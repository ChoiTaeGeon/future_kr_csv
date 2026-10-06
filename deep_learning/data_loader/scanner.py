"""
Tick Data Scanner & Sanitizer
- Scans directory recursively for Korean futures tick CSV files
- Extracts date from filename or content and sorts chronologically
- Auto-infers column mapping [Timestamp, Price, Volume, Side]
- Detects missing values, duplicates, abnormal price spikes, and halts
- Emits detailed data quality report
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class FileIntegrityResult:
    file_path: str
    file_name: str
    date_str: Optional[str]
    is_valid: bool
    file_size_bytes: int
    encoding_detected: str
    row_count: int
    issues: List[str] = field(default_factory=list)
    sha256_hash: str = ""

@dataclass
class DailyQualityStats:
    date_str: str
    file_name: str
    total_ticks: int
    cleaned_ticks: int
    missing_count: int
    duplicate_count: int
    spike_count: int
    zero_or_negative_price_count: int = 0
    time_inversion_count: int = 0
    halt_periods: List[str] = field(default_factory=list)
    integrity_status: str = "PASSED"  # PASSED, WARNING, CORRUPTED


@dataclass
class DataSanitizeReport:
    total_files_scanned: int
    valid_files_count: int
    corrupted_files_count: int
    total_raw_ticks: int
    total_cleaned_ticks: int
    integrity_results: List[FileIntegrityResult] = field(default_factory=list)
    daily_stats: List[DailyQualityStats] = field(default_factory=list)

    def summary(self) -> str:
        dup = sum(d.duplicate_count for d in self.daily_stats)
        spk = sum(d.spike_count for d in self.daily_stats)
        miss = sum(d.missing_count for d in self.daily_stats)
        return (
            f"Files: {self.valid_files_count} Valid / {self.corrupted_files_count} Corrupted / {self.total_files_scanned} Total | "
            f"Ticks: {self.total_cleaned_ticks:,} Cleaned (Raw: {self.total_raw_ticks:,}) | "
            f"Filtered: {miss:,} Missing, {dup:,} Dups, {spk:,} Spikes"
        )


class TickDataScanner:
    def __init__(self, data_dir: str | Path, symbol: str = "KOSPI_F"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol

    def extract_date_from_name(self, filename: str) -> Optional[str]:
        """Parse date from file name like 170207.csv or 20170207.csv or KOSPI_2017-02-07.csv."""
        digits = re.findall(r'\d+', filename)
        for m in digits:
            if len(m) == 8:
                try:
                    dt = datetime.strptime(m, "%Y%m%d")
                    if 1990 <= dt.year <= 2035:
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            elif len(m) == 6:
                try:
                    yy = int(m[:2])
                    mm = int(m[2:4])
                    dd = int(m[4:6])
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        prefix = "19" if yy >= 70 else "20"
                        dt = datetime.strptime(f"{prefix}{m}", "%Y%m%d")
                        if 1990 <= dt.year <= 2035:
                            return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return None

    def scan_and_sort_files(self) -> List[Tuple[str, Path]]:
        """
        Recursively scans data_dir and returns sorted list of (date_str, file_path).
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        all_csvs = list(self.data_dir.rglob("*.csv"))
        file_list: List[Tuple[str, Path]] = []

        for p in all_csvs:
            dt_str = self.extract_date_from_name(p.name)
            if dt_str:
                file_list.append((dt_str, p))

        # Sort strictly ascending by date
        file_list.sort(key=lambda x: x[0])
        return file_list

    def infer_columns_and_encoding(self, file_path: Path) -> Tuple[str, Dict[str, int]]:
        """
        Detects file encoding and identifies column indexes for (time, price, volume, side).
        """
        encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig', 'latin1']
        df_sample = None
        chosen_enc = 'cp949'

        for enc in encodings:
            try:
                df_sample = pd.read_csv(file_path, encoding=enc, nrows=10, low_memory=False)
                chosen_enc = enc
                break
            except Exception:
                continue

        if df_sample is None:
            raise ValueError(f"Unable to decode CSV file: {file_path}")

        # Search for column names by Korean keywords
        col_map: Dict[str, int] = {}
        cleaned_headers = [re.sub(r'\s+', '', str(c)).lower() for c in df_sample.columns]

        for idx, h in enumerate(cleaned_headers):
            if any(k in h for k in ['시간', 'time', '체결시간']):
                if 'time' not in col_map: col_map['time'] = idx
            elif any(k in h for k in ['현재가', '체결가', 'price', 'close', '종가']):
                if 'price' not in col_map: col_map['price'] = idx
            elif any(k in h for k in ['체결량', 'volume', 'qty', '수량']):
                if 'volume' not in col_map: col_map['volume'] = idx
            elif any(k in h for k in ['매도수', '매수매도', '구분', 'side']):
                if 'side' not in col_map: col_map['side'] = idx

        # Positional fallback for 19-column KOSPI HTS format
        if 'time' not in col_map:
            col_map['time'] = 1 if len(df_sample.columns) > 1 else 0
        if 'price' not in col_map:
            col_map['price'] = 5 if len(df_sample.columns) > 5 else 1
        if 'volume' not in col_map:
            col_map['volume'] = 8 if len(df_sample.columns) > 8 else 2

        return chosen_enc, col_map

    def verify_file_integrity(self, file_path: Path) -> FileIntegrityResult:
        """
        Automated integrity check for a single CSV file:
        1. Non-empty file size (> 100 bytes)
        2. Valid text encoding detection (CP949, EUC-KR, UTF-8)
        3. Header and required column presence (Time, Price, Volume)
        4. Row count validity (> 10 rows)
        5. SHA-256 checksum generation for change tracking
        """
        import hashlib
        issues = []
        file_size = file_path.stat().st_size
        date_str = self.extract_date_from_name(file_path.name)

        # 1. File size check
        if file_size < 100:
            issues.append(f"비정상적인 최소 파일 크기 ({file_size} bytes)")
            return FileIntegrityResult(
                file_path=str(file_path),
                file_name=file_path.name,
                date_str=date_str,
                is_valid=False,
                file_size_bytes=file_size,
                encoding_detected="none",
                row_count=0,
                issues=issues
            )

        # Compute SHA-256
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        sha_str = sha256.hexdigest()

        # 2. Encoding detection & header check
        chosen_enc = None
        col_map = {}
        for enc in ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig', 'latin1']:
            try:
                sample = pd.read_csv(file_path, encoding=enc, nrows=5)
                chosen_enc = enc
                break
            except Exception:
                continue

        if chosen_enc is None:
            issues.append("인코딩 디코딩 실패 (지원 인코딩 형식 불일치)")
            return FileIntegrityResult(
                file_path=str(file_path),
                file_name=file_path.name,
                date_str=date_str,
                is_valid=False,
                file_size_bytes=file_size,
                encoding_detected="corrupted",
                row_count=0,
                issues=issues,
                sha256_hash=sha_str
            )

        try:
            _, col_map = self.infer_columns_and_encoding(file_path)
            if 'price' not in col_map or 'time' not in col_map:
                issues.append("필수 컬럼(체결시각 또는 체결가) 누락")
        except Exception as e:
            issues.append(f"컬럼 매핑 실패: {e}")

        # 3. Fast line count estimation
        row_count = 0
        try:
            with open(file_path, "rb") as f:
                row_count = sum(1 for _ in f) - 1
            if row_count < 10:
                issues.append(f"체결 틱 수 부족 ({row_count} 행)")
        except Exception as e:
            issues.append(f"행 개수 집계 오류: {e}")

        is_valid = (len(issues) == 0)
        return FileIntegrityResult(
            file_path=str(file_path),
            file_name=file_path.name,
            date_str=date_str,
            is_valid=is_valid,
            file_size_bytes=file_size,
            encoding_detected=chosen_enc,
            row_count=max(0, row_count),
            issues=issues,
            sha256_hash=sha_str
        )

    def scan_all_files_with_integrity(self) -> Tuple[List[Tuple[str, Path]], DataSanitizeReport]:
        """
        Runs complete automated integrity verification over all files in data_dir.
        Filters out corrupted files and builds comprehensive integrity report.
        """
        raw_files = self.scan_and_sort_files()
        valid_files: List[Tuple[str, Path]] = []
        integrity_results: List[FileIntegrityResult] = []

        for dt_str, path in raw_files:
            res = self.verify_file_integrity(path)
            integrity_results.append(res)
            if res.is_valid:
                valid_files.append((dt_str, path))

        corrupted_count = len(raw_files) - len(valid_files)
        report = DataSanitizeReport(
            total_files_scanned=len(raw_files),
            valid_files_count=len(valid_files),
            corrupted_files_count=corrupted_count,
            total_raw_ticks=0,
            total_cleaned_ticks=0,
            integrity_results=integrity_results
        )
        return valid_files, report

    def clean_daily_ticks(
        self,
        file_path: Path,
        date_str: str,
        encoding: str,
        col_map: Dict[str, int]
    ) -> Tuple[pd.DataFrame, DailyQualityStats]:
        """
        Reads, standardizes, cleans ticks for a single trading day.
        Returns cleaned DataFrame and quality stats.
        """
        usecols = [col_map['time'], col_map['price'], col_map['volume']]
        if 'side' in col_map:
            usecols.append(col_map['side'])

        df_raw = pd.read_csv(
            file_path,
            encoding=encoding,
            usecols=usecols,
            header=0,
            low_memory=False
        )

        # Standardize column naming
        reverse_map = {idx: name for name, idx in col_map.items()}
        new_cols = [reverse_map.get(c, f"col_{c}") for c in df_raw.columns]
        df_raw.columns = new_cols

        total_ticks = len(df_raw)

        # 1. Clean Time & construct full datetime timestamp
        raw_time = df_raw['time'].astype(str).str.strip()
        # Filter out invalid time strings
        valid_time_mask = raw_time.str.match(r'^\d{1,2}:\d{2}:\d{2}') | raw_time.str.match(r'^\d{6}$')
        missing_count = int((~valid_time_mask).sum())
        df = df_raw[valid_time_mask].copy()

        # Format times
        def fmt_time(t_str: str) -> str:
            if ':' in t_str:
                parts = t_str.split(':')
                return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
            elif len(t_str) == 6:
                return f"{t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}"
            return "09:00:00"

        time_formatted = df['time'].apply(fmt_time)
        df['datetime'] = pd.to_datetime(date_str + ' ' + time_formatted, errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)

        # 2. Convert Price & Volume
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(1.0)
        df.dropna(subset=['price', 'volume'], inplace=True)

        # 3. Handle Side (1: Buy, -1: Sell, 0: Neutral)
        if 'side' in df.columns:
            side_str = df['side'].astype(str).str.strip()
            df['side'] = np.where(side_str.str.contains(r'매수|buy|\+|▲', case=False), 1,
                         np.where(side_str.str.contains(r'매도|sell|\-|▼', case=False), -1, 0))
        else:
            # If no explicit side column, infer side by tick rule (price change)
            price_diff = df['price'].diff().fillna(0)
            df['side'] = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))

        # 4. Abnormal price detection (<= 0 or outside realistic bounds 50~600 for KOSPI futures)
        zero_neg_mask = (df['price'] <= 0.0)
        zero_neg_count = int(zero_neg_mask.sum())
        spike_mask = (df['price'] < 50.0) | (df['price'] > 600.0)
        spike_count = int(spike_mask.sum())
        df = df[~spike_mask & ~zero_neg_mask].copy()

        # 5. Chronological sort & duplicate detection
        # Korean HTS files often store ticks in reverse order (15:45:00 at top, 09:00:00 at bottom)
        time_inversion_count = 0
        if len(df) > 1 and df['datetime'].iloc[0] > df['datetime'].iloc[-1]:
            time_inversion_count = len(df)
            df = df.iloc[::-1].reset_index(drop=True)

        initial_len = len(df)
        df.drop_duplicates(subset=['datetime', 'price', 'volume'], inplace=True)
        dup_count = initial_len - len(df)

        # 6. Trading halt detection (> 15 minutes silence during regular session 09:00~15:45)
        df = df.sort_values('datetime').reset_index(drop=True)
        halts = []
        if len(df) > 1:
            time_deltas = df['datetime'].diff().dt.total_seconds()
            long_gaps = df[time_deltas > 900]
            for idx in long_gaps.index:
                t_prev = df.loc[idx - 1, 'datetime'].strftime('%H:%M:%S')
                t_curr = df.loc[idx, 'datetime'].strftime('%H:%M:%S')
                # Ignore regular lunch or after-market gap
                if "09:00:00" <= t_prev <= "15:30:00":
                    halts.append(f"{t_prev}~{t_curr}")

        # Assess integrity status
        status = "PASSED"
        if spike_count > 0 or zero_neg_count > 0:
            status = "WARNING"
        if len(df) < 50:
            status = "CORRUPTED"

        stats = DailyQualityStats(
            date_str=date_str,
            file_name=file_path.name,
            total_ticks=total_ticks,
            cleaned_ticks=len(df),
            missing_count=missing_count,
            duplicate_count=dup_count,
            spike_count=spike_count,
            zero_or_negative_price_count=zero_neg_count,
            time_inversion_count=time_inversion_count,
            halt_periods=halts,
            integrity_status=status
        )

        return df[['datetime', 'price', 'volume', 'side']], stats
