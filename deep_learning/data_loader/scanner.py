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
class DailyQualityStats:
    date_str: str
    file_name: str
    total_ticks: int
    cleaned_ticks: int
    missing_count: int
    duplicate_count: int
    spike_count: int
    halt_periods: List[str] = field(default_factory=list)


@dataclass
class DataSanitizeReport:
    total_files_scanned: int
    valid_files_count: int
    total_raw_ticks: int
    total_cleaned_ticks: int
    daily_stats: List[DailyQualityStats] = field(default_factory=list)

    def summary(self) -> str:
        dup = sum(d.duplicate_count for d in self.daily_stats)
        spk = sum(d.spike_count for d in self.daily_stats)
        miss = sum(d.missing_count for d in self.daily_stats)
        return (
            f"Files: {self.valid_files_count}/{self.total_files_scanned} | "
            f"Ticks: {self.total_cleaned_ticks:,} (Raw: {self.total_raw_ticks:,}) | "
            f"Filtered: {miss:,} Missing, {dup:,} Dup, {spk:,} Spikes"
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

        # 4. Spike detection (abnormal filter: price > 500 or price < 50 for KOSPI futures)
        spike_mask = (df['price'] < 50.0) | (df['price'] > 600.0)
        spike_count = int(spike_mask.sum())
        df = df[~spike_mask].copy()

        # 5. Chronological sort & duplicate detection
        # Korean HTS files often store ticks in reverse order (15:45:00 at top, 09:00:00 at bottom)
        if len(df) > 1 and df['datetime'].iloc[0] > df['datetime'].iloc[-1]:
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

        stats = DailyQualityStats(
            date_str=date_str,
            file_name=file_path.name,
            total_ticks=total_ticks,
            cleaned_ticks=len(df),
            missing_count=missing_count,
            duplicate_count=dup_count,
            spike_count=spike_count,
            halt_periods=halts
        )

        return df[['datetime', 'price', 'volume', 'side']], stats
