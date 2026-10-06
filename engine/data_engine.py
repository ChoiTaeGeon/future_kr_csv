"""
Data Pipeline & Local DB Engine using DuckDB
- High-performance ingestion of hundreds of daily tick CSV files
- Incremental updates & sync tracking with hash/mtime verification
- Safe transactions and fast column queries
"""
import os
import hashlib
import threading
import duckdb
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from engine.logger import log_error, log_sync, log_info

_DATA_ENGINE_LOCK = threading.Lock()
_SHARED_DATA_ENGINE: Optional['DataEngine'] = None

def get_data_engine(db_path: Optional[str] = None) -> 'DataEngine':
    """Return a thread-safe process-wide singleton DataEngine instance to prevent DuckDB file lock conflicts."""
    global _SHARED_DATA_ENGINE
    with _DATA_ENGINE_LOCK:
        if _SHARED_DATA_ENGINE is None:
            from config import DB_PATH
            target_path = db_path if db_path else str(DB_PATH)
            _SHARED_DATA_ENGINE = DataEngine(db_path=target_path)
        return _SHARED_DATA_ENGINE

class DataEngine:
    def __init__(self, db_path: str = "data/market_data.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.is_syncing = False
        self._cached_dates: Optional[List[str]] = None
        self._cached_total_ticks: Optional[int] = None
        self.conn = self._safe_connect()
        self._init_db()

    def clear_resample_cache(self) -> int:
        """Delete all cached resampled candle files from CACHE_DIR."""
        from config import CACHE_DIR
        deleted_count = 0
        try:
            if CACHE_DIR.exists():
                for f in CACHE_DIR.glob("*.parquet"):
                    try:
                        f.unlink()
                        deleted_count += 1
                    except Exception:
                        pass
            if deleted_count > 0:
                print(f"[*] Cleared {deleted_count} resampled candle cache files.")
        except Exception as e:
            print(f"[Error clearing resample cache]: {e}")
        return deleted_count

    def invalidate_cache(self):
        """Invalidate cached date lists, tick counts, and resampled bars cache when data changes."""
        with self.lock:
            self._cached_dates = None
            self._cached_total_ticks = None
        self.clear_resample_cache()

    def _safe_connect(self):
        """Connect to DuckDB with automatic WAL corruption recovery and process-lock handling."""
        import time
        import shutil
        max_lock_retries = 3
        for attempt in range(max_lock_retries):
            try:
                conn = duckdb.connect(str(self.db_path))
                # Ensure clean checkpoint state
                try:
                    conn.execute("CHECKPOINT")
                except Exception:
                    pass
                return conn
            except Exception as e:
                err_msg = str(e).lower()
                # 1. Detect process-lock conflict (another process is holding the file)
                if any(k in err_msg for k in ["already open", "사용 중", "access", "lock", "conflict"]):
                    if attempt < max_lock_retries - 1:
                        time.sleep(1.0)
                        continue
                    log_error(f"DuckDB database '{self.db_path}' is locked by another running process: {e}")
                    raise

                # 2. Detect genuine corruption or WAL replay failure
                is_corruption = any(k in err_msg for k in ["replay", "corrupt", "checksum", "wal", "out of range", "dictionary"])
                log_error(f"DuckDB connect failed on '{self.db_path}': {e}. Corruption detected: {is_corruption}. Attempting recovery...", exc=e)

                wal_path = Path(str(self.db_path) + ".wal")
                if wal_path.exists():
                    try:
                        wal_path.unlink()
                        conn = duckdb.connect(str(self.db_path))
                        try:
                            conn.execute("CHECKPOINT")
                        except Exception:
                            pass
                        log_sync("DuckDB auto-recovered successfully after removing corrupted .wal file.", level="WARNING")
                        return conn
                    except Exception as wal_err:
                        log_error("WAL removal recovery failed. Backing up corrupted DB.", exc=wal_err)

                # Check if a pristine root data/market_data.duckdb exists to restore from
                root_src = Path("data/market_data.duckdb")
                if root_src.exists() and root_src.resolve() != self.db_path.resolve():
                    try:
                        shutil.copy2(root_src, self.db_path)
                        conn = duckdb.connect(str(self.db_path))
                        log_sync(f"Restored clean database from '{root_src}' to '{self.db_path}'.", level="WARNING")
                        return conn
                    except Exception as restore_err:
                        log_error("Restoring from root data/market_data.duckdb failed.", exc=restore_err)

                if is_corruption:
                    backup_path = self.db_path.with_suffix(f".corrupted_{int(time.time())}")
                    try:
                        if self.db_path.exists():
                            self.db_path.rename(backup_path)
                        conn = duckdb.connect(str(self.db_path))
                        log_sync(f"Corrupted database backed up to '{backup_path.name}'. Re-initialized clean database.", level="WARNING")
                        return conn
                    except Exception as final_err:
                        log_error("Fatal: failed to create new DB after corruption backup", exc=final_err)
                        raise
                raise

    def _init_db(self):
        """Initialize required tables and indexes."""
        # 1. Sync metadata table to track loaded CSV files
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                file_name VARCHAR PRIMARY KEY,
                file_path VARCHAR,
                file_hash VARCHAR,
                file_size_bytes BIGINT,
                row_count BIGINT,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Main ticks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                price DOUBLE,
                volume BIGINT,
                bid DOUBLE,
                ask DOUBLE,
                file_source VARCHAR
            )
        """)
        
        # 3. Create index for fast time-series & symbol filtering
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_ts_sym 
            ON ticks (timestamp, symbol)
        """)

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute quick hash of file based on header + size + tail for fast verification."""
        hasher = hashlib.md5()
        size = file_path.stat().st_size
        hasher.update(str(size).encode('utf-8'))
        with open(file_path, 'rb') as f:
            # Read first 8KB and last 8KB
            hasher.update(f.read(8192))
            if size > 16384:
                f.seek(size - 8192)
                hasher.update(f.read(8192))
        return hasher.hexdigest()

    @staticmethod
    def extract_date_from_filename(filename: str) -> Optional[str]:
        """
        Extracts date (YYYY-MM-DD) from filenames like '170328.csv', '20170328.csv', 'ticks_170328.csv'.
        Supports 6-digit (YYMMDD) and 8-digit (YYYYMMDD) patterns.
        """
        import re
        # Find 6 or 8 consecutive digits
        matches = re.findall(r'(?:\b|\D|^)(\d{6}|\d{8})(?:\b|\D|$)', filename)
        if not matches:
            return None

        # Take the most plausible date string
        for m in matches:
            if len(m) == 8:
                try:
                    dt = datetime.strptime(m, "%Y%m%d")
                    if 1990 <= dt.year <= 2035:
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            elif len(m) == 6:
                try:
                    yy = int(m[:2])
                    mm = int(m[2:4])
                    dd = int(m[4:6])
                    if not (1 <= mm <= 12 and 1 <= dd <= 31):
                        continue
                    prefix = "19" if yy >= 70 else "20"
                    dt = datetime.strptime(f"{prefix}{m}", "%Y%m%d")
                    if 1990 <= dt.year <= 2035:
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        return None

    def _read_csv_smart(self, file: Path, file_date: Optional[str] = None, symbol_default: str = "KOSPI_F") -> pd.DataFrame:
        """
        Robustly reads Korean tick CSV files with various encodings (CP949, UTF-8, etc.),
        auto-maps columns (Time, Price, Volume), synthesizes complete timestamps,
        and ensures strict ascending chronological order.
        """
        encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig', 'latin1']
        df_raw = None
        
        for enc in encodings:
            try:
                df_raw = pd.read_csv(file, encoding=enc, low_memory=False, engine='c')
                break
            except Exception:
                continue

        if df_raw is None or df_raw.empty:
            return pd.DataFrame()

        # Korean HTS Column auto-mapping
        ts_col = None
        p_col = None
        v_col = None

        import re
        col_names = [re.sub(r'\s+', '', str(c)).lower() for c in df_raw.columns]
        for idx, c in enumerate(col_names):
            if any(k in c for k in ['시간', 'time', '체결시간', '일자']):
                if not ts_col: ts_col = df_raw.columns[idx]
            elif any(k in c for k in ['현재가', '체결가', 'price', 'close', '종가', '지수']):
                if not p_col: p_col = df_raw.columns[idx]
            elif any(k in c for k in ['체결량', 'volume', 'qty', '수량', '거래량']):
                if not v_col: v_col = df_raw.columns[idx]

        # Positional Fallback for 19-column HTS formats
        if not ts_col or not p_col:
            if df_raw.shape[1] >= 19:
                ts_col = df_raw.columns[1]
                p_col = df_raw.columns[5]
                v_col = df_raw.columns[8] if df_raw.shape[1] > 8 else None
            elif df_raw.shape[1] >= 3:
                ts_col = df_raw.columns[0]
                p_col = df_raw.columns[1]
                v_col = df_raw.columns[4] if df_raw.shape[1] > 4 else df_raw.columns[2]

        if not ts_col or not p_col:
            return pd.DataFrame()

        # Detect and reverse descending chronological order (e.g. Kiwoom CSV with 15:45:00 at top)
        first_t = str(df_raw[ts_col].iloc[0]).strip()
        last_t = str(df_raw[ts_col].iloc[-1]).strip()
        if first_t > last_t:
            df_raw = df_raw.iloc[::-1].reset_index(drop=True)

        # Extract Time strings & combine with file_date
        time_series = df_raw[ts_col].astype(str).str.strip()
        price_series = pd.to_numeric(df_raw[p_col].astype(str).str.replace(',', ''), errors='coerce')
        
        if v_col and v_col in df_raw.columns:
            vol_series = pd.to_numeric(df_raw[v_col].astype(str).str.replace(',', ''), errors='coerce').abs().fillna(1).astype(np.int64)
        else:
            vol_series = pd.Series(1, index=df_raw.index, dtype=np.int64)

        # Drop NaN prices
        valid_mask = price_series.notna()
        time_series = time_series[valid_mask]
        price_series = price_series[valid_mask]
        vol_series = vol_series[valid_mask]

        if time_series.empty:
            return pd.DataFrame()

        # Check if time column itself contains date information
        sample_val = str(time_series.iloc[0])
        has_internal_date = ("-" in sample_val or "/" in sample_val) and len(sample_val) >= 10
        if not file_date and not has_internal_date:
            # File has no date in filename and no date in data columns.
            # Do NOT fall back to st_mtime to avoid falsely attributing historical files to today.
            log_sync(f"[Skip File] Cannot determine date for {file.name}: filename has no valid date and content has only time.", level="WARNING")
            return pd.DataFrame()

        # Vectorized deterministic timestamp conversion (Instant C-speed)
        if has_internal_date:
            clean_dt = time_series.str.replace(r'[^0-9: \-\/]', '', regex=True).str.slice(0, 19)
            clean_dt = clean_dt.str.replace('/', '-')
            ts_parsed = pd.to_datetime(clean_dt, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        else:
            raw_t = time_series.str.strip()
            if raw_t.str.contains(':', regex=False).any():
                clean_time = raw_t.str.slice(0, 8)
                clean_time = clean_time.apply(lambda s: s if len(s) == 8 else (f"0{s}" if len(s) == 7 else "00:00:00"))
            else:
                digits = raw_t.str.replace(r'[^0-9]', '', regex=True).str.zfill(6).str.slice(0, 6)
                clean_time = digits.str.slice(0, 2) + ":" + digits.str.slice(2, 4) + ":" + digits.str.slice(4, 6)

            full_ts = file_date + " " + clean_time
            ts_parsed = pd.to_datetime(full_ts, format='%Y-%m-%d %H:%M:%S', errors='coerce')

        clean_df = pd.DataFrame({
            'timestamp': ts_parsed,
            'symbol': symbol_default,
            'price': price_series,
            'volume': vol_series,
            'bid': price_series,
            'ask': price_series,
            'file_source': file.name
        }).dropna(subset=['timestamp'])

        clean_df.sort_values('timestamp', ascending=True, kind='stable', inplace=True)
        clean_df.reset_index(drop=True, inplace=True)
        return clean_df

    def scan_and_sync(self, csv_target: Union[str, Path, List[Union[str, Path]]] = None, symbol_default: str = "KOSPI_F", force_full: bool = False, csv_folder: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        High-performance incremental or full sync with multithreaded parsing and batch DuckDB inserts.
        Supports:
        - Directory path (str / Path): ingests all .csv files inside.
        - Single CSV file path (str / Path).
        - List of CSV file paths (List[str] / List[Path]).
        - Semicolon-delimited file path string (e.g. "f1.csv;f2.csv").
        """
        if csv_target is None and csv_folder is not None:
            csv_target = csv_folder
        if csv_target is None:
            csv_target = RAW_CSV_DIR
        if self.is_syncing:
            log_sync("Sync already in progress. Ignoring concurrent scan_and_sync request.", level="WARNING")
            return {
                "status": "already_syncing",
                "message": "틱데이터 동기화가 이미 백그라운드에서 진행 중입니다.",
                "synced_files": 0,
                "total_rows_added": 0,
                "total_ticks": self.get_total_ticks()
            }

        with self.lock:
            self.is_syncing = True
            try:
                return self._scan_and_sync_internal(
                    csv_target=csv_target,
                    symbol_default=symbol_default,
                    force_full=force_full
                )
            finally:
                self.is_syncing = False

    def _scan_and_sync_internal(self, csv_target: Any, symbol_default: str = "KOSPI_F", force_full: bool = False) -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor
        from engine.tracker import tracker

        csv_files: List[Path] = []
        if isinstance(csv_target, (list, tuple)):
            for p in csv_target:
                item = Path(p)
                if item.exists():
                    if item.is_file() and item.suffix.lower() == '.csv':
                        csv_files.append(item)
                    elif item.is_dir():
                        csv_files.extend(list(item.glob("*.[cC][sS][vV]")))
        else:
            target_str = str(csv_target).strip()
            if ';' in target_str:
                raw_parts = [p.strip() for p in target_str.split(';') if p.strip()]
                for rp in raw_parts:
                    item = Path(rp)
                    if item.exists():
                        if item.is_file() and item.suffix.lower() == '.csv':
                            csv_files.append(item)
                        elif item.is_dir():
                            csv_files.extend(list(item.glob("*.[cC][sS][vV]")))
            else:
                p = Path(target_str)
                if not p.exists():
                    return {"status": "error", "message": f"Target '{csv_target}' does not exist."}
                if p.is_file():
                    csv_files = [p]
                elif p.is_dir():
                    csv_files = list(p.glob("*.[cC][sS][vV]"))

        # Deduplicate files by filename
        csv_map = {f.name.lower(): f for f in csv_files}
        csv_files = sorted(list(csv_map.values()), key=lambda x: x.name)
        if not csv_files:
            return {"status": "ok", "synced_files": 0, "total_rows_added": 0, "total_ticks": self.get_total_ticks()}

        if force_full:
            print("[*] Performing FULL RE-SYNC: clearing existing table & metadata...")
            self.conn.execute("DELETE FROM ticks")
            self.conn.execute("DELETE FROM sync_metadata")
            files_to_load = csv_files
        else:
            existing_meta = self.conn.execute("SELECT file_name, file_hash, file_path, row_count FROM sync_metadata").fetchdf()
            synced_map = dict(zip(existing_meta['file_name'], existing_meta['file_hash'])) if not existing_meta.empty else {}
            meta_rows_map = dict(zip(existing_meta['file_name'], existing_meta['row_count'])) if not existing_meta.empty else {}

            # Detect corrupt or abnormal files in ticks table (e.g. volume mapped to price, truncated row count)
            corrupt_files = set()
            try:
                corrupt_df = self.conn.execute("""
                    SELECT file_source 
                    FROM ticks 
                    GROUP BY file_source 
                    HAVING MAX(price) > 5000 OR MIN(price) < 10 OR COUNT(*) < 2000
                """).fetchdf()
                if not corrupt_df.empty:
                    corrupt_files = set(corrupt_df['file_source'].dropna())
            except Exception:
                pass

            files_to_load = []
            for f in csv_files:
                f_size = f.stat().st_size
                rec_rows = meta_rows_map.get(f.name, 0)
                is_corrupt = (f.name in corrupt_files) or (f_size > 500000 and rec_rows < 2000)
                if f.name not in synced_map or synced_map[f.name] != self._compute_file_hash(f) or is_corrupt:
                    files_to_load.append(f)

        if not files_to_load:
            tracker.finish("All files already up to date.")
            return {
                "status": "up_to_date",
                "synced_files": 0,
                "total_rows_added": 0,
                "total_ticks": self.get_total_ticks()
            }

        total_rows_added = 0
        loaded_file_count = 0
        mode_str = "Full Re-Sync" if force_full else "Incremental Sync"
        self.clear_resample_cache()
        print(f"[*] [{mode_str}] Ingesting {len(files_to_load)} CSV files into DuckDB...")
        tracker.start("INGESTION", len(files_to_load), f"Ingesting {len(files_to_load)} CSV files ({mode_str})...")

        def _parse_single(file_item: Path):
            if tracker.is_cancelled:
                return None
            f_name = file_item.name
            f_hash = self._compute_file_hash(file_item)
            f_size = file_item.stat().st_size
            f_date = self.extract_date_from_filename(f_name)
            try:
                cdf = self._read_csv_smart(file_item, file_date=f_date, symbol_default=symbol_default)
                return (file_item, f_name, f_hash, f_size, cdf)
            except Exception as e:
                print(f"[Error reading {f_name}]: {e}")
                return None

        # Parallel batch parsing & transaction batch insertion (50 files per chunk)
        chunk_size = 50
        with ThreadPoolExecutor() as executor:
            for chunk_idx in range(0, len(files_to_load), chunk_size):
                if tracker.is_cancelled:
                    print("\n[*] Ingestion cancelled by user.")
                    break

                chunk_files = files_to_load[chunk_idx : chunk_idx + chunk_size]
                results = list(executor.map(_parse_single, chunk_files))

                valid_dfs = []
                meta_records = []
                names_to_delete = []

                for res in results:
                    if tracker.is_cancelled:
                        break
                    if not res or res[4].empty:
                        tracker.step(1, message="Skipped empty file")
                        continue
                    file_item, f_name, f_hash, f_size, cdf = res
                    row_count = len(cdf)
                    valid_dfs.append(cdf)
                    names_to_delete.append(f_name)
                    meta_records.append((f_name, str(file_item), f_hash, f_size, row_count))
                    total_rows_added += row_count
                    loaded_file_count += 1
                    tracker.step(1, message=f"Parsed {f_name} (+{row_count:,} rows)", details=f"Progress: {loaded_file_count}/{len(files_to_load)}")

                if valid_dfs:
                    combined_chunk_df = pd.concat(valid_dfs, ignore_index=True)
                    # Enforce strict schema & types before bulk append to DuckDB
                    combined_chunk_df = combined_chunk_df[['timestamp', 'symbol', 'price', 'volume', 'bid', 'ask', 'file_source']].copy()
                    combined_chunk_df.dropna(subset=['timestamp', 'price'], inplace=True)
                    combined_chunk_df['price'] = combined_chunk_df['price'].astype('float64')
                    combined_chunk_df['volume'] = combined_chunk_df['volume'].fillna(1).astype('int64')
                    combined_chunk_df['bid'] = combined_chunk_df['bid'].fillna(combined_chunk_df['price']).astype('float64')
                    combined_chunk_df['ask'] = combined_chunk_df['ask'].fillna(combined_chunk_df['price']).astype('float64')
                    combined_chunk_df['symbol'] = combined_chunk_df['symbol'].astype(str)
                    combined_chunk_df['file_source'] = combined_chunk_df['file_source'].astype(str)

                    # Pre-emptively rollback any dangling/aborted transaction
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass

                    self.conn.execute("BEGIN TRANSACTION")
                    try:
                        # Clean existing entries if updating
                        for fn in names_to_delete:
                            self.conn.execute("DELETE FROM ticks WHERE file_source = ?", [fn])
                            self.conn.execute("DELETE FROM sync_metadata WHERE file_name = ?", [fn])

                        self.conn.append("ticks", combined_chunk_df)

                        for m in meta_records:
                            self.conn.execute("""
                                INSERT INTO sync_metadata (file_name, file_path, file_hash, file_size_bytes, row_count, synced_at)
                                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            """, list(m))

                        self.conn.execute("COMMIT")
                    except Exception as e:
                        try:
                            self.conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        log_error(f"[Batch Insert Error in chunk {chunk_idx // chunk_size}] ({len(valid_dfs)} files)", exc=e)
                        log_sync(f"[Batch Insert Error in chunk {chunk_idx // chunk_size}]: {e}", level="ERROR", exc=e)

        self.invalidate_cache()
        tracker.finish(f"Sync complete (+{total_rows_added:,} rows)")
        return {
            "status": "success",
            "synced_files": loaded_file_count,
            "total_rows_added": total_rows_added,
            "total_ticks": self.get_total_ticks()
        }

    def export_ticks_to_csv(self, output_path: Union[str, Path], symbol: str = "KOSPI_F", 
                            start_time: Optional[str] = None, end_time: Optional[str] = None,
                            start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """High-speed native C-level DuckDB export to single unified CSV file."""
        st = start_time or start_date
        et = end_time or end_date
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sql_out_path = str(out_p).replace('\\', '/')

        where_clauses = ["symbol = ?"]
        params = [symbol]
        if st:
            where_clauses.append("timestamp >= ?")
            params.append(st)
        if et:
            where_clauses.append("timestamp <= ?")
            params.append(et)

        where_sql = " AND ".join(where_clauses)
        with self.lock:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM ticks WHERE {where_sql}", params).fetchone()[0]
                if count == 0:
                    return {"status": "error", "message": "No ticks found for specified filters.", "count": 0}

                query = f"""
                    COPY (
                        SELECT timestamp, symbol, price, volume, bid, ask, file_source
                        FROM ticks
                        WHERE {where_sql}
                        ORDER BY timestamp ASC
                    ) TO '{sql_out_path}' (HEADER, DELIMITER ',')
                """
                self.conn.execute(query, params)
                return {"status": "success", "count": count, "output_path": str(out_p)}
            except Exception as e:
                print(f"[Export Ticks Error]: {e}")
                return {"status": "error", "message": str(e), "count": 0}

    def get_total_ticks(self, symbol: Optional[str] = None) -> int:
        """Return total tick count in DB."""
        with self.lock:
            if symbol:
                res = self.conn.execute("SELECT COUNT(*) FROM ticks WHERE symbol = ?", [symbol]).fetchone()
            else:
                res = self.conn.execute("SELECT COUNT(*) FROM ticks").fetchone()
            return res[0] if res else 0

    def get_date_range(self, symbol: Optional[str] = None):
        """Return min and max timestamp in ticks table."""
        with self.lock:
            if symbol:
                query = "SELECT MIN(timestamp), MAX(timestamp) FROM ticks WHERE symbol = ?"
                return self.conn.execute(query, [symbol]).fetchone()
            return self.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ticks").fetchone()

    def get_symbols(self) -> List[str]:
        """Return list of distinct symbols."""
        with self.lock:
            df = self.conn.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol").fetchdf()
            if df is not None and not df.empty and 'symbol' in df.columns:
                return df['symbol'].dropna().tolist()
            return []

    def load_ticks_df(self, symbol: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
        """Fetch raw tick DataFrame ordered by timestamp."""
        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT timestamp, symbol, price, volume, bid, ask FROM ticks {where_clause} ORDER BY timestamp ASC"
        with self.lock:
            return self.conn.execute(query, params).fetchdf()

    get_ticks_df = load_ticks_df

    def get_available_years(self, symbol: Optional[str] = None) -> List[str]:
        """Returns sorted list of distinct years available in DB (fast derivation from dates)."""
        dates = self.get_available_dates(symbol)
        if dates:
            return sorted(list(set(d[:4] for d in dates)))
        return []

    def get_available_months(self, symbol: Optional[str] = None) -> List[str]:
        """Returns sorted list of distinct year-months available in DB (fast derivation from dates)."""
        dates = self.get_available_dates(symbol)
        if dates:
            return sorted(list(set(d[:7] for d in dates)))
        return []

    def get_available_dates(self, symbol: Optional[str] = None) -> List[str]:
        """Returns sorted list of distinct trading dates (YYYY-MM-DD) available in DB."""
        if symbol is None and self._cached_dates is not None:
            return self._cached_dates
        try:
            where = "WHERE symbol = ? AND timestamp IS NOT NULL" if symbol else "WHERE timestamp IS NOT NULL"
            params = [symbol] if symbol else []
            query = f"SELECT DISTINCT STRFTIME(timestamp, '%Y-%m-%d') AS d FROM ticks {where} ORDER BY d ASC"
            with self.lock:
                df = self.conn.execute(query, params).fetchdf()
            if df is not None and not df.empty and 'd' in df.columns:
                res = [str(d)[:10] for d in df['d'].dropna().tolist()]
                if symbol is None:
                    self._cached_dates = res
                return res
            return []
        except Exception:
            return []

    def get_sync_status(self) -> Dict[str, Any]:
        """Returns quick sync status with total ticks and file count."""
        total_ticks = self.get_total_ticks()
        with self.lock:
            try:
                df_files = self.conn.execute("SELECT COUNT(*) AS total_files FROM sync_metadata").fetchdf()
                total_files = int(df_files['total_files'].iloc[0]) if not df_files.empty else 0
            except Exception:
                total_files = 0
        return {
            "total_ticks": total_ticks,
            "total_files": total_files
        }

    def get_dataset_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns comprehensive dataset statistics:
        - first_date, last_date
        - total_days (unique trading days)
        - total_ticks
        - total_files (synced CSV files)
        - yearly_stats: list of {year, days, ticks, start_date, end_date}
        """
        with self.lock:
            try:
                # 1. Overall stats
                where = "WHERE symbol = ? AND timestamp IS NOT NULL" if symbol else "WHERE timestamp IS NOT NULL"
                params = [symbol] if symbol else []
                
                query_overall = f"""
                    SELECT 
                        MIN(timestamp) AS first_date,
                        MAX(timestamp) AS last_date,
                        COUNT(DISTINCT CAST(timestamp AS DATE)) AS total_days,
                        COUNT(*) AS total_ticks
                    FROM ticks
                    {where}
                """
                df_overall = self.conn.execute(query_overall, params).fetchdf()
                
                first_date = str(df_overall['first_date'].iloc[0]) if not df_overall.empty and pd.notna(df_overall['first_date'].iloc[0]) else None
                last_date = str(df_overall['last_date'].iloc[0]) if not df_overall.empty and pd.notna(df_overall['last_date'].iloc[0]) else None
                total_days = int(df_overall['total_days'].iloc[0]) if not df_overall.empty and pd.notna(df_overall['total_days'].iloc[0]) else 0
                total_ticks = int(df_overall['total_ticks'].iloc[0]) if not df_overall.empty and pd.notna(df_overall['total_ticks'].iloc[0]) else 0
                
                # 2. Total synced files
                df_files = self.conn.execute("SELECT COUNT(*) AS total_files FROM sync_metadata").fetchdf()
                total_files = int(df_files['total_files'].iloc[0]) if not df_files.empty else 0

                # 3. Yearly stats
                query_yearly = f"""
                    SELECT 
                        STRFTIME(timestamp, '%Y') AS year,
                        COUNT(DISTINCT CAST(timestamp AS DATE)) AS trading_days,
                        COUNT(*) AS ticks,
                        MIN(timestamp) AS start_date,
                        MAX(timestamp) AS end_date
                    FROM ticks
                    {where}
                    GROUP BY STRFTIME(timestamp, '%Y')
                    ORDER BY year ASC
                """
                df_yearly = self.conn.execute(query_yearly, params).fetchdf()
                
                yearly_stats = []
                if not df_yearly.empty:
                    for _, row in df_yearly.iterrows():
                        yearly_stats.append({
                            "year": str(row['year']),
                            "trading_days": int(row['trading_days']),
                            "ticks": int(row['ticks']),
                            "start_date": str(row['start_date'])[:10] if pd.notna(row['start_date']) else "",
                            "end_date": str(row['end_date'])[:10] if pd.notna(row['end_date']) else ""
                        })

                return {
                    "status": "success",
                    "symbol": symbol or "KOSPI_F",
                    "first_date": first_date,
                    "last_date": last_date,
                    "total_days": total_days,
                    "total_ticks": total_ticks,
                    "total_files": total_files,
                    "yearly_stats": yearly_stats
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e),
                    "first_date": None,
                    "last_date": None,
                    "total_days": 0,
                    "total_ticks": 0,
                    "total_files": 0,
                    "yearly_stats": []
                }

    def get_storage_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Alias for get_dataset_summary for backward compatibility."""
        return self.get_dataset_summary(symbol=symbol)

    def close(self):
        """Close duckdb connection."""
        self.conn.close()
