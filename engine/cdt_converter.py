"""
Kiwoom .cdt Binary Converter Engine
- Converts OHLCV & Tick DataFrames to Kiwoom HTS / Youngwoom .cdt binary files.
- Converts .cdt binary files back to DataFrames / CSV.
- Supports single file and batch folder conversion.
"""
import struct
import io
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class CDTConverter:
    """
    Kiwoom .cdt format handler:
    Header: <2i (h1=1, h2=num_rows) -> 8 bytes
    Record: <8d (date_time, open, high, low, close, volume, trading_val, open_int) -> 64 bytes
    """

    @staticmethod
    def dataframe_to_cdt_bytes(df: pd.DataFrame) -> bytes:
        """
        Converts DataFrame to Kiwoom .cdt binary bytes.
        Input df can have:
        - timestamp / open_time / close_time (datetime or str)
        - open, high, low, close (or price for single ticks)
        - volume
        - optional: trading_value, open_interest
        """
        if df.empty:
            return struct.pack('<2i', 1, 0)

        df = df.copy()

        # 1. Normalize Timestamp to 14-digit numeric double (YYYYMMDDHHMMSS)
        ts_col = None
        for col in ['timestamp', 'open_time', 'close_time', '일자 / 시간', '일자', 'time', 'datetime']:
            if col in df.columns:
                ts_col = col
                break

        if ts_col:
            ts_series = pd.to_datetime(df[ts_col], errors='coerce')
        else:
            ts_series = pd.date_range(end=datetime.now(), periods=len(df), freq='1s')

        # Format as numeric YYYYMMDDHHMMSS float
        def _ts_to_num(dt):
            if pd.isna(dt):
                return 20240101090000.0
            return float(dt.strftime('%Y%m%d%H%M%S'))

        date_nums = ts_series.apply(_ts_to_num).values

        # 2. Normalize Price & Volume
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            opens = df['open'].fillna(0.0).astype(float).values
            highs = df['high'].fillna(0.0).astype(float).values
            lows = df['low'].fillna(0.0).astype(float).values
            closes = df['close'].fillna(0.0).astype(float).values
        elif 'price' in df.columns:
            p = df['price'].fillna(0.0).astype(float).values
            opens = p
            highs = p
            lows = p
            closes = p
        elif '종가' in df.columns:
            opens = df['시가'].fillna(df['종가']).astype(float).values
            highs = df['고가'].fillna(df['종가']).astype(float).values
            lows = df['저가'].fillna(df['종가']).astype(float).values
            closes = df['종가'].astype(float).values
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                p = df[num_cols[0]].fillna(0.0).astype(float).values
                opens, highs, lows, closes = p, p, p, p
            else:
                opens = np.zeros(len(df), dtype=float)
                highs, lows, closes = opens, opens, opens

        if 'volume' in df.columns:
            volumes = df['volume'].fillna(0).astype(float).values
        elif '거래량' in df.columns:
            volumes = df['거래량'].fillna(0).astype(float).values
        else:
            volumes = np.ones(len(df), dtype=float)

        if 'trading_value' in df.columns:
            values = df['trading_value'].fillna(0.0).astype(float).values
        elif '거래대금' in df.columns:
            values = df['거래대금'].fillna(0.0).astype(float).values
        else:
            values = np.zeros(len(df), dtype=float)

        if 'open_interest' in df.columns:
            open_ints = df['open_interest'].fillna(0.0).astype(float).values
        elif '미결제약정' in df.columns:
            open_ints = df['미결제약정'].fillna(0.0).astype(float).values
        else:
            open_ints = np.zeros(len(df), dtype=float)

        num_rows = len(df)
        header = struct.pack('<2i', 1, num_rows)
        
        record_bytes = bytearray()
        for i in range(num_rows):
            record_bytes.extend(
                struct.pack('<8d', 
                            float(date_nums[i]), 
                            float(opens[i]), 
                            float(highs[i]), 
                            float(lows[i]), 
                            float(closes[i]), 
                            float(volumes[i]), 
                            float(values[i]), 
                            float(open_ints[i]))
            )

        return header + bytes(record_bytes)

    @staticmethod
    def cdt_bytes_to_dataframe(file_content: bytes) -> pd.DataFrame:
        """
        Parses Kiwoom .cdt binary bytes back into a Pandas DataFrame.
        """
        if len(file_content) < 8:
            return pd.DataFrame()

        try:
            h1, h2 = struct.unpack('<2i', file_content[:8])
            data_len = len(file_content) - 8

            record_size = 64
            fmt = '<8d'
            if h1 == 3 and data_len % 80 == 0:
                record_size = 80
                fmt = '<10d'
            elif data_len % 64 == 0:
                record_size = 64
                fmt = '<8d'
            elif data_len % 80 == 0:
                record_size = 80
                fmt = '<10d'
            else:
                record_size = 64
                fmt = '<8d'

            num_rows = data_len // record_size
            if num_rows <= 0:
                return pd.DataFrame()

            data = []
            for i in range(num_rows):
                offset = 8 + i * record_size
                unpacked = struct.unpack(fmt, file_content[offset : offset + record_size])

                date_val = unpacked[0]
                date_str = str(int(date_val))

                if len(date_str) == 14:
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {date_str[8:10]}:{date_str[10:12]}:{date_str[12:14]}"
                elif len(date_str) == 8:
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00"
                else:
                    formatted_date = date_str

                data.append({
                    'timestamp': formatted_date,
                    'open': round(float(unpacked[1]), 2),
                    'high': round(float(unpacked[2]), 2),
                    'low': round(float(unpacked[3]), 2),
                    'close': round(float(unpacked[4]), 2),
                    'volume': int(unpacked[5]) if len(unpacked) > 5 else 0,
                    'trading_value': float(unpacked[6]) if len(unpacked) > 6 else 0.0,
                    'open_interest': float(unpacked[7]) if len(unpacked) > 7 else 0.0
                })

            df = pd.DataFrame(data)
            return df
        except Exception as e:
            print(f"[CDTConverter] Error parsing CDT bytes: {e}")
            return pd.DataFrame()

    @classmethod
    def convert_csv_to_cdt_file(cls, csv_path: Union[str, Path], output_cdt_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Converts a single CSV file to a .cdt binary file.
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        if output_cdt_path is None:
            output_cdt_path = csv_file.with_suffix('.cdt')
        else:
            output_cdt_path = Path(output_cdt_path)

        output_cdt_path.parent.mkdir(parents=True, exist_ok=True)

        from engine.data_engine import get_data_engine
        de = get_data_engine()
        f_date = de.extract_date_from_filename(csv_file.name)
        df_clean = de._read_csv_smart(csv_file, file_date=f_date)

        if df_clean.empty:
            df_clean = pd.read_csv(csv_file)

        cdt_bytes = cls.dataframe_to_cdt_bytes(df_clean)
        with open(output_cdt_path, 'wb') as f:
            f.write(cdt_bytes)

        return output_cdt_path

    @classmethod
    def batch_convert_target(cls, target_path: Union[str, Path, List[Union[str, Path]]], target_folder: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Batch converts CSV file(s) or folder into .cdt files.
        """
        if isinstance(target_path, (list, tuple)):
            csv_files = [Path(p) for p in target_path if Path(p).exists() and Path(p).suffix.lower() == '.csv']
            if not csv_files:
                return []
            parent_dir = csv_files[0].parent
        else:
            p = Path(target_path)
            if p.is_file():
                csv_files = [p]
                parent_dir = p.parent
            elif p.is_dir():
                csv_files = sorted(list(p.glob("*.[cC][sS][vV]")), key=lambda x: x.name)
                parent_dir = p
            else:
                return []

        if target_folder is None:
            tgt = parent_dir / "cdt_converted"
        else:
            tgt = Path(target_folder)

        tgt.mkdir(parents=True, exist_ok=True)

        results = []
        for cf in csv_files:
            out_file = tgt / f"{cf.stem}.cdt"
            try:
                cls.convert_csv_to_cdt_file(cf, out_file)
                results.append(out_file)
            except Exception as e:
                print(f"[CDT Batch Error on {cf.name}]: {e}")

        return results
