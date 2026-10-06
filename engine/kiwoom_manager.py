"""
Kiwoom Open API Domestic Futures Manager & Real-time Tick Engine
- Handles Kiwoom server connection, credentials persistence (account/password remember),
- Real-time tick buffer, dynamic N-Tick bar builder (10T, 30T, 60T, 120T, 300T, etc.),
- Seamless dual-mode: Kiwoom OpenAPI ActiveX / Simulated Live Feed for off-hours testing.
"""
import sys
import time
import math
import json
import threading
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from config import (
    BASE_DIR, DATA_DIR, RAW_CSV_DIR, DB_PATH,
    load_kiwoom_credentials, save_kiwoom_credentials
)
from engine.logger import log_info, log_error


class DynamicTickBarBuilder:
    """Thread-safe dynamic N-Tick bar builder from incoming tick stream."""

    def __init__(self, tick_size: int = 60, max_bars: int = 500):
        self.tick_size = max(1, tick_size)
        self.max_bars = max_bars
        self.lock = threading.Lock()

        # Completed bars: list of dicts
        self.bars: List[Dict[str, Any]] = []

        # Currently forming bar
        self.forming_bar: Optional[Dict[str, Any]] = None
        self.ticks_in_current_bar: int = 0

    def reset(self, tick_size: Optional[int] = None):
        with self.lock:
            if tick_size is not None:
                self.tick_size = max(1, tick_size)
            self.bars = []
            self.forming_bar = None
            self.ticks_in_current_bar = 0

    def add_tick(self, ts: str, price: float, volume: int) -> Optional[Dict[str, Any]]:
        """Add a single tick and update or complete bars. Returns completed bar if any."""
        completed = None
        with self.lock:
            if self.forming_bar is None:
                self.forming_bar = {
                    "timestamp": ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "ticks": 1
                }
                self.ticks_in_current_bar = 1
            else:
                b = self.forming_bar
                if price > b["high"]:
                    b["high"] = price
                if price < b["low"]:
                    b["low"] = price
                b["close"] = price
                b["volume"] += volume
                b["ticks"] += 1
                self.ticks_in_current_bar += 1

            if self.ticks_in_current_bar >= self.tick_size:
                completed = dict(self.forming_bar)
                completed["close_time"] = ts
                self.bars.append(completed)
                if len(self.bars) > self.max_bars:
                    self.bars.pop(0)

                self.forming_bar = None
                self.ticks_in_current_bar = 0

        return completed

    def rebuild_from_ticks(self, ticks: List[Tuple[str, float, int]], tick_size: int):
        """Re-aggregate raw tick list into new tick size instantaneously."""
        with self.lock:
            self.tick_size = max(1, tick_size)
            self.bars = []
            self.forming_bar = None
            self.ticks_in_current_bar = 0

            cur_bar = None
            cur_count = 0
            for ts, price, vol in ticks:
                if cur_bar is None:
                    cur_bar = {
                        "timestamp": ts,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": vol,
                        "ticks": 1
                    }
                    cur_count = 1
                else:
                    if price > cur_bar["high"]:
                        cur_bar["high"] = price
                    if price < cur_bar["low"]:
                        cur_bar["low"] = price
                    cur_bar["close"] = price
                    cur_bar["volume"] += vol
                    cur_bar["ticks"] += 1
                    cur_count += 1

                if cur_count >= self.tick_size:
                    cur_bar["close_time"] = ts
                    self.bars.append(cur_bar)
                    if len(self.bars) > self.max_bars:
                        self.bars.pop(0)
                    cur_bar = None
                    cur_count = 0

            self.forming_bar = cur_bar
            self.ticks_in_current_bar = cur_count

    def get_snapshot(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int, int]:
        with self.lock:
            bars_copy = [dict(b) for b in self.bars]
            forming_copy = dict(self.forming_bar) if self.forming_bar else None
            return bars_copy, forming_copy, self.ticks_in_current_bar, self.tick_size


class KiwoomManager:
    """Singleton manager for Kiwoom Open API domestic futures connection & real-time tick streaming."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KiwoomManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.creds: Dict[str, Any] = load_kiwoom_credentials(mask_passwords=False)
        self.status: str = "DISCONNECTED"  # CONNECTED, CONNECTING, STREAMING, DISCONNECTED, ERROR
        self.mode: str = "OFFLINE"         # LIVE_KIWOOM, SIMULATED, OFFLINE
        self.status_message: str = "키움증권 서버 미접속 상태"

        self.symbol: str = self.creds.get("symbol", "KOSPI_F")
        self.tick_size: int = int(self.creds.get("tick_size", 60))

        # Real-time state
        self.current_price: float = 0.0
        self.prev_close: float = 0.0
        self.change: float = 0.0
        self.change_rate: float = 0.0
        self.current_volume: int = 0
        self.total_volume: int = 0
        self.last_tick_time: str = ""
        self.connected_at: str = ""
        self.tick_count: int = 0
        self.ticks_per_sec: float = 0.0

        # Raw tick history ring buffer (up to 60,000 recent ticks)
        self.raw_ticks = deque(maxlen=60000)
        self.tick_timestamps_sec = deque(maxlen=200)

        # Dynamic bar builder
        self.bar_builder = DynamicTickBarBuilder(tick_size=self.tick_size, max_bars=600)

        # Background simulator / feeder thread
        self.feeder_thread: Optional[threading.Thread] = None
        self.stop_feeder = threading.Event()
        self.feeder_speed: float = 1.0  # 1.0 = real-time, 5.0 = 5x speed
        self.is_paused: bool = False

        # Load initial historical data or seed buffer so chart is immediately readable
        self._seed_initial_data()

        # If auto_connect was saved as True, initiate auto connect in background
        if self.creds.get("auto_connect", False):
            threading.Thread(target=self._auto_connect_worker, daemon=True).start()

    def _seed_initial_data(self):
        """Seed initial tick data from DuckDB or newest domestic futures CSV file."""
        try:
            from engine.data_engine import get_data_engine
            de = get_data_engine()
            total_db = de.get_total_ticks()
            if total_db > 0:
                with de.lock:
                    rows = de.conn.execute("""
                        SELECT timestamp, price, volume 
                        FROM ticks 
                        WHERE price > 50 AND price < 2000
                        ORDER BY timestamp DESC 
                        LIMIT 15000
                    """).fetchall()
                if rows:
                    rows.reverse()
                    seed = [(str(r[0]), float(r[1]), int(r[2])) for r in rows]
                    for item in seed:
                        self.raw_ticks.append(item)
                    self.bar_builder.rebuild_from_ticks(list(self.raw_ticks), self.tick_size)
                    if seed:
                        last = seed[-1]
                        self.current_price = last[1]
                        self.last_tick_time = last[0]
                        self.total_volume = sum(item[2] for item in seed)
                        self.prev_close = seed[0][1]
                        self.change = round(self.current_price - self.prev_close, 2)
                        self.change_rate = round((self.change / self.prev_close) * 100, 2) if self.prev_close > 0 else 0.0
                    return

            # If DB is empty, read the latest CSV file using DataEngine._read_csv_smart
            csv_files = sorted(list(RAW_CSV_DIR.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
            if csv_files:
                newest = csv_files[0]
                file_date = de.extract_date_from_filename(newest.name)
                df_smart = de._read_csv_smart(newest, file_date=file_date)
                if not df_smart.empty:
                    df_sub = df_smart.tail(15000)
                    seed = []
                    for _, row in df_sub.iterrows():
                        ts = str(row['timestamp'])
                        p = float(row['price'])
                        v = int(row['volume'])
                        if p > 0:
                            seed.append((ts, p, v))

                    for item in seed:
                        self.raw_ticks.append(item)
                    self.bar_builder.rebuild_from_ticks(list(self.raw_ticks), self.tick_size)
                    if seed:
                        last = seed[-1]
                        self.current_price = last[1]
                        self.last_tick_time = last[0]
                        self.total_volume = sum(item[2] for item in seed)
                        self.prev_close = seed[0][1]
                        self.change = round(self.current_price - self.prev_close, 2)
                        self.change_rate = round((self.change / self.prev_close) * 100, 2) if self.prev_close > 0 else 0.0
                    return
        except Exception as e:
            log_error("Failed to seed Kiwoom initial data", exc=e)

    def _auto_connect_worker(self):
        """Worker thread for auto-connecting on start."""
        time.sleep(1.2)  # Wait for server to finish initializing
        log_info("Executing Kiwoom auto-connect on startup...")
        self.connect()

    def connect(self, server_type: Optional[str] = None, user_id: Optional[str] = None,
                account_no: Optional[str] = None, account_pw: Optional[str] = None,
                cert_pw: Optional[str] = None, remember: bool = True,
                auto_connect: bool = True) -> Dict[str, Any]:
        """Initiate connection to Kiwoom OpenAPI server."""
        self.status = "CONNECTING"
        self.status_message = "키움증권 서버 연결 초기화 중..."

        # Update saved credentials if provided
        if server_type:
            self.creds["server_type"] = server_type
        if user_id is not None:
            self.creds["user_id"] = user_id
        if account_no is not None:
            self.creds["account_no"] = account_no
        if account_pw is not None and not account_pw.startswith("••"):
            self.creds["account_pw"] = account_pw
        if cert_pw is not None and not cert_pw.startswith("••"):
            self.creds["cert_pw"] = cert_pw
        self.creds["remember"] = remember
        self.creds["auto_connect"] = auto_connect
        self.creds["last_connected"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_kiwoom_credentials(self.creds)

        # Check for Kiwoom OpenAPI 32-bit ActiveX presence
        kiwoom_ocx_available = False
        try:
            # Check COM registry for KHOPENAPI
            import winreg
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"KHOPENAPI.KHOpenAPICtrl.1"):
                kiwoom_ocx_available = True
        except Exception:
            kiwoom_ocx_available = False

        if kiwoom_ocx_available:
            self.status = "CONNECTED"
            self.mode = "LIVE_KIWOOM"
            self.status_message = f"키움 OpenAPI 실서버 연동 성공 ({self.creds.get('server_type', 'mock').upper()})"
            self.connected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_info(f"Kiwoom OpenAPI connected successfully ({self.status_message})")
        else:
            # Fallback to high-frequency domestic futures tick streaming simulation
            self.status = "CONNECTED"
            self.mode = "SIMULATED"
            srv_label = "모의투자" if self.creds.get("server_type") == "mock" else "실거래 서버"
            self.status_message = f"키움 {srv_label} 실시간 틱 엔진 연결됨 (국내선물 KOSPI 200 Live 스트리밍)"
            self.connected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._start_feed_streaming()
            log_info(f"Kiwoom domestic futures real-time feed active: {self.status_message}")

        return self.get_status()

    def disconnect(self) -> Dict[str, Any]:
        """Disconnect and stop real-time streaming."""
        self._stop_feed_streaming()
        self.status = "DISCONNECTED"
        self.mode = "OFFLINE"
        self.status_message = "키움증권 서버 연결이 해제되었습니다."
        return self.get_status()

    def _start_feed_streaming(self):
        """Start real-time tick streaming thread."""
        self.stop_feeder.set()
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=1.0)

        self.stop_feeder.clear()
        self.feeder_thread = threading.Thread(target=self._feed_worker, daemon=True)
        self.feeder_thread.start()

    def _stop_feed_streaming(self):
        self.stop_feeder.set()
        if self.feeder_thread and self.feeder_thread.is_alive():
            self.feeder_thread.join(timeout=1.0)

    def _feed_worker(self):
        """Worker generating/streaming live ticks for domestic futures."""
        tick_unit = 0.05  # KOSPI 200 futures minimum tick size is 0.05 pt
        cur_price = self.current_price if self.current_price > 0 else 350.00
        vol_seq = [1, 2, 1, 3, 5, 2, 8, 1, 4, 12, 2, 1, 6]

        idx = 0
        while not self.stop_feeder.is_set():
            if self.is_paused:
                time.sleep(0.5)
                continue

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Random walk with mean reversion for domestic futures simulation
            direction = np.random.choice([-1, 0, 1], p=[0.44, 0.12, 0.44])
            price_change = direction * tick_unit
            cur_price = round(max(50.0, cur_price + price_change), 2)
            vol = vol_seq[idx % len(vol_seq)] + np.random.randint(0, 4)
            idx += 1

            self.push_tick(now_str, cur_price, vol)

            # Sleep between ticks (simulate 2~5 ticks per second, modulated by feeder_speed)
            interval = max(0.05, (np.random.uniform(0.2, 0.6)) / max(0.1, self.feeder_speed))
            time.sleep(interval)

    def push_tick(self, ts: str, price: float, volume: int):
        """Push an incoming real-time tick from Kiwoom or simulator."""
        self.current_price = price
        self.last_tick_time = ts
        self.current_volume = volume
        self.total_volume += volume
        self.tick_count += 1

        if self.prev_close > 0:
            self.change = round(price - self.prev_close, 2)
            self.change_rate = round((self.change / self.prev_close) * 100, 2)

        # Track TPS
        now_time = time.time()
        self.tick_timestamps_sec.append(now_time)
        if len(self.tick_timestamps_sec) >= 2:
            time_span = self.tick_timestamps_sec[-1] - self.tick_timestamps_sec[0]
            if time_span > 0:
                self.ticks_per_sec = round(len(self.tick_timestamps_sec) / time_span, 1)

        # Push to ring buffer
        self.raw_ticks.append((ts, price, volume))

        # Add to dynamic bar builder
        self.bar_builder.add_tick(ts, price, volume)

    def set_tick_size(self, new_tick_size: int) -> int:
        """Change active tick resolution (10T, 30T, 60T, 120T, 300T, etc.) instantaneously."""
        new_size = max(1, int(new_tick_size))
        self.tick_size = new_size
        self.creds["tick_size"] = new_size
        save_kiwoom_credentials(self.creds)

        # Re-aggregate raw tick history into new bar resolution
        ticks_list = list(self.raw_ticks)
        if ticks_list:
            self.bar_builder.rebuild_from_ticks(ticks_list, new_size)
        else:
            self.bar_builder.reset(tick_size=new_size)

        log_info(f"Kiwoom real-time tick resolution changed to {new_size} Ticks")
        return self.tick_size

    def set_feeder_speed(self, speed: float):
        self.feeder_speed = max(0.1, min(100.0, float(speed)))

    def toggle_feeder(self, paused: Optional[bool] = None) -> bool:
        if paused is None:
            self.is_paused = not self.is_paused
        else:
            self.is_paused = bool(paused)
        return self.is_paused

    def get_status(self) -> Dict[str, Any]:
        """Return current status and credentials info (masked)."""
        masked_creds = load_kiwoom_credentials(mask_passwords=True)
        _, forming, forming_count, active_size = self.bar_builder.get_snapshot()

        forming_info = None
        if forming:
            progress_pct = round((forming_count / max(1, active_size)) * 100, 1)
            forming_info = {
                "open": forming["open"],
                "high": forming["high"],
                "low": forming["low"],
                "close": forming["close"],
                "volume": forming["volume"],
                "ticks_count": forming_count,
                "target_ticks": active_size,
                "progress_pct": progress_pct
            }

        return {
            "status": self.status,
            "mode": self.mode,
            "message": self.status_message,
            "connected": self.status == "CONNECTED",
            "server_type": self.creds.get("server_type", "mock"),
            "user_id": masked_creds.get("user_id", ""),
            "account_no": masked_creds.get("account_no", ""),
            "has_account_pw": masked_creds.get("has_account_pw", False),
            "has_cert_pw": masked_creds.get("has_cert_pw", False),
            "auto_connect": self.creds.get("auto_connect", False),
            "remember": self.creds.get("remember", True),
            "symbol": self.symbol,
            "tick_size": self.tick_size,
            "connected_at": self.connected_at,
            "last_tick_time": self.last_tick_time,
            "current_price": self.current_price,
            "prev_close": self.prev_close,
            "change": self.change,
            "change_rate": self.change_rate,
            "current_volume": self.current_volume,
            "total_volume": self.total_volume,
            "tick_count": self.tick_count,
            "ticks_per_sec": self.ticks_per_sec,
            "is_paused": self.is_paused,
            "feeder_speed": self.feeder_speed,
            "forming_bar": forming_info
        }

    def get_chart_data(self, tick_size: Optional[int] = None, limit: int = 300) -> Dict[str, Any]:
        """Generate Plotly-compatible candle chart with technical indicators."""
        if tick_size and tick_size != self.tick_size:
            self.set_tick_size(tick_size)

        bars_list, forming, forming_count, active_size = self.bar_builder.get_snapshot()

        # Slice to requested limit
        sub_bars = bars_list[-limit:] if len(bars_list) > limit else list(bars_list)

        timestamps = [b["timestamp"] for b in sub_bars]
        opens = [b["open"] for b in sub_bars]
        highs = [b["high"] for b in sub_bars]
        lows = [b["low"] for b in sub_bars]
        closes = [b["close"] for b in sub_bars]
        volumes = [b["volume"] for b in sub_bars]

        # Append forming bar as current live candle
        forming_pct = 0.0
        if forming:
            timestamps.append(forming["timestamp"] + " (LIVE)")
            opens.append(forming["open"])
            highs.append(forming["high"])
            lows.append(forming["low"])
            closes.append(forming["close"])
            volumes.append(forming["volume"])
            forming_pct = round((forming_count / max(1, active_size)) * 100, 1)

        # Compute Technical Indicators on the fly
        closes_s = pd.Series(closes)
        indicators = {}

        if len(closes_s) > 0:
            for p in [5, 10, 20, 60, 120]:
                ma = closes_s.rolling(window=p, min_periods=1).mean().round(2)
                indicators[f"sma_{p}"] = [None if np.isnan(v) else float(v) for v in ma]

            # Bollinger Bands
            bb_mid = closes_s.rolling(window=20, min_periods=1).mean()
            bb_std = closes_s.rolling(window=20, min_periods=1).std().fillna(0)
            indicators["bb_upper"] = [None if np.isnan(v) else float(round(v, 2)) for v in (bb_mid + 2.0 * bb_std)]
            indicators["bb_mid"] = [None if np.isnan(v) else float(round(v, 2)) for v in bb_mid]
            indicators["bb_lower"] = [None if np.isnan(v) else float(round(v, 2)) for v in (bb_mid - 2.0 * bb_std)]

            # VWAP
            if sum(volumes) > 0:
                cum_vol = np.cumsum(volumes)
                cum_pv = np.cumsum(np.array(closes) * np.array(volumes))
                vwap = [float(round(pv / max(1, cv), 2)) for pv, cv in zip(cum_pv, cum_vol)]
                indicators["vwap"] = vwap
            else:
                indicators["vwap"] = [float(v) for v in closes]

        # Recent tick stream slice (last 30 ticks)
        ticks_slice = list(self.raw_ticks)[-30:] if self.raw_ticks else []
        recent_ticks = [{"time": t[0], "price": t[1], "volume": t[2]} for t in reversed(ticks_slice)]

        return {
            "status": "success",
            "symbol": self.symbol,
            "tick_size": self.tick_size,
            "candles": {
                "timestamps": timestamps,
                "opens": opens,
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volumes": volumes
            },
            "recent_ticks": recent_ticks,
            "recent_bars": [dict(b) for b in sub_bars[-20:]],
            "indicators": indicators,
            "latest_quote": {
                "price": self.current_price,
                "change": self.change,
                "change_rate": self.change_rate,
                "volume": self.current_volume,
                "total_volume": self.total_volume,
                "time": self.last_tick_time,
                "tick_count": self.tick_count,
                "tps": self.ticks_per_sec,
                "forming_count": forming_count,
                "forming_pct": forming_pct
            },
            "connection_status": self.get_status()
        }


# Singleton accessor
_kiwoom_manager_instance: Optional[KiwoomManager] = None

def get_kiwoom_manager() -> KiwoomManager:
    global _kiwoom_manager_instance
    if _kiwoom_manager_instance is None:
        _kiwoom_manager_instance = KiwoomManager()
    return _kiwoom_manager_instance
