"""
System Configuration Module
- Handles paths, database settings, trading parameters (commission, slippage), and default indicators.
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Base directories (Supports PyInstaller frozen executable)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CSV_DIR = BASE_DIR / "csv"
RAW_CSV_DIR = CSV_DIR  # Alias for backward compatibility
DB_PATH = DATA_DIR / "market_data.duckdb"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHARTS_DIR = OUTPUT_DIR / "charts"
LOGS_DIR = OUTPUT_DIR / "logs"

SETTINGS_PATH = DATA_DIR / "settings.json"
KIWOOM_CONFIG_PATH = DATA_DIR / "kiwoom_credentials.json"
CACHE_DIR = DATA_DIR / "cache" / "resampled"

# Ensure directories exist
for path in [DATA_DIR, CSV_DIR, OUTPUT_DIR, REPORTS_DIR, CHARTS_DIR, LOGS_DIR, CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def load_settings() -> dict:
    """Load persistent settings (e.g. last used CSV folder)."""
    import json
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_csv_folder": "csv"}

def save_settings(settings: dict):
    """Save persistent settings."""
    import json
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _obfuscate(text: str) -> str:
    import base64
    if not text:
        return ""
    key = 0x5A
    encoded = bytes([b ^ key for b in text.encode('utf-8')])
    return base64.b64encode(encoded).decode('ascii')

def _deobfuscate(encoded_text: str) -> str:
    import base64
    if not encoded_text:
        return ""
    try:
        raw = base64.b64decode(encoded_text.encode('ascii'))
        key = 0x5A
        return bytes([b ^ key for b in raw]).decode('utf-8', errors='ignore')
    except Exception:
        return ""

def load_kiwoom_credentials(mask_passwords: bool = False) -> dict:
    """Load saved Kiwoom credentials and connection settings."""
    import json
    default_creds = {
        "server_type": "mock",      # 'mock' (모의투자) or 'real' (실서버)
        "user_id": "",
        "account_no": "",
        "account_pw": "",
        "cert_pw": "",
        "auto_connect": False,
        "remember": True,
        "symbol": "KOSPI_F",
        "tick_size": 60,
        "last_connected": ""
    }
    if KIWOOM_CONFIG_PATH.exists():
        try:
            with open(KIWOOM_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_creds.update(data)
                default_creds["account_pw"] = _deobfuscate(default_creds.get("account_pw", ""))
                default_creds["cert_pw"] = _deobfuscate(default_creds.get("cert_pw", ""))
        except Exception:
            pass

    if mask_passwords:
        result = dict(default_creds)
        result["has_account_pw"] = bool(result.get("account_pw"))
        result["has_cert_pw"] = bool(result.get("cert_pw"))
        result["account_pw"] = "********" if result["has_account_pw"] else ""
        result["cert_pw"] = "********" if result["has_cert_pw"] else ""
        return result
    return default_creds

def save_kiwoom_credentials(creds: dict):
    """Save Kiwoom credentials with obfuscation if remember is checked."""
    import json
    data_to_save = dict(creds)
    # If not remembering credentials, blank out passwords
    if not data_to_save.get("remember", True):
        data_to_save["account_pw"] = ""
        data_to_save["cert_pw"] = ""
    else:
        # Obfuscate passwords before writing
        raw_pw = data_to_save.get("account_pw", "")
        raw_cert = data_to_save.get("cert_pw", "")
        # If UI sent masked bullets or asterisks, keep previously stored password
        if raw_pw.startswith("••") or raw_pw.startswith("**"):
            prev = load_kiwoom_credentials(mask_passwords=False)
            data_to_save["account_pw"] = _obfuscate(prev.get("account_pw", ""))
        else:
            data_to_save["account_pw"] = _obfuscate(raw_pw)

        if raw_cert.startswith("••") or raw_cert.startswith("**"):
            prev = load_kiwoom_credentials(mask_passwords=False)
            data_to_save["cert_pw"] = _obfuscate(prev.get("cert_pw", ""))
        else:
            data_to_save["cert_pw"] = _obfuscate(raw_cert)

    try:
        with open(KIWOOM_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

@dataclass
class BacktestConfig:
    """Trading and Backtesting Configuration"""
    initial_capital: float = 100_000_000.0  # 기본 자본금: 1억 원 (KRW)
    commission_rate: float = 0.00003        # 수수료: 0.003% (선물/주식 대략적 기준)
    slippage_ticks: float = 1.0             # 체결 슬리피지: 1틱
    tick_value: float = 0.05                # 지수선물 호가단위 (예: KOSPI200 선물 0.05pt)
    multiplier: float = 250_000.0           # 1포인트당 가치 (KOSPI200 기준 250,000원, 주식은 1.0)
    eod_close_time: str = "15:35:00"        # 데이트레이딩 당일 장마감 청산 시각
    allow_short: bool = True                # 공매도/숏 포지션 허용 여부
    
@dataclass
class ResampleConfig:
    """Tick Resampling Configuration"""
    default_tick_sizes: List[int] = field(default_factory=lambda: [1000, 2000, 3000, 5000, 10000])
    start_tick: int = 1000
    end_tick: int = 10000
    step_tick: int = 1000
