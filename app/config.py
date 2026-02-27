from __future__ import annotations
import os
from dataclasses import dataclass

def _bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1","true","yes","y","on")

def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip()=="":
        return default
    try:
        return int(v)
    except Exception:
        return default

@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_data_feed: str
    timezone: str

    scan_interval_minutes: int

    admin_password: str
    train_lookback_days: int
    train_max_symbols: int

    model_dir: str

    demo_mode: bool
    disable_scheduler: bool

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            alpaca_api_key=os.getenv("ALPACA_API_KEY",""),
            alpaca_api_secret=os.getenv("ALPACA_API_SECRET",""),
            alpaca_data_feed=os.getenv("ALPACA_DATA_FEED","sip"),
            timezone=os.getenv("TIMEZONE","America/New_York"),
            scan_interval_minutes=max(1, _int("SCAN_INTERVAL_MINUTES", 5)),
            admin_password=os.getenv("ADMIN_PASSWORD",""),
            train_lookback_days=max(5, _int("TRAIN_LOOKBACK_DAYS", 60)),
            # 0 or <0 means "no cap"
            train_max_symbols=_int("TRAIN_MAX_SYMBOLS", 0),
            model_dir=os.getenv("MODEL_DIR","./runtime/model"),
            demo_mode=_bool("DEMO_MODE", False),
            disable_scheduler=_bool("DISABLE_SCHEDULER", False),
        )

    def normalized_feed(self) -> str:
        # Enforce SIP for consistent historical+live (Algo Trader Plus)
        return "sip"
