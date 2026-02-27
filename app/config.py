import os
from dataclasses import dataclass

def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default

@dataclass(frozen=True)
class Settings:
    # Alpaca
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_api_secret: str = os.getenv("ALPACA_API_SECRET", "")
    alpaca_data_feed: str = os.getenv("ALPACA_DATA_FEED", "sip")  # enforce sip
    timezone: str = os.getenv("TIMEZONE", "America/New_York")

    # Scanner
    scan_interval_minutes: int = _get_int("SCAN_INTERVAL_MINUTES", 5)

    # Training
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    train_lookback_days: int = _get_int("TRAIN_LOOKBACK_DAYS", 60)
    train_max_symbols: int = _get_int("TRAIN_MAX_SYMBOLS", 200)

    # Storage
    model_dir: str = os.getenv("MODEL_DIR", "./runtime/model")

    # Debug
    demo_mode: bool = _get_bool("DEMO_MODE", False)
    disable_scheduler: bool = _get_bool("DISABLE_SCHEDULER", False)

    def alpaca_live_ready(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_api_secret) and (not self.demo_mode)

    def normalized_feed(self) -> str:
        # Hard-enforce SIP for consistency (Algo Trader Plus recommended).
        # If ALPACA_DATA_FEED is set to something else, we still use 'sip'.
        return "sip"
