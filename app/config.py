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

def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip()=="":
        return default
    try:
        return float(v)
    except Exception:
        return default

@dataclass(frozen=True)
class Settings:
    # Alpaca
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_data_feed: str
    timezone: str

    # Scanner
    scan_interval_minutes: int
    min_bars_5m: int

    # Training
    admin_password: str
    train_lookback_days: int
    train_max_symbols: int

    # Storage
    model_dir: str

    # Debug
    demo_mode: bool
    disable_scheduler: bool
    debug_password: str

    # ToD-RVOL
    tod_rvol_lookback_days: int
    tod_rvol_min_days: int

    # Liquidity risk thresholds
    liq_rolling_bars: int
    liq_dvol_min_usd: float
    liq_range_pct_max: float
    liq_wick_atr_max: float

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            alpaca_api_key=os.getenv("ALPACA_API_KEY",""),
            alpaca_api_secret=os.getenv("ALPACA_API_SECRET",""),
            alpaca_data_feed=os.getenv("ALPACA_DATA_FEED","sip"),
            timezone=os.getenv("TIMEZONE","America/New_York"),

            scan_interval_minutes=max(1, _int("SCAN_INTERVAL_MINUTES", 5)),
            min_bars_5m=max(3, _int("MIN_BARS_5M", 7)),

            admin_password=os.getenv("ADMIN_PASSWORD",""),
            train_lookback_days=max(5, _int("TRAIN_LOOKBACK_DAYS", 60)),
            train_max_symbols=_int("TRAIN_MAX_SYMBOLS", 0),  # <=0 means no cap

            model_dir=os.getenv("MODEL_DIR","./runtime/model"),

            demo_mode=_bool("DEMO_MODE", False),
            disable_scheduler=_bool("DISABLE_SCHEDULER", False),
            debug_password=os.getenv("DEBUG_PASSWORD",""),

            tod_rvol_lookback_days=max(5, _int("TOD_RVOL_LOOKBACK_DAYS", 20)),
            tod_rvol_min_days=max(1, _int("TOD_RVOL_MIN_DAYS", 8)),

            liq_rolling_bars=max(3, _int("LIQ_ROLLING_BARS", 12)),
            liq_dvol_min_usd=_float("LIQ_DVOL_MIN_USD", 2_000_000.0),
            liq_range_pct_max=_float("LIQ_RANGE_PCT_MAX", 0.012),
            liq_wick_atr_max=_float("LIQ_WICK_ATR_MAX", 0.8),
        )

    def normalized_feed(self) -> str:
        # Enforce SIP for consistent historical+live (Algo Trader Plus)
        return "sip"

    def debug_gate_password(self) -> str:
        return self.debug_password or self.admin_password
