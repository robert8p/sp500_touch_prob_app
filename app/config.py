from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _csv_floats(name: str, default_csv: str) -> List[float]:
    raw = os.getenv(name, default_csv)
    vals: List[float] = []
    for part in (raw or "").split(','):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except Exception:
            continue
    return vals or [float(x) for x in default_csv.split(',')]


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_data_feed: str
    timezone: str
    scan_interval_minutes: int
    min_bars_5m: int
    admin_password: str
    train_lookback_days: int
    train_max_symbols: int
    calib_min_bucket_samples: int
    enet_c_values: List[float]
    enet_l1_values: List[float]
    prior_alpha_values: List[float]
    selection_min_count_70: int
    selection_min_count_75: int
    selection_min_precision_70: float
    selection_min_precision_75: float
    tail_not_ready_prob_cap: float
    model_dir: str
    demo_mode: bool
    disable_scheduler: bool
    debug_password: str
    tod_rvol_lookback_days: int
    tod_rvol_min_days: int
    liq_rolling_bars: int
    liq_dvol_min_usd: float
    liq_range_pct_max: float
    liq_wick_atr_max: float
    stage1_candidate_cap: int
    stage1_min_score: float
    stage1_min_rvol: float
    stage1_min_dollar_volume_mult: float
    stage1_min_minutes_since_open: int
    stage1_min_minutes_to_close: int
    blocked_ret20d_max: float
    blocked_ret60d_max: float
    blocked_dist50dma_max: float
    blocked_ret_since_open_max: float
    blocked_damage_from_high_atr_min: float
    blocked_below_vwap_frac_min: float
    blocked_prob_cap: float
    event_gap_abs_min: float
    event_rvol_min: float
    event_range_pct_min: float
    event_prob_cap: float
    downside_high_threshold: float
    downside_medium_threshold: float
    downside_prob_cap_high: float
    downside_prob_cap_medium: float
    uncertainty_z_thresh: float
    uncertainty_extreme_features_min: int
    uncertainty_prob_cap: float
    diag_track_min_prob: float
    diag_track_top_n: int
    diag_cap_buffer: float
    diag_sane_mae_pct: float
    diag_held_minutes: int
    diag_held_fraction: float

    @staticmethod
    def from_env() -> 'Settings':
        return Settings(
            alpaca_api_key=os.getenv('ALPACA_API_KEY', ''),
            alpaca_api_secret=os.getenv('ALPACA_API_SECRET', ''),
            alpaca_data_feed=os.getenv('ALPACA_DATA_FEED', 'sip'),
            timezone=os.getenv('TIMEZONE', 'America/New_York'),
            scan_interval_minutes=max(1, _int('SCAN_INTERVAL_MINUTES', 5)),
            min_bars_5m=max(3, _int('MIN_BARS_5M', 7)),
            admin_password=os.getenv('ADMIN_PASSWORD', ''),
            train_lookback_days=max(5, _int('TRAIN_LOOKBACK_DAYS', 60)),
            train_max_symbols=_int('TRAIN_MAX_SYMBOLS', 0),
            calib_min_bucket_samples=max(50, _int('CALIB_MIN_BUCKET_SAMPLES', 200)),
            enet_c_values=_csv_floats('ENET_C_VALUES', '0.25,0.5,1.0,2.0'),
            enet_l1_values=_csv_floats('ENET_L1_VALUES', '0.0,0.25,0.5,0.75'),
            prior_alpha_values=_csv_floats('PRIOR_ALPHA_VALUES', '0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85'),
            selection_min_count_70=max(1, _int('SELECTION_MIN_COUNT_70', 6)),
            selection_min_count_75=max(1, _int('SELECTION_MIN_COUNT_75', 8)),
            selection_min_precision_70=_float('SELECTION_MIN_PRECISION_70', 0.58),
            selection_min_precision_75=_float('SELECTION_MIN_PRECISION_75', 0.62),
            tail_not_ready_prob_cap=_float('TAIL_NOT_READY_PROB_CAP', 0.74),
            model_dir=os.getenv('MODEL_DIR', './runtime/model'),
            demo_mode=_bool('DEMO_MODE', False),
            disable_scheduler=_bool('DISABLE_SCHEDULER', False),
            debug_password=os.getenv('DEBUG_PASSWORD', ''),
            tod_rvol_lookback_days=max(5, _int('TOD_RVOL_LOOKBACK_DAYS', 20)),
            tod_rvol_min_days=max(1, _int('TOD_RVOL_MIN_DAYS', 8)),
            liq_rolling_bars=max(3, _int('LIQ_ROLLING_BARS', 12)),
            liq_dvol_min_usd=_float('LIQ_DVOL_MIN_USD', 2_000_000.0),
            liq_range_pct_max=_float('LIQ_RANGE_PCT_MAX', 0.012),
            liq_wick_atr_max=_float('LIQ_WICK_ATR_MAX', 0.8),
            stage1_candidate_cap=max(20, _int('STAGE1_CANDIDATE_CAP', 120)),
            stage1_min_score=_float('STAGE1_MIN_SCORE', 2.0),
            stage1_min_rvol=_float('STAGE1_MIN_RVOL', 0.85),
            stage1_min_dollar_volume_mult=_float('STAGE1_MIN_DOLLAR_VOLUME_MULT', 0.7),
            stage1_min_minutes_since_open=max(5, _int('STAGE1_MIN_MINUTES_SINCE_OPEN', 40)),
            stage1_min_minutes_to_close=max(10, _int('STAGE1_MIN_MINUTES_TO_CLOSE', 35)),
            blocked_ret20d_max=_float('BLOCKED_RET20D_MAX', -0.08),
            blocked_ret60d_max=_float('BLOCKED_RET60D_MAX', -0.15),
            blocked_dist50dma_max=_float('BLOCKED_DIST50DMA_MAX', -0.06),
            blocked_ret_since_open_max=_float('BLOCKED_RET_SINCE_OPEN_MAX', -0.025),
            blocked_damage_from_high_atr_min=_float('BLOCKED_DAMAGE_FROM_HIGH_ATR_MIN', 2.5),
            blocked_below_vwap_frac_min=_float('BLOCKED_BELOW_VWAP_FRAC_MIN', 0.85),
            blocked_prob_cap=_float('BLOCKED_PROB_CAP', 0.25),
            event_gap_abs_min=_float('EVENT_GAP_ABS_MIN', 0.08),
            event_rvol_min=_float('EVENT_RVOL_MIN', 2.2),
            event_range_pct_min=_float('EVENT_RANGE_PCT_MIN', 0.035),
            event_prob_cap=_float('EVENT_PROB_CAP', 0.35),
            downside_high_threshold=_float('DOWNSIDE_HIGH_THRESHOLD', 0.72),
            downside_medium_threshold=_float('DOWNSIDE_MEDIUM_THRESHOLD', 0.48),
            downside_prob_cap_high=_float('DOWNSIDE_PROB_CAP_HIGH', 0.42),
            downside_prob_cap_medium=_float('DOWNSIDE_PROB_CAP_MEDIUM', 0.62),
            uncertainty_z_thresh=_float('UNCERTAINTY_Z_THRESH', 3.0),
            uncertainty_extreme_features_min=max(1, _int('UNCERTAINTY_EXTREME_FEATURES_MIN', 3)),
            uncertainty_prob_cap=_float('UNCERTAINTY_PROB_CAP', 0.55),
            diag_track_min_prob=_float('DIAG_TRACK_MIN_PROB', 0.60),
            diag_track_top_n=max(5, _int('DIAG_TRACK_TOP_N', 15)),
            diag_cap_buffer=_float('DIAG_CAP_BUFFER', 0.01),
            diag_sane_mae_pct=_float('DIAG_SANE_MAE_PCT', -0.006),
            diag_held_minutes=max(3, _int('DIAG_HELD_MINUTES', 10)),
            diag_held_fraction=_float('DIAG_HELD_FRACTION', 0.7),
        )

    def normalized_feed(self) -> str:
        return 'sip'

    def debug_gate_password(self) -> str:
        return self.debug_password or self.admin_password
