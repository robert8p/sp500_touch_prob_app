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


def _csv_strs(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    vals: List[str] = []
    for part in (raw or '').split(','):
        part = part.strip()
        if part:
            vals.append(part)
    return vals or [x.strip() for x in default_csv.split(',') if x.strip()]


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
    # ── Stage 1 (relaxed in v10) ──
    stage1_candidate_cap: int
    stage1_min_score: float           # v11 default 2.0
    stage1_min_rvol: float            # 1.15 → 1.00
    stage1_midday_min_rvol: float     # 0.98 → 0.90
    stage1_late_min_rvol: float       # 0.90 → 0.85
    stage1_strong_rvol_minutes: int
    stage1_min_dollar_volume_mult: float
    stage1_min_minutes_since_open: int
    stage1_min_minutes_to_close: int
    stage1_min_rel_spy_30m: float     # -0.0015 → -0.0025
    stage1_min_rel_spy_5m: float      # -0.0010 → -0.0018
    stage1_min_dist_pct_to_vwap: float
    stage1_deadcat_damage_atr: float
    stage1_deadcat_rel_spy_30m: float
    stage1_strong_override_score: float  # NEW: bypass minor filter failures
    # ── Surfacing discipline (NEW in v11) ──
    surfacing_cooldown_minutes: int
    surfacing_min_touch_delta: float
    surfacing_min_path_delta: float
    unvalidated_messy_suppress: bool
    unvalidated_messy_risk_flags: List[str]
    watchlist_rescue_enabled: bool
    watchlist_rescue_max_rows: int
    watchlist_rescue_min_stage1_score: float
    watchlist_rescue_touch_frac: float
    watchlist_rescue_combined_frac: float
    watchlist_rescue_path_min: float
    watchlist_rescue_allow_medium_downside: bool
    # ── Guardrails ──
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
    # ── Label (loosened in v10) ──
    strict_touch_mae_threshold: float   # -0.004 → -0.006
    worthy_close_vs_scan_min: float
    worthy_missed_prob_max: float
    # ── Selection ──
    selection_ugly_touch_penalty: float
    selection_missed_worthy_penalty: float
    selection_challenge_fp_penalty: float
    selection_sparse_tail_penalty: float
    # ── Diagnostics ──
    diag_track_min_prob: float
    diag_track_top_n: int
    diag_sane_mae_pct: float
    diag_held_minutes: int
    diag_held_fraction: float
    diag_near_miss_top_n: int        # NEW
    diag_near_miss_min_score: float  # NEW
    # ── Decomposed models (NEW in v10) ──
    model_b_enabled: bool
    model_b_num_leaves: int
    model_b_learning_rate: float
    model_b_n_estimators: int
    model_b_min_samples: int
    train_recency_halflife_days: float
    train_strict_positive_boost: float
    train_ugly_positive_weight: float
    train_hard_negative_boost: float
    train_path_negative_boost: float
    strict_calibration_enabled: bool
    strict_calibration_min_samples: int
    strict_calibration_blend: float
    # ── Touch-tail readiness (adaptive in v10) ──
    touch_tail_threshold_mult: float   # tail = obs where P(touch) >= base_rate * this
    touch_tail_min_count: int
    touch_tail_min_lift: float
    path_quality_action_min: float
    # ── Market regime overlay (NEW in v12) ──
    regime_enabled: bool
    regime_oil_proxy: str
    regime_vol_proxy: str
    regime_safe_haven_proxy: str
    regime_energy_proxy: str
    regime_amber_oil_move_1h: float
    regime_red_oil_move_1h: float
    regime_amber_vol_move_1h: float
    regime_red_vol_move_1h: float
    regime_amber_safe_haven_move_1h: float
    regime_red_safe_haven_move_1h: float
    regime_amber_energy_rel_spy_1h: float
    regime_red_energy_rel_spy_1h: float
    regime_amber_spy_drop_1h: float
    regime_red_spy_drop_1h: float
    regime_amber_spy_drop_since_open: float
    regime_red_spy_drop_since_open: float
    regime_amber_multiplier: float
    regime_red_multiplier: float
    regime_amber_prob_cap: float
    regime_red_prob_cap: float
    regime_amber_touch_threshold_mult: float
    regime_red_touch_threshold_mult: float
    regime_amber_path_floor_add: float
    regime_red_path_floor_add: float
    regime_amber_suppress_signals: bool
    regime_red_suppress_signals: bool
    regime_energy_amber_multiplier: float
    regime_energy_red_multiplier: float
    # ── AI strategy interpreter (NEW in v12.3) ──
    ai_strategy_enabled: bool
    ai_strategy_model: str
    ai_strategy_timeout_seconds: int
    ai_strategy_reasoning_effort: str
    ai_strategy_max_candidates: int
    ai_strategy_cache_minutes: int
    ai_strategy_base_url: str
    # ── Specialist families + top-K reranker (NEW in v12.4) ──
    specialist_families_enabled: bool
    rerank_prob1_weight: float
    rerank_touch_weight: float
    rerank_path_weight: float
    rerank_stage1_weight: float
    rerank_family_bonus_weight: float
    rerank_downside_weight: float
    rerank_uncertainty_weight: float
    rerank_actionable_bonus: float
    rerank_candidate_bonus: float
    rerank_watchlist_penalty: float
    specialist_min_profile_count: int
    specialist_suppress_below_lift: float
    specialist_promote_above_lift: float
    specialist_threshold_loosen_mult: float
    specialist_threshold_tighten_mult: float
    specialist_calib_bin_count: int
    specialist_context_min_count: int
    specialist_context_min_lift_delta: float
    # ── Shadow (kept for backward compat) ──
    shadow_model_enabled: bool
    shadow_model_auto_promote: bool
    shadow_model_num_leaves: int
    shadow_model_learning_rate: float
    shadow_model_n_estimators: int

    @staticmethod
    def from_env() -> 'Settings':
        strict_mae = _float('STRICT_TOUCH_MAE_THRESHOLD', -0.006)
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
            # ── Stage 1 relaxed ──
            stage1_candidate_cap=max(20, _int('STAGE1_CANDIDATE_CAP', 120)),
            stage1_min_score=_float('STAGE1_MIN_SCORE', 2.0),
            stage1_min_rvol=_float('STAGE1_MIN_RVOL', 1.00),
            stage1_midday_min_rvol=_float('STAGE1_MIDDAY_MIN_RVOL', 0.90),
            stage1_late_min_rvol=_float('STAGE1_LATE_MIN_RVOL', 0.85),
            stage1_strong_rvol_minutes=max(30, _int('STAGE1_STRONG_RVOL_MINUTES', 90)),
            stage1_min_dollar_volume_mult=_float('STAGE1_MIN_DOLLAR_VOLUME_MULT', 0.75),
            stage1_min_minutes_since_open=max(5, _int('STAGE1_MIN_MINUTES_SINCE_OPEN', 40)),
            stage1_min_minutes_to_close=max(10, _int('STAGE1_MIN_MINUTES_TO_CLOSE', 35)),
            stage1_min_rel_spy_30m=_float('STAGE1_MIN_REL_SPY_30M', -0.0025),
            stage1_min_rel_spy_5m=_float('STAGE1_MIN_REL_SPY_5M', -0.0018),
            stage1_min_dist_pct_to_vwap=_float('STAGE1_MIN_DIST_PCT_TO_VWAP', -0.0015),
            stage1_deadcat_damage_atr=_float('STAGE1_DEADCAT_DAMAGE_ATR', 1.8),
            stage1_deadcat_rel_spy_30m=_float('STAGE1_DEADCAT_REL_SPY_30M', -0.0030),
            stage1_strong_override_score=_float('STAGE1_STRONG_OVERRIDE_SCORE', 3.5),
            # ── Surfacing discipline ──
            surfacing_cooldown_minutes=max(0, _int('SURFACING_COOLDOWN_MINUTES', 15)),
            surfacing_min_touch_delta=_float('SURFACING_MIN_TOUCH_DELTA', 0.03),
            surfacing_min_path_delta=_float('SURFACING_MIN_PATH_DELTA', 0.04),
            unvalidated_messy_suppress=_bool('UNVALIDATED_MESSY_SUPPRESS', True),
            unvalidated_messy_risk_flags=_csv_strs('UNVALIDATED_MESSY_RISK_FLAGS', 'WICKY,WIDE_RANGE,FALLING_KNIFE'),
            watchlist_rescue_enabled=_bool('WATCHLIST_RESCUE_ENABLED', True),
            watchlist_rescue_max_rows=max(0, _int('WATCHLIST_RESCUE_MAX_ROWS', 12)),
            watchlist_rescue_min_stage1_score=_float('WATCHLIST_RESCUE_MIN_STAGE1_SCORE', 2.25),
            watchlist_rescue_touch_frac=_float('WATCHLIST_RESCUE_TOUCH_FRAC', 0.78),
            watchlist_rescue_combined_frac=_float('WATCHLIST_RESCUE_COMBINED_FRAC', 0.90),
            watchlist_rescue_path_min=_float('WATCHLIST_RESCUE_PATH_MIN', 0.62),
            watchlist_rescue_allow_medium_downside=_bool('WATCHLIST_RESCUE_ALLOW_MEDIUM_DOWNSIDE', True),
            # ── Guardrails ──
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
            # ── Label loosened ──
            strict_touch_mae_threshold=strict_mae,
            worthy_close_vs_scan_min=_float('WORTHY_CLOSE_VS_SCAN_MIN', 0.0),
            worthy_missed_prob_max=_float('WORTHY_MISSED_PROB_MAX', 0.55),
            # ── Selection ──
            selection_ugly_touch_penalty=_float('SELECTION_UGLY_TOUCH_PENALTY', 1.8),
            selection_missed_worthy_penalty=_float('SELECTION_MISSED_WORTHY_PENALTY', 1.1),
            selection_challenge_fp_penalty=_float('SELECTION_CHALLENGE_FP_PENALTY', 1.4),
            selection_sparse_tail_penalty=_float('SELECTION_SPARSE_TAIL_PENALTY', 1.4),
            # ── Diagnostics ──
            diag_track_min_prob=_float('DIAG_TRACK_MIN_PROB', 0.03),
            diag_track_top_n=max(5, _int('DIAG_TRACK_TOP_N', 15)),
            diag_sane_mae_pct=_float('DIAG_SANE_MAE_PCT', strict_mae),
            diag_held_minutes=max(3, _int('DIAG_HELD_MINUTES', 10)),
            diag_held_fraction=_float('DIAG_HELD_FRACTION', 0.7),
            diag_near_miss_top_n=max(5, _int('DIAG_NEAR_MISS_TOP_N', 50)),
            diag_near_miss_min_score=_float('DIAG_NEAR_MISS_MIN_SCORE', 1.2),
            # ── Decomposed Model B (path quality) ──
            model_b_enabled=_bool('MODEL_B_ENABLED', True),
            model_b_num_leaves=max(8, _int('MODEL_B_NUM_LEAVES', 31)),
            model_b_learning_rate=_float('MODEL_B_LEARNING_RATE', 0.05),
            model_b_n_estimators=max(50, _int('MODEL_B_N_ESTIMATORS', 250)),
            model_b_min_samples=max(50, _int('MODEL_B_MIN_SAMPLES', 200)),
            train_recency_halflife_days=max(1.0, _float('TRAIN_RECENCY_HALFLIFE_DAYS', 12.0)),
            train_strict_positive_boost=_float('TRAIN_STRICT_POSITIVE_BOOST', 1.35),
            train_ugly_positive_weight=_float('TRAIN_UGLY_POSITIVE_WEIGHT', 0.82),
            train_hard_negative_boost=_float('TRAIN_HARD_NEGATIVE_BOOST', 1.30),
            train_path_negative_boost=_float('TRAIN_PATH_NEGATIVE_BOOST', 1.15),
            strict_calibration_enabled=_bool('STRICT_CALIBRATION_ENABLED', True),
            strict_calibration_min_samples=max(100, _int('STRICT_CALIBRATION_MIN_SAMPLES', 200)),
            strict_calibration_blend=_float('STRICT_CALIBRATION_BLEND', 0.80),
            # ── Touch-tail readiness ──
            touch_tail_threshold_mult=_float('TOUCH_TAIL_THRESHOLD_MULT', 1.5),
            touch_tail_min_count=max(3, _int('TOUCH_TAIL_MIN_COUNT', 8)),
            touch_tail_min_lift=_float('TOUCH_TAIL_MIN_LIFT', 2.0),
            path_quality_action_min=_float('PATH_QUALITY_ACTION_MIN', 0.65),
            # ── Market regime overlay ──
            regime_enabled=_bool('REGIME_ENABLED', True),
            regime_oil_proxy=os.getenv('REGIME_OIL_PROXY', 'USO').strip().upper() or 'USO',
            regime_vol_proxy=os.getenv('REGIME_VOL_PROXY', 'VIXY').strip().upper() or 'VIXY',
            regime_safe_haven_proxy=os.getenv('REGIME_SAFE_HAVEN_PROXY', 'GLD').strip().upper() or 'GLD',
            regime_energy_proxy=os.getenv('REGIME_ENERGY_PROXY', 'XLE').strip().upper() or 'XLE',
            regime_amber_oil_move_1h=_float('REGIME_AMBER_OIL_MOVE_1H', 0.010),
            regime_red_oil_move_1h=_float('REGIME_RED_OIL_MOVE_1H', 0.020),
            regime_amber_vol_move_1h=_float('REGIME_AMBER_VOL_MOVE_1H', 0.030),
            regime_red_vol_move_1h=_float('REGIME_RED_VOL_MOVE_1H', 0.060),
            regime_amber_safe_haven_move_1h=_float('REGIME_AMBER_SAFE_HAVEN_MOVE_1H', 0.004),
            regime_red_safe_haven_move_1h=_float('REGIME_RED_SAFE_HAVEN_MOVE_1H', 0.008),
            regime_amber_energy_rel_spy_1h=_float('REGIME_AMBER_ENERGY_REL_SPY_1H', 0.005),
            regime_red_energy_rel_spy_1h=_float('REGIME_RED_ENERGY_REL_SPY_1H', 0.010),
            regime_amber_spy_drop_1h=_float('REGIME_AMBER_SPY_DROP_1H', -0.003),
            regime_red_spy_drop_1h=_float('REGIME_RED_SPY_DROP_1H', -0.006),
            regime_amber_spy_drop_since_open=_float('REGIME_AMBER_SPY_DROP_SINCE_OPEN', -0.006),
            regime_red_spy_drop_since_open=_float('REGIME_RED_SPY_DROP_SINCE_OPEN', -0.012),
            regime_amber_multiplier=_float('REGIME_AMBER_MULTIPLIER', 0.80),
            regime_red_multiplier=_float('REGIME_RED_MULTIPLIER', 0.60),
            regime_amber_prob_cap=_float('REGIME_AMBER_PROB_CAP', 0.70),
            regime_red_prob_cap=_float('REGIME_RED_PROB_CAP', 0.62),
            regime_amber_touch_threshold_mult=_float('REGIME_AMBER_TOUCH_THRESHOLD_MULT', 1.10),
            regime_red_touch_threshold_mult=_float('REGIME_RED_TOUCH_THRESHOLD_MULT', 1.25),
            regime_amber_path_floor_add=_float('REGIME_AMBER_PATH_FLOOR_ADD', 0.03),
            regime_red_path_floor_add=_float('REGIME_RED_PATH_FLOOR_ADD', 0.08),
            regime_amber_suppress_signals=_bool('REGIME_AMBER_SUPPRESS_SIGNALS', False),
            regime_red_suppress_signals=_bool('REGIME_RED_SUPPRESS_SIGNALS', True),
            regime_energy_amber_multiplier=_float('REGIME_ENERGY_AMBER_MULTIPLIER', 0.90),
            regime_energy_red_multiplier=_float('REGIME_ENERGY_RED_MULTIPLIER', 0.75),
            # ── AI strategy interpreter ──
            ai_strategy_enabled=_bool('AI_STRATEGY_ENABLED', True),
            ai_strategy_model=os.getenv('AI_STRATEGY_MODEL', 'gpt-5.4').strip() or 'gpt-5.4',
            ai_strategy_timeout_seconds=max(5, _int('AI_STRATEGY_TIMEOUT_SECONDS', 45)),
            ai_strategy_reasoning_effort=(os.getenv('AI_STRATEGY_REASONING_EFFORT', 'medium').strip().lower() or 'medium'),
            ai_strategy_max_candidates=max(3, _int('AI_STRATEGY_MAX_CANDIDATES', 8)),
            ai_strategy_cache_minutes=max(0, _int('AI_STRATEGY_CACHE_MINUTES', 15)),
            ai_strategy_base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
            # ── Specialist families + top-K reranker ──
            specialist_families_enabled=_bool('SPECIALIST_FAMILIES_ENABLED', True),
            rerank_prob1_weight=_float('RERANK_PROB1_WEIGHT', 0.50),
            rerank_touch_weight=_float('RERANK_TOUCH_WEIGHT', 0.20),
            rerank_path_weight=_float('RERANK_PATH_WEIGHT', 0.12),
            rerank_stage1_weight=_float('RERANK_STAGE1_WEIGHT', 0.08),
            rerank_family_bonus_weight=_float('RERANK_FAMILY_BONUS_WEIGHT', 1.00),
            rerank_downside_weight=_float('RERANK_DOWNSIDE_WEIGHT', 0.18),
            rerank_uncertainty_weight=_float('RERANK_UNCERTAINTY_WEIGHT', 0.07),
            rerank_actionable_bonus=_float('RERANK_ACTIONABLE_BONUS', 0.08),
            rerank_candidate_bonus=_float('RERANK_CANDIDATE_BONUS', 0.04),
            rerank_watchlist_penalty=_float('RERANK_WATCHLIST_PENALTY', 0.03),
            specialist_min_profile_count=max(100, _int('SPECIALIST_MIN_PROFILE_COUNT', 300)),
            specialist_suppress_below_lift=_float('SPECIALIST_SUPPRESS_BELOW_LIFT', 0.90),
            specialist_promote_above_lift=_float('SPECIALIST_PROMOTE_ABOVE_LIFT', 1.15),
            specialist_threshold_loosen_mult=_float('SPECIALIST_THRESHOLD_LOOSEN_MULT', 0.92),
            specialist_threshold_tighten_mult=_float('SPECIALIST_THRESHOLD_TIGHTEN_MULT', 1.10),
            specialist_calib_bin_count=max(3, _int('SPECIALIST_CALIB_BIN_COUNT', 5)),
            specialist_context_min_count=max(80, _int('SPECIALIST_CONTEXT_MIN_COUNT', 180)),
            specialist_context_min_lift_delta=_float('SPECIALIST_CONTEXT_MIN_LIFT_DELTA', 0.15),
            # ── Shadow (backward compat) ──
            shadow_model_enabled=_bool('SHADOW_MODEL_ENABLED', False),
            shadow_model_auto_promote=_bool('SHADOW_MODEL_AUTO_PROMOTE', False),
            shadow_model_num_leaves=max(8, _int('SHADOW_MODEL_NUM_LEAVES', 31)),
            shadow_model_learning_rate=_float('SHADOW_MODEL_LEARNING_RATE', 0.05),
            shadow_model_n_estimators=max(50, _int('SHADOW_MODEL_N_ESTIMATORS', 250)),
        )

    def normalized_feed(self) -> str:
        return 'sip'

    def debug_gate_password(self) -> str:
        return self.debug_password or self.admin_password
