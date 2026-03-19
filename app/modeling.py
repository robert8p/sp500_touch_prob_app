from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from .features import FEATURE_NAMES

TTC_IDX = FEATURE_NAMES.index('ttc_frac')
TOD_FRAC_IDX = FEATURE_NAMES.index('tod_frac')
RET5_IDX = FEATURE_NAMES.index('ret_5m')
RET30_IDX = FEATURE_NAMES.index('ret_30m')
REL_SPY5_IDX = FEATURE_NAMES.index('rel_strength_spy_5m')
REL_SPY_IDX = FEATURE_NAMES.index('rel_strength_spy_30m')
REL_SECTOR_IDX = FEATURE_NAMES.index('rel_strength_sector_30m')
RET20D_IDX = FEATURE_NAMES.index('ret_20d')
RET60D_IDX = FEATURE_NAMES.index('ret_60d')
DIST50DMA_IDX = FEATURE_NAMES.index('dist_50dma_pct')
DIST200DMA_IDX = FEATURE_NAMES.index('dist_200dma_pct')
DD60_IDX = FEATURE_NAMES.index('drawdown_60d_pct')
EMA_IDX = FEATURE_NAMES.index('ema_diff_pct')
ADX_IDX = FEATURE_NAMES.index('adx')
RVOL_IDX = FEATURE_NAMES.index('rvol_tod')
VWAP_LOC_IDX = FEATURE_NAMES.index('vwap_loc')
DIST_VWAP_IDX = FEATURE_NAMES.index('dist_pct_to_vwap')
VWAP_FRAC30_IDX = FEATURE_NAMES.index('closes_above_vwap_frac_30m')
VWAP_SLOPE_IDX = FEATURE_NAMES.index('vwap_slope_30m')
DONCH_IDX = FEATURE_NAMES.index('donch_dist')
RET_SINCE_OPEN_IDX = FEATURE_NAMES.index('ret_since_open_pct')
DIST_OPEN_ATR_IDX = FEATURE_NAMES.index('dist_open_atr')
BARS_BELOW_VWAP_IDX = FEATURE_NAMES.index('bars_below_vwap_frac')
NO_RECLAIM_IDX = FEATURE_NAMES.index('no_reclaim_vwap')
DAMAGE_HIGH_IDX = FEATURE_NAMES.index('damage_from_high_atr')
PATH_SMOOTH_IDX = FEATURE_NAMES.index('path_smoothness_30m')
REVERSAL_IDX = FEATURE_NAMES.index('reversal_count_30m')
OR_STATE_IDX = FEATURE_NAMES.index('or_breakout_state')
DLOG_IDX = FEATURE_NAMES.index('dvol_usd_med_log')
RANGE_IDX = FEATURE_NAMES.index('range_pct_med')
WICK_IDX = FEATURE_NAMES.index('wick_atr_med')
GAP_IDX = FEATURE_NAMES.index('gap_prev_close_pct')
# v10 new feature indices
CHH_IDX = FEATURE_NAMES.index('consecutive_higher_highs_5m')
DDR_IDX = FEATURE_NAMES.index('drawdown_recovery_30m')
SMOOTH15_IDX = FEATURE_NAMES.index('path_smoothness_15m')
RS_OPEN_IDX = FEATURE_NAMES.index('rel_strength_spy_since_open')
DVR_IDX = FEATURE_NAMES.index('directional_volume_ratio')
RVOL_DIR_IDX = FEATURE_NAMES.index('rvol_directional')
SPY_5M_RAW_IDX = FEATURE_NAMES.index('spy_ret_5m_raw')
SPY_CONSEC_DN_IDX = FEATURE_NAMES.index('spy_consecutive_down')
SPY_RV_IDX = FEATURE_NAMES.index('spy_rv_1h')
SECTOR_OPEN_IDX = FEATURE_NAMES.index('sector_ret_since_open')
EXTENSION_IDX = FEATURE_NAMES.index('extension_ratio')
REJECTION_IDX = FEATURE_NAMES.index('intraday_rejection_count')
VOL_TREND_IDX = FEATURE_NAMES.index('volume_trend_5bar')
RW_TOUCH_IDX = FEATURE_NAMES.index('rw_implied_touch_prob')

BUCKETS = [
    ('0_30', 0.0, 30.0),
    ('30_60', 30.0, 60.0),
    ('60_120', 60.0, 120.0),
    ('120_240', 120.0, 240.0),
    ('240_390', 240.0, 9999.0),
]


def bucket_name_from_ttc(ttc_frac: float) -> str:
    mins = float(ttc_frac) * 390.0
    for name, lo, hi in BUCKETS:
        if mins >= lo and mins < hi:
            return name
    return '240_390'


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Calibrator:
    method: str
    model: Any

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        rs = raw_scores.reshape(-1).astype(float)
        if self.method == 'isotonic':
            p = self.model.predict(rs)
            return np.clip(p, 0.0, 1.0)
        p = self.model.predict_proba(rs.reshape(-1, 1))[:, 1]
        return np.clip(p, 0.0, 1.0)


def _meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(meta or {})


def _dynamic_stage1_rvol_floor(mins_since_open: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    early = float(meta.get('stage1_min_rvol', 1.00))
    midday = float(meta.get('stage1_midday_min_rvol', max(0.8, early - 0.10)))
    late = float(meta.get('stage1_late_min_rvol', max(0.75, midday - 0.05)))
    strong_minutes = float(meta.get('stage1_strong_rvol_minutes', 90))
    out = np.full(mins_since_open.shape, late, dtype=float)
    out = np.where(mins_since_open < strong_minutes, early, out)
    out = np.where((mins_since_open >= strong_minutes) & (mins_since_open < 240.0), midday, out)
    return out


def weak_long_structure_mask_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    rel30_floor = float(m.get('stage1_deadcat_rel_spy_30m', -0.003))
    dist_vwap_floor = float(m.get('stage1_min_dist_pct_to_vwap', -0.0015))
    damage_floor = float(m.get('stage1_deadcat_damage_atr', 1.8))
    below_vwap = X[:, DIST_VWAP_IDX] < dist_vwap_floor
    below_open = X[:, RET_SINCE_OPEN_IDX] < 0.0
    weak_vs_spy = X[:, REL_SPY_IDX] < rel30_floor
    damage = X[:, DAMAGE_HIGH_IDX] >= damage_floor
    no_reclaim = (X[:, BARS_BELOW_VWAP_IDX] >= 0.75) & (X[:, NO_RECLAIM_IDX] > 0.5)
    dead_cat = below_vwap & below_open & weak_vs_spy & (damage | no_reclaim)
    vwap_state_bad = (X[:, DIST_VWAP_IDX] < -0.0035) & (X[:, VWAP_FRAC30_IDX] < 0.35) & (X[:, VWAP_SLOPE_IDX] <= 0.0)
    micro_failure = (X[:, OR_STATE_IDX] < -0.5) & below_open & (X[:, REL_SPY5_IDX] < -0.001)
    # Sell-off on heavy volume: below VWAP, negative momentum, selling-dominated volume, elevated RVOL
    selloff_volume = below_vwap & below_open & (X[:, DVR_IDX] < 0.5) & (X[:, RVOL_DIR_IDX] < -1.2)
    return dead_cat | vwap_state_bad | micro_failure | selloff_volume


def risk_bucket_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    dvol_min = float(m.get('liq_dvol_min_usd', 2_000_000.0))
    range_max = float(m.get('liq_range_pct_max', 0.012))
    wick_max = float(m.get('liq_wick_atr_max', 0.8))
    dvol_floor = np.log1p(max(0.0, dvol_min))
    cond_dvol = X[:, DLOG_IDX] < dvol_floor
    cond_range = X[:, RANGE_IDX] > range_max
    cond_wick = X[:, WICK_IDX] > wick_max
    risk = np.where(cond_dvol | cond_range | cond_wick, 'HIGH', 'OK')

    ret20_max = float(m.get('blocked_ret20d_max', -0.08))
    ret60_max = float(m.get('blocked_ret60d_max', -0.15))
    dist50_max = float(m.get('blocked_dist50dma_max', -0.06))
    open_max = float(m.get('blocked_ret_since_open_max', -0.025))
    damage_min = float(m.get('blocked_damage_from_high_atr_min', 2.5))
    below_vwap_min = float(m.get('blocked_below_vwap_frac_min', 0.85))

    lt_down = ((X[:, RET20D_IDX] <= ret20_max) & (X[:, DIST50DMA_IDX] <= dist50_max)) | ((X[:, RET60D_IDX] <= ret60_max) & (X[:, DIST200DMA_IDX] < -0.08)) | (X[:, DD60_IDX] <= -0.22)
    open_weak = X[:, RET_SINCE_OPEN_IDX] <= open_max
    no_reclaim = (X[:, BARS_BELOW_VWAP_IDX] >= below_vwap_min) & (X[:, NO_RECLAIM_IDX] > 0.5) & (X[:, VWAP_LOC_IDX] < -0.6)
    damage = X[:, DAMAGE_HIGH_IDX] >= damage_min
    gap_down = X[:, GAP_IDX] <= -0.05
    or_fail = (X[:, OR_STATE_IDX] < -0.5) & open_weak & no_reclaim
    weak_structure = weak_long_structure_mask_from_X(X, m)
    blocked_hits = lt_down.astype(int) + open_weak.astype(int) + no_reclaim.astype(int) + damage.astype(int) + or_fail.astype(int) + weak_structure.astype(int)
    blocked = (blocked_hits >= 3) | (lt_down & open_weak & (no_reclaim | damage)) | (gap_down & open_weak & no_reclaim) | weak_structure
    return np.where(blocked, 'BLOCKED', risk)


def event_risk_mask_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    gap_abs_min = float(m.get('event_gap_abs_min', 0.08))
    rvol_min = float(m.get('event_rvol_min', 2.2))
    range_min = float(m.get('event_range_pct_min', 0.035))
    gap_abs = np.abs(X[:, GAP_IDX])
    rvol = X[:, RVOL_IDX]
    rng = X[:, RANGE_IDX]
    return (gap_abs >= gap_abs_min) & ((rvol >= rvol_min) | (rng >= range_min))


def downside_risk_score_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ret20 = np.clip(-X[:, RET20D_IDX] / 0.12, 0.0, 2.0)
    ret60 = np.clip(-X[:, RET60D_IDX] / 0.20, 0.0, 2.0)
    dist50 = np.clip(-X[:, DIST50DMA_IDX] / 0.10, 0.0, 2.0)
    dd60 = np.clip(-X[:, DD60_IDX] / 0.25, 0.0, 2.0)
    open_weak = np.clip(-X[:, RET_SINCE_OPEN_IDX] / 0.03, 0.0, 2.0)
    below_vwap = np.clip(-X[:, DIST_VWAP_IDX] / 0.006, 0.0, 2.0)
    vwap_frac_bad = np.clip((0.6 - X[:, VWAP_FRAC30_IDX]) / 0.4, 0.0, 2.0)
    no_reclaim = np.clip(X[:, NO_RECLAIM_IDX], 0.0, 1.0)
    damage = np.clip(X[:, DAMAGE_HIGH_IDX] / 3.0, 0.0, 2.0)
    gap_down = np.clip(-X[:, GAP_IDX] / 0.08, 0.0, 2.0)
    rs_weak = np.clip(-X[:, REL_SPY_IDX] / 0.008, 0.0, 2.0)
    selloff_vol = np.clip(-X[:, RVOL_DIR_IDX] / 2.0, 0.0, 2.0)  # negative rvol_directional = sell-off
    overextended = np.clip((X[:, EXTENSION_IDX] - 1.5) / 2.0, 0.0, 2.0)  # far from normal range
    spy_turning = np.clip(X[:, SPY_CONSEC_DN_IDX] / 4.0, 0.0, 1.5)  # market rolling over
    raw = -1.9 + 0.50 * ret20 + 0.45 * ret60 + 0.30 * dist50 + 0.30 * dd60 + 0.50 * open_weak + 0.45 * below_vwap + 0.30 * vwap_frac_bad + 0.40 * no_reclaim + 0.55 * damage + 0.25 * gap_down + 0.35 * rs_weak + 0.30 * selloff_vol + 0.25 * overextended + 0.20 * spy_turning
    return sigmoid(raw)


def uncertainty_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    means = np.asarray(m.get('feature_mean', []), dtype=float)
    stds = np.asarray(m.get('feature_std', []), dtype=float)
    z_thresh = float(m.get('uncertainty_z_thresh', 3.0))
    extreme_min = int(m.get('uncertainty_extreme_features_min', 3))
    if means.size != X.shape[1] or stds.size != X.shape[1]:
        return np.array(['LOW'] * X.shape[0], dtype=object), np.array([''] * X.shape[0], dtype=object)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    z = np.abs((X - means) / stds)
    watch_idxs = [GAP_IDX, RET20D_IDX, DIST50DMA_IDX, RET_SINCE_OPEN_IDX, DIST_VWAP_IDX, DAMAGE_HIGH_IDX, RANGE_IDX, WICK_IDX, RVOL_IDX, REL_SPY_IDX]
    zw = z[:, watch_idxs]
    counts = np.sum(zw >= z_thresh, axis=1)
    level = np.where(counts >= extreme_min, 'HIGH', np.where(counts >= max(1, extreme_min - 1), 'MED', 'LOW')).astype(object)
    names = ['GAP', '20D', '50DMA', 'OPEN', 'VWAP', 'DAMAGE', 'RANGE', 'WICK', 'RVOL', 'RS']
    reasons = []
    for i in range(X.shape[0]):
        tags = [names[j] for j, val in enumerate(zw[i]) if val >= z_thresh]
        reasons.append(' '.join(tags[:5]))
    return level, np.array(reasons, dtype=object)


def acceptable_long_mask_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    risk = risk_bucket_from_X(X, m)
    event = event_risk_mask_from_X(X, m)
    downside = downside_risk_score_from_X(X, m)
    unc, _ = uncertainty_from_X(X, m)
    med_downside = float(m.get('downside_medium_threshold', 0.60))
    dist_vwap_floor = float(m.get('stage1_min_dist_pct_to_vwap', -0.0015)) - 0.001
    rel30_floor = float(m.get('stage1_min_rel_spy_30m', -0.0025)) - 0.0015
    rel5_floor = float(m.get('stage1_min_rel_spy_5m', -0.0018)) - 0.0010
    return (risk == 'OK') & (~event) & (downside < med_downside) & (unc != 'HIGH') & (X[:, DIST_VWAP_IDX] >= dist_vwap_floor) & (X[:, REL_SPY_IDX] >= rel30_floor) & (X[:, REL_SPY5_IDX] >= rel5_floor) & (~weak_long_structure_mask_from_X(X, m))


def stage1_diagnostics_from_X(X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=float)
    m = _meta(meta)
    mins_to_close = X[:, TTC_IDX] * 390.0
    mins_since_open = X[:, TOD_FRAC_IDX] * 390.0
    dvol_floor = np.log1p(max(0.0, float(m.get('liq_dvol_min_usd', 2_000_000.0)) * float(m.get('stage1_min_dollar_volume_mult', 0.75))))
    min_score = float(m.get('stage1_min_score', 1.75))
    min_mins = float(m.get('stage1_min_minutes_to_close', 35))
    min_since_open = float(m.get('stage1_min_minutes_since_open', 40))
    rvol_floor = _dynamic_stage1_rvol_floor(mins_since_open, m)
    rel30_floor = float(m.get('stage1_min_rel_spy_30m', -0.0025))
    rel5_floor = float(m.get('stage1_min_rel_spy_5m', -0.0018))
    dist_vwap_floor = float(m.get('stage1_min_dist_pct_to_vwap', -0.0015))
    strong_override_score = float(m.get('stage1_strong_override_score', 3.5))

    score = -0.40
    score += np.where(X[:, DLOG_IDX] >= dvol_floor, 1.15, -1.05)
    score += np.where(X[:, RVOL_IDX] >= rvol_floor, 1.00, -0.55)
    score += np.clip(X[:, RET30_IDX] / 0.008, -1.6, 1.8)
    score += 0.45 * np.clip(X[:, RET5_IDX] / 0.003, -1.4, 1.4)
    score += 0.85 * np.clip(X[:, REL_SPY_IDX] / 0.006, -1.3, 1.3)
    score += 0.45 * np.clip(X[:, REL_SPY5_IDX] / 0.0025, -1.2, 1.2)
    score += 0.55 * np.clip(X[:, REL_SECTOR_IDX] / 0.006, -1.2, 1.2)
    score += 0.75 * np.clip(X[:, EMA_IDX] / 0.004, -1.0, 1.5)
    score += 0.40 * np.clip((X[:, ADX_IDX] - 18.0) / 12.0, -1.0, 1.25)
    score += 0.65 * np.clip(X[:, DIST_VWAP_IDX] / 0.004, -1.4, 1.4)
    score += 0.40 * np.clip(X[:, VWAP_FRAC30_IDX] - 0.5, -0.8, 0.8)
    score += 0.35 * np.clip(X[:, VWAP_SLOPE_IDX] / 0.003, -1.0, 1.0)
    score -= 0.40 * np.clip(X[:, DONCH_IDX], 0.0, 3.0)
    score += 0.30 * np.clip(X[:, RET20D_IDX] / 0.08, -1.2, 1.0)
    score += 0.20 * np.clip(X[:, RET60D_IDX] / 0.15, -1.2, 1.0)
    score += 0.20 * np.clip(X[:, DIST50DMA_IDX] / 0.08, -1.2, 1.0)
    score += 0.35 * np.clip(X[:, PATH_SMOOTH_IDX], -1.0, 1.0)
    score -= 0.20 * np.clip(X[:, REVERSAL_IDX] / 4.0, 0.0, 1.5)
    score += 0.35 * X[:, OR_STATE_IDX]
    score -= 0.65 * np.clip(-X[:, RET_SINCE_OPEN_IDX] / 0.02, 0.0, 2.0)
    score -= 0.55 * np.clip(-X[:, DIST_VWAP_IDX] / 0.005, 0.0, 2.0)
    score -= 0.35 * X[:, BARS_BELOW_VWAP_IDX]
    score -= 0.55 * X[:, NO_RECLAIM_IDX]
    score -= 0.55 * np.clip(X[:, DAMAGE_HIGH_IDX] / 2.5, 0.0, 2.0)
    score -= 0.25 * np.clip(-X[:, GAP_IDX] / 0.05, 0.0, 1.5)
    # v10 bonus for new path features
    score += 0.25 * np.clip(X[:, CHH_IDX] / 4.0, 0.0, 1.0)
    score += 0.15 * np.clip(X[:, DDR_IDX] - 0.5, -0.5, 0.5)
    score += 0.20 * np.clip(X[:, SMOOTH15_IDX], -0.5, 0.5)
    score += 0.30 * np.clip(X[:, RS_OPEN_IDX] / 0.008, -1.0, 1.0)
    # Directional volume: reward buying volume, penalize sell-off volume
    score += 0.40 * np.clip(X[:, DVR_IDX] - 1.0, -1.5, 1.5)
    score += 0.35 * np.clip(X[:, RVOL_DIR_IDX] / 1.5, -1.5, 1.5)
    # Blindspot features: market context, exhaustion, supply
    score -= 0.35 * np.clip(X[:, SPY_CONSEC_DN_IDX] / 3.0, 0.0, 1.5)  # SPY rolling over
    score += 0.20 * np.clip(X[:, SECTOR_OPEN_IDX] / 0.01, -1.0, 1.0)  # Sector tailwind
    score -= 0.45 * np.clip((X[:, EXTENSION_IDX] - 1.5) / 1.5, 0.0, 1.5)  # Over-extended
    score -= 0.30 * np.clip(X[:, REJECTION_IDX] / 3.0, 0.0, 1.5)  # Failed breakouts
    score += 0.25 * np.clip(X[:, VOL_TREND_IDX], -1.0, 1.0)  # Rising volume = good

    risk = risk_bucket_from_X(X, m)
    blocked = risk == 'BLOCKED'
    event = event_risk_mask_from_X(X, m)
    weak_structure = weak_long_structure_mask_from_X(X, m)
    late_ok = mins_to_close >= min_mins
    warm_ok = mins_since_open >= min_since_open
    time_ok = late_ok & warm_ok
    relative_ok = (X[:, REL_SPY_IDX] >= rel30_floor) & (X[:, REL_SPY5_IDX] >= rel5_floor)
    vwap_ok = X[:, DIST_VWAP_IDX] >= dist_vwap_floor
    rvol_ok = X[:, RVOL_IDX] >= rvol_floor

    # v10: strong-override bypass — very high score lets marginal filter failures through
    strong_override = score >= strong_override_score

    # Standard pass: everything meets thresholds
    standard_pass = (~blocked) & (~event) & (~weak_structure) & time_ok & relative_ok & vwap_ok & rvol_ok & (score >= min_score)

    # Override pass: strong score overrides RVOL/relative-strength shortfalls (NOT blocked/event/structure/time)
    override_pass = (~blocked) & (~event) & (~weak_structure) & time_ok & strong_override

    passed = standard_pass | override_pass

    reasons = []
    for i in range(X.shape[0]):
        tags = []
        if X[i, DLOG_IDX] >= dvol_floor:
            tags.append('LIQ')
        if X[i, RVOL_IDX] >= rvol_floor[i]:
            tags.append('RVOL')
        if X[i, RET30_IDX] > 0.003:
            tags.append('MOM')
        if X[i, REL_SPY_IDX] > 0.002:
            tags.append('RS')
        if X[i, REL_SPY5_IDX] > 0.001:
            tags.append('RS5')
        if X[i, EMA_IDX] > 0 and X[i, ADX_IDX] > 18:
            tags.append('TREND')
        if X[i, DIST_VWAP_IDX] > 0:
            tags.append('VWAP+')
        if X[i, VWAP_SLOPE_IDX] > 0:
            tags.append('VWAP_UP')
        if X[i, PATH_SMOOTH_IDX] > 0.3:
            tags.append('SMOOTH')
        if override_pass[i] and not standard_pass[i]:
            tags.append('STRONG_OVERRIDE')
        if blocked[i]:
            tags.append('BLOCKED')
        elif event[i]:
            tags.append('EVENT')
        elif weak_structure[i]:
            tags.append('WEAK_STRUCT')
        elif not warm_ok[i]:
            tags.append('EARLY')
        elif not late_ok[i]:
            tags.append('LATE')
        elif not passed[i]:
            if not vwap_ok[i]:
                tags.append('VWAP-')
            elif not relative_ok[i]:
                tags.append('RS-')
            elif not rvol_ok[i]:
                tags.append('RVOL_LOW')
            elif score[i] < min_score:
                tags.append('WEAK')
        reasons.append(' '.join(tags[:9]))

    flags = {
        'blocked': blocked,
        'event': event,
        'weak_structure': weak_structure,
        'time_ok': time_ok,
        'warm_ok': warm_ok,
        'late_ok': late_ok,
        'relative_ok': relative_ok,
        'vwap_ok': vwap_ok,
        'rvol_ok': rvol_ok,
        'strong_override': override_pass & (~standard_pass),
    }
    return score.astype(float), passed.astype(bool), np.array(reasons, dtype=object), flags


# ── v10: Decomposed model bundle ──

@dataclass
class ModelBundle:
    """Model A: predicts P(raw_touch_1pct | features)."""
    pipeline: Any
    bucket_calibrators: Dict[str, Calibrator]
    priors: Dict[str, float]
    alpha: float
    feature_names: list
    meta: Dict[str, Any]
    strict_calibrator: Optional[Calibrator] = None
    strict_calibration_blend: float = 1.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.pipeline.decision_function(X), dtype=float)
        ttc = np.asarray(X[:, TTC_IDX], dtype=float) if X.shape[1] > TTC_IDX else np.zeros(X.shape[0])
        risk_keys = risk_bucket_from_X(X, self.meta)
        out = np.empty(X.shape[0], dtype=float)
        group_to_idx: Dict[Tuple[str, str], list] = {}
        for i in range(X.shape[0]):
            b = bucket_name_from_ttc(ttc[i])
            rk = str(risk_keys[i])
            group_to_idx.setdefault((rk, b), []).append(i)
        for (rk, b), idxs in group_to_idx.items():
            idxs_arr = np.array(idxs, dtype=int)
            cal = self.bucket_calibrators.get(f'{rk}|{b}') or self.bucket_calibrators.get(f'{rk}|global') or self.bucket_calibrators.get(b) or self.bucket_calibrators.get('global')
            p = sigmoid(raw[idxs_arr]) if cal is None else cal.predict(raw[idxs_arr])
            prior = float(self.priors.get(f'{rk}|{b}', self.priors.get(f'{rk}|global', self.priors.get(b, self.priors.get('global', 0.5)))))
            a = float(self.alpha)
            out[idxs_arr] = np.clip(a * p + (1.0 - a) * prior, 0.0, 1.0)
        return out


@dataclass
class PathQualityModel:
    """Model B: predicts P(acceptable_path | touch, features). LightGBM or logistic."""
    model: Any
    calibrator: Optional[Calibrator]
    method: str  # 'lightgbm' or 'logistic'
    feature_names: list
    meta: Dict[str, Any]

    def predict_path_quality(self, X: np.ndarray) -> np.ndarray:
        """Returns P(clean_path | touch) for each row."""
        if self.method == 'lightgbm':
            raw = self.model.predict_proba(X)[:, 1]
        else:
            raw = sigmoid(self.model.decision_function(X))
        if self.calibrator is not None:
            raw = self.calibrator.predict(np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6)))
        return np.clip(raw, 0.01, 0.99)


def bundle_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'pt1', 'bundle.joblib')


def path_model_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'pt1', 'path_quality.joblib')


def try_load_bundle(model_dir: str) -> Tuple[Optional[ModelBundle], str]:
    p = bundle_path(model_dir)
    if not os.path.exists(p):
        return None, 'missing'
    try:
        b = joblib.load(p)
        if getattr(b, 'feature_names', None) != list(FEATURE_NAMES):
            return None, 'incompatible'
        return b, 'ok'
    except Exception:
        return None, 'error'


def try_load_path_model(model_dir: str) -> Tuple[Optional[PathQualityModel], str]:
    p = path_model_path(model_dir)
    if not os.path.exists(p):
        return None, 'missing'
    try:
        m = joblib.load(p)
        if getattr(m, 'feature_names', None) != list(FEATURE_NAMES):
            return None, 'incompatible'
        return m, 'ok'
    except Exception:
        return None, 'error'


def heuristic_prob(features: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> float:
    X = np.asarray(features, dtype=float).reshape(1, -1)
    score, _passed, _reasons, flags = stage1_diagnostics_from_X(X, meta)
    downside = downside_risk_score_from_X(X, meta)[0]
    unc, _ = uncertainty_from_X(X, meta)
    p = float(sigmoid(np.array([
        -1.55
        + 0.60 * score[0]
        + 0.55 * (X[0, RET30_IDX] / 0.01)
        + 0.25 * (X[0, RVOL_IDX] - 1.0)
        + 0.30 * (X[0, DIST_VWAP_IDX] / 0.004)
        + 0.20 * (X[0, REL_SPY_IDX] / 0.006)
    ]))[0])
    # v10: multiplicative discounts, no artificial cap
    if flags['blocked'][0] or flags.get('weak_structure', np.array([False]))[0]:
        p *= 0.15
    if flags['event'][0]:
        p *= 0.40
    if unc[0] == 'HIGH':
        p *= 0.50
    if downside >= float((meta or {}).get('downside_high_threshold', 0.75)):
        p *= 0.45
    elif downside >= float((meta or {}).get('downside_medium_threshold', 0.60)):
        p *= 0.75
    return float(np.clip(p, 0.001, 1.0))


def predict_probs(model_dir: str, X: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, str, str, np.ndarray, np.ndarray]:
    """
    Returns (combined_prob, source, tail_status, touch_prob, path_prob).
    source: 'trained' | 'trained_no_path' | 'heuristic' -- explicit degradation
    tail_status: 'validated' | 'unvalidated' | 'heuristic'
    """
    bundle, status = try_load_bundle(model_dir)
    if bundle is None:
        probs = np.array([heuristic_prob(x, meta) for x in X], dtype=float)
        ones = np.ones_like(probs)
        return probs, 'heuristic', 'heuristic', probs, ones

    touch_probs = bundle.predict_proba(X)

    path_model, path_status = try_load_path_model(model_dir)
    if path_model is not None:
        path_probs = path_model.predict_path_quality(X)
        combined_raw = touch_probs * path_probs
        source = 'trained'
    else:
        path_probs = np.ones_like(touch_probs)
        combined_raw = touch_probs
        source = 'trained_no_path'

    strict_cal = getattr(bundle, 'strict_calibrator', None)
    strict_blend = float(getattr(bundle, 'strict_calibration_blend', bundle.meta.get('strict_calibration_blend', 1.0)) or 1.0)
    if strict_cal is not None:
        logits = np.log(np.clip(combined_raw, 1e-6, 1 - 1e-6) / np.clip(1 - combined_raw, 1e-6, 1 - 1e-6))
        calibrated = strict_cal.predict(logits)
        combined = np.clip((strict_blend * calibrated) + ((1.0 - strict_blend) * combined_raw), 0.0, 1.0)
    else:
        combined = combined_raw

    tail_status = 'validated' if bool(bundle.meta.get('decision_tail_validated', bundle.meta.get('touch_tail_validated', False))) else 'unvalidated'
    return combined, source, tail_status, touch_probs, path_probs
