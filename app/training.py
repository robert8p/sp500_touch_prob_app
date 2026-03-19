from __future__ import annotations
import math
from datetime import date, datetime, timedelta, timezone
import os
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .alpaca import AlpacaClient
from .config import Settings
from .specialist import (
    classify_setup_families,
    compute_post_policy_metrics,
    compute_topk_metrics,
    market_context_bucket_from_X,
    sector_groups_from_array,
    summarize_setup_family_profiles,
    time_bucket_from_ttc,
)
from .features import FEATURE_NAMES, compute_features_from_5m
from .modeling import (
    Calibrator,
    ModelBundle,
    PathQualityModel,
    acceptable_long_mask_from_X,
    bucket_name_from_ttc,
    downside_risk_score_from_X,
    event_risk_mask_from_X,
    risk_bucket_from_X,
    sigmoid,
    stage1_diagnostics_from_X,
    uncertainty_from_X,
)
from .sectors import sector_etf_for_sector
from .volume_profiles import compute_profiles, save_profiles, slot_index_from_ts


TTC_IDX = FEATURE_NAMES.index('ttc_frac')


def _parse_ts(ts: str) -> datetime:
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts)


def _trading_days(end_local: date, lookback_days: int, tz_name: str) -> List[date]:
    start = end_local - timedelta(days=lookback_days * 3)
    out: List[date] = []
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('XNYS')
        sched = cal.schedule(start_date=start, end_date=end_local)
        for idx in sched.index:
            out.append(idx.date())
    except Exception:
        d = start
        while d <= end_local:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
    return out[-lookback_days:] if len(out) > lookback_days else out


def _session_utc_for_day(d: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    open_local = datetime(d.year, d.month, d.day, 9, 30, tzinfo=tz)
    close_local = datetime(d.year, d.month, d.day, 16, 0, tzinfo=tz)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('XNYS')
        sched = cal.schedule(start_date=d, end_date=d)
        if len(sched) == 1:
            open_local = sched.iloc[0]['market_open'].to_pydatetime().astimezone(tz)
            close_local = sched.iloc[0]['market_close'].to_pydatetime().astimezone(tz)
    except Exception:
        pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


def _daily_ctx_from_bars(daily_bars: List[dict], trade_day: date) -> Dict[str, float]:
    rows = []
    for b in daily_bars:
        try:
            ts = _parse_ts(b['t']).date()
        except Exception:
            continue
        if ts < trade_day:
            rows.append(b)
    if not rows:
        return {}
    closes = np.array([float(b.get('c') or 0.0) for b in rows], dtype=float)
    highs = np.array([float(b.get('h') or 0.0) for b in rows], dtype=float)
    valid = closes > 0
    if not np.any(valid):
        return {}
    closes = closes[valid]
    highs = highs[valid]
    last = float(closes[-1])
    def ret_n(n):
        if closes.size <= n: return 0.0
        ref = float(closes[-(n + 1)])
        return float(last / ref - 1.0) if ref > 0 else 0.0
    def sma_dist(n):
        if closes.size < n: return 0.0
        ma = float(np.mean(closes[-n:]))
        return float(last / ma - 1.0) if ma > 0 else 0.0
    def drawdown_n(n):
        peak = float(np.max(highs[-n:])) if highs.size >= n else (float(np.max(highs)) if highs.size else 0.0)
        return float(last / peak - 1.0) if peak > 0 else 0.0
    return {
        'ret_5d': ret_n(5), 'ret_20d': ret_n(20), 'ret_60d': ret_n(60),
        'dist_20dma_pct': sma_dist(20), 'dist_50dma_pct': sma_dist(50), 'dist_200dma_pct': sma_dist(200),
        'drawdown_20d_pct': drawdown_n(20), 'drawdown_60d_pct': drawdown_n(60),
    }


def _suffix_max(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    m = -np.inf
    for i in range(arr.size - 1, -1, -1):
        if arr[i] > m: m = arr[i]
        out[i] = m
    return out


def _split_day_masks(day_idx: np.ndarray):
    unique_days = sorted(int(x) for x in np.unique(day_idx))
    n = len(unique_days)
    if n < 8:
        raise RuntimeError('Need at least 8 trading days for time-aware split.')
    n_hold = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_cal = max(1, int(round(n * 0.15)))
    n_tr = max(1, n - n_hold - n_val - n_cal)
    while n_tr + n_cal + n_val + n_hold > n: n_tr -= 1
    tr_days = unique_days[:n_tr]
    cal_days = unique_days[n_tr:n_tr + n_cal]
    val_days = unique_days[n_tr + n_cal:n_tr + n_cal + n_val]
    hold_days = unique_days[n_tr + n_cal + n_val:]
    return (np.isin(day_idx, tr_days), np.isin(day_idx, cal_days),
            np.isin(day_idx, val_days), np.isin(day_idx, hold_days),
            {'train_days': tr_days, 'cal_days': cal_days, 'val_days': val_days, 'holdout_days': hold_days})


def _brier(y, p):
    return float(np.mean((p - y) ** 2)) if y.size else float('nan')


def _auc_safe(y, p):
    try:
        if len(np.unique(y)) < 2: return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _fit_platt(raw, y, sample_weight=None):
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float).reshape(-1)
    lr.fit(raw.reshape(-1, 1), y, **fit_kwargs)
    return Calibrator(method='platt', model=lr)


def _fit_isotonic(raw, y, sample_weight=None):
    if raw.size < 800 or len(np.unique(y)) < 2: return None
    iso = IsotonicRegression(out_of_bounds='clip')
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float).reshape(-1)
    iso.fit(raw.reshape(-1), y.reshape(-1), **fit_kwargs)
    return Calibrator(method='isotonic', model=iso)


def _fit_best_calibrator(raw, y, min_samples, sample_weight=None):
    raw = np.asarray(raw, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if raw.size < max(min_samples, 200):
        return _fit_platt(raw, y, weights)
    split = int(0.8 * raw.size)
    fit_idx = np.arange(split)
    eval_idx = np.arange(split, raw.size) if split < raw.size else np.arange(raw.size)
    fit_w = None if weights is None else weights[fit_idx]
    pl = _fit_platt(raw[fit_idx], y[fit_idx], fit_w)
    pl_b = _brier(y[eval_idx].astype(float), pl.predict(raw[eval_idx]).astype(float))
    iso = _fit_isotonic(raw[fit_idx], y[fit_idx], fit_w)
    if iso is None: return pl
    iso_b = _brier(y[eval_idx].astype(float), iso.predict(raw[eval_idx]).astype(float))
    return iso if iso_b <= pl_b else pl


def _weighted_mean(y, sample_weight=None):
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        return 0.5
    if sample_weight is None:
        return float(np.mean(y))
    w = np.asarray(sample_weight, dtype=float).reshape(-1)
    denom = float(np.sum(w))
    return float(np.sum(y * w) / denom) if denom > 0 else float(np.mean(y))


def _bucket_priors_segmented(y, ttc, risk, sample_weight=None):
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    mins = np.asarray(ttc, dtype=float).reshape(-1) * 390.0
    priors = {'global': _weighted_mean(y, weights)}
    for rk in ['OK', 'HIGH', 'BLOCKED']:
        mask_r = risk == rk
        if np.any(mask_r): priors[f'{rk}|global'] = _weighted_mean(y[mask_r], None if weights is None else weights[mask_r])
        for bname, lo, hi in [('0_30', 0.0, 30.0), ('30_60', 30.0, 60.0), ('60_120', 60.0, 120.0), ('120_240', 120.0, 240.0), ('240_390', 240.0, 9999.0)]:
            mask = mask_r & (mins >= lo) & (mins < hi)
            if np.any(mask): priors[f'{rk}|{bname}'] = _weighted_mean(y[mask], None if weights is None else weights[mask])
    return priors


def _fit_segmented_calibrators(raw, y, ttc, risk, min_samples, mode, sample_weight=None):
    calibs = {}; methods = {}
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    global_cal = _fit_best_calibrator(raw, y, min_samples, weights)
    calibs['global'] = global_cal; methods['global'] = global_cal.method
    if mode == 'global': return calibs, methods
    mins = ttc * 390.0
    for rk in ['OK', 'HIGH', 'BLOCKED']:
        mask_r = risk == rk
        if np.sum(mask_r) >= min_samples and len(np.unique(y[mask_r])) >= 2:
            cal = _fit_best_calibrator(raw[mask_r], y[mask_r], min_samples, None if weights is None else weights[mask_r])
            calibs[f'{rk}|global'] = cal; methods[f'{rk}|global'] = cal.method
        for bname, lo, hi in [('0_30', 0.0, 30.0), ('30_60', 30.0, 60.0), ('60_120', 60.0, 120.0), ('120_240', 120.0, 240.0), ('240_390', 240.0, 9999.0)]:
            mask = mask_r & (mins >= lo) & (mins < hi)
            if np.sum(mask) >= min_samples and len(np.unique(y[mask])) >= 2:
                cal = _fit_best_calibrator(raw[mask], y[mask], min_samples, None if weights is None else weights[mask])
                calibs[f'{rk}|{bname}'] = cal; methods[f'{rk}|{bname}'] = cal.method
    return calibs, methods


def _apply_calibration(raw, ttc, risk, calibs, priors, alpha):
    out = np.empty(raw.shape[0], dtype=float)
    for i in range(raw.shape[0]):
        b = bucket_name_from_ttc(ttc[i])
        rk = str(risk[i])
        cal = calibs.get(f'{rk}|{b}') or calibs.get(f'{rk}|global') or calibs.get('global')
        p = sigmoid(np.array([raw[i]]))[0] if cal is None else cal.predict(np.array([raw[i]]))[0]
        prior = float(priors.get(f'{rk}|{b}', priors.get(f'{rk}|global', priors.get('global', 0.5))))
        out[i] = np.clip(alpha * p + (1.0 - alpha) * prior, 0.0, 1.0)
    return out


def _precision_at_threshold(y, p, mask, thr):
    m = mask & (p >= thr)
    n = int(np.sum(m))
    if n == 0: return None, 0
    return float(np.mean(y[m])), n


def _median_or_none(arr):
    return float(np.median(arr)) if arr.size else None

def _mean_or_none(arr):
    return float(np.mean(arr)) if arr.size else None

def _logit_clip(p):
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def _recency_weights(day_idx, half_life_days):
    days = np.asarray(day_idx, dtype=int).reshape(-1)
    if days.size == 0:
        return np.array([], dtype=float)
    unique_days = sorted(int(x) for x in np.unique(days))
    pos_map = {d: i for i, d in enumerate(unique_days)}
    latest_pos = len(unique_days) - 1
    hl = max(1.0, float(half_life_days))
    weights = np.empty(days.shape[0], dtype=float)
    for i, d in enumerate(days):
        age = max(0, latest_pos - pos_map.get(int(d), latest_pos))
        weights[i] = max(0.35, 0.5 ** (age / hl))
    return weights


def _touch_sample_weights(X, y_touch, y_strict, day_idx, meta_cfg, settings):
    weights = _recency_weights(day_idx, settings.train_recency_halflife_days)
    y_touch = np.asarray(y_touch, dtype=int)
    y_strict = np.asarray(y_strict, dtype=int)
    acceptable = acceptable_long_mask_from_X(X, meta_cfg)
    event_mask = event_risk_mask_from_X(X, meta_cfg)
    stage1_score, _, _, _ = stage1_diagnostics_from_X(X, meta_cfg)
    hard_neg = (y_touch == 0) & acceptable & (~event_mask) & (stage1_score >= max(float(settings.stage1_min_score) + 0.75, 2.75))
    ugly_positive = (y_touch == 1) & (y_strict == 0)
    strict_positive = y_strict == 1
    weights = weights * np.where(strict_positive, float(settings.train_strict_positive_boost), 1.0)
    weights = weights * np.where(ugly_positive, float(settings.train_ugly_positive_weight), 1.0)
    weights = weights * np.where(hard_neg, float(settings.train_hard_negative_boost), 1.0)
    return np.clip(weights, 0.20, 5.0)


def _path_sample_weights(y_path, day_idx, settings):
    weights = _recency_weights(day_idx, settings.train_recency_halflife_days)
    y_path = np.asarray(y_path, dtype=int)
    weights = weights * np.where(y_path == 0, float(settings.train_path_negative_boost), 1.0)
    return np.clip(weights, 0.20, 5.0)


def _model_a_candidate_score(auc_touch, brier_touch, touch_tail_metrics, topk_metrics=None):
    metrics = dict(touch_tail_metrics or {})
    topk = dict(topk_metrics or {})
    by_k = dict(topk.get('by_k') or {})
    top3 = dict(by_k.get('top_3') or {})
    top5 = dict(by_k.get('top_5') or {})
    top10 = dict(by_k.get('top_10') or {})
    touch_lift = float(metrics.get('touch_lift') or 0.0)
    tail_n = int(metrics.get('tail_n') or 0)
    action_n = int(metrics.get('action_n') or 0)
    tail_strict = float(metrics.get('tail_strict_rate') or 0.0)
    action_strict = float(metrics.get('action_strict_rate') or 0.0)
    top3_prec = float(top3.get('precision') or 0.0)
    top5_prec = float(top5.get('precision') or 0.0)
    top10_prec = float(top10.get('precision') or 0.0)
    top5_lift = float(top5.get('lift') or 0.0)
    top10_lift = float(top10.get('lift') or 0.0)
    return (
        6.0 * touch_lift +
        5.0 * action_strict +
        3.0 * tail_strict +
        0.9 * math.log1p(tail_n) +
        0.6 * math.log1p(action_n) +
        5.5 * top3_prec +
        4.5 * top5_prec +
        3.5 * top10_prec +
        2.5 * top5_lift +
        1.8 * top10_lift -
        10.0 * float(brier_touch or 0.25) +
        0.4 * float(auc_touch or 0.5)
    )


def _fit_strict_outcome_calibrator(p_combined_val, y_val_strict, settings):
    result = {'enabled': bool(settings.strict_calibration_enabled), 'trained': False, 'method': 'none'}
    if not settings.strict_calibration_enabled:
        result['reason'] = 'disabled'
        return result, None
    p = np.asarray(p_combined_val, dtype=float).reshape(-1)
    y = np.asarray(y_val_strict, dtype=int).reshape(-1)
    if p.size < int(settings.strict_calibration_min_samples):
        result['reason'] = f'validation sample too small ({p.size})'
        return result, None
    if len(np.unique(y)) < 2:
        result['reason'] = 'strict labels have only one class'
        return result, None
    cal = _fit_best_calibrator(_logit_clip(p), y, int(settings.strict_calibration_min_samples))
    result.update({'trained': True, 'method': cal.method})
    return result, cal


def _apply_strict_outcome_calibration(p_raw, calibrator, blend):
    p = np.asarray(p_raw, dtype=float).reshape(-1)
    if calibrator is None:
        return np.clip(p, 0.0, 1.0)
    calibrated = calibrator.predict(_logit_clip(p))
    a = float(np.clip(blend, 0.0, 1.0))
    return np.clip((a * calibrated) + ((1.0 - a) * p), 0.0, 1.0)



def _threshold_path_metrics(diag, mask):
    n = int(np.sum(mask))
    out = {'n': n}
    if n == 0:
        out.update({k: None for k in ['raw_touch_rate', 'strict_touch_rate', 'clean_touch_rate', 'bouncy_touch_rate', 'ugly_touch_rate', 'worthy_rate', 'median_time_to_touch_min', 'mean_mae_before_touch_pct', 'median_mae_before_touch_pct', 'median_close_vs_scan_pct']})
        return out
    out['raw_touch_rate'] = float(np.mean(diag['raw_touch'][mask]))
    out['strict_touch_rate'] = float(np.mean(diag['strict_touch'][mask]))
    out['clean_touch_rate'] = float(np.mean(diag['clean_touch'][mask]))
    out['bouncy_touch_rate'] = float(np.mean(diag['bouncy_touch'][mask]))
    out['ugly_touch_rate'] = float(np.mean(diag['ugly_touch'][mask]))
    out['worthy_rate'] = float(np.mean(diag['worthy_touch'][mask]))
    time_valid = mask & np.isfinite(diag['time_to_touch_min']) & (diag['time_to_touch_min'] >= 0)
    mae_valid = mask & np.isfinite(diag['mae_before_touch_pct'])
    close_valid = mask & np.isfinite(diag['close_vs_scan_pct'])
    out['median_time_to_touch_min'] = _median_or_none(diag['time_to_touch_min'][time_valid])
    out['mean_mae_before_touch_pct'] = _mean_or_none(diag['mae_before_touch_pct'][mae_valid])
    out['median_mae_before_touch_pct'] = _median_or_none(diag['mae_before_touch_pct'][mae_valid])
    out['median_close_vs_scan_pct'] = _median_or_none(diag['close_vs_scan_pct'][close_valid])
    return out


def _final_metrics(X, y_touch, y_strict, p_combined, meta, diag, acceptable):
    """v10: metrics on the combined probability against the strict-touch label."""
    worthy_prob_max = float(meta.get('worthy_missed_prob_max', 0.55))
    challenge_mask = (~acceptable) | (risk_bucket_from_X(X, meta) == 'BLOCKED') | event_risk_mask_from_X(X, meta) | (downside_risk_score_from_X(X, meta) >= float(meta.get('downside_high_threshold', 0.75)))
    worthy_mask = diag['worthy_touch'] > 0.5
    out = {
        'brier_touch': _brier(y_touch.astype(float), p_combined.astype(float)),
        'brier_strict': _brier(y_strict.astype(float), p_combined.astype(float)),
        'auc_touch': _auc_safe(y_touch.astype(int), p_combined.astype(float)),
        'auc_strict': _auc_safe(y_strict.astype(int), p_combined.astype(float)),
        'raw_touch_count': int(np.sum(diag['raw_touch'] > 0.5)),
        'strict_touch_count': int(np.sum(diag['strict_touch'] > 0.5)),
        'clean_touch_count': int(np.sum(diag['clean_touch'] > 0.5)),
        'bouncy_touch_count': int(np.sum(diag['bouncy_touch'] > 0.5)),
        'ugly_touch_count': int(np.sum(diag['ugly_touch'] > 0.5)),
        'worthy_count': int(np.sum(worthy_mask)),
        'missed_worthy_count': int(np.sum(worthy_mask & (p_combined < worthy_prob_max))),
        'time_to_touch': {'median_min': _median_or_none(diag['time_to_touch_min'][(diag['raw_touch'] > 0.5) & np.isfinite(diag['time_to_touch_min'])])},
        'mae_before_touch': {
            'mean_pct': _mean_or_none(diag['mae_before_touch_pct'][(diag['raw_touch'] > 0.5) & np.isfinite(diag['mae_before_touch_pct'])]),
            'median_pct': _median_or_none(diag['mae_before_touch_pct'][(diag['raw_touch'] > 0.5) & np.isfinite(diag['mae_before_touch_pct'])]),
        },
        'close_vs_scan': {'median_pct': _median_or_none(diag['close_vs_scan_pct'][np.isfinite(diag['close_vs_scan_pct'])])},
    }
    # Evaluate combined prob against STRICT touch (the real decision target)
    for thr in [0.60, 0.70, 0.75, 0.80]:
        tag = str(int(round(thr * 100)))
        prec, n = _precision_at_threshold(y_strict, p_combined, np.ones_like(y_strict, dtype=bool), thr)
        prec_ok, n_ok = _precision_at_threshold(y_strict, p_combined, acceptable, thr)
        m = p_combined >= thr
        out[f'precision{tag}'] = prec
        out[f'precision{tag}_n'] = n
        out[f'precision{tag}_ok'] = prec_ok
        out[f'precision{tag}_ok_n'] = n_ok
        out[f'unacceptable_ge{tag}_n'] = int(np.sum((~acceptable) & m))
        out[f'ugly_touch_ge{tag}_n'] = int(np.sum(m & (diag['ugly_touch'] > 0.5)))
        out[f'clean_touch_ge{tag}_n'] = int(np.sum(m & (diag['clean_touch'] > 0.5)))
        out[f'worthy_ge{tag}_n'] = int(np.sum(m & worthy_mask))
        out[f'path_ge{tag}'] = _threshold_path_metrics(diag, m)
        worthies = int(np.sum(worthy_mask))
        out[f'worthy_capture_ge{tag}'] = (float(np.sum(worthy_mask & m)) / worthies) if worthies > 0 else None
    out['tail_path_75_ok'] = _threshold_path_metrics(diag, acceptable & (p_combined >= 0.75))
    out['challenge_set'] = {
        'count': int(np.sum(challenge_mask)),
        'avg_prob': float(np.mean(p_combined[challenge_mask])) if np.any(challenge_mask) else None,
        'ge75_count': int(np.sum(challenge_mask & (p_combined >= 0.75))),
        'ge75_false_positive_count': int(np.sum(challenge_mask & (p_combined >= 0.75) & (y_strict == 0))),
        'ugly_touch_ge75_count': int(np.sum(challenge_mask & (p_combined >= 0.75) & (diag['ugly_touch'] > 0.5))),
        'strict_touch_rate': float(np.mean(y_strict[challenge_mask])) if np.any(challenge_mask) else None,
        'path': _threshold_path_metrics(diag, challenge_mask & (p_combined >= 0.60)),
    }
    return out


def _make_base_pipeline(C, l1_ratio, class_weight):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=float(l1_ratio), C=float(C), class_weight=class_weight, max_iter=2500, random_state=42)),
    ])


def _selection_score(c, settings):
    auc_strict = float(c.get('auc_strict') or c.get('auc') or 0.5)
    brier_strict = float(c.get('brier_strict') or c.get('brier', 0.25))
    def pv(keyp, keyn, cap):
        p = c.get(keyp); n = int(c.get(keyn, 0) or 0)
        if p is None or n <= 0: return 0.0
        return float(p) * math.log1p(min(n, cap))
    tail = (
        10.0 * pv('precision75_ok', 'precision75_ok_n', max(10, settings.selection_min_count_75)) +
        7.5 * pv('precision80_ok', 'precision80_ok_n', 25) +
        6.5 * pv('precision70_ok', 'precision70_ok_n', max(10, settings.selection_min_count_70)) +
        2.5 * pv('precision60_ok', 'precision60_ok_n', 60)
    )
    density_bonus = min(int(c.get('precision75_ok_n', 0) or 0), settings.selection_min_count_75) * 0.18
    ugly_penalty = settings.selection_ugly_touch_penalty * (1.2 * int(c.get('ugly_touch_ge75_n', 0) or 0) + 0.6 * int(c.get('ugly_touch_ge70_n', 0) or 0))
    challenge_penalty = settings.selection_challenge_fp_penalty * (1.3 * int((((c.get('challenge_set') or {}).get('ge75_false_positive_count')) or 0)) + 0.7 * int((((c.get('challenge_set') or {}).get('ugly_touch_ge75_count')) or 0)))
    missed_worthy_penalty = settings.selection_missed_worthy_penalty * int(c.get('missed_worthy_count', 0) or 0)
    sparse_tail_penalty = settings.selection_sparse_tail_penalty if int(c.get('precision75_ok_n', 0) or 0) == 0 else 0.0
    sparse_tail_penalty += 0.75 if int(c.get('precision70_ok_n', 0) or 0) == 0 else 0.0
    unacceptable_penalty = 0.55 * int(c.get('unacceptable_ge75_n', 0) or 0)
    worthy_bonus = 0.9 * float(c.get('worthy_capture_ge75') or 0.0) + 0.45 * float(c.get('worthy_capture_ge70') or 0.0)
    return tail + density_bonus + worthy_bonus - ugly_penalty - challenge_penalty - missed_worthy_penalty - sparse_tail_penalty - unacceptable_penalty - 18.0 * brier_strict + 0.30 * auc_strict


def _touch_tail_readiness(p_touch_hold, p_path_hold, y_touch_hold, y_strict_hold, acceptable_hold, settings):
    """v10: Adaptive touch-tail validation.
    Threshold = base_rate * threshold_mult, so the tail captures the TOP of the
    model's distribution, not the entire scored population.
    """
    threshold_mult = float(settings.touch_tail_threshold_mult)
    min_count = int(settings.touch_tail_min_count)
    min_lift = float(settings.touch_tail_min_lift)
    path_min = float(settings.path_quality_action_min)

    base_touch_rate = float(np.mean(y_touch_hold)) if y_touch_hold.size else 0.0
    base_strict_rate = float(np.mean(y_strict_hold)) if y_strict_hold.size else 0.0

    # Adaptive threshold: tail = top of the distribution
    adaptive_threshold = base_touch_rate * threshold_mult
    adaptive_threshold = max(adaptive_threshold, 0.03)  # floor at 3% even if base is very low

    # Tail = observations where P(touch) >= adaptive threshold AND acceptable long
    tail_mask = (p_touch_hold >= adaptive_threshold) & acceptable_hold
    tail_n = int(np.sum(tail_mask))
    tail_touch_rate = float(np.mean(y_touch_hold[tail_mask])) if tail_n > 0 else 0.0
    tail_strict_rate = float(np.mean(y_strict_hold[tail_mask])) if tail_n > 0 else 0.0
    touch_lift = (tail_touch_rate / base_touch_rate) if base_touch_rate > 0 else 0.0

    # Actionable tail = tail AND path quality >= threshold
    if p_path_hold is not None:
        action_mask = tail_mask & (p_path_hold >= path_min)
    else:
        action_mask = tail_mask
    action_n = int(np.sum(action_mask))
    action_touch_rate = float(np.mean(y_touch_hold[action_mask])) if action_n > 0 else 0.0
    action_strict_rate = float(np.mean(y_strict_hold[action_mask])) if action_n > 0 else 0.0

    validated = bool(
        tail_n >= min_count
        and touch_lift >= min_lift
        and action_strict_rate >= 0.40
        and action_n >= max(3, min_count // 2)
    )

    metrics = {
        'base_touch_rate': base_touch_rate,
        'base_strict_rate': base_strict_rate,
        'adaptive_threshold': adaptive_threshold,
        'threshold_mult': threshold_mult,
        'tail_n': tail_n,
        'tail_touch_rate': tail_touch_rate,
        'tail_strict_rate': tail_strict_rate,
        'touch_lift': touch_lift,
        'action_n': action_n,
        'action_touch_rate': action_touch_rate,
        'action_strict_rate': action_strict_rate,
        'path_quality_min': path_min,
        'validated': validated,
    }

    warning = None
    if not validated:
        if tail_n < min_count:
            warning = f'Touch tail has only {tail_n} observations (need {min_count}) at adaptive threshold {adaptive_threshold:.1%}; tail unvalidated.'
        elif touch_lift < min_lift:
            warning = f'Touch tail lift is {touch_lift:.2f}x (need {min_lift:.1f}x) at threshold {adaptive_threshold:.1%}; tail unvalidated.'
        elif action_strict_rate < 0.40:
            warning = f'Actionable strict-touch rate is {action_strict_rate:.1%} (need >=40%) at threshold {adaptive_threshold:.1%}; tail unvalidated.'
        elif action_n < max(3, min_count // 2):
            warning = f'Actionable subset has only {action_n} observations (need >={max(3, min_count // 2)}); tail unvalidated.'
    return validated, metrics, warning




def _decision_tail_readiness(p_final_hold, y_strict_hold, acceptable_hold, settings):
    """Validate the actionable tail on the final investability score, not just touch."""
    threshold_mult = float(settings.touch_tail_threshold_mult)
    min_count = int(settings.touch_tail_min_count)
    min_lift = float(settings.touch_tail_min_lift)

    base_mask = np.asarray(acceptable_hold, dtype=bool)
    if base_mask.size == 0 or not np.any(base_mask):
        base_mask = np.ones_like(np.asarray(y_strict_hold, dtype=bool), dtype=bool)
    base_rate = float(np.mean(y_strict_hold[base_mask])) if np.any(base_mask) else 0.0
    adaptive_threshold = max(base_rate * threshold_mult, 0.03)

    tail_mask = (p_final_hold >= adaptive_threshold) & base_mask
    tail_n = int(np.sum(tail_mask))
    tail_rate = float(np.mean(y_strict_hold[tail_mask])) if tail_n > 0 else 0.0
    strict_lift = (tail_rate / base_rate) if base_rate > 0 else 0.0

    validated = bool(
        tail_n >= min_count
        and strict_lift >= min_lift
        and tail_rate >= 0.40
    )

    metrics = {
        'base_strict_rate': base_rate,
        'adaptive_threshold': adaptive_threshold,
        'threshold_mult': threshold_mult,
        'tail_n': tail_n,
        'tail_strict_rate': tail_rate,
        'strict_lift': strict_lift,
        'validated': validated,
    }

    warning = None
    if not validated:
        if tail_n < min_count:
            warning = f'Final-score tail has only {tail_n} observations (need {min_count}) at adaptive threshold {adaptive_threshold:.1%}; tail unvalidated.'
        elif strict_lift < min_lift:
            warning = f'Final-score tail lift is {strict_lift:.2f}x (need {min_lift:.1f}x) at threshold {adaptive_threshold:.1%}; tail unvalidated.'
        elif tail_rate < 0.40:
            warning = f'Final-score tail strict-hit rate is {tail_rate:.1%} (need >=40%) at threshold {adaptive_threshold:.1%}; tail unvalidated.'
    return validated, metrics, warning


def _learn_guardrail_profiles(X_hold, y_hold_strict, acceptable_hold, meta_cfg, settings):
    """Learn empirical shrinkage factors for live risk overlays from holdout data."""
    y = np.asarray(y_hold_strict, dtype=float)
    acceptable = np.asarray(acceptable_hold, dtype=bool)
    if y.size == 0:
        return {
            'base_strict_rate': 0.0,
            'min_count': 0,
            'event': {'multiplier': 0.40},
            'uncertainty': {'HIGH': {'multiplier': 0.50}, 'MED': {'multiplier': 0.85}},
            'downside': {'HIGH': {'multiplier': 0.45}, 'MED': {'multiplier': 0.75}},
        }

    base_mask = acceptable if np.any(acceptable) else np.ones_like(acceptable, dtype=bool)
    base_rate = float(np.mean(y[base_mask])) if np.any(base_mask) else float(np.mean(y))
    min_count = max(50, int(0.005 * y.size))

    event_mask = event_risk_mask_from_X(X_hold, meta_cfg)
    downside = downside_risk_score_from_X(X_hold, meta_cfg)
    unc_level, _ = uncertainty_from_X(X_hold, meta_cfg)
    high_down = downside >= float(settings.downside_high_threshold)
    med_down = (downside >= float(settings.downside_medium_threshold)) & (~high_down)
    med_unc = np.isin(unc_level, ['MED', 'MEDIUM'])
    high_unc = unc_level == 'HIGH'

    def _profile(mask, fallback):
        joint = base_mask & np.asarray(mask, dtype=bool)
        n = int(np.sum(joint))
        if n < min_count or base_rate <= 0:
            return {
                'count': n,
                'rate': float(np.mean(y[joint])) if n > 0 else None,
                'multiplier': float(fallback),
                'fallback': True,
            }
        rate = float(np.mean(y[joint]))
        mult = float(np.clip(rate / max(base_rate, 1e-9), 0.25, 1.10))
        return {'count': n, 'rate': rate, 'multiplier': mult, 'fallback': False}

    return {
        'base_strict_rate': base_rate,
        'min_count': min_count,
        'event': _profile(event_mask, 0.40),
        'uncertainty': {
            'HIGH': _profile(high_unc, 0.50),
            'MED': _profile(med_unc, 0.85),
        },
        'downside': {
            'HIGH': _profile(high_down, 0.45),
            'MED': _profile(med_down, 0.75),
        },
    }

def _settings_meta(settings):
    return {
        'liq_dvol_min_usd': float(settings.liq_dvol_min_usd), 'liq_range_pct_max': float(settings.liq_range_pct_max), 'liq_wick_atr_max': float(settings.liq_wick_atr_max),
        'blocked_ret20d_max': float(settings.blocked_ret20d_max), 'blocked_ret60d_max': float(settings.blocked_ret60d_max),
        'blocked_dist50dma_max': float(settings.blocked_dist50dma_max), 'blocked_ret_since_open_max': float(settings.blocked_ret_since_open_max),
        'blocked_damage_from_high_atr_min': float(settings.blocked_damage_from_high_atr_min), 'blocked_below_vwap_frac_min': float(settings.blocked_below_vwap_frac_min),
        'blocked_prob_cap': float(settings.blocked_prob_cap),
        'event_gap_abs_min': float(settings.event_gap_abs_min), 'event_rvol_min': float(settings.event_rvol_min), 'event_range_pct_min': float(settings.event_range_pct_min), 'event_prob_cap': float(settings.event_prob_cap),
        'uncertainty_z_thresh': float(settings.uncertainty_z_thresh), 'uncertainty_extreme_features_min': int(settings.uncertainty_extreme_features_min), 'uncertainty_prob_cap': float(settings.uncertainty_prob_cap),
        'downside_prob_cap_high': float(settings.downside_prob_cap_high), 'downside_prob_cap_medium': float(settings.downside_prob_cap_medium),
        'downside_high_threshold': float(settings.downside_high_threshold), 'downside_medium_threshold': float(settings.downside_medium_threshold),
        'stage1_candidate_cap': int(settings.stage1_candidate_cap), 'stage1_min_score': float(settings.stage1_min_score),
        'stage1_min_minutes_since_open': int(settings.stage1_min_minutes_since_open), 'stage1_min_minutes_to_close': int(settings.stage1_min_minutes_to_close),
        'stage1_min_rvol': float(settings.stage1_min_rvol), 'stage1_midday_min_rvol': float(settings.stage1_midday_min_rvol),
        'stage1_late_min_rvol': float(settings.stage1_late_min_rvol), 'stage1_strong_rvol_minutes': int(settings.stage1_strong_rvol_minutes),
        'stage1_min_dollar_volume_mult': float(settings.stage1_min_dollar_volume_mult),
        'stage1_min_rel_spy_30m': float(settings.stage1_min_rel_spy_30m), 'stage1_min_rel_spy_5m': float(settings.stage1_min_rel_spy_5m),
        'stage1_min_dist_pct_to_vwap': float(settings.stage1_min_dist_pct_to_vwap),
        'stage1_deadcat_damage_atr': float(settings.stage1_deadcat_damage_atr), 'stage1_deadcat_rel_spy_30m': float(settings.stage1_deadcat_rel_spy_30m),
        'stage1_strong_override_score': float(settings.stage1_strong_override_score),
        'strict_touch_mae_threshold': float(settings.strict_touch_mae_threshold),
        'worthy_close_vs_scan_min': float(settings.worthy_close_vs_scan_min), 'worthy_missed_prob_max': float(settings.worthy_missed_prob_max),
        'touch_tail_threshold_mult': float(settings.touch_tail_threshold_mult), 'touch_tail_min_count': int(settings.touch_tail_min_count),
        'touch_tail_min_lift': float(settings.touch_tail_min_lift), 'path_quality_action_min': float(settings.path_quality_action_min),
    }


# ── Dataset builder (same structure, new spy_ret_since_open plumbing) ──

def build_training_dataset(client, symbols, sector_map, lookback_days, tz_name, scan_interval_minutes, tod_min_days, liq_rolling_bars, liq_thresholds, blocked_params, strict_touch_mae_threshold, worthy_close_vs_scan_min, diag_held_minutes, diag_held_fraction):
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    days = _trading_days(today_local, lookback_days, tz_name)
    X_rows, y_touch_rows, y_strict_rows, day_idx_rows, sector_rows = [], [], [], [], []
    raw_touch_rows, strict_touch_rows, clean_touch_rows, bouncy_touch_rows = [], [], [], []
    ugly_touch_rows, worthy_touch_rows, time_to_touch_rows, mae_before_touch_rows = [], [], [], []
    held_above_rows, close_vs_scan_rows = [], []
    slot_hist = {s: {i: [] for i in range(78)} for s in symbols}
    day_count_hist = {s: 0 for s in symbols}
    step = max(1, int(scan_interval_minutes / 5)) if scan_interval_minutes >= 5 else 1
    sector_etfs = sorted({sector_etf_for_sector(sector_map.get(sym, '')) for sym in symbols})
    symbols_daily = list(dict.fromkeys(symbols + ['SPY'] + sector_etfs))
    first_day = days[0] if days else today_local
    first_open_utc, _ = _session_utc_for_day(first_day, tz_name)
    daily_start = first_open_utc - timedelta(days=500)
    daily_bars, errd, _ = client.get_bars(symbols_daily, '1Day', start_utc=daily_start, end_utc=datetime.now(timezone.utc))
    if errd: return None, f'daily bars fetch failed: {errd}'

    for di, d in enumerate(days):
        prev_d = days[di - 1] if di > 0 else d
        prev_open_utc, prev_close_utc = _session_utc_for_day(prev_d, tz_name)
        open_utc, close_utc = _session_utc_for_day(d, tz_name)
        symbols_5m = list(dict.fromkeys(symbols + ['SPY'] + sector_etfs))
        bars5, err5, _ = client.get_bars(symbols_5m, '5Min', start_utc=prev_open_utc, end_utc=close_utc)
        if err5: return None, f'bars5 fetch failed {d}: {err5}'
        bars1, err1, _ = client.get_bars(symbols_5m, '1Min', start_utc=open_utc, end_utc=close_utc)
        if err1: return None, f'bars1 fetch failed {d}: {err1}'

        def full_session_and_current(lst):
            full, cur, cur_idx = [], [], []
            for idx, b in enumerate(lst):
                t = _parse_ts(b['t'])
                if prev_open_utc <= t <= prev_close_utc or open_utc <= t <= close_utc:
                    full.append(b)
                    if open_utc <= t <= close_utc:
                        cur.append(b); cur_idx.append(len(full) - 1)
            return full, cur, np.array(cur_idx, dtype=int)

        _spy_full, spy_cur, _ = full_session_and_current(bars5.get('SPY', []))
        spy_ret_5m_arr = spy_ret_30m_arr = spy_ret_since_open_arr = None
        spy_rv_1h_arr = spy_consec_down_arr = None
        if len(spy_cur) >= 2:
            spy_c = np.array([float(b.get('c') or 0.0) for b in spy_cur], dtype=float)
            spy_open = spy_c[0] if spy_c.size else 1.0
            spy_ret_5m_arr = np.zeros_like(spy_c)
            for i in range(1, spy_c.size):
                spy_ret_5m_arr[i] = (spy_c[i] / spy_c[i - 1] - 1.0) if spy_c[i - 1] else 0.0
            spy_ret_30m_arr = np.zeros_like(spy_c)
            for i in range(spy_c.size):
                spy_ret_30m_arr[i] = (spy_c[i] / spy_c[i - 6] - 1.0) if i >= 6 and spy_c[i - 6] else 0.0
            spy_ret_since_open_arr = np.zeros_like(spy_c)
            for i in range(spy_c.size):
                spy_ret_since_open_arr[i] = (spy_c[i] / spy_open - 1.0) if spy_open > 0 else 0.0
            # SPY realized vol (rolling 12-bar window)
            spy_logrets = np.diff(np.log(np.maximum(spy_c, 1e-9)))
            spy_rv_1h_arr = np.zeros_like(spy_c)
            for i in range(2, spy_c.size):
                seg = spy_logrets[max(0, i - 12):i]
                spy_rv_1h_arr[i] = float(np.std(seg) * np.sqrt(max(1.0, float(seg.size)))) if seg.size > 2 else 0.0
            # SPY consecutive down bars at each position
            spy_consec_down_arr = np.zeros_like(spy_c)
            for i in range(1, spy_c.size):
                count = 0
                for k in range(i, 0, -1):
                    if spy_c[k] < spy_c[k - 1]:
                        count += 1
                    else:
                        break
                spy_consec_down_arr[i] = float(min(count, 7))

        sector_ret_arrs, sector_ret5_arrs = {}, {}
        sector_open_arrs = {}
        for etf in sector_etfs:
            _f5, cur, _ = full_session_and_current(bars5.get(etf, []))
            if len(cur) >= 2:
                cc = np.array([float(b.get('c') or 0.0) for b in cur], dtype=float)
                sec_open = cc[0] if cc.size else 1.0
                arr5 = np.zeros_like(cc)
                for i in range(1, cc.size): arr5[i] = (cc[i] / cc[i - 1] - 1.0) if cc[i - 1] else 0.0
                arr30 = np.zeros_like(cc)
                for i in range(cc.size): arr30[i] = (cc[i] / cc[i - 6] - 1.0) if i >= 6 and cc[i - 6] else 0.0
                arr_open = np.zeros_like(cc)
                for i in range(cc.size): arr_open[i] = (cc[i] / sec_open - 1.0) if sec_open > 0 else 0.0
                sector_ret_arrs[etf] = arr30; sector_ret5_arrs[etf] = arr5; sector_open_arrs[etf] = arr_open
            else:
                sector_ret_arrs[etf] = np.zeros(0); sector_ret5_arrs[etf] = np.zeros(0); sector_open_arrs[etf] = np.zeros(0)

        today_slot_vols = {s: {i: [] for i in range(78)} for s in symbols}

        for sym in symbols:
            full5, cur5, cur_indices = full_session_and_current(bars5.get(sym, []))
            cur1 = [b for b in bars1.get(sym, []) if open_utc <= _parse_ts(b['t']) <= close_utc]
            if not cur5 or not cur1 or len(full5) < 7: continue
            prev_session = [b for b in full5 if prev_open_utc <= _parse_ts(b['t']) <= prev_close_utc]
            prev_day_close = float(prev_session[-1].get('c') or 0.0) if prev_session else None
            prev_day_high = float(max(float(b.get('h') or 0.0) for b in prev_session)) if prev_session else None
            prev_day_low = float(min(float(b.get('l') or 0.0) for b in prev_session)) if prev_session else None
            t1 = np.array([_parse_ts(b['t']).timestamp() for b in cur1], dtype=float)
            h1 = np.array([float(b.get('h') or 0.0) for b in cur1], dtype=float)
            l1 = np.array([float(b.get('l') or 0.0) for b in cur1], dtype=float)
            c1 = np.array([float(b.get('c') or 0.0) for b in cur1], dtype=float)
            if h1.size < 2: continue
            suf = _suffix_max(h1)
            etf = sector_etf_for_sector(sector_map.get(sym, ''))
            etf_arr30 = sector_ret_arrs.get(etf)
            daily_ctx = _daily_ctx_from_bars(daily_bars.get(sym, []), d)

            for cur_pos, full_idx in enumerate(cur_indices):
                if (cur_pos % step) != 0: continue
                hist5 = full5[:full_idx + 1]
                if len(hist5) < 7: continue
                current_bar = cur5[cur_pos]
                current_ts = _parse_ts(current_bar['t'])
                slot = slot_index_from_ts(current_ts, tz_name)
                if slot is not None:
                    try: today_slot_vols[sym][slot].append(float(current_bar.get('v') or 0.0))
                    except Exception: pass
                baseline = None
                if slot is not None and day_count_hist.get(sym, 0) >= tod_min_days:
                    vals = slot_hist[sym][slot]
                    if vals: baseline = float(np.median(np.array(vals, dtype=float)))
                price = float(current_bar.get('c') or 0.0)
                scan_end_ts = current_ts.timestamp() + 5 * 60.0
                j = int(np.searchsorted(t1, scan_end_ts, side='left'))
                if j >= t1.size or price <= 0: continue
                h_future = float(suf[j])
                touch_raw = 1 if h_future >= 1.01 * price else 0
                mins_to_close = max(0.0, (close_utc - datetime.fromtimestamp(scan_end_ts, tz=timezone.utc)).total_seconds() / 60.0)
                spy_ret5 = float(spy_ret_5m_arr[cur_pos]) if spy_ret_5m_arr is not None and cur_pos < spy_ret_5m_arr.size else 0.0
                spy_ret30 = float(spy_ret_30m_arr[cur_pos]) if spy_ret_30m_arr is not None and cur_pos < spy_ret_30m_arr.size else 0.0
                spy_ret_open = float(spy_ret_since_open_arr[cur_pos]) if spy_ret_since_open_arr is not None and cur_pos < spy_ret_since_open_arr.size else 0.0
                spy_rv = float(spy_rv_1h_arr[cur_pos]) if spy_rv_1h_arr is not None and cur_pos < spy_rv_1h_arr.size else 0.0
                spy_cd = int(spy_consec_down_arr[cur_pos]) if spy_consec_down_arr is not None and cur_pos < spy_consec_down_arr.size else 0
                etf_open_arr = sector_open_arrs.get(etf)
                sec_open_ret = float(etf_open_arr[cur_pos]) if etf_open_arr is not None and cur_pos < etf_open_arr.size else 0.0
                sec_ret30 = float(etf_arr30[cur_pos]) if etf_arr30 is not None and cur_pos < etf_arr30.size else 0.0
                fr = compute_features_from_5m(
                    bars_5m=hist5, spy_ret_5m=spy_ret5, spy_ret_30m=spy_ret30, sector_ret_30m=sec_ret30,
                    mins_to_close=mins_to_close, tod_baseline_vol_median=baseline, rolling_rvol_window=20,
                    risk_params=(liq_rolling_bars, liq_thresholds[0], liq_thresholds[1], liq_thresholds[2]),
                    prev_day_close=prev_day_close, prev_day_high=prev_day_high, prev_day_low=prev_day_low,
                    daily_ctx=daily_ctx, tz_name=tz_name, blocked_params=blocked_params,
                    spy_ret_since_open=spy_ret_open,
                    spy_ret_5m_raw=spy_ret5,
                    spy_consecutive_down=spy_cd,
                    spy_rv_1h=spy_rv,
                    sector_ret_since_open=sec_open_ret,
                )
                if fr is None: continue

                future_h, future_l, future_c = h1[j:], l1[j:], c1[j:]
                touch_rel_idx = np.where(future_h >= 1.01 * price)[0]
                if touch_rel_idx.size:
                    touch_idx = int(touch_rel_idx[0])
                    mae_before_touch = float(np.min(future_l[:touch_idx + 1]) / price - 1.0)
                    time_to_touch_min = float((touch_idx + 1))
                    hold_end = min(future_c.size, touch_idx + max(1, diag_held_minutes))
                    future_after_touch = future_c[touch_idx:hold_end]
                    held_above = 1.0 if (future_after_touch.size >= 3 and np.mean(future_after_touch >= price) >= diag_held_fraction) else 0.0
                else:
                    mae_before_touch = float(np.min(future_l) / price - 1.0) if future_l.size else 0.0
                    time_to_touch_min = float('nan'); held_above = 0.0
                close_vs_scan = float(future_c[-1] / price - 1.0) if future_c.size else 0.0
                strict_touch = 1 if (touch_raw == 1 and mae_before_touch >= strict_touch_mae_threshold) else 0
                clean_touch = 1 if (strict_touch == 1 and held_above >= 0.5) else 0
                bouncy_touch = 1 if (touch_raw == 1 and strict_touch == 1 and clean_touch == 0) else 0
                ugly_touch = 1 if (touch_raw == 1 and strict_touch == 0) else 0
                worthy_touch = 1 if (strict_touch == 1 and (held_above >= 0.5 or close_vs_scan >= worthy_close_vs_scan_min)) else 0

                X_rows.append(fr.features)
                y_touch_rows.append(touch_raw)       # Model A target
                y_strict_rows.append(strict_touch)    # Evaluation target
                day_idx_rows.append(di)
                sector_rows.append(sector_map.get(sym, ''))
                raw_touch_rows.append(float(touch_raw)); strict_touch_rows.append(float(strict_touch))
                clean_touch_rows.append(float(clean_touch)); bouncy_touch_rows.append(float(bouncy_touch))
                ugly_touch_rows.append(float(ugly_touch)); worthy_touch_rows.append(float(worthy_touch))
                time_to_touch_rows.append(time_to_touch_min); mae_before_touch_rows.append(mae_before_touch)
                held_above_rows.append(held_above); close_vs_scan_rows.append(close_vs_scan)

        for sym in symbols:
            any_day_obs = False
            for slot, vals in today_slot_vols[sym].items():
                if vals: any_day_obs = True; slot_hist[sym][slot].append(float(np.sum(vals)))
            if any_day_obs: day_count_hist[sym] += 1

    if not X_rows: return None, 'no training rows produced'
    diag = {
        'raw_touch': np.array(raw_touch_rows, dtype=float),
        'strict_touch': np.array(strict_touch_rows, dtype=float),
        'clean_touch': np.array(clean_touch_rows, dtype=float),
        'bouncy_touch': np.array(bouncy_touch_rows, dtype=float),
        'ugly_touch': np.array(ugly_touch_rows, dtype=float),
        'worthy_touch': np.array(worthy_touch_rows, dtype=float),
        'time_to_touch_min': np.array(time_to_touch_rows, dtype=float),
        'mae_before_touch_pct': np.array(mae_before_touch_rows, dtype=float),
        'held_above_scan_10m': np.array(held_above_rows, dtype=float),
        'close_vs_scan_pct': np.array(close_vs_scan_rows, dtype=float),
    }
    return {
        'X': np.vstack(X_rows),
        'y_touch': np.array(y_touch_rows, dtype=int),
        'y_strict': np.array(y_strict_rows, dtype=int),
        'day_idx': np.array(day_idx_rows, dtype=int),
        'sectors': np.array(sector_rows, dtype=object),
        'diag': diag,
    }, None


# ── v10: Train Model B (path quality | touch) ──

def _fit_path_quality_model(X_touch, y_path, X_val_touch, y_val_path, settings, sample_weight=None):
    """Train LightGBM to predict P(acceptable_path | touch)."""
    result = {'enabled': bool(settings.model_b_enabled), 'trained': False, 'method': 'none'}
    if not settings.model_b_enabled or X_touch.shape[0] < settings.model_b_min_samples:
        result['reason'] = 'disabled or insufficient touch samples'
        return result, None
    if len(np.unique(y_path)) < 2:
        result['reason'] = 'only one class in path labels'
        return result, None
    if X_val_touch.shape[0] < 20:
        result['reason'] = f'validation touch set too small ({X_val_touch.shape[0]} rows)'
        return result, None
    if len(np.unique(y_val_path)) < 2:
        result['reason'] = 'validation path labels have only one class'
        return result, None
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        # Fallback to logistic regression
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))])
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['clf__sample_weight'] = np.asarray(sample_weight, dtype=float)
        pipe.fit(X_touch, y_path, **fit_kwargs)
        raw_val = pipe.decision_function(X_val_touch)
        cal = _fit_best_calibrator(raw_val, y_val_path, 100)
        pq = PathQualityModel(model=pipe, calibrator=cal, method='logistic', feature_names=list(FEATURE_NAMES), meta={})
        result.update({'trained': True, 'method': 'logistic', 'auc': _auc_safe(y_val_path, sigmoid(raw_val))})
        return result, pq
    model = LGBMClassifier(
        objective='binary', num_leaves=settings.model_b_num_leaves,
        learning_rate=settings.model_b_learning_rate, n_estimators=settings.model_b_n_estimators,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = np.asarray(sample_weight, dtype=float)
    model.fit(X_touch, y_path, **fit_kwargs)
    raw_val = model.predict_proba(X_val_touch)[:, 1]
    log_odds = np.log(np.clip(raw_val, 1e-6, 1 - 1e-6) / np.clip(1 - raw_val, 1e-6, 1 - 1e-6))
    cal = _fit_best_calibrator(log_odds, y_val_path, 100)
    pq = PathQualityModel(model=model, calibrator=cal, method='lightgbm', feature_names=list(FEATURE_NAMES), meta={})
    auc_b = _auc_safe(y_val_path, raw_val)
    result.update({'trained': True, 'method': 'lightgbm', 'auc': auc_b, 'n_touch_train': int(X_touch.shape[0]), 'n_touch_val': int(X_val_touch.shape[0])})
    return result, pq


def _select_and_train_bundle(dataset, settings, model_dir):
    X_full = np.asarray(dataset['X'], dtype=float)
    y_touch_full = np.asarray(dataset['y_touch'], dtype=int)
    y_strict_full = np.asarray(dataset['y_strict'], dtype=int)
    day_idx_full = np.asarray(dataset['day_idx'], dtype=int)
    sectors_full = np.asarray(dataset.get('sectors', np.array(['OTHER'] * X_full.shape[0], dtype=object)), dtype=object)
    diag_full = {k: np.asarray(v) for k, v in dict(dataset['diag']).items()}

    meta_cfg = _settings_meta(settings)
    stage1_score, stage1_pass, _, _ = stage1_diagnostics_from_X(X_full, meta_cfg)
    if np.sum(stage1_pass) < 1500:
        raise RuntimeError(f'Too few stage-1 training rows ({int(np.sum(stage1_pass))}). Lower STAGE1_MIN_SCORE or widen training window.')

    X = X_full[stage1_pass]; y_touch = y_touch_full[stage1_pass]; y_strict = y_strict_full[stage1_pass]
    day_idx = day_idx_full[stage1_pass]; diag = {k: v[stage1_pass] for k, v in diag_full.items()}
    tr_mask, cal_mask, val_mask, hold_mask, split_days = _split_day_masks(day_idx)
    X_tr, y_tr_touch = X[tr_mask], y_touch[tr_mask]
    X_cal, y_cal_touch = X[cal_mask], y_touch[cal_mask]
    X_val, y_val_touch, y_val_strict = X[val_mask], y_touch[val_mask], y_strict[val_mask]
    X_hold, y_hold_touch, y_hold_strict = X[hold_mask], y_touch[hold_mask], y_strict[hold_mask]
    y_tr_strict = y_strict[tr_mask]
    y_cal_strict = y_strict[cal_mask]
    day_tr = day_idx[tr_mask]
    day_cal = day_idx[cal_mask]
    day_val = day_idx[val_mask]
    diag_val = {k: v[val_mask] for k, v in diag.items()}
    diag_hold = {k: v[hold_mask] for k, v in diag.items()}
    acceptable_hold = acceptable_long_mask_from_X(X_hold, meta_cfg)
    acceptable_val = acceptable_long_mask_from_X(X_val, meta_cfg)

    if len(np.unique(y_tr_touch)) < 2:
        raise RuntimeError('Training labels have only one class.')

    risk_cal = risk_bucket_from_X(X_cal, meta_cfg); risk_val = risk_bucket_from_X(X_val, meta_cfg); risk_hold = risk_bucket_from_X(X_hold, meta_cfg)
    ttc_cal = X_cal[:, TTC_IDX]; ttc_val = X_val[:, TTC_IDX]; ttc_hold = X_hold[:, TTC_IDX]
    X_traincal = np.vstack([X_tr, X_cal]); y_traincal_touch = np.concatenate([y_tr_touch, y_cal_touch])
    weights_tr = _touch_sample_weights(X_tr, y_tr_touch, y_tr_strict, day_tr, meta_cfg, settings)
    weights_cal = _touch_sample_weights(X_cal, y_cal_touch, y_cal_strict, day_cal, meta_cfg, settings)
    weights_traincal = np.concatenate([weights_tr, weights_cal])
    risk_traincal = risk_bucket_from_X(X_traincal, meta_cfg); ttc_traincal = X_traincal[:, TTC_IDX]

    # ── Model A: touch probability (elastic-net grid search) ──
    candidates = []
    for cw in [None, 'balanced']:
        for C in settings.enet_c_values:
            for l1 in settings.enet_l1_values:
                pipe = _make_base_pipeline(C=C, l1_ratio=l1, class_weight=cw)
                pipe.fit(X_tr, y_tr_touch, clf__sample_weight=weights_tr)
                raw_cal = pipe.decision_function(X_cal)
                raw_val = pipe.decision_function(X_val)
                priors = _bucket_priors_segmented(y_traincal_touch, ttc_traincal, risk_traincal, weights_traincal)
                for cal_mode in ['bucketed', 'global']:
                    calibs, methods = _fit_segmented_calibrators(raw_cal, y_cal_touch, ttc_cal, risk_cal, settings.calib_min_bucket_samples, cal_mode, weights_cal)
                    for alpha in settings.prior_alpha_values:
                        p_touch_val = _apply_calibration(raw_val, ttc_val, risk_val, calibs, priors, float(alpha))
                        # For Model A selection, use touch AUC as primary signal
                        auc_a = _auc_safe(y_val_touch, p_touch_val)
                        brier_a = _brier(y_val_touch.astype(float), p_touch_val)
                        touch_tail_ok_val, touch_tail_metrics_val, _ = _touch_tail_readiness(p_touch_val, None, y_val_touch, y_val_strict, acceptable_val, settings)
                        topk_metrics_val = compute_topk_metrics(p_touch_val, y_val_strict, acceptable_val)
                        candidate_score = _model_a_candidate_score(auc_a, brier_a, touch_tail_metrics_val, topk_metrics_val)
                        candidates.append({
                            'cw': 'balanced' if cw == 'balanced' else 'none',
                            'C': float(C), 'l1': float(l1), 'alpha': float(alpha),
                            'cal_mode': cal_mode, 'methods': methods,
                            'auc_touch_val': auc_a, 'brier_touch_val': brier_a,
                            'candidate_score': candidate_score, 'touch_tail_validated_val': touch_tail_ok_val,
                            'touch_tail_metrics_val': touch_tail_metrics_val,
                            'topk_metrics_val': topk_metrics_val,
                            'p_touch_val': p_touch_val,
                        })

    # Select best Model A by tail-aware validation score, not generic AUC alone
    best_a = max(candidates, key=lambda c: float(c.get('candidate_score', c.get('auc_touch_val') or 0.5)))

    # Refit Model A on train+cal
    X_fit = np.vstack([X_tr, X_cal]); y_fit_touch = np.concatenate([y_tr_touch, y_cal_touch])
    y_fit_strict = np.concatenate([y_tr_strict, y_cal_strict])
    day_fit = np.concatenate([day_tr, day_cal])
    weights_fit = _touch_sample_weights(X_fit, y_fit_touch, y_fit_strict, day_fit, meta_cfg, settings)
    risk_fit = risk_bucket_from_X(X_fit, meta_cfg); ttc_fit = X_fit[:, TTC_IDX]
    priors_final = _bucket_priors_segmented(y_fit_touch, ttc_fit, risk_fit, weights_fit)
    pipe_final = _make_base_pipeline(C=best_a['C'], l1_ratio=best_a['l1'], class_weight=('balanced' if best_a['cw'] == 'balanced' else None))
    pipe_final.fit(X_fit, y_fit_touch, clf__sample_weight=weights_fit)
    raw_val_final = pipe_final.decision_function(X_val)
    val_touch_weights = _touch_sample_weights(X_val, y_val_touch, y_val_strict, day_val, meta_cfg, settings)
    calibs_final, methods_final = _fit_segmented_calibrators(raw_val_final, y_val_touch, ttc_val, risk_val, settings.calib_min_bucket_samples, best_a['cal_mode'], val_touch_weights)
    raw_hold = pipe_final.decision_function(X_hold)
    p_touch_hold = _apply_calibration(raw_hold, ttc_hold, risk_hold, calibs_final, priors_final, float(best_a['alpha']))
    p_touch_val_final = _apply_calibration(raw_val_final, ttc_val, risk_val, calibs_final, priors_final, float(best_a['alpha']))

    feature_mean = np.mean(X_fit, axis=0).astype(float).tolist()
    feature_std = np.where(np.std(X_fit, axis=0) <= 1e-6, 1.0, np.std(X_fit, axis=0)).astype(float).tolist()

    # ── Model B: path quality conditional on touch ──
    # Train on rows where raw_touch == 1; target is strict_touch (i.e., acceptable path)
    touch_mask_tr = y_tr_touch == 1
    touch_mask_cal = y_cal_touch == 1
    touch_mask_val = y_val_touch == 1
    y_tr_path = y_strict[tr_mask][touch_mask_tr]
    X_tr_touch = X_tr[touch_mask_tr]
    y_cal_path = y_strict[cal_mask][touch_mask_cal]
    X_cal_touch_b = X_cal[touch_mask_cal]
    y_val_path = y_val_strict[touch_mask_val]
    X_val_touch_b = X_val[touch_mask_val]
    X_fit_b = np.vstack([X_tr_touch, X_cal_touch_b]) if X_tr_touch.shape[0] > 0 and X_cal_touch_b.shape[0] > 0 else X_tr_touch
    y_fit_b = np.concatenate([y_tr_path, y_cal_path]) if y_tr_path.size > 0 and y_cal_path.size > 0 else y_tr_path
    day_fit_b = np.concatenate([day_tr[touch_mask_tr], day_cal[touch_mask_cal]]) if y_tr_path.size > 0 and y_cal_path.size > 0 else day_tr[touch_mask_tr]
    weights_fit_b = _path_sample_weights(y_fit_b, day_fit_b, settings) if y_fit_b.size > 0 else None

    model_b_result, path_model = _fit_path_quality_model(X_fit_b, y_fit_b, X_val_touch_b, y_val_path, settings, sample_weight=weights_fit_b)

    # ── Compute combined probabilities on holdout ──
    if path_model is not None:
        p_path_hold = path_model.predict_path_quality(X_hold)
        p_combined_hold_raw = p_touch_hold * p_path_hold
        p_path_val = path_model.predict_path_quality(X_val)
        p_combined_val_raw = p_touch_val_final * p_path_val
    else:
        p_path_hold = np.ones_like(p_touch_hold)
        p_combined_hold_raw = p_touch_hold
        p_path_val = np.ones_like(p_touch_val_final)
        p_combined_val_raw = p_touch_val_final

    strict_calibration_result, strict_calibrator = _fit_strict_outcome_calibrator(p_combined_val_raw, y_val_strict, settings)
    strict_blend = float(np.clip(settings.strict_calibration_blend, 0.0, 1.0))
    p_combined_hold = _apply_strict_outcome_calibration(p_combined_hold_raw, strict_calibrator, strict_blend)
    p_combined_val = _apply_strict_outcome_calibration(p_combined_val_raw, strict_calibrator, strict_blend)

    # Evaluate combined probability
    hold_metrics = _final_metrics(X_hold, y_hold_touch, y_hold_strict, p_combined_hold, meta_cfg, diag_hold, acceptable_hold)
    val_metrics = _final_metrics(X_val, y_val_touch, y_val_strict, p_combined_val, meta_cfg, diag_val, acceptable_val)
    topk_metrics_holdout = compute_topk_metrics(p_combined_hold, y_hold_strict, acceptable_hold)
    topk_metrics_val_final = compute_topk_metrics(p_combined_val, y_val_strict, acceptable_val)
    val_families = classify_setup_families(X_val)
    holdout_families = classify_setup_families(X_hold)
    val_market_context = market_context_bucket_from_X(X_val)
    hold_market_context = market_context_bucket_from_X(X_hold)
    val_time_bucket = time_bucket_from_ttc(ttc_val)
    hold_time_bucket = time_bucket_from_ttc(ttc_hold)
    val_sector_groups = sector_groups_from_array(sectors_val)
    hold_sector_groups = sector_groups_from_array(sectors_hold)
    setup_family_profiles = summarize_setup_family_profiles(
        val_families,
        y_val_strict,
        p_combined_val,
        acceptable_val,
        sectors=sectors_val,
        ttc_frac=ttc_val,
        market_context=val_market_context,
        min_count=int(settings.specialist_min_profile_count),
        suppress_below_lift=float(settings.specialist_suppress_below_lift),
        promote_above_lift=float(settings.specialist_promote_above_lift),
        threshold_loosen_mult=float(settings.specialist_threshold_loosen_mult),
        threshold_tighten_mult=float(settings.specialist_threshold_tighten_mult),
        bin_count=int(settings.specialist_calib_bin_count),
        context_min_count=int(settings.specialist_context_min_count),
        context_min_lift_delta=float(settings.specialist_context_min_lift_delta),
    )

    # Tail readiness metrics
    p_path_hold_for_eval = p_path_hold if path_model is not None else None
    touch_tail_validated, touch_tail_metrics, touch_tail_warning = _touch_tail_readiness(
        p_touch_hold, p_path_hold_for_eval, y_hold_touch, y_hold_strict, acceptable_hold, settings)
    decision_tail_validated, decision_tail_metrics, decision_tail_warning = _decision_tail_readiness(
        p_combined_hold, y_hold_strict, acceptable_hold, settings)
    policy_adaptive_threshold = float(decision_tail_metrics.get('adaptive_threshold') or max(0.03, np.mean(p_combined_val) * float(settings.touch_tail_threshold_mult)))
    family_policy_metrics_holdout = compute_post_policy_metrics(
        p_combined_hold, y_hold_strict, acceptable_hold, holdout_families, policy_adaptive_threshold, setup_family_profiles,
        regime_states=hold_market_context, time_buckets=hold_time_bucket, sector_groups=hold_sector_groups,
        watchlist_frac=float(settings.watchlist_rescue_combined_frac),
    )
    family_policy_metrics_validation = compute_post_policy_metrics(
        p_combined_val, y_val_strict, acceptable_val, val_families, policy_adaptive_threshold, setup_family_profiles,
        regime_states=val_market_context, time_buckets=val_time_bucket, sector_groups=val_sector_groups,
        watchlist_frac=float(settings.watchlist_rescue_combined_frac),
    )
    learned_guardrail_profiles = _learn_guardrail_profiles(X_hold, y_hold_strict, acceptable_hold, meta_cfg, settings)

    # Selection score on combined holdout metrics
    sel_score = _selection_score(hold_metrics, settings)

    # ── Save Model A ──
    bundle_meta = {
        **meta_cfg,
        'trained_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'n_rows_full': int(X_full.shape[0]), 'n_rows_stage1': int(X.shape[0]),
        'stage1_pass_rate': float(np.mean(stage1_pass)),
        'split_days': split_days,
        'target_name': 'decomposed_touch_x_path_1pct',
        'target_definition': f'P(touch_1pct) * P(MAE >= {settings.strict_touch_mae_threshold:.4f} | touch)',
        'model_a': {'family': 'elastic_net', 'cw': best_a['cw'], 'C': best_a['C'], 'l1': best_a['l1'], 'alpha': best_a['alpha'], 'cal_mode': best_a['cal_mode'], 'auc_touch_val': best_a['auc_touch_val'], 'brier_touch_val': best_a['brier_touch_val'], 'candidate_score': best_a.get('candidate_score'), 'touch_tail_metrics_val': best_a.get('touch_tail_metrics_val'), 'topk_metrics_val': best_a.get('topk_metrics_val')},
        'model_b': model_b_result,
        'strict_calibration': {**strict_calibration_result, 'blend': strict_blend},
        'bucket_calibration_methods': methods_final,
        'prior_means': priors_final,
        'calib_min_bucket_samples': int(settings.calib_min_bucket_samples),
        'calibrator': best_a['cal_mode'], 'class_weight': best_a['cw'], 'alpha': float(best_a['alpha']),
        'strict_calibration_blend': strict_blend,
        'feature_mean': feature_mean, 'feature_std': feature_std,
        'selection_tier': 'decision_tail_validated' if decision_tail_validated else 'tail_not_ready',
        'selection_warning': decision_tail_warning or touch_tail_warning,
        'auc_val': hold_metrics.get('auc_strict'), 'brier_val': hold_metrics.get('brier_strict'),
        'touch_tail_validated': touch_tail_validated,
        'touch_tail_metrics': touch_tail_metrics,
        'decision_tail_validated': decision_tail_validated,
        'decision_tail_metrics': decision_tail_metrics,
        'guardrail_profiles': learned_guardrail_profiles,
        'probability_contract': 'strict_calibrated_decomposed' if strict_calibrator is not None else 'uncapped_decomposed',
        'selection_score': sel_score,
        'topk_metrics_holdout': topk_metrics_holdout,
        'topk_metrics_val_final': topk_metrics_val_final,
        'setup_family_profiles': setup_family_profiles,
        'family_policy_metrics_holdout': family_policy_metrics_holdout,
        'family_policy_metrics_validation': family_policy_metrics_validation,
        'policy_profile_source': 'validation',
    }

    # Add all holdout precision metrics
    for tag in ['60', '70', '75', '80']:
        for k in [f'precision{tag}', f'precision{tag}_n', f'precision{tag}_ok', f'precision{tag}_ok_n']:
            bundle_meta[f'{k}_val'] = hold_metrics.get(k)

    bundle = ModelBundle(pipeline=pipe_final, bucket_calibrators=calibs_final, priors=priors_final, alpha=float(best_a['alpha']), feature_names=list(FEATURE_NAMES), meta=bundle_meta, strict_calibrator=strict_calibrator, strict_calibration_blend=strict_blend)

    pt_dir = model_dir + '/pt1'
    os.makedirs(pt_dir, exist_ok=True)
    joblib.dump(bundle, pt_dir + '/bundle.joblib')

    # Save Model B
    if path_model is not None:
        joblib.dump(path_model, pt_dir + '/path_quality.joblib')

    return {
        'target_name': 'decomposed_touch_x_path_1pct',
        'target_definition': bundle_meta['target_definition'],
        'brier_val': hold_metrics.get('brier_strict'),
        'auc_val': hold_metrics.get('auc_strict'),
        'auc_touch_val': hold_metrics.get('auc_touch'),
        'model_a': bundle_meta['model_a'],
        'model_b': model_b_result,
        **{f'precision{tag}_val': hold_metrics.get(f'precision{tag}') for tag in ['60', '70', '75', '80']},
        **{f'precision{tag}_n_val': hold_metrics.get(f'precision{tag}_n') for tag in ['60', '70', '75', '80']},
        **{f'precision{tag}_ok_val': hold_metrics.get(f'precision{tag}_ok') for tag in ['60', '70', '75', '80']},
        **{f'precision{tag}_ok_n_val': hold_metrics.get(f'precision{tag}_ok_n') for tag in ['60', '70', '75', '80']},
        'selection_tier': bundle_meta['selection_tier'], 'selection_warning': bundle_meta['selection_warning'],
        'selection_score': sel_score, 'strict_calibration': strict_calibration_result,
        'class_weight': best_a['cw'], 'calibrator': best_a['cal_mode'], 'alpha': float(best_a['alpha']),
        'C': float(best_a['C']), 'l1_ratio': float(best_a['l1']),
        'stage1_pass_rate': float(np.mean(stage1_pass)),
        'rows_full': int(X_full.shape[0]), 'rows_stage1': int(X.shape[0]),
        'raw_touch_count': hold_metrics['raw_touch_count'], 'strict_touch_count': hold_metrics['strict_touch_count'],
        'clean_touch_count': hold_metrics['clean_touch_count'], 'bouncy_touch_count': hold_metrics['bouncy_touch_count'],
        'ugly_touch_count': hold_metrics['ugly_touch_count'], 'worthy_count': hold_metrics['worthy_count'],
        'missed_worthy_count': hold_metrics['missed_worthy_count'],
        'time_to_touch': hold_metrics['time_to_touch'], 'mae_before_touch': hold_metrics['mae_before_touch'],
        'close_vs_scan': hold_metrics['close_vs_scan'],
        'tail_path_75_ok': hold_metrics['tail_path_75_ok'], 'challenge_set': hold_metrics['challenge_set'],
        'touch_tail_validated': touch_tail_validated, 'touch_tail_metrics': touch_tail_metrics,
        'decision_tail_validated': decision_tail_validated, 'decision_tail_metrics': decision_tail_metrics,
        'guardrail_profiles': learned_guardrail_profiles,
        'probability_contract': bundle_meta['probability_contract'],
        'topk_metrics_holdout': topk_metrics_holdout,
        'topk_metrics_val_final': topk_metrics_val_final,
        'setup_family_profiles': setup_family_profiles,
        'family_policy_metrics_holdout': family_policy_metrics_holdout,
        'family_policy_metrics_validation': family_policy_metrics_validation,
        'policy_profile_source': 'validation',
    }


def run_training(settings, symbols, sector_map):
    if settings.demo_mode:
        raise RuntimeError('Training is disabled in DEMO_MODE.')
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        raise RuntimeError('Missing Alpaca keys.')
    client = AlpacaClient(settings.alpaca_api_key, settings.alpaca_api_secret, feed=settings.normalized_feed())
    profiles = compute_profiles(client=client, symbols=symbols, tz_name=settings.timezone, lookback_days=settings.tod_rvol_lookback_days, min_days=settings.tod_rvol_min_days)
    save_profiles(settings.model_dir, profiles)
    dataset, err = build_training_dataset(
        client=client, symbols=symbols, sector_map=sector_map,
        lookback_days=settings.train_lookback_days, tz_name=settings.timezone,
        scan_interval_minutes=settings.scan_interval_minutes, tod_min_days=settings.tod_rvol_min_days,
        liq_rolling_bars=settings.liq_rolling_bars,
        liq_thresholds=(settings.liq_dvol_min_usd, settings.liq_range_pct_max, settings.liq_wick_atr_max),
        blocked_params={
            'ret20d_max': settings.blocked_ret20d_max, 'ret60d_max': settings.blocked_ret60d_max,
            'dist50dma_max': settings.blocked_dist50dma_max, 'ret_since_open_max': settings.blocked_ret_since_open_max,
            'damage_from_high_atr_min': settings.blocked_damage_from_high_atr_min, 'below_vwap_frac_min': settings.blocked_below_vwap_frac_min,
            'event_gap_abs_min': settings.event_gap_abs_min, 'event_rvol_min': settings.event_rvol_min, 'event_range_pct_min': settings.event_range_pct_min,
        },
        strict_touch_mae_threshold=settings.strict_touch_mae_threshold,
        worthy_close_vs_scan_min=settings.worthy_close_vs_scan_min,
        diag_held_minutes=settings.diag_held_minutes, diag_held_fraction=settings.diag_held_fraction,
    )
    if err: raise RuntimeError(err)
    meta1 = _select_and_train_bundle(dataset, settings, settings.model_dir)
    avail = sum(1 for p in profiles.values() if p.available)
    return {
        'pt1': meta1,
        'volume_profiles': {'symbols': len(profiles), 'available': avail, 'missing': len(profiles) - avail, 'lookback_days': settings.tod_rvol_lookback_days, 'min_days': settings.tod_rvol_min_days},
    }
