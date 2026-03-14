from __future__ import annotations
import math
import os
from datetime import date, datetime, timedelta, timezone
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
from .features import FEATURE_NAMES, compute_features_from_5m
from .modeling import (
    Calibrator,
    ModelBundle,
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

    def ret_n(n: int) -> float:
        if closes.size <= n:
            return 0.0
        ref = float(closes[-(n + 1)])
        return float(last / ref - 1.0) if ref > 0 else 0.0

    def sma_dist(n: int) -> float:
        if closes.size < n:
            return 0.0
        ma = float(np.mean(closes[-n:]))
        return float(last / ma - 1.0) if ma > 0 else 0.0

    def drawdown_n(n: int) -> float:
        if highs.size >= n:
            peak = float(np.max(highs[-n:]))
        else:
            peak = float(np.max(highs)) if highs.size else 0.0
        return float(last / peak - 1.0) if peak > 0 else 0.0

    return {
        'ret_5d': ret_n(5),
        'ret_20d': ret_n(20),
        'ret_60d': ret_n(60),
        'dist_20dma_pct': sma_dist(20),
        'dist_50dma_pct': sma_dist(50),
        'dist_200dma_pct': sma_dist(200),
        'drawdown_20d_pct': drawdown_n(20),
        'drawdown_60d_pct': drawdown_n(60),
    }


def _suffix_max(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    m = -np.inf
    for i in range(arr.size - 1, -1, -1):
        if arr[i] > m:
            m = arr[i]
        out[i] = m
    return out


def _split_day_masks(day_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
    unique_days = sorted(int(x) for x in np.unique(day_idx))
    n = len(unique_days)
    if n < 8:
        raise RuntimeError('Need at least 8 trading days for time-aware split.')
    n_hold = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_cal = max(1, int(round(n * 0.15)))
    n_tr = max(1, n - n_hold - n_val - n_cal)
    while n_tr + n_cal + n_val + n_hold > n:
        n_tr -= 1
    tr_days = unique_days[:n_tr]
    cal_days = unique_days[n_tr:n_tr + n_cal]
    val_days = unique_days[n_tr + n_cal:n_tr + n_cal + n_val]
    hold_days = unique_days[n_tr + n_cal + n_val:]
    tr_mask = np.isin(day_idx, tr_days)
    cal_mask = np.isin(day_idx, cal_days)
    val_mask = np.isin(day_idx, val_days)
    hold_mask = np.isin(day_idx, hold_days)
    return tr_mask, cal_mask, val_mask, hold_mask, {
        'train_days': tr_days,
        'cal_days': cal_days,
        'val_days': val_days,
        'holdout_days': hold_days,
    }


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if y.size else float('nan')


def _auc_safe(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _fit_platt(raw: np.ndarray, y: np.ndarray) -> Calibrator:
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(raw.reshape(-1, 1), y)
    return Calibrator(method='platt', model=lr)


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> Optional[Calibrator]:
    if raw.size < 800 or len(np.unique(y)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw.reshape(-1), y.reshape(-1))
    return Calibrator(method='isotonic', model=iso)


def _fit_best_calibrator(raw: np.ndarray, y: np.ndarray, min_samples: int) -> Calibrator:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if raw.size < max(min_samples, 200):
        return _fit_platt(raw, y)
    split = int(0.8 * raw.size)
    fit_idx = np.arange(split)
    eval_idx = np.arange(split, raw.size) if split < raw.size else np.arange(raw.size)
    pl = _fit_platt(raw[fit_idx], y[fit_idx])
    pl_b = _brier(y[eval_idx].astype(float), pl.predict(raw[eval_idx]).astype(float))
    iso = _fit_isotonic(raw[fit_idx], y[fit_idx])
    if iso is None:
        return pl
    iso_b = _brier(y[eval_idx].astype(float), iso.predict(raw[eval_idx]).astype(float))
    return iso if iso_b <= pl_b else pl


def _bucket_priors_segmented(y: np.ndarray, ttc: np.ndarray, risk: np.ndarray) -> Dict[str, float]:
    priors: Dict[str, float] = {'global': float(np.mean(y)) if y.size else 0.5}
    for rk in ['OK', 'HIGH', 'BLOCKED']:
        mask_r = risk == rk
        if np.any(mask_r):
            priors[f'{rk}|global'] = float(np.mean(y[mask_r]))
        for bname, lo, hi in [('0_30', 0.0, 30.0), ('30_60', 30.0, 60.0), ('60_120', 60.0, 120.0), ('120_240', 120.0, 240.0), ('240_390', 240.0, 9999.0)]:
            mins = ttc * 390.0
            mask = mask_r & (mins >= lo) & (mins < hi)
            if np.any(mask):
                priors[f'{rk}|{bname}'] = float(np.mean(y[mask]))
    return priors


def _fit_segmented_calibrators(raw: np.ndarray, y: np.ndarray, ttc: np.ndarray, risk: np.ndarray, min_samples: int, mode: str) -> Tuple[Dict[str, Calibrator], Dict[str, str]]:
    calibs: Dict[str, Calibrator] = {}
    methods: Dict[str, str] = {}
    global_cal = _fit_best_calibrator(raw, y, min_samples)
    calibs['global'] = global_cal
    methods['global'] = global_cal.method
    if mode == 'global':
        return calibs, methods
    mins = ttc * 390.0
    for rk in ['OK', 'HIGH', 'BLOCKED']:
        mask_r = risk == rk
        if np.sum(mask_r) >= min_samples and len(np.unique(y[mask_r])) >= 2:
            cal = _fit_best_calibrator(raw[mask_r], y[mask_r], min_samples)
            calibs[f'{rk}|global'] = cal
            methods[f'{rk}|global'] = cal.method
        for bname, lo, hi in [('0_30', 0.0, 30.0), ('30_60', 30.0, 60.0), ('60_120', 60.0, 120.0), ('120_240', 120.0, 240.0), ('240_390', 240.0, 9999.0)]:
            mask = mask_r & (mins >= lo) & (mins < hi)
            if np.sum(mask) >= min_samples and len(np.unique(y[mask])) >= 2:
                cal = _fit_best_calibrator(raw[mask], y[mask], min_samples)
                calibs[f'{rk}|{bname}'] = cal
                methods[f'{rk}|{bname}'] = cal.method
    return calibs, methods


def _apply_calibration(raw: np.ndarray, ttc: np.ndarray, risk: np.ndarray, calibs: Dict[str, Calibrator], priors: Dict[str, float], alpha: float) -> np.ndarray:
    out = np.empty(raw.shape[0], dtype=float)
    for i in range(raw.shape[0]):
        b = bucket_name_from_ttc(ttc[i])
        rk = str(risk[i])
        cal = calibs.get(f'{rk}|{b}') or calibs.get(f'{rk}|global') or calibs.get('global')
        p = sigmoid(np.array([raw[i]]))[0] if cal is None else cal.predict(np.array([raw[i]]))[0]
        prior = float(priors.get(f'{rk}|{b}', priors.get(f'{rk}|global', priors.get('global', 0.5))))
        out[i] = np.clip(alpha * p + (1.0 - alpha) * prior, 0.0, 1.0)
    return out


def _precision_at_threshold(y: np.ndarray, p: np.ndarray, mask: np.ndarray, thr: float) -> Tuple[Optional[float], int]:
    m = mask & (p >= thr)
    n = int(np.sum(m))
    if n == 0:
        return None, 0
    return float(np.mean(y[m])), n


def _path_diagnostics(y: np.ndarray, diag: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    n = int(np.sum(mask))
    out['n'] = n
    if n == 0:
        out.update({
            'touch_rate': None,
            'sane_touch_rate': None,
            'held_above_scan_10m_rate': None,
            'mean_mae_before_touch_pct': None,
            'median_close_vs_scan_pct': None,
        })
        return out
    out['touch_rate'] = float(np.mean(y[mask]))
    out['sane_touch_rate'] = float(np.mean(diag['sane_touch'][mask]))
    out['held_above_scan_10m_rate'] = float(np.mean(diag['held_above_scan_10m'][mask]))
    out['mean_mae_before_touch_pct'] = float(np.mean(diag['mae_before_touch_pct'][mask]))
    out['median_close_vs_scan_pct'] = float(np.median(diag['close_vs_scan_pct'][mask]))
    return out


def _final_metrics(X: np.ndarray, y: np.ndarray, p: np.ndarray, meta: Dict[str, object], diag: Dict[str, np.ndarray]) -> Dict[str, object]:
    acceptable = acceptable_long_mask_from_X(X, meta)
    out: Dict[str, object] = {
        'brier': _brier(y.astype(float), p.astype(float)),
        'auc': _auc_safe(y.astype(int), p.astype(float)),
    }
    for thr in [0.60, 0.70, 0.75, 0.80]:
        tag = str(int(round(thr * 100)))
        prec, n = _precision_at_threshold(y, p, np.ones_like(y, dtype=bool), thr)
        prec_ok, n_ok = _precision_at_threshold(y, p, acceptable, thr)
        out[f'precision{tag}'] = prec
        out[f'precision{tag}_n'] = n
        out[f'precision{tag}_ok'] = prec_ok
        out[f'precision{tag}_ok_n'] = n_ok
        out[f'unacceptable_ge{tag}_n'] = int(np.sum((~acceptable) & (p >= thr)))
    out['tail_path_75_ok'] = _path_diagnostics(y, diag, acceptable & (p >= 0.75))
    challenge_mask = (~acceptable) | (risk_bucket_from_X(X, meta) == 'BLOCKED') | event_risk_mask_from_X(X, meta) | (downside_risk_score_from_X(X, meta) >= float(meta.get('downside_high_threshold', 0.75)))
    out['challenge_set'] = {
        'count': int(np.sum(challenge_mask)),
        'avg_prob': float(np.mean(p[challenge_mask])) if np.any(challenge_mask) else None,
        'ge75_count': int(np.sum(challenge_mask & (p >= 0.75))),
        'touch_rate': float(np.mean(y[challenge_mask])) if np.any(challenge_mask) else None,
        'path': _path_diagnostics(y, diag, challenge_mask & (p >= 0.60)),
    }
    return out


def _make_base_pipeline(C: float, l1_ratio: float, class_weight: Optional[str]) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=float(l1_ratio),
            C=float(C),
            class_weight=class_weight,
            max_iter=2500,
            random_state=42,
        )),
    ])


def _selection_score(c: Dict[str, object], selection_min_count_75: int, selection_min_count_70: int) -> float:
    auc = float(c['auc']) if c['auc'] is not None else 0.5
    brier = float(c['brier'])
    def pv(keyp: str, keyn: str, cap: int) -> float:
        p = c.get(keyp)
        n = int(c.get(keyn, 0) or 0)
        if p is None or n <= 0:
            return 0.0
        return float(p) * math.log1p(min(n, cap))
    tail = (
        10.0 * pv('precision75_ok', 'precision75_ok_n', max(10, selection_min_count_75)) +
        7.0 * pv('precision80_ok', 'precision80_ok_n', 25) +
        6.0 * pv('precision70_ok', 'precision70_ok_n', max(10, selection_min_count_70)) +
        2.5 * pv('precision60_ok', 'precision60_ok_n', 60)
    )
    density_bonus = min(int(c.get('precision75_ok_n', 0) or 0), selection_min_count_75) * 0.18
    penalty = 0.45 * int(c.get('unacceptable_ge75_n', 0) or 0) + 0.30 * int(c.get('unacceptable_ge80_n', 0) or 0)
    sparse_tail_penalty = 1.40 if int(c.get('precision75_ok_n', 0) or 0) == 0 else 0.0
    sparse_tail_penalty += 0.75 if int(c.get('precision70_ok_n', 0) or 0) == 0 else 0.0
    return tail + density_bonus - penalty - sparse_tail_penalty - 20.0 * brier + 0.35 * auc


def _candidate_selection_tier(c: Dict[str, object], settings: Settings) -> Tuple[int, str, Optional[str]]:
    p75 = c.get('precision75_ok')
    n75 = int(c.get('precision75_ok_n', 0) or 0)
    p70 = c.get('precision70_ok')
    n70 = int(c.get('precision70_ok_n', 0) or 0)
    p60 = c.get('precision60_ok')
    n60 = int(c.get('precision60_ok_n', 0) or 0)
    if p75 is not None and n75 >= settings.selection_min_count_75 and float(p75) >= settings.selection_min_precision_75:
        return 3, 'tail_ready_75', None
    if p70 is not None and n70 >= settings.selection_min_count_70 and float(p70) >= settings.selection_min_precision_70:
        return 2, 'tail_ready_70_only', 'No validated >=0.75 acceptable tail on validation; live probabilities above the cap will be suppressed until holdout proves readiness.'
    if p60 is not None and n60 >= max(3, settings.selection_min_count_70 // 2) and float(p60) >= 0.55:
        return 1, 'tail_ready_60_only', 'Validation only supports weak top-tail evidence; live probabilities above the cap will be suppressed.'
    return 0, 'tail_not_ready', 'Validation produced no credible high-confidence acceptable tail; live probabilities above the cap will be suppressed.'


def _holdout_tail_readiness(hold: Dict[str, object], settings: Settings) -> Tuple[bool, bool, Optional[str]]:
    p75 = hold.get('precision75_ok')
    n75 = int(hold.get('precision75_ok_n', 0) or 0)
    p70 = hold.get('precision70_ok')
    n70 = int(hold.get('precision70_ok_n', 0) or 0)
    tail70 = bool(p70 is not None and n70 >= settings.selection_min_count_70 and float(p70) >= settings.selection_min_precision_70)
    tail75 = bool(p75 is not None and n75 >= settings.selection_min_count_75 and float(p75) >= settings.selection_min_precision_75)
    warning = None
    if not tail75:
        warning = 'Holdout did not validate a trustworthy >=0.75 acceptable tail; live probabilities are capped below the action threshold.'
    elif not tail70:
        warning = 'Holdout validated >=0.75 tail but not enough >=0.70 density; use discretion.'
    return tail70, tail75, warning


def _feature_group_ablation(bundle: ModelBundle, X_hold: np.ndarray, y_hold: np.ndarray, diag_hold: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    groups = {
        'previous_day': ['gap_prev_close_pct', 'dist_prev_high_atr', 'dist_prev_low_atr', 'prev_day_range_pct'],
        'daily_trend': ['ret_5d', 'ret_20d', 'ret_60d', 'dist_20dma_pct', 'dist_50dma_pct', 'dist_200dma_pct', 'drawdown_20d_pct', 'drawdown_60d_pct'],
        'trend_participation': ['ema_diff_pct', 'adx', 'rvol_tod', 'obv_slope_norm', 'vwap_loc'],
        'intraday_damage': ['ret_since_open_pct', 'bars_below_vwap_frac', 'no_reclaim_vwap', 'damage_from_high_atr'],
        'path_quality': ['path_smoothness_30m', 'reversal_count_30m', 'or_breakout_state'],
    }
    base_p = bundle.predict_proba(X_hold)
    base = _final_metrics(X_hold, y_hold, base_p, bundle.meta, diag_hold)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for g, cols in groups.items():
        X_mod = X_hold.copy()
        for col in cols:
            if col in FEATURE_NAMES:
                X_mod[:, FEATURE_NAMES.index(col)] = 0.0
        p = bundle.predict_proba(X_mod)
        m = _final_metrics(X_mod, y_hold, p, bundle.meta, diag_hold)
        out[g] = {
            'auc': m['auc'],
            'brier': m['brier'],
            'precision75_ok': m['precision75_ok'],
            'precision75_ok_n': m['precision75_ok_n'],
            'delta_auc': None if (base['auc'] is None or m['auc'] is None) else float(m['auc'] - base['auc']),
            'delta_brier': float(m['brier'] - base['brier']),
        }
    return out


def _settings_meta(settings: Settings) -> Dict[str, object]:
    return {
        'liq_dvol_min_usd': float(settings.liq_dvol_min_usd),
        'liq_range_pct_max': float(settings.liq_range_pct_max),
        'liq_wick_atr_max': float(settings.liq_wick_atr_max),
        'blocked_ret20d_max': float(settings.blocked_ret20d_max),
        'blocked_ret60d_max': float(settings.blocked_ret60d_max),
        'blocked_dist50dma_max': float(settings.blocked_dist50dma_max),
        'blocked_ret_since_open_max': float(settings.blocked_ret_since_open_max),
        'blocked_damage_from_high_atr_min': float(settings.blocked_damage_from_high_atr_min),
        'blocked_below_vwap_frac_min': float(settings.blocked_below_vwap_frac_min),
        'blocked_prob_cap': float(settings.blocked_prob_cap),
        'event_gap_abs_min': float(settings.event_gap_abs_min),
        'event_rvol_min': float(settings.event_rvol_min),
        'event_range_pct_min': float(settings.event_range_pct_min),
        'event_prob_cap': float(settings.event_prob_cap),
        'uncertainty_z_thresh': float(settings.uncertainty_z_thresh),
        'uncertainty_extreme_features_min': int(settings.uncertainty_extreme_features_min),
        'uncertainty_prob_cap': float(settings.uncertainty_prob_cap),
        'downside_prob_cap_high': float(settings.downside_prob_cap_high),
        'downside_prob_cap_medium': float(settings.downside_prob_cap_medium),
        'downside_high_threshold': float(settings.downside_high_threshold),
        'downside_medium_threshold': float(settings.downside_medium_threshold),
        'stage1_candidate_cap': int(settings.stage1_candidate_cap),
        'stage1_min_score': float(settings.stage1_min_score),
        'stage1_min_minutes_since_open': int(settings.stage1_min_minutes_since_open),
        'stage1_min_minutes_to_close': int(settings.stage1_min_minutes_to_close),
        'stage1_min_rvol': float(settings.stage1_min_rvol),
        'stage1_min_dollar_volume_mult': float(settings.stage1_min_dollar_volume_mult),
        'tail_not_ready_prob_cap': float(settings.tail_not_ready_prob_cap),
    }


def build_training_dataset(
    client: AlpacaClient,
    symbols: List[str],
    sector_map: Dict[str, str],
    lookback_days: int,
    tz_name: str,
    scan_interval_minutes: int,
    tod_min_days: int,
    liq_rolling_bars: int,
    liq_thresholds: Tuple[float, float, float],
    blocked_params: Dict[str, float],
):
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    days = _trading_days(today_local, lookback_days, tz_name)
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    day_idx_rows: List[int] = []
    mae_before_touch_rows: List[float] = []
    sane_touch_rows: List[float] = []
    held_above_rows: List[float] = []
    close_vs_scan_rows: List[float] = []

    slot_hist: Dict[str, Dict[int, List[float]]] = {s: {i: [] for i in range(78)} for s in symbols}
    day_count_hist: Dict[str, int] = {s: 0 for s in symbols}
    step = max(1, int(scan_interval_minutes / 5)) if scan_interval_minutes >= 5 else 1

    sector_etfs = sorted({sector_etf_for_sector(sector_map.get(sym, '')) for sym in symbols})
    symbols_daily = list(dict.fromkeys(symbols + ['SPY'] + sector_etfs))
    first_day = days[0] if days else today_local
    first_open_utc, _ = _session_utc_for_day(first_day, tz_name)
    daily_start = first_open_utc - timedelta(days=500)
    daily_bars, errd, _ = client.get_bars(symbols_daily, '1Day', start_utc=daily_start, end_utc=datetime.now(timezone.utc))
    if errd:
        return None, f'daily bars fetch failed: {errd}'

    for di, d in enumerate(days):
        prev_d = days[di - 1] if di > 0 else d
        prev_open_utc, prev_close_utc = _session_utc_for_day(prev_d, tz_name)
        open_utc, close_utc = _session_utc_for_day(d, tz_name)

        symbols_5m = list(dict.fromkeys(symbols + ['SPY'] + sector_etfs))
        bars5, err5, _ = client.get_bars(symbols_5m, '5Min', start_utc=prev_open_utc, end_utc=close_utc)
        if err5:
            return None, f'bars5 fetch failed {d}: {err5}'
        bars1, err1, _ = client.get_bars(symbols_5m, '1Min', start_utc=open_utc, end_utc=close_utc)
        if err1:
            return None, f'bars1 fetch failed {d}: {err1}'

        def full_session_and_current(lst: List[dict]):
            full: List[dict] = []
            cur: List[dict] = []
            cur_idx: List[int] = []
            for idx, b in enumerate(lst):
                t = _parse_ts(b['t'])
                if prev_open_utc <= t <= prev_close_utc or open_utc <= t <= close_utc:
                    full.append(b)
                    if open_utc <= t <= close_utc:
                        cur.append(b)
                        cur_idx.append(len(full) - 1)
            return full, cur, np.array(cur_idx, dtype=int)

        _spy_full, spy_cur, _ = full_session_and_current(bars5.get('SPY', []))
        spy_ret_30m_arr = None
        if len(spy_cur) >= 7:
            spy_c = np.array([float(b.get('c') or 0.0) for b in spy_cur], dtype=float)
            spy_ret_30m_arr = np.zeros_like(spy_c)
            for i in range(spy_c.size):
                spy_ret_30m_arr[i] = (spy_c[i] / spy_c[i - 6] - 1.0) if i >= 6 and spy_c[i - 6] else 0.0

        sector_ret_arrs: Dict[str, np.ndarray] = {}
        for etf in sector_etfs:
            _f5, cur, _ = full_session_and_current(bars5.get(etf, []))
            if len(cur) >= 7:
                cc = np.array([float(b.get('c') or 0.0) for b in cur], dtype=float)
                arr = np.zeros_like(cc)
                for i in range(cc.size):
                    arr[i] = (cc[i] / cc[i - 6] - 1.0) if i >= 6 and cc[i - 6] else 0.0
                sector_ret_arrs[etf] = arr
            else:
                sector_ret_arrs[etf] = np.zeros(0, dtype=float)

        today_slot_vols: Dict[str, Dict[int, List[float]]] = {s: {i: [] for i in range(78)} for s in symbols}

        for sym in symbols:
            full5, cur5, cur_indices = full_session_and_current(bars5.get(sym, []))
            cur1 = []
            for b in bars1.get(sym, []):
                t = _parse_ts(b['t'])
                if open_utc <= t <= close_utc:
                    cur1.append(b)
            if not cur5 or not cur1 or len(full5) < 7:
                continue
            prev_session = [b for b in full5 if prev_open_utc <= _parse_ts(b['t']) <= prev_close_utc]
            prev_day_close = float(prev_session[-1].get('c') or 0.0) if prev_session else None
            prev_day_high = float(max(float(b.get('h') or 0.0) for b in prev_session)) if prev_session else None
            prev_day_low = float(min(float(b.get('l') or 0.0) for b in prev_session)) if prev_session else None

            t1 = np.array([_parse_ts(b['t']).timestamp() for b in cur1], dtype=float)
            h1 = np.array([float(b.get('h') or 0.0) for b in cur1], dtype=float)
            l1 = np.array([float(b.get('l') or 0.0) for b in cur1], dtype=float)
            c1 = np.array([float(b.get('c') or 0.0) for b in cur1], dtype=float)
            if h1.size < 2:
                continue
            suf = _suffix_max(h1)

            etf = sector_etf_for_sector(sector_map.get(sym, ''))
            etf_arr = sector_ret_arrs.get(etf)
            daily_ctx = _daily_ctx_from_bars(daily_bars.get(sym, []), d)

            for cur_pos, full_idx in enumerate(cur_indices):
                if (cur_pos % step) != 0:
                    continue
                hist5 = full5[: full_idx + 1]
                if len(hist5) < 7:
                    continue
                current_bar = cur5[cur_pos]
                current_ts = _parse_ts(current_bar['t'])
                slot = slot_index_from_ts(current_ts, tz_name)
                if slot is not None:
                    try:
                        today_slot_vols[sym][slot].append(float(current_bar.get('v') or 0.0))
                    except Exception:
                        pass
                baseline = None
                if slot is not None and day_count_hist.get(sym, 0) >= tod_min_days:
                    vals = slot_hist[sym][slot]
                    if vals:
                        baseline = float(np.median(np.array(vals, dtype=float)))

                price = float(current_bar.get('c') or 0.0)
                scan_end_ts = current_ts.timestamp() + 5 * 60.0
                j = int(np.searchsorted(t1, scan_end_ts, side='left'))
                if j >= t1.size or price <= 0:
                    continue
                h_future = float(suf[j])
                y = 1 if h_future >= 1.01 * price else 0
                mins_to_close = max(0.0, (close_utc - datetime.fromtimestamp(scan_end_ts, tz=timezone.utc)).total_seconds() / 60.0)
                spy_ret = float(spy_ret_30m_arr[cur_pos]) if spy_ret_30m_arr is not None and cur_pos < spy_ret_30m_arr.size else 0.0
                sec_ret = float(etf_arr[cur_pos]) if etf_arr is not None and cur_pos < etf_arr.size else 0.0
                fr = compute_features_from_5m(
                    bars_5m=hist5,
                    spy_ret_30m=spy_ret,
                    sector_ret_30m=sec_ret,
                    mins_to_close=mins_to_close,
                    tod_baseline_vol_median=baseline,
                    rolling_rvol_window=20,
                    risk_params=(liq_rolling_bars, liq_thresholds[0], liq_thresholds[1], liq_thresholds[2]),
                    prev_day_close=prev_day_close,
                    prev_day_high=prev_day_high,
                    prev_day_low=prev_day_low,
                    daily_ctx=daily_ctx,
                    tz_name=tz_name,
                    blocked_params=blocked_params,
                )
                if fr is None:
                    continue

                future_h = h1[j:]
                future_l = l1[j:]
                future_c = c1[j:]
                touch_rel_idx = np.where(future_h >= 1.01 * price)[0]
                if touch_rel_idx.size:
                    touch_idx = int(touch_rel_idx[0])
                    mae_before_touch = float(np.min(future_l[:touch_idx + 1]) / price - 1.0)
                    sane_touch = 1.0 if mae_before_touch > -0.0075 else 0.0
                    hold_end = min(future_c.size, touch_idx + 10)
                    future_after_touch = future_c[touch_idx:hold_end]
                    held_above = 1.0 if (future_after_touch.size >= 3 and np.mean(future_after_touch >= price) >= 0.7) else 0.0
                else:
                    mae_before_touch = float(np.min(future_l) / price - 1.0) if future_l.size else 0.0
                    sane_touch = 0.0
                    held_above = 0.0
                close_vs_scan = float(future_c[-1] / price - 1.0) if future_c.size else 0.0

                X_rows.append(fr.features)
                y_rows.append(y)
                day_idx_rows.append(di)
                mae_before_touch_rows.append(mae_before_touch)
                sane_touch_rows.append(sane_touch)
                held_above_rows.append(held_above)
                close_vs_scan_rows.append(close_vs_scan)

        for sym in symbols:
            any_day_obs = False
            for slot, vals in today_slot_vols[sym].items():
                if vals:
                    any_day_obs = True
                    slot_hist[sym][slot].append(float(np.sum(vals)))
            if any_day_obs:
                day_count_hist[sym] += 1

    if not X_rows:
        return None, 'no training rows produced'
    X = np.vstack(X_rows)
    diag = {
        'mae_before_touch_pct': np.array(mae_before_touch_rows, dtype=float),
        'sane_touch': np.array(sane_touch_rows, dtype=float),
        'held_above_scan_10m': np.array(held_above_rows, dtype=float),
        'close_vs_scan_pct': np.array(close_vs_scan_rows, dtype=float),
    }
    return {
        'X': X,
        'y': np.array(y_rows, dtype=int),
        'day_idx': np.array(day_idx_rows, dtype=int),
        'diag': diag,
    }, None


def _select_and_train_bundle(dataset: Dict[str, object], settings: Settings, model_dir: str) -> Dict[str, object]:
    X_full = np.asarray(dataset['X'], dtype=float)
    y_full = np.asarray(dataset['y'], dtype=int)
    day_idx_full = np.asarray(dataset['day_idx'], dtype=int)
    diag_full = {k: np.asarray(v) for k, v in dict(dataset['diag']).items()}

    meta_cfg = _settings_meta(settings)
    stage1_score, stage1_pass, _stage1_reasons, _flags = stage1_diagnostics_from_X(X_full, meta_cfg)
    if np.sum(stage1_pass) < 1500:
        raise RuntimeError(f'Too few stage-1 training rows ({int(np.sum(stage1_pass))}). Lower STAGE1_MIN_SCORE or widen training window.')

    X = X_full[stage1_pass]
    y = y_full[stage1_pass]
    day_idx = day_idx_full[stage1_pass]
    diag = {k: v[stage1_pass] for k, v in diag_full.items()}

    tr_mask, cal_mask, val_mask, hold_mask, split_days = _split_day_masks(day_idx)
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_hold, y_hold = X[hold_mask], y[hold_mask]
    diag_hold = {k: v[hold_mask] for k, v in diag.items()}

    if len(np.unique(y_tr)) < 2:
        raise RuntimeError('Training labels have only one class after stage-1 filtering; widen lookback or symbols.')

    risk_cal = risk_bucket_from_X(X_cal, meta_cfg)
    risk_val = risk_bucket_from_X(X_val, meta_cfg)
    risk_hold = risk_bucket_from_X(X_hold, meta_cfg)
    ttc_cal = X_cal[:, FEATURE_NAMES.index('ttc_frac')]
    ttc_val = X_val[:, FEATURE_NAMES.index('ttc_frac')]
    ttc_hold = X_hold[:, FEATURE_NAMES.index('ttc_frac')]

    X_traincal = np.vstack([X_tr, X_cal])
    y_traincal = np.concatenate([y_tr, y_cal])
    risk_traincal = risk_bucket_from_X(X_traincal, meta_cfg)
    ttc_traincal = X_traincal[:, FEATURE_NAMES.index('ttc_frac')]

    candidates = []
    for cw in [None, 'balanced']:
        for C in settings.enet_c_values:
            for l1 in settings.enet_l1_values:
                pipe = _make_base_pipeline(C=C, l1_ratio=l1, class_weight=cw)
                pipe.fit(X_tr, y_tr)
                raw_cal = pipe.decision_function(X_cal)
                raw_val = pipe.decision_function(X_val)
                priors = _bucket_priors_segmented(y_traincal, ttc_traincal, risk_traincal)
                for cal_mode in ['bucketed', 'global']:
                    calibs, methods = _fit_segmented_calibrators(raw_cal, y_cal, ttc_cal, risk_cal, settings.calib_min_bucket_samples, cal_mode)
                    for alpha in settings.prior_alpha_values:
                        p_final = _apply_calibration(raw_val, ttc_val, risk_val, calibs, priors, float(alpha))
                        metrics = _final_metrics(X_val, y_val, p_final, meta_cfg, {k: v[val_mask] for k, v in diag.items()})
                        cand = {
                            'cw': 'balanced' if cw == 'balanced' else 'none',
                            'C': float(C),
                            'l1': float(l1),
                            'alpha': float(alpha),
                            'brier': float(metrics['brier']),
                            'auc': metrics['auc'],
                            'precision60': metrics['precision60'],
                            'precision60_n': metrics['precision60_n'],
                            'precision70': metrics['precision70'],
                            'precision70_n': metrics['precision70_n'],
                            'precision75': metrics['precision75'],
                            'precision75_n': metrics['precision75_n'],
                            'precision80': metrics['precision80'],
                            'precision80_n': metrics['precision80_n'],
                            'precision60_ok': metrics['precision60_ok'],
                            'precision60_ok_n': metrics['precision60_ok_n'],
                            'precision70_ok': metrics['precision70_ok'],
                            'precision70_ok_n': metrics['precision70_ok_n'],
                            'precision75_ok': metrics['precision75_ok'],
                            'precision75_ok_n': metrics['precision75_ok_n'],
                            'precision80_ok': metrics['precision80_ok'],
                            'precision80_ok_n': metrics['precision80_ok_n'],
                            'unacceptable_ge75_n': metrics['unacceptable_ge75_n'],
                            'unacceptable_ge80_n': metrics['unacceptable_ge80_n'],
                            'methods': methods,
                            'cal_mode': cal_mode,
                        }
                        tier_rank, selection_tier, selection_warning = _candidate_selection_tier(cand, settings)
                        cand['selection_tier_rank'] = int(tier_rank)
                        cand['selection_tier'] = selection_tier
                        cand['selection_warning'] = selection_warning
                        cand['selection_score'] = _selection_score(cand, settings.selection_min_count_75, settings.selection_min_count_70)
                        candidates.append(cand)

    best = max(candidates, key=lambda c: (int(c['selection_tier_rank']), float(c['selection_score'])))

    X_fit = np.vstack([X_tr, X_cal])
    y_fit = np.concatenate([y_tr, y_cal])
    risk_fit = risk_bucket_from_X(X_fit, meta_cfg)
    ttc_fit = X_fit[:, FEATURE_NAMES.index('ttc_frac')]
    priors_final = _bucket_priors_segmented(y_fit, ttc_fit, risk_fit)
    pipe_final = _make_base_pipeline(C=best['C'], l1_ratio=best['l1'], class_weight=('balanced' if best['cw'] == 'balanced' else None))
    pipe_final.fit(X_fit, y_fit)
    raw_val_final = pipe_final.decision_function(X_val)
    calibs_final, methods_final = _fit_segmented_calibrators(raw_val_final, y_val, ttc_val, risk_val, settings.calib_min_bucket_samples, best['cal_mode'])
    raw_hold = pipe_final.decision_function(X_hold)
    p_hold = _apply_calibration(raw_hold, ttc_hold, risk_hold, calibs_final, priors_final, float(best['alpha']))

    feature_mean = np.mean(X_fit, axis=0).astype(float).tolist()
    feature_std = np.where(np.std(X_fit, axis=0) <= 1e-6, 1.0, np.std(X_fit, axis=0)).astype(float).tolist()
    bundle_meta = {
        **meta_cfg,
        'trained_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'n_rows_full': int(X_full.shape[0]),
        'n_rows_stage1': int(X.shape[0]),
        'stage1_pass_rate': float(np.mean(stage1_pass)),
        'split_days': split_days,
        'selected': {k: best[k] for k in ['cw', 'C', 'l1', 'alpha', 'brier', 'auc', 'precision60_ok', 'precision70_ok', 'precision75_ok', 'precision80_ok', 'precision60_ok_n', 'precision70_ok_n', 'precision75_ok_n', 'precision80_ok_n', 'selection_score', 'selection_tier', 'selection_tier_rank', 'selection_warning', 'cal_mode']},
        'bucket_calibration_methods': methods_final,
        'prior_means': priors_final,
        'calib_min_bucket_samples': int(settings.calib_min_bucket_samples),
        'calibrator': best['cal_mode'],
        'class_weight': best['cw'],
        'alpha': float(best['alpha']),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'selection_tier': best['selection_tier'],
        'selection_warning': best['selection_warning'],
    }
    bundle = ModelBundle(
        pipeline=pipe_final,
        bucket_calibrators=calibs_final,
        priors=priors_final,
        alpha=float(best['alpha']),
        feature_names=list(FEATURE_NAMES),
        meta=bundle_meta,
    )

    hold = _final_metrics(X_hold, y_hold, p_hold, bundle_meta, diag_hold)
    tail_ready_70, tail_ready_75, holdout_warning = _holdout_tail_readiness(hold, settings)
    live_prob_cap = 1.0 if tail_ready_75 else float(settings.tail_not_ready_prob_cap)
    bundle.meta.update({
        'auc_val': float(hold['auc']) if hold['auc'] is not None else None,
        'brier_val': float(hold['brier']),
        'precision60_val': hold['precision60'],
        'precision60_n_val': hold['precision60_n'],
        'precision70_val': hold['precision70'],
        'precision70_n_val': hold['precision70_n'],
        'precision75_val': hold['precision75'],
        'precision75_n_val': hold['precision75_n'],
        'precision80_val': hold['precision80'],
        'precision80_n_val': hold['precision80_n'],
        'precision60_ok_val': hold['precision60_ok'],
        'precision60_ok_n_val': hold['precision60_ok_n'],
        'precision70_ok_val': hold['precision70_ok'],
        'precision70_ok_n_val': hold['precision70_ok_n'],
        'precision75_ok_val': hold['precision75_ok'],
        'precision75_ok_n_val': hold['precision75_ok_n'],
        'precision80_ok_val': hold['precision80_ok'],
        'precision80_ok_n_val': hold['precision80_ok_n'],
        'challenge_set': hold['challenge_set'],
        'tail_path_75_ok': hold['tail_path_75_ok'],
        'tail_ready_70': tail_ready_70,
        'tail_ready_75': tail_ready_75,
        'selection_warning': holdout_warning or best.get('selection_warning'),
        'selection_tier': best['selection_tier'],
        'live_prob_cap': live_prob_cap,
    })

    pt_dir = os.path.join(model_dir, 'pt1')
    os.makedirs(pt_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(pt_dir, 'bundle.joblib'))
    ablation = _feature_group_ablation(bundle, X_hold, y_hold, diag_hold)

    return {
        'brier_val': float(hold['brier']),
        'auc_val': float(hold['auc']) if hold['auc'] is not None else None,
        'precision60_val': hold['precision60'],
        'precision60_n_val': hold['precision60_n'],
        'precision70_val': hold['precision70'],
        'precision70_n_val': hold['precision70_n'],
        'precision75_val': hold['precision75'],
        'precision75_n_val': hold['precision75_n'],
        'precision80_val': hold['precision80'],
        'precision80_n_val': hold['precision80_n'],
        'precision60_ok_val': hold['precision60_ok'],
        'precision60_ok_n_val': hold['precision60_ok_n'],
        'precision70_ok_val': hold['precision70_ok'],
        'precision70_ok_n_val': hold['precision70_ok_n'],
        'precision75_ok_val': hold['precision75_ok'],
        'precision75_ok_n_val': hold['precision75_ok_n'],
        'precision80_ok_val': hold['precision80_ok'],
        'precision80_ok_n_val': hold['precision80_ok_n'],
        'tail_ready_70': tail_ready_70,
        'tail_ready_75': tail_ready_75,
        'selection_tier': best['selection_tier'],
        'selection_warning': holdout_warning or best.get('selection_warning'),
        'live_prob_cap': live_prob_cap,
        'class_weight': best['cw'],
        'calibrator': best['cal_mode'],
        'alpha': float(best['alpha']),
        'C': float(best['C']),
        'l1_ratio': float(best['l1']),
        'selection_score': float(best['selection_score']),
        'stage1_pass_rate': float(np.mean(stage1_pass)),
        'rows_full': int(X_full.shape[0]),
        'rows_stage1': int(X.shape[0]),
        'tail_path_75_ok': hold['tail_path_75_ok'],
        'challenge_set': hold['challenge_set'],
        'ablation_holdout': ablation,
    }


def run_training(settings: Settings, symbols: List[str], sector_map: Dict[str, str]) -> Dict[str, object]:
    if settings.demo_mode:
        raise RuntimeError('Training is disabled in DEMO_MODE. Set DEMO_MODE=false and provide Alpaca keys.')
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        raise RuntimeError('Missing Alpaca keys.')
    client = AlpacaClient(settings.alpaca_api_key, settings.alpaca_api_secret, feed=settings.normalized_feed())
    profiles = compute_profiles(client=client, symbols=symbols, tz_name=settings.timezone, lookback_days=settings.tod_rvol_lookback_days, min_days=settings.tod_rvol_min_days)
    save_profiles(settings.model_dir, profiles)
    dataset, err = build_training_dataset(
        client=client,
        symbols=symbols,
        sector_map=sector_map,
        lookback_days=settings.train_lookback_days,
        tz_name=settings.timezone,
        scan_interval_minutes=settings.scan_interval_minutes,
        tod_min_days=settings.tod_rvol_min_days,
        liq_rolling_bars=settings.liq_rolling_bars,
        liq_thresholds=(settings.liq_dvol_min_usd, settings.liq_range_pct_max, settings.liq_wick_atr_max),
        blocked_params={
            'ret20d_max': settings.blocked_ret20d_max,
            'ret60d_max': settings.blocked_ret60d_max,
            'dist50dma_max': settings.blocked_dist50dma_max,
            'ret_since_open_max': settings.blocked_ret_since_open_max,
            'damage_from_high_atr_min': settings.blocked_damage_from_high_atr_min,
            'below_vwap_frac_min': settings.blocked_below_vwap_frac_min,
            'event_gap_abs_min': settings.event_gap_abs_min,
            'event_rvol_min': settings.event_rvol_min,
            'event_range_pct_min': settings.event_range_pct_min,
        },
    )
    if err:
        raise RuntimeError(err)
    meta1 = _select_and_train_bundle(dataset, settings, settings.model_dir)
    avail = sum(1 for p in profiles.values() if p.available)
    missing = len(profiles) - avail
    return {
        'pt1': meta1,
        'volume_profiles': {
            'symbols': len(profiles),
            'available': avail,
            'missing': missing,
            'lookback_days': settings.tod_rvol_lookback_days,
            'min_days': settings.tod_rvol_min_days,
        },
    }
