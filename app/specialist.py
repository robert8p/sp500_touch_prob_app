from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .features import FEATURE_NAMES

RET5_IDX = FEATURE_NAMES.index('ret_5m')
RET30_IDX = FEATURE_NAMES.index('ret_30m')
REL_SPY5_IDX = FEATURE_NAMES.index('rel_strength_spy_5m')
REL_SPY30_IDX = FEATURE_NAMES.index('rel_strength_spy_30m')
REL_SECTOR30_IDX = FEATURE_NAMES.index('rel_strength_sector_30m')
GAP_IDX = FEATURE_NAMES.index('gap_prev_close_pct')
EMA_IDX = FEATURE_NAMES.index('ema_diff_pct')
ADX_IDX = FEATURE_NAMES.index('adx')
RVOL_IDX = FEATURE_NAMES.index('rvol_tod')
DIST_VWAP_IDX = FEATURE_NAMES.index('dist_pct_to_vwap')
VWAP_FRAC30_IDX = FEATURE_NAMES.index('closes_above_vwap_frac_30m')
VWAP_SLOPE_IDX = FEATURE_NAMES.index('vwap_slope_30m')
RET_OPEN_IDX = FEATURE_NAMES.index('ret_since_open_pct')
PATH_SMOOTH_IDX = FEATURE_NAMES.index('path_smoothness_30m')
REVERSAL_IDX = FEATURE_NAMES.index('reversal_count_30m')
OR_STATE_IDX = FEATURE_NAMES.index('or_breakout_state')
DRAWDOWN_RECOVERY_IDX = FEATURE_NAMES.index('drawdown_recovery_30m')
RS_OPEN_IDX = FEATURE_NAMES.index('rel_strength_spy_since_open')
SECTOR_OPEN_IDX = FEATURE_NAMES.index('sector_ret_since_open')
EXTENSION_IDX = FEATURE_NAMES.index('extension_ratio')
REJECTION_IDX = FEATURE_NAMES.index('intraday_rejection_count')
VOL_TREND_IDX = FEATURE_NAMES.index('volume_trend_5bar')
RW_TOUCH_IDX = FEATURE_NAMES.index('rw_implied_touch_prob')
SPY_RET5_RAW_IDX = FEATURE_NAMES.index('spy_ret_5m_raw')
SPY_RV_IDX = FEATURE_NAMES.index('spy_rv_1h')
TTC_IDX = FEATURE_NAMES.index('ttc_frac')

FAMILIES: List[str] = [
    'GAP_TREND',
    'ORB_CONTINUATION',
    'SECTOR_LEADER',
    'PULLBACK_RECLAIM',
    'RS_TREND',
    'OTHER',
]

POLICY_ORDER = ['SUPPRESS', 'WATCHLIST_ONLY', 'CANDIDATE_ONLY', 'FULL_ACTIONABLE']


def _as_2d(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def classify_setup_families(X: Any) -> np.ndarray:
    arr = _as_2d(X)
    out = np.empty(arr.shape[0], dtype=object)
    for i, x in enumerate(arr):
        gap = float(x[GAP_IDX])
        or_state = float(x[OR_STATE_IDX])
        rel_spy30 = float(x[REL_SPY30_IDX])
        rel_spy5 = float(x[REL_SPY5_IDX])
        rel_sector = float(x[REL_SECTOR30_IDX])
        rel_open = float(x[RS_OPEN_IDX])
        sector_open = float(x[SECTOR_OPEN_IDX])
        ret_open = float(x[RET_OPEN_IDX])
        ret5 = float(x[RET5_IDX])
        ret30 = float(x[RET30_IDX])
        ema = float(x[EMA_IDX])
        adx = float(x[ADX_IDX])
        rvol = float(x[RVOL_IDX])
        dist_vwap = float(x[DIST_VWAP_IDX])
        vwap_frac = float(x[VWAP_FRAC30_IDX])
        vwap_slope = float(x[VWAP_SLOPE_IDX])
        drawdown_recovery = float(x[DRAWDOWN_RECOVERY_IDX])
        extension = float(x[EXTENSION_IDX])
        rejections = float(x[REJECTION_IDX])
        vol_trend = float(x[VOL_TREND_IDX])
        path_smooth = float(x[PATH_SMOOTH_IDX])
        reversals = float(x[REVERSAL_IDX])
        rw_touch = float(x[RW_TOUCH_IDX])

        fam = 'OTHER'
        if (
            or_state > 0.45 and ret_open > 0.003 and rel_spy30 > 0 and rvol >= 1.0
            and vwap_slope >= 0 and dist_vwap >= -0.0025 and rejections <= 2.5
        ):
            fam = 'ORB_CONTINUATION'
        elif (
            gap >= 0.003 and ret_open > 0.002 and ema > 0 and rel_spy30 > 0
            and vwap_slope >= 0 and dist_vwap >= -0.003 and extension <= 1.7
        ):
            fam = 'GAP_TREND'
        elif (
            rel_sector >= 0.003 and sector_open > 0 and rel_open >= 0.0015 and rel_spy30 >= -0.0005
            and ema >= -0.0005 and vwap_slope >= -0.0001
        ):
            fam = 'SECTOR_LEADER'
        elif (
            drawdown_recovery >= 0.35 and ret5 > 0 and dist_vwap >= -0.0035 and vwap_frac >= 0.50
            and rejections <= 2.5 and rw_touch >= 0.08
        ):
            fam = 'PULLBACK_RECLAIM'
        elif (
            rel_spy30 >= 0.0025 and rel_open >= 0.002 and ema > 0 and adx >= 16.0
            and vwap_slope >= 0 and path_smooth >= 0.45 and reversals <= 3.5 and vol_trend >= -0.15
        ):
            fam = 'RS_TREND'
        out[i] = fam
    return out


def classify_setup_family_vector(x: Any) -> str:
    return str(classify_setup_families(x)[0])


def compute_topk_metrics(probs: Any, y_strict: Any, acceptable: Any, ks: Iterable[int] = (3, 5, 10)) -> Dict[str, Any]:
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(y_strict, dtype=int).reshape(-1)
    acc = np.asarray(acceptable, dtype=bool).reshape(-1)
    n = p.size
    if n == 0:
        return {'base_strict_rate': None, 'n': 0, 'by_k': {f'top_{int(k)}': {'precision': None, 'lift': None, 'acceptable_share': None, 'count': 0} for k in ks}}
    order = np.argsort(p)[::-1]
    base = float(np.mean(y)) if n else None
    by_k: Dict[str, Any] = {}
    for k in ks:
        kk = min(int(k), n)
        idx = order[:kk]
        precision = float(np.mean(y[idx])) if kk > 0 else None
        acceptable_share = float(np.mean(acc[idx])) if kk > 0 else None
        lift = (precision / base) if (precision is not None and base and base > 0) else None
        by_k[f'top_{int(k)}'] = {
            'precision': precision,
            'lift': lift,
            'acceptable_share': acceptable_share,
            'count': kk,
        }
    return {'base_strict_rate': base, 'n': int(n), 'by_k': by_k}


def sector_group_from_sector(sector: str) -> str:
    s = (sector or '').strip().lower()
    if s in {'information technology', 'communication services'}:
        return 'TECH_COMM'
    if s in {'utilities', 'consumer staples', 'health care', 'real estate'}:
        return 'DEFENSIVE'
    if s in {'consumer discretionary', 'materials'}:
        return 'CYCLICAL'
    if s in {'industrials', 'energy'}:
        return 'INDUSTRIAL_ENERGY'
    if s == 'financials':
        return 'FINANCIAL'
    return 'OTHER'


def sector_groups_from_array(sectors: Any) -> np.ndarray:
    arr = np.asarray(sectors, dtype=object).reshape(-1)
    return np.array([sector_group_from_sector(str(x)) for x in arr], dtype=object)


def time_bucket_from_ttc(ttc_frac: Any) -> np.ndarray:
    mins_to_close = np.asarray(ttc_frac, dtype=float).reshape(-1) * 390.0
    mins_since_open = 390.0 - mins_to_close
    out = np.empty(mins_since_open.size, dtype=object)
    out[:] = 'MID'
    out[mins_since_open < 120.0] = 'EARLY'
    out[mins_since_open >= 240.0] = 'LATE'
    return out


def market_context_bucket_from_X(X: Any) -> np.ndarray:
    arr = _as_2d(X)
    spy_ret5 = arr[:, SPY_RET5_RAW_IDX]
    spy_rv = arr[:, SPY_RV_IDX]
    out = np.empty(arr.shape[0], dtype=object)
    out[:] = 'NEUTRAL'
    stress = (spy_ret5 <= -0.0025) | (spy_rv >= 0.012)
    supportive = (spy_ret5 >= 0.0015) & (spy_rv <= 0.006)
    out[stress] = 'STRESS'
    out[supportive] = 'SUPPORTIVE'
    return out


def _rank_weight(count: int, prior_strength: float) -> float:
    return float(count / max(count + prior_strength, 1.0))


def _quantile_edges(p: np.ndarray, bin_count: int) -> np.ndarray:
    if p.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0.0, 1.0, max(2, int(bin_count)) + 1)
    edges = np.quantile(np.clip(p, 0.0, 1.0), qs)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([0.0, 1.0], dtype=float)
    return edges.astype(float)


def _bin_index(prob: float, edges: np.ndarray) -> int:
    if edges.size <= 1:
        return 0
    idx = int(np.searchsorted(edges, float(np.clip(prob, 0.0, 1.0)), side='right') - 1)
    return max(0, min(idx, edges.size - 2))


def _compute_global_bin_stats(p: np.ndarray, y: np.ndarray, edges: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == edges.size - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        avg_prob = float(np.mean(p[mask])) if count else 0.0
        strict_rate = float(np.mean(y[mask])) if count else 0.0
        out.append({'bin': i, 'lo': lo, 'hi': hi, 'count': count, 'avg_prob': avg_prob, 'strict_rate': strict_rate})
    return out


def _conservative_bin_multiplier(
    family_prob: float,
    family_rate: float,
    family_count: int,
    global_rate: float,
    *,
    prior_strength: float,
    lower_clip: float = 0.88,
    upper_clip: float = 1.08,
) -> float:
    if family_prob <= 1e-9:
        return 1.0
    w = _rank_weight(family_count, prior_strength)
    shrunk_rate = (w * family_rate) + ((1.0 - w) * global_rate)
    # Conservative: never let family overlay become materially more optimistic than observed empirical fit.
    target_rate = min(shrunk_rate, max(global_rate, shrunk_rate))
    mult = target_rate / family_prob if family_prob > 0 else 1.0
    return float(np.clip(mult, lower_clip, upper_clip))


def _policy_level_from_metrics(
    count: int,
    lift: Optional[float],
    acceptable_rate: Optional[float],
    *,
    min_count: int,
    suppress_below_lift: float,
    promote_above_lift: float,
) -> str:
    acc = float(acceptable_rate or 0.0)
    if count < max(60, int(0.5 * min_count)):
        return 'WATCHLIST_ONLY'
    if count < min_count:
        return 'CANDIDATE_ONLY'
    if lift is None:
        return 'CANDIDATE_ONLY'
    if lift < max(0.65, suppress_below_lift - 0.12) and acc < 0.56:
        return 'SUPPRESS'
    if lift < suppress_below_lift:
        return 'WATCHLIST_ONLY'
    if lift >= promote_above_lift and acc >= 0.58:
        return 'FULL_ACTIONABLE'
    if lift >= 1.0:
        return 'FULL_ACTIONABLE'
    return 'CANDIDATE_ONLY'


def _threshold_multiplier_for_policy(policy: str, *, loosen_mult: float, tighten_mult: float) -> float:
    if policy == 'FULL_ACTIONABLE':
        return float(loosen_mult)
    if policy == 'CANDIDATE_ONLY':
        return 1.03
    if policy == 'WATCHLIST_ONLY':
        return max(1.08, float(tighten_mult) - 0.01)
    if policy == 'SUPPRESS':
        return max(1.15, float(tighten_mult) + 0.03)
    return 1.0


def _context_summary(
    values: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    acceptable: np.ndarray,
    base_strict_rate: float,
    *,
    min_count: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in sorted({str(x) for x in values.tolist()}):
        mask = values == key
        count = int(np.sum(mask))
        if count == 0:
            continue
        strict_rate = float(np.mean(y[mask]))
        avg_prob = float(np.mean(p[mask]))
        acceptable_rate = float(np.mean(acceptable[mask]))
        lift = (strict_rate / base_strict_rate) if base_strict_rate > 0 else None
        result[key] = {
            'count': count,
            'strict_rate': strict_rate,
            'avg_prob': avg_prob,
            'acceptable_rate': acceptable_rate,
            'lift_vs_base': lift,
            'supported': bool(count >= min_count),
        }
    return result


def summarize_setup_family_profiles(
    families: Any,
    y_strict: Any,
    probs: Any,
    acceptable: Any,
    *,
    sectors: Any = None,
    ttc_frac: Any = None,
    market_context: Any = None,
    min_count: int = 300,
    suppress_below_lift: float = 0.90,
    promote_above_lift: float = 1.15,
    threshold_loosen_mult: float = 0.92,
    threshold_tighten_mult: float = 1.10,
    bin_count: int = 5,
    context_min_count: int = 180,
    context_min_lift_delta: float = 0.15,
) -> Dict[str, Any]:
    fam = np.asarray(families, dtype=object).reshape(-1)
    y = np.asarray(y_strict, dtype=int).reshape(-1)
    p = np.asarray(probs, dtype=float).reshape(-1)
    acc = np.asarray(acceptable, dtype=bool).reshape(-1)
    sector_groups = sector_groups_from_array(sectors) if sectors is not None else np.array(['OTHER'] * fam.size, dtype=object)
    tod = time_bucket_from_ttc(ttc_frac) if ttc_frac is not None else np.array(['MID'] * fam.size, dtype=object)
    mkt = np.asarray(market_context, dtype=object).reshape(-1) if market_context is not None else np.array(['NEUTRAL'] * fam.size, dtype=object)
    base_rate = float(np.mean(y)) if y.size else 0.0
    prior_strength = max(60.0, float(min_count) * 0.60)
    edges = _quantile_edges(p, bin_count)
    global_bins = _compute_global_bin_stats(p, y, edges)
    global_by_bin = {int(x['bin']): x for x in global_bins}

    profiles: Dict[str, Any] = {
        'base_strict_rate': base_rate,
        'min_count': int(min_count),
        'context_min_count': int(context_min_count),
        'context_min_lift_delta': float(context_min_lift_delta),
        'suppress_below_lift': float(suppress_below_lift),
        'promote_above_lift': float(promote_above_lift),
        'bin_edges': edges.astype(float).tolist(),
        'profile_source': 'validation',
    }
    for name in FAMILIES:
        mask = fam == name
        count = int(np.sum(mask))
        if count == 0:
            profiles[name] = {
                'count': 0,
                'strict_rate': None,
                'avg_prob': None,
                'acceptable_rate': None,
                'lift_vs_base': None,
                'bonus': 0.0,
                'signal_policy': 'NO_DATA',
                'threshold_multiplier': 1.0,
                'calibration_bins': [],
                'context_profiles': {},
            }
            continue
        strict_rate = float(np.mean(y[mask]))
        avg_prob = float(np.mean(p[mask]))
        acceptable_rate = float(np.mean(acc[mask]))
        lift = (strict_rate / base_rate) if base_rate > 0 else None
        policy = _policy_level_from_metrics(
            count,
            lift,
            acceptable_rate,
            min_count=min_count,
            suppress_below_lift=suppress_below_lift,
            promote_above_lift=promote_above_lift,
        )
        threshold_multiplier = _threshold_multiplier_for_policy(policy, loosen_mult=threshold_loosen_mult, tighten_mult=threshold_tighten_mult)

        family_bins: List[Dict[str, Any]] = []
        bin_indices = np.array([_bin_index(float(x), edges) for x in p[mask]], dtype=int)
        for bi in range(edges.size - 1):
            bmask = bin_indices == bi
            bcount = int(np.sum(bmask))
            global_bin = global_by_bin.get(int(bi), {'strict_rate': base_rate, 'avg_prob': 0.0, 'count': 0})
            if bcount == 0:
                family_bins.append({
                    'bin': int(bi),
                    'lo': float(edges[bi]),
                    'hi': float(edges[bi + 1]),
                    'count': 0,
                    'avg_prob': None,
                    'strict_rate': None,
                    'global_strict_rate': float(global_bin.get('strict_rate') or base_rate),
                    'multiplier': 1.0,
                })
                continue
            bprob = float(np.mean(p[mask][bmask]))
            brate = float(np.mean(y[mask][bmask]))
            mult = _conservative_bin_multiplier(
                bprob,
                brate,
                bcount,
                float(global_bin.get('strict_rate') or base_rate),
                prior_strength=prior_strength,
            )
            family_bins.append({
                'bin': int(bi),
                'lo': float(edges[bi]),
                'hi': float(edges[bi + 1]),
                'count': int(bcount),
                'avg_prob': bprob,
                'strict_rate': brate,
                'global_strict_rate': float(global_bin.get('strict_rate') or base_rate),
                'multiplier': mult,
            })

        context_profiles: Dict[str, Any] = {}
        for dim_name, vals in [('market_context', mkt[mask]), ('time_bucket', tod[mask]), ('sector_group', sector_groups[mask])]:
            ctx = _context_summary(vals, y[mask], p[mask], acc[mask], base_rate, min_count=context_min_count)
            for key, node in ctx.items():
                if not node.get('supported'):
                    node['policy_delta'] = 0
                    continue
                ctx_lift = float(node.get('lift_vs_base') or 1.0)
                fam_lift = float(lift or 1.0)
                delta = 0
                if ctx_lift >= promote_above_lift and ctx_lift >= fam_lift + context_min_lift_delta and float(node.get('acceptable_rate') or 0.0) >= 0.58:
                    delta = 1
                elif ctx_lift <= suppress_below_lift and ctx_lift <= fam_lift - context_min_lift_delta:
                    delta = -1
                node['policy_delta'] = int(delta)
            context_profiles[dim_name] = ctx

        bonus = 0.0
        if lift is not None:
            bonus += 0.10 * max(-1.0, min(1.0, lift - 1.0))
        bonus += 0.05 * (acceptable_rate - 0.5)
        bonus += 0.05 * (avg_prob - (base_rate or 0.0))
        bonus = float(np.clip(bonus, -0.16, 0.16))
        profiles[name] = {
            'count': count,
            'strict_rate': strict_rate,
            'avg_prob': avg_prob,
            'acceptable_rate': acceptable_rate,
            'lift_vs_base': lift,
            'bonus': bonus,
            'signal_policy': policy,
            'threshold_multiplier': float(threshold_multiplier),
            'calibration_bins': family_bins,
            'context_profiles': context_profiles,
        }
    return profiles


def family_profile_for_live_row(profiles: Dict[str, Any], family: str) -> Dict[str, Any]:
    if not isinstance(profiles, dict):
        return {}
    node = profiles.get(str(family)) or {}
    return node if isinstance(node, dict) else {}


def _policy_to_index(policy: str) -> int:
    return POLICY_ORDER.index(policy) if policy in POLICY_ORDER else POLICY_ORDER.index('CANDIDATE_ONLY')


def _index_to_policy(idx: int) -> str:
    return POLICY_ORDER[max(0, min(int(idx), len(POLICY_ORDER) - 1))]


def _family_context_delta(item: Dict[str, Any], regime_state: str, time_bucket: str, sector_group: str) -> Tuple[int, List[str]]:
    context_profiles = item.get('context_profiles') if isinstance(item, dict) else None
    if not isinstance(context_profiles, dict):
        return 0, []
    deltas: List[int] = []
    reasons: List[str] = []
    for dim_name, key, label in [
        ('market_context', regime_state, 'CTX_REGIME'),
        ('time_bucket', time_bucket, 'CTX_TIME'),
        ('sector_group', sector_group, 'CTX_SECTOR'),
    ]:
        node = (context_profiles.get(dim_name) or {}).get(str(key)) if isinstance(context_profiles.get(dim_name), dict) else None
        if not isinstance(node, dict) or not bool(node.get('supported')):
            continue
        delta = int(node.get('policy_delta') or 0)
        if delta == 0:
            continue
        deltas.append(delta)
        reasons.append(label)
    if not deltas:
        return 0, []
    # Conservative combination: any tightening dominates; promotions only apply if no tightening is present.
    if any(d < 0 for d in deltas):
        return min(deltas), reasons
    return max(deltas), reasons


def _family_calibration_multiplier(item: Dict[str, Any], prob: float) -> float:
    bins = item.get('calibration_bins') if isinstance(item, dict) else None
    edges = []
    if not isinstance(bins, list) or not bins:
        return 1.0
    # derive edges from bin list ordering
    for i, node in enumerate(bins):
        if i == 0:
            edges.append(0.0)
        edges.append(float(node.get('hi', 1.0)) if 'hi' in node else 1.0)
    # fallback from stored bin index/multiplier
    idx = max(0, min(len(bins) - 1, int(np.floor(float(np.clip(prob, 0.0, 0.999)) * len(bins)))))
    for bi, node in enumerate(bins):
        lo = float(node.get('lo', bi / max(len(bins), 1))) if isinstance(node, dict) else 0.0
        hi = float(node.get('hi', (bi + 1) / max(len(bins), 1))) if isinstance(node, dict) else 1.0
        if (bi == len(bins) - 1 and prob <= hi) or (prob >= lo and prob < hi):
            idx = bi
            break
    node = bins[idx] if idx < len(bins) else {}
    try:
        return float(node.get('multiplier') or 1.0)
    except Exception:
        return 1.0


def family_live_policy_for_row(
    profiles: Dict[str, Any],
    family: str,
    prob: float,
    *,
    regime_state: str = 'NEUTRAL',
    time_bucket: str = 'MID',
    sector_group: str = 'OTHER',
) -> Dict[str, Any]:
    item = family_profile_for_live_row(profiles, family)
    base_policy = str(item.get('signal_policy') or 'CANDIDATE_ONLY')
    threshold_multiplier = float(item.get('threshold_multiplier') or 1.0)
    cal_mult = _family_calibration_multiplier(item, prob)
    ctx_delta, ctx_reasons = _family_context_delta(item, regime_state, time_bucket, sector_group)
    policy_idx = _policy_to_index(base_policy) + int(ctx_delta)
    effective_policy = _index_to_policy(policy_idx)
    if ctx_delta > 0:
        threshold_multiplier *= 0.97
    elif ctx_delta < 0:
        threshold_multiplier *= 1.04
    if effective_policy == 'FULL_ACTIONABLE':
        threshold_multiplier = min(threshold_multiplier, float(item.get('threshold_multiplier') or 1.0))
    reason_flags: List[str] = []
    if base_policy == 'FULL_ACTIONABLE' and effective_policy != 'FULL_ACTIONABLE':
        reason_flags.append('FAMILY_TIGHTEN')
    elif base_policy != 'FULL_ACTIONABLE' and effective_policy == 'FULL_ACTIONABLE':
        reason_flags.append('FAMILY_PROMOTE')
    if base_policy == 'WATCHLIST_ONLY':
        reason_flags.append('FAMILY_WATCHLIST_ONLY')
    elif base_policy == 'CANDIDATE_ONLY':
        reason_flags.append('FAMILY_CANDIDATE_ONLY')
    elif base_policy == 'SUPPRESS':
        reason_flags.append('FAMILY_SUPPRESS')
    if base_policy in {'WATCHLIST_ONLY', 'CANDIDATE_ONLY'} and int(item.get('count') or 0) < int((profiles.get('min_count') or 300)):
        reason_flags.append('FAMILY_LIMITED_DATA')
    if ctx_delta > 0:
        reason_flags.append('FAMILY_PROMOTE_CTX')
    elif ctx_delta < 0:
        reason_flags.append('FAMILY_TIGHTEN_CTX')
    return {
        'base_policy': base_policy,
        'effective_policy': effective_policy,
        'threshold_multiplier': float(threshold_multiplier),
        'calibration_multiplier': float(cal_mult),
        'actionable_allowed': effective_policy == 'FULL_ACTIONABLE',
        'candidate_allowed': effective_policy in {'FULL_ACTIONABLE', 'CANDIDATE_ONLY'},
        'watchlist_allowed': effective_policy in {'FULL_ACTIONABLE', 'CANDIDATE_ONLY', 'WATCHLIST_ONLY'},
        'suppress_signals': effective_policy == 'SUPPRESS',
        'reason_flags': reason_flags,
        'context_reasons': ctx_reasons,
        'bonus': float(item.get('bonus') or 0.0),
    }


def family_bonus_for_live_row(profiles: Dict[str, Any], family: str) -> float:
    item = family_profile_for_live_row(profiles, family)
    try:
        return float(item.get('bonus') or 0.0)
    except Exception:
        return 0.0


def family_calibration_multiplier_for_live_row(profiles: Dict[str, Any], family: str, prob: float = 0.0) -> float:
    item = family_profile_for_live_row(profiles, family)
    return _family_calibration_multiplier(item, prob)


def family_threshold_multiplier_for_live_row(profiles: Dict[str, Any], family: str) -> float:
    item = family_profile_for_live_row(profiles, family)
    try:
        return float(item.get('threshold_multiplier') or 1.0)
    except Exception:
        return 1.0


def family_signal_policy_for_live_row(profiles: Dict[str, Any], family: str) -> str:
    item = family_profile_for_live_row(profiles, family)
    try:
        return str(item.get('signal_policy') or 'CANDIDATE_ONLY')
    except Exception:
        return 'CANDIDATE_ONLY'


def family_suppressed_for_live_row(profiles: Dict[str, Any], family: str) -> bool:
    return family_signal_policy_for_live_row(profiles, family) == 'SUPPRESS'


def family_actionable_allowed_for_live_row(profiles: Dict[str, Any], family: str) -> bool:
    return family_signal_policy_for_live_row(profiles, family) == 'FULL_ACTIONABLE'


def apply_family_policy_profiles(
    probs: Any,
    acceptable: Any,
    families: Any,
    adaptive_threshold: float,
    profiles: Dict[str, Any],
    *,
    regime_states: Any = None,
    time_buckets: Any = None,
    sector_groups: Any = None,
    watchlist_frac: float = 0.90,
) -> Dict[str, Any]:
    p = np.asarray(probs, dtype=float).reshape(-1)
    acc = np.asarray(acceptable, dtype=bool).reshape(-1)
    fam = np.asarray(families, dtype=object).reshape(-1)
    regime = np.asarray(regime_states if regime_states is not None else ['NEUTRAL'] * p.size, dtype=object).reshape(-1)
    tod = np.asarray(time_buckets if time_buckets is not None else ['MID'] * p.size, dtype=object).reshape(-1)
    sgrp = np.asarray(sector_groups if sector_groups is not None else ['OTHER'] * p.size, dtype=object).reshape(-1)

    adj_prob = np.zeros_like(p, dtype=float)
    thresholds = np.full_like(p, float(max(0.03, adaptive_threshold)), dtype=float)
    policy_levels: List[str] = []
    score = np.zeros_like(p, dtype=float)
    for i in range(p.size):
        pol = family_live_policy_for_row(profiles, str(fam[i]), float(p[i]), regime_state=str(regime[i]), time_bucket=str(tod[i]), sector_group=str(sgrp[i]))
        q = float(np.clip(float(p[i]) * float(pol['calibration_multiplier']), 0.0, 0.999))
        thr = float(max(0.03, adaptive_threshold * float(pol['threshold_multiplier'])))
        adj_prob[i] = q
        thresholds[i] = thr
        policy_levels.append(str(pol['effective_policy']))
        if pol['effective_policy'] == 'SUPPRESS':
            score[i] = 0.0
        elif pol['effective_policy'] == 'WATCHLIST_ONLY':
            score[i] = q * 0.60
        elif pol['effective_policy'] == 'CANDIDATE_ONLY':
            score[i] = q * 0.90
        else:
            score[i] = q

    policy_levels_arr = np.asarray(policy_levels, dtype=object)
    actionable_mask = (policy_levels_arr == 'FULL_ACTIONABLE') & acc & (adj_prob >= thresholds)
    candidate_mask = np.isin(policy_levels_arr, ['FULL_ACTIONABLE', 'CANDIDATE_ONLY']) & acc & (adj_prob >= thresholds)
    watchlist_mask = np.isin(policy_levels_arr, ['FULL_ACTIONABLE', 'CANDIDATE_ONLY', 'WATCHLIST_ONLY']) & acc & (adj_prob >= np.maximum(0.03, thresholds * float(watchlist_frac))) & (~candidate_mask)
    suppress_mask = policy_levels_arr == 'SUPPRESS'
    return {
        'adjusted_prob': adj_prob,
        'thresholds': thresholds,
        'policy_levels': policy_levels_arr,
        'actionable_mask': actionable_mask,
        'candidate_mask': candidate_mask,
        'watchlist_mask': watchlist_mask,
        'suppress_mask': suppress_mask,
        'score_for_rank': score,
        'policy_counts': {lvl: int(np.sum(policy_levels_arr == lvl)) for lvl in POLICY_ORDER},
    }


def compute_post_policy_metrics(
    probs: Any,
    y_strict: Any,
    acceptable: Any,
    families: Any,
    adaptive_threshold: float,
    profiles: Dict[str, Any],
    *,
    regime_states: Any = None,
    time_buckets: Any = None,
    sector_groups: Any = None,
    watchlist_frac: float = 0.90,
) -> Dict[str, Any]:
    y = np.asarray(y_strict, dtype=int).reshape(-1)
    acc = np.asarray(acceptable, dtype=bool).reshape(-1)
    applied = apply_family_policy_profiles(
        probs,
        acc,
        families,
        adaptive_threshold,
        profiles,
        regime_states=regime_states,
        time_buckets=time_buckets,
        sector_groups=sector_groups,
        watchlist_frac=watchlist_frac,
    )
    topk = compute_topk_metrics(applied['score_for_rank'], y, acc)
    return {
        'base_strict_rate': float(np.mean(y)) if y.size else None,
        'candidate_count': int(np.sum(applied['candidate_mask'])),
        'actionable_count': int(np.sum(applied['actionable_mask'])),
        'watchlist_count': int(np.sum(applied['watchlist_mask'])),
        'suppressed_count': int(np.sum(applied['suppress_mask'])),
        'policy_counts': applied['policy_counts'],
        'topk': topk,
    }
