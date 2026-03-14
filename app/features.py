from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


def _f(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if close.size < 2:
        return np.zeros_like(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.zeros_like(close, dtype=float)
    out[0] = tr[0]
    for i in range(1, tr.size):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    if n < period + 2:
        return np.zeros_like(close, dtype=float)
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    tr_s = np.zeros_like(tr)
    p_s = np.zeros_like(tr)
    m_s = np.zeros_like(tr)
    tr_s[0] = tr[0]
    p_s[0] = plus_dm[0]
    m_s[0] = minus_dm[0]
    for i in range(1, tr.size):
        tr_s[i] = tr_s[i - 1] - tr_s[i - 1] / period + tr[i]
        p_s[i] = p_s[i - 1] - p_s[i - 1] / period + plus_dm[i]
        m_s[i] = m_s[i - 1] - m_s[i - 1] / period + minus_dm[i]
    plus_di = 100.0 * (p_s / np.where(tr_s == 0, 1.0, tr_s))
    minus_di = 100.0 * (m_s / np.where(tr_s == 0, 1.0, tr_s))
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1.0, (plus_di + minus_di))
    adx_sm = np.zeros_like(dx)
    adx_sm[0] = dx[0]
    for i in range(1, dx.size):
        adx_sm[i] = (adx_sm[i - 1] * (period - 1) + dx[i]) / period
    out = np.zeros(n, dtype=float)
    out[1:] = adx_sm
    return out


def obv(close: np.ndarray, vol: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
    return out


def slope_last(y: np.ndarray, window: int = 12) -> float:
    if y.size < 3:
        return 0.0
    w = min(window, y.size)
    ys = y[-w:]
    xs = np.arange(w, dtype=float)
    xs = xs - xs.mean()
    denom = float((xs * xs).sum())
    if denom == 0:
        return 0.0
    return float((xs * (ys - ys.mean())).sum() / denom)


@dataclass
class FeatureRow:
    price: float
    vwap: float
    features: np.ndarray
    risk: str
    risk_reasons: str
    reasons: str
    used_tod_profile: bool


FEATURE_NAMES = [
    'ret_5m', 'ret_30m', 'rel_strength_spy_30m', 'rel_strength_sector_30m',
    'gap_prev_close_pct', 'dist_prev_high_atr', 'dist_prev_low_atr', 'prev_day_range_pct',
    'ret_5d', 'ret_20d', 'ret_60d', 'dist_20dma_pct', 'dist_50dma_pct', 'dist_200dma_pct', 'drawdown_20d_pct', 'drawdown_60d_pct',
    'ema_diff_pct', 'adx', 'atr_pct', 'rv_1h', 'rvol_tod', 'obv_slope_norm', 'vwap_loc', 'donch_dist',
    'ret_since_open_pct', 'dist_open_atr', 'dist_orh_atr', 'dist_orl_atr', 'bars_below_vwap_frac', 'no_reclaim_vwap', 'downside_impulse_30m', 'damage_from_high_atr',
    'path_smoothness_30m', 'reversal_count_30m', 'or_breakout_state',
    'ttc_frac', 'log_mins_to_close', 'tod_frac', 'dvol_usd_med_log', 'range_pct_med', 'wick_atr_med',
    'int_mom_rvol', 'int_mom_vwap', 'int_trend_adx', 'int_mom_ttc',
]

DIRECTIONAL_REASON_CAP = 12


def _risk_and_micro(o, h, l, c, v, atr_arr, rolling_bars, dvol_min_usd, range_pct_max, wick_atr_max):
    n = c.size
    if n < 2:
        return 'OK', [], 0.0, 0.0, 0.0
    w = min(rolling_bars, n - 1)
    o_w = o[-(w + 1):-1]
    h_w = h[-(w + 1):-1]
    l_w = l[-(w + 1):-1]
    c_w = c[-(w + 1):-1]
    v_w = v[-(w + 1):-1]
    atr_w = atr_arr[-(w + 1):-1]
    dvol_series = c_w * np.maximum(v_w, 0.0)
    dvol_med = float(np.median(dvol_series)) if dvol_series.size else 0.0
    rng_series = np.where(c_w > 0, (h_w - l_w) / c_w, 0.0)
    rng_med = float(np.median(rng_series)) if rng_series.size else 0.0
    rng_cur = float((h[-1] - l[-1]) / c[-1]) if c[-1] else 0.0
    upper = h_w - np.maximum(o_w, c_w)
    lower = np.minimum(o_w, c_w) - l_w
    wick = np.maximum(upper, lower)
    wick_norm = np.where(atr_w > 0, wick / atr_w, 0.0)
    wick_med = float(np.median(wick_norm)) if wick_norm.size else 0.0
    atr_cur = float(atr_arr[-1]) if atr_arr.size else 0.0
    upper_c = float(h[-1] - max(o[-1], c[-1]))
    lower_c = float(min(o[-1], c[-1]) - l[-1])
    wick_cur = max(upper_c, lower_c)
    wick_cur_n = float(wick_cur / atr_cur) if atr_cur > 0 else 0.0
    cond_dvol = dvol_med > 0 and dvol_med < dvol_min_usd
    cond_range = (rng_med > range_pct_max) or (rng_cur > range_pct_max)
    cond_wick = (wick_med > wick_atr_max) or (wick_cur_n > wick_atr_max)
    codes = []
    if cond_dvol:
        codes.append('LOW_LIQ')
    if cond_wick:
        codes.append('WICKY')
    if cond_range:
        codes.append('WIDE_RANGE')
    k = int(cond_dvol) + int(cond_range) + int(cond_wick)
    risk = 'HIGH' if k >= 2 else ('CAUTION' if k == 1 else 'OK')
    return risk, codes, dvol_med, rng_med, wick_med


def _directional_reasons(ret30, rel_spy, rel_sector, ema_pct, adx_v, rvol_tod, vwap_loc, donch, gap_prev_close, daily_ctx, ret_since_open, smoothness, reversal_count, or_state):
    codes = []
    if ret30 > 0.006:
        codes.append('MOM30+')
    if ret30 < -0.006:
        codes.append('MOM30-')
    if rel_spy > 0.003:
        codes.append('RS+')
    if rel_spy < -0.003:
        codes.append('RS-')
    if rel_sector > 0.003:
        codes.append('SECTOR+')
    if rel_sector < -0.003:
        codes.append('SECTOR-')
    if gap_prev_close > 0.003:
        codes.append('GAP+')
    if gap_prev_close < -0.003:
        codes.append('GAP-')
    if daily_ctx.get('ret_20d', 0.0) > 0.05:
        codes.append('UP20D')
    if daily_ctx.get('ret_20d', 0.0) < -0.08:
        codes.append('DN20D')
    if ema_pct > 0 and adx_v > 18:
        codes.append('TREND+')
    if ema_pct < 0 and adx_v > 18:
        codes.append('TREND-')
    if vwap_loc > 0.3:
        codes.append('>VWAP')
    if vwap_loc < -0.3:
        codes.append('<VWAP')
    if rvol_tod > 1.4:
        codes.append('RVOL')
    if donch < 0.6:
        codes.append('NEARHIGH')
    if ret_since_open > 0.01:
        codes.append('OPEN+')
    if ret_since_open < -0.02:
        codes.append('OPEN-')
    if smoothness > 0.5:
        codes.append('SMOOTH')
    if reversal_count >= 3:
        codes.append('CHOPPY')
    if or_state > 0.5:
        codes.append('ORB+')
    if or_state < -0.5:
        codes.append('ORB-')
    return ' '.join(codes[:DIRECTIONAL_REASON_CAP])


def compute_features_from_5m(
    bars_5m: List[dict],
    spy_ret_30m: float,
    sector_ret_30m: float,
    mins_to_close: float,
    tod_baseline_vol_median: Optional[float],
    rolling_rvol_window: int,
    risk_params: Tuple[int, float, float, float],
    prev_day_close: Optional[float],
    prev_day_high: Optional[float],
    prev_day_low: Optional[float],
    daily_ctx: Optional[Dict[str, float]],
    tz_name: str,
    blocked_params: Dict[str, float],
) -> Optional[FeatureRow]:
    if not bars_5m or len(bars_5m) < 7:
        return None
    daily_ctx = daily_ctx or {}
    o = np.array([_f(b.get('o')) for b in bars_5m], dtype=float)
    h = np.array([_f(b.get('h')) for b in bars_5m], dtype=float)
    l = np.array([_f(b.get('l')) for b in bars_5m], dtype=float)
    c = np.array([_f(b.get('c')) for b in bars_5m], dtype=float)
    v = np.array([_f(b.get('v')) for b in bars_5m], dtype=float)
    vw = np.array([_f(b.get('vw')) for b in bars_5m], dtype=float)
    price = float(c[-1])
    if price <= 0:
        return None
    vwap = float(vw[-1]) if vw[-1] > 0 else float((c * np.maximum(v, 1.0)).sum() / max(1.0, float(v.sum())))
    if not (vwap > 0):
        return None
    ret5 = float(price / c[-2] - 1.0) if c[-2] else 0.0
    ret30 = float(price / c[-7] - 1.0) if c[-7] else 0.0
    rel_spy = float(ret30 - float(spy_ret_30m))
    rel_sector = float(ret30 - float(sector_ret_30m))
    ema_fast = ema(c, span=6)
    ema_slow = ema(c, span=18)
    ema_diff_pct = float((ema_fast[-1] - ema_slow[-1]) / price)
    adx_v = float(adx(h, l, c, period=14)[-1])
    atr_arr = atr(h, l, c, period=14)
    atr_v = float(atr_arr[-1])
    atr_pct = float(atr_v / price) if atr_v > 0 else 0.0
    logrets = np.diff(np.log(np.maximum(c, 1e-9)))
    seg = logrets[-12:] if logrets.size >= 12 else logrets
    rv_1h = float(np.std(seg) * math.sqrt(max(1.0, float(seg.size)))) if seg.size > 2 else 0.0
    current_vol = float(v[-1])
    prev_v = v[:-1]
    med_prev = float(np.median(prev_v[-rolling_rvol_window:])) if prev_v.size > 0 else 0.0
    fallback_rvol = (current_vol / med_prev) if med_prev > 0 else 1.0
    used_profile = False
    if tod_baseline_vol_median is not None and tod_baseline_vol_median > 0:
        rvol_tod = float(current_vol / tod_baseline_vol_median)
        used_profile = True
    else:
        rvol_tod = float(fallback_rvol)
    obv_series = obv(c, v)
    obv_sl = slope_last(obv_series, window=12)
    obv_slope_norm = float(obv_sl / max(1.0, med_prev))
    vwap_loc = float((price - vwap) / atr_v) if atr_v > 0 else 0.0
    look = min(20, h.size)
    hi = float(np.max(h[-look:])) if look > 0 else float(h[-1])
    donch = float((hi - price) / atr_v) if atr_v > 0 else 0.0

    ts_dates = []
    for b in bars_5m:
        ts = str(b.get('t') or '')
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        try:
            ts_dates.append(np.datetime64(np.datetime64(ts).astype('datetime64[D]')))
        except Exception:
            ts_dates.append(None)
    cur_day = next((d for d in reversed(ts_dates) if d is not None), None)
    cur_mask = np.array([(d == cur_day) if (d is not None and cur_day is not None) else True for d in ts_dates], dtype=bool)
    cur_o = o[cur_mask]
    cur_h = h[cur_mask]
    cur_l = l[cur_mask]
    cur_c = c[cur_mask]
    cur_vw = np.where(vw[cur_mask] > 0, vw[cur_mask], cur_c)
    open_price = float(cur_o[0]) if cur_o.size else price
    ret_since_open = float(price / open_price - 1.0) if open_price > 0 else 0.0
    dist_open_atr = float((price - open_price) / atr_v) if atr_v > 0 else 0.0
    or_n = min(6, cur_h.size) if cur_h.size else 0
    orh = float(np.max(cur_h[:or_n])) if or_n else price
    orl = float(np.min(cur_l[:or_n])) if or_n else price
    dist_orh_atr = float((orh - price) / atr_v) if atr_v > 0 else 0.0
    dist_orl_atr = float((price - orl) / atr_v) if atr_v > 0 else 0.0
    bars_below_vwap_frac = float(np.mean(cur_c < cur_vw)) if cur_c.size else 0.0
    no_reclaim_vwap = 1.0 if (cur_c.size >= 4 and np.all(cur_c[-4:] < cur_vw[-4:])) else 0.0
    downside_impulse_30m = float(min(0.0, ret30))
    session_high = float(np.max(cur_h)) if cur_h.size else price
    damage_from_high_atr = float((session_high - price) / atr_v) if atr_v > 0 else 0.0

    ret_seq = np.diff(cur_c[-7:]) / np.maximum(cur_c[-7:-1], 1e-9) if cur_c.size >= 7 else np.diff(cur_c) / np.maximum(cur_c[:-1], 1e-9)
    abs_sum = float(np.sum(np.abs(ret_seq))) if ret_seq.size else 0.0
    path_smoothness = float(ret30 / (abs_sum + 1e-9)) if abs_sum > 0 else 0.0
    if ret_seq.size >= 2:
        signs = np.sign(ret_seq)
        reversal_count = float(np.sum((signs[1:] * signs[:-1]) < 0))
    else:
        reversal_count = 0.0
    or_breakout_state = 1.0 if price > orh else (-1.0 if price < orl else 0.0)

    ttc_frac = float(np.clip(mins_to_close / 390.0, 0.0, 1.0))
    log_mins = float(math.log1p(max(0.0, mins_to_close)))
    tod_frac = float(np.clip((390.0 - mins_to_close) / 390.0, 0.0, 1.0))
    gap_prev_close_pct = float(price / prev_day_close - 1.0) if prev_day_close and prev_day_close > 0 else 0.0
    dist_prev_high_atr = float((prev_day_high - price) / atr_v) if prev_day_high and atr_v > 0 else 0.0
    dist_prev_low_atr = float((price - prev_day_low) / atr_v) if prev_day_low and atr_v > 0 else 0.0
    prev_day_range_pct = float((prev_day_high - prev_day_low) / prev_day_close) if prev_day_close and prev_day_high and prev_day_low and prev_day_close > 0 else 0.0

    ret_5d = float(daily_ctx.get('ret_5d', 0.0))
    ret_20d = float(daily_ctx.get('ret_20d', 0.0))
    ret_60d = float(daily_ctx.get('ret_60d', 0.0))
    dist_20dma_pct = float(daily_ctx.get('dist_20dma_pct', 0.0))
    dist_50dma_pct = float(daily_ctx.get('dist_50dma_pct', 0.0))
    dist_200dma_pct = float(daily_ctx.get('dist_200dma_pct', 0.0))
    drawdown_20d_pct = float(daily_ctx.get('drawdown_20d_pct', 0.0))
    drawdown_60d_pct = float(daily_ctx.get('drawdown_60d_pct', 0.0))

    liq_rolling_bars, liq_dvol_min_usd, liq_range_pct_max, liq_wick_atr_max = risk_params
    risk, risk_codes, dvol_med, rng_med, wick_med = _risk_and_micro(o, h, l, c, v, atr_arr, liq_rolling_bars, liq_dvol_min_usd, liq_range_pct_max, liq_wick_atr_max)
    dvol_usd_med_log = float(math.log1p(max(0.0, dvol_med)))
    range_pct_med = float(max(0.0, rng_med))
    wick_atr_med = float(max(0.0, wick_med))

    lt_down = (
        (ret_20d <= blocked_params.get('ret20d_max', -0.08) and dist_50dma_pct <= blocked_params.get('dist50dma_max', -0.06))
        or (ret_60d <= blocked_params.get('ret60d_max', -0.15) and dist_200dma_pct < -0.08)
        or (drawdown_60d_pct <= -0.22 and ret_20d < -0.04)
    )
    open_weak = ret_since_open <= blocked_params.get('ret_since_open_max', -0.025) or dist_open_atr <= -2.0
    no_reclaim = bars_below_vwap_frac >= blocked_params.get('below_vwap_frac_min', 0.85) and no_reclaim_vwap > 0.5 and vwap_loc < -0.6
    damage = damage_from_high_atr >= blocked_params.get('damage_from_high_atr_min', 2.5)
    gap_down = gap_prev_close_pct <= -0.05
    event_risk = (abs(gap_prev_close_pct) >= blocked_params.get('event_gap_abs_min', 0.08)) and ((rvol_tod >= blocked_params.get('event_rvol_min', 2.2)) or (range_pct_med >= blocked_params.get('event_range_pct_min', 0.035)))
    broken_or = (or_breakout_state < -0.5) and open_weak and no_reclaim
    blocked_hits = int(lt_down) + int(open_weak) + int(no_reclaim) + int(damage) + int(broken_or)
    if lt_down:
        risk_codes.append('LT_DOWNTREND')
    if open_weak:
        risk_codes.append('OPEN_WEAK')
    if no_reclaim:
        risk_codes.append('NO_RECLAIM')
    if damage:
        risk_codes.append('INTRADAY_DAMAGE')
    if gap_down:
        risk_codes.append('GAP_DOWN')
    if broken_or:
        risk_codes.append('OR_FAIL')
    if event_risk:
        risk_codes.append('EVENT_RISK')
    if (blocked_hits >= 3) or (lt_down and open_weak and (no_reclaim or damage)) or (gap_down and open_weak and no_reclaim) or (event_risk and (open_weak or no_reclaim or damage)):
        risk = 'BLOCKED'
        if 'FALLING_KNIFE' not in risk_codes:
            risk_codes.insert(0, 'FALLING_KNIFE')

    int_mom_rvol = float(ret30 * (rvol_tod - 1.0))
    int_mom_vwap = float(ret30 * vwap_loc)
    int_trend_adx = float(ema_diff_pct * adx_v)
    int_mom_ttc = float(ret30 * ttc_frac)
    feats = np.array([
        ret5, ret30, rel_spy, rel_sector,
        gap_prev_close_pct, dist_prev_high_atr, dist_prev_low_atr, prev_day_range_pct,
        ret_5d, ret_20d, ret_60d, dist_20dma_pct, dist_50dma_pct, dist_200dma_pct, drawdown_20d_pct, drawdown_60d_pct,
        ema_diff_pct, adx_v, atr_pct, rv_1h, rvol_tod, obv_slope_norm, vwap_loc, donch,
        ret_since_open, dist_open_atr, dist_orh_atr, dist_orl_atr, bars_below_vwap_frac, no_reclaim_vwap, downside_impulse_30m, damage_from_high_atr,
        path_smoothness, reversal_count, or_breakout_state,
        ttc_frac, log_mins, tod_frac, dvol_usd_med_log, range_pct_med, wick_atr_med,
        int_mom_rvol, int_mom_vwap, int_trend_adx, int_mom_ttc,
    ], dtype=float)
    reasons = _directional_reasons(ret30, rel_spy, rel_sector, ema_diff_pct, adx_v, rvol_tod, vwap_loc, donch, gap_prev_close_pct, daily_ctx, ret_since_open, path_smoothness, reversal_count, or_breakout_state)
    risk_reasons = ' '.join(dict.fromkeys(risk_codes))
    return FeatureRow(price=price, vwap=vwap, features=feats, risk=risk, risk_reasons=risk_reasons, reasons=reasons, used_tod_profile=used_profile)
