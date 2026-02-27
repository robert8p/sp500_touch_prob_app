from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    if span <= 1 or arr.size == 0:
        return arr.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if close.size < 2:
        return np.zeros_like(close, dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Wilder's smoothing
    out = np.zeros_like(close, dtype=float)
    if tr.size == 0:
        return out
    out[0] = tr[0]
    for i in range(1, tr.size):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    if n < period + 2:
        return np.zeros_like(close, dtype=float)
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    # Wilder smoothing
    tr_s = np.zeros_like(tr, dtype=float)
    p_s = np.zeros_like(tr, dtype=float)
    m_s = np.zeros_like(tr, dtype=float)
    tr_s[0], p_s[0], m_s[0] = tr[0], plus_dm[0], minus_dm[0]
    for i in range(1, tr.size):
        tr_s[i] = tr_s[i - 1] - (tr_s[i - 1] / period) + tr[i]
        p_s[i] = p_s[i - 1] - (p_s[i - 1] / period) + plus_dm[i]
        m_s[i] = m_s[i - 1] - (m_s[i - 1] / period) + minus_dm[i]

    plus_di = 100.0 * (p_s / np.where(tr_s == 0, 1.0, tr_s))
    minus_di = 100.0 * (m_s / np.where(tr_s == 0, 1.0, tr_s))
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1.0, (plus_di + minus_di))

    adx_arr = np.zeros(n, dtype=float)
    # align dx (length n-1) into adx_arr[1:]
    adx_sm = np.zeros_like(dx, dtype=float)
    adx_sm[0] = dx[0]
    for i in range(1, dx.size):
        adx_sm[i] = (adx_sm[i - 1] * (period - 1) + dx[i]) / period
    adx_arr[1:] = adx_sm
    return adx_arr

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    out = np.zeros_like(close, dtype=float)
    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
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
    if denom == 0.0:
        return 0.0
    return float((xs * (ys - ys.mean())).sum() / denom)

@dataclass
class FeatureRow:
    price: float
    vwap: float
    features: np.ndarray
    reasons: str

FEATURE_NAMES = [
    "ret_5m",
    "ret_30m",
    "ema_fast_minus_slow",
    "adx",
    "atr_pct",
    "rv_1h",
    "rvol_rolling",
    "obv_slope",
    "vwap_loc",
    "donch_dist",
    "spy_ret_30m",
    "mins_to_close",
]

def compute_features_from_5m(
    bars_5m: List[dict],
    spy_ret_30m: float,
    mins_to_close: float,
) -> Optional[FeatureRow]:
    if not bars_5m or len(bars_5m) < 8:
        return None

    c = np.array([_to_float(b.get("c")) for b in bars_5m], dtype=float)
    h = np.array([_to_float(b.get("h")) for b in bars_5m], dtype=float)
    l = np.array([_to_float(b.get("l")) for b in bars_5m], dtype=float)
    v = np.array([_to_float(b.get("v")) for b in bars_5m], dtype=float)
    vw = np.array([_to_float(b.get("vw")) for b in bars_5m], dtype=float)

    price = float(c[-1])
    vwap = float(vw[-1]) if vw[-1] > 0 else float((c * np.maximum(v, 1.0)).sum() / np.maximum(1.0, v.sum()))

    ret_5m = float(price / c[-2] - 1.0) if c[-2] != 0 else 0.0
    ret_30m = float(price / c[-7] - 1.0) if c.size >= 7 and c[-7] != 0 else 0.0

    ema_fast = ema(c, span=6)
    ema_slow = ema(c, span=18)
    ema_diff = float(ema_fast[-1] - ema_slow[-1])

    adx_arr = adx(h, l, c, period=14)
    adx_v = float(adx_arr[-1])

    atr_arr = atr(h, l, c, period=14)
    atr_v = float(atr_arr[-1])
    atr_pct = float(atr_v / price) if price > 0 else 0.0

    # Realized vol over last 12 5m bars (~1h)
    rets = np.diff(np.log(np.maximum(c, 1e-9)))
    if rets.size >= 12:
        recent = rets[-12:]
    else:
        recent = rets
    rv_1h = float(np.std(recent) * np.sqrt(max(1.0, float(recent.size)))) if recent.size > 2 else 0.0

    rvol = 0.0
    if v.size >= 20:
        med = float(np.median(v[-20:]))
        rvol = float(v[-1] / med) if med > 0 else 0.0
    else:
        med = float(np.median(v)) if v.size > 0 else 0.0
        rvol = float(v[-1] / med) if med > 0 else 0.0

    obv_series = obv(c, v)
    obv_sl = slope_last(obv_series, window=12)

    vwap_loc = float((price - vwap) / atr_v) if atr_v > 0 else 0.0

    look = min(20, h.size)
    donch_hi = float(np.max(h[-look:])) if look > 0 else float(h[-1])
    donch_dist = float((donch_hi - price) / atr_v) if atr_v > 0 else 0.0

    feats = np.array([
        ret_5m,
        ret_30m,
        ema_diff,
        adx_v,
        atr_pct,
        rv_1h,
        rvol,
        obv_sl,
        vwap_loc,
        donch_dist,
        float(spy_ret_30m),
        float(mins_to_close),
    ], dtype=float)

    reasons = reason_codes(ret_30m, ema_diff, adx_v, rvol, vwap_loc, donch_dist)

    return FeatureRow(price=price, vwap=vwap, features=feats, reasons=reasons)

def reason_codes(ret_30m: float, ema_diff: float, adx_v: float, rvol: float, vwap_loc: float, donch_dist: float) -> str:
    codes: List[str] = []
    if ret_30m > 0.006:
        codes.append("MOM30+")
    if ret_30m < -0.006:
        codes.append("MOM30-")
    if ema_diff > 0 and adx_v > 18:
        codes.append("TREND+")
    if ema_diff < 0 and adx_v > 18:
        codes.append("TREND-")
    if vwap_loc > 0.3:
        codes.append(">VWAP")
    if vwap_loc < -0.3:
        codes.append("<VWAP")
    if rvol > 1.4:
        codes.append("RVOL")
    if donch_dist < 0.6:
        codes.append("NEARHIGH")
    return " ".join(codes[:4])
