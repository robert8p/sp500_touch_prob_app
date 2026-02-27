from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import math

def _f(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    alpha = 2.0/(span+1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha*arr[i] + (1-alpha)*out[i-1]
    return out

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=14) -> np.ndarray:
    if close.size < 2:
        return np.zeros_like(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high-low, np.maximum(np.abs(high-prev), np.abs(low-prev)))
    out = np.zeros_like(close, dtype=float)
    out[0] = tr[0]
    for i in range(1, tr.size):
        out[i] = (out[i-1]*(period-1)+tr[i])/period
    return out

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int=14) -> np.ndarray:
    n = close.size
    if n < period+2:
        return np.zeros_like(close, dtype=float)
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm= np.where((dn>up) & (dn>0), dn, 0.0)
    prev = close[:-1]
    tr = np.maximum(high[1:]-low[1:], np.maximum(np.abs(high[1:]-prev), np.abs(low[1:]-prev)))
    tr_s = np.zeros_like(tr); p_s=np.zeros_like(tr); m_s=np.zeros_like(tr)
    tr_s[0]=tr[0]; p_s[0]=plus_dm[0]; m_s[0]=minus_dm[0]
    for i in range(1, tr.size):
        tr_s[i]=tr_s[i-1]-tr_s[i-1]/period + tr[i]
        p_s[i]=p_s[i-1]-p_s[i-1]/period + plus_dm[i]
        m_s[i]=m_s[i-1]-m_s[i-1]/period + minus_dm[i]
    plus_di = 100.0 * (p_s / np.where(tr_s==0, 1.0, tr_s))
    minus_di= 100.0 * (m_s / np.where(tr_s==0, 1.0, tr_s))
    dx = 100.0 * np.abs(plus_di-minus_di) / np.where((plus_di+minus_di)==0, 1.0, (plus_di+minus_di))
    adx_sm = np.zeros_like(dx)
    adx_sm[0]=dx[0]
    for i in range(1, dx.size):
        adx_sm[i]=(adx_sm[i-1]*(period-1)+dx[i])/period
    out = np.zeros(n, dtype=float)
    out[1:] = adx_sm
    return out

def obv(close: np.ndarray, vol: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    for i in range(1, close.size):
        if close[i] > close[i-1]:
            out[i]=out[i-1]+vol[i]
        elif close[i] < close[i-1]:
            out[i]=out[i-1]-vol[i]
        else:
            out[i]=out[i-1]
    return out

def slope_last(y: np.ndarray, window: int=12) -> float:
    if y.size < 3:
        return 0.0
    w=min(window, y.size)
    ys=y[-w:]
    xs=np.arange(w, dtype=float)
    xs=xs-xs.mean()
    denom=float((xs*xs).sum())
    if denom==0:
        return 0.0
    return float((xs*(ys-ys.mean())).sum()/denom)

@dataclass
class FeatureRow:
    price: float
    vwap: float
    features: np.ndarray
    reasons: str

FEATURE_NAMES = [
    "ret_5m",
    "ret_30m",
    "rel_strength_30m",
    "ema_diff_pct",
    "adx",
    "atr_pct",
    "rv_1h",
    "rvol_rolling",
    "obv_slope_norm",
    "vwap_loc",
    "donch_dist",
    "ttc_frac",
    "log_mins_to_close",
    "tod_frac",
]

def reason_codes(ret30: float, rel: float, ema_pct: float, adx_v: float, rvol: float, vwap_loc: float, donch: float) -> str:
    codes=[]
    if ret30 > 0.006: codes.append("MOM30+")
    if ret30 < -0.006: codes.append("MOM30-")
    if rel > 0.003: codes.append("RS+")
    if rel < -0.003: codes.append("RS-")
    if ema_pct > 0 and adx_v > 18: codes.append("TREND+")
    if ema_pct < 0 and adx_v > 18: codes.append("TREND-")
    if vwap_loc > 0.3: codes.append(">VWAP")
    if vwap_loc < -0.3: codes.append("<VWAP")
    if rvol > 1.4: codes.append("RVOL")
    if donch < 0.6: codes.append("NEARHIGH")
    return " ".join(codes[:4])

def compute_features_from_5m(
    bars_5m: List[dict],
    spy_ret_30m: float,
    mins_to_close: float,
) -> Optional[FeatureRow]:
    if not bars_5m or len(bars_5m) < 7:
        return None

    c = np.array([_f(b.get("c")) for b in bars_5m], dtype=float)
    h = np.array([_f(b.get("h")) for b in bars_5m], dtype=float)
    l = np.array([_f(b.get("l")) for b in bars_5m], dtype=float)
    v = np.array([_f(b.get("v")) for b in bars_5m], dtype=float)
    vw= np.array([_f(b.get("vw")) for b in bars_5m], dtype=float)

    price=float(c[-1])
    if price <= 0:
        return None

    vwap=float(vw[-1]) if vw[-1] > 0 else float((c*np.maximum(v,1.0)).sum()/max(1.0, float(v.sum())))

    ret5 = float(price/c[-2]-1.0) if c[-2] else 0.0
    ret30= float(price/c[-7]-1.0) if c[-7] else 0.0
    rel_strength = float(ret30 - float(spy_ret_30m))

    ema_fast = ema(c, span=6)
    ema_slow = ema(c, span=18)
    ema_diff_pct = float((ema_fast[-1]-ema_slow[-1]) / price)

    adx_v = float(adx(h,l,c,period=14)[-1])
    atr_v = float(atr(h,l,c,period=14)[-1])
    atr_pct = float(atr_v/price) if atr_v>0 else 0.0

    logrets = np.diff(np.log(np.maximum(c, 1e-9)))
    seg = logrets[-12:] if logrets.size >= 12 else logrets
    rv_1h = float(np.std(seg)*math.sqrt(max(1.0,float(seg.size)))) if seg.size>2 else 0.0

    medv = float(np.median(v[-20:])) if v.size>=20 else float(np.median(v)) if v.size>0 else 0.0
    rvol = float(v[-1]/medv) if medv>0 else 0.0

    obv_series = obv(c, v)
    obv_sl = slope_last(obv_series, window=12)
    obv_slope_norm = float(obv_sl / max(1.0, medv))

    vwap_loc = float((price - vwap)/atr_v) if atr_v>0 else 0.0
    look = min(20, h.size)
    hi = float(np.max(h[-look:])) if look>0 else float(h[-1])
    donch = float((hi-price)/atr_v) if atr_v>0 else 0.0

    ttc_frac = float(np.clip(mins_to_close/390.0, 0.0, 1.0))
    log_mins = float(math.log1p(max(0.0, mins_to_close)))
    tod_frac = float(np.clip((390.0 - mins_to_close)/390.0, 0.0, 1.0))

    feats = np.array([
        ret5, ret30, rel_strength, ema_diff_pct, adx_v, atr_pct, rv_1h, rvol, obv_slope_norm,
        vwap_loc, donch, ttc_frac, log_mins, tod_frac
    ], dtype=float)

    reasons = reason_codes(ret30, rel_strength, ema_diff_pct, adx_v, rvol, vwap_loc, donch)
    return FeatureRow(price=price, vwap=vwap, features=feats, reasons=reasons)
