from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    tr_s=np.zeros_like(tr); p_s=np.zeros_like(tr); m_s=np.zeros_like(tr)
    tr_s[0]=tr[0]; p_s[0]=plus_dm[0]; m_s[0]=minus_dm[0]
    for i in range(1, tr.size):
        tr_s[i]=tr_s[i-1]-tr_s[i-1]/period + tr[i]
        p_s[i]=p_s[i-1]-p_s[i-1]/period + plus_dm[i]
        m_s[i]=m_s[i-1]-m_s[i-1]/period + minus_dm[i]
    plus_di = 100.0*(p_s/np.where(tr_s==0,1.0,tr_s))
    minus_di= 100.0*(m_s/np.where(tr_s==0,1.0,tr_s))
    dx = 100.0*np.abs(plus_di-minus_di)/np.where((plus_di+minus_di)==0,1.0,(plus_di+minus_di))
    adx_sm=np.zeros_like(dx); adx_sm[0]=dx[0]
    for i in range(1, dx.size):
        adx_sm[i]=(adx_sm[i-1]*(period-1)+dx[i])/period
    out=np.zeros(n, dtype=float); out[1:]=adx_sm
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
    risk: str
    reasons: str
    used_tod_profile: bool

FEATURE_NAMES = [
    "ret_5m",
    "ret_30m",
    "rel_strength_30m",
    "ema_diff_pct",
    "adx",
    "atr_pct",
    "rv_1h",
    "rvol_tod",
    "obv_slope_norm",
    "vwap_loc",
    "donch_dist",
    "ttc_frac",
    "log_mins_to_close",
    "tod_frac",
    "dvol_usd_med_log",
    "range_pct_med",
    "wick_atr_med",
    "int_mom_rvol",
    "int_mom_vwap",
    "int_trend_adx",
    "int_mom_ttc",
]

def _risk_and_micro(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, atr_arr: np.ndarray,
    rolling_bars: int,
    dvol_min_usd: float,
    range_pct_max: float,
    wick_atr_max: float,
) -> Tuple[str, List[str], float, float, float]:
    n=c.size
    if n < 2:
        return "OK", [], 0.0, 0.0, 0.0
    w = min(rolling_bars, n-1)
    # exclude current bar for rolling baselines
    o_w=o[-(w+1):-1]; h_w=h[-(w+1):-1]; l_w=l[-(w+1):-1]; c_w=c[-(w+1):-1]; v_w=v[-(w+1):-1]; atr_w=atr_arr[-(w+1):-1]

    dvol_series = c_w * np.maximum(v_w, 0.0)
    dvol_med = float(np.median(dvol_series)) if dvol_series.size else 0.0

    rng_series = np.where(c_w>0, (h_w-l_w)/c_w, 0.0)
    rng_med = float(np.median(rng_series)) if rng_series.size else 0.0
    rng_cur = float((h[-1]-l[-1])/c[-1]) if c[-1] else 0.0

    upper = h_w - np.maximum(o_w, c_w)
    lower = np.minimum(o_w, c_w) - l_w
    wick = np.maximum(upper, lower)
    wick_norm = np.where(atr_w>0, wick/atr_w, 0.0)
    wick_med = float(np.median(wick_norm)) if wick_norm.size else 0.0

    atr_cur=float(atr_arr[-1]) if atr_arr.size else 0.0
    upper_c=float(h[-1]-max(o[-1],c[-1]))
    lower_c=float(min(o[-1],c[-1])-l[-1])
    wick_cur=max(upper_c, lower_c)
    wick_cur_n=float(wick_cur/atr_cur) if atr_cur>0 else 0.0

    cond_dvol = dvol_med>0 and dvol_med < dvol_min_usd
    cond_range = (rng_med > range_pct_max) or (rng_cur > range_pct_max)
    cond_wick = (wick_med > wick_atr_max) or (wick_cur_n > wick_atr_max)

    codes=[]
    if cond_dvol: codes.append("LOW_LIQ")
    if cond_wick: codes.append("WICKY")
    if cond_range: codes.append("WIDE_RANGE")

    k = int(cond_dvol)+int(cond_range)+int(cond_wick)
    if k>=2:
        risk="HIGH"
    elif k==1:
        risk="CAUTION"
    else:
        risk="OK"

    return risk, codes, dvol_med, rng_med, wick_med

def _reason_codes(ret30: float, rel: float, ema_pct: float, adx_v: float, rvol_tod: float, vwap_loc: float, donch: float, risk_codes: List[str]) -> str:
    codes=[]
    if ret30 > 0.006: codes.append("MOM30+")
    if ret30 < -0.006: codes.append("MOM30-")
    if rel > 0.003: codes.append("RS+")
    if rel < -0.003: codes.append("RS-")
    if ema_pct > 0 and adx_v > 18: codes.append("TREND+")
    if ema_pct < 0 and adx_v > 18: codes.append("TREND-")
    if vwap_loc > 0.3: codes.append(">VWAP")
    if vwap_loc < -0.3: codes.append("<VWAP")
    if rvol_tod > 1.4: codes.append("RVOL")
    if donch < 0.6: codes.append("NEARHIGH")
    for rc in risk_codes:
        if rc not in codes:
            codes.append(rc)
    return " ".join(codes[:6])

def compute_features_from_5m(
    bars_5m: List[dict],
    spy_ret_30m: float,
    mins_to_close: float,
    tod_baseline_vol_median: Optional[float],
    rolling_rvol_window: int,
    risk_params: Tuple[int, float, float, float],
) -> Optional[FeatureRow]:
    if not bars_5m or len(bars_5m) < 7:
        return None

    o = np.array([_f(b.get("o")) for b in bars_5m], dtype=float)
    h = np.array([_f(b.get("h")) for b in bars_5m], dtype=float)
    l = np.array([_f(b.get("l")) for b in bars_5m], dtype=float)
    c = np.array([_f(b.get("c")) for b in bars_5m], dtype=float)
    v = np.array([_f(b.get("v")) for b in bars_5m], dtype=float)
    vw= np.array([_f(b.get("vw")) for b in bars_5m], dtype=float)

    price=float(c[-1])
    if price <= 0:
        return None
    vwap=float(vw[-1]) if vw[-1] > 0 else float((c*np.maximum(v,1.0)).sum()/max(1.0,float(v.sum())))
    if not (vwap>0):
        return None

    ret5=float(price/c[-2]-1.0) if c[-2] else 0.0
    ret30=float(price/c[-7]-1.0) if c[-7] else 0.0
    rel_strength=float(ret30 - float(spy_ret_30m))

    ema_fast=ema(c, span=6)
    ema_slow=ema(c, span=18)
    ema_diff_pct=float((ema_fast[-1]-ema_slow[-1])/price)

    adx_v=float(adx(h,l,c,period=14)[-1])
    atr_arr=atr(h,l,c,period=14)
    atr_v=float(atr_arr[-1])
    atr_pct=float(atr_v/price) if atr_v>0 else 0.0

    logrets=np.diff(np.log(np.maximum(c, 1e-9)))
    seg=logrets[-12:] if logrets.size>=12 else logrets
    rv_1h=float(np.std(seg)*math.sqrt(max(1.0,float(seg.size)))) if seg.size>2 else 0.0

    current_vol=float(v[-1])
    prev_v=v[:-1]
    med_prev=float(np.median(prev_v[-rolling_rvol_window:])) if prev_v.size>0 else 0.0
    fallback_rvol=(current_vol/med_prev) if med_prev>0 else 1.0

    used_profile=False
    if tod_baseline_vol_median is not None and tod_baseline_vol_median>0:
        rvol_tod=float(current_vol/tod_baseline_vol_median)
        used_profile=True
    else:
        rvol_tod=float(fallback_rvol)

    obv_series=obv(c,v)
    obv_sl=slope_last(obv_series, window=12)
    obv_slope_norm=float(obv_sl/max(1.0, med_prev))

    vwap_loc=float((price-vwap)/atr_v) if atr_v>0 else 0.0
    look=min(20, h.size)
    hi=float(np.max(h[-look:])) if look>0 else float(h[-1])
    donch=float((hi-price)/atr_v) if atr_v>0 else 0.0

    ttc_frac=float(np.clip(mins_to_close/390.0, 0.0, 1.0))
    log_mins=float(math.log1p(max(0.0, mins_to_close)))
    tod_frac=float(np.clip((390.0-mins_to_close)/390.0, 0.0, 1.0))

    liq_rolling_bars, liq_dvol_min_usd, liq_range_pct_max, liq_wick_atr_max = risk_params
    risk, risk_codes, dvol_med, rng_med, wick_med = _risk_and_micro(
        o=o,h=h,l=l,c=c,v=v,atr_arr=atr_arr,
        rolling_bars=liq_rolling_bars,
        dvol_min_usd=liq_dvol_min_usd,
        range_pct_max=liq_range_pct_max,
        wick_atr_max=liq_wick_atr_max,
    )
    dvol_usd_med_log=float(math.log1p(max(0.0, dvol_med)))
    range_pct_med=float(max(0.0, rng_med))
    wick_atr_med=float(max(0.0, wick_med))

    # interactions (auditable)
    int_mom_rvol=float(ret30*(rvol_tod-1.0))
    int_mom_vwap=float(ret30*vwap_loc)
    int_trend_adx=float(ema_diff_pct*adx_v)
    int_mom_ttc=float(ret30*ttc_frac)

    feats=np.array([
        ret5, ret30, rel_strength, ema_diff_pct, adx_v, atr_pct, rv_1h, rvol_tod,
        obv_slope_norm, vwap_loc, donch, ttc_frac, log_mins, tod_frac,
        dvol_usd_med_log, range_pct_med, wick_atr_med,
        int_mom_rvol, int_mom_vwap, int_trend_adx, int_mom_ttc
    ], dtype=float)

    reasons=_reason_codes(ret30, rel_strength, ema_diff_pct, adx_v, rvol_tod, vwap_loc, donch, risk_codes)
    return FeatureRow(price=price, vwap=vwap, features=feats, risk=risk, reasons=reasons, used_tod_profile=used_profile)
