from __future__ import annotations
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer

from .alpaca import AlpacaClient
from .config import Settings
from .features import FEATURE_NAMES
from .modeling import Calibrator, ModelBundle, BUCKETS, bucket_name_from_ttc
from .volume_profiles import compute_profiles, save_profiles, slot_index_from_ts, VolumeProfile

def _parse_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def _trading_days(end_local: date, lookback_days: int, tz_name: str) -> List[date]:
    start = end_local - timedelta(days=lookback_days*2)
    out=[]
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=start, end_date=end_local)
        for idx in sched.index:
            out.append(idx.date())
    except Exception:
        d=start
        while d<=end_local:
            if d.weekday()<5:
                out.append(d)
            d += timedelta(days=1)
    return out[-lookback_days:] if len(out)>lookback_days else out

def _session_utc_for_day(d: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    open_local = datetime(d.year,d.month,d.day,9,30,tzinfo=tz)
    close_local= datetime(d.year,d.month,d.day,16,0,tzinfo=tz)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=d, end_date=d)
        if len(sched)==1:
            open_local = sched.iloc[0]["market_open"].to_pydatetime().astimezone(tz)
            close_local= sched.iloc[0]["market_close"].to_pydatetime().astimezone(tz)
    except Exception:
        pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)

def _suffix_max(arr: np.ndarray) -> np.ndarray:
    out=np.empty_like(arr, dtype=float)
    m=-np.inf
    for i in range(arr.size-1,-1,-1):
        if arr[i]>m:
            m=arr[i]
        out[i]=m
    return out

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    if x.size==0:
        return x.copy()
    alpha=2.0/(span+1.0)
    out=np.empty_like(x, dtype=float)
    out[0]=x[0]
    for i in range(1,x.size):
        out[i]=alpha*x[i]+(1-alpha)*out[i-1]
    return out

def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int=14) -> np.ndarray:
    if c.size<2:
        return np.zeros_like(c, dtype=float)
    prev=np.concatenate([[c[0]], c[:-1]])
    tr=np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    out=np.zeros_like(c, dtype=float); out[0]=tr[0]
    for i in range(1,tr.size):
        out[i]=(out[i-1]*(period-1)+tr[i])/period
    return out

def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int=14) -> np.ndarray:
    n=c.size
    if n<period+2:
        return np.zeros_like(c, dtype=float)
    up=h[1:]-h[:-1]; dn=l[:-1]-l[1:]
    plus=np.where((up>dn)&(up>0), up, 0.0)
    minus=np.where((dn>up)&(dn>0), dn, 0.0)
    prev=c[:-1]
    tr=np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-prev), np.abs(l[1:]-prev)))
    tr_s=np.zeros_like(tr); p_s=np.zeros_like(tr); m_s=np.zeros_like(tr)
    tr_s[0]=tr[0]; p_s[0]=plus[0]; m_s[0]=minus[0]
    for i in range(1,tr.size):
        tr_s[i]=tr_s[i-1]-tr_s[i-1]/period + tr[i]
        p_s[i]=p_s[i-1]-p_s[i-1]/period + plus[i]
        m_s[i]=m_s[i-1]-m_s[i-1]/period + minus[i]
    pdi=100*(p_s/np.where(tr_s==0,1.0,tr_s)); mdi=100*(m_s/np.where(tr_s==0,1.0,tr_s))
    dx=100*np.abs(pdi-mdi)/np.where((pdi+mdi)==0,1.0,(pdi+mdi))
    adx_sm=np.zeros_like(dx); adx_sm[0]=dx[0]
    for i in range(1,dx.size):
        adx_sm[i]=(adx_sm[i-1]*(period-1)+dx[i])/period
    out=np.zeros(n, dtype=float); out[1:]=adx_sm
    return out

def _obv(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    out=np.zeros_like(c, dtype=float)
    for i in range(1,c.size):
        if c[i]>c[i-1]: out[i]=out[i-1]+v[i]
        elif c[i]<c[i-1]: out[i]=out[i-1]-v[i]
        else: out[i]=out[i-1]
    return out

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p-y)**2)) if y.size else float("nan")

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0/(1.0+np.exp(-x))

class AppendSplineFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, idxs: List[int], n_knots: int=5, degree: int=3):
        self.idxs=list(idxs)
        self.n_knots=n_knots
        self.degree=degree
        self._spline=SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    def fit(self, X, y=None):
        X=np.asarray(X, dtype=float)
        self._spline.fit(X[:, self.idxs])
        return self
    def transform(self, X):
        X=np.asarray(X, dtype=float)
        return np.hstack([X, self._spline.transform(X[:, self.idxs])])

def _fit_platt(raw: np.ndarray, y: np.ndarray) -> Calibrator:
    lr=LogisticRegression(solver="lbfgs")
    lr.fit(raw.reshape(-1,1), y)
    return Calibrator(method="platt", model=lr)

def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> Optional[Calibrator]:
    if raw.size < 2000 or len(np.unique(y)) < 2:
        return None
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw.reshape(-1), y.reshape(-1))
    return Calibrator(method="isotonic", model=iso)

def _fit_best_calibrator(raw: np.ndarray, y: np.ndarray, min_samples: int) -> Calibrator:
    raw=np.asarray(raw, dtype=float).reshape(-1); y=np.asarray(y, dtype=int).reshape(-1)
    if raw.size < max(min_samples, 200):
        return _fit_platt(raw, y)
    split=int(0.8*raw.size)
    fit_idx=np.arange(split)
    eval_idx=np.arange(split, raw.size) if split < raw.size else np.arange(raw.size)
    pl=_fit_platt(raw[fit_idx], y[fit_idx]); pl_p=pl.predict(raw[eval_idx]); pl_b=_brier(y[eval_idx].astype(float), pl_p.astype(float))
    iso=_fit_isotonic(raw[fit_idx], y[fit_idx])
    if iso is None:
        return _fit_platt(raw, y)
    iso_p=iso.predict(raw[eval_idx]); iso_b=_brier(y[eval_idx].astype(float), iso_p.astype(float))
    if iso_b <= pl_b:
        iso_full=_fit_isotonic(raw, y)
        return iso_full if iso_full is not None else _fit_platt(raw, y)
    return _fit_platt(raw, y)

def _fit_bucket_calibrators(raw: np.ndarray, y: np.ndarray, ttc_frac: np.ndarray, min_bucket_samples: int):
    raw=np.asarray(raw, dtype=float).reshape(-1); y=np.asarray(y, dtype=int).reshape(-1); ttc_frac=np.asarray(ttc_frac, dtype=float).reshape(-1)
    calibs={}; methods={}
    cal_global=_fit_best_calibrator(raw, y, min_bucket_samples)
    calibs["global"]=cal_global; methods["global"]=cal_global.method
    for name,_,_ in BUCKETS:
        mask=np.array([bucket_name_from_ttc(ttc_frac[i])==name for i in range(ttc_frac.size)], dtype=bool)
        idx=np.where(mask)[0]
        if idx.size < min_bucket_samples or len(np.unique(y[idx])) < 2:
            continue
        c=_fit_best_calibrator(raw[idx], y[idx], min_bucket_samples)
        calibs[name]=c; methods[name]=c.method
    return calibs, methods

def _bucket_priors(y: np.ndarray, ttc_frac: np.ndarray):
    y=np.asarray(y, dtype=float).reshape(-1); ttc_frac=np.asarray(ttc_frac, dtype=float).reshape(-1)
    priors={"global": float(y.mean()) if y.size else 0.5}
    for name,_,_ in BUCKETS:
        mask=np.array([bucket_name_from_ttc(ttc_frac[i])==name for i in range(ttc_frac.size)], dtype=bool)
        idx=np.where(mask)[0]
        if idx.size:
            priors[name]=float(y[idx].mean())
    return priors

def build_training_dataset(client: AlpacaClient, symbols: List[str], lookback_days: int, tz_name: str, scan_interval_minutes: int, vol_profiles: Dict[str, VolumeProfile], liq_rolling_bars: int):
    tz=ZoneInfo(tz_name)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    days = _trading_days(today_local, lookback_days, tz_name)
    X_rows=[]; y1_rows=[]; y2_rows=[]; day_idx_rows=[]
    symbols_all=list(dict.fromkeys(symbols + (["SPY"] if "SPY" not in symbols else [])))
    step=max(1, int(scan_interval_minutes/5)) if scan_interval_minutes>=5 else 1
    for di,d in enumerate(days):
        open_utc, close_utc = _session_utc_for_day(d, tz_name)
        bars5, err5, _ = client.get_bars(symbols_all, "5Min", start_utc=open_utc, end_utc=close_utc)
        if err5:
            return np.empty((0,len(FEATURE_NAMES))), np.array([]), np.array([]), np.array([]), f"bars5 fetch failed {d}: {err5}"
        bars1, err1, _ = client.get_bars(symbols_all, "1Min", start_utc=open_utc, end_utc=close_utc)
        if err1:
            return np.empty((0,len(FEATURE_NAMES))), np.array([]), np.array([]), np.array([]), f"bars1 fetch failed {d}: {err1}"
        spy5=bars5.get("SPY", [])
        spy_ret_30m=None
        if spy5 and len(spy5) >= 7:
            spy_c=np.array([float(b.get("c") or 0.0) for b in spy5], dtype=float)
            spy_ret_30m=np.zeros_like(spy_c)
            for i in range(spy_c.size):
                spy_ret_30m[i]=(spy_c[i]/spy_c[i-6]-1.0) if i>=6 and spy_c[i-6] else 0.0
        for sym in symbols:
            b5=bars5.get(sym, []); b1=bars1.get(sym, [])
            if not b5 or not b1 or len(b5) < 25 or len(b1) < 60:
                continue
            t5=np.array([_parse_ts(b["t"]).timestamp() for b in b5], dtype=float)
            o5=np.array([float(b.get("o") or 0.0) for b in b5], dtype=float)
            h5=np.array([float(b.get("h") or 0.0) for b in b5], dtype=float)
            l5=np.array([float(b.get("l") or 0.0) for b in b5], dtype=float)
            c5=np.array([float(b.get("c") or 0.0) for b in b5], dtype=float)
            v5=np.array([float(b.get("v") or 0.0) for b in b5], dtype=float)
            vw5=np.array([float(b.get("vw") or 0.0) for b in b5], dtype=float)
            vw5=np.where(vw5>0, vw5, c5)
            price=np.maximum(c5, 1e-9)
            ema_fast=_ema(c5, span=6); ema_slow=_ema(c5, span=18); ema_diff_pct=(ema_fast-ema_slow)/price
            adx_arr=_adx(h5,l5,c5,period=14); atr_arr=_atr(h5,l5,c5,period=14); atr_pct=np.where(price>0, atr_arr/price, 0.0)
            logrets=np.diff(np.log(np.maximum(c5,1e-9))); rv=np.zeros_like(c5)
            for i in range(c5.size):
                j0=max(0, i-12); seg=logrets[j0:i]
                rv[i]=float(np.std(seg)*np.sqrt(max(1.0,float(seg.size)))) if seg.size>2 else 0.0
            obv=_obv(c5, v5); obv_slope=np.zeros_like(c5)
            for i in range(c5.size):
                j0=max(0, i-12); denom=max(1.0, float(i-j0)); obv_slope[i]=(obv[i]-obv[j0])/denom
            donch=np.zeros_like(c5)
            for i in range(c5.size):
                j0=max(0, i-19); hi=float(np.max(h5[j0:i+1])); donch[i]=(hi-c5[i])/atr_arr[i] if atr_arr[i]>0 else 0.0
            vwap_loc=np.where(atr_arr>0, (c5-vw5)/atr_arr, 0.0)
            t1=np.array([_parse_ts(b["t"]).timestamp() for b in b1], dtype=float)
            h1=np.array([float(b.get("h") or 0.0) for b in b1], dtype=float)
            suf=_suffix_max(h1)
            prof=vol_profiles.get(sym)
            for i in range(0, c5.size):
                if i < 24 or (i % step) != 0:
                    continue
                scan_end_ts=t5[i] + 5*60.0
                j=int(np.searchsorted(t1, scan_end_ts, side="left"))
                if j >= t1.size:
                    continue
                p0=float(c5[i])
                if p0 <= 0:
                    continue
                h_future=float(suf[j]); y1=1 if h_future >= 1.01*p0 else 0; y2=1 if h_future >= 1.02*p0 else 0
                ret5=(p0/c5[i-1]-1.0) if c5[i-1] else 0.0; ret30=(p0/c5[i-6]-1.0) if c5[i-6] else 0.0
                spy_val=float(spy_ret_30m[i]) if spy_ret_30m is not None and i < spy_ret_30m.size else 0.0
                rel_strength=float(ret30-spy_val)
                slot_idx=None
                try:
                    slot_idx = slot_index_from_ts(_parse_ts(b5[i]["t"]), tz_name)
                except Exception:
                    slot_idx=None
                baseline=None
                if prof and prof.available and slot_idx is not None:
                    baseline = prof.slot_median[slot_idx]
                prev=v5[max(0,i-20):i]; med_prev=float(np.median(prev)) if prev.size>0 else 0.0
                fallback_rvol=(float(v5[i])/med_prev) if med_prev>0 else 1.0
                rvol_tod=(float(v5[i])/float(baseline)) if (baseline is not None and float(baseline)>0) else float(fallback_rvol)
                obv_slope_norm=float(obv_slope[i]/max(1.0, med_prev))
                scan_dt=datetime.fromtimestamp(scan_end_ts, tz=timezone.utc)
                mins_to_close=max(0.0, (close_utc - scan_dt).total_seconds()/60.0)
                ttc_frac=float(np.clip(mins_to_close/390.0, 0.0, 1.0)); log_m=float(np.log1p(max(0.0, mins_to_close))); tod_frac=float(np.clip((390.0-mins_to_close)/390.0, 0.0, 1.0))
                j0=max(0, i-liq_rolling_bars)
                if j0 >= i:
                    dvol_med=0.0; range_med=0.0; wick_med=0.0
                else:
                    c_w=c5[j0:i]; v_w=v5[j0:i]; h_w=h5[j0:i]; l_w=l5[j0:i]; o_w=o5[j0:i]; atr_w=atr_arr[j0:i]
                    dvol_med=float(np.median(c_w*np.maximum(v_w,0.0))) if c_w.size else 0.0
                    range_med=float(np.median(np.where(c_w>0, (h_w-l_w)/c_w, 0.0))) if c_w.size else 0.0
                    upper=h_w-np.maximum(o_w,c_w); lower=np.minimum(o_w,c_w)-l_w
                    wick=np.maximum(upper, lower); wick_norm=np.where(atr_w>0, wick/atr_w, 0.0)
                    wick_med=float(np.median(wick_norm)) if wick_norm.size else 0.0
                dvol_log=float(np.log1p(max(0.0, dvol_med))); range_pct_med=float(max(0.0, range_med)); wick_atr_med=float(max(0.0, wick_med))
                int_mom_rvol=float(ret30*(rvol_tod-1.0)); int_mom_vwap=float(ret30*float(vwap_loc[i])); int_trend_adx=float(float(ema_diff_pct[i])*float(adx_arr[i])); int_mom_ttc=float(ret30*ttc_frac)
                feats=np.array([float(ret5), float(ret30), float(rel_strength), float(ema_diff_pct[i]), float(adx_arr[i]), float(atr_pct[i]), float(rv[i]), float(rvol_tod), float(obv_slope_norm), float(vwap_loc[i]), float(donch[i]), float(ttc_frac), float(log_m), float(tod_frac), float(dvol_log), float(range_pct_med), float(wick_atr_med), float(int_mom_rvol), float(int_mom_vwap), float(int_trend_adx), float(int_mom_ttc)], dtype=float)
                X_rows.append(feats); y1_rows.append(y1); y2_rows.append(y2); day_idx_rows.append(di)
    if not X_rows:
        return np.empty((0,len(FEATURE_NAMES))), np.array([]), np.array([]), np.array([]), "no training rows produced"
    X=np.vstack(X_rows)
    return X, np.array(y1_rows,dtype=int), np.array(y2_rows,dtype=int), np.array(day_idx_rows,dtype=int), None

def _make_base_pipeline(C: float, l1_ratio: float, class_weight: Optional[str]) -> Pipeline:
    spline_cols = [FEATURE_NAMES.index(n) for n in ["ttc_frac","rvol_tod","vwap_loc","donch_dist"]]
    return Pipeline([
        ("spline", AppendSplineFeatures(idxs=spline_cols, n_knots=5, degree=3)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=float(l1_ratio), C=float(C), max_iter=2000, n_jobs=1, class_weight=class_weight)),
    ])

def _select_and_train_bundle(X: np.ndarray, y: np.ndarray, day_idx: np.ndarray, settings: Settings, threshold_pct: int, model_dir: str) -> Dict[str, object]:
    uniq=np.unique(day_idx)
    if uniq.size < 12:
        raise RuntimeError("Not enough trading days for time-split training (need >=12).")
    n=uniq.size; cal_days=max(2,int(round(n*0.15))); val_days=max(2,int(round(n*0.15))); train_end=max(1, n-(cal_days+val_days))
    train_set=set(uniq[:train_end]); cal_set=set(uniq[train_end:train_end+cal_days]); val_set=set(uniq[train_end+cal_days:])
    tr_mask=np.array([d in train_set for d in day_idx]); cal_mask=np.array([d in cal_set for d in day_idx]); val_mask=np.array([d in val_set for d in day_idx])
    X_tr,y_tr=X[tr_mask],y[tr_mask]; X_cal,y_cal=X[cal_mask],y[cal_mask]; X_val,y_val=X[val_mask],y[val_mask]
    if len(np.unique(y_tr)) < 2:
        raise RuntimeError("Training labels have only one class; widen lookback or symbols.")
    X_traincal=np.vstack([X_tr, X_cal]) if X_cal.size else X_tr
    y_traincal=np.concatenate([y_tr, y_cal]) if y_cal.size else y_tr
    ttc_traincal=X_traincal[:, FEATURE_NAMES.index("ttc_frac")]
    candidates=[]
    for cw in [None, "balanced"]:
        for C in settings.enet_c_values:
            for l1 in settings.enet_l1_values:
                pipe=_make_base_pipeline(C=C, l1_ratio=l1, class_weight=cw)
                pipe.fit(X_tr, y_tr)
                raw_cal=pipe.decision_function(X_cal); raw_val=pipe.decision_function(X_val)
                ttc_cal=X_cal[:, FEATURE_NAMES.index("ttc_frac")]; ttc_val=X_val[:, FEATURE_NAMES.index("ttc_frac")]
                calibs, methods=_fit_bucket_calibrators(raw_cal, y_cal, ttc_cal, settings.calib_min_bucket_samples)
                priors=_bucket_priors(y_traincal, ttc_traincal)
                p_cal=np.empty_like(raw_val, dtype=float)
                for i in range(raw_val.size):
                    b=bucket_name_from_ttc(ttc_val[i]); cal=calibs.get(b) or calibs.get("global")
                    p_cal[i]=float(cal.predict(np.array([raw_val[i]]))[0]) if cal is not None else float(_sigmoid(np.array([raw_val[i]]))[0])
                for alpha in settings.prior_alpha_values:
                    p_final=np.empty_like(p_cal, dtype=float)
                    for i in range(p_cal.size):
                        b=bucket_name_from_ttc(ttc_val[i]); prior=float(priors.get(b, priors.get("global", 0.5)))
                        p_final[i]=float(np.clip(alpha*p_cal[i] + (1.0-alpha)*prior, 0.0, 1.0))
                    brier=_brier(y_val.astype(float), p_final.astype(float))
                    auc=None
                    try:
                        if len(np.unique(y_val))==2:
                            auc=float(roc_auc_score(y_val, p_final))
                    except Exception:
                        auc=None
                    candidates.append({"cw":"balanced" if cw=="balanced" else "none", "C":float(C), "l1":float(l1), "alpha":float(alpha), "brier":float(brier), "auc":auc, "methods":methods})
    def key(c):
        return (c["brier"], -(c["auc"] if c["auc"] is not None else -1.0))
    best=min(candidates, key=key)
    pipe_final=_make_base_pipeline(C=best["C"], l1_ratio=best["l1"], class_weight=("balanced" if best["cw"]=="balanced" else None))
    pipe_final.fit(X_traincal, y_traincal)
    raw_val=pipe_final.decision_function(X_val); ttc_val=X_val[:, FEATURE_NAMES.index("ttc_frac")]
    calibs_final, methods_final=_fit_bucket_calibrators(raw_val, y_val, ttc_val, settings.calib_min_bucket_samples)
    priors=_bucket_priors(y_traincal.astype(float), ttc_traincal)
    pt_dir=os.path.join(model_dir, f"pt{threshold_pct}"); os.makedirs(pt_dir, exist_ok=True)
    bundle=ModelBundle(pipeline=pipe_final, bucket_calibrators=calibs_final, priors=priors, alpha=float(best["alpha"]), feature_names=list(FEATURE_NAMES), meta={"threshold_pct": threshold_pct, "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "n_rows": int(X.shape[0]), "split_days": {"train": int(len(train_set)), "cal": int(len(cal_set)), "val": int(len(val_set))}, "selected": {k: best[k] for k in ["cw","C","l1","alpha","brier","auc"]}, "bucket_calibration_methods": methods_final, "prior_means": priors, "calib_min_bucket_samples": int(settings.calib_min_bucket_samples), "auc_val": float(best["auc"]) if best["auc"] is not None else None, "brier_val": float(best["brier"]), "calibrator": "bucketed", "class_weight": best["cw"], "alpha": float(best["alpha"]),})
    joblib.dump(bundle, os.path.join(pt_dir, "bundle.joblib"))
    return {"threshold_pct": threshold_pct, "brier_val": float(best["brier"]), "auc_val": float(best["auc"]) if best["auc"] is not None else None, "class_weight": best["cw"], "calibrator": "bucketed", "alpha": float(best["alpha"]), "C": float(best["C"]), "l1_ratio": float(best["l1"])}

def run_training(settings: Settings, symbols: List[str]) -> Dict[str, object]:
    if settings.demo_mode:
        raise RuntimeError("Training is disabled in DEMO_MODE. Set DEMO_MODE=false and provide Alpaca keys.")
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        raise RuntimeError("Missing Alpaca keys.")
    client=AlpacaClient(settings.alpaca_api_key, settings.alpaca_api_secret, feed=settings.normalized_feed())
    profiles=compute_profiles(client=client, symbols=symbols, tz_name=settings.timezone, lookback_days=settings.tod_rvol_lookback_days, min_days=settings.tod_rvol_min_days)
    save_profiles(settings.model_dir, profiles)
    X,y1,y2,day_idx,err = build_training_dataset(client=client, symbols=symbols, lookback_days=settings.train_lookback_days, tz_name=settings.timezone, scan_interval_minutes=settings.scan_interval_minutes, vol_profiles=profiles, liq_rolling_bars=settings.liq_rolling_bars)
    if err:
        raise RuntimeError(err)
    if X.shape[0] < 2500:
        raise RuntimeError(f"Too few training rows ({X.shape[0]}). Increase TRAIN_LOOKBACK_DAYS or TRAIN_MAX_SYMBOLS.")
    meta1=_select_and_train_bundle(X,y1,day_idx,settings,1,settings.model_dir)
    meta2=_select_and_train_bundle(X,y2,day_idx,settings,2,settings.model_dir)
    avail=sum(1 for p in profiles.values() if p.available); missing=len(profiles)-avail
    return {"pt1": meta1, "pt2": meta2, "volume_profiles": {"symbols": len(profiles), "available": avail, "missing": missing, "lookback_days": settings.tod_rvol_lookback_days, "min_days": settings.tod_rvol_min_days}}
