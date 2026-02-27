from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

from .alpaca import AlpacaClient
from .config import Settings
from .features import FEATURE_NAMES
from .market import get_market_times
from .modeling import Calibrator, ModelBundle
from .state import AppState

def _parse_ts(ts: str) -> datetime:
    # Alpaca returns RFC3339 with Z
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def _trading_days(end_local_date: date, lookback_days: int, tz_name: str) -> List[date]:
    # Prefer NYSE calendar for holidays; fallback weekdays.
    tz = ZoneInfo(tz_name)
    start_local_date = end_local_date - timedelta(days=lookback_days * 2)
    days: List[date] = []
    try:
        import pandas_market_calendars as mcal  # type: ignore
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=start_local_date, end_date=end_local_date)
        for idx in sched.index:
            days.append(idx.date())
    except Exception:
        d = start_local_date
        while d <= end_local_date:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
    # keep last lookback_days trading days
    return days[-lookback_days:] if len(days) > lookback_days else days

def _session_utc_for_day(d: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    open_local = datetime(d.year, d.month, d.day, 9, 30, tzinfo=tz)
    close_local = datetime(d.year, d.month, d.day, 16, 0, tzinfo=tz)
    # if calendar library is available, use it for that day's exact open/close (DST/early close/holiday)
    try:
        import pandas_market_calendars as mcal  # type: ignore
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=d, end_date=d)
        if len(sched) == 1:
            open_local = sched.iloc[0]["market_open"].to_pydatetime().astimezone(tz)
            close_local = sched.iloc[0]["market_close"].to_pydatetime().astimezone(tz)
    except Exception:
        pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)

def _suffix_max(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    m = -np.inf
    for i in range(arr.size - 1, -1, -1):
        if arr[i] > m:
            m = arr[i]
        out[i] = m
    return out

def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.size):
        j0 = max(0, i - window + 1)
        out[i] = float(np.median(x[j0:i+1]))
    return out

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if close.size < 2:
        return np.zeros_like(close, dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out = np.zeros_like(close, dtype=float)
    out[0] = tr[0]
    for i in range(1, tr.size):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out

def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.size
    if n < period + 2:
        return np.zeros_like(close, dtype=float)
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
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
    adx_sm = np.zeros_like(dx, dtype=float)
    adx_sm[0] = dx[0]
    for i in range(1, dx.size):
        adx_sm[i] = (adx_sm[i - 1] * (period - 1) + dx[i]) / period
    out = np.zeros(n, dtype=float)
    out[1:] = adx_sm
    return out

def _obv(close: np.ndarray, vol: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close, dtype=float)
    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
    return out

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def _fit_calibrator(raw: np.ndarray, y: np.ndarray) -> Calibrator:
    raw = raw.reshape(-1).astype(float)
    y = y.reshape(-1).astype(int)
    # choose isotonic if enough points and both classes present
    if raw.size >= 2000 and len(np.unique(y)) == 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, y)
        return Calibrator(method="isotonic", model=iso)
    # Platt scaling
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(raw.reshape(-1, 1), y)
    return Calibrator(method="platt", model=lr)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def build_training_dataset(
    client: AlpacaClient,
    symbols: List[str],
    lookback_days: int,
    tz_name: str,
    scan_interval_minutes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[date], Optional[str]]:
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    days = _trading_days(today_local, lookback_days, tz_name)

    X_rows: List[np.ndarray] = []
    y1_rows: List[int] = []
    y2_rows: List[int] = []
    day_idx_rows: List[int] = []

    # Ensure SPY included for regime
    symbols_all = list(dict.fromkeys(symbols + (["SPY"] if "SPY" not in symbols else [])))

    for di, d in enumerate(days):
        open_utc, close_utc = _session_utc_for_day(d, tz_name)
        # pull 5m and 1m for the day
        bars5, err5, _ = client.get_bars(symbols_all, timeframe="5Min", start_utc=open_utc, end_utc=close_utc)
        if err5:
            return np.empty((0, len(FEATURE_NAMES))), np.array([]), np.array([]), [], f"bars5 fetch failed for {d}: {err5}"
        bars1, err1, _ = client.get_bars(symbols_all, timeframe="1Min", start_utc=open_utc, end_utc=close_utc)
        if err1:
            return np.empty((0, len(FEATURE_NAMES))), np.array([]), np.array([]), [], f"bars1 fetch failed for {d}: {err1}"

        # Precompute SPY 30m returns aligned by 5m bar index (approx)
        spy5 = bars5.get("SPY", [])
        spy_ret_30m = None
        if spy5 and len(spy5) >= 7:
            spy_c = np.array([float(b.get("c") or 0.0) for b in spy5], dtype=float)
            spy_ret_30m = np.zeros_like(spy_c, dtype=float)
            for i in range(spy_c.size):
                if i >= 6 and spy_c[i - 6] != 0:
                    spy_ret_30m[i] = spy_c[i] / spy_c[i - 6] - 1.0
                else:
                    spy_ret_30m[i] = 0.0

        # For each symbol, build per-bar rows
        for sym in symbols:
            b5 = bars5.get(sym, [])
            b1 = bars1.get(sym, [])
            if not b5 or not b1 or len(b5) < 20 or len(b1) < 30:
                continue

            # parse 5m arrays
            t5 = np.array([_parse_ts(b["t"]).timestamp() for b in b5], dtype=float)
            c5 = np.array([float(b.get("c") or 0.0) for b in b5], dtype=float)
            h5 = np.array([float(b.get("h") or 0.0) for b in b5], dtype=float)
            l5 = np.array([float(b.get("l") or 0.0) for b in b5], dtype=float)
            v5 = np.array([float(b.get("v") or 0.0) for b in b5], dtype=float)
            vw5 = np.array([float(b.get("vw") or 0.0) for b in b5], dtype=float)
            vw5 = np.where(vw5 > 0, vw5, c5)

            ema_fast = _ema(c5, span=6)
            ema_slow = _ema(c5, span=18)
            ema_diff = ema_fast - ema_slow

            adx_arr = _adx(h5, l5, c5, period=14)
            atr_arr = _atr(h5, l5, c5, period=14)
            atr_pct = np.where(c5 > 0, atr_arr / c5, 0.0)

            logrets = np.diff(np.log(np.maximum(c5, 1e-9)))
            rv = np.zeros_like(c5, dtype=float)
            for i in range(c5.size):
                if i < 3:
                    rv[i] = 0.0
                    continue
                j0 = max(0, i - 12)
                seg = logrets[j0:i]  # length up to 12
                rv[i] = float(np.std(seg) * np.sqrt(max(1.0, float(seg.size)))) if seg.size > 2 else 0.0

            medv = _rolling_median(v5, window=20)
            rvol = np.where(medv > 0, v5 / medv, 0.0)

            obv = _obv(c5, v5)
            obv_slope = np.zeros_like(c5, dtype=float)
            for i in range(c5.size):
                j0 = max(0, i - 12)
                denom = max(1.0, float(i - j0))
                obv_slope[i] = (obv[i] - obv[j0]) / denom

            donch_dist = np.zeros_like(c5, dtype=float)
            for i in range(c5.size):
                j0 = max(0, i - 19)
                hi = float(np.max(h5[j0:i+1]))
                donch_dist[i] = (hi - c5[i]) / atr_arr[i] if atr_arr[i] > 0 else 0.0

            vwap_loc = np.where(atr_arr > 0, (c5 - vw5) / atr_arr, 0.0)

            # parse 1m arrays + suffix max of highs
            t1 = np.array([_parse_ts(b["t"]).timestamp() for b in b1], dtype=float)
            h1 = np.array([float(b.get("h") or 0.0) for b in b1], dtype=float)
            sufmax = _suffix_max(h1)

            # scan bars aligned to scan_interval_minutes (based on bar index)
            step = max(1, int(scan_interval_minutes / 5)) if scan_interval_minutes >= 5 else 1
            for i in range(0, c5.size):
                if i < 20 or i < 6:
                    continue
                if (i % step) != 0:
                    continue

                # scan end time is bar start + 5 minutes
                scan_end_ts = t5[i] + 5 * 60.0
                # find first 1m bar >= scan_end_ts
                j = int(np.searchsorted(t1, scan_end_ts, side="left"))
                if j >= t1.size:
                    continue
                h_future = float(sufmax[j])
                p0 = float(c5[i])

                y1 = 1 if h_future >= (1.01 * p0) else 0
                y2 = 1 if h_future >= (1.02 * p0) else 0

                ret5 = (p0 / c5[i - 1] - 1.0) if c5[i - 1] != 0 else 0.0
                ret30 = (p0 / c5[i - 6] - 1.0) if c5[i - 6] != 0 else 0.0

                # mins to close in local tz
                scan_dt_utc = datetime.fromtimestamp(scan_end_ts, tz=timezone.utc)
                mins_to_close = max(0.0, (close_utc - scan_dt_utc).total_seconds() / 60.0)

                spy_val = 0.0
                if spy_ret_30m is not None:
                    # approximate by index i if lengths match else 0
                    if i < spy_ret_30m.size:
                        spy_val = float(spy_ret_30m[i])

                feats = np.array([
                    float(ret5),
                    float(ret30),
                    float(ema_diff[i]),
                    float(adx_arr[i]),
                    float(atr_pct[i]),
                    float(rv[i]),
                    float(rvol[i]),
                    float(obv_slope[i]),
                    float(vwap_loc[i]),
                    float(donch_dist[i]),
                    float(spy_val),
                    float(mins_to_close),
                ], dtype=float)

                X_rows.append(feats)
                y1_rows.append(y1)
                y2_rows.append(y2)
                day_idx_rows.append(di)

    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.array([]), np.array([]), [], "no training rows produced"
    X = np.vstack(X_rows)
    y1 = np.array(y1_rows, dtype=int)
    y2 = np.array(y2_rows, dtype=int)
    day_idx = np.array(day_idx_rows, dtype=int)
    return X, y1, y2, day_idx, days, None

def _train_one(
    X: np.ndarray,
    y: np.ndarray,
    day_idx: np.ndarray,
    days: List[date],
    threshold_pct: int,
    model_dir: str,
) -> Dict[str, object]:
    # time split by day (expanding train, holdout calibration, holdout validation)
    unique_days = np.unique(day_idx)
    if unique_days.size < 10:
        raise RuntimeError("Not enough trading days for time-split training (need >= 10).")

    n_days = unique_days.size
    cal_days = max(1, int(round(n_days * 0.10)))
    val_days = max(1, int(round(n_days * 0.10)))
    train_end = n_days - (cal_days + val_days)
    train_days = set(unique_days[:train_end])
    cal_set = set(unique_days[train_end:train_end + cal_days])
    val_set = set(unique_days[train_end + cal_days:])

    tr_mask = np.array([d in train_days for d in day_idx], dtype=bool)
    cal_mask = np.array([d in cal_set for d in day_idx], dtype=bool)
    val_mask = np.array([d in val_set for d in day_idx], dtype=bool)

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(np.unique(y_tr)) < 2:
        raise RuntimeError("Training labels have only one class; widen TRAIN_LOOKBACK_DAYS or TRAIN_MAX_SYMBOLS.")

    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=600,
            n_jobs=1,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_tr, y_tr)

    raw_cal = pipeline.decision_function(X_cal)
    calibrator = _fit_calibrator(np.asarray(raw_cal, dtype=float), y_cal)
    raw_val = pipeline.decision_function(X_val)
    p_val = calibrator.predict(np.asarray(raw_val, dtype=float))

    auc = None
    try:
        if len(np.unique(y_val)) == 2 and y_val.size > 0:
            auc = float(roc_auc_score(y_val, p_val))
    except Exception:
        auc = None
    brier = _brier(y_val.astype(float), p_val.astype(float)) if y_val.size > 0 else None

    pt_dir = os.path.join(model_dir, f"pt{threshold_pct}")
    _ensure_dir(pt_dir)
    bundle = ModelBundle(
        pipeline=pipeline,
        calibrator=calibrator,
        feature_names=list(FEATURE_NAMES),
        meta={
            "threshold_pct": threshold_pct,
            "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "n_rows": int(X.shape[0]),
            "n_train": int(X_tr.shape[0]),
            "n_cal": int(X_cal.shape[0]),
            "n_val": int(X_val.shape[0]),
            "auc_val": auc,
            "brier_val": brier,
            "calibrator": calibrator.method,
        },
    )
    joblib.dump(bundle, os.path.join(pt_dir, "bundle.joblib"))
    return bundle.meta

def run_training(settings: Settings, state: AppState, symbols: List[str]) -> Dict[str, object]:
    # Preconditions
    if settings.demo_mode:
        raise RuntimeError("Training is disabled in DEMO_MODE. Set DEMO_MODE=false and provide Alpaca keys.")
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        raise RuntimeError("Missing Alpaca keys. Set ALPACA_API_KEY and ALPACA_API_SECRET.")

    client = AlpacaClient(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        feed=settings.normalized_feed(),
    )
    # dataset
    X, y1, y2, day_idx, days, err = build_training_dataset(
        client=client,
        symbols=symbols,
        lookback_days=settings.train_lookback_days,
        tz_name=settings.timezone,
        scan_interval_minutes=settings.scan_interval_minutes,
    )
    if err:
        raise RuntimeError(err)
    if X.shape[0] < 2000:
        raise RuntimeError(f"Too few training rows ({X.shape[0]}). Increase TRAIN_LOOKBACK_DAYS or TRAIN_MAX_SYMBOLS.")

    meta1 = _train_one(X, y1, day_idx, days, threshold_pct=1, model_dir=settings.model_dir)
    meta2 = _train_one(X, y2, day_idx, days, threshold_pct=2, model_dir=settings.model_dir)

    # update state model status
    with state.lock:
        state.model.pt1.trained = True
        state.model.pt1.path = os.path.join(settings.model_dir, "pt1")
        state.model.pt1.auc_val = meta1.get("auc_val")
        state.model.pt1.brier_val = meta1.get("brier_val")
        state.model.pt1.calibrator = meta1.get("calibrator")

        state.model.pt2.trained = True
        state.model.pt2.path = os.path.join(settings.model_dir, "pt2")
        state.model.pt2.auc_val = meta2.get("auc_val")
        state.model.pt2.brier_val = meta2.get("brier_val")
        state.model.pt2.calibrator = meta2.get("calibrator")

    return {"pt1": meta1, "pt2": meta2}
