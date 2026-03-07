from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from .features import FEATURE_NAMES

TTC_IDX = FEATURE_NAMES.index("ttc_frac")
BUCKETS = [
    ("0_30", 0.0, 30.0),
    ("30_60", 30.0, 60.0),
    ("60_120", 60.0, 120.0),
    ("120_240", 120.0, 240.0),
    ("240_390", 240.0, 9999.0),
]

def bucket_name_from_ttc(ttc_frac: float) -> str:
    mins = float(ttc_frac) * 390.0
    for name, lo, hi in BUCKETS:
        if mins >= lo and mins < hi:
            return name
    return "240_390"

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0/(1.0+np.exp(-x))

@dataclass
class Calibrator:
    method: str
    model: Any
    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        rs = raw_scores.reshape(-1).astype(float)
        if self.method == "isotonic":
            p = self.model.predict(rs)
            return np.clip(p, 0.0, 1.0)
        p = self.model.predict_proba(rs.reshape(-1,1))[:,1]
        return np.clip(p, 0.0, 1.0)

@dataclass
class ModelBundle:
    pipeline: Any
    bucket_calibrators: Dict[str, Calibrator]
    priors: Dict[str, float]
    alpha: float
    feature_names: list
    meta: Dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.pipeline.decision_function(X), dtype=float)
        ttc = np.asarray(X[:, TTC_IDX], dtype=float) if X.shape[1] > TTC_IDX else np.zeros(X.shape[0])
        out = np.empty(X.shape[0], dtype=float)
        bucket_to_idx: Dict[str, list] = {}
        for i in range(X.shape[0]):
            b = bucket_name_from_ttc(ttc[i])
            bucket_to_idx.setdefault(b, []).append(i)
        for b, idxs in bucket_to_idx.items():
            idxs_arr = np.array(idxs, dtype=int)
            cal = self.bucket_calibrators.get(b) or self.bucket_calibrators.get("global")
            p = sigmoid(raw[idxs_arr]) if cal is None else cal.predict(raw[idxs_arr])
            prior = float(self.priors.get(b, self.priors.get("global", 0.5)))
            a = float(self.alpha)
            out[idxs_arr] = np.clip(a*p + (1.0-a)*prior, 0.0, 1.0)
        return out

def bundle_path(model_dir: str, threshold_pct: int) -> str:
    return os.path.join(model_dir, f"pt{threshold_pct}", "bundle.joblib")

def try_load_bundle(model_dir: str, threshold_pct: int) -> Tuple[Optional[ModelBundle], str]:
    p = bundle_path(model_dir, threshold_pct)
    if not os.path.exists(p):
        return None, "missing"
    try:
        b = joblib.load(p)
        if getattr(b, "feature_names", None) != list(FEATURE_NAMES):
            return None, "incompatible"
        return b, "ok"
    except Exception:
        return None, "error"

def heuristic_prob(features: np.ndarray, threshold_pct: int) -> float:
    f = features.astype(float)
    ret5, ret30, rel = f[0], f[1], f[2]
    ema = f[3]
    adx = np.clip(f[4], 0, 60)/30.0
    atrp = np.clip(f[5], 0, 0.08)/0.02
    rv = np.clip(f[6], 0, 0.10)/0.03
    rvol = np.clip(f[7], 0, 4.0)/1.5
    vwap_loc = np.clip(f[9], -2.0, 2.0)
    donch = np.clip(f[10], 0.0, 4.0)
    ttc = np.clip(f[11], 0.0, 1.0)
    logm = np.clip(f[12], 0.0, 7.0)/3.0
    dvol_log = np.clip(f[14], 0.0, 18.0)/10.0
    wick = np.clip(f[16], 0.0, 2.0)
    score = (0.55*(ret30/0.02) + 0.25*(ret5/0.01) + 0.25*(rel/0.02) + 0.22*(ema/0.01) + 0.15*adx + 0.18*rvol + 0.15*vwap_loc + 0.10*logm - 0.20*donch - 0.22*(1.0-ttc) - 0.18*atrp - 0.10*rv + 0.06*dvol_log - 0.08*wick)
    intercept = -0.55 if threshold_pct >= 2 else 0.0
    p = sigmoid(np.array([intercept + score]))[0]
    hi = 0.78 if threshold_pct >= 2 else 0.88
    return float(np.clip(p, 0.01, hi))

def predict_probs(model_dir: str, X: np.ndarray, threshold_pct: int) -> Tuple[np.ndarray, str, str]:
    bundle, status = try_load_bundle(model_dir, threshold_pct)
    if bundle is None:
        probs = np.array([heuristic_prob(x, threshold_pct) for x in X], dtype=float)
        return probs, "heuristic", status
    return bundle.predict_proba(X), "trained", "ok"
