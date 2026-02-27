from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from .features import FEATURE_NAMES

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class Calibrator:
    method: str  # "isotonic" or "platt"
    model: Any

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        rs = raw_scores.reshape(-1)
        if self.method == "isotonic":
            # isotonic regression expects 1d
            p = self.model.predict(rs)
            return np.clip(p, 0.0, 1.0)
        # platt uses logistic regression on raw score
        p = self.model.predict_proba(rs.reshape(-1, 1))[:, 1]
        return np.clip(p, 0.0, 1.0)

@dataclass
class ModelBundle:
    pipeline: Any
    calibrator: Calibrator
    feature_names: list
    meta: Dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.pipeline.decision_function(X)
        return self.calibrator.predict(np.asarray(raw, dtype=float))

def bundle_path(model_dir: str, threshold_pct: int) -> str:
    return os.path.join(model_dir, f"pt{threshold_pct}", "bundle.joblib")

def load_bundle(model_dir: str, threshold_pct: int) -> Optional[ModelBundle]:
    path = bundle_path(model_dir, threshold_pct)
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        return obj
    except Exception:
        return None

def heuristic_prob(features: np.ndarray) -> float:
    # Map feature vector -> pseudo-probability (stable day-1 fallback)
    # Features: see FEATURE_NAMES
    f = features.copy().astype(float)
    # basic scaling/clipping
    # momentum helps
    ret5 = np.clip(f[0], -0.03, 0.03) / 0.01
    ret30 = np.clip(f[1], -0.08, 0.08) / 0.02
    ema = np.clip(f[2], -2.0, 2.0)
    adx = np.clip(f[3], 0, 50) / 25.0
    atr_pct = np.clip(f[4], 0.0, 0.08) / 0.02
    rv = np.clip(f[5], 0.0, 0.08) / 0.02
    rvol = np.clip(f[6], 0.0, 4.0) / 1.5
    obv_sl = np.clip(f[7], -5e7, 5e7) / 2e7
    vwap_loc = np.clip(f[8], -2.0, 2.0)
    donch = np.clip(f[9], 0.0, 4.0)
    spy = np.clip(f[10], -0.03, 0.03) / 0.01
    ttc = np.clip(f[11], 0.0, 390.0) / 390.0

    # Heuristic: momentum + trend + participation + regime, penalize too-far-from-high and too-late-in-day
    score = (
        0.55 * ret30 +
        0.25 * ret5 +
        0.25 * ema +
        0.20 * adx +
        0.22 * rvol +
        0.10 * obv_sl +
        0.18 * vwap_loc +
        0.20 * spy -
        0.18 * donch -
        0.25 * (1.0 - ttc)
    )
    return float(sigmoid(np.array([score]))[0])

def predict_probs(
    model_dir: str,
    X: np.ndarray,
    threshold_pct: int,
) -> Tuple[np.ndarray, str]:
    bundle = load_bundle(model_dir, threshold_pct)
    if bundle is None:
        # heuristic fallback
        probs = np.array([heuristic_prob(x) for x in X], dtype=float)
        return probs, "heuristic"
    probs = bundle.predict_proba(X)
    return probs, "trained"
