from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import joblib

from .features import FEATURE_NAMES

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def training_state_path(model_dir: str) -> str:
    return os.path.join(model_dir, "training_last.json")

def save_training_last(model_dir: str, payload: Dict[str, Any]) -> None:
    os.makedirs(model_dir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("saved_at_utc", _utc_now())
    with open(training_state_path(model_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_training_last(model_dir: str) -> Optional[Dict[str, Any]]:
    p = training_state_path(model_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _bundle_path(model_dir: str, threshold_pct: int) -> str:
    return os.path.join(model_dir, f"pt{threshold_pct}", "bundle.joblib")

def load_model_meta(model_dir: str, threshold_pct: int) -> Tuple[Optional[Dict[str, Any]], str]:
    p = _bundle_path(model_dir, threshold_pct)
    if not os.path.exists(p):
        return None, "missing"
    try:
        b = joblib.load(p)
        if getattr(b, "feature_names", None) != list(FEATURE_NAMES):
            return None, "incompatible"
        meta = getattr(b, "meta", None)
        return (dict(meta) if isinstance(meta, dict) else {}), "ok"
    except Exception:
        return None, "error"
