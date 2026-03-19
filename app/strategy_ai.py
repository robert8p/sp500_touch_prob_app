from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .config import Settings
from .state import AIStrategyStatus, AppState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any, limit: int = 280) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _pick_rows(rows: List[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    def sort_key(r: Dict[str, Any]):
        signal = str(r.get("signal") or "")
        signal_rank = 2 if signal == "ACTIONABLE" else 1 if signal == "CANDIDATE" else 0
        return (
            -int(r.get("relative_strength_rank") or 999999),
            signal_rank,
            1 if r.get("watchlist_rescue") else 0,
            float(r.get("prob_1") or 0.0),
            float(r.get("prob_touch") or 0.0),
            float(r.get("prob_path") or 0.0),
        )

    chosen: List[Dict[str, Any]] = []
    for r in sorted(rows, key=sort_key, reverse=True)[:max_candidates]:
        chosen.append({
            "symbol": r.get("symbol"),
            "sector": r.get("sector"),
            "price": r.get("price"),
            "prob_1": r.get("prob_1"),
            "prob_touch": r.get("prob_touch"),
            "prob_path": r.get("prob_path"),
            "signal": r.get("signal"),
            "setup_family": r.get("setup_family"),
            "relative_strength_rank": r.get("relative_strength_rank"),
            "watchlist_rescue": bool(r.get("watchlist_rescue")),
            "risk": r.get("risk"),
            "uncertainty": r.get("uncertainty"),
            "reasons": _clean_text(r.get("reasons"), 220),
            "risk_reasons": _clean_text(r.get("risk_reasons"), 180),
            "regime_state": r.get("regime_state"),
            "regime_reasons": _clean_text(r.get("regime_reasons"), 160),
            "display_touch_threshold": r.get("display_touch_threshold"),
            "path_action_min": r.get("path_action_min"),
        })
    return chosen


def build_strategy_payload(state: AppState, settings: Settings, diagnostics_summary: Dict[str, Any]) -> Dict[str, Any]:
    status = state.snapshot_status()
    scores = state.snapshot_scores()
    rows = scores.get("rows") or []
    watchlist_rows = scores.get("watchlist_rescue_rows") or []
    top_rows = _pick_rows(rows, settings.ai_strategy_max_candidates)
    if not top_rows and watchlist_rows:
        top_rows = _pick_rows(watchlist_rows, settings.ai_strategy_max_candidates)

    model = (status.get("model") or {}).get("pt1") or {}
    regime = status.get("regime") or {}
    training = status.get("training") or {}
    coverage = status.get("coverage") or {}

    return {
        "generated_at_utc": _utc_now(),
        "scanner_version": "12.4.0",
        "market_open": bool(status.get("market_open")),
        "time_to_close_seconds": status.get("time_to_close_seconds"),
        "last_run_utc": status.get("last_run_utc"),
        "regime": {
            "state": regime.get("state"),
            "reasons": regime.get("reasons"),
            "note": regime.get("note"),
            "suppress_new_signals": bool(regime.get("suppress_new_signals")),
            "live_evaluated": bool(regime.get("live_evaluated")),
            "data_complete": bool(regime.get("data_complete")),
            "market_session": regime.get("market_session"),
            "multiplier": regime.get("multiplier"),
            "prob_cap": regime.get("prob_cap"),
            "last_live_state": regime.get("last_live_state"),
        },
        "model": {
            "trained": bool(model.get("trained")),
            "selection_tier": model.get("selection_tier"),
            "selection_warning": model.get("selection_warning"),
            "touch_tail_validated": bool(model.get("touch_tail_validated")) if model.get("touch_tail_validated") is not None else False,
            "probability_contract": model.get("probability_contract"),
            "adaptive_touch_threshold": model.get("adaptive_touch_threshold"),
            "auc_val": model.get("auc_val"),
            "brier_val": model.get("brier_val"),
        },
        "training": {
            "running": bool(training.get("running")),
            "started_at_utc": training.get("started_at_utc"),
            "finished_at_utc": training.get("finished_at_utc"),
        },
        "coverage": {
            "stage1_candidate_count": coverage.get("stage1_candidate_count"),
            "stage2_scored_count": coverage.get("stage2_scored_count"),
            "symbols_scored_count": coverage.get("symbols_scored_count"),
            "guardrail_stats": coverage.get("guardrail_stats") or {},
        },
        "diagnostics_summary": diagnostics_summary or {},
        "candidates": top_rows,
    }


SYSTEM_PROMPT = """
You are the AI Strategy Interpreter for an intraday S&P 500 scanner.

Your job is to translate the scanner state into a bounded tactical posture, not to make autonomous trading decisions.

Rules you must obey:
1. Respect the scanner's hard controls. Never recommend ignoring a RED regime, NOT_EVALUATED regime, CLOSED session, training-in-progress state, or suppress_new_signals flag.
2. If touch_tail_validated is false, do not act as if the scanner has a validated high-confidence edge. In that case, maximum posture is watchlist_only unless the payload explicitly says otherwise.
3. Prefer stand_aside when evidence is weak, stale, contradictory, or when no strong candidates are present.
4. Use only the payload provided. Do not invent macro facts, symbols, or prices.
5. Keep recommendations concrete, cautious, and operational.
6. Do not output prose outside the schema.
""".strip()


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary_headline": {"type": "string"},
        "market_posture": {"type": "string", "enum": ["stand_aside", "watchlist_only", "pilot_size", "normal_size"]},
        "confidence_label": {"type": "string", "enum": ["low", "moderate", "high"]},
        "model_readiness": {"type": "string", "enum": ["not_ready", "cautious", "ready"]},
        "entry_style": {"type": "string", "enum": ["none", "breakout_only", "pullback_only", "either"]},
        "sizing_note": {"type": "string"},
        "reason_summary": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
        "risk_flags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
        "fail_conditions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
        "top_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "symbol": {"type": "string"},
                    "stance": {"type": "string", "enum": ["avoid", "watchlist", "candidate"]},
                    "rationale": {"type": "string"},
                    "entry_trigger": {"type": "string"},
                    "invalidation": {"type": "string"}
                },
                "required": ["symbol", "stance", "rationale", "entry_trigger", "invalidation"]
            },
            "minItems": 0,
            "maxItems": 8
        },
        "compliance_note": {"type": "string"}
    },
    "required": [
        "summary_headline",
        "market_posture",
        "confidence_label",
        "model_readiness",
        "entry_style",
        "sizing_note",
        "reason_summary",
        "recommended_actions",
        "risk_flags",
        "fail_conditions",
        "top_candidates",
        "compliance_note"
    ]
}


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    texts: List[str] = []
    for item in response_json.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(t for t in texts if t).strip()


def _cap_posture(strategy: Dict[str, Any], max_posture: str, note: str) -> Dict[str, Any]:
    order = {"stand_aside": 0, "watchlist_only": 1, "pilot_size": 2, "normal_size": 3}
    current = str(strategy.get("market_posture") or "stand_aside")
    if order.get(current, 0) > order.get(max_posture, 0):
        strategy["market_posture"] = max_posture
        strategy.setdefault("risk_flags", [])
        if note not in strategy["risk_flags"]:
            strategy["risk_flags"].append(note)
    if max_posture == "stand_aside":
        strategy["entry_style"] = "none"
        strategy["sizing_note"] = "No new entries; stand aside until the scanner state improves."
    elif max_posture == "watchlist_only" and strategy.get("entry_style") == "either":
        strategy["entry_style"] = "none"
    return strategy


def apply_hard_bounds(strategy: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    regime = payload.get("regime") or {}
    model = payload.get("model") or {}
    training = payload.get("training") or {}
    candidates = payload.get("candidates") or []

    posture_limit = None
    posture_note = None
    regime_state = str(regime.get("state") or "NOT_EVALUATED").upper()
    if training.get("running"):
        posture_limit = "stand_aside"
        posture_note = "training_running"
    elif regime_state in {"NOT_EVALUATED", "CLOSED"}:
        posture_limit = "stand_aside"
        posture_note = f"regime_{regime_state.lower()}"
    elif regime.get("suppress_new_signals"):
        posture_limit = "stand_aside"
        posture_note = "regime_suppresses_signals"
    elif not model.get("touch_tail_validated"):
        posture_limit = "watchlist_only"
        posture_note = "tail_unvalidated"
    elif not candidates:
        posture_limit = "watchlist_only"
        posture_note = "no_viable_candidates"

    if posture_limit:
        strategy = _cap_posture(strategy, posture_limit, posture_note)

    if not model.get("trained"):
        strategy["model_readiness"] = "not_ready"
    elif not model.get("touch_tail_validated"):
        strategy["model_readiness"] = "cautious"
    else:
        strategy["model_readiness"] = strategy.get("model_readiness") or "ready"

    limited_symbols = {str(c.get("symbol") or "") for c in candidates}
    cleaned = []
    for row in strategy.get("top_candidates") or []:
        symbol = str(row.get("symbol") or "")
        if symbol and limited_symbols and symbol not in limited_symbols:
            continue
        stance = row.get("stance") or "watchlist"
        if strategy.get("market_posture") == "stand_aside":
            stance = "avoid"
        elif strategy.get("market_posture") == "watchlist_only" and stance == "candidate":
            stance = "watchlist"
        cleaned.append({
            "symbol": symbol,
            "stance": stance,
            "rationale": _clean_text(row.get("rationale"), 220),
            "entry_trigger": _clean_text(row.get("entry_trigger"), 160),
            "invalidation": _clean_text(row.get("invalidation"), 160),
        })
    strategy["top_candidates"] = cleaned[:8]
    strategy["summary_headline"] = _clean_text(strategy.get("summary_headline"), 120)
    strategy["reason_summary"] = _clean_text(strategy.get("reason_summary"), 300)
    strategy["sizing_note"] = _clean_text(strategy.get("sizing_note"), 220)
    strategy["compliance_note"] = _clean_text(strategy.get("compliance_note") or "Advisory only; scanner hard rules still control.", 180)
    strategy["recommended_actions"] = [_clean_text(x, 120) for x in (strategy.get("recommended_actions") or [])][:5]
    strategy["risk_flags"] = [_clean_text(x, 120) for x in (strategy.get("risk_flags") or [])][:6]
    strategy["fail_conditions"] = [_clean_text(x, 120) for x in (strategy.get("fail_conditions") or [])][:6]
    return strategy


def _call_openai(settings: Settings, payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    body = {
        "model": settings.ai_strategy_model,
        "store": False,
        "reasoning": {"effort": settings.ai_strategy_reasoning_effort},
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Interpret this scanner payload and return only JSON matching the schema.\n" + json.dumps(payload, ensure_ascii=False),
                }],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scanner_strategy_v1",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
    }

    resp = requests.post(
        f"{settings.ai_strategy_base_url}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=settings.ai_strategy_timeout_seconds,
    )
    if resp.status_code >= 400:
        try:
            data = resp.json()
            msg = (data.get("error") or {}).get("message") or resp.text
        except Exception:
            msg = resp.text
        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {msg}")
    data = resp.json()
    text = _extract_output_text(data)
    if not text:
        raise RuntimeError("OpenAI API returned no text output.")
    try:
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"OpenAI API returned non-JSON output: {exc}") from exc


def generate_strategy(state: AppState, settings: Settings, diagnostics_summary: Dict[str, Any]) -> AIStrategyStatus:
    payload = build_strategy_payload(state, settings, diagnostics_summary)
    strategy = _call_openai(settings, payload)
    strategy = apply_hard_bounds(strategy, payload)
    now = _utc_now()
    return AIStrategyStatus(
        enabled=settings.ai_strategy_enabled,
        configured=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        model=settings.ai_strategy_model,
        status="ready",
        last_generated_at_utc=now,
        generated_for_run_utc=payload.get("last_run_utc"),
        error=None,
        summary_headline=strategy.get("summary_headline") or "",
        strategy={"generated_at_utc": now, "input_snapshot": payload, "output": strategy},
    )


def load_cached_strategy(settings: Settings, cached_payload: Optional[Dict[str, Any]]) -> AIStrategyStatus:
    base = AIStrategyStatus(
        enabled=settings.ai_strategy_enabled,
        configured=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        model=settings.ai_strategy_model,
        status="idle",
    )
    if not cached_payload:
        return base
    strategy = cached_payload.get("strategy") or {}
    output = strategy.get("output") or {}
    base.status = cached_payload.get("status") or "ready"
    base.last_generated_at_utc = cached_payload.get("last_generated_at_utc")
    base.generated_for_run_utc = cached_payload.get("generated_for_run_utc")
    base.error = cached_payload.get("error")
    base.summary_headline = cached_payload.get("summary_headline") or output.get("summary_headline") or ""
    base.strategy = strategy
    return base
