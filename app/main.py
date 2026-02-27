from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings
from .scanner import Scanner
from .state import AppState
from .training import run_training
from .market import get_market_times, iso

app = FastAPI(title="S&P 500 Prob Scanner", version="1.0.0")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

STATE = AppState()
SETTINGS = Settings()
SCANNER = Scanner(SETTINGS, STATE)

def get_settings() -> Settings:
    return SETTINGS

@app.on_event("startup")
def _startup() -> None:
    # Ensure model dirs exist
    os.makedirs(os.path.join(SETTINGS.model_dir, "pt1"), exist_ok=True)
    os.makedirs(os.path.join(SETTINGS.model_dir, "pt2"), exist_ok=True)

    # Load cached last scores if any
    try:
        cache_path = os.path.join(os.path.dirname(SETTINGS.model_dir), "last_scores.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            with STATE.lock:
                STATE.last_run_utc = payload.get("last_run_utc")
                # Keep rows in memory only if schema matches
                rows = payload.get("rows") or []
                # delay parsing into dataclasses; api/scores can use raw rows if needed
                # We'll just store parsed rows through scanner helper on first scan. For startup, keep empty.
    except Exception:
        pass

    # Load constituents best-effort
    try:
        SCANNER.load_constituents()
    except Exception as e:
        with STATE.lock:
            STATE.constituents.source = "fallback"
            STATE.constituents.warning = f"failed to load constituents: {e}"

    # Start scheduler
    SCANNER.start()

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sp500-prob-scanner"}

@app.api_route("/", methods=["GET","HEAD"], response_class=HTMLResponse)
def dashboard(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
def api_status(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    # Update market status on-demand (so status is correct even if scheduler is disabled)
    now = datetime.now(timezone.utc)
    open_utc, close_utc, is_open, ttc = get_market_times(now, settings.timezone)
    with STATE.lock:
        STATE.market.market_open = is_open
        STATE.market.time_to_close_seconds = ttc
        STATE.market.market_open_time = iso(open_utc)
        STATE.market.market_close_time = iso(close_utc)
    snap = STATE.snapshot_status()
    snap["demo_mode"] = settings.demo_mode
    snap["scan_interval_minutes"] = settings.scan_interval_minutes
    snap["timezone"] = settings.timezone
    snap["model_dir"] = settings.model_dir
    return snap

@app.get("/api/scores")
def api_scores() -> Dict[str, Any]:
    return STATE.snapshot_scores()

@app.get("/api/training/status")
def training_status() -> Dict[str, Any]:
    with STATE.lock:
        return STATE.training.__dict__.copy()

def _training_thread(settings: Settings, password: str) -> None:
    # Resolve symbols for training: use constituents limited by TRAIN_MAX_SYMBOLS
    try:
        symbols = [c.symbol for c in SCANNER.constituents]
        if not symbols:
            SCANNER.load_constituents()
            symbols = [c.symbol for c in SCANNER.constituents]
        symbols = symbols[: max(1, settings.train_max_symbols)]
        with STATE.lock:
            STATE.training.running = True
            STATE.training.started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            STATE.training.last_error = None
            STATE.training.last_result = None
            STATE.training.finished_at_utc = None

        res = run_training(settings, STATE, symbols=symbols)

        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            STATE.training.last_result = res
            STATE.training.last_error = None
    except Exception as e:
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            STATE.training.last_error = str(e)
            STATE.training.last_result = None

@app.post("/train")
def train(
    admin_password: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={"ok": False, "error": "ADMIN_PASSWORD is not set on the server."})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={"ok": False, "error": "Invalid admin password."})

    with STATE.lock:
        if STATE.training.running:
            return JSONResponse(status_code=409, content={"ok": False, "error": "Training is already running."})

    # Preflight: require live keys and DEMO_MODE=false
    if settings.demo_mode:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Training requires DEMO_MODE=false."})
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return JSONResponse(status_code=400, content={"ok": False, "error": "Training requires ALPACA_API_KEY and ALPACA_API_SECRET."})

    # Start background thread
    t = threading.Thread(target=_training_thread, args=(settings, admin_password), daemon=True)
    t.start()
    return JSONResponse(content={"ok": True, "message": "Training started (1% and 2%).", "status_endpoint": "/api/training/status"})
