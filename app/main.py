from __future__ import annotations
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings
from .market import get_market_times, iso
from .diagnostics import DiagnosticJournal
from .persist import load_model_meta, load_training_last, save_training_last
from .scanner import Scanner
from .state import AppState
from .training import run_training

app = FastAPI(title='S&P 500 Prob Scanner', version='9.6.3')
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))
app.mount('/static', StaticFiles(directory=os.path.join(BASE_DIR, 'static')), name='static')
STATE = AppState()
SETTINGS = Settings.from_env()
SCANNER = Scanner(SETTINGS, STATE)
DIAGNOSTICS = DiagnosticJournal(SETTINGS)


def get_settings() -> Settings:
    return SETTINGS


@app.on_event('startup')
def _startup() -> None:
    os.makedirs(os.path.join(SETTINGS.model_dir, 'pt1'), exist_ok=True)
    try:
        SCANNER.load_constituents()
    except Exception as e:
        with STATE.lock:
            STATE.constituents.source = 'fallback'
            STATE.constituents.warning = f'failed to load constituents: {e}'
    try:
        last = load_training_last(SETTINGS.model_dir)
        if last:
            with STATE.lock:
                STATE.training.running = False
                STATE.training.started_at_utc = last.get('started_at_utc')
                STATE.training.finished_at_utc = last.get('finished_at_utc')
                STATE.training.last_result = last.get('last_result')
                STATE.training.last_error = last.get('last_error')
    except Exception:
        pass
    meta, st = load_model_meta(SETTINGS.model_dir, 1)
    if meta and st == 'ok':
        with STATE.lock:
            STATE.model.pt1.trained = True
            STATE.model.pt1.path = os.path.join(SETTINGS.model_dir, 'pt1')
            STATE.model.pt1.auc_val = meta.get('auc_val')
            STATE.model.pt1.brier_val = meta.get('brier_val')
            STATE.model.pt1.calibrator = meta.get('calibrator')
            STATE.model.pt1.class_weight = meta.get('class_weight')
            STATE.model.pt1.alpha = meta.get('alpha')
            STATE.model.pt1.tail_ready_70 = meta.get('tail_ready_70')
            STATE.model.pt1.tail_ready_75 = meta.get('tail_ready_75')
            STATE.model.pt1.selection_tier = meta.get('selection_tier')
            STATE.model.pt1.selection_warning = meta.get('selection_warning')
            STATE.model.pt1.live_prob_cap = meta.get('live_prob_cap')
    SCANNER.start()


@app.get('/health')
def health() -> Dict[str, Any]:
    return {'ok': True, 'service': 'sp500-prob-scanner', 'version': '9.6.3'}


@app.api_route('/', methods=['GET', 'HEAD'], response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/api/status')
def api_status(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    open_utc, close_utc, is_open, ttc = get_market_times(now, settings.timezone)
    with STATE.lock:
        STATE.market.market_open = is_open
        STATE.market.time_to_close_seconds = ttc
        STATE.market.market_open_time = iso(open_utc)
        STATE.market.market_close_time = iso(close_utc)
        STATE.alpaca.feed = settings.normalized_feed()
        if STATE.constituents.count and not STATE.coverage.universe_count:
            STATE.coverage.universe_count = STATE.constituents.count
        if STATE.constituents.count and not STATE.coverage.symbols_requested_count:
            STATE.coverage.symbols_requested_count = STATE.constituents.count + 1
    snap = STATE.snapshot_status()
    snap['demo_mode'] = settings.demo_mode
    snap['scan_interval_minutes'] = settings.scan_interval_minutes
    snap['timezone'] = settings.timezone
    snap['model_dir'] = settings.model_dir
    snap['min_bars_5m'] = settings.min_bars_5m
    snap['tod_rvol'] = {'lookback_days': settings.tod_rvol_lookback_days, 'min_days': settings.tod_rvol_min_days}
    snap['liq_thresholds'] = {
        'rolling_bars': settings.liq_rolling_bars,
        'dvol_min_usd': settings.liq_dvol_min_usd,
        'range_pct_max': settings.liq_range_pct_max,
        'wick_atr_max': settings.liq_wick_atr_max,
    }
    snap['stage1'] = {
        'candidate_cap': settings.stage1_candidate_cap,
        'min_score': settings.stage1_min_score,
        'min_minutes_since_open': settings.stage1_min_minutes_since_open,
        'min_minutes_to_close': settings.stage1_min_minutes_to_close,
        'min_rvol': settings.stage1_min_rvol,
        'min_dollar_volume_mult': settings.stage1_min_dollar_volume_mult,
    }
    snap['guardrail_thresholds'] = {
        'blocked_prob_cap': settings.blocked_prob_cap,
        'event_prob_cap': settings.event_prob_cap,
        'uncertainty_prob_cap': settings.uncertainty_prob_cap,
        'downside_prob_cap_high': settings.downside_prob_cap_high,
        'downside_prob_cap_medium': settings.downside_prob_cap_medium,
        'downside_high_threshold': settings.downside_high_threshold,
        'downside_medium_threshold': settings.downside_medium_threshold,
    }
    snap['training_selection'] = {
        'calib_min_bucket_samples': settings.calib_min_bucket_samples,
        'enet_c_values': settings.enet_c_values,
        'enet_l1_values': settings.enet_l1_values,
        'prior_alpha_values': settings.prior_alpha_values,
        'selection_min_count_70': settings.selection_min_count_70,
        'selection_min_count_75': settings.selection_min_count_75,
        'selection_min_precision_70': settings.selection_min_precision_70,
        'selection_min_precision_75': settings.selection_min_precision_75,
        'tail_not_ready_prob_cap': settings.tail_not_ready_prob_cap,
    }

    snap['diagnostics'] = DIAGNOSTICS.load_latest_summary()
    return snap


@app.get('/api/scores')
def api_scores() -> Dict[str, Any]:
    return STATE.snapshot_scores()


@app.get('/api/diagnostics')
def api_diagnostics(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.load_day_for_api(trade_day=trade_date or None)


@app.get('/api/training/status')
def training_status() -> Dict[str, Any]:
    with STATE.lock:
        return STATE.training.__dict__.copy()


@app.get('/api/debug/coverage')
def debug_coverage(password: str = Query(''), settings: Settings = Depends(get_settings)) -> JSONResponse:
    gate = settings.debug_gate_password()
    if not gate:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'No ADMIN_PASSWORD/DEBUG_PASSWORD configured.'})
    if password != gate:
        return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid password.'})
    with STATE.lock:
        items = [s.__dict__ for s in STATE.skipped[:200]]
    return JSONResponse(content={'ok': True, 'count': len(items), 'items': items})


def _training_thread(settings: Settings) -> None:
    try:
        if not SCANNER.constituents:
            SCANNER.load_constituents()
        constituents = list(SCANNER.constituents)
        if settings.train_max_symbols and settings.train_max_symbols > 0:
            constituents = constituents[:settings.train_max_symbols]
        symbols = [c.symbol for c in constituents]
        sector_map = {c.symbol: c.sector for c in constituents}
        with STATE.lock:
            STATE.training.running = True
            STATE.training.started_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_error = None
            STATE.training.last_result = None
            STATE.training.finished_at_utc = None
        res = run_training(settings, symbols, sector_map)
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_result = res
            STATE.training.last_error = None
            pt1 = res.get('pt1', {})
            STATE.model.pt1.trained = True
            STATE.model.pt1.path = os.path.join(settings.model_dir, 'pt1')
            STATE.model.pt1.auc_val = pt1.get('auc_val')
            STATE.model.pt1.brier_val = pt1.get('brier_val')
            STATE.model.pt1.calibrator = pt1.get('calibrator')
            STATE.model.pt1.class_weight = pt1.get('class_weight')
            STATE.model.pt1.alpha = pt1.get('alpha')
            STATE.model.pt1.tail_ready_70 = pt1.get('tail_ready_70')
            STATE.model.pt1.tail_ready_75 = pt1.get('tail_ready_75')
            STATE.model.pt1.selection_tier = pt1.get('selection_tier')
            STATE.model.pt1.selection_warning = pt1.get('selection_warning')
            STATE.model.pt1.live_prob_cap = pt1.get('live_prob_cap')
        try:
            save_training_last(settings.model_dir, {
                'started_at_utc': STATE.training.started_at_utc,
                'finished_at_utc': STATE.training.finished_at_utc,
                'last_result': res,
                'last_error': None,
            })
        except Exception:
            pass
    except Exception as e:
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_error = str(e)
            STATE.training.last_result = None
        try:
            save_training_last(settings.model_dir, {
                'started_at_utc': STATE.training.started_at_utc,
                'finished_at_utc': STATE.training.finished_at_utc,
                'last_result': None,
                'last_error': str(e),
            })
        except Exception:
            pass


@app.post('/train')
def train(admin_password: str = Form(''), settings: Settings = Depends(get_settings)) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'ADMIN_PASSWORD is not set on the server.'})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid admin password.'})
    if settings.demo_mode:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'Training requires DEMO_MODE=false.'})
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'Training requires ALPACA_API_KEY and ALPACA_API_SECRET.'})
    with STATE.lock:
        if STATE.training.running:
            return JSONResponse(status_code=409, content={'ok': False, 'error': 'Training is already running.'})
    t = threading.Thread(target=_training_thread, args=(settings,), daemon=True)
    t.start()
    return JSONResponse(content={'ok': True, 'message': 'Training started (pt1 only).'})
