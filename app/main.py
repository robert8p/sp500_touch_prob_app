from __future__ import annotations
import io
import json
import os
import threading
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings
from .market import get_market_times, iso
from .diagnostics import DiagnosticJournal
from .persist import load_model_meta, load_training_last, save_training_last
from .scanner import Scanner
from .state import AppState
from .training import run_training

app = FastAPI(title='S&P 500 Prob Scanner', version='13.3.0')
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
        # Probe actual runtime state: can we load Model B?
        from .modeling import try_load_path_model
        path_model, path_st = try_load_path_model(SETTINGS.model_dir)
        runtime_source = 'trained' if path_model is not None else 'trained_no_path'
        runtime_b_method = (meta.get('model_b') or {}).get('method') if path_model is not None else None
        with STATE.lock:
            STATE.model.pt1.trained = True
            STATE.model.pt1.path = os.path.join(SETTINGS.model_dir, 'pt1')
            STATE.model.pt1.auc_val = meta.get('auc_val')
            STATE.model.pt1.brier_val = meta.get('brier_val')
            STATE.model.pt1.calibrator = meta.get('calibrator')
            STATE.model.pt1.class_weight = meta.get('class_weight')
            STATE.model.pt1.alpha = meta.get('alpha')
            STATE.model.pt1.touch_tail_validated = meta.get('touch_tail_validated')
            STATE.model.pt1.decision_tail_validated = meta.get('decision_tail_validated')
            STATE.model.pt1.selection_tier = meta.get('selection_tier')
            STATE.model.pt1.selection_warning = meta.get('selection_warning')
            STATE.model.pt1.model_b_method = runtime_b_method
            STATE.model.pt1.probability_contract = meta.get('probability_contract')
            STATE.model.pt1.model_source = runtime_source
            ttm = meta.get('touch_tail_metrics') or {}
            dtm = meta.get('decision_tail_metrics') or {}
            STATE.model.pt1.adaptive_touch_threshold = ttm.get('adaptive_threshold')
            STATE.model.pt1.adaptive_decision_threshold = dtm.get('adaptive_threshold')
    now = datetime.now(timezone.utc)
    open_utc, close_utc, is_open, _ = get_market_times(now, SETTINGS.timezone)
    try:
        SCANNER._publish_regime(SCANNER.regime_controller.bootstrap_status(now, is_open))
    except Exception:
        pass
    SCANNER.start()


@app.get('/health')
def health() -> Dict[str, Any]:
    return {'ok': True, 'service': 'sp500-prob-scanner', 'version': '13.3.0'}


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
    snap = STATE.snapshot_status()
    snap['demo_mode'] = settings.demo_mode
    snap['scan_interval_minutes'] = settings.scan_interval_minutes
    snap['timezone'] = settings.timezone
    snap['model_dir'] = settings.model_dir
    snap['min_bars_5m'] = settings.min_bars_5m
    snap['tod_rvol'] = {'lookback_days': settings.tod_rvol_lookback_days, 'min_days': settings.tod_rvol_min_days}
    snap['liq_thresholds'] = {'rolling_bars': settings.liq_rolling_bars, 'dvol_min_usd': settings.liq_dvol_min_usd, 'range_pct_max': settings.liq_range_pct_max, 'wick_atr_max': settings.liq_wick_atr_max}
    snap['stage1'] = {
        'candidate_cap': settings.stage1_candidate_cap, 'min_score': settings.stage1_min_score,
        'min_minutes_since_open': settings.stage1_min_minutes_since_open, 'min_minutes_to_close': settings.stage1_min_minutes_to_close,
        'min_rvol_early': settings.stage1_min_rvol, 'min_rvol_midday': settings.stage1_midday_min_rvol, 'min_rvol_late': settings.stage1_late_min_rvol,
        'strong_rvol_minutes': settings.stage1_strong_rvol_minutes, 'min_dollar_volume_mult': settings.stage1_min_dollar_volume_mult,
        'min_rel_spy_30m': settings.stage1_min_rel_spy_30m, 'min_rel_spy_5m': settings.stage1_min_rel_spy_5m,
        'min_dist_pct_to_vwap': settings.stage1_min_dist_pct_to_vwap,
        'strong_override_score': settings.stage1_strong_override_score,
    }
    snap['surfacing'] = {
        'cooldown_minutes': settings.surfacing_cooldown_minutes,
        'min_touch_delta': settings.surfacing_min_touch_delta,
        'min_path_delta': settings.surfacing_min_path_delta,
        'unvalidated_messy_suppress': settings.unvalidated_messy_suppress,
        'unvalidated_messy_risk_flags': settings.unvalidated_messy_risk_flags,
    }
    snap['watchlist_rescue'] = {
        'enabled': settings.watchlist_rescue_enabled,
        'max_rows': settings.watchlist_rescue_max_rows,
        'min_stage1_score': settings.watchlist_rescue_min_stage1_score,
        'touch_frac': settings.watchlist_rescue_touch_frac,
        'combined_frac': settings.watchlist_rescue_combined_frac,
        'path_min': settings.watchlist_rescue_path_min,
        'allow_medium_downside': settings.watchlist_rescue_allow_medium_downside,
    }
    snap['regime_config'] = {
        'enabled': settings.regime_enabled,
        'oil_proxy': settings.regime_oil_proxy,
        'vol_proxy': settings.regime_vol_proxy,
        'safe_haven_proxy': settings.regime_safe_haven_proxy,
        'energy_proxy': settings.regime_energy_proxy,
        'amber_multiplier': settings.regime_amber_multiplier,
        'red_multiplier': settings.regime_red_multiplier,
        'amber_prob_cap': settings.regime_amber_prob_cap,
        'red_prob_cap': settings.regime_red_prob_cap,
        'amber_touch_threshold_mult': settings.regime_amber_touch_threshold_mult,
        'red_touch_threshold_mult': settings.regime_red_touch_threshold_mult,
        'amber_path_floor_add': settings.regime_amber_path_floor_add,
        'red_path_floor_add': settings.regime_red_path_floor_add,
    }
    snap['guardrail_thresholds'] = {
        'blocked_prob_cap': settings.blocked_prob_cap, 'event_prob_cap': settings.event_prob_cap, 'uncertainty_prob_cap': settings.uncertainty_prob_cap,
        'downside_prob_cap_high': settings.downside_prob_cap_high, 'downside_prob_cap_medium': settings.downside_prob_cap_medium,
        'downside_high_threshold': settings.downside_high_threshold, 'downside_medium_threshold': settings.downside_medium_threshold,
    }
    snap['training_selection'] = {
        'target_name': 'decomposed_touch_x_path_1pct',
        'strict_touch_mae_threshold': settings.strict_touch_mae_threshold,
        'worthy_close_vs_scan_min': settings.worthy_close_vs_scan_min,
        'model_b_enabled': settings.model_b_enabled,
        'touch_tail_threshold_mult': settings.touch_tail_threshold_mult,
        'touch_tail_min_count': settings.touch_tail_min_count,
        'touch_tail_min_lift': settings.touch_tail_min_lift,
        'path_quality_action_min': settings.path_quality_action_min,
    }
    snap['specialist_config'] = {
        'enabled': settings.specialist_families_enabled,
        'rerank_prob1_weight': settings.rerank_prob1_weight,
        'rerank_touch_weight': settings.rerank_touch_weight,
        'rerank_path_weight': settings.rerank_path_weight,
        'rerank_stage1_weight': settings.rerank_stage1_weight,
        'rerank_family_bonus_weight': settings.rerank_family_bonus_weight,
        'rerank_downside_weight': settings.rerank_downside_weight,
        'rerank_uncertainty_weight': settings.rerank_uncertainty_weight,
        'rerank_actionable_bonus': settings.rerank_actionable_bonus,
        'rerank_candidate_bonus': settings.rerank_candidate_bonus,
        'rerank_watchlist_penalty': settings.rerank_watchlist_penalty,
        'min_profile_count': settings.specialist_min_profile_count,
        'suppress_below_lift': settings.specialist_suppress_below_lift,
        'promote_above_lift': settings.specialist_promote_above_lift,
        'threshold_loosen_mult': settings.specialist_threshold_loosen_mult,
        'threshold_tighten_mult': settings.specialist_threshold_tighten_mult,
        'calib_bin_count': settings.specialist_calib_bin_count,
        'context_min_count': settings.specialist_context_min_count,
        'context_min_lift_delta': settings.specialist_context_min_lift_delta,
    }
    snap['diagnostics'] = DIAGNOSTICS.load_latest_summary()
    return snap



@app.get('/api/scores')
def api_scores() -> Dict[str, Any]:
    return STATE.snapshot_scores()


@app.get('/api/regime')
def api_regime() -> Dict[str, Any]:
    with STATE.lock:
        return STATE.regime.__dict__.copy()


@app.post('/api/regime/override')
def api_regime_override(
    admin_password: str = Form(''),
    state: str = Form(''),
    reason: str = Form(''),
    duration_minutes: int = Form(0),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'ADMIN_PASSWORD is not set.'})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid admin password.'})
    state_norm = (state or '').strip().upper()
    if state_norm not in {'GREEN', 'AMBER', 'RED'}:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'state must be GREEN, AMBER, or RED.'})
    payload = SCANNER.regime_controller.set_override(state=state_norm, reason=reason or f'MANUAL_{state_norm}', duration_minutes=duration_minutes)
    active = SCANNER.regime_controller.get_active_override()
    if active is not None:
        now = datetime.now(timezone.utc)
        _, _, is_open, _ = get_market_times(now, settings.timezone)
        SCANNER._publish_regime(active.to_decision(settings, market_session='LIVE' if is_open else 'CLOSED', evaluated_at_utc=now.isoformat().replace('+00:00', 'Z')))
    return JSONResponse(content={'ok': True, 'override': payload})


@app.post('/api/regime/override/clear')
def api_regime_override_clear(admin_password: str = Form(''), settings: Settings = Depends(get_settings)) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'ADMIN_PASSWORD is not set.'})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid admin password.'})
    SCANNER.regime_controller.clear_override()
    STATE.set_regime(STATE.regime.__class__())
    return JSONResponse(content={'ok': True})


@app.get('/api/diagnostics')
def api_diagnostics(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.load_day_for_api(trade_day=trade_date or None)


@app.get('/api/eod-review')
def api_eod_review(trade_date: str = Query(''), refresh: bool = Query(False)) -> Dict[str, Any]:
    if not SCANNER.constituents:
        try:
            SCANNER.load_constituents()
        except Exception:
            pass
    client = SCANNER._make_client()
    symbols = [c.symbol for c in SCANNER.constituents]
    return DIAGNOSTICS.build_eod_review(client=client, symbols=symbols, trade_day=trade_date or None, now_utc=datetime.now(timezone.utc), refresh=bool(refresh))


@app.get('/api/scan-history')
def api_scan_history(trade_date: str = Query(''), symbol: str = Query(''), include_unsurfaced: bool = Query(True)) -> Dict[str, Any]:
    return DIAGNOSTICS.build_scan_history(trade_day=trade_date or None, symbol=symbol or None, include_unsurfaced=bool(include_unsurfaced))


@app.get('/api/blocker-attribution')
def api_blocker_attribution(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_blocker_attribution(trade_day=trade_date or None)


@app.get('/api/promotion-attribution')
def api_promotion_attribution(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_promotion_attribution(trade_day=trade_date or None)


@app.get('/api/calibration-review')
def api_calibration_review(
    trade_date: str = Query(''),
    metric: str = Query('prob_touch'),
    bucket_mode: str = Query('decile'),
    basis: str = Query('all_rows'),
) -> Dict[str, Any]:
    return DIAGNOSTICS.build_calibration_review(trade_day=trade_date or None, metric=metric, bucket_mode=bucket_mode, basis=basis)


@app.get('/api/stage1-review')
def api_stage1_review(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_stage1_review(trade_day=trade_date or None)


@app.get('/api/guardrail-review')
def api_guardrail_review(trade_date: str = Query(''), symbol: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_guardrail_review(trade_day=trade_date or None, symbol=symbol or None)


@app.get('/api/threshold-review')
def api_threshold_review(
    trade_date: str = Query(''),
    touch_threshold: float | None = Query(None),
    path_min: float | None = Query(None),
    ignore_downside: bool = Query(False),
    ignore_uncertainty: bool = Query(False),
    ignore_event: bool = Query(False),
    include_tail_unvalidated: bool = Query(True),
) -> Dict[str, Any]:
    return DIAGNOSTICS.build_threshold_review(
        trade_day=trade_date or None,
        touch_threshold=touch_threshold,
        path_min=path_min,
        ignore_downside=bool(ignore_downside),
        ignore_uncertainty=bool(ignore_uncertainty),
        ignore_event=bool(ignore_event),
        include_tail_unvalidated=bool(include_tail_unvalidated),
    )


@app.get('/api/review-slices')
def api_review_slices(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_review_slices(trade_day=trade_date or None)


@app.get('/api/review-export')
def api_review_export(trade_date: str = Query('')) -> Dict[str, Any]:
    return DIAGNOSTICS.build_review_export(trade_day=trade_date or None)


def _resolved_review_trade_day(explicit_trade_date: str = '') -> str | None:
    if explicit_trade_date:
        return explicit_trade_date
    latest = DIAGNOSTICS.load_latest_summary() or {}
    return latest.get('trade_date')


def _build_review_export_payload(trade_day: str | None) -> Dict[str, Any]:
    payload = DIAGNOSTICS.build_review_export(trade_day=trade_day)
    payload['status'] = api_status(SETTINGS)
    payload['near_misses'] = api_near_misses()
    payload['eod_review'] = api_eod_review(trade_date=trade_day or '', refresh=False)
    payload['threshold_review'] = DIAGNOSTICS.build_threshold_review(trade_day=trade_day)
    payload['watchlist_rescue'] = api_watchlist_rescue()
    return payload


def _add_txt_to_zip(zf: zipfile.ZipFile, folder: str, filename: str, payload: Dict[str, Any]) -> None:
    pretty = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
    zf.writestr(f'{folder}/{filename}', pretty + '\n')


@app.get('/api/review-export/download')
def api_review_export_download(trade_date: str = Query('')) -> StreamingResponse:
    trade_day = _resolved_review_trade_day(trade_date)
    payload = _build_review_export_payload(trade_day)
    resolved_trade_day = payload.get('trade_date') or payload.get('status', {}).get('diagnostics', {}).get('trade_date') or trade_day or datetime.now(timezone.utc).date().isoformat()
    folder = f'post_close_review_{resolved_trade_day}'
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            'version': 1,
            'generated_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'trade_date': resolved_trade_day,
            'files': [
                'manifest.txt',
                f'{resolved_trade_day}_status.txt',
                f'{resolved_trade_day}_diagnostics.txt',
                f'{resolved_trade_day}_near_misses.txt',
                f'{resolved_trade_day}_eod_review.txt',
                f'{resolved_trade_day}_scan_history.txt',
                f'{resolved_trade_day}_blocker_attribution.txt',
                f'{resolved_trade_day}_promotion_attribution.txt',
                f'{resolved_trade_day}_calibration_prob_touch.txt',
                f'{resolved_trade_day}_calibration_prob_1.txt',
                f'{resolved_trade_day}_calibration_prob_path.txt',
                f'{resolved_trade_day}_stage1_review.txt',
                f'{resolved_trade_day}_guardrail_review.txt',
                f'{resolved_trade_day}_threshold_review.txt',
                f'{resolved_trade_day}_review_slices.txt',
                f'{resolved_trade_day}_watchlist_rescue.txt',
            ],
        }
        _add_txt_to_zip(zf, folder, 'manifest.txt', manifest)
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_status.txt', payload.get('status', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_diagnostics.txt', payload.get('diagnostics', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_near_misses.txt', payload.get('near_misses', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_eod_review.txt', payload.get('eod_review', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_scan_history.txt', payload.get('scan_history', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_blocker_attribution.txt', payload.get('blocker_attribution', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_promotion_attribution.txt', payload.get('promotion_attribution', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_calibration_prob_touch.txt', payload.get('calibration_prob_touch', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_calibration_prob_1.txt', payload.get('calibration_prob_1', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_calibration_prob_path.txt', payload.get('calibration_prob_path', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_stage1_review.txt', payload.get('stage1_review', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_guardrail_review.txt', payload.get('guardrail_review', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_threshold_review.txt', payload.get('threshold_review', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_review_slices.txt', payload.get('review_slices', {}))
        _add_txt_to_zip(zf, folder, f'{resolved_trade_day}_watchlist_rescue.txt', payload.get('watchlist_rescue', {}))
    mem.seek(0)
    filename = f'post_close_review_{resolved_trade_day}.zip'
    headers = {'Content-Disposition': f'attachment; filename={filename}'}
    return StreamingResponse(mem, media_type='application/zip', headers=headers)


@app.get('/api/watchlist-rescue')
def api_watchlist_rescue() -> Dict[str, Any]:
    snap = STATE.snapshot_scores()
    return {'last_run_utc': snap.get('last_run_utc'), 'rows': snap.get('watchlist_rescue_rows', []), 'last_error': snap.get('last_error')}


@app.get('/api/near-misses')
def api_near_misses() -> Dict[str, Any]:
    """Stage 1 near-miss symbols."""
    with STATE.lock:
        return {'near_misses': [nm.__dict__ for nm in STATE.near_misses]}


@app.get('/api/training/status')
def training_status() -> Dict[str, Any]:
    with STATE.lock:
        return STATE.training.__dict__.copy()


@app.get('/api/debug/coverage')
def debug_coverage(password: str = Query(''), settings: Settings = Depends(get_settings)) -> JSONResponse:
    gate = settings.debug_gate_password()
    if not gate: return JSONResponse(status_code=400, content={'ok': False, 'error': 'No password configured.'})
    if password != gate: return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid password.'})
    with STATE.lock:
        items = [s.__dict__ for s in STATE.skipped[:200]]
    return JSONResponse(content={'ok': True, 'count': len(items), 'items': items})


def _training_thread(settings: Settings) -> None:
    try:
        if not SCANNER.constituents: SCANNER.load_constituents()
        constituents = list(SCANNER.constituents)
        if settings.train_max_symbols and settings.train_max_symbols > 0: constituents = constituents[:settings.train_max_symbols]
        symbols = [c.symbol for c in constituents]
        sector_map = {c.symbol: c.sector for c in constituents}
        with STATE.lock:
            STATE.training.running = True
            STATE.training.started_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_error = None; STATE.training.last_result = None; STATE.training.finished_at_utc = None
        res = run_training(settings, symbols, sector_map)
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_result = res; STATE.training.last_error = None
            pt1 = res.get('pt1', {})
            STATE.model.pt1.trained = True
            STATE.model.pt1.path = os.path.join(settings.model_dir, 'pt1')
            STATE.model.pt1.auc_val = pt1.get('auc_val')
            STATE.model.pt1.brier_val = pt1.get('brier_val')
            STATE.model.pt1.calibrator = pt1.get('calibrator')
            STATE.model.pt1.class_weight = pt1.get('class_weight')
            STATE.model.pt1.alpha = pt1.get('alpha')
            STATE.model.pt1.touch_tail_validated = pt1.get('touch_tail_validated')
            STATE.model.pt1.decision_tail_validated = pt1.get('decision_tail_validated')
            ttm = pt1.get('touch_tail_metrics') or {}
            dtm = pt1.get('decision_tail_metrics') or {}
            STATE.model.pt1.adaptive_touch_threshold = ttm.get('adaptive_threshold')
            STATE.model.pt1.adaptive_decision_threshold = dtm.get('adaptive_threshold')
            STATE.model.pt1.selection_tier = pt1.get('selection_tier')
            STATE.model.pt1.selection_warning = pt1.get('selection_warning')
            STATE.model.pt1.model_b_method = (pt1.get('model_b') or {}).get('method')
            STATE.model.pt1.probability_contract = pt1.get('probability_contract') or 'strict_calibrated_decomposed'
            # Re-probe actual runtime Model B availability
            from .modeling import try_load_path_model
            _pb, _ps = try_load_path_model(settings.model_dir)
            STATE.model.pt1.model_source = 'trained' if _pb is not None else 'trained_no_path'
            if _pb is None:
                STATE.model.pt1.model_b_method = None
        try:
            save_training_last(settings.model_dir, {'started_at_utc': STATE.training.started_at_utc, 'finished_at_utc': STATE.training.finished_at_utc, 'last_result': res, 'last_error': None})
        except Exception: pass
    except Exception as e:
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            STATE.training.last_error = str(e); STATE.training.last_result = None
        try:
            save_training_last(settings.model_dir, {'started_at_utc': STATE.training.started_at_utc, 'finished_at_utc': STATE.training.finished_at_utc, 'last_result': None, 'last_error': str(e)})
        except Exception: pass


@app.post('/train')
def train(admin_password: str = Form(''), settings: Settings = Depends(get_settings)) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'ADMIN_PASSWORD is not set.'})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={'ok': False, 'error': 'Invalid admin password.'})
    if settings.demo_mode:
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'Training requires DEMO_MODE=false.'})
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return JSONResponse(status_code=400, content={'ok': False, 'error': 'Training requires Alpaca keys.'})
    with STATE.lock:
        if STATE.training.running:
            return JSONResponse(status_code=409, content={'ok': False, 'error': 'Training is already running.'})
    t = threading.Thread(target=_training_thread, args=(settings,), daemon=True)
    t.start()
    return JSONResponse(content={'ok': True, 'message': 'Training started (v12.2 tail-aware weighted decomposed model with strict calibration and geopolitical regime overlay).'})
