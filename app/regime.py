from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


LIVE_REGIME_STATES = {'GREEN', 'AMBER', 'RED'}


@dataclass
class RegimeDecision:
    state: str = 'NOT_EVALUATED'
    source: str = 'auto'
    reasons: List[str] = field(default_factory=list)
    multiplier: float = 1.0
    prob_cap: Optional[float] = None
    touch_threshold_mult: float = 1.0
    path_floor_add: float = 0.0
    suppress_new_signals: bool = False
    cooldown_until_utc: Optional[str] = None
    is_manual_override: bool = False
    metrics: Dict[str, Dict[str, float | int | bool | str | None]] = field(default_factory=dict)
    note: str = ''
    evaluated_at_utc: Optional[str] = None
    last_live_state: Optional[str] = None
    market_session: Optional[str] = None
    live_evaluated: bool = False
    data_complete: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ManualRegimeOverride:
    state: str
    reason: str
    source: str = 'manual'
    expires_at_utc: Optional[str] = None
    created_at_utc: Optional[str] = None

    def is_active(self, now_utc: datetime) -> bool:
        if not self.expires_at_utc:
            return True
        try:
            return _parse_iso(self.expires_at_utc) > now_utc
        except Exception:
            return False

    def to_decision(self, settings, *, market_session: str = 'LIVE', evaluated_at_utc: Optional[str] = None) -> RegimeDecision:
        state = (self.state or 'GREEN').upper()
        decision = _decision_from_state(settings, state)
        decision.source = self.source or 'manual'
        decision.is_manual_override = True
        decision.reasons = [self.reason or f'MANUAL_{state}']
        decision.cooldown_until_utc = self.expires_at_utc
        decision.note = 'Manual override active'
        decision.market_session = market_session
        decision.evaluated_at_utc = evaluated_at_utc or _iso(_utcnow())
        decision.live_evaluated = False
        decision.data_complete = False
        decision.last_live_state = state if state in LIVE_REGIME_STATES else None
        return decision


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> datetime:
    value = str(ts or '')
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    return datetime.fromisoformat(value)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _current_session_bars(all_bars: List[dict], open_utc: datetime, close_utc: datetime) -> List[dict]:
    rows: List[dict] = []
    for bar in all_bars or []:
        try:
            ts = _parse_iso(bar['t'])
        except Exception:
            continue
        if open_utc <= ts <= close_utc:
            rows.append(bar)
    return rows


def _ret_over_bars(cur: List[dict], intervals: int) -> Optional[float]:
    if not cur or len(cur) < max(2, intervals + 1):
        return None
    closes: List[float] = []
    for bar in cur:
        try:
            closes.append(float(bar.get('c') or 0.0))
        except Exception:
            closes.append(0.0)
    closes = [x for x in closes if x > 0]
    if len(closes) < max(2, intervals + 1):
        return None
    ref = closes[-1 - max(1, intervals)]
    last = closes[-1]
    if ref <= 0:
        return None
    return float(last / ref - 1.0)


def _metric_snapshot(symbol: str, all_bars: List[dict], open_utc: datetime, close_utc: datetime) -> Dict[str, float | int | bool | str | None]:
    cur = _current_session_bars(all_bars, open_utc, close_utc)
    ret_1h = _ret_over_bars(cur, 12)
    ret_30m = _ret_over_bars(cur, 6)
    ret_since_open = None
    bars_used = len(cur)
    last_bar_utc = None
    if cur:
        try:
            first_close = float(cur[0].get('c') or 0.0)
            last_close = float(cur[-1].get('c') or 0.0)
            last_bar_utc = cur[-1].get('t')
            if first_close > 0:
                ret_since_open = float(last_close / first_close - 1.0)
        except Exception:
            ret_since_open = None
    return {
        'symbol': symbol,
        'bars_used': bars_used,
        'last_bar_utc': last_bar_utc,
        'ret_30m': ret_30m,
        'ret_1h': ret_1h,
        'ret_since_open': ret_since_open,
        'complete_1h': ret_1h is not None,
    }


def _decision_from_state(settings, state: str) -> RegimeDecision:
    state = (state or 'GREEN').upper()
    if state == 'RED':
        return RegimeDecision(
            state='RED',
            multiplier=float(settings.regime_red_multiplier),
            prob_cap=float(settings.regime_red_prob_cap),
            touch_threshold_mult=float(settings.regime_red_touch_threshold_mult),
            path_floor_add=float(settings.regime_red_path_floor_add),
            suppress_new_signals=bool(settings.regime_red_suppress_signals),
        )
    if state == 'AMBER':
        return RegimeDecision(
            state='AMBER',
            multiplier=float(settings.regime_amber_multiplier),
            prob_cap=float(settings.regime_amber_prob_cap),
            touch_threshold_mult=float(settings.regime_amber_touch_threshold_mult),
            path_floor_add=float(settings.regime_amber_path_floor_add),
            suppress_new_signals=bool(settings.regime_amber_suppress_signals),
        )
    return RegimeDecision(state='GREEN')


class RegimeController:
    def __init__(self, settings):
        self.settings = settings
        self.override_path = os.path.join(settings.model_dir, 'regime_override.json')
        self.last_path = os.path.join(settings.model_dir, 'regime_last.json')

    def _read_override(self) -> Optional[ManualRegimeOverride]:
        try:
            if not os.path.exists(self.override_path):
                return None
            with open(self.override_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh) or {}
            ov = ManualRegimeOverride(
                state=str(payload.get('state') or 'GREEN').upper(),
                reason=str(payload.get('reason') or ''),
                source=str(payload.get('source') or 'manual'),
                expires_at_utc=payload.get('expires_at_utc'),
                created_at_utc=payload.get('created_at_utc'),
            )
            if not ov.is_active(_utcnow()):
                self.clear_override()
                return None
            return ov
        except Exception:
            return None

    def get_active_override(self, now_utc: Optional[datetime] = None) -> Optional[ManualRegimeOverride]:
        now_utc = now_utc or _utcnow()
        ov = self._read_override()
        if ov and ov.is_active(now_utc):
            return ov
        return None

    def set_override(self, state: str, reason: str, duration_minutes: int = 0) -> Dict[str, object]:
        now_utc = _utcnow()
        expires_at = None
        if duration_minutes and duration_minutes > 0:
            expires_at = _iso(now_utc + timedelta(minutes=int(duration_minutes)))
        payload = {
            'state': str(state or 'GREEN').upper(),
            'reason': str(reason or '').strip(),
            'source': 'manual',
            'created_at_utc': _iso(now_utc),
            'expires_at_utc': expires_at,
        }
        os.makedirs(os.path.dirname(self.override_path), exist_ok=True)
        with open(self.override_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return payload

    def clear_override(self) -> None:
        try:
            if os.path.exists(self.override_path):
                os.remove(self.override_path)
        except Exception:
            pass

    def sector_multiplier(self, state: str, sector: str) -> float:
        state = (state or 'GREEN').upper()
        sector_norm = str(sector or '').strip()
        if sector_norm == 'Energy':
            if state == 'RED':
                return float(self.settings.regime_energy_red_multiplier)
            if state == 'AMBER':
                return float(self.settings.regime_energy_amber_multiplier)
        return 1.0

    def save_last_auto_decision(self, decision: RegimeDecision) -> None:
        if decision.is_manual_override:
            return
        if not decision.live_evaluated:
            return
        if (decision.state or '').upper() not in LIVE_REGIME_STATES:
            return
        try:
            os.makedirs(os.path.dirname(self.last_path), exist_ok=True)
            with open(self.last_path, 'w', encoding='utf-8') as fh:
                json.dump(decision.to_dict(), fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def load_last_auto_decision(self) -> Optional[RegimeDecision]:
        try:
            if not os.path.exists(self.last_path):
                return None
            with open(self.last_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh) or {}
            decision = RegimeDecision(
                state=str(payload.get('state') or 'NOT_EVALUATED').upper(),
                source=str(payload.get('source') or 'persisted'),
                reasons=[str(x) for x in (payload.get('reasons') or []) if str(x).strip()],
                multiplier=float(payload.get('multiplier') or 1.0),
                prob_cap=float(payload['prob_cap']) if payload.get('prob_cap') is not None else None,
                touch_threshold_mult=float(payload.get('touch_threshold_mult') or 1.0),
                path_floor_add=float(payload.get('path_floor_add') or 0.0),
                suppress_new_signals=bool(payload.get('suppress_new_signals')),
                cooldown_until_utc=payload.get('cooldown_until_utc'),
                is_manual_override=bool(payload.get('is_manual_override')),
                metrics=payload.get('metrics') or {},
                note=str(payload.get('note') or ''),
                evaluated_at_utc=payload.get('evaluated_at_utc'),
                last_live_state=payload.get('last_live_state'),
                market_session=payload.get('market_session'),
                live_evaluated=bool(payload.get('live_evaluated')),
                data_complete=bool(payload.get('data_complete')),
            )
            return decision
        except Exception:
            return None

    def bootstrap_status(self, now_utc: datetime, market_open: bool) -> RegimeDecision:
        manual = self.get_active_override(now_utc)
        if manual is not None:
            return manual.to_decision(
                self.settings,
                market_session='LIVE' if market_open else 'CLOSED',
                evaluated_at_utc=_iso(now_utc),
            )
        last = self.load_last_auto_decision()
        if market_open:
            if last is not None:
                return RegimeDecision(
                    state='NOT_EVALUATED',
                    source='persisted',
                    reasons=['WAITING_FOR_LIVE_PROXY_REFRESH', f'LAST_LIVE_{last.state}'],
                    note=f'Waiting for fresh live proxy evaluation. Last live regime was {last.state} at {last.evaluated_at_utc or "unknown time"}.',
                    multiplier=1.0,
                    prob_cap=None,
                    touch_threshold_mult=1.0,
                    path_floor_add=0.0,
                    suppress_new_signals=False,
                    metrics=last.metrics or {},
                    evaluated_at_utc=last.evaluated_at_utc,
                    last_live_state=last.state if last.state in LIVE_REGIME_STATES else None,
                    market_session='LIVE',
                    live_evaluated=False,
                    data_complete=False,
                )
            return RegimeDecision(
                state='NOT_EVALUATED',
                source='auto',
                reasons=['WAITING_FOR_LIVE_PROXY_REFRESH'],
                note='No fresh live proxy evaluation yet in this process.',
                evaluated_at_utc=None,
                market_session='LIVE',
                live_evaluated=False,
                data_complete=False,
            )
        return self.closed_status(now_utc, None, None, {})

    def unavailable_status(self, reason: str, note: str, *, now_utc: Optional[datetime] = None, market_session: str = 'LIVE') -> RegimeDecision:
        now_utc = now_utc or _utcnow()
        last = self.load_last_auto_decision()
        reasons = [str(reason or 'PROXY_EVAL_UNAVAILABLE').strip().upper() or 'PROXY_EVAL_UNAVAILABLE']
        if last is not None and last.state in LIVE_REGIME_STATES:
            reasons.append(f'LAST_LIVE_{last.state}')
        return RegimeDecision(
            state='NOT_EVALUATED',
            source='auto',
            reasons=reasons,
            note=str(note or 'Live regime evaluation unavailable.'),
            multiplier=1.0,
            prob_cap=None,
            touch_threshold_mult=1.0,
            path_floor_add=0.0,
            suppress_new_signals=False,
            metrics=last.metrics or {} if last else {},
            evaluated_at_utc=last.evaluated_at_utc if last else None,
            last_live_state=last.state if last and last.state in LIVE_REGIME_STATES else None,
            market_session=market_session,
            live_evaluated=False,
            data_complete=False,
        )

    def closed_status(
        self,
        now_utc: datetime,
        open_utc: Optional[datetime],
        close_utc: Optional[datetime],
        bars_by_symbol: Dict[str, List[dict]],
    ) -> RegimeDecision:
        manual = self.get_active_override(now_utc)
        if manual is not None:
            return manual.to_decision(self.settings, market_session='CLOSED', evaluated_at_utc=_iso(now_utc))

        live_candidate = None
        if open_utc is not None and close_utc is not None and bars_by_symbol:
            try:
                maybe_live = self._evaluate_live(now_utc, open_utc, close_utc, bars_by_symbol)
                if maybe_live.live_evaluated and maybe_live.state in LIVE_REGIME_STATES:
                    live_candidate = maybe_live
                    self.save_last_auto_decision(maybe_live)
            except Exception:
                live_candidate = None
        last = live_candidate or self.load_last_auto_decision()
        if last is not None and (last.state or '').upper() in LIVE_REGIME_STATES:
            return RegimeDecision(
                state='CLOSED',
                source='persisted' if live_candidate is None else 'auto_cached',
                reasons=['MARKET_CLOSED', f'LAST_LIVE_{last.state}'],
                note=f'Market closed; not live-evaluated now. Last live regime was {last.state} at {last.evaluated_at_utc or "unknown time"}.',
                multiplier=float(last.multiplier or 1.0),
                prob_cap=last.prob_cap,
                touch_threshold_mult=float(last.touch_threshold_mult or 1.0),
                path_floor_add=float(last.path_floor_add or 0.0),
                suppress_new_signals=True,
                metrics=last.metrics or {},
                evaluated_at_utc=last.evaluated_at_utc,
                last_live_state=last.state,
                market_session='CLOSED',
                live_evaluated=False,
                data_complete=bool(last.data_complete),
            )
        return RegimeDecision(
            state='NOT_EVALUATED',
            source='auto',
            reasons=['MARKET_CLOSED', 'NO_RECENT_PROXY_EVALUATION'],
            note='Market closed and no recent live regime evaluation is available.',
            suppress_new_signals=True,
            evaluated_at_utc=None,
            market_session='CLOSED',
            live_evaluated=False,
            data_complete=False,
        )

    def evaluate(
        self,
        now_utc: datetime,
        open_utc: Optional[datetime],
        close_utc: Optional[datetime],
        bars_by_symbol: Dict[str, List[dict]],
        *,
        market_open: bool = True,
    ) -> RegimeDecision:
        if not self.settings.regime_enabled:
            return RegimeDecision(state='NOT_EVALUATED', note='Regime layer disabled', market_session='LIVE' if market_open else 'CLOSED')
        if market_open:
            return self._evaluate_live(now_utc, open_utc, close_utc, bars_by_symbol)
        return self.closed_status(now_utc, open_utc, close_utc, bars_by_symbol)

    def _evaluate_live(self, now_utc: datetime, open_utc: Optional[datetime], close_utc: Optional[datetime], bars_by_symbol: Dict[str, List[dict]]) -> RegimeDecision:
        manual = self.get_active_override(now_utc)
        if manual is not None:
            decision = manual.to_decision(self.settings, market_session='LIVE', evaluated_at_utc=_iso(now_utc))
            if open_utc is not None and close_utc is not None:
                decision.metrics = self._build_metrics(open_utc, close_utc, bars_by_symbol)
            return decision

        if open_utc is None or close_utc is None:
            return RegimeDecision(
                state='NOT_EVALUATED',
                source='auto',
                reasons=['MISSING_SESSION_WINDOW'],
                note='Missing session window for live proxy evaluation.',
                evaluated_at_utc=_iso(now_utc),
                market_session='LIVE',
                live_evaluated=False,
                data_complete=False,
            )

        metrics = self._build_metrics(open_utc, close_utc, bars_by_symbol)
        missing = self._missing_metrics(metrics)
        if missing:
            last = self.load_last_auto_decision()
            reasons = ['INSUFFICIENT_PROXY_BARS'] + [f'MISSING_{name.upper()}' for name in missing]
            if last is not None and last.state in LIVE_REGIME_STATES:
                reasons.append(f'LAST_LIVE_{last.state}')
            return RegimeDecision(
                state='NOT_EVALUATED',
                source='auto',
                reasons=reasons,
                note='Proxy regime not yet evaluated; need a full 1h window for SPY and all regime proxies.',
                multiplier=1.0,
                prob_cap=None,
                touch_threshold_mult=1.0,
                path_floor_add=0.0,
                suppress_new_signals=False,
                metrics=metrics,
                evaluated_at_utc=_iso(now_utc),
                last_live_state=last.state if last and last.state in LIVE_REGIME_STATES else None,
                market_session='LIVE',
                live_evaluated=False,
                data_complete=False,
            )

        oil = metrics.get('oil', {})
        vol = metrics.get('volatility', {})
        haven = metrics.get('safe_haven', {})
        xle = metrics.get('energy_equity', {})
        spy = metrics.get('spy', {})

        oil_1h = _safe_float(oil.get('ret_1h'))
        vol_1h = _safe_float(vol.get('ret_1h'))
        haven_1h = _safe_float(haven.get('ret_1h'))
        xle_1h = _safe_float(xle.get('ret_1h'))
        spy_1h = _safe_float(spy.get('ret_1h'))
        spy_open = _safe_float(spy.get('ret_since_open'))
        xle_rel_spy = None
        if xle_1h is not None and spy_1h is not None:
            xle_rel_spy = float(xle_1h - spy_1h)
            metrics['energy_vs_spy'] = {'ret_1h': xle_rel_spy}

        red_reasons: List[str] = []
        amber_reasons: List[str] = []
        if oil_1h is not None:
            if abs(oil_1h) >= float(self.settings.regime_red_oil_move_1h):
                red_reasons.append(f'OIL_1H={oil_1h:+.2%}')
            elif abs(oil_1h) >= float(self.settings.regime_amber_oil_move_1h):
                amber_reasons.append(f'OIL_1H={oil_1h:+.2%}')
        if vol_1h is not None:
            if vol_1h >= float(self.settings.regime_red_vol_move_1h):
                red_reasons.append(f'VOL_1H={vol_1h:+.2%}')
            elif vol_1h >= float(self.settings.regime_amber_vol_move_1h):
                amber_reasons.append(f'VOL_1H={vol_1h:+.2%}')
        if haven_1h is not None:
            if haven_1h >= float(self.settings.regime_red_safe_haven_move_1h):
                red_reasons.append(f'SAFE_HAVEN_1H={haven_1h:+.2%}')
            elif haven_1h >= float(self.settings.regime_amber_safe_haven_move_1h):
                amber_reasons.append(f'SAFE_HAVEN_1H={haven_1h:+.2%}')
        if xle_rel_spy is not None:
            if xle_rel_spy >= float(self.settings.regime_red_energy_rel_spy_1h):
                red_reasons.append(f'XLE_REL_SPY_1H={xle_rel_spy:+.2%}')
            elif xle_rel_spy >= float(self.settings.regime_amber_energy_rel_spy_1h):
                amber_reasons.append(f'XLE_REL_SPY_1H={xle_rel_spy:+.2%}')
        if spy_1h is not None:
            if spy_1h <= float(self.settings.regime_red_spy_drop_1h):
                red_reasons.append(f'SPY_1H={spy_1h:+.2%}')
            elif spy_1h <= float(self.settings.regime_amber_spy_drop_1h):
                amber_reasons.append(f'SPY_1H={spy_1h:+.2%}')
        if spy_open is not None:
            if spy_open <= float(self.settings.regime_red_spy_drop_since_open):
                red_reasons.append(f'SPY_OPEN={spy_open:+.2%}')
            elif spy_open <= float(self.settings.regime_amber_spy_drop_since_open):
                amber_reasons.append(f'SPY_OPEN={spy_open:+.2%}')

        if red_reasons:
            decision = _decision_from_state(self.settings, 'RED')
            decision.reasons = _dedupe(red_reasons)
            decision.metrics = metrics
            decision.note = 'Automated market-wide geopolitical stress regime'
            decision.evaluated_at_utc = _iso(now_utc)
            decision.market_session = 'LIVE'
            decision.live_evaluated = True
            decision.data_complete = True
            decision.last_live_state = 'RED'
            return decision
        if amber_reasons:
            decision = _decision_from_state(self.settings, 'AMBER')
            decision.reasons = _dedupe(amber_reasons)
            decision.metrics = metrics
            decision.note = 'Elevated geopolitical/event-risk regime'
            decision.evaluated_at_utc = _iso(now_utc)
            decision.market_session = 'LIVE'
            decision.live_evaluated = True
            decision.data_complete = True
            decision.last_live_state = 'AMBER'
            return decision

        decision = RegimeDecision(
            state='GREEN',
            source='auto',
            reasons=['PROXIES_NORMAL'],
            metrics=metrics,
            note='Proxy regime normal',
            evaluated_at_utc=_iso(now_utc),
            market_session='LIVE',
            live_evaluated=True,
            data_complete=True,
            last_live_state='GREEN',
        )
        return decision

    def _build_metrics(self, open_utc: datetime, close_utc: datetime, bars_by_symbol: Dict[str, List[dict]]) -> Dict[str, Dict[str, float | int | bool | str | None]]:
        return {
            'spy': _metric_snapshot('SPY', bars_by_symbol.get('SPY', []), open_utc, close_utc),
            'oil': _metric_snapshot(self.settings.regime_oil_proxy, bars_by_symbol.get(self.settings.regime_oil_proxy, []), open_utc, close_utc),
            'volatility': _metric_snapshot(self.settings.regime_vol_proxy, bars_by_symbol.get(self.settings.regime_vol_proxy, []), open_utc, close_utc),
            'safe_haven': _metric_snapshot(self.settings.regime_safe_haven_proxy, bars_by_symbol.get(self.settings.regime_safe_haven_proxy, []), open_utc, close_utc),
            'energy_equity': _metric_snapshot(self.settings.regime_energy_proxy, bars_by_symbol.get(self.settings.regime_energy_proxy, []), open_utc, close_utc),
        }

    def _missing_metrics(self, metrics: Dict[str, Dict[str, float | int | bool | str | None]]) -> List[str]:
        missing: List[str] = []
        for key in ['spy', 'oil', 'volatility', 'safe_haven', 'energy_equity']:
            snap = metrics.get(key) or {}
            if snap.get('ret_1h') is None:
                missing.append(key)
        return missing


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
