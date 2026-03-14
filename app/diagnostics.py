from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .alpaca import AlpacaClient
from .config import Settings
from .state import CoverageStatus, ScoreRow


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


@dataclass
class DiagnosticSummary:
    trade_date: Optional[str] = None
    latest_snapshot_utc: Optional[str] = None
    snapshots_count: int = 0
    tracked_count: int = 0
    cap_bound_count: int = 0
    evaluated: bool = False
    evaluated_at_utc: Optional[str] = None
    clean_touch_count: int = 0
    ugly_touch_count: int = 0
    no_touch_count: int = 0
    worthy_count: int = 0


class DiagnosticJournal:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = os.path.join(settings.model_dir, 'diagnostics')

    def _ensure_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def _day_path(self, trade_day: str) -> str:
        return os.path.join(self.base_dir, f'{trade_day}.json')

    def _latest_path(self) -> str:
        return os.path.join(self.base_dir, 'latest_summary.json')

    def _load_day(self, trade_day: str) -> Dict[str, Any]:
        self._ensure_dir()
        p = self._day_path(trade_day)
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'trade_date': trade_day,
            'session_open_utc': None,
            'session_close_utc': None,
            'created_at_utc': _iso(datetime.now(timezone.utc)),
            'snapshots': [],
            'tracked': {},
            'evaluation': None,
        }

    def _save_day(self, trade_day: str, payload: Dict[str, Any]) -> None:
        self._ensure_dir()
        with open(self._day_path(trade_day), 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        self._save_latest_summary(self._build_summary(payload))

    def _save_latest_summary(self, summary: Dict[str, Any]) -> None:
        self._ensure_dir()
        with open(self._latest_path(), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    def _build_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tracked = payload.get('tracked', {}) or {}
        evaluation = payload.get('evaluation') or {}
        verdict_rows = evaluation.get('rows', []) if isinstance(evaluation, dict) else []
        summary = asdict(DiagnosticSummary(
            trade_date=payload.get('trade_date'),
            latest_snapshot_utc=(payload.get('snapshots') or [{}])[-1].get('run_utc') if payload.get('snapshots') else None,
            snapshots_count=len(payload.get('snapshots') or []),
            tracked_count=len(tracked),
            cap_bound_count=sum(1 for v in tracked.values() if v.get('cap_bound_seen')),
            evaluated=bool(evaluation),
            evaluated_at_utc=evaluation.get('evaluated_at_utc') if isinstance(evaluation, dict) else None,
            clean_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') == 'CLEAN_TOUCH'),
            ugly_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') in ('UGLY_TOUCH', 'BOUNCY_TOUCH')),
            no_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') == 'NO_TOUCH'),
            worthy_count=sum(1 for r in verdict_rows if r.get('review_bucket') == 'WORTH_REVIEW'),
        ))
        return summary

    def load_latest_summary(self) -> Dict[str, Any]:
        p = self._latest_path()
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return asdict(DiagnosticSummary())

    def load_day_for_api(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            latest = self.load_latest_summary()
            trade_day = latest.get('trade_date')
        if not trade_day:
            return {'trade_date': None, 'summary': asdict(DiagnosticSummary()), 'snapshots': [], 'tracked': [], 'evaluation': None}
        payload = self._load_day(trade_day)
        tracked_rows = []
        for symbol, info in sorted((payload.get('tracked') or {}).items(), key=lambda kv: kv[1].get('max_prob_1', 0.0), reverse=True):
            row = dict(info)
            row['symbol'] = symbol
            tracked_rows.append(row)
        evaluation = payload.get('evaluation')
        return {
            'trade_date': trade_day,
            'summary': self._build_summary(payload),
            'snapshots': payload.get('snapshots', []),
            'tracked': tracked_rows,
            'evaluation': evaluation,
        }

    def record_scan(
        self,
        trade_day: date,
        run_utc: str,
        session_open_utc: datetime,
        session_close_utc: datetime,
        coverage: CoverageStatus,
        rows: List[ScoreRow],
        live_prob_cap: Optional[float],
    ) -> None:
        trade_day_str = trade_day.isoformat()
        payload = self._load_day(trade_day_str)
        payload['session_open_utc'] = _iso(session_open_utc)
        payload['session_close_utc'] = _iso(session_close_utc)
        tracked_rows = self._select_rows_to_track(rows, live_prob_cap)
        snapshot = {
            'run_utc': run_utc,
            'stage1_candidate_count': int(coverage.stage1_candidate_count),
            'stage2_scored_count': int(coverage.stage2_scored_count),
            'threshold_counts': dict(coverage.threshold_counts or {}),
            'acceptable_threshold_counts': dict(coverage.acceptable_threshold_counts or {}),
            'stage1_blocked_count': int(coverage.stage1_blocked_count),
            'stage1_event_count': int(coverage.stage1_event_count),
            'capped_by_downside_count': int(coverage.capped_by_downside_count),
            'capped_by_uncertainty_count': int(coverage.capped_by_uncertainty_count),
            'capped_by_event_count': int(coverage.capped_by_event_count),
            'tracked_rows': [self._row_brief(r, live_prob_cap, run_utc) for r in tracked_rows],
        }
        payload.setdefault('snapshots', []).append(snapshot)
        payload['snapshots'] = payload['snapshots'][-100:]
        tracked = payload.setdefault('tracked', {})
        for row in tracked_rows:
            item = tracked.setdefault(row.symbol, {
                'symbol': row.symbol,
                'sector': row.sector,
                'first_seen_utc': run_utc,
                'last_seen_utc': run_utc,
                'seen_count': 0,
                'max_prob_1': 0.0,
                'max_prob_1_raw': 0.0,
                'cap_bound_seen': False,
                'times_ge_0_70': 0,
                'times_ge_0_74': 0,
                'risk_set': [],
                'uncertainty_set': [],
                'max_downside_risk': None,
                'best_snapshot': None,
                'snapshots': [],
            })
            item['last_seen_utc'] = run_utc
            item['seen_count'] = _safe_int(item.get('seen_count')) + 1
            prob = _safe_float(row.prob_1, 0.0) or 0.0
            raw_prob = _safe_float(row.prob_1_raw, prob) or prob
            if prob >= item.get('max_prob_1', 0.0):
                item['max_prob_1'] = prob
                item['max_prob_1_raw'] = max(raw_prob, _safe_float(item.get('max_prob_1_raw'), 0.0) or 0.0)
                item['best_snapshot'] = self._row_brief(row, live_prob_cap, run_utc)
            item['cap_bound_seen'] = bool(item.get('cap_bound_seen')) or self._is_cap_bound(prob, live_prob_cap)
            if prob >= 0.70:
                item['times_ge_0_70'] = _safe_int(item.get('times_ge_0_70')) + 1
            if prob >= 0.74:
                item['times_ge_0_74'] = _safe_int(item.get('times_ge_0_74')) + 1
            risk_set = list(item.get('risk_set') or [])
            if row.risk not in risk_set:
                risk_set.append(row.risk)
            item['risk_set'] = risk_set
            unc_set = list(item.get('uncertainty_set') or [])
            if row.uncertainty not in unc_set:
                unc_set.append(row.uncertainty)
            item['uncertainty_set'] = unc_set
            ds = _safe_float(row.downside_risk)
            current_max_ds = _safe_float(item.get('max_downside_risk'))
            item['max_downside_risk'] = ds if current_max_ds is None else max(current_max_ds, ds if ds is not None else current_max_ds)
            item.setdefault('snapshots', []).append(self._row_brief(row, live_prob_cap, run_utc))
            item['snapshots'] = item['snapshots'][-50:]
        self._save_day(trade_day_str, payload)

    def maybe_evaluate_closed_trade_day(self, client: Optional[AlpacaClient], now_utc: datetime) -> Optional[Dict[str, Any]]:
        if client is None:
            return None
        latest = self.load_latest_summary()
        trade_day = latest.get('trade_date')
        if not trade_day:
            return None
        payload = self._load_day(trade_day)
        session_close = _parse_iso(payload.get('session_close_utc'))
        if session_close is None or now_utc < session_close + timedelta(minutes=1):
            return None
        if payload.get('evaluation'):
            return payload.get('evaluation')
        tracked = payload.get('tracked') or {}
        if not tracked:
            return None
        symbols = sorted(tracked.keys())
        start_utc = min((_parse_iso((tracked[s].get('best_snapshot') or {}).get('run_utc')) for s in symbols), default=None)
        if start_utc is None:
            return None
        end_utc = session_close + timedelta(minutes=1)
        bars_1m, err, warn = client.get_bars(symbols, timeframe='1Min', start_utc=start_utc, end_utc=end_utc, limit=None)
        rows = []
        for sym in symbols:
            info = tracked.get(sym) or {}
            best = info.get('best_snapshot') or {}
            scan_ts = _parse_iso(best.get('run_utc'))
            scan_price = _safe_float(best.get('price'))
            if scan_ts is None or scan_price is None or scan_price <= 0:
                continue
            one_min = [b for b in (bars_1m.get(sym) or []) if (_parse_iso(b.get('t')) or start_utc) >= scan_ts and (_parse_iso(b.get('t')) or start_utc) <= session_close]
            rows.append(self._evaluate_symbol(sym, info, best, one_min))
        evaluation = {
            'evaluated_at_utc': _iso(now_utc),
            'bars_warning': warn,
            'bars_error': err,
            'rows': rows,
        }
        payload['evaluation'] = evaluation
        self._save_day(trade_day, payload)
        return evaluation

    def _row_brief(self, row: ScoreRow, live_prob_cap: Optional[float], run_utc: Optional[str] = None) -> Dict[str, Any]:
        prob = _safe_float(row.prob_1, 0.0) or 0.0
        return {
            'run_utc': run_utc,
            'symbol': row.symbol,
            'sector': row.sector,
            'price': _safe_float(row.price),
            'vwap': _safe_float(row.vwap),
            'prob_1': prob,
            'prob_1_raw': _safe_float(row.prob_1_raw, prob),
            'risk': row.risk,
            'risk_reasons': row.risk_reasons,
            'reasons': row.reasons,
            'downside_risk': _safe_float(row.downside_risk),
            'uncertainty': row.uncertainty,
            'uncertainty_reasons': row.uncertainty_reasons,
            'stage1_score': _safe_float(row.stage1_score),
            'stage1_reasons': row.stage1_reasons,
            'cap_bound': self._is_cap_bound(prob, live_prob_cap),
        }

    def _select_rows_to_track(self, rows: List[ScoreRow], live_prob_cap: Optional[float]) -> List[ScoreRow]:
        if not rows:
            return []
        keep: List[ScoreRow] = []
        for i, row in enumerate(rows):
            prob = _safe_float(row.prob_1, 0.0) or 0.0
            keep_row = False
            if prob >= self.settings.diag_track_min_prob:
                keep_row = True
            if self._is_cap_bound(prob, live_prob_cap):
                keep_row = True
            if i < self.settings.diag_track_top_n:
                keep_row = True
            if keep_row:
                keep.append(row)
        return keep[: max(self.settings.diag_track_top_n, len(keep))]

    def _is_cap_bound(self, prob: float, live_prob_cap: Optional[float]) -> bool:
        if live_prob_cap is None:
            return False
        return abs(prob - float(live_prob_cap)) <= float(self.settings.diag_cap_buffer)

    def _evaluate_symbol(self, symbol: str, info: Dict[str, Any], best: Dict[str, Any], one_min: List[dict]) -> Dict[str, Any]:
        scan_price = _safe_float(best.get('price'), 0.0) or 0.0
        target_price = scan_price * 1.01
        highs: List[Tuple[datetime, float]] = []
        lows: List[Tuple[datetime, float]] = []
        closes: List[Tuple[datetime, float]] = []
        for b in one_min:
            ts = _parse_iso(b.get('t'))
            if ts is None:
                continue
            highs.append((ts, _safe_float(b.get('h'), 0.0) or 0.0))
            lows.append((ts, _safe_float(b.get('l'), 0.0) or 0.0))
            closes.append((ts, _safe_float(b.get('c'), 0.0) or 0.0))
        max_high = max((h for _, h in highs), default=None)
        touch_1pct = bool(max_high is not None and max_high >= target_price)
        first_touch_utc = None
        first_touch_idx = None
        if touch_1pct:
            for idx, (ts, high) in enumerate(highs):
                if high >= target_price:
                    first_touch_utc = _iso(ts)
                    first_touch_idx = idx
                    break
        if first_touch_idx is None:
            lows_before = [x for _, x in lows]
        else:
            lows_before = [x for _, x in lows[: first_touch_idx + 1]]
        mae_before_touch_pct = None
        if lows_before and scan_price > 0:
            mae_before_touch_pct = min((x / scan_price) - 1.0 for x in lows_before if x is not None)
        close_vs_scan_pct = None
        if closes and scan_price > 0:
            close_vs_scan_pct = (closes[-1][1] / scan_price) - 1.0
        held_above_scan_10m = None
        held_closes_fraction = None
        if touch_1pct and first_touch_idx is not None and closes:
            future = [c for _, c in closes[first_touch_idx + 1:first_touch_idx + 1 + self.settings.diag_held_minutes]]
            if future:
                frac = sum(1 for c in future if c >= scan_price) / float(len(future))
                held_closes_fraction = frac
                held_above_scan_10m = frac >= self.settings.diag_held_fraction
        sane_touch = bool(touch_1pct and (mae_before_touch_pct is not None) and mae_before_touch_pct >= self.settings.diag_sane_mae_pct and held_above_scan_10m)
        if not touch_1pct:
            path_verdict = 'NO_TOUCH'
        elif sane_touch:
            path_verdict = 'CLEAN_TOUCH'
        elif held_above_scan_10m:
            path_verdict = 'BOUNCY_TOUCH'
        else:
            path_verdict = 'UGLY_TOUCH'
        review_bucket = 'REJECT'
        risk = best.get('risk')
        downside = _safe_float(best.get('downside_risk'))
        uncertainty = (best.get('uncertainty') or 'LOW').upper()
        if sane_touch and risk == 'OK' and (downside is None or downside < 0.48) and uncertainty != 'HIGH':
            review_bucket = 'WORTH_REVIEW'
        elif touch_1pct:
            review_bucket = 'WATCHLIST_ONLY'
        return {
            'symbol': symbol,
            'sector': info.get('sector'),
            'scan_utc': best.get('run_utc'),
            'scan_price': scan_price,
            'target_price': target_price,
            'best_prob_1': _safe_float(best.get('prob_1')),
            'best_prob_1_raw': _safe_float(best.get('prob_1_raw')),
            'cap_bound': bool(best.get('cap_bound')),
            'risk': risk,
            'risk_reasons': best.get('risk_reasons'),
            'downside_risk': downside,
            'uncertainty': best.get('uncertainty'),
            'touch_1pct': touch_1pct,
            'first_touch_utc': first_touch_utc,
            'max_high': max_high,
            'mae_before_touch_pct': mae_before_touch_pct,
            'held_above_scan_10m': held_above_scan_10m,
            'held_closes_fraction': held_closes_fraction,
            'close_vs_scan_pct': close_vs_scan_pct,
            'path_verdict': path_verdict,
            'review_bucket': review_bucket,
            'times_seen': info.get('seen_count'),
            'times_ge_0_70': info.get('times_ge_0_70'),
            'times_ge_0_74': info.get('times_ge_0_74'),
        }
