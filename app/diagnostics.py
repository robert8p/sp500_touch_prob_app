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


def _tokenize(s: Any) -> List[str]:
    raw = str(s or '').replace('·', ' ')
    return [tok for tok in raw.split() if tok]


def _yn(v: Any) -> bool:
    return bool(v)


@dataclass
class DiagnosticSummary:
    trade_date: Optional[str] = None
    latest_snapshot_utc: Optional[str] = None
    snapshots_count: int = 0
    tracked_count: int = 0
    signaled_count: int = 0
    evaluated: bool = False
    evaluated_at_utc: Optional[str] = None
    strict_touch_count: int = 0
    clean_touch_count: int = 0
    bouncy_touch_count: int = 0
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
                    payload = json.load(f)
                payload.setdefault('eod_review', None)
                payload.setdefault('stage1_rejects', {})
                return payload
            except Exception:
                pass
        return {
            'trade_date': trade_day,
            'session_open_utc': None,
            'session_close_utc': None,
            'created_at_utc': _iso(datetime.now(timezone.utc)),
            'snapshots': [],
            'tracked': {},
            'stage1_rejects': {},
            'evaluation': None,
            'eod_review': None,
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
            signaled_count=sum(1 for v in tracked.values() if v.get('times_signal', 0) > 0),
            evaluated=bool(evaluation),
            evaluated_at_utc=evaluation.get('evaluated_at_utc') if isinstance(evaluation, dict) else None,
            strict_touch_count=sum(1 for r in verdict_rows if r.get('strict_touch')),
            clean_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') == 'CLEAN_TOUCH'),
            bouncy_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') == 'BOUNCY_TOUCH'),
            ugly_touch_count=sum(1 for r in verdict_rows if r.get('path_verdict') == 'UGLY_TOUCH'),
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
            return {'trade_date': None, 'summary': asdict(DiagnosticSummary()), 'snapshots': [], 'tracked': [], 'stage1_rejects': [], 'evaluation': None, 'eod_review': None}
        payload = self._load_day(trade_day)
        tracked_rows = []
        for symbol, info in sorted((payload.get('tracked') or {}).items(), key=lambda kv: kv[1].get('max_prob_touch', kv[1].get('max_prob_1', 0.0)), reverse=True):
            row = dict(info)
            row['symbol'] = symbol
            tracked_rows.append(row)
        stage1_rows = []
        for symbol, info in sorted((payload.get('stage1_rejects') or {}).items(), key=lambda kv: kv[1].get('max_score', 0.0), reverse=True):
            row = dict(info)
            row['symbol'] = symbol
            stage1_rows.append(row)
        evaluation = payload.get('evaluation')
        return {
            'trade_date': trade_day,
            'summary': self._build_summary(payload),
            'snapshots': payload.get('snapshots', []),
            'tracked': tracked_rows,
            'stage1_rejects': stage1_rows,
            'evaluation': evaluation,
            'eod_review': payload.get('eod_review'),
        }

    def record_scan(
        self,
        trade_day: date,
        run_utc: str,
        session_open_utc: datetime,
        session_close_utc: datetime,
        coverage: CoverageStatus,
        rows: List[ScoreRow],
        near_misses: Optional[list] = None,
        stage1_rejects: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        trade_day_str = trade_day.isoformat()
        payload = self._load_day(trade_day_str)
        payload['session_open_utc'] = _iso(session_open_utc)
        payload['session_close_utc'] = _iso(session_close_utc)
        tracked_rows = self._select_rows_to_track(rows)
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
            'tracked_rows': [self._row_brief(r, run_utc) for r in tracked_rows],
            'near_misses': [{'symbol': nm.symbol, 'score': nm.score, 'reason': nm.rejection_reason} for nm in (near_misses or [])][:200],
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
                'max_prob_touch': 0.0,
                'times_touch_ge_006': 0,
                'times_touch_ge_010': 0,
                'times_signal': 0,
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
            touch_prob = _safe_float(getattr(row, 'prob_touch', None), prob) or prob
            if touch_prob >= item.get('max_prob_touch', 0.0):
                item['max_prob_touch'] = touch_prob
                item['best_snapshot'] = self._row_brief(row, run_utc)
            if prob >= item.get('max_prob_1', 0.0):
                item['max_prob_1'] = prob
                item['max_prob_1_raw'] = max(raw_prob, _safe_float(item.get('max_prob_1_raw'), 0.0) or 0.0)
            if touch_prob >= 0.06:
                item['times_touch_ge_006'] = _safe_int(item.get('times_touch_ge_006')) + 1
            if touch_prob >= 0.10:
                item['times_touch_ge_010'] = _safe_int(item.get('times_touch_ge_010')) + 1
            if getattr(row, 'signal', '') in ('ACTIONABLE', 'CANDIDATE'):
                item['times_signal'] = _safe_int(item.get('times_signal')) + 1
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
            item.setdefault('snapshots', []).append(self._row_brief(row, run_utc))
            item['snapshots'] = item['snapshots'][-100:]

        reject_map = payload.setdefault('stage1_rejects', {})
        for rej in (stage1_rejects or []):
            sym = str(rej.get('symbol') or '').strip().upper()
            if not sym:
                continue
            item = reject_map.setdefault(sym, {
                'symbol': sym,
                'sector': rej.get('sector'),
                'first_reject_utc': run_utc,
                'last_reject_utc': run_utc,
                'reject_count': 0,
                'max_score': float(rej.get('score') or 0.0),
                'best_reason': str(rej.get('reason') or ''),
                'reasons_set': [],
                'ever_blocked': False,
                'ever_event': False,
                'ever_time_filtered': False,
                'ever_strong_override': False,
            })
            item['last_reject_utc'] = run_utc
            item['reject_count'] = _safe_int(item.get('reject_count')) + 1
            score = float(rej.get('score') or 0.0)
            if score >= _safe_float(item.get('max_score'), 0.0):
                item['max_score'] = score
                item['best_reason'] = str(rej.get('reason') or '')
            rs = list(item.get('reasons_set') or [])
            reason = str(rej.get('reason') or '')
            if reason and reason not in rs:
                rs.append(reason)
            item['reasons_set'] = rs
            item['ever_blocked'] = bool(item.get('ever_blocked')) or bool(rej.get('blocked'))
            item['ever_event'] = bool(item.get('ever_event')) or bool(rej.get('event'))
            item['ever_time_filtered'] = bool(item.get('ever_time_filtered')) or bool(rej.get('time_filtered'))
            item['ever_strong_override'] = bool(item.get('ever_strong_override')) or bool(rej.get('strong_override'))

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

    def _row_brief(self, row: ScoreRow, run_utc: Optional[str] = None) -> Dict[str, Any]:
        prob = _safe_float(row.prob_1, 0.0) or 0.0
        touch_prob = _safe_float(getattr(row, 'prob_touch', None), prob) or prob
        return {
            'run_utc': run_utc,
            'symbol': row.symbol,
            'sector': row.sector,
            'price': _safe_float(row.price),
            'vwap': _safe_float(row.vwap),
            'prob_1': prob,
            'prob_1_raw': _safe_float(row.prob_1_raw, prob),
            'prob_touch': touch_prob,
            'prob_touch_raw': _safe_float(getattr(row, 'prob_touch_raw', None), touch_prob),
            'prob_path': _safe_float(getattr(row, 'prob_path', None)),
            'signal': getattr(row, 'signal', ''),
            'raw_signal': getattr(row, 'raw_signal', ''),
            'suppression_reason': getattr(row, 'suppression_reason', ''),
            'watchlist_rescue': bool(getattr(row, 'watchlist_rescue', False)),
            'watchlist_reason': getattr(row, 'watchlist_reason', ''),
            'watchlist_score': _safe_float(getattr(row, 'watchlist_score', None)),
            'risk': row.risk,
            'risk_reasons': row.risk_reasons,
            'reasons': row.reasons,
            'downside_risk': _safe_float(row.downside_risk),
            'uncertainty': row.uncertainty,
            'uncertainty_reasons': row.uncertainty_reasons,
            'stage1_score': _safe_float(row.stage1_score),
            'stage1_reasons': row.stage1_reasons,
            'guardrail_flags': getattr(row, 'guardrail_flags', ''),
            'acceptable': getattr(row, 'acceptable', None),
            'event_risk': getattr(row, 'event_risk', None),
            'high_downside': getattr(row, 'high_downside', None),
            'medium_downside': getattr(row, 'medium_downside', None),
            'high_uncertainty': getattr(row, 'high_uncertainty', None),
            'tail_validated': getattr(row, 'tail_validated', None),
            'display_touch_threshold': _safe_float(getattr(row, 'display_touch_threshold', None)),
            'path_action_min': _safe_float(getattr(row, 'path_action_min', None)),
        }

    def _select_rows_to_track(self, rows: List[ScoreRow]) -> List[ScoreRow]:
        if not rows:
            return []
        keep: List[ScoreRow] = []
        for i, row in enumerate(rows):
            touch_prob = _safe_float(getattr(row, 'prob_touch', None) or row.prob_1, 0.0) or 0.0
            keep_row = False
            if touch_prob >= self.settings.diag_track_min_prob:
                keep_row = True
            if i < self.settings.diag_track_top_n:
                keep_row = True
            if getattr(row, 'signal', '') in ('ACTIONABLE', 'CANDIDATE'):
                keep_row = True
            if bool(getattr(row, 'watchlist_rescue', False)):
                keep_row = True
            if keep_row:
                keep.append(row)
        return keep[: max(self.settings.diag_track_top_n, len(keep))]

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
            future = [c for _, c in closes[first_touch_idx:first_touch_idx + max(1, self.settings.diag_held_minutes)]]
            if future:
                frac = sum(1 for c in future if c >= scan_price) / float(len(future))
                held_closes_fraction = frac
                held_above_scan_10m = frac >= self.settings.diag_held_fraction
        strict_touch = bool(touch_1pct and (mae_before_touch_pct is not None) and mae_before_touch_pct >= self.settings.strict_touch_mae_threshold)
        worthy_touch = bool(strict_touch and ((held_above_scan_10m is True) or ((close_vs_scan_pct or -999.0) >= self.settings.worthy_close_vs_scan_min)))
        if not touch_1pct:
            path_verdict = 'NO_TOUCH'
        elif strict_touch and held_above_scan_10m:
            path_verdict = 'CLEAN_TOUCH'
        elif strict_touch:
            path_verdict = 'BOUNCY_TOUCH'
        else:
            path_verdict = 'UGLY_TOUCH'
        review_bucket = 'REJECT'
        risk = best.get('risk')
        downside = _safe_float(best.get('downside_risk'))
        uncertainty = (best.get('uncertainty') or 'LOW').upper()
        if worthy_touch and risk == 'OK' and (downside is None or downside < 0.48) and uncertainty != 'HIGH':
            review_bucket = 'WORTH_REVIEW'
        elif strict_touch:
            review_bucket = 'WATCHLIST_ONLY'
        return {
            'symbol': symbol,
            'sector': info.get('sector'),
            'scan_utc': best.get('run_utc'),
            'scan_price': scan_price,
            'target_price': target_price,
            'best_prob_1': _safe_float(best.get('prob_1')),
            'best_prob_1_raw': _safe_float(best.get('prob_1_raw')),
            'best_prob_touch': _safe_float(best.get('prob_touch')),
            'best_prob_touch_raw': _safe_float(best.get('prob_touch_raw')),
            'best_prob_path': _safe_float(best.get('prob_path')),
            'signal': best.get('signal', ''),
            'risk': risk,
            'risk_reasons': best.get('risk_reasons'),
            'reasons': best.get('reasons'),
            'downside_risk': downside,
            'uncertainty': best.get('uncertainty'),
            'guardrail_flags': best.get('guardrail_flags'),
            'touch_1pct': touch_1pct,
            'strict_touch': strict_touch,
            'worthy_touch': worthy_touch,
            'first_touch_utc': first_touch_utc,
            'max_high': max_high,
            'mae_before_touch_pct': mae_before_touch_pct,
            'held_above_scan_10m': held_above_scan_10m,
            'held_closes_fraction': held_closes_fraction,
            'close_vs_scan_pct': close_vs_scan_pct,
            'path_verdict': path_verdict,
            'review_bucket': review_bucket,
            'times_seen': info.get('seen_count'),
            'times_touch_ge_006': info.get('times_touch_ge_006'),
            'times_touch_ge_010': info.get('times_touch_ge_010'),
            'times_signal': info.get('times_signal'),
        }

    # ---- Review builders ----
    def build_eod_review(
        self,
        client: Optional[AlpacaClient],
        symbols: Optional[List[str]] = None,
        trade_day: Optional[str] = None,
        now_utc: Optional[datetime] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        now_utc = now_utc or datetime.now(timezone.utc)
        if trade_day is None:
            latest = self.load_latest_summary()
            trade_day = latest.get('trade_date')
        if not trade_day:
            return self._empty_eod_review(trade_day, 'No trade day available yet.')
        payload = self._load_day(trade_day)
        cached = payload.get('eod_review')
        if isinstance(cached, dict) and cached.get('version') == 1 and not refresh:
            return cached
        session_open = _parse_iso(payload.get('session_open_utc'))
        session_close = _parse_iso(payload.get('session_close_utc'))
        if session_open is None or session_close is None:
            return self._empty_eod_review(trade_day, 'Session timings are not available yet.')
        if now_utc < session_close + timedelta(minutes=1):
            return self._empty_eod_review(trade_day, 'End-of-day review is only available after the market closes.')
        if client is None:
            return self._empty_eod_review(trade_day, 'A live Alpaca data client is required for end-of-day review.')
        evaluation = payload.get('evaluation')
        if not evaluation or refresh:
            payload['evaluation'] = None if refresh else payload.get('evaluation')
            if refresh:
                self._save_day(trade_day, payload)
            evaluation = self.maybe_evaluate_closed_trade_day(client, now_utc)
            payload = self._load_day(trade_day)
            evaluation = payload.get('evaluation')
        tracked = payload.get('tracked') or {}
        surfaced_symbols = {sym for sym, info in tracked.items() if _safe_int(info.get('times_signal')) > 0}
        tracked_symbols = set(tracked.keys())
        eligible_start = session_open + timedelta(minutes=int(self.settings.stage1_min_minutes_since_open))
        end_utc = session_close + timedelta(minutes=1)
        universe_symbols = [s for s in sorted({str(s).strip().upper() for s in (symbols or []) if str(s).strip()})]
        if not universe_symbols:
            universe_symbols = sorted(tracked_symbols)
        bars_1m, bars_error, bars_warning = client.get_bars(universe_symbols, timeframe='1Min', start_utc=eligible_start, end_utc=end_utc, limit=None)
        eval_by_symbol = {r.get('symbol'): r for r in ((evaluation or {}).get('rows') or []) if r.get('symbol')}
        universe_rows = []
        for sym in universe_symbols:
            row = self._evaluate_universe_symbol(sym, tracked.get(sym) or {}, eval_by_symbol.get(sym), bars_1m.get(sym) or [], eligible_start, session_close, sym in surfaced_symbols)
            if row is not None:
                universe_rows.append(row)
        surfaced_winners = sorted([r for r in universe_rows if r.get('touch_1pct_after_eligible') and r.get('surfaced')], key=lambda x: (x.get('max_gain_from_start_pct') or 0.0, x.get('best_prob_touch') or 0.0), reverse=True)
        missed_winners = sorted([r for r in universe_rows if r.get('touch_1pct_after_eligible') and not r.get('surfaced')], key=lambda x: (x.get('max_gain_from_start_pct') or 0.0, x.get('tracked') or False), reverse=True)
        false_positives = []
        worthy_misses = []
        for sym, row in eval_by_symbol.items():
            info = tracked.get(sym) or {}
            merged = self._merge_eval_with_tracking(row, info)
            if _safe_int(row.get('times_signal')) > 0 and not row.get('touch_1pct'):
                false_positives.append(merged)
            if row.get('worthy_touch') and _safe_int(row.get('times_signal')) == 0:
                worthy_misses.append(merged)
        false_positives.sort(key=lambda x: (x.get('best_prob_touch') or 0.0, x.get('best_prob_1') or 0.0), reverse=True)
        worthy_misses.sort(key=lambda x: (x.get('best_prob_touch') or 0.0, x.get('best_prob_1') or 0.0), reverse=True)
        review = {
            'version': 1,
            'trade_date': trade_day,
            'review_ready': True,
            'scope': {
                'winner_basis': 'full_universe_from_first_1m_bar_after_stage1_start',
                'eligible_start_utc': _iso(eligible_start),
                'session_close_utc': _iso(session_close),
                'stage1_min_minutes_since_open': int(self.settings.stage1_min_minutes_since_open),
                'universe_symbols': len(universe_symbols),
                'tracked_symbols': len(tracked_symbols),
                'surfaced_symbols': len(surfaced_symbols),
            },
            'summary': {
                'full_universe_winners_count': sum(1 for r in universe_rows if r.get('touch_1pct_after_eligible')),
                'surfaced_winners_count': len(surfaced_winners),
                'missed_winners_count': len(missed_winners),
                'untracked_missed_winners_count': sum(1 for r in missed_winners if not r.get('tracked')),
                'tracked_missed_winners_count': sum(1 for r in missed_winners if r.get('tracked')),
                'false_positives_count': len(false_positives),
                'worthy_misses_count': len(worthy_misses),
                'tracked_review_rows_count': len((evaluation or {}).get('rows') or []),
            },
            'bars_warning': bars_warning,
            'bars_error': bars_error,
            'surfaced_winners': surfaced_winners[:300],
            'missed_winners': missed_winners[:300],
            'false_positives': false_positives[:300],
            'worthy_misses': worthy_misses[:300],
        }
        payload['eod_review'] = review
        self._save_day(trade_day, payload)
        return review

    def build_scan_history(self, trade_day: Optional[str] = None, symbol: Optional[str] = None, include_unsurfaced: bool = True) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        if not trade_day:
            return {'trade_date': None, 'rows': [], 'summary': {'row_count': 0, 'symbol_count': 0}}
        payload = self._load_day(trade_day)
        evaluation = payload.get('evaluation') or {}
        eval_by_symbol = {r.get('symbol'): r for r in evaluation.get('rows', []) if r.get('symbol')}
        tracked = payload.get('tracked') or {}
        rows = []
        for sym, info in tracked.items():
            if symbol and sym.upper() != symbol.upper():
                continue
            if (not include_unsurfaced) and _safe_int(info.get('times_signal')) <= 0:
                continue
            for snap in info.get('snapshots', []) or []:
                row = dict(snap)
                row['tracked'] = True
                row['times_seen_total'] = _safe_int(info.get('seen_count'))
                row['times_signal_total'] = _safe_int(info.get('times_signal'))
                row['max_prob_touch_symbol'] = _safe_float(info.get('max_prob_touch'))
                ev = eval_by_symbol.get(sym) or {}
                row['touch_1pct_eod'] = ev.get('touch_1pct')
                row['strict_touch_eod'] = ev.get('strict_touch')
                row['worthy_touch_eod'] = ev.get('worthy_touch')
                row['path_verdict_eod'] = ev.get('path_verdict')
                row['review_bucket_eod'] = ev.get('review_bucket')
                rows.append(row)
        rows.sort(key=lambda r: (r.get('run_utc') or '', r.get('symbol') or ''))
        return {
            'trade_date': trade_day,
            'summary': {'row_count': len(rows), 'symbol_count': len({r.get('symbol') for r in rows})},
            'rows': rows,
        }

    def build_stage1_review(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        if not trade_day:
            return {'trade_date': None, 'rows': [], 'summary': {'reject_symbol_count': 0}}
        payload = self._load_day(trade_day)
        evaluation = payload.get('evaluation') or {}
        eval_by_symbol = {r.get('symbol'): r for r in evaluation.get('rows', []) if r.get('symbol')}
        eod = payload.get('eod_review') or {}
        missed_by_symbol = {r.get('symbol'): r for r in (eod.get('missed_winners') or []) if r.get('symbol')}
        rows = []
        for sym, info in (payload.get('stage1_rejects') or {}).items():
            ev = eval_by_symbol.get(sym) or {}
            miss = missed_by_symbol.get(sym) or {}
            rows.append({
                'symbol': sym,
                'sector': info.get('sector'),
                'first_reject_utc': info.get('first_reject_utc'),
                'last_reject_utc': info.get('last_reject_utc'),
                'reject_count': _safe_int(info.get('reject_count')),
                'max_score': _safe_float(info.get('max_score')),
                'best_reason': info.get('best_reason'),
                'reasons_set': info.get('reasons_set') or [],
                'ever_blocked': _yn(info.get('ever_blocked')),
                'ever_event': _yn(info.get('ever_event')),
                'ever_time_filtered': _yn(info.get('ever_time_filtered')),
                'touch_1pct_eod': ev.get('touch_1pct') or miss.get('touch_1pct_after_eligible'),
                'strict_touch_eod': ev.get('strict_touch'),
                'worthy_touch_eod': ev.get('worthy_touch'),
                'path_verdict_eod': ev.get('path_verdict'),
                'missed_winner': bool(miss),
            })
        rows.sort(key=lambda r: (r.get('missed_winner', False), r.get('worthy_touch_eod', False), r.get('max_score') or 0.0), reverse=True)
        return {'trade_date': trade_day, 'summary': {'reject_symbol_count': len(rows)}, 'rows': rows}

    def build_blocker_attribution(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        if not trade_day:
            return {'trade_date': None, 'rows': [], 'summary': {'count': 0}}
        payload = self._load_day(trade_day)
        eod = payload.get('eod_review') or {}
        if not eod:
            return {'trade_date': trade_day, 'rows': [], 'summary': {'count': 0, 'warning': 'No eod_review cached yet'}}
        tracked = payload.get('tracked') or {}
        stage1_rejects = payload.get('stage1_rejects') or {}
        worthy_symbols = {r.get('symbol') for r in (eod.get('worthy_misses') or [])}
        rows = []
        for miss in (eod.get('missed_winners') or []):
            sym = miss.get('symbol')
            info = tracked.get(sym) or {}
            best = info.get('best_snapshot') or {}
            rej = stage1_rejects.get(sym) or {}
            risk = (best.get('risk') or miss.get('risk') or '').upper()
            reasons = ' '.join(_tokenize(best.get('reasons') or miss.get('reasons')))
            risk_reasons = ' '.join(_tokenize(best.get('risk_reasons') or miss.get('risk_reasons')))
            downside = _safe_float(best.get('downside_risk') or miss.get('downside_risk'))
            uncertainty = (best.get('uncertainty') or miss.get('uncertainty') or 'LOW').upper()
            p_touch = _safe_float(best.get('prob_touch') or miss.get('best_prob_touch'))
            p_path = _safe_float(best.get('prob_path') or miss.get('best_prob_path'))
            ath = _safe_float(best.get('display_touch_threshold'))
            pmin = _safe_float(best.get('path_action_min'), self.settings.path_quality_action_min)
            if not miss.get('tracked'):
                failure_stage = 'stage1_untracked'
            elif risk == 'BLOCKED':
                failure_stage = 'risk_block'
            elif _yn(best.get('event_risk')):
                failure_stage = 'event_gate'
            elif _yn(best.get('high_downside')):
                failure_stage = 'downside_gate'
            elif _yn(best.get('high_uncertainty')) or uncertainty == 'HIGH':
                failure_stage = 'uncertainty_gate'
            elif best.get('acceptable') is False:
                failure_stage = 'acceptable_long_gate'
            elif ath is not None and p_touch is not None and p_touch < ath:
                failure_stage = 'touch_threshold'
            elif p_path is not None and pmin is not None and p_path < pmin:
                failure_stage = 'path_threshold'
            else:
                failure_stage = 'watchlist_suppressed'
            rows.append({
                'symbol': sym,
                'sector': miss.get('sector'),
                'tracked': miss.get('tracked'),
                'worthy_miss': sym in worthy_symbols,
                'failure_stage': failure_stage,
                'best_row_run_utc': best.get('run_utc') or miss.get('scan_utc'),
                'best_prob_touch': p_touch,
                'best_prob_path': p_path,
                'best_prob_1': _safe_float(best.get('prob_1') or miss.get('best_prob_1')),
                'touch_threshold': ath,
                'path_threshold': pmin,
                'risk': risk or None,
                'risk_reasons': risk_reasons,
                'reasons': reasons,
                'downside_risk': downside,
                'uncertainty': uncertainty,
                'stage1_best_reason': rej.get('best_reason'),
                'stage1_reject_count': _safe_int(rej.get('reject_count')),
                'would_surface_without_event_gate': bool(failure_stage == 'event_gate' and _base_promotable(best)),
                'would_surface_without_downside_gate': bool(failure_stage == 'downside_gate' and _base_promotable(best)),
                'would_surface_without_uncertainty_gate': bool(failure_stage == 'uncertainty_gate' and _base_promotable(best)),
                'distance_to_touch_threshold': None if (ath is None or p_touch is None) else (p_touch - ath),
                'distance_to_path_threshold': None if (pmin is None or p_path is None) else (p_path - pmin),
            })
        rows.sort(key=lambda r: (r.get('worthy_miss', False), r.get('tracked', False), r.get('best_prob_touch') or 0.0), reverse=True)
        return {'trade_date': trade_day, 'summary': {'count': len(rows)}, 'rows': rows}

    def build_promotion_attribution(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        if not trade_day:
            return {'trade_date': None, 'rows': [], 'summary': {'count': 0}}
        payload = self._load_day(trade_day)
        eod = payload.get('eod_review') or {}
        rows = []
        for group, label in [((eod.get('surfaced_winners') or []), 'surfaced_winner'), ((eod.get('false_positives') or []), 'false_positive')]:
            for row in group:
                reasons = _tokenize(row.get('reasons'))
                rows.append({
                    'symbol': row.get('symbol'),
                    'promotion_type': label,
                    'run_utc': row.get('scan_utc') or row.get('eligible_start_utc'),
                    'signal': row.get('signal') or '',
                    'best_prob_touch': _safe_float(row.get('best_prob_touch')),
                    'best_prob_path': _safe_float(row.get('best_prob_path')),
                    'best_prob_1': _safe_float(row.get('best_prob_1')),
                    'risk': row.get('risk'),
                    'risk_reasons': row.get('risk_reasons'),
                    'positive_contributors': [t for t in reasons if t.endswith('+') or t in {'STRONG_OVERRIDE', 'SMOOTH', 'NEARHIGH'}],
                    'negative_contributors': [t for t in reasons if t.startswith('DOWNSIDE_') or t.startswith('UNCERT_') or t == 'TAIL_UNVALIDATED'],
                    'path_verdict': row.get('path_verdict'),
                    'review_bucket': row.get('review_bucket'),
                })
        rows.sort(key=lambda r: (r.get('promotion_type') == 'false_positive', r.get('best_prob_touch') or 0.0), reverse=True)
        return {'trade_date': trade_day, 'summary': {'count': len(rows)}, 'rows': rows}

    def build_calibration_review(self, trade_day: Optional[str] = None, metric: str = 'prob_touch', bucket_mode: str = 'decile', basis: str = 'all_rows') -> Dict[str, Any]:
        hist = self.build_scan_history(trade_day=trade_day)
        rows = hist.get('rows') or []
        if basis == 'surfaced_only':
            rows = [r for r in rows if (r.get('signal') or '') in ('ACTIONABLE', 'CANDIDATE')]
        elif basis == 'best_snapshot':
            by_sym = {}
            for r in rows:
                cur = by_sym.get(r['symbol'])
                if cur is None or (_safe_float(r.get(metric), 0.0) or 0.0) > (_safe_float(cur.get(metric), 0.0) or 0.0):
                    by_sym[r['symbol']] = r
            rows = list(by_sym.values())
        vals = [(_safe_float(r.get(metric)), r) for r in rows if _safe_float(r.get(metric)) is not None]
        vals.sort(key=lambda x: x[0])
        buckets = []
        if not vals:
            return {'trade_date': hist.get('trade_date'), 'metric': metric, 'bucket_mode': bucket_mode, 'buckets': []}
        if bucket_mode == 'fixed':
            edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.01]
            for lo, hi in zip(edges[:-1], edges[1:]):
                bucket_rows = [r for v, r in vals if v >= lo and v < hi]
                buckets.append(self._bucket_stats(f'[{lo:.2f},{hi:.2f})', bucket_rows, metric))
        else:
            n = len(vals)
            for i in range(10):
                lo = int(i * n / 10)
                hi = int((i + 1) * n / 10)
                bucket_rows = [r for _, r in vals[lo:hi]]
                if bucket_rows:
                    label = f'decile_{i+1}'
                    buckets.append(self._bucket_stats(label, bucket_rows, metric))
        return {'trade_date': hist.get('trade_date'), 'metric': metric, 'basis': basis, 'bucket_mode': bucket_mode, 'buckets': buckets}

    def build_guardrail_review(self, trade_day: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        hist = self.build_scan_history(trade_day=trade_day, symbol=symbol)
        rows = []
        for r in hist.get('rows') or []:
            flags = _tokenize(r.get('guardrail_flags'))
            raw_touch = _safe_float(r.get('prob_touch_raw'))
            disp_touch = _safe_float(r.get('prob_touch'))
            if raw_touch is None and disp_touch is None:
                continue
            if flags or (raw_touch is not None and disp_touch is not None and abs(raw_touch - disp_touch) > 1e-9):
                rows.append({
                    'run_utc': r.get('run_utc'),
                    'symbol': r.get('symbol'),
                    'guardrail_flags': flags,
                    'pre_guardrail_prob_touch': raw_touch,
                    'post_guardrail_prob_touch': disp_touch,
                    'delta_touch': None if (raw_touch is None or disp_touch is None) else (disp_touch - raw_touch),
                    'pre_guardrail_prob_1': _safe_float(r.get('prob_1_raw')),
                    'post_guardrail_prob_1': _safe_float(r.get('prob_1')),
                    'later_touch_1pct': r.get('touch_1pct_eod'),
                    'later_strict_touch': r.get('strict_touch_eod'),
                    'later_worthy_touch': r.get('worthy_touch_eod'),
                })
        rows.sort(key=lambda r: (r.get('run_utc') or '', r.get('symbol') or ''))
        return {'trade_date': hist.get('trade_date'), 'summary': {'count': len(rows)}, 'rows': rows}

    def build_threshold_review(
        self,
        trade_day: Optional[str] = None,
        touch_threshold: Optional[float] = None,
        path_min: Optional[float] = None,
        ignore_downside: bool = False,
        ignore_uncertainty: bool = False,
        ignore_event: bool = False,
        include_tail_unvalidated: bool = True,
    ) -> Dict[str, Any]:
        hist = self.build_scan_history(trade_day=trade_day)
        rows = hist.get('rows') or []
        by_sym: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            sym = r.get('symbol')
            if not sym:
                continue
            ath = _safe_float(r.get('display_touch_threshold'), touch_threshold)
            pmin = _safe_float(r.get('path_action_min'), path_min or self.settings.path_quality_action_min)
            cur_touch_thr = touch_threshold if touch_threshold is not None else ath
            cur_path_thr = path_min if path_min is not None else pmin
            p_touch = _safe_float(r.get('prob_touch'))
            p_path = _safe_float(r.get('prob_path'))
            if p_touch is None or p_path is None or cur_touch_thr is None or cur_path_thr is None:
                promoted = False
            else:
                promoted = (
                    p_touch >= cur_touch_thr
                    and p_path >= cur_path_thr
                    and (ignore_event or not _yn(r.get('event_risk')))
                    and (ignore_downside or not _yn(r.get('high_downside')))
                    and (ignore_uncertainty or not _yn(r.get('high_uncertainty')))
                    and (include_tail_unvalidated or _yn(r.get('tail_validated')))
                    and (r.get('acceptable') is not False)
                )
            if promoted:
                prev = by_sym.get(sym)
                if prev is None or (_safe_float(r.get('prob_touch'), 0.0) or 0.0) > (_safe_float(prev.get('prob_touch'), 0.0) or 0.0):
                    by_sym[sym] = r
        outcome = {}
        for r in rows:
            outcome[r.get('symbol')] = {
                'touch_1pct': r.get('touch_1pct_eod'),
                'strict_touch': r.get('strict_touch_eod'),
                'worthy_touch': r.get('worthy_touch_eod'),
            }
        promoted_rows = []
        for sym, row in by_sym.items():
            o = outcome.get(sym) or {}
            promoted_rows.append({
                'symbol': sym,
                'run_utc': row.get('run_utc'),
                'prob_touch': row.get('prob_touch'),
                'prob_path': row.get('prob_path'),
                'touch_1pct_eod': o.get('touch_1pct'),
                'strict_touch_eod': o.get('strict_touch'),
                'worthy_touch_eod': o.get('worthy_touch'),
            })
        summary = {
            'promoted_symbols': len(promoted_rows),
            'touch_winners': sum(1 for r in promoted_rows if r.get('touch_1pct_eod')),
            'strict_touches': sum(1 for r in promoted_rows if r.get('strict_touch_eod')),
            'worthy_touches': sum(1 for r in promoted_rows if r.get('worthy_touch_eod')),
            'false_positives': sum(1 for r in promoted_rows if not r.get('touch_1pct_eod')),
        }
        promoted_rows.sort(key=lambda r: (_safe_float(r.get('prob_touch'), 0.0) or 0.0), reverse=True)
        return {
            'trade_date': hist.get('trade_date'),
            'params': {
                'touch_threshold': touch_threshold,
                'path_min': path_min,
                'ignore_downside': ignore_downside,
                'ignore_uncertainty': ignore_uncertainty,
                'ignore_event': ignore_event,
                'include_tail_unvalidated': include_tail_unvalidated,
            },
            'summary': summary,
            'rows': promoted_rows[:500],
        }

    def build_review_slices(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        if not trade_day:
            return {'trade_date': None}
        payload = self._load_day(trade_day)
        eod = payload.get('eod_review') or {}
        if not eod:
            return {'trade_date': trade_day, 'warning': 'No eod_review cached yet'}
        blocked_winners = [r for r in (eod.get('missed_winners') or []) if (str(r.get('risk') or '').upper() == 'BLOCKED')]
        watchlist_only_clean_winners = [r for r in (eod.get('worthy_misses') or []) if r.get('path_verdict') == 'CLEAN_TOUCH']
        tracked_missed = [r for r in (eod.get('missed_winners') or []) if r.get('tracked')]
        untracked_missed = [r for r in (eod.get('missed_winners') or []) if not r.get('tracked')]
        return {
            'trade_date': trade_day,
            'surfaced_winners': eod.get('surfaced_winners') or [],
            'tracked_missed_winners': tracked_missed,
            'untracked_missed_winners': untracked_missed,
            'false_positives': eod.get('false_positives') or [],
            'worthy_misses': eod.get('worthy_misses') or [],
            'blocked_winners': blocked_winners,
            'watchlist_only_clean_winners': watchlist_only_clean_winners,
        }

    def build_review_export(self, trade_day: Optional[str] = None) -> Dict[str, Any]:
        if trade_day is None:
            trade_day = self.load_latest_summary().get('trade_date')
        return {
            'trade_date': trade_day,
            'diagnostics': self.load_day_for_api(trade_day),
            'scan_history': self.build_scan_history(trade_day),
            'stage1_review': self.build_stage1_review(trade_day),
            'blocker_attribution': self.build_blocker_attribution(trade_day),
            'promotion_attribution': self.build_promotion_attribution(trade_day),
            'calibration_prob_touch': self.build_calibration_review(trade_day, metric='prob_touch', basis='best_snapshot'),
            'calibration_prob_1': self.build_calibration_review(trade_day, metric='prob_1', basis='best_snapshot'),
            'calibration_prob_path': self.build_calibration_review(trade_day, metric='prob_path', basis='best_snapshot'),
            'guardrail_review': self.build_guardrail_review(trade_day),
            'review_slices': self.build_review_slices(trade_day),
        }

    # ---- Helpers ----
    def _empty_eod_review(self, trade_day: Optional[str], reason: str) -> Dict[str, Any]:
        return {
            'version': 1,
            'trade_date': trade_day,
            'review_ready': False,
            'reason': reason,
            'scope': {
                'winner_basis': 'full_universe_from_first_1m_bar_after_stage1_start',
                'eligible_start_utc': None,
                'session_close_utc': None,
                'stage1_min_minutes_since_open': int(self.settings.stage1_min_minutes_since_open),
                'universe_symbols': 0,
                'tracked_symbols': 0,
                'surfaced_symbols': 0,
            },
            'summary': {
                'full_universe_winners_count': 0,
                'surfaced_winners_count': 0,
                'missed_winners_count': 0,
                'untracked_missed_winners_count': 0,
                'tracked_missed_winners_count': 0,
                'false_positives_count': 0,
                'worthy_misses_count': 0,
                'tracked_review_rows_count': 0,
            },
            'bars_warning': None,
            'bars_error': None,
            'surfaced_winners': [],
            'missed_winners': [],
            'false_positives': [],
            'worthy_misses': [],
        }

    def _evaluate_universe_symbol(
        self,
        symbol: str,
        tracked_info: Dict[str, Any],
        eval_row: Optional[Dict[str, Any]],
        one_min: List[dict],
        eligible_start: datetime,
        session_close: datetime,
        surfaced: bool,
    ) -> Optional[Dict[str, Any]]:
        bars = []
        for b in one_min or []:
            ts = _parse_iso(b.get('t'))
            if ts is None:
                continue
            if ts < eligible_start or ts > session_close:
                continue
            bars.append((ts, b))
        if not bars:
            return None
        first_bar = bars[0][1]
        start_price = _safe_float(first_bar.get('o')) or _safe_float(first_bar.get('c'))
        if start_price is None or start_price <= 0:
            return None
        target_price = start_price * 1.01
        highs = [(_safe_float(b.get('h')) or 0.0, ts) for ts, b in bars]
        closes = [(_safe_float(b.get('c')) or 0.0, ts) for ts, b in bars]
        max_high = max((h for h, _ in highs), default=None)
        first_touch_utc = None
        if max_high is not None and max_high >= target_price:
            for high, ts in highs:
                if high >= target_price:
                    first_touch_utc = _iso(ts)
                    break
        close_vs_start_pct = None
        if closes and start_price > 0:
            close_vs_start_pct = (closes[-1][0] / start_price) - 1.0
        max_gain = None
        if max_high is not None and start_price > 0:
            max_gain = (max_high / start_price) - 1.0
        best = tracked_info.get('best_snapshot') or {}
        return {
            'symbol': symbol,
            'sector': tracked_info.get('sector') or (eval_row or {}).get('sector'),
            'tracked': bool(tracked_info),
            'surfaced': bool(surfaced),
            'eligible_start_utc': _iso(bars[0][0]),
            'eligible_start_price': start_price,
            'target_price': target_price,
            'touch_1pct_after_eligible': bool(max_high is not None and max_high >= target_price),
            'first_touch_utc': first_touch_utc,
            'max_high': max_high,
            'max_gain_from_start_pct': max_gain,
            'close_vs_start_pct': close_vs_start_pct,
            'times_signal': _safe_int(tracked_info.get('times_signal')),
            'times_seen': _safe_int(tracked_info.get('seen_count')),
            'best_prob_touch': _safe_float(best.get('prob_touch')) or _safe_float((eval_row or {}).get('best_prob_touch')),
            'best_prob_1': _safe_float(best.get('prob_1')) or _safe_float((eval_row or {}).get('best_prob_1')),
            'best_prob_path': _safe_float(best.get('prob_path')) or _safe_float((eval_row or {}).get('best_prob_path')),
            'signal': best.get('signal') or (eval_row or {}).get('signal') or '',
            'risk': best.get('risk') or (eval_row or {}).get('risk'),
            'risk_reasons': best.get('risk_reasons') or (eval_row or {}).get('risk_reasons'),
            'downside_risk': _safe_float(best.get('downside_risk')) or _safe_float((eval_row or {}).get('downside_risk')),
            'uncertainty': best.get('uncertainty') or (eval_row or {}).get('uncertainty'),
            'reasons': best.get('reasons') or (eval_row or {}).get('reasons'),
        }

    def _merge_eval_with_tracking(self, eval_row: Dict[str, Any], tracked_info: Dict[str, Any]) -> Dict[str, Any]:
        best = tracked_info.get('best_snapshot') or {}
        merged = dict(eval_row)
        merged['tracked'] = bool(tracked_info)
        merged['best_prob_touch'] = _safe_float(best.get('prob_touch')) or _safe_float(eval_row.get('best_prob_touch'))
        merged['best_prob_touch_raw'] = _safe_float(best.get('prob_touch_raw')) or _safe_float(eval_row.get('best_prob_touch_raw'))
        merged['best_prob_path'] = _safe_float(best.get('prob_path')) or _safe_float(eval_row.get('best_prob_path'))
        merged['best_prob_1'] = _safe_float(eval_row.get('best_prob_1')) or _safe_float(best.get('prob_1'))
        merged['signal'] = eval_row.get('signal') or best.get('signal') or ''
        merged['reasons'] = eval_row.get('reasons') or best.get('reasons')
        merged['risk_reasons'] = eval_row.get('risk_reasons') or best.get('risk_reasons')
        merged['times_seen'] = _safe_int(eval_row.get('times_seen')) or _safe_int(tracked_info.get('seen_count'))
        return merged

    def _bucket_stats(self, label: str, rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        n = len(rows)
        return {
            'bucket': label,
            'metric': metric,
            'rows': n,
            'symbols': len({r.get('symbol') for r in rows}),
            'avg_metric': None if not rows else sum((_safe_float(r.get(metric), 0.0) or 0.0) for r in rows) / float(n),
            'touch_rate': None if not rows else sum(1 for r in rows if r.get('touch_1pct_eod')) / float(n),
            'strict_touch_rate': None if not rows else sum(1 for r in rows if r.get('strict_touch_eod')) / float(n),
            'worthy_touch_rate': None if not rows else sum(1 for r in rows if r.get('worthy_touch_eod')) / float(n),
        }


def _base_promotable(best: Dict[str, Any]) -> bool:
    p_touch = _safe_float(best.get('prob_touch'))
    p_path = _safe_float(best.get('prob_path'))
    ath = _safe_float(best.get('display_touch_threshold'))
    pmin = _safe_float(best.get('path_action_min'))
    if p_touch is None or p_path is None or ath is None or pmin is None:
        return False
    return bool(
        p_touch >= ath
        and p_path >= pmin
        and (best.get('acceptable') is not False)
    )
