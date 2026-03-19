from __future__ import annotations
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .alpaca import AlpacaClient
from .config import Settings
from .diagnostics import DiagnosticJournal
from .constituents import Constituent, load_fallback, normalize_symbol, try_refresh_from_wikipedia
from .features import FEATURE_NAMES, compute_features_from_5m
from .market import get_market_times, iso, next_aligned_run
from .modeling import (
    acceptable_long_mask_from_X,
    downside_risk_score_from_X,
    event_risk_mask_from_X,
    predict_probs,
    risk_bucket_from_X,
    stage1_diagnostics_from_X,
    try_load_bundle,
    uncertainty_from_X,
)
from .sectors import sector_etf_for_sector, unique_sector_etfs
from .state import AppState, CoverageStatus, NearMiss, RegimeStatus, ScoreRow, SkippedSymbol
from .specialist import (
    classify_setup_families,
    family_bonus_for_live_row,
    family_live_policy_for_row,
    sector_group_from_sector,
)
from .regime import RegimeController
from .volume_profiles import VolumeProfileStore, slot_index_from_ts

TTC_IDX = FEATURE_NAMES.index('ttc_frac')


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_ts(ts):
    if ts.endswith('Z'): ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts)


def _previous_trading_day(d, tz_name):
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('XNYS')
        start = d - timedelta(days=10)
        sched = cal.schedule(start_date=start, end_date=d)
        days = [idx.date() for idx in sched.index if idx.date() < d]
        if days: return days[-1]
    except Exception: pass
    x = d - timedelta(days=1)
    while x.weekday() >= 5: x -= timedelta(days=1)
    return x


def _session_utc_for_day(d, tz_name):
    tz = ZoneInfo(tz_name)
    open_local = datetime(d.year, d.month, d.day, 9, 30, tzinfo=tz)
    close_local = datetime(d.year, d.month, d.day, 16, 0, tzinfo=tz)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('XNYS')
        sched = cal.schedule(start_date=d, end_date=d)
        if len(sched) == 1:
            open_local = sched.iloc[0]['market_open'].to_pydatetime().astimezone(tz)
            close_local = sched.iloc[0]['market_close'].to_pydatetime().astimezone(tz)
    except Exception: pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


def _daily_ctx_from_bars(daily_bars, trade_day):
    rows = []
    for b in daily_bars:
        try: ts = _parse_ts(b['t']).date()
        except Exception: continue
        if ts < trade_day: rows.append(b)
    if not rows: return {}
    closes = np.array([float(b.get('c') or 0.0) for b in rows], dtype=float)
    highs = np.array([float(b.get('h') or 0.0) for b in rows], dtype=float)
    valid = closes > 0
    if not np.any(valid): return {}
    closes = closes[valid]; highs = highs[valid]; last = float(closes[-1])
    def ret_n(n):
        if closes.size <= n: return 0.0
        ref = float(closes[-(n + 1)]); return float(last / ref - 1.0) if ref > 0 else 0.0
    def sma_dist(n):
        if closes.size < n: return 0.0
        ma = float(np.mean(closes[-n:])); return float(last / ma - 1.0) if ma > 0 else 0.0
    def drawdown_n(n):
        peak = float(np.max(highs[-n:])) if highs.size >= n else (float(np.max(highs)) if highs.size else 0.0)
        return float(last / peak - 1.0) if peak > 0 else 0.0
    return {'ret_5d': ret_n(5), 'ret_20d': ret_n(20), 'ret_60d': ret_n(60), 'dist_20dma_pct': sma_dist(20), 'dist_50dma_pct': sma_dist(50), 'dist_200dma_pct': sma_dist(200), 'drawdown_20d_pct': drawdown_n(20), 'drawdown_60d_pct': drawdown_n(60)}


def _runtime_meta(settings):
    return {
        'liq_dvol_min_usd': float(settings.liq_dvol_min_usd), 'liq_range_pct_max': float(settings.liq_range_pct_max), 'liq_wick_atr_max': float(settings.liq_wick_atr_max),
        'blocked_ret20d_max': float(settings.blocked_ret20d_max), 'blocked_ret60d_max': float(settings.blocked_ret60d_max),
        'blocked_dist50dma_max': float(settings.blocked_dist50dma_max), 'blocked_ret_since_open_max': float(settings.blocked_ret_since_open_max),
        'blocked_damage_from_high_atr_min': float(settings.blocked_damage_from_high_atr_min), 'blocked_below_vwap_frac_min': float(settings.blocked_below_vwap_frac_min),
        'blocked_prob_cap': float(settings.blocked_prob_cap),
        'event_gap_abs_min': float(settings.event_gap_abs_min), 'event_rvol_min': float(settings.event_rvol_min), 'event_range_pct_min': float(settings.event_range_pct_min), 'event_prob_cap': float(settings.event_prob_cap),
        'uncertainty_z_thresh': float(settings.uncertainty_z_thresh), 'uncertainty_extreme_features_min': int(settings.uncertainty_extreme_features_min), 'uncertainty_prob_cap': float(settings.uncertainty_prob_cap),
        'downside_prob_cap_high': float(settings.downside_prob_cap_high), 'downside_prob_cap_medium': float(settings.downside_prob_cap_medium),
        'downside_high_threshold': float(settings.downside_high_threshold), 'downside_medium_threshold': float(settings.downside_medium_threshold),
        'stage1_candidate_cap': int(settings.stage1_candidate_cap), 'stage1_min_score': float(settings.stage1_min_score),
        'stage1_min_minutes_since_open': int(settings.stage1_min_minutes_since_open), 'stage1_min_minutes_to_close': int(settings.stage1_min_minutes_to_close),
        'stage1_min_rvol': float(settings.stage1_min_rvol), 'stage1_midday_min_rvol': float(settings.stage1_midday_min_rvol),
        'stage1_late_min_rvol': float(settings.stage1_late_min_rvol), 'stage1_strong_rvol_minutes': int(settings.stage1_strong_rvol_minutes),
        'stage1_min_dollar_volume_mult': float(settings.stage1_min_dollar_volume_mult),
        'stage1_min_rel_spy_30m': float(settings.stage1_min_rel_spy_30m), 'stage1_min_rel_spy_5m': float(settings.stage1_min_rel_spy_5m),
        'stage1_min_dist_pct_to_vwap': float(settings.stage1_min_dist_pct_to_vwap),
        'stage1_deadcat_damage_atr': float(settings.stage1_deadcat_damage_atr), 'stage1_deadcat_rel_spy_30m': float(settings.stage1_deadcat_rel_spy_30m),
        'stage1_strong_override_score': float(settings.stage1_strong_override_score),
        'strict_touch_mae_threshold': float(settings.strict_touch_mae_threshold),
        'worthy_close_vs_scan_min': float(settings.worthy_close_vs_scan_min), 'worthy_missed_prob_max': float(settings.worthy_missed_prob_max),
        'touch_tail_threshold_mult': float(settings.touch_tail_threshold_mult),
        'touch_tail_min_count': int(settings.touch_tail_min_count),
        'touch_tail_min_lift': float(settings.touch_tail_min_lift),
        'path_quality_action_min': float(settings.path_quality_action_min),
    }


@dataclass
class BarsCache:
    timeframe: str
    bars: Dict[str, List[dict]] = field(default_factory=dict)
    last_fetch_utc: Optional[datetime] = None
    last_bar_ts: Optional[str] = None
    def merge(self, symbol, new_bars, keep=900):
        existing = self.bars.get(symbol, [])
        seen = {b.get('t') for b in existing}
        merged = existing + [b for b in new_bars if b.get('t') not in seen]
        merged.sort(key=lambda x: x.get('t', ''))
        if len(merged) > keep: merged = merged[-keep:]
        self.bars[symbol] = merged


class Scanner:
    def __init__(self, settings, state):
        self.settings = settings
        self.state = state
        self._thread = None
        self._stop = threading.Event()
        self.constituents = []
        self.symbol_meta = {}
        self.cache_5m = BarsCache(timeframe='5Min')
        self.cache_1d = BarsCache(timeframe='1Day')
        self.daily_cache_trade_day = None
        self.vol_profiles = VolumeProfileStore(settings.model_dir)
        self.diagnostics = DiagnosticJournal(settings)
        self.regime_controller = RegimeController(settings)
        self._surfacing_trade_day = None
        self._surface_memory: Dict[str, Dict[str, object]] = {}

    def _regime_proxy_symbols(self) -> List[str]:
        return [
            self.settings.regime_oil_proxy,
            self.settings.regime_vol_proxy,
            self.settings.regime_safe_haven_proxy,
            self.settings.regime_energy_proxy,
        ]

    def _publish_regime(self, decision) -> None:
        self.state.set_regime(RegimeStatus(
            state=str(decision.state or 'NOT_EVALUATED'),
            source=str(decision.source or 'auto'),
            reasons=' · '.join(decision.reasons or []),
            note=str(decision.note or ''),
            multiplier=float(decision.multiplier or 1.0),
            prob_cap=float(decision.prob_cap) if decision.prob_cap is not None else None,
            touch_threshold_mult=float(decision.touch_threshold_mult or 1.0),
            path_floor_add=float(decision.path_floor_add or 0.0),
            suppress_new_signals=bool(decision.suppress_new_signals),
            is_manual_override=bool(decision.is_manual_override),
            cooldown_until_utc=decision.cooldown_until_utc,
            metrics=decision.metrics or {},
            evaluated_at_utc=getattr(decision, 'evaluated_at_utc', None),
            last_live_state=getattr(decision, 'last_live_state', None),
            market_session=getattr(decision, 'market_session', None),
            live_evaluated=bool(getattr(decision, 'live_evaluated', False)),
            data_complete=bool(getattr(decision, 'data_complete', False)),
        ))

    def _reset_intraday_memory_if_needed(self, trade_day: date) -> None:
        if self._surfacing_trade_day != trade_day:
            self._surfacing_trade_day = trade_day
            self._surface_memory = {}

    @staticmethod
    def _signal_priority(signal: str) -> int:
        if signal == 'ACTIONABLE':
            return 2
        if signal == 'CANDIDATE':
            return 1
        return 0

    def _is_unvalidated_messy(self, risk: str, risk_reasons: str) -> bool:
        risk_norm = str(risk or '').upper()
        if risk_norm in {'HIGH', 'BLOCKED'}:
            return True
        tokens = {tok.strip().upper() for tok in str(risk_reasons or '').replace('·', ' ').replace(',', ' ').split() if tok.strip()}
        configured = {tok.strip().upper() for tok in (self.settings.unvalidated_messy_risk_flags or []) if tok.strip()}
        return any(tok in tokens for tok in configured)

    def _apply_surfacing_cooldown(
        self,
        symbol: str,
        run_utc: str,
        signal: str,
        prob_touch: float,
        prob_path: float,
    ) -> Tuple[str, str]:
        if not signal:
            return '', ''
        cooldown = max(0, int(self.settings.surfacing_cooldown_minutes))
        if cooldown <= 0:
            self._surface_memory[symbol] = {
                'run_utc': run_utc,
                'signal': signal,
                'prob_touch': float(prob_touch),
                'prob_path': float(prob_path),
            }
            return signal, ''
        prev = self._surface_memory.get(symbol)
        if prev:
            prev_run = _parse_ts(str(prev.get('run_utc'))) if prev.get('run_utc') else None
            curr_run = _parse_ts(run_utc)
            if prev_run is not None and curr_run - prev_run < timedelta(minutes=cooldown):
                same_or_lower = self._signal_priority(signal) <= self._signal_priority(str(prev.get('signal') or ''))
                touch_improved = float(prob_touch) >= float(prev.get('prob_touch') or 0.0) + float(self.settings.surfacing_min_touch_delta)
                path_improved = float(prob_path) >= float(prev.get('prob_path') or 0.0) + float(self.settings.surfacing_min_path_delta)
                if same_or_lower and not (touch_improved or path_improved):
                    return '', 'COOLDOWN'
        self._surface_memory[symbol] = {
            'run_utc': run_utc,
            'signal': signal,
            'prob_touch': float(prob_touch),
            'prob_path': float(prob_path),
        }
        return signal, ''


    @staticmethod
    def _guardrail_multiplier_for_row(profiles: Optional[dict], *, event_risk: bool, uncertainty: str, high_downside: bool, medium_downside: bool) -> float:
        profiles = dict(profiles or {})

        def _read_mult(container, key, fallback):
            node = ((container or {}).get(key) if isinstance(container, dict) else None)
            if isinstance(node, dict):
                try:
                    return float(node.get('multiplier', fallback))
                except Exception:
                    return float(fallback)
            try:
                return float(node)
            except Exception:
                return float(fallback)

        mult = 1.0
        if event_risk:
            mult *= _read_mult(profiles, 'event', 0.40)
        unc_profiles = profiles.get('uncertainty') if isinstance(profiles.get('uncertainty'), dict) else {}
        unc_norm = str(uncertainty or '').upper()
        if unc_norm == 'HIGH':
            mult *= _read_mult(unc_profiles, 'HIGH', 0.50)
        elif unc_norm in {'MED', 'MEDIUM'}:
            mult *= _read_mult(unc_profiles, 'MED', 0.85)
        ds_profiles = profiles.get('downside') if isinstance(profiles.get('downside'), dict) else {}
        if high_downside:
            mult *= _read_mult(ds_profiles, 'HIGH', 0.45)
        elif medium_downside:
            mult *= _read_mult(ds_profiles, 'MED', 0.75)
        return float(np.clip(mult, 0.05, 1.0))


    def _compute_relative_strength_score(self, row: ScoreRow) -> float:
        signal_bonus = 0.0
        if row.signal == 'ACTIONABLE':
            signal_bonus += float(self.settings.rerank_actionable_bonus)
        elif row.signal == 'CANDIDATE':
            signal_bonus += float(self.settings.rerank_candidate_bonus)
        if row.watchlist_rescue:
            signal_bonus -= float(self.settings.rerank_watchlist_penalty)
        unc_pen = 1.0 if str(row.uncertainty or '').upper() == 'HIGH' else 0.5 if str(row.uncertainty or '').upper() in {'MED', 'MEDIUM'} else 0.0
        score = (
            float(self.settings.rerank_prob1_weight) * float(row.prob_1 or 0.0) +
            float(self.settings.rerank_touch_weight) * float(row.prob_touch or 0.0) +
            float(self.settings.rerank_path_weight) * float(row.prob_path or 0.0) +
            float(self.settings.rerank_stage1_weight) * max(0.0, float(row.stage1_score or 0.0) / 5.0) +
            float(self.settings.rerank_family_bonus_weight) * float(row.setup_family_bonus or 0.0) +
            signal_bonus -
            float(self.settings.rerank_downside_weight) * float(row.downside_risk or 0.0) -
            float(self.settings.rerank_uncertainty_weight) * unc_pen
        )
        if str(row.risk or '').upper() in {'HIGH', 'BLOCKED'}:
            score -= 0.08
        if row.suppression_reason:
            score -= 0.04
        if row.high_downside:
            score -= 0.05
        if row.high_uncertainty:
            score -= 0.04
        return float(score)

    def _apply_relative_strength_ranks(self, rows: List[ScoreRow]) -> None:
        ranked = sorted(
            rows,
            key=lambda r: (
                float(r.relative_strength_score if r.relative_strength_score is not None else self._compute_relative_strength_score(r)),
                self._signal_priority(str(r.signal or '')),
                float(r.prob_1 or 0.0),
                float(r.prob_touch or 0.0),
                float(r.prob_path or 0.0),
            ),
            reverse=True,
        )
        for idx, row in enumerate(ranked, start=1):
            row.relative_strength_rank = idx

    def _select_watchlist_rescue_rows(
        self,
        rows: List[ScoreRow],
        adaptive_touch_threshold: float,
        path_action_min: float,
    ) -> List[ScoreRow]:
        if not self.settings.watchlist_rescue_enabled or self.settings.watchlist_rescue_max_rows <= 0:
            return []
        path_floor = max(0.0, float(self.settings.watchlist_rescue_path_min))
        rescue: List[ScoreRow] = []
        for row in rows:
            row_touch_threshold = max(0.03, float(row.display_touch_threshold or adaptive_touch_threshold))
            row_decision_threshold = max(0.03, float(row.display_prob_threshold or (adaptive_touch_threshold * path_action_min)))
            touch_floor = max(0.03, row_touch_threshold * float(self.settings.watchlist_rescue_touch_frac))
            combined_floor = max(0.03, row_decision_threshold * float(self.settings.watchlist_rescue_combined_frac))
            if row.signal:
                continue
            if row.raw_signal:
                continue
            if row.suppression_reason in {'COOLDOWN', 'FAMILY_SUPPRESS'}:
                continue
            if not bool(row.acceptable):
                continue
            if bool(row.event_risk) or bool(row.high_uncertainty):
                continue
            if str(row.risk or '').upper() in {'HIGH', 'BLOCKED'}:
                continue
            if bool(row.high_downside):
                continue
            if bool(row.medium_downside) and not self.settings.watchlist_rescue_allow_medium_downside:
                continue
            if (row.stage1_score or 0.0) < float(self.settings.watchlist_rescue_min_stage1_score):
                continue
            if (row.prob_path or 0.0) < path_floor:
                continue
            if (row.prob_touch or 0.0) < touch_floor and (row.prob_1 or 0.0) < combined_floor:
                continue
            touch_ratio = (row.prob_touch or 0.0) / max(touch_floor, 1e-9)
            combined_ratio = (row.prob_1 or 0.0) / max(combined_floor, 1e-9)
            path_ratio = (row.prob_path or 0.0) / max(path_floor or 1e-9, 1e-9)
            watch_score = (0.45 * touch_ratio) + (0.30 * combined_ratio) + (0.20 * path_ratio) + (0.05 * float(row.stage1_score or 0.0))
            row.watchlist_rescue = True
            row.watchlist_reason = 'NEAR_THRESHOLD'
            row.watchlist_score = float(watch_score)
            rescue.append(row)
        rescue.sort(key=lambda r: (r.watchlist_score or 0.0, r.prob_touch or 0.0, r.prob_1 or 0.0), reverse=True)
        return rescue[: int(self.settings.watchlist_rescue_max_rows)]

    def load_constituents(self):
        from .constituents import Constituent, load_fallback, try_refresh_from_wikipedia, normalize_symbol
        fallback = load_fallback(); source = 'fallback'; warning = None; data = fallback
        refreshed, err = try_refresh_from_wikipedia()
        if refreshed is not None: data = refreshed; source = 'wikipedia'
        elif err: warning = f'refresh blocked/unavailable: {err}'
        normed = [Constituent(symbol=normalize_symbol(c.symbol), name=c.name, sector=c.sector, industry=c.industry) for c in data]
        self.constituents = normed; self.symbol_meta = {c.symbol: c for c in normed}
        with self.state.lock:
            self.state.constituents.source = source; self.state.constituents.warning = warning; self.state.constituents.count = len(normed)
            self.state.coverage.universe_count = len(normed)

    def start(self):
        if self.settings.disable_scheduler: return
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        t = threading.Thread(target=self._loop, daemon=True); self._thread = t; t.start()

    def _make_client(self):
        if self.settings.demo_mode: return None
        if not (self.settings.alpaca_api_key and self.settings.alpaca_api_secret): return None
        return AlpacaClient(self.settings.alpaca_api_key, self.settings.alpaca_api_secret, feed=self.settings.normalized_feed())

    def _update_market_status(self):
        now = _utcnow()
        open_utc, close_utc, is_open, ttc = get_market_times(now, self.settings.timezone)
        with self.state.lock:
            self.state.market.market_open = is_open; self.state.market.time_to_close_seconds = ttc
            self.state.market.market_open_time = iso(open_utc); self.state.market.market_close_time = iso(close_utc)
        return open_utc, close_utc, is_open, ttc

    def _demo_scores(self):
        demo = [('AAPL', 'Information Technology'), ('MSFT', 'Information Technology'), ('NVDA', 'Information Technology'), ('AMZN', 'Consumer Discretionary'), ('JPM', 'Financials')]
        rows = []
        for i, (sym, sec) in enumerate(demo):
            price = 100 + i * 25; vwap = price - 0.3 + 0.1 * i; p1 = min(0.95, 0.56 + 0.08 * i); risk = 'OK' if i < 3 else 'CAUTION'
            rows.append(ScoreRow(sym, sec, float(price), float(vwap), float(p1), risk, 'DEMO', 'DEMO', 0.18 + 0.08 * i, 'LOW', '', 2.0 + i * 0.35, 'LIQ TREND DEMO', float(p1), float(p1)))
        rows.sort(key=lambda r: r.prob_1, reverse=True)
        return rows

    def _refresh_daily_cache_if_needed(self, client, symbols_requested, trade_day):
        if self.daily_cache_trade_day == trade_day and self.cache_1d.bars: return None
        start_utc = datetime.now(timezone.utc) - timedelta(days=500)
        bars1d, err, _ = client.get_bars(symbols_requested, timeframe='1Day', start_utc=start_utc, end_utc=datetime.now(timezone.utc), limit=None)
        if err: return err
        self.cache_1d.bars = {k: list(v) for k, v in bars1d.items()}; self.cache_1d.last_fetch_utc = datetime.now(timezone.utc); self.daily_cache_trade_day = trade_day
        return None

    def scan_once(self, open_utc, close_utc, now_utc):
        run_utc = now_utc.isoformat().replace('+00:00', 'Z')
        if not self.constituents: self.load_constituents()
        universe_symbols = [c.symbol for c in self.constituents]
        sector_etfs = unique_sector_etfs([self.symbol_meta[s].sector for s in universe_symbols if s in self.symbol_meta])
        regime_symbols = [s for s in self._regime_proxy_symbols() if s]
        helper_symbols = list(dict.fromkeys(['SPY'] + sector_etfs + regime_symbols))
        symbols_requested = list(dict.fromkeys(universe_symbols + helper_symbols))
        helper_symbol_set = set(helper_symbols)
        cov = CoverageStatus(universe_count=len(universe_symbols), symbols_requested_count=len(symbols_requested))
        skip_counts = {'no_bars': 0, 'insufficient_bars': 0, 'missing_price_or_vwap': 0, 'other_errors': 0, 'model_schema_incompatible': 0}
        skipped = []

        if self.settings.demo_mode:
            with self.state.lock:
                self.state.alpaca.ok = True; self.state.alpaca.message = 'DEMO_MODE'; self.state.alpaca.feed = 'sip'
                self.state.alpaca.last_request_utc = run_utc; self.state.alpaca.last_bar_timestamp = run_utc; self.state.alpaca.rate_limit_warn = None
                self.state.model.pt1.trained = False; self.state.model.pt1.path = os.path.join(self.settings.model_dir, 'pt1')
            rows = self._demo_scores()
            self._publish_regime(self.regime_controller.unavailable_status(
                reason='DEMO_MODE',
                note='Demo mode active; live geopolitical regime is not being evaluated.',
                now_utc=now_utc,
                market_session='LIVE',
            ))
            cov.symbols_scored_count = len(rows); cov.stage1_candidate_count = len(rows); cov.stage2_scored_count = len(rows)
            self.state.set_scores(rows, run_utc); self.state.set_coverage(cov, [])
            return

        client = self._make_client()
        if client is None:
            self.state.set_error('Alpaca keys missing.')
            with self.state.lock: self.state.alpaca.ok = False; self.state.alpaca.message = 'Missing keys'
            self._publish_regime(self.regime_controller.unavailable_status(
                reason='ALPACA_UNAVAILABLE',
                note='Cannot live-evaluate geopolitical regime because Alpaca credentials are missing.',
                now_utc=now_utc,
                market_session='LIVE',
            ))
            self.state.set_coverage(cov, []); return

        trade_day = open_utc.astimezone(ZoneInfo(self.settings.timezone)).date()
        self._reset_intraday_memory_if_needed(trade_day)
        daily_err = self._refresh_daily_cache_if_needed(client, symbols_requested, trade_day)
        if daily_err:
            self.state.set_error(f'Alpaca daily error: {daily_err}')
            self._publish_regime(self.regime_controller.unavailable_status(
                reason='ALPACA_DAILY_ERROR',
                note=f'Cannot refresh daily data required for scanner context: {daily_err}',
                now_utc=now_utc,
                market_session='LIVE',
            ))
            self.state.set_coverage(cov, []); return

        if self.cache_5m.last_fetch_utc is None:
            prev_d = _previous_trading_day(trade_day, self.settings.timezone)
            prev_open_utc, _ = _session_utc_for_day(prev_d, self.settings.timezone)
            start = prev_open_utc
        else:
            start = self.cache_5m.last_fetch_utc - timedelta(minutes=10)
        end = now_utc
        bars_by_sym, err, warn = client.get_bars(symbols_requested, timeframe='5Min', start_utc=start, end_utc=end, limit=None)
        with self.state.lock:
            self.state.alpaca.feed = client.feed; self.state.alpaca.last_request_utc = run_utc; self.state.alpaca.rate_limit_warn = warn
            if err: self.state.alpaca.ok = False; self.state.alpaca.message = err
            else: self.state.alpaca.ok = True; self.state.alpaca.message = 'OK'
        if err:
            self.state.set_error(f'Alpaca error: {err}')
            self._publish_regime(self.regime_controller.unavailable_status(
                reason='ALPACA_INTRADAY_ERROR',
                note=f'Cannot live-evaluate geopolitical regime because intraday proxy bars failed: {err}',
                now_utc=now_utc,
                market_session='LIVE',
            ))
            self.state.set_coverage(cov, []); return

        cov.symbols_returned_with_bars_count = sum(1 for k, v in bars_by_sym.items() if k not in helper_symbol_set and v)
        for sym, lst in bars_by_sym.items():
            self.cache_5m.merge(sym, lst)
            if lst: self.cache_5m.last_bar_ts = lst[-1].get('t')
        self.cache_5m.last_fetch_utc = end
        with self.state.lock: self.state.alpaca.last_bar_timestamp = self.cache_5m.last_bar_ts

        regime_decision = self.regime_controller.evaluate(now_utc, open_utc, close_utc, self.cache_5m.bars, market_open=True)
        self._publish_regime(regime_decision)
        self.regime_controller.save_last_auto_decision(regime_decision)

        mins_to_close = max(0.0, (close_utc - now_utc).total_seconds() / 60.0)
        prev_d = _previous_trading_day(trade_day, self.settings.timezone)
        prev_open_utc, prev_close_utc = _session_utc_for_day(prev_d, self.settings.timezone)

        def current_session_bars(lst):
            return [b for b in lst if open_utc <= _parse_ts(b['t']) <= close_utc]

        spy_cur = current_session_bars(self.cache_5m.bars.get('SPY', []))
        spy_ret_5m = spy_ret_30m = spy_ret_since_open = 0.0
        spy_ret_5m_raw = 0.0
        spy_consecutive_down = 0
        spy_rv_1h = 0.0
        if len(spy_cur) >= 2:
            c_arr = np.array([float(b.get('c') or 0.0) for b in spy_cur], dtype=float)
            spy_open = c_arr[0] if c_arr.size else 1.0
            spy_ret_5m = float(c_arr[-1] / c_arr[-2] - 1.0) if c_arr[-2] else 0.0
            spy_ret_5m_raw = spy_ret_5m
            if len(spy_cur) >= 7: spy_ret_30m = float(c_arr[-1] / c_arr[-7] - 1.0) if c_arr[-7] else 0.0
            spy_ret_since_open = float(c_arr[-1] / spy_open - 1.0) if spy_open > 0 else 0.0
            # Consecutive negative SPY 5m bars
            for k in range(2, min(8, c_arr.size + 1)):
                if c_arr[-k + 1] < c_arr[-k]:
                    spy_consecutive_down += 1
                else:
                    break
            # SPY realized vol over last 12 bars (~1 hour)
            if c_arr.size >= 3:
                spy_logrets = np.diff(np.log(np.maximum(c_arr, 1e-9)))
                spy_seg = spy_logrets[-12:] if spy_logrets.size >= 12 else spy_logrets
                spy_rv_1h = float(np.std(spy_seg) * np.sqrt(max(1.0, float(spy_seg.size)))) if spy_seg.size > 2 else 0.0

        sector_ret_map = {}
        sector_ret_since_open_map = {}
        for etf in sector_etfs:
            cur = current_session_bars(self.cache_5m.bars.get(etf, []))
            if len(cur) >= 7:
                cc = np.array([float(b.get('c') or 0.0) for b in cur], dtype=float)
                sector_ret_map[etf] = float(cc[-1] / cc[-7] - 1.0) if cc[-7] else 0.0
                sec_open = cc[0] if cc.size else 1.0
                sector_ret_since_open_map[etf] = float(cc[-1] / sec_open - 1.0) if sec_open > 0 else 0.0
            else:
                sector_ret_map[etf] = 0.0
                sector_ret_since_open_map[etf] = 0.0

        avail, _ = self.vol_profiles.availability_counts()
        cov.profile_symbols_available = avail; cov.profile_symbols_missing = max(0, cov.universe_count - avail)

        feats = []; meta = []; sufficient_count = 0
        risk_params = (self.settings.liq_rolling_bars, self.settings.liq_dvol_min_usd, self.settings.liq_range_pct_max, self.settings.liq_wick_atr_max)
        blocked_params = {
            'ret20d_max': self.settings.blocked_ret20d_max, 'ret60d_max': self.settings.blocked_ret60d_max,
            'dist50dma_max': self.settings.blocked_dist50dma_max, 'ret_since_open_max': self.settings.blocked_ret_since_open_max,
            'damage_from_high_atr_min': self.settings.blocked_damage_from_high_atr_min, 'below_vwap_frac_min': self.settings.blocked_below_vwap_frac_min,
            'event_gap_abs_min': self.settings.event_gap_abs_min, 'event_rvol_min': self.settings.event_rvol_min, 'event_range_pct_min': self.settings.event_range_pct_min,
        }

        for sym in universe_symbols:
            bars_raw = self.cache_5m.bars.get(sym, [])
            bars = [b for b in bars_raw if (prev_open_utc <= _parse_ts(b['t']) <= prev_close_utc) or (open_utc <= _parse_ts(b['t']) <= close_utc)]
            if not bars: skip_counts['no_bars'] += 1; skipped.append(SkippedSymbol(symbol=sym, reason='no_bars')); continue
            if len(bars) < self.settings.min_bars_5m: skip_counts['insufficient_bars'] += 1; skipped.append(SkippedSymbol(symbol=sym, reason='insufficient_bars', last_bar_timestamp=bars[-1].get('t'))); continue
            current_bars = current_session_bars(bars)
            if not current_bars: skip_counts['no_bars'] += 1; skipped.append(SkippedSymbol(symbol=sym, reason='no_bars', last_bar_timestamp=bars[-1].get('t'))); continue
            sufficient_count += 1
            last_ts = current_bars[-1].get('t')
            slot_idx = None
            if last_ts:
                try: slot_idx = slot_index_from_ts(datetime.fromisoformat(last_ts.replace('Z', '+00:00')), self.settings.timezone)
                except Exception: pass
            baseline = self.vol_profiles.get_slot_median(sym, slot_idx) if slot_idx is not None else None
            prev_session = [b for b in bars if prev_open_utc <= _parse_ts(b['t']) <= prev_close_utc]
            prev_day_close = float(prev_session[-1].get('c') or 0.0) if prev_session else None
            prev_day_high = float(max(float(b.get('h') or 0.0) for b in prev_session)) if prev_session else None
            prev_day_low = float(min(float(b.get('l') or 0.0) for b in prev_session)) if prev_session else None
            sector_etf = sector_etf_for_sector(self.symbol_meta.get(sym).sector if sym in self.symbol_meta else '')
            sector_ret = sector_ret_map.get(sector_etf, 0.0)
            sector_ret_open = sector_ret_since_open_map.get(sector_etf, 0.0)
            daily_ctx = _daily_ctx_from_bars(self.cache_1d.bars.get(sym, []), trade_day)
            try:
                fr = compute_features_from_5m(
                    bars_5m=bars, spy_ret_5m=spy_ret_5m, spy_ret_30m=spy_ret_30m, sector_ret_30m=sector_ret,
                    mins_to_close=mins_to_close, tod_baseline_vol_median=baseline, rolling_rvol_window=20,
                    risk_params=risk_params, prev_day_close=prev_day_close, prev_day_high=prev_day_high, prev_day_low=prev_day_low,
                    daily_ctx=daily_ctx, tz_name=self.settings.timezone, blocked_params=blocked_params,
                    spy_ret_since_open=spy_ret_since_open,
                    spy_ret_5m_raw=spy_ret_5m_raw,
                    spy_consecutive_down=spy_consecutive_down,
                    spy_rv_1h=spy_rv_1h,
                    sector_ret_since_open=sector_ret_open,
                )
            except Exception: fr = None
            if fr is None: skip_counts['missing_price_or_vwap'] += 1; skipped.append(SkippedSymbol(symbol=sym, reason='missing_price_or_vwap', last_bar_timestamp=last_ts)); continue
            sector = self.symbol_meta.get(sym).sector if sym in self.symbol_meta else 'Unknown'
            feats.append(fr.features); meta.append((sym, sector, fr.price, fr.vwap, fr.risk, fr.risk_reasons, fr.reasons))

        cov.symbols_with_sufficient_bars_count = sufficient_count
        if not feats:
            cov.symbols_scored_count = 0; cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
            self.state.set_scores([], run_utc); self.state.set_coverage(cov, skipped, watchlist_rescue_rows=[]); return

        X_all = np.vstack(feats)
        bundle, bundle_status = try_load_bundle(self.settings.model_dir)
        runtime_meta = _runtime_meta(self.settings)
        merged_meta = {**(bundle.meta if bundle is not None else {}), **runtime_meta}
        stage1_score, stage1_pass, stage1_reasons, stage1_flags = stage1_diagnostics_from_X(X_all, merged_meta)
        pass_idxs = np.where(stage1_pass)[0]
        cov.stage1_candidate_count = int(pass_idxs.size)
        cov.stage1_blocked_count = int(np.sum(stage1_flags['blocked']))
        cov.stage1_event_count = int(np.sum(stage1_flags['event']))
        cov.stage1_time_filtered_count = int(np.sum(~stage1_flags['time_ok']))
        cov.stage1_rejected_count = int(X_all.shape[0] - np.sum(stage1_pass))
        cov.stage1_strong_override_count = int(np.sum(stage1_flags.get('strong_override', np.zeros(X_all.shape[0], dtype=bool))))

        # ── Near-miss tracking (NEW in v10) ──
        near_misses = []
        near_miss_min = self.settings.diag_near_miss_min_score
        rejected_idxs = np.where(~stage1_pass & (stage1_score >= near_miss_min))[0]
        sorted_rej = rejected_idxs[np.argsort(stage1_score[rejected_idxs])[::-1]][:self.settings.diag_near_miss_top_n]
        for idx in sorted_rej:
            sym = meta[idx][0] if idx < len(meta) else '?'
            near_misses.append(NearMiss(symbol=sym, score=float(stage1_score[idx]), rejection_reason=str(stage1_reasons[idx])))

        # Full stage-1 reject ledger (aggregated downstream by diagnostics)
        stage1_reject_rows = []
        all_rejected = np.where(~stage1_pass)[0]
        for idx in all_rejected:
            sym, sector, *_ = meta[idx]
            stage1_reject_rows.append({
                'symbol': sym,
                'sector': sector,
                'score': float(stage1_score[idx]),
                'reason': str(stage1_reasons[idx]),
                'blocked': bool(stage1_flags['blocked'][idx]),
                'event': bool(stage1_flags['event'][idx]),
                'time_filtered': bool(~stage1_flags['time_ok'][idx]),
                'strong_override': bool(stage1_flags.get('strong_override', np.zeros(X_all.shape[0], dtype=bool))[idx]),
            })

        if pass_idxs.size > self.settings.stage1_candidate_cap:
            order = np.argsort(stage1_score[pass_idxs])[::-1]
            pass_idxs = pass_idxs[order[:self.settings.stage1_candidate_cap]]

        with self.state.lock:
            self.state.model.pt1.trained = bundle is not None
            self.state.model.pt1.path = os.path.join(self.settings.model_dir, 'pt1')
            if bundle is not None:
                self.state.model.pt1.auc_val = bundle.meta.get('auc_val')
                self.state.model.pt1.brier_val = bundle.meta.get('brier_val')
                self.state.model.pt1.calibrator = bundle.meta.get('calibrator')
                self.state.model.pt1.class_weight = bundle.meta.get('class_weight')
                self.state.model.pt1.alpha = bundle.meta.get('alpha')
                self.state.model.pt1.touch_tail_validated = bundle.meta.get('touch_tail_validated')
                self.state.model.pt1.selection_tier = bundle.meta.get('selection_tier')
                self.state.model.pt1.selection_warning = bundle.meta.get('selection_warning')
                self.state.model.pt1.model_b_method = (bundle.meta.get('model_b') or {}).get('method')
                self.state.model.pt1.probability_contract = bundle.meta.get('probability_contract', 'uncapped_decomposed')
            else:
                # Explicit: model failed to load — clear ALL stale metadata
                self.state.model.pt1.trained = False
                self.state.model.pt1.model_source = 'heuristic'
                self.state.model.pt1.selection_warning = f'Model load failed: {bundle_status}'
                self.state.model.pt1.model_b_method = None
                self.state.model.pt1.auc_val = None
                self.state.model.pt1.brier_val = None
                self.state.model.pt1.calibrator = None
                self.state.model.pt1.class_weight = None
                self.state.model.pt1.alpha = None
                self.state.model.pt1.touch_tail_validated = None
                self.state.model.pt1.selection_tier = 'heuristic'
                self.state.model.pt1.probability_contract = None
                self.state.model.pt1.adaptive_touch_threshold = None

        if pass_idxs.size == 0:
            cov.stage2_scored_count = 0; cov.symbols_scored_count = 0
            cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
            self.state.set_scores([], run_utc); self.state.set_coverage(cov, skipped, near_misses, watchlist_rescue_rows=[])
            try: self.diagnostics.record_scan(trade_day, run_utc, open_utc, close_utc, cov, [], near_misses, stage1_reject_rows)
            except Exception: pass
            return

        X = X_all[pass_idxs]; meta_pass = [meta[i] for i in pass_idxs]
        # v10 Path A: predict_probs returns 5 values
        p1, model_source, _status1, p_touch, p_path = predict_probs(self.settings.model_dir, X, merged_meta)
        p1_raw = np.asarray(p1, dtype=float).copy()
        p_touch_arr = np.asarray(p_touch, dtype=float)
        p_path_arr = np.asarray(p_path, dtype=float)
        tail_validated = _status1 == 'validated'
        is_degraded = model_source in ('heuristic', 'trained_no_path')
        has_path_model = model_source == 'trained'

        risk_arr = np.array([m[4] for m in meta_pass], dtype=object)
        downside_risk = downside_risk_score_from_X(X, merged_meta)
        uncertainty_level, uncertainty_reasons = uncertainty_from_X(X, merged_meta)
        event_mask = event_risk_mask_from_X(X, merged_meta)
        high_downside = downside_risk >= self.settings.downside_high_threshold
        med_downside = (downside_risk >= self.settings.downside_medium_threshold) & (~high_downside)
        high_unc = uncertainty_level == 'HIGH'

        # Final-score guardrail discounting: learned shrinkage on the final calibrated contract
        acceptable = acceptable_long_mask_from_X(X, merged_meta)
        guardrail_profiles = (bundle.meta.get('guardrail_profiles') or {}) if bundle else {}
        guardrail_mult = np.ones_like(p1_raw, dtype=float)
        p_touch_pre_regime = p_touch_arr.copy()
        p1_pre_regime = p1_raw.copy()
        rows = []
        cov.capped_by_event_count = int(np.sum(event_mask))
        cov.capped_by_uncertainty_count = int(np.sum(high_unc))
        cov.capped_by_downside_count = int(np.sum(high_downside | med_downside))
        cov.guardrail_stats = {
            'blocked_in_universe': int(np.sum(stage1_flags['blocked'])),
            'event_in_universe': int(np.sum(stage1_flags['event'])),
            'weak_structure_in_universe': int(np.sum(stage1_flags.get('weak_structure', np.zeros(X_all.shape[0], dtype=bool)))),
            'high_downside_in_candidates': int(np.sum(high_downside)),
            'med_downside_in_candidates': int(np.sum(med_downside)),
            'high_uncertainty_in_candidates': int(np.sum(high_unc)),
            'event_in_candidates': int(np.sum(event_mask)),
            'regime_state': str(regime_decision.state or 'GREEN'),
        }

        # Adaptive thresholds: touch for context, decision for actual surfacing
        touch_tail_metrics = (bundle.meta.get('touch_tail_metrics') or {}) if bundle else {}
        adaptive_touch_threshold = float(touch_tail_metrics.get('adaptive_threshold', 0.0))
        if adaptive_touch_threshold <= 0:
            base_rate = float(touch_tail_metrics.get('base_touch_rate', 0.0))
            adaptive_touch_threshold = max(0.03, base_rate * self.settings.touch_tail_threshold_mult)

        decision_tail_metrics = (bundle.meta.get('decision_tail_metrics') or {}) if bundle else {}
        adaptive_decision_threshold = float(decision_tail_metrics.get('adaptive_threshold', 0.0))
        if adaptive_decision_threshold <= 0:
            base_rate = float(decision_tail_metrics.get('base_strict_rate', 0.0))
            adaptive_decision_threshold = max(0.03, base_rate * self.settings.touch_tail_threshold_mult)

        effective_touch_threshold = adaptive_touch_threshold * float(regime_decision.touch_threshold_mult or 1.0)
        effective_decision_threshold = adaptive_decision_threshold * float(regime_decision.touch_threshold_mult or 1.0)
        path_action_min = min(0.98, self.settings.path_quality_action_min + float(regime_decision.path_floor_add or 0.0))
        setup_families = classify_setup_families(X) if self.settings.specialist_families_enabled else np.array(['OTHER'] * X.shape[0], dtype=object)
        family_profiles = (bundle.meta.get('setup_family_profiles') or {}) if bundle else {}
        decision_tail_validated = bool((bundle.meta.get('decision_tail_validated')) if bundle else False)
        family_policy_counts = {'FULL_ACTIONABLE': 0, 'CANDIDATE_ONLY': 0, 'WATCHLIST_ONLY': 0, 'SUPPRESS': 0}
        family_actionable_block_count = 0
        family_watchlist_only_count = 0
        family_suppressed_count = 0
        limited_data_family_count = 0
        promoted_family_row_count = 0
        tightened_family_row_count = 0

        for j, (sym, sector, price, vwap, risk, risk_reasons, reasons) in enumerate(meta_pass):
            rr = risk_reasons
            if event_mask[j] and 'EVENT_RISK' not in rr:
                rr = (rr + ' EVENT_RISK').strip()
            if not decision_tail_validated and 'TAIL_UNVALIDATED' not in rr:
                rr = (rr + ' TAIL_UNVALIDATED').strip()
            display_tags = []
            if high_downside[j]:
                display_tags.append('DOWNSIDE_HIGH')
            elif med_downside[j]:
                display_tags.append('DOWNSIDE_MED')
            if uncertainty_level[j] == 'HIGH':
                display_tags.append('UNCERT_HIGH')
            elif uncertainty_level[j] == 'MED':
                display_tags.append('UNCERT_MED')
            display_reasons = reasons
            if stage1_reasons[pass_idxs[j]]:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + stage1_reasons[pass_idxs[j]]).strip()
            if display_tags:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + ' · '.join(display_tags)).strip()
            if not decision_tail_validated:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + 'TAIL_UNVALIDATED').strip()
            if is_degraded:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + f'DEGRADED({model_source})').strip()

            setup_family = str(setup_families[j])
            setup_family_bonus = family_bonus_for_live_row(family_profiles, setup_family)
            ttc_minutes = float(X[j, TTC_IDX]) * 390.0
            mins_since_open = 390.0 - ttc_minutes
            time_bucket = 'EARLY' if mins_since_open < 120.0 else ('LATE' if mins_since_open >= 240.0 else 'MID')
            _reg_state = str(regime_decision.state or 'NEUTRAL').upper()
            live_regime_bucket = 'STRESS' if _reg_state in {'AMBER', 'RED'} else 'NEUTRAL'
            family_live_policy = family_live_policy_for_row(
                family_profiles, setup_family, float(p1_raw[j]),
                regime_state=live_regime_bucket,
                time_bucket=time_bucket,
                sector_group=sector_group_from_sector(sector),
            )
            family_cal_mult = float(family_live_policy.get('calibration_multiplier') or 1.0)
            family_threshold_mult = float(family_live_policy.get('threshold_multiplier') or 1.0)
            family_signal_policy = str(family_live_policy.get('effective_policy') or 'CANDIDATE_ONLY')
            family_suppress = bool(family_live_policy.get('suppress_signals'))
            family_actionable_allowed = bool(family_live_policy.get('actionable_allowed'))
            family_candidate_allowed = bool(family_live_policy.get('candidate_allowed'))
            family_watchlist_allowed = bool(family_live_policy.get('watchlist_allowed'))
            family_reason_flags = list(family_live_policy.get('reason_flags') or [])
            family_policy_counts[family_signal_policy] = family_policy_counts.get(family_signal_policy, 0) + 1
            if 'FAMILY_LIMITED_DATA' in family_reason_flags:
                limited_data_family_count += 1
            if 'FAMILY_PROMOTE' in family_reason_flags or 'FAMILY_PROMOTE_CTX' in family_reason_flags:
                promoted_family_row_count += 1
            if 'FAMILY_TIGHTEN' in family_reason_flags or 'FAMILY_TIGHTEN_CTX' in family_reason_flags:
                tightened_family_row_count += 1

            row_guardrail_mult = self._guardrail_multiplier_for_row(
                guardrail_profiles,
                event_risk=bool(event_mask[j]),
                uncertainty=str(uncertainty_level[j]),
                high_downside=bool(high_downside[j]),
                medium_downside=bool(med_downside[j]),
            )
            guardrail_mult[j] = row_guardrail_mult
            p_touch_pre_regime[j] = float(np.clip(p_touch_arr[j] * row_guardrail_mult, 0.0, 0.999))
            p1_pre_regime[j] = float(np.clip(p1_raw[j] * row_guardrail_mult, 0.0, 0.999))

            regime_sector_mult = float(regime_decision.multiplier or 1.0) * float(self.regime_controller.sector_multiplier(regime_decision.state, sector))
            live_touch = float(np.clip(p_touch_pre_regime[j] * regime_sector_mult, 0.0, 0.999))
            live_prob = float(np.clip(p1_pre_regime[j] * regime_sector_mult * family_cal_mult, 0.0, 0.999))
            if regime_decision.prob_cap is not None:
                live_prob = min(live_prob, float(regime_decision.prob_cap))
            family_effective_threshold = max(0.03, float(effective_decision_threshold) * float(family_threshold_mult))
            regime_reason_text = ' · '.join(regime_decision.reasons or [])
            if regime_decision.state != 'GREEN':
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + f'REGIME_{regime_decision.state}').strip()
                if regime_reason_text:
                    display_reasons = (display_reasons + ' · ' + regime_reason_text).strip()
            for fam_flag in family_reason_flags:
                if fam_flag and fam_flag not in display_reasons:
                    display_reasons = (display_reasons + (' · ' if display_reasons else '') + fam_flag).strip()

            meets_criteria = (
                live_prob >= family_effective_threshold
                and p_path_arr[j] >= path_action_min
                and acceptable[j]
                and not event_mask[j]
                and not high_downside[j]
                and uncertainty_level[j] != 'HIGH'
                and not bool(regime_decision.suppress_new_signals)
            )
            raw_signal = ''
            if meets_criteria and decision_tail_validated and has_path_model and family_actionable_allowed:
                raw_signal = 'ACTIONABLE'
            elif meets_criteria and has_path_model and family_candidate_allowed:
                raw_signal = 'CANDIDATE'

            suppression_reason = ''
            signal = raw_signal
            if signal and family_suppress:
                signal = ''
                suppression_reason = 'FAMILY_SUPPRESS'
                family_suppressed_count += 1
            elif signal == 'ACTIONABLE' and not family_actionable_allowed:
                signal = 'CANDIDATE'
                family_actionable_block_count += 1
            elif signal == 'CANDIDATE' and not family_candidate_allowed:
                signal = ''
                suppression_reason = 'FAMILY_CANDIDATE_ONLY' if family_watchlist_allowed else 'FAMILY_SUPPRESS'
                if family_watchlist_allowed:
                    family_watchlist_only_count += 1
                else:
                    family_suppressed_count += 1
            if signal and (not decision_tail_validated) and self.settings.unvalidated_messy_suppress and self._is_unvalidated_messy(str(risk_arr[j]), rr):
                signal = ''
                suppression_reason = 'UNVALIDATED_MESSY'
            if not signal and not suppression_reason:
                if family_suppress:
                    suppression_reason = 'FAMILY_SUPPRESS'
                elif family_signal_policy == 'WATCHLIST_ONLY':
                    suppression_reason = 'FAMILY_WATCHLIST_ONLY'
                    family_watchlist_only_count += 1
                elif family_signal_policy == 'CANDIDATE_ONLY' and not family_actionable_allowed:
                    suppression_reason = 'FAMILY_CANDIDATE_ONLY'
            if signal:
                signal, cooldown_reason = self._apply_surfacing_cooldown(
                    sym,
                    run_utc,
                    signal,
                    float(live_prob),
                    float(p_path_arr[j]),
                )
                if cooldown_reason:
                    suppression_reason = cooldown_reason

            guardrail_flags = []
            if event_mask[j]:
                guardrail_flags.append('EVENT')
            if high_unc[j]:
                guardrail_flags.append('UNCERT_HIGH')
            elif str(uncertainty_level[j]) in {'MED', 'MEDIUM'}:
                guardrail_flags.append('UNCERT_MED')
            if high_downside[j]:
                guardrail_flags.append('DOWNSIDE_HIGH')
            elif med_downside[j]:
                guardrail_flags.append('DOWNSIDE_MED')
            if regime_decision.suppress_new_signals:
                suppression_reason = 'REGIME_SUPPRESS' if not suppression_reason else suppression_reason
            if suppression_reason:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + suppression_reason).strip()
            rows.append(ScoreRow(
                symbol=sym, sector=sector, price=float(price), vwap=float(vwap),
                prob_1=float(live_prob), risk=str(risk_arr[j]), risk_reasons=rr, reasons=display_reasons,
                downside_risk=float(downside_risk[j]), uncertainty=str(uncertainty_level[j]),
                uncertainty_reasons=str(uncertainty_reasons[j]), stage1_score=float(stage1_score[pass_idxs[j]]),
                stage1_reasons=str(stage1_reasons[pass_idxs[j]]), prob_1_raw=float(p1_raw[j]),
                prob_touch=float(live_touch), prob_touch_raw=float(p_touch_arr[j]), prob_touch_pre_regime=float(p_touch_pre_regime[j]),
                prob_path=float(p_path_arr[j]), prob_1_pre_regime=float(p1_pre_regime[j]), guardrail_multiplier=float(row_guardrail_mult), display_prob_threshold=float(family_effective_threshold),
                regime_state=str(regime_decision.state or 'GREEN'), regime_multiplier=float(regime_sector_mult),
                regime_prob_cap=float(regime_decision.prob_cap) if regime_decision.prob_cap is not None else None, regime_reasons=regime_reason_text,
                signal=signal, raw_signal=raw_signal, suppression_reason=suppression_reason,
                guardrail_flags=' '.join(guardrail_flags), acceptable=bool(acceptable[j]),
                event_risk=bool(event_mask[j]), high_downside=bool(high_downside[j]),
                medium_downside=bool(med_downside[j]), high_uncertainty=bool(high_unc[j]),
                tail_validated=bool(decision_tail_validated), display_touch_threshold=float(effective_touch_threshold),
                path_action_min=float(path_action_min), setup_family=setup_family, setup_family_bonus=float(setup_family_bonus),
            ))
        threshold_counts = {}
        acceptable_counts = {}
        for mult, tag in [
            (0.5, 'ge_low'),
            (1.0, 'ge_mid'),
            (1.5, 'ge_high'),
            (2.0, 'ge_top'),
        ]:
            threshold_counts[tag] = int(sum(1 for r in rows if (r.prob_1 or 0.0) >= (mult * float(r.display_prob_threshold or effective_decision_threshold))))
            acceptable_counts[tag] = int(sum(1 for r in rows if (r.prob_1 or 0.0) >= (mult * float(r.display_prob_threshold or effective_decision_threshold)) and r.acceptable))
        cov.threshold_counts = threshold_counts
        cov.acceptable_threshold_counts = acceptable_counts

        watchlist_rescue_rows = [] if regime_decision.suppress_new_signals else self._select_watchlist_rescue_rows(rows, effective_decision_threshold, path_action_min)
        watchlist_symbols = {r.symbol for r in watchlist_rescue_rows}
        for row in rows:
            if row.symbol in watchlist_symbols and 'WATCHLIST_RESCUE' not in row.reasons:
                row.reasons = (row.reasons + (' · ' if row.reasons else '') + 'WATCHLIST_RESCUE').strip()
        cov.guardrail_stats['watchlist_rescue_count'] = len(watchlist_rescue_rows)
        cov.guardrail_stats['family_signal_policy_counts'] = family_policy_counts
        cov.guardrail_stats['family_actionable_block_count'] = int(family_actionable_block_count)
        cov.guardrail_stats['family_watchlist_only_count'] = int(family_watchlist_only_count)
        cov.guardrail_stats['family_suppressed_count'] = int(family_suppressed_count)
        cov.guardrail_stats['limited_data_family_count'] = int(limited_data_family_count)
        cov.guardrail_stats['promoted_family_row_count'] = int(promoted_family_row_count)
        cov.guardrail_stats['tightened_family_row_count'] = int(tightened_family_row_count)
        for row in rows:
            row.relative_strength_score = self._compute_relative_strength_score(row)
        self._apply_relative_strength_ranks(rows)
        for row in watchlist_rescue_rows:
            row.relative_strength_score = self._compute_relative_strength_score(row)
        self._apply_relative_strength_ranks(watchlist_rescue_rows)
        rows.sort(key=lambda r: (r.relative_strength_rank or 999999))
        watchlist_rescue_rows.sort(key=lambda r: (r.relative_strength_rank or 999999))

        # Update model status with ACTUAL runtime state, not just metadata
        with self.state.lock:
            self.state.model.pt1.model_source = model_source
            if model_source == 'trained':
                self.state.model.pt1.model_b_method = (bundle.meta.get('model_b') or {}).get('method') if bundle else None
            elif model_source == 'trained_no_path':
                self.state.model.pt1.model_b_method = None  # Actually missing at runtime
            else:
                self.state.model.pt1.model_b_method = None

        # Concentration warning: flag if top candidates are sector-clustered
        top_n = min(10, len(rows))
        if top_n >= 3:
            from collections import Counter
            sub_sectors = Counter()
            for r in rows[:top_n]:
                sub_sectors[r.sector] += 1
            dominant = sub_sectors.most_common(1)[0] if sub_sectors else ('', 0)
            if dominant[1] >= 3:
                for r in rows[:top_n]:
                    if r.sector == dominant[0] and 'CONCENTRATED' not in r.reasons:
                        r.reasons = (r.reasons + (' · ' if r.reasons else '') + f'CONCENTRATED({dominant[0][:12]}:{dominant[1]}/{top_n})').strip()
        cov.stage2_scored_count = len(rows); cov.symbols_scored_count = len(rows)
        cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
        self.state.set_scores(rows, run_utc); self.state.set_coverage(cov, skipped, near_misses, watchlist_rescue_rows=watchlist_rescue_rows)
        try: self.diagnostics.record_scan(trade_day, run_utc, open_utc, close_utc, cov, rows, near_misses, stage1_reject_rows)
        except Exception: pass

    def _loop(self):
        try: self.load_constituents()
        except Exception as e:
            with self.state.lock: self.state.constituents.source = 'fallback'; self.state.constituents.warning = f'failed: {e}'
        while not self._stop.is_set():
            open_utc, close_utc, is_open, _ = self._update_market_status()
            now = _utcnow()
            if is_open:
                try:
                    self.scan_once(open_utc, close_utc, now)
                except Exception as e:
                    self.state.set_error(f'scan error: {e}')
                    self._publish_regime(self.regime_controller.unavailable_status(
                        reason='SCAN_ERROR',
                        note=f'Live geopolitical regime evaluation failed during scan: {e}',
                        now_utc=now,
                        market_session='LIVE',
                    ))
            else:
                try:
                    self._publish_regime(self.regime_controller.closed_status(now, open_utc, close_utc, self.cache_5m.bars))
                except Exception as e:
                    self._publish_regime(self.regime_controller.unavailable_status(
                        reason='CLOSED_STATUS_ERROR',
                        note=f'Could not restore last live regime state after market close: {e}',
                        now_utc=now,
                        market_session='CLOSED',
                    ))
                try:
                    client = self._make_client()
                    self.diagnostics.maybe_evaluate_closed_trade_day(client, now)
                except Exception:
                    pass
            nxt = next_aligned_run(now, self.settings.timezone, self.settings.scan_interval_minutes, 3)
            self._stop.wait(timeout=max(1.0, (nxt - _utcnow()).total_seconds()))
