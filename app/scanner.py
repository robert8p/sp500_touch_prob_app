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
from .features import compute_features_from_5m
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
from .state import AppState, CoverageStatus, ScoreRow, SkippedSymbol
from .volume_profiles import VolumeProfileStore, slot_index_from_ts


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts: str) -> datetime:
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts)


def _previous_trading_day(d: date, tz_name: str) -> date:
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('XNYS')
        start = d - timedelta(days=10)
        sched = cal.schedule(start_date=start, end_date=d)
        days = [idx.date() for idx in sched.index if idx.date() < d]
        if days:
            return days[-1]
    except Exception:
        pass
    x = d - timedelta(days=1)
    while x.weekday() >= 5:
        x -= timedelta(days=1)
    return x


def _session_utc_for_day(d: date, tz_name: str) -> Tuple[datetime, datetime]:
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
    except Exception:
        pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


def _daily_ctx_from_bars(daily_bars: List[dict], trade_day: date) -> Dict[str, float]:
    rows = []
    for b in daily_bars:
        try:
            ts = _parse_ts(b['t']).date()
        except Exception:
            continue
        if ts < trade_day:
            rows.append(b)
    if not rows:
        return {}
    closes = np.array([float(b.get('c') or 0.0) for b in rows], dtype=float)
    highs = np.array([float(b.get('h') or 0.0) for b in rows], dtype=float)
    valid = closes > 0
    if not np.any(valid):
        return {}
    closes = closes[valid]
    highs = highs[valid]
    last = float(closes[-1])

    def ret_n(n: int) -> float:
        if closes.size <= n:
            return 0.0
        ref = float(closes[-(n + 1)])
        return float(last / ref - 1.0) if ref > 0 else 0.0

    def sma_dist(n: int) -> float:
        if closes.size < n:
            return 0.0
        ma = float(np.mean(closes[-n:]))
        return float(last / ma - 1.0) if ma > 0 else 0.0

    def drawdown_n(n: int) -> float:
        if highs.size >= n:
            peak = float(np.max(highs[-n:]))
        else:
            peak = float(np.max(highs)) if highs.size else 0.0
        return float(last / peak - 1.0) if peak > 0 else 0.0

    return {
        'ret_5d': ret_n(5),
        'ret_20d': ret_n(20),
        'ret_60d': ret_n(60),
        'dist_20dma_pct': sma_dist(20),
        'dist_50dma_pct': sma_dist(50),
        'dist_200dma_pct': sma_dist(200),
        'drawdown_20d_pct': drawdown_n(20),
        'drawdown_60d_pct': drawdown_n(60),
    }


def _runtime_meta(settings: Settings) -> Dict[str, object]:
    return {
        'liq_dvol_min_usd': float(settings.liq_dvol_min_usd),
        'liq_range_pct_max': float(settings.liq_range_pct_max),
        'liq_wick_atr_max': float(settings.liq_wick_atr_max),
        'blocked_ret20d_max': float(settings.blocked_ret20d_max),
        'blocked_ret60d_max': float(settings.blocked_ret60d_max),
        'blocked_dist50dma_max': float(settings.blocked_dist50dma_max),
        'blocked_ret_since_open_max': float(settings.blocked_ret_since_open_max),
        'blocked_damage_from_high_atr_min': float(settings.blocked_damage_from_high_atr_min),
        'blocked_below_vwap_frac_min': float(settings.blocked_below_vwap_frac_min),
        'blocked_prob_cap': float(settings.blocked_prob_cap),
        'event_gap_abs_min': float(settings.event_gap_abs_min),
        'event_rvol_min': float(settings.event_rvol_min),
        'event_range_pct_min': float(settings.event_range_pct_min),
        'event_prob_cap': float(settings.event_prob_cap),
        'uncertainty_z_thresh': float(settings.uncertainty_z_thresh),
        'uncertainty_extreme_features_min': int(settings.uncertainty_extreme_features_min),
        'uncertainty_prob_cap': float(settings.uncertainty_prob_cap),
        'downside_prob_cap_high': float(settings.downside_prob_cap_high),
        'downside_prob_cap_medium': float(settings.downside_prob_cap_medium),
        'downside_high_threshold': float(settings.downside_high_threshold),
        'downside_medium_threshold': float(settings.downside_medium_threshold),
        'stage1_candidate_cap': int(settings.stage1_candidate_cap),
        'stage1_min_score': float(settings.stage1_min_score),
        'stage1_min_minutes_to_close': int(settings.stage1_min_minutes_to_close),
        'stage1_min_rvol': float(settings.stage1_min_rvol),
        'stage1_min_dollar_volume_mult': float(settings.stage1_min_dollar_volume_mult),
    }


@dataclass
class BarsCache:
    timeframe: str
    bars: Dict[str, List[dict]] = field(default_factory=dict)
    last_fetch_utc: Optional[datetime] = None
    last_bar_ts: Optional[str] = None

    def merge(self, symbol: str, new_bars: List[dict], keep: int = 900) -> None:
        existing = self.bars.get(symbol, [])
        seen = {b.get('t') for b in existing}
        merged = existing + [b for b in new_bars if b.get('t') not in seen]
        merged.sort(key=lambda x: x.get('t', ''))
        if len(merged) > keep:
            merged = merged[-keep:]
        self.bars[symbol] = merged


class Scanner:
    def __init__(self, settings: Settings, state: AppState):
        self.settings = settings
        self.state = state
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.constituents: List[Constituent] = []
        self.symbol_meta: Dict[str, Constituent] = {}
        self.cache_5m = BarsCache(timeframe='5Min')
        self.cache_1d = BarsCache(timeframe='1Day')
        self.daily_cache_trade_day: Optional[date] = None
        self.vol_profiles = VolumeProfileStore(settings.model_dir)
        self.diagnostics = DiagnosticJournal(settings)

    def load_constituents(self) -> None:
        fallback = load_fallback()
        source = 'fallback'
        warning = None
        data = fallback
        refreshed, err = try_refresh_from_wikipedia()
        if refreshed is not None:
            data = refreshed
            source = 'wikipedia'
        elif err:
            warning = f'refresh blocked/unavailable: {err}'
        normed = []
        for c in data:
            sym = normalize_symbol(c.symbol)
            normed.append(Constituent(symbol=sym, name=c.name, sector=c.sector, industry=c.industry))
        self.constituents = normed
        self.symbol_meta = {c.symbol: c for c in normed}
        with self.state.lock:
            self.state.constituents.source = source
            self.state.constituents.warning = warning
            self.state.constituents.count = len(normed)
            self.state.coverage.universe_count = len(normed)

    def start(self) -> None:
        if self.settings.disable_scheduler:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._loop, daemon=True)
        self._thread = t
        t.start()

    def _make_client(self) -> Optional[AlpacaClient]:
        if self.settings.demo_mode:
            return None
        if not (self.settings.alpaca_api_key and self.settings.alpaca_api_secret):
            return None
        return AlpacaClient(self.settings.alpaca_api_key, self.settings.alpaca_api_secret, feed=self.settings.normalized_feed())

    def _update_market_status(self):
        now = _utcnow()
        open_utc, close_utc, is_open, ttc = get_market_times(now, self.settings.timezone)
        with self.state.lock:
            self.state.market.market_open = is_open
            self.state.market.time_to_close_seconds = ttc
            self.state.market.market_open_time = iso(open_utc)
            self.state.market.market_close_time = iso(close_utc)
        return open_utc, close_utc, is_open, ttc

    def _demo_scores(self) -> List[ScoreRow]:
        demo = [('AAPL', 'Information Technology'), ('MSFT', 'Information Technology'), ('NVDA', 'Information Technology'), ('AMZN', 'Consumer Discretionary'), ('JPM', 'Financials')]
        rows = []
        for i, (sym, sec) in enumerate(demo):
            price = 100 + i * 25
            vwap = price - 0.3 + 0.1 * i
            p1 = min(0.95, 0.56 + 0.08 * i)
            risk = 'OK' if i < 3 else 'CAUTION'
            rows.append(ScoreRow(sym, sec, float(price), float(vwap), float(p1), risk, 'DEMO', 'DEMO', 0.18 + 0.08 * i, 'LOW', '', 2.0 + i * 0.35, 'LIQ TREND DEMO', float(p1)))
        rows.sort(key=lambda r: r.prob_1, reverse=True)
        return rows

    def _refresh_daily_cache_if_needed(self, client: AlpacaClient, symbols_requested: List[str], trade_day: date) -> Optional[str]:
        if self.daily_cache_trade_day == trade_day and self.cache_1d.bars:
            return None
        start_utc = datetime.now(timezone.utc) - timedelta(days=500)
        bars1d, err, _ = client.get_bars(symbols_requested, timeframe='1Day', start_utc=start_utc, end_utc=datetime.now(timezone.utc), limit=None)
        if err:
            return err
        self.cache_1d.bars = {k: list(v) for k, v in bars1d.items()}
        self.cache_1d.last_fetch_utc = datetime.now(timezone.utc)
        self.daily_cache_trade_day = trade_day
        return None

    def scan_once(self, open_utc: datetime, close_utc: datetime, now_utc: datetime) -> None:
        run_utc = now_utc.isoformat().replace('+00:00', 'Z')
        if not self.constituents:
            self.load_constituents()
        universe_symbols = [c.symbol for c in self.constituents]
        sector_etfs = unique_sector_etfs([self.symbol_meta[s].sector for s in universe_symbols if s in self.symbol_meta])
        symbols_requested = list(dict.fromkeys(universe_symbols + ['SPY'] + sector_etfs))
        cov = CoverageStatus(universe_count=len(universe_symbols), symbols_requested_count=len(symbols_requested))
        skip_counts = {'no_bars': 0, 'insufficient_bars': 0, 'missing_price_or_vwap': 0, 'other_errors': 0, 'model_schema_incompatible': 0}
        skipped: List[SkippedSymbol] = []

        if self.settings.demo_mode:
            with self.state.lock:
                self.state.alpaca.ok = True
                self.state.alpaca.message = 'DEMO_MODE'
                self.state.alpaca.feed = 'sip'
                self.state.alpaca.last_request_utc = run_utc
                self.state.alpaca.last_bar_timestamp = run_utc
                self.state.alpaca.rate_limit_warn = None
                self.state.model.pt1.trained = False
                self.state.model.pt1.path = os.path.join(self.settings.model_dir, 'pt1')
            rows = self._demo_scores()
            cov.symbols_scored_count = len(rows)
            cov.stage1_candidate_count = len(rows)
            cov.stage2_scored_count = len(rows)
            cov.threshold_counts = {'ge_0_60': int(sum(r.prob_1 >= 0.60 for r in rows)), 'ge_0_70': int(sum(r.prob_1 >= 0.70 for r in rows)), 'ge_0_75': int(sum(r.prob_1 >= 0.75 for r in rows)), 'ge_0_80': int(sum(r.prob_1 >= 0.80 for r in rows))}
            cov.acceptable_threshold_counts = dict(cov.threshold_counts)
            self.state.set_scores(rows, run_utc)
            self.state.set_coverage(cov, [])
            return

        client = self._make_client()
        if client is None:
            self.state.set_error('Alpaca keys missing (set ALPACA_API_KEY/ALPACA_API_SECRET) or DEMO_MODE=true')
            with self.state.lock:
                self.state.alpaca.ok = False
                self.state.alpaca.message = 'Missing keys'
                self.state.alpaca.feed = self.settings.normalized_feed()
            cov.top_skip_reasons = {'missing_keys': len(universe_symbols)}
            self.state.set_coverage(cov, [])
            return

        trade_day = open_utc.astimezone(ZoneInfo(self.settings.timezone)).date()
        daily_err = self._refresh_daily_cache_if_needed(client, symbols_requested, trade_day)
        if daily_err:
            self.state.set_error(f'Alpaca daily error: {daily_err}')
            cov.top_skip_reasons = {'alpaca_daily_error': len(universe_symbols)}
            self.state.set_coverage(cov, [])
            return

        if self.cache_5m.last_fetch_utc is None:
            prev_d = _previous_trading_day(trade_day, self.settings.timezone)
            prev_open_utc, _ = _session_utc_for_day(prev_d, self.settings.timezone)
            start = prev_open_utc
        else:
            start = self.cache_5m.last_fetch_utc - timedelta(minutes=10)
        end = now_utc
        bars_by_sym, err, warn = client.get_bars(symbols_requested, timeframe='5Min', start_utc=start, end_utc=end, limit=None)
        with self.state.lock:
            self.state.alpaca.feed = client.feed
            self.state.alpaca.last_request_utc = run_utc
            self.state.alpaca.rate_limit_warn = warn
            if err:
                self.state.alpaca.ok = False
                self.state.alpaca.message = err
            else:
                self.state.alpaca.ok = True
                self.state.alpaca.message = 'OK'
        if err:
            self.state.set_error(f'Alpaca error: {err}')
            cov.top_skip_reasons = {'alpaca_error': len(universe_symbols)}
            self.state.set_coverage(cov, [])
            return

        cov.symbols_returned_with_bars_count = sum(1 for k, v in bars_by_sym.items() if k not in set(['SPY'] + sector_etfs) and v)
        for sym, lst in bars_by_sym.items():
            self.cache_5m.merge(sym, lst)
            if lst:
                self.cache_5m.last_bar_ts = lst[-1].get('t')
        self.cache_5m.last_fetch_utc = end
        with self.state.lock:
            self.state.alpaca.last_bar_timestamp = self.cache_5m.last_bar_ts

        mins_to_close = max(0.0, (close_utc - now_utc).total_seconds() / 60.0)
        prev_d = _previous_trading_day(trade_day, self.settings.timezone)
        prev_open_utc, prev_close_utc = _session_utc_for_day(prev_d, self.settings.timezone)

        def current_session_bars(lst: List[dict]) -> List[dict]:
            out = []
            for b in lst:
                t = _parse_ts(b['t'])
                if open_utc <= t <= close_utc:
                    out.append(b)
            return out

        spy_cur = current_session_bars(self.cache_5m.bars.get('SPY', []))
        spy_ret_30m = 0.0
        if len(spy_cur) >= 7:
            c = np.array([float(b.get('c') or 0.0) for b in spy_cur], dtype=float)
            spy_ret_30m = float(c[-1] / c[-7] - 1.0) if c[-7] else 0.0

        sector_ret_map: Dict[str, float] = {}
        for etf in sector_etfs:
            cur = current_session_bars(self.cache_5m.bars.get(etf, []))
            if len(cur) >= 7:
                cc = np.array([float(b.get('c') or 0.0) for b in cur], dtype=float)
                sector_ret_map[etf] = float(cc[-1] / cc[-7] - 1.0) if cc[-7] else 0.0
            else:
                sector_ret_map[etf] = 0.0

        avail, _ = self.vol_profiles.availability_counts()
        cov.profile_symbols_available = avail
        cov.profile_symbols_missing = max(0, cov.universe_count - avail)
        if avail == 0:
            cov.profile_note = 'ToD RVOL profiles not found; using leakage-free rolling fallback'

        feats = []
        meta = []
        sufficient_count = 0
        risk_params = (self.settings.liq_rolling_bars, self.settings.liq_dvol_min_usd, self.settings.liq_range_pct_max, self.settings.liq_wick_atr_max)
        blocked_params = {
            'ret20d_max': self.settings.blocked_ret20d_max,
            'ret60d_max': self.settings.blocked_ret60d_max,
            'dist50dma_max': self.settings.blocked_dist50dma_max,
            'ret_since_open_max': self.settings.blocked_ret_since_open_max,
            'damage_from_high_atr_min': self.settings.blocked_damage_from_high_atr_min,
            'below_vwap_frac_min': self.settings.blocked_below_vwap_frac_min,
            'event_gap_abs_min': self.settings.event_gap_abs_min,
            'event_rvol_min': self.settings.event_rvol_min,
            'event_range_pct_min': self.settings.event_range_pct_min,
        }

        for sym in universe_symbols:
            bars_raw = self.cache_5m.bars.get(sym, [])
            bars = []
            for b in bars_raw:
                t = _parse_ts(b['t'])
                if (prev_open_utc <= t <= prev_close_utc) or (open_utc <= t <= close_utc):
                    bars.append(b)
            if not bars:
                skip_counts['no_bars'] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason='no_bars', last_bar_timestamp=None))
                continue
            if len(bars) < self.settings.min_bars_5m:
                skip_counts['insufficient_bars'] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason='insufficient_bars', last_bar_timestamp=bars[-1].get('t')))
                continue
            current_bars = current_session_bars(bars)
            if not current_bars:
                skip_counts['no_bars'] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason='no_bars', last_bar_timestamp=bars[-1].get('t')))
                continue
            sufficient_count += 1
            last_ts = current_bars[-1].get('t')
            slot_idx = None
            if last_ts:
                try:
                    slot_idx = slot_index_from_ts(datetime.fromisoformat(last_ts.replace('Z', '+00:00')), self.settings.timezone)
                except Exception:
                    slot_idx = None
            baseline = self.vol_profiles.get_slot_median(sym, slot_idx) if slot_idx is not None else None
            prev_session = []
            for b in bars:
                t = _parse_ts(b['t'])
                if prev_open_utc <= t <= prev_close_utc:
                    prev_session.append(b)
            prev_day_close = float(prev_session[-1].get('c') or 0.0) if prev_session else None
            prev_day_high = float(max(float(b.get('h') or 0.0) for b in prev_session)) if prev_session else None
            prev_day_low = float(min(float(b.get('l') or 0.0) for b in prev_session)) if prev_session else None
            sector_etf = sector_etf_for_sector(self.symbol_meta.get(sym).sector if sym in self.symbol_meta else '')
            sector_ret = sector_ret_map.get(sector_etf, 0.0)
            daily_ctx = _daily_ctx_from_bars(self.cache_1d.bars.get(sym, []), trade_day)
            try:
                fr = compute_features_from_5m(
                    bars_5m=bars,
                    spy_ret_30m=spy_ret_30m,
                    sector_ret_30m=sector_ret,
                    mins_to_close=mins_to_close,
                    tod_baseline_vol_median=baseline,
                    rolling_rvol_window=20,
                    risk_params=risk_params,
                    prev_day_close=prev_day_close,
                    prev_day_high=prev_day_high,
                    prev_day_low=prev_day_low,
                    daily_ctx=daily_ctx,
                    tz_name=self.settings.timezone,
                    blocked_params=blocked_params,
                )
            except Exception:
                fr = None
            if fr is None:
                skip_counts['missing_price_or_vwap'] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason='missing_price_or_vwap', last_bar_timestamp=last_ts))
                continue
            sector = self.symbol_meta.get(sym).sector if sym in self.symbol_meta else 'Unknown'
            feats.append(fr.features)
            meta.append((sym, sector, fr.price, fr.vwap, fr.risk, fr.risk_reasons, fr.reasons))

        cov.symbols_with_sufficient_bars_count = sufficient_count
        if not feats:
            cov.symbols_scored_count = 0
            cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
            self.state.set_scores([], run_utc)
            self.state.set_coverage(cov, skipped)
            return

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

        if pass_idxs.size > self.settings.stage1_candidate_cap:
            order = np.argsort(stage1_score[pass_idxs])[::-1]
            pass_idxs = pass_idxs[order[:self.settings.stage1_candidate_cap]]

        if bundle_status == 'incompatible':
            skip_counts['model_schema_incompatible'] = len(universe_symbols)
            with self.state.lock:
                if not self.state.last_error:
                    self.state.last_error = 'Model schema mismatch; retrain required.'

        with self.state.lock:
            self.state.model.pt1.trained = bundle is not None
            self.state.model.pt1.path = os.path.join(self.settings.model_dir, 'pt1')
            if bundle is not None:
                self.state.model.pt1.auc_val = bundle.meta.get('auc_val')
                self.state.model.pt1.brier_val = bundle.meta.get('brier_val')
                self.state.model.pt1.calibrator = bundle.meta.get('calibrator')
                self.state.model.pt1.class_weight = bundle.meta.get('class_weight')
                self.state.model.pt1.alpha = bundle.meta.get('alpha')
                self.state.model.pt1.tail_ready_70 = bundle.meta.get('tail_ready_70')
                self.state.model.pt1.tail_ready_75 = bundle.meta.get('tail_ready_75')
                self.state.model.pt1.selection_tier = bundle.meta.get('selection_tier')
                self.state.model.pt1.selection_warning = bundle.meta.get('selection_warning')
                self.state.model.pt1.live_prob_cap = bundle.meta.get('live_prob_cap')

        if pass_idxs.size == 0:
            cov.stage2_scored_count = 0
            cov.symbols_scored_count = 0
            cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
            self.state.set_scores([], run_utc)
            self.state.set_coverage(cov, skipped)
            try:
                self.diagnostics.record_scan(trade_day, run_utc, open_utc, close_utc, cov, [], self.state.model.pt1.live_prob_cap)
            except Exception:
                pass
            return

        X = X_all[pass_idxs]
        meta_pass = [meta[i] for i in pass_idxs]
        p1, _src1, _status1 = predict_probs(self.settings.model_dir, X, merged_meta)
        p1_raw = np.asarray(p1, dtype=float).copy()
        tail_cap_active = _status1 == 'tail_capped'
        risk_arr = np.array([m[4] for m in meta_pass], dtype=object)
        downside_risk = downside_risk_score_from_X(X, merged_meta)
        uncertainty_level, uncertainty_reasons = uncertainty_from_X(X, merged_meta)
        event_mask = event_risk_mask_from_X(X, merged_meta)
        high_downside = downside_risk >= self.settings.downside_high_threshold
        med_downside = (downside_risk >= self.settings.downside_medium_threshold) & (~high_downside)
        high_unc = uncertainty_level == 'HIGH'

        if np.any(event_mask):
            p1_raw[event_mask] = np.minimum(p1_raw[event_mask], self.settings.event_prob_cap)
        if np.any(high_unc):
            p1_raw[high_unc] = np.minimum(p1_raw[high_unc], self.settings.uncertainty_prob_cap)
        if np.any(high_downside):
            p1_raw[high_downside] = np.minimum(p1_raw[high_downside], self.settings.downside_prob_cap_high)
        if np.any(med_downside):
            p1_raw[med_downside] = np.minimum(p1_raw[med_downside], self.settings.downside_prob_cap_medium)

        rows: List[ScoreRow] = []
        acceptable = acceptable_long_mask_from_X(X, merged_meta)
        cov.capped_by_event_count = int(np.sum(event_mask))
        cov.capped_by_uncertainty_count = int(np.sum(high_unc))
        cov.capped_by_downside_count = int(np.sum(high_downside | med_downside))
        cov.guardrail_stats = {
            'blocked_in_universe': int(np.sum(stage1_flags['blocked'])),
            'event_in_universe': int(np.sum(stage1_flags['event'])),
            'high_downside_in_candidates': int(np.sum(high_downside)),
            'high_uncertainty_in_candidates': int(np.sum(high_unc)),
        }

        threshold_counts = {}
        acceptable_counts = {}
        for thr, tag in [(0.60, 'ge_0_60'), (0.70, 'ge_0_70'), (0.75, 'ge_0_75'), (0.80, 'ge_0_80')]:
            threshold_counts[tag] = int(np.sum(p1_raw >= thr))
            acceptable_counts[tag] = int(np.sum((p1_raw >= thr) & acceptable))
        cov.threshold_counts = threshold_counts
        cov.acceptable_threshold_counts = acceptable_counts

        for j, (sym, sector, price, vwap, risk, risk_reasons, reasons) in enumerate(meta_pass):
            rr = risk_reasons
            if event_mask[j] and 'EVENT_RISK' not in rr:
                rr = (rr + ' EVENT_RISK').strip()
            if tail_cap_active and 'MODEL_TAIL_UNVALIDATED' not in rr:
                rr = (rr + ' MODEL_TAIL_UNVALIDATED').strip()
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
            if stage1_reasons[j]:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + stage1_reasons[j]).strip()
            if display_tags:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + ' · '.join(display_tags)).strip()
            if tail_cap_active:
                display_reasons = (display_reasons + (' · ' if display_reasons else '') + 'MODEL_TAIL_CAP').strip()
            rows.append(
                ScoreRow(
                    symbol=sym,
                    sector=sector,
                    price=float(price),
                    vwap=float(vwap),
                    prob_1=float(p1_raw[j]),
                    risk=str(risk_arr[j]),
                    risk_reasons=rr,
                    reasons=display_reasons,
                    downside_risk=float(downside_risk[j]),
                    uncertainty=str(uncertainty_level[j]),
                    uncertainty_reasons=str(uncertainty_reasons[j]),
                    stage1_score=float(stage1_score[pass_idxs[j]]),
                    stage1_reasons=str(stage1_reasons[pass_idxs[j]]),
                    prob_1_raw=float(p1_raw[j]),
                )
            )
        rows.sort(key=lambda r: r.prob_1, reverse=True)
        cov.stage2_scored_count = len(rows)
        cov.symbols_scored_count = len(rows)
        cov.top_skip_reasons = {k: v for k, v in skip_counts.items() if v > 0}
        self.state.set_scores(rows, run_utc)
        self.state.set_coverage(cov, skipped)
        try:
            self.diagnostics.record_scan(trade_day, run_utc, open_utc, close_utc, cov, rows, self.state.model.pt1.live_prob_cap)
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(self.settings.model_dir), exist_ok=True)
            with open(os.path.join(os.path.dirname(self.settings.model_dir), 'last_scores.json'), 'w', encoding='utf-8') as f:
                json.dump({'last_run_utc': run_utc, 'rows': [r.__dict__ for r in rows]}, f)
        except Exception:
            pass

    def _loop(self) -> None:
        try:
            self.load_constituents()
        except Exception as e:
            with self.state.lock:
                self.state.constituents.source = 'fallback'
                self.state.constituents.warning = f'failed to load constituents: {e}'
        while not self._stop.is_set():
            open_utc, close_utc, is_open, _ = self._update_market_status()
            now = _utcnow()
            if is_open:
                try:
                    self.scan_once(open_utc, close_utc, now)
                except Exception as e:
                    self.state.set_error(f'scan error: {e}')
            else:
                try:
                    client = self._make_client()
                    self.diagnostics.maybe_evaluate_closed_trade_day(client, now)
                except Exception:
                    pass
            nxt = next_aligned_run(now, self.settings.timezone, self.settings.scan_interval_minutes, 3)
            self._stop.wait(timeout=max(1.0, (nxt - _utcnow()).total_seconds()))
