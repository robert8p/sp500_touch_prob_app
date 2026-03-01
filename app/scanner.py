from __future__ import annotations
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alpaca import AlpacaClient
from .config import Settings
from .constituents import Constituent, load_fallback, try_refresh_from_wikipedia, normalize_symbol
from .features import compute_features_from_5m
from .market import get_market_times, iso, next_aligned_run
from .modeling import predict_probs
from .state import AppState, CoverageStatus, ScoreRow, SkippedSymbol
from .volume_profiles import VolumeProfileStore, slot_index_from_ts

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

@dataclass
class BarsCache:
    timeframe: str
    bars: Dict[str, List[dict]] = field(default_factory=dict)
    last_fetch_utc: Optional[datetime] = None
    last_bar_ts: Optional[str] = None

    def merge(self, symbol: str, new_bars: List[dict], keep: int = 160) -> None:
        if not new_bars:
            return
        existing = self.bars.get(symbol, [])
        seen = {b.get("t") for b in existing}
        merged = existing + [b for b in new_bars if b.get("t") not in seen]
        merged.sort(key=lambda x: x.get("t",""))
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
        self.cache_5m = BarsCache(timeframe="5Min")

        self.vol_profiles = VolumeProfileStore(settings.model_dir)

    def load_constituents(self) -> None:
        fallback = load_fallback()
        source = "fallback"
        warning = None
        data = fallback

        refreshed, err = try_refresh_from_wikipedia()
        if refreshed is not None:
            data = refreshed
            source = "wikipedia"
        elif err:
            warning = f"refresh blocked/unavailable: {err}"

        normed: List[Constituent] = []
        for c in data:
            sym = normalize_symbol(c.symbol)
            normed.append(Constituent(symbol=sym, name=c.name, sector=c.sector, industry=c.industry))

        self.constituents = normed
        self.symbol_meta = {c.symbol: c for c in normed}

        with self.state.lock:
            self.state.constituents.source = source
            self.state.constituents.warning = warning
            self.state.constituents.count = len(normed)

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

    def _update_market_status(self) -> Tuple[datetime, datetime, bool, int]:
        now = _utcnow()
        open_utc, close_utc, is_open, ttc = get_market_times(now, self.settings.timezone)
        with self.state.lock:
            self.state.market.market_open = is_open
            self.state.market.time_to_close_seconds = ttc
            self.state.market.market_open_time = iso(open_utc)
            self.state.market.market_close_time = iso(close_utc)
        return open_utc, close_utc, is_open, ttc

    def _demo_scores(self) -> List[ScoreRow]:
        demo=[("AAPL","Information Technology"),("MSFT","Information Technology"),("NVDA","Information Technology"),
              ("AMZN","Consumer Discretionary"),("JPM","Financials")]
        rows=[]
        for i,(sym,sec) in enumerate(demo):
            price=100+i*20
            vwap=price-0.4
            p2=min(0.95, 0.55+0.08*i)
            p1=min(0.98, p2+0.15)
            rows.append(ScoreRow(sym, sec, float(price), float(vwap), float(p1), float(p2), "OK", "DEMO"))
        rows.sort(key=lambda r: r.prob_2, reverse=True)
        return rows

    def scan_once(self, open_utc: datetime, close_utc: datetime, now_utc: datetime) -> None:
        run_utc = now_utc.isoformat().replace("+00:00","Z")

        # Ensure constituents loaded
        if not self.constituents:
            self.load_constituents()

        universe_symbols = [c.symbol for c in self.constituents]
        symbols_requested = universe_symbols + ["SPY"]

        cov = CoverageStatus(
            universe_count=len(universe_symbols),
            symbols_requested_count=len(symbols_requested),
        )
        skip_counts: Dict[str,int] = {"no_bars":0, "insufficient_bars":0, "missing_price_or_vwap":0, "other_errors":0, "model_schema_incompatible":0}
        skipped: List[SkippedSymbol] = []

        if self.settings.demo_mode:
            with self.state.lock:
                self.state.alpaca.ok = True
                self.state.alpaca.message = "DEMO_MODE"
                self.state.alpaca.feed = "sip"
                self.state.alpaca.last_request_utc = run_utc
                self.state.alpaca.last_bar_timestamp = run_utc
                self.state.alpaca.rate_limit_warn = None
            rows = self._demo_scores()
            cov.symbols_scored_count = len(rows)
            cov.top_skip_reasons = {}
            self.state.set_scores(rows, run_utc)
            self.state.set_coverage(cov, [])
            return

        client = self._make_client()
        if client is None:
            self.state.set_error("Alpaca keys missing (set ALPACA_API_KEY/ALPACA_API_SECRET) or DEMO_MODE=true")
            with self.state.lock:
                self.state.alpaca.ok = False
                self.state.alpaca.message = "Missing keys"
                self.state.alpaca.feed = self.settings.normalized_feed()
            cov.top_skip_reasons = {"missing_keys": len(universe_symbols)}
            self.state.set_coverage(cov, [])
            return

        # First-run backfill capped to last ~3h; we only need ~20 bars for features
        if self.cache_5m.last_fetch_utc is None:
            start = max(open_utc, now_utc - timedelta(hours=3))
        else:
            start = self.cache_5m.last_fetch_utc
        start = start - timedelta(minutes=10)
        end = now_utc

        bars_by_sym, err, warn = client.get_bars(symbols_requested, timeframe="5Min", start_utc=start, end_utc=end, limit=None)

        with self.state.lock:
            self.state.alpaca.feed = client.feed
            self.state.alpaca.last_request_utc = run_utc
            self.state.alpaca.rate_limit_warn = warn
            if err:
                self.state.alpaca.ok = False
                self.state.alpaca.message = err
            else:
                self.state.alpaca.ok = True
                self.state.alpaca.message = "OK"

        if err:
            self.state.set_error(f"Alpaca error: {err}")
            cov.top_skip_reasons = {"alpaca_error": len(universe_symbols)}
            self.state.set_coverage(cov, [])
            return

        cov.symbols_returned_with_bars_count = len([k for k,v in bars_by_sym.items() if k != "SPY" and v is not None])

        for sym, lst in bars_by_sym.items():
            self.cache_5m.merge(sym, lst)
            if lst:
                self.cache_5m.last_bar_ts = lst[-1].get("t")
        self.cache_5m.last_fetch_utc = end
        with self.state.lock:
            self.state.alpaca.last_bar_timestamp = self.cache_5m.last_bar_ts

        mins_to_close = max(0.0, (close_utc - now_utc).total_seconds()/60.0)

        # SPY regime (30m return)
        spy_bars = self.cache_5m.bars.get("SPY", [])
        spy_ret_30m = 0.0
        if spy_bars and len(spy_bars) >= 7:
            c = float(spy_bars[-1].get("c") or 0.0)
            c0= float(spy_bars[-7].get("c") or c)
            spy_ret_30m = (c/c0 - 1.0) if c0 else 0.0

        # ToD profiles availability
        avail, missing = self.vol_profiles.availability_counts()
        cov.profile_symbols_available = avail
        # Interpret missing relative to the current universe for auditability
        cov.profile_symbols_missing = max(0, cov.universe_count - avail)
        if avail == 0:
            cov.profile_note = "ToD RVOL profiles not found; using leakage-free rolling fallback"

        feats=[]
        meta=[]
        sufficient_count = 0
        used_profile_count = 0

        risk_params = (self.settings.liq_rolling_bars, self.settings.liq_dvol_min_usd, self.settings.liq_range_pct_max, self.settings.liq_wick_atr_max)

        for sym in universe_symbols:
            bars = self.cache_5m.bars.get(sym, [])
            if not bars:
                skip_counts["no_bars"] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason="no_bars", last_bar_timestamp=None))
                continue
            if len(bars) < self.settings.min_bars_5m:
                skip_counts["insufficient_bars"] += 1
                last_ts = bars[-1].get("t") if bars else None
                skipped.append(SkippedSymbol(symbol=sym, reason="insufficient_bars", last_bar_timestamp=last_ts))
                continue
            sufficient_count += 1

            # Determine slot for ToD baseline from latest bar timestamp
            last_ts = bars[-1].get("t")
            slot_idx = None
            if last_ts:
                try:
                    slot_idx = slot_index_from_ts(datetime.fromisoformat(last_ts.replace("Z","+00:00")), self.settings.timezone)
                except Exception:
                    slot_idx = None
            baseline = self.vol_profiles.get_slot_median(sym, slot_idx) if slot_idx is not None else None

            try:
                fr = compute_features_from_5m(
                    bars_5m=bars,
                    spy_ret_30m=spy_ret_30m,
                    mins_to_close=mins_to_close,
                    tod_baseline_vol_median=baseline,
                    rolling_rvol_window=20,
                    risk_params=risk_params,
                )
            except Exception:
                fr = None

            if fr is None:
                skip_counts["missing_price_or_vwap"] += 1
                skipped.append(SkippedSymbol(symbol=sym, reason="missing_price_or_vwap", last_bar_timestamp=last_ts))
                continue
            if fr.used_tod_profile:
                used_profile_count += 1

            sector = self.symbol_meta.get(sym).sector if sym in self.symbol_meta else "Unknown"
            feats.append(fr.features)
            meta.append((sym, sector, fr.price, fr.vwap, fr.risk, fr.reasons))

        cov.symbols_with_sufficient_bars_count = sufficient_count

        if not feats:
            cov.symbols_scored_count = 0
            cov.top_skip_reasons = {k:v for k,v in skip_counts.items() if v>0}
            self.state.set_scores([], run_utc)
            self.state.set_coverage(cov, skipped)
            return

        X = np.vstack(feats)
        p1, src1, status1 = predict_probs(self.settings.model_dir, X, 1)
        p2, src2, status2 = predict_probs(self.settings.model_dir, X, 2)

        # model compatibility diagnostics
        schema_bad = (status1 == "incompatible") or (status2 == "incompatible")
        if schema_bad:
            skip_counts["model_schema_incompatible"] = len(universe_symbols)
            # don't overwrite a real error
            with self.state.lock:
                if not self.state.last_error:
                    self.state.last_error = "Model schema mismatch; retrain required."

        with self.state.lock:
            self.state.model.pt1.trained = (src1=="trained")
            self.state.model.pt1.path = os.path.join(self.settings.model_dir, "pt1")
            self.state.model.pt2.trained = (src2=="trained")
            self.state.model.pt2.path = os.path.join(self.settings.model_dir, "pt2")

        rows=[]
        for i,(sym,sector,price,vwap,risk,reasons) in enumerate(meta):
            rows.append(ScoreRow(sym, sector, float(price), float(vwap), float(p1[i]), float(p2[i]), risk, reasons))
        rows.sort(key=lambda r: r.prob_2, reverse=True)

        cov.symbols_scored_count = len(rows)
        cov.top_skip_reasons = {k:v for k,v in skip_counts.items() if v>0}

        self.state.set_scores(rows, run_utc)
        self.state.set_coverage(cov, skipped)

        # cache last scores to disk
        try:
            os.makedirs(os.path.dirname(self.settings.model_dir), exist_ok=True)
            cache_path = os.path.join(os.path.dirname(self.settings.model_dir), "last_scores.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"last_run_utc": run_utc, "rows": [r.__dict__ for r in rows]}, f)
        except Exception:
            pass

    def _loop(self) -> None:
        try:
            self.load_constituents()
        except Exception as e:
            with self.state.lock:
                self.state.constituents.source="fallback"
                self.state.constituents.warning=f"failed to load constituents: {e}"

        while not self._stop.is_set():
            open_utc, close_utc, is_open, _ = self._update_market_status()
            now = _utcnow()
            if is_open:
                try:
                    self.scan_once(open_utc, close_utc, now)
                except Exception as e:
                    self.state.set_error(f"scan error: {e}")
            nxt = next_aligned_run(now, self.settings.timezone, self.settings.scan_interval_minutes, 3)
            self._stop.wait(timeout=max(1.0, (nxt - _utcnow()).total_seconds()))
