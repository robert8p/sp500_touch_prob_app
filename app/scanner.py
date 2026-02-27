from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .alpaca import AlpacaClient
from .config import Settings
from .constituents import Constituent, load_fallback, try_refresh_from_wikipedia, normalize_symbol
from .features import FeatureRow, compute_features_from_5m
from .market import get_market_times, iso, next_aligned_run
from .modeling import predict_probs
from .state import AppState, ScoreRow

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

@dataclass
class BarsCache:
    timeframe: str
    bars: Dict[str, List[dict]] = field(default_factory=dict)
    last_fetch_utc: Optional[datetime] = None
    last_bar_ts: Optional[str] = None

    def merge(self, symbol: str, new_bars: List[dict], keep: int = 140) -> None:
        if not new_bars:
            return
        existing = self.bars.get(symbol, [])
        # Deduplicate by timestamp 't'
        seen = {b.get("t") for b in existing}
        merged = existing + [b for b in new_bars if b.get("t") not in seen]
        merged.sort(key=lambda x: x.get("t", ""))
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
        # Normalize for Alpaca symbols (e.g., BRK-B -> BRK.B)
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
        t = threading.Thread(target=self._loop, name="scanner-loop", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def _make_client(self) -> Optional[AlpacaClient]:
        if self.settings.demo_mode:
            return None
        if not (self.settings.alpaca_api_key and self.settings.alpaca_api_secret):
            return None
        return AlpacaClient(
            api_key=self.settings.alpaca_api_key,
            api_secret=self.settings.alpaca_api_secret,
            feed=self.settings.normalized_feed(),
        )

    def _update_market_status(self) -> Tuple[datetime, datetime, bool, int]:
        now = _utcnow()
        open_utc, close_utc, is_open, ttc = get_market_times(now, self.settings.timezone)
        with self.state.lock:
            self.state.market.market_open = is_open
            self.state.market.time_to_close_seconds = ttc
            self.state.market.market_open_time = iso(open_utc)
            self.state.market.market_close_time = iso(close_utc)
        return open_utc, close_utc, is_open, ttc

    def _loop(self) -> None:
        # Load constituents once at startup
        try:
            self.load_constituents()
        except Exception as e:
            with self.state.lock:
                self.state.constituents.source = "fallback"
                self.state.constituents.warning = f"failed to load constituents: {e}"
                self.state.constituents.count = 0

        while not self._stop.is_set():
            open_utc, close_utc, is_open, _ = self._update_market_status()
            now = _utcnow()

            if is_open:
                try:
                    self.scan_once(open_utc=open_utc, close_utc=close_utc, now_utc=now)
                except Exception as e:
                    self.state.set_error(f"scan error: {e}")
            # Sleep to next aligned run
            nxt = next_aligned_run(now, self.settings.timezone, self.settings.scan_interval_minutes, offset_seconds=3)
            sleep_s = max(1.0, (nxt - _utcnow()).total_seconds())
            self._stop.wait(timeout=sleep_s)

    def _demo_scores(self) -> List[ScoreRow]:
        # Deterministic demo rows
        demo = [
            ("AAPL", "Information Technology"),
            ("MSFT", "Information Technology"),
            ("NVDA", "Information Technology"),
            ("AMZN", "Consumer Discretionary"),
            ("JPM", "Financials"),
        ]
        rows: List[ScoreRow] = []
        for i, (sym, sec) in enumerate(demo):
            base = 100 + i * 20
            price = float(base + (i * 1.3))
            vwap = price - 0.4
            # simple monotonic demo probs
            prob2 = min(0.95, 0.55 + 0.08 * i)
            prob1 = min(0.98, prob2 + 0.15)
            rows.append(ScoreRow(symbol=sym, sector=sec, price=price, vwap=vwap, prob_1=prob1, prob_2=prob2, reasons="DEMO"))
        rows.sort(key=lambda r: r.prob_2, reverse=True)
        return rows

    def scan_once(self, open_utc: datetime, close_utc: datetime, now_utc: datetime) -> None:
        run_utc = now_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

        if self.settings.demo_mode:
            with self.state.lock:
                self.state.alpaca.ok = True
                self.state.alpaca.message = "DEMO_MODE"
                self.state.alpaca.feed = "sip"
                self.state.alpaca.last_request_utc = run_utc
                self.state.alpaca.last_bar_timestamp = run_utc
                self.state.alpaca.rate_limit_warn = None
            rows = self._demo_scores()
            self.state.set_scores(rows, run_utc)
            return

        client = self._make_client()
        if client is None:
            self.state.set_error("Alpaca keys missing (set ALPACA_API_KEY/ALPACA_API_SECRET) or DEMO_MODE=true")
            with self.state.lock:
                self.state.alpaca.ok = False
                self.state.alpaca.message = "Missing keys"
                self.state.alpaca.feed = self.settings.normalized_feed()
            return

        # Ensure constituents loaded
        if not self.constituents:
            self.load_constituents()

        symbols = [c.symbol for c in self.constituents]
        # Always include SPY for regime
        if "SPY" not in symbols:
            symbols = symbols + ["SPY"]

        # Incremental fetch
        start = self.cache_5m.last_fetch_utc or open_utc
        # back up 10 minutes to avoid gaps
        start = start - timedelta(minutes=10)
        end = now_utc

        bars_by_sym, err, warn = client.get_bars(symbols, timeframe="5Min", start_utc=start, end_utc=end, limit=None)
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
            return

        # Merge into cache
        for sym, lst in bars_by_sym.items():
            self.cache_5m.merge(sym, lst)
            if lst:
                self.cache_5m.last_bar_ts = lst[-1].get("t")
        self.cache_5m.last_fetch_utc = end
        with self.state.lock:
            self.state.alpaca.last_bar_timestamp = self.cache_5m.last_bar_ts

        # Determine mins to close
        mins_to_close = max(0.0, (close_utc - now_utc).total_seconds() / 60.0)

        # SPY return 30m
        spy_bars = self.cache_5m.bars.get("SPY", [])
        spy_ret_30m = 0.0
        if spy_bars and len(spy_bars) >= 7:
            c = float(spy_bars[-1].get("c") or 0.0)
            c0 = float(spy_bars[-7].get("c") or c)
            spy_ret_30m = (c / c0 - 1.0) if c0 else 0.0

        # Compute features for each symbol (exclude SPY from rows)
        feats: List[np.ndarray] = []
        meta: List[Tuple[str, str, float, float, str]] = []  # sym, sector, price, vwap, reasons
        for sym in symbols:
            if sym == "SPY":
                continue
            bars = self.cache_5m.bars.get(sym, [])
            fr = compute_features_from_5m(bars, spy_ret_30m=spy_ret_30m, mins_to_close=mins_to_close)
            if fr is None:
                continue
            sector = self.symbol_meta.get(sym).sector if sym in self.symbol_meta else "Unknown"
            feats.append(fr.features)
            meta.append((sym, sector, fr.price, fr.vwap, fr.reasons))

        if not feats:
            self.state.set_scores([], run_utc)
            return

        X = np.vstack(feats)

        p1, src1 = predict_probs(self.settings.model_dir, X, threshold_pct=1)
        p2, src2 = predict_probs(self.settings.model_dir, X, threshold_pct=2)

        # record model status
        with self.state.lock:
            self.state.model.pt1.trained = (src1 == "trained")
            self.state.model.pt1.path = os.path.join(self.settings.model_dir, "pt1")
            self.state.model.pt2.trained = (src2 == "trained")
            self.state.model.pt2.path = os.path.join(self.settings.model_dir, "pt2")

        rows: List[ScoreRow] = []
        for i, (sym, sector, price, vwap, reasons) in enumerate(meta):
            rows.append(
                ScoreRow(
                    symbol=sym,
                    sector=sector,
                    price=float(price),
                    vwap=float(vwap),
                    prob_1=float(p1[i]),
                    prob_2=float(p2[i]),
                    reasons=reasons,
                )
            )
        rows.sort(key=lambda r: r.prob_2, reverse=True)
        self.state.set_scores(rows, run_utc)

        # persist last scores to disk for resilience across restarts (best effort)
        try:
            os.makedirs(os.path.join(self.settings.model_dir, ".."), exist_ok=True)
            cache_path = os.path.join(os.path.dirname(self.settings.model_dir), "last_scores.json")
            payload = {"last_run_utc": run_utc, "rows": [r.__dict__ for r in rows]}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass
