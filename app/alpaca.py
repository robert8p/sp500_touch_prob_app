from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

@dataclass
class AlpacaClient:
    api_key: str
    api_secret: str
    feed: str = "sip"
    base_url: str = "https://data.alpaca.markets"

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _get(self, path: str, params: Dict[str, str], timeout_s: int = 20) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        # returns (json, error, rate_limit_warn)
        url = self.base_url.rstrip("/") + path
        backoff = 1.0
        rate_warn = None
        for attempt in range(6):
            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=timeout_s)
                if resp.status_code == 429:
                    rate_warn = "HTTP 429 rate-limited; backing off"
                    time.sleep(backoff)
                    backoff = min(30.0, backoff * 2.0)
                    continue
                if resp.status_code >= 400:
                    return None, f"HTTP {resp.status_code}: {resp.text[:200]}", rate_warn
                return resp.json(), None, rate_warn
            except Exception as e:
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)
                err = str(e)
        return None, err, rate_warn

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start_utc: Optional[datetime] = None,
        end_utc: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: str = "raw",
    ) -> Tuple[Dict[str, List[dict]], Optional[str], Optional[str]]:
        # timeframe examples: "1Min", "5Min"
        if not symbols:
            return {}, None, None
        params: Dict[str, str] = {
            "timeframe": timeframe,
            "feed": (self.feed or "sip").lower(),
            "adjustment": adjustment,
        }
        if start_utc is not None:
            params["start"] = start_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        if end_utc is not None:
            params["end"] = end_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        if limit is not None:
            params["limit"] = str(limit)

        out: Dict[str, List[dict]] = {}
        # Chunk symbols to keep URL reasonable
        chunk_size = 200
        err_any = None
        warn_any = None
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            params["symbols"] = ",".join(chunk)
            js, err, warn = self._get("/v2/stocks/bars", params=params)
            if warn and not warn_any:
                warn_any = warn
            if err:
                err_any = err
                continue
            bars = js.get("bars", {}) if isinstance(js, dict) else {}
            for sym, lst in bars.items():
                out[sym] = lst or []
        return out, err_any, warn_any
