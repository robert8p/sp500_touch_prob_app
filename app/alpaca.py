from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests


def normalize_symbol(sym: str) -> str:
    """Normalize tickers for Alpaca query params (e.g., BRK-B -> BRK.B)."""
    s = (sym or "").strip().upper()
    if not s:
        return s
    if "-" in s and s.count("-") == 1:
        left, right = s.split("-", 1)
        if left and len(right) == 1 and right.isalnum():
            return f"{left}.{right}"
    return s


def _extract_invalid_symbol(err_text: str) -> Optional[str]:
    """Extract `invalid symbol: XYZ` from common Alpaca error payload strings."""
    if not err_text:
        return None
    m = re.search(r'invalid symbol\s*:\s*([^"\s}]+)', err_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _to_utc_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


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
        """Return (json, error_str, rate_limit_warn)."""
        url = self.base_url.rstrip("/") + path
        backoff = 1.0
        rate_warn = None
        last_err = None

        for _ in range(6):
            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=timeout_s)

                if resp.status_code == 429:
                    rate_warn = "HTTP 429 rate-limited; backing off"
                    time.sleep(backoff)
                    backoff = min(30.0, backoff * 2.0)
                    continue

                if resp.status_code >= 400:
                    return None, f"HTTP {resp.status_code}: {resp.text[:400]}", rate_warn

                return resp.json(), None, rate_warn

            except Exception as e:
                last_err = str(e)
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)

        return None, last_err or "request failed", rate_warn

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start_utc: Optional[datetime] = None,
        end_utc: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: str = "raw",
    ) -> Tuple[Dict[str, List[dict]], Optional[str], Optional[str]]:
        """Fetch stock bars for a list of symbols from Alpaca Market Data v2.

        Returns:
          out_by_symbol, err_any, rate_limit_warning
        """
        if not symbols:
            return {}, None, None

        # Normalize and de-duplicate symbols (preserve order)
        symbols = [normalize_symbol(s) for s in symbols if s and str(s).strip()]
        seen = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]

        params: Dict[str, str] = {
            "timeframe": timeframe,
            "feed": (self.feed or "sip").lower(),
            "adjustment": adjustment,
        }
        if start_utc is not None:
            params["start"] = _to_utc_iso(start_utc)
        if end_utc is not None:
            params["end"] = _to_utc_iso(end_utc)
        if limit is not None:
            params["limit"] = str(limit)

        out: Dict[str, List[dict]] = {}
        chunk_size = 200
        err_any = None
        warn_any = None

        for i in range(0, len(symbols), chunk_size):
            base_chunk = symbols[i : i + chunk_size]
            retry_chunk = list(base_chunk)

            # If one invalid symbol poisons the batch, remove it and retry.
            for _ in range(5):  # remove up to 5 invalid symbols per chunk
                params["symbols"] = ",".join(retry_chunk)
                js, err, warn = self._get("/v2/stocks/bars", params=params)
                if warn and not warn_any:
                    warn_any = warn

                if err:
                    err_any = err
                    bad = _extract_invalid_symbol(err)
                    if bad and bad in retry_chunk and len(retry_chunk) > 1:
                        retry_chunk.remove(bad)
                        continue
                    break  # cannot isolate, skip chunk

                bars = js.get("bars", {}) if isinstance(js, dict) else {}
                for sym, lst in bars.items():
                    out[sym] = lst or []
                break

        return out, err_any, warn_any
