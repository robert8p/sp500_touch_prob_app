from __future__ import annotations
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

def normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    if "-" in s and s.count("-")==1:
        left,right = s.split("-",1)
        if left and len(right)==1 and right.isalnum():
            return f"{left}.{right}"
    return s

def _extract_invalid_symbol(err_text: str) -> Optional[str]:
    if not err_text:
        return None
    m = re.search(r'invalid symbol\s*:\s*([^"\s}]+)', err_text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def _to_utc_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z")

@dataclass
class AlpacaClient:
    api_key: str
    api_secret: str
    feed: str = "sip"
    base_url: str = "https://data.alpaca.markets"

    def _headers(self) -> Dict[str,str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}

    def _get(self, path: str, params: Dict[str,str], timeout_s: int=25) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        url = self.base_url.rstrip("/") + path
        backoff = 1.0
        warn = None
        last_err = None
        for _ in range(6):
            try:
                r = requests.get(url, headers=self._headers(), params=params, timeout=timeout_s)
                if r.status_code == 429:
                    warn = "HTTP 429 rate-limited; backing off"
                    time.sleep(backoff)
                    backoff = min(30.0, backoff*2)
                    continue
                if r.status_code >= 400:
                    return None, f"HTTP {r.status_code}: {r.text[:400]}", warn
                return r.json(), None, warn
            except Exception as e:
                last_err = str(e)
                time.sleep(backoff)
                backoff = min(30.0, backoff*2)
        return None, last_err or "request failed", warn

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start_utc: Optional[datetime]=None,
        end_utc: Optional[datetime]=None,
        limit: Optional[int]=None,
        adjustment: str="raw",
    ) -> Tuple[Dict[str, List[dict]], Optional[str], Optional[str]]:
        if not symbols:
            return {}, None, None

        symbols = [normalize_symbol(s) for s in symbols if s and str(s).strip()]
        seen=set()
        symbols=[s for s in symbols if not (s in seen or seen.add(s))]

        params: Dict[str,str] = {"timeframe": timeframe, "feed": (self.feed or "sip").lower(), "adjustment": adjustment}
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
            base_chunk = symbols[i:i+chunk_size]
            retry_chunk = list(base_chunk)

            for _ in range(5):
                params["symbols"] = ",".join(retry_chunk)

                page_token: Optional[str] = None
                pages = 0
                merged: Dict[str, List[dict]] = {}

                while True:
                    if page_token:
                        params["page_token"] = page_token
                    else:
                        params.pop("page_token", None)

                    js, err, warn = self._get("/v2/stocks/bars", params=params)
                    if warn and not warn_any:
                        warn_any = warn

                    if err:
                        err_any = err
                        bad = _extract_invalid_symbol(err)
                        if bad and bad in retry_chunk and len(retry_chunk) > 1:
                            retry_chunk.remove(bad)
                            merged = {}
                            page_token = None
                            break
                        merged = {}
                        page_token = None
                        break

                    bars = js.get("bars", {}) if isinstance(js, dict) else {}
                    for sym, lst in bars.items():
                        if lst:
                            merged.setdefault(sym, []).extend(lst)

                    page_token = js.get("next_page_token") if isinstance(js, dict) else None
                    pages += 1
                    if not page_token:
                        break
                    if pages >= 20:
                        warn_any = warn_any or "pagination cap hit"
                        break

                if merged:
                    for sym, lst in merged.items():
                        out[sym] = lst
                    break

                if err_any:
                    bad = _extract_invalid_symbol(err_any)
                    if not (bad and bad in base_chunk and len(retry_chunk) > 0):
                        break

        return out, err_any, warn_any
