from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

FALLBACK_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "sp500_fallback.csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

@dataclass
class Constituent:
    symbol: str
    name: str
    sector: str
    industry: str


def normalize_symbol(sym: str) -> str:
    """Normalize tickers for Alpaca.
    - Keeps dot-class tickers like BRK.B and BF.B (Alpaca format).
    - Converts hyphen-class tickers like BRK-B -> BRK.B.
    """
    s = (sym or "").strip().upper()
    if not s:
        return s
    if "-" in s and s.count("-") == 1:
        left, right = s.split("-", 1)
        if left and len(right) == 1 and right.isalnum():
            return f"{left}.{right}"
    return s

def load_fallback() -> List[Constituent]:
    out: List[Constituent] = []
    if not os.path.exists(FALLBACK_CSV_PATH):
        return out
    with open(FALLBACK_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("Symbol") or row.get("symbol") or "").strip()
            if not sym:
                continue
            out.append(
                Constituent(
                    symbol=normalize_symbol(sym),
                    name=(row.get("Name") or row.get("Security") or row.get("name") or sym).strip(),
                    sector=(row.get("Sector") or row.get("GICS Sector") or row.get("sector") or "Unknown").strip(),
                    industry=(row.get("Industry") or row.get("GICS Sub-Industry") or row.get("industry") or "").strip(),
                )
            )
    return out

def try_refresh_from_wikipedia(timeout_s: int = 8) -> Tuple[Optional[List[Constituent]], Optional[str]]:
    # Best-effort only. Non-fatal if blocked.
    try:
        import pandas as pd  # noqa
    except Exception:
        return None, "pandas not available for refresh"

    try:
        resp = requests.get(WIKI_URL, timeout=timeout_s, headers={"User-Agent": "sp500-prob-scanner/1.0"})
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            return None, "no tables found"
        df = tables[0]
        # Wikipedia table typically has columns: Symbol, Security, GICS Sector, GICS Sub-Industry
        cols = [c.lower() for c in df.columns]
        sym_col = None
        for candidate in ["symbol", "ticker symbol", "ticker"]:
            if candidate in cols:
                sym_col = df.columns[cols.index(candidate)]
                break
        if sym_col is None:
            sym_col = df.columns[0]
        sec_col = None
        for candidate in ["security", "name"]:
            if candidate in cols:
                sec_col = df.columns[cols.index(candidate)]
                break
        gics_sector = None
        for candidate in ["gics sector", "sector"]:
            if candidate in cols:
                gics_sector = df.columns[cols.index(candidate)]
                break
        gics_ind = None
        for candidate in ["gics sub-industry", "sub-industry", "industry"]:
            if candidate in cols:
                gics_ind = df.columns[cols.index(candidate)]
                break

        out: List[Constituent] = []
        for _, r in df.iterrows():
            sym = str(r.get(sym_col, "")).strip()
            if not sym or sym.lower() == "nan":
                continue
            out.append(
                Constituent(
                    symbol=normalize_symbol(sym),
                    name=str(r.get(sec_col, sym)).strip() if sec_col is not None else sym,
                    sector=str(r.get(gics_sector, "Unknown")).strip() if gics_sector is not None else "Unknown",
                    industry=str(r.get(gics_ind, "")).strip() if gics_ind is not None else "",
                )
            )
        if len(out) < 400:
            return None, f"refresh produced too few rows ({len(out)})"
        return out, None
    except Exception as e:
        return None, str(e)
