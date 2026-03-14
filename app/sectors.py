from __future__ import annotations
from typing import Dict, Iterable, List

SECTOR_TO_ETF: Dict[str, str] = {
    'Information Technology': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}

def sector_etf_for_sector(sector: str) -> str:
    return SECTOR_TO_ETF.get((sector or '').strip(), 'SPY')

def unique_sector_etfs(sectors: Iterable[str]) -> List[str]:
    out=[]
    seen=set()
    for s in sectors:
        e=sector_etf_for_sector(s)
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
