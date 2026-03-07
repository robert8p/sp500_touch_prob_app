from __future__ import annotations
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .alpaca import AlpacaClient

def profiles_dir(model_dir: str) -> str:
    return os.path.join(model_dir, "volume_profiles")

def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def _parse_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def slot_index_from_ts(ts_utc: datetime, tz_name: str="America/New_York") -> Optional[int]:
    tz = ZoneInfo(tz_name)
    tloc = ts_utc.astimezone(tz)
    start = tloc.replace(hour=9, minute=30, second=0, microsecond=0)
    mins = (tloc - start).total_seconds()/60.0
    if mins < 0 or mins >= 390:
        return None
    idx = int(mins // 5)
    return idx if 0 <= idx < 78 else None

def _trading_days(end_local: date, lookback_days: int, tz_name: str) -> List[date]:
    start = end_local - timedelta(days=lookback_days*2)
    out=[]
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=start, end_date=end_local)
        for idx in sched.index:
            out.append(idx.date())
    except Exception:
        d=start
        while d<=end_local:
            if d.weekday()<5:
                out.append(d)
            d += timedelta(days=1)
    return out[-lookback_days:] if len(out)>lookback_days else out

def _session_utc_for_day(d: date, tz_name: str) -> Tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    open_local = datetime(d.year,d.month,d.day,9,30,tzinfo=tz)
    close_local = datetime(d.year,d.month,d.day,16,0,tzinfo=tz)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=d, end_date=d)
        if len(sched)==1:
            open_local = sched.iloc[0]["market_open"].to_pydatetime().astimezone(tz)
            close_local = sched.iloc[0]["market_close"].to_pydatetime().astimezone(tz)
    except Exception:
        pass
    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)

@dataclass
class VolumeProfile:
    symbol: str
    lookback_days: int
    min_days: int
    days_used: int
    slot_median: List[Optional[float]]
    slot_iqr: List[Optional[float]]
    available: bool
    computed_at_utc: str

def compute_profiles(client: AlpacaClient, symbols: List[str], tz_name: str, lookback_days: int, min_days: int) -> Dict[str, VolumeProfile]:
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    days = _trading_days(today_local, lookback_days, tz_name)
    volumes: Dict[str, List[List[float]]] = {s:[[] for _ in range(78)] for s in symbols}
    day_presence: Dict[str, set] = {s:set() for s in symbols}
    for d in days:
        open_utc, close_utc = _session_utc_for_day(d, tz_name)
        bars_by_sym, err, _ = client.get_bars(symbols, timeframe="5Min", start_utc=open_utc, end_utc=close_utc)
        if err:
            continue
        for sym, bars in bars_by_sym.items():
            if sym not in volumes or not bars:
                continue
            day_presence[sym].add(d.isoformat())
            for b in bars:
                ts = b.get("t")
                if not ts:
                    continue
                try:
                    dt = _parse_ts(ts)
                except Exception:
                    continue
                idx = slot_index_from_ts(dt, tz_name)
                if idx is None:
                    continue
                try:
                    vv = float(b.get("v"))
                except Exception:
                    continue
                if vv >= 0:
                    volumes[sym][idx].append(vv)
    out={}
    for sym in symbols:
        days_used = len(day_presence.get(sym,set()))
        available = days_used >= min_days
        if not available:
            medians=[None]*78; iqrs=[None]*78
        else:
            medians=[]; iqrs=[]
            for idx in range(78):
                vals=np.array(volumes[sym][idx], dtype=float)
                if vals.size == 0:
                    medians.append(None); iqrs.append(None); continue
                med=float(np.median(vals)); medians.append(med if med>0 else None)
                if vals.size >= 4:
                    q75=float(np.quantile(vals,0.75)); q25=float(np.quantile(vals,0.25)); iqrs.append(max(0.0,q75-q25))
                else:
                    iqrs.append(None)
        out[sym]=VolumeProfile(symbol=sym, lookback_days=lookback_days, min_days=min_days, days_used=days_used, slot_median=medians, slot_iqr=iqrs, available=available, computed_at_utc=_utc_iso_now())
    return out

def save_profiles(model_dir: str, profiles: Dict[str, VolumeProfile]) -> None:
    out_dir = profiles_dir(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    for sym, prof in profiles.items():
        with open(os.path.join(out_dir, f"{sym}.json"), "w", encoding="utf-8") as f:
            json.dump(prof.__dict__, f)

class VolumeProfileStore:
    def __init__(self, model_dir: str):
        self.model_dir=model_dir
        self.dir=profiles_dir(model_dir)
        self._loaded=False
        self._last_mtime=0.0
        self._profiles: Dict[str, VolumeProfile] = {}

    def _dir_mtime(self) -> float:
        try:
            return os.path.getmtime(self.dir)
        except Exception:
            return 0.0

    def load_if_changed(self) -> None:
        m=self._dir_mtime()
        if self._loaded and m <= self._last_mtime:
            return
        self._loaded=True
        self._last_mtime=m
        self._profiles={}
        if not os.path.isdir(self.dir):
            return
        for fn in os.listdir(self.dir):
            if not fn.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.dir, fn), 'r', encoding='utf-8') as f:
                    d=json.load(f)
                sym=d.get('symbol') or fn[:-5]
                self._profiles[sym]=VolumeProfile(**d)
            except Exception:
                continue

    def get_slot_median(self, symbol: str, slot: int) -> Optional[float]:
        self.load_if_changed()
        prof=self._profiles.get(symbol)
        if not prof or not prof.available or slot < 0 or slot >= 78:
            return None
        v=prof.slot_median[slot]
        return float(v) if v is not None and v > 0 else None

    def availability_counts(self) -> Tuple[int, int]:
        self.load_if_changed()
        avail=sum(1 for p in self._profiles.values() if p.available)
        missing=sum(1 for p in self._profiles.values() if not p.available)
        return avail, missing
