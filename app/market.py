from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

def get_market_times(now_utc: datetime, tz_name: str="America/New_York") -> Tuple[datetime, datetime, bool, int]:
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)
    open_local = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    close_local = now_local.replace(hour=16, minute=0, second=0, microsecond=0)
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=now_local.date(), end_date=now_local.date())
        if len(sched) == 1:
            open_local = sched.iloc[0]["market_open"].to_pydatetime().astimezone(tz)
            close_local = sched.iloc[0]["market_close"].to_pydatetime().astimezone(tz)
        else:
            return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), False, 0
    except Exception:
        if now_local.weekday() >= 5:
            return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), False, 0
    is_open = (now_local >= open_local) and (now_local < close_local)
    ttc = int(max(0, (close_local - now_local).total_seconds())) if is_open else 0
    return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), is_open, ttc

def next_aligned_run(now_utc: datetime, tz_name: str, interval_minutes: int, offset_seconds: int=3) -> datetime:
    tz = ZoneInfo(tz_name)
    local = now_utc.astimezone(tz)
    interval_minutes = max(1, interval_minutes)
    base = local.replace(second=0, microsecond=0)
    m = base.minute
    nxt_m = ((m // interval_minutes) + 1) * interval_minutes
    add_h = 0
    if nxt_m >= 60:
        nxt_m %= 60
        add_h = 1
    nxt = base.replace(minute=nxt_m) + timedelta(hours=add_h, seconds=offset_seconds)
    return nxt.astimezone(ZoneInfo("UTC"))
