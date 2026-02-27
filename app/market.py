from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

def _safe_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()

def get_market_times(now_utc: datetime, tz_name: str = "America/New_York") -> Tuple[datetime, datetime, bool, int]:
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)

    # Default open/close
    open_local = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    close_local = now_local.replace(hour=16, minute=0, second=0, microsecond=0)

    # Use NYSE calendar if available
    try:
        import pandas_market_calendars as mcal  # type: ignore
        cal = mcal.get_calendar("XNYS")
        schedule = cal.schedule(start_date=now_local.date(), end_date=now_local.date())
        if len(schedule) == 1:
            open_ts = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(tz)
            close_ts = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(tz)
            open_local, close_local = open_ts, close_ts
        else:
            # holiday: market closed
            return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), False, 0
    except Exception:
        # best-effort; fallback weekday schedule
        if now_local.weekday() >= 5:
            return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), False, 0

    is_open = (now_local >= open_local) and (now_local < close_local)
    seconds_to_close = int(max(0, (close_local - now_local).total_seconds())) if is_open else 0
    return open_local.astimezone(ZoneInfo("UTC")), close_local.astimezone(ZoneInfo("UTC")), is_open, seconds_to_close

def next_aligned_run(now_utc: datetime, tz_name: str, interval_minutes: int, offset_seconds: int = 3) -> datetime:
    tz = ZoneInfo(tz_name)
    local = now_utc.astimezone(tz)
    if interval_minutes <= 0:
        interval_minutes = 5
    # Align to minute boundary
    minute = local.minute
    next_min = ((minute // interval_minutes) + 1) * interval_minutes
    add_hours = 0
    if next_min >= 60:
        next_min = next_min % 60
        add_hours = 1
    nxt = local.replace(second=0, microsecond=0) + timedelta(hours=add_hours)
    nxt = nxt.replace(minute=next_min)
    nxt = nxt + timedelta(seconds=offset_seconds)
    return nxt.astimezone(ZoneInfo("UTC"))

def iso(dt: Optional[datetime]) -> Optional[str]:
    return _safe_iso(dt)
