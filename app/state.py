from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TrainingStatus:
    running: bool = False
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

@dataclass
class AlpacaStatus:
    ok: bool = False
    message: str = "Not checked"
    feed: str = "sip"
    last_request_utc: Optional[str] = None
    last_bar_timestamp: Optional[str] = None
    rate_limit_warn: Optional[str] = None

@dataclass
class ConstituentsStatus:
    source: str = "fallback"
    warning: Optional[str] = None
    count: int = 0

@dataclass
class ModelThresholdStatus:
    trained: bool = False
    path: str = ""
    auc_val: Optional[float] = None
    brier_val: Optional[float] = None
    calibrator: Optional[str] = None
    class_weight: Optional[str] = None

@dataclass
class ModelStatus:
    pt1: ModelThresholdStatus = field(default_factory=ModelThresholdStatus)
    pt2: ModelThresholdStatus = field(default_factory=ModelThresholdStatus)

@dataclass
class MarketStatus:
    market_open: bool = False
    time_to_close_seconds: int = 0
    market_open_time: Optional[str] = None
    market_close_time: Optional[str] = None

@dataclass
class ScoreRow:
    symbol: str
    sector: str
    price: float
    vwap: float
    prob_1: float
    prob_2: float
    reasons: str

@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    alpaca: AlpacaStatus = field(default_factory=AlpacaStatus)
    constituents: ConstituentsStatus = field(default_factory=ConstituentsStatus)
    model: ModelStatus = field(default_factory=ModelStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    training: TrainingStatus = field(default_factory=TrainingStatus)

    last_run_utc: Optional[str] = None
    scores: List[ScoreRow] = field(default_factory=list)
    last_error: Optional[str] = None

    def set_scores(self, rows: List[ScoreRow], run_utc: str) -> None:
        with self.lock:
            self.scores = rows
            self.last_run_utc = run_utc
            self.last_error = None

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.last_error = msg

    def snapshot_scores(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "last_run_utc": self.last_run_utc,
                "rows": [r.__dict__ for r in self.scores],
                "last_error": self.last_error,
            }

    def snapshot_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "demo_mode": None,
                "market_open": self.market.market_open,
                "time_to_close_seconds": self.market.time_to_close_seconds,
                "market_open_time": self.market.market_open_time,
                "market_close_time": self.market.market_close_time,
                "alpaca": self.alpaca.__dict__,
                "constituents": self.constituents.__dict__,
                "model": {"pt1": self.model.pt1.__dict__, "pt2": self.model.pt2.__dict__},
                "training": self.training.__dict__,
                "last_run_utc": self.last_run_utc,
                "last_error": self.last_error,
            }
