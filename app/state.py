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
    alpha: Optional[float] = None
    tail_ready_70: Optional[bool] = None
    tail_ready_75: Optional[bool] = None
    selection_tier: Optional[str] = None
    selection_warning: Optional[str] = None
    live_prob_cap: Optional[float] = None


@dataclass
class ModelStatus:
    pt1: ModelThresholdStatus = field(default_factory=ModelThresholdStatus)


@dataclass
class MarketStatus:
    market_open: bool = False
    time_to_close_seconds: int = 0
    market_open_time: Optional[str] = None
    market_close_time: Optional[str] = None


@dataclass
class CoverageStatus:
    universe_count: int = 0
    symbols_requested_count: int = 0
    symbols_returned_with_bars_count: int = 0
    symbols_with_sufficient_bars_count: int = 0
    symbols_scored_count: int = 0
    top_skip_reasons: Dict[str, int] = field(default_factory=dict)
    profile_symbols_available: int = 0
    profile_symbols_missing: int = 0
    profile_note: Optional[str] = None
    stage1_candidate_count: int = 0
    stage2_scored_count: int = 0
    stage1_blocked_count: int = 0
    stage1_event_count: int = 0
    stage1_time_filtered_count: int = 0
    stage1_rejected_count: int = 0
    capped_by_downside_count: int = 0
    capped_by_uncertainty_count: int = 0
    capped_by_event_count: int = 0
    threshold_counts: Dict[str, int] = field(default_factory=dict)
    acceptable_threshold_counts: Dict[str, int] = field(default_factory=dict)
    guardrail_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class SkippedSymbol:
    symbol: str
    reason: str
    last_bar_timestamp: Optional[str] = None


@dataclass
class ScoreRow:
    symbol: str
    sector: str
    price: float
    vwap: float
    prob_1: float
    risk: str
    risk_reasons: str
    reasons: str
    downside_risk: Optional[float] = None
    uncertainty: str = 'LOW'
    uncertainty_reasons: str = ''
    stage1_score: Optional[float] = None
    stage1_reasons: str = ''
    prob_1_raw: Optional[float] = None


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    alpaca: AlpacaStatus = field(default_factory=AlpacaStatus)
    constituents: ConstituentsStatus = field(default_factory=ConstituentsStatus)
    model: ModelStatus = field(default_factory=ModelStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    training: TrainingStatus = field(default_factory=TrainingStatus)
    coverage: CoverageStatus = field(default_factory=CoverageStatus)
    last_run_utc: Optional[str] = None
    scores: List[ScoreRow] = field(default_factory=list)
    last_error: Optional[str] = None
    skipped: List[SkippedSymbol] = field(default_factory=list)

    def set_scores(self, rows: List[ScoreRow], run_utc: str) -> None:
        with self.lock:
            self.scores = rows
            self.last_run_utc = run_utc
            self.last_error = None

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.last_error = msg

    def set_coverage(self, cov: CoverageStatus, skipped: List[SkippedSymbol]) -> None:
        with self.lock:
            self.coverage = cov
            self.skipped = skipped[:200]

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
                "model": {"pt1": self.model.pt1.__dict__},
                "training": self.training.__dict__,
                "coverage": {
                    "universe_count": self.coverage.universe_count,
                    "symbols_requested_count": self.coverage.symbols_requested_count,
                    "symbols_returned_with_bars_count": self.coverage.symbols_returned_with_bars_count,
                    "symbols_with_sufficient_bars_count": self.coverage.symbols_with_sufficient_bars_count,
                    "symbols_scored_count": self.coverage.symbols_scored_count,
                    "top_skip_reasons": self.coverage.top_skip_reasons,
                    "profile_symbols_available": self.coverage.profile_symbols_available,
                    "profile_symbols_missing": self.coverage.profile_symbols_missing,
                    "profile_note": self.coverage.profile_note,
                    "stage1_candidate_count": self.coverage.stage1_candidate_count,
                    "stage2_scored_count": self.coverage.stage2_scored_count,
                    "stage1_blocked_count": self.coverage.stage1_blocked_count,
                    "stage1_event_count": self.coverage.stage1_event_count,
                    "stage1_time_filtered_count": self.coverage.stage1_time_filtered_count,
                    "stage1_rejected_count": self.coverage.stage1_rejected_count,
                    "capped_by_downside_count": self.coverage.capped_by_downside_count,
                    "capped_by_uncertainty_count": self.coverage.capped_by_uncertainty_count,
                    "capped_by_event_count": self.coverage.capped_by_event_count,
                    "threshold_counts": self.coverage.threshold_counts,
                    "acceptable_threshold_counts": self.coverage.acceptable_threshold_counts,
                    "guardrail_stats": self.coverage.guardrail_stats,
                },
                "last_run_utc": self.last_run_utc,
                "last_error": self.last_error,
            }
