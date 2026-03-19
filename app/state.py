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
    touch_tail_validated: Optional[bool] = None
    decision_tail_validated: Optional[bool] = None
    selection_tier: Optional[str] = None
    selection_warning: Optional[str] = None
    model_b_method: Optional[str] = None    # NEW: 'lightgbm', 'logistic', or None
    probability_contract: Optional[str] = None  # 'strict_calibrated_decomposed'
    model_source: Optional[str] = None     # runtime: 'trained' | 'trained_no_path' | 'heuristic'
    adaptive_touch_threshold: Optional[float] = None  # legacy touch threshold
    adaptive_decision_threshold: Optional[float] = None


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
class RegimeStatus:
    state: str = 'NOT_EVALUATED'
    source: str = 'auto'
    reasons: str = ''
    note: str = ''
    multiplier: float = 1.0
    prob_cap: Optional[float] = None
    touch_threshold_mult: float = 1.0
    path_floor_add: float = 0.0
    suppress_new_signals: bool = False
    is_manual_override: bool = False
    cooldown_until_utc: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    evaluated_at_utc: Optional[str] = None
    last_live_state: Optional[str] = None
    market_session: Optional[str] = None
    live_evaluated: bool = False
    data_complete: bool = False


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
    stage1_strong_override_count: int = 0   # NEW
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
    prob_1: float                  # combined = P(touch) * P(path | touch)
    risk: str
    risk_reasons: str
    reasons: str
    downside_risk: Optional[float] = None
    uncertainty: str = 'LOW'
    uncertainty_reasons: str = ''
    stage1_score: Optional[float] = None
    stage1_reasons: str = ''
    prob_1_raw: Optional[float] = None
    prob_touch: Optional[float] = None   # display-touch after guardrails and regime overlay
    prob_touch_raw: Optional[float] = None
    prob_touch_pre_regime: Optional[float] = None
    prob_path: Optional[float] = None    # Model B output (path quality)
    prob_1_pre_regime: Optional[float] = None
    guardrail_multiplier: Optional[float] = None
    display_prob_threshold: Optional[float] = None
    regime_state: str = 'GREEN'
    regime_multiplier: float = 1.0
    regime_prob_cap: Optional[float] = None
    regime_reasons: str = ''
    signal: str = ''
    raw_signal: str = ''
    suppression_reason: str = ''
    watchlist_rescue: bool = False
    watchlist_reason: str = ''
    watchlist_score: Optional[float] = None
    guardrail_flags: str = ''
    acceptable: Optional[bool] = None
    event_risk: Optional[bool] = None
    high_downside: Optional[bool] = None
    medium_downside: Optional[bool] = None
    high_uncertainty: Optional[bool] = None
    tail_validated: Optional[bool] = None
    display_touch_threshold: Optional[float] = None
    path_action_min: Optional[float] = None
    setup_family: str = 'OTHER'
    setup_family_bonus: float = 0.0
    relative_strength_score: Optional[float] = None
    relative_strength_rank: Optional[int] = None


@dataclass
class NearMiss:
    """Stage 1 near-miss: passed score threshold but failed another gate."""
    symbol: str
    score: float
    rejection_reason: str


@dataclass
class AIStrategyStatus:
    enabled: bool = False
    configured: bool = False
    model: str = ''
    status: str = 'idle'
    last_generated_at_utc: Optional[str] = None
    generated_for_run_utc: Optional[str] = None
    error: Optional[str] = None
    summary_headline: str = ''
    strategy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    alpaca: AlpacaStatus = field(default_factory=AlpacaStatus)
    constituents: ConstituentsStatus = field(default_factory=ConstituentsStatus)
    model: ModelStatus = field(default_factory=ModelStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    regime: RegimeStatus = field(default_factory=RegimeStatus)
    training: TrainingStatus = field(default_factory=TrainingStatus)
    coverage: CoverageStatus = field(default_factory=CoverageStatus)
    last_run_utc: Optional[str] = None
    scores: List[ScoreRow] = field(default_factory=list)
    last_error: Optional[str] = None
    skipped: List[SkippedSymbol] = field(default_factory=list)
    near_misses: List[NearMiss] = field(default_factory=list)  # NEW
    watchlist_rescue_rows: List[ScoreRow] = field(default_factory=list)

    def set_scores(self, rows: List[ScoreRow], run_utc: str) -> None:
        with self.lock:
            self.scores = rows
            self.last_run_utc = run_utc
            self.last_error = None

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.last_error = msg

    def set_regime(self, regime: RegimeStatus) -> None:
        with self.lock:
            self.regime = regime

    def set_coverage(self, cov: CoverageStatus, skipped: List[SkippedSymbol], near_misses: Optional[List[NearMiss]] = None, watchlist_rescue_rows: Optional[List[ScoreRow]] = None) -> None:
        with self.lock:
            self.coverage = cov
            self.skipped = skipped[:200]
            self.near_misses = (near_misses or [])[:50]
            self.watchlist_rescue_rows = (watchlist_rescue_rows or [])[:50]


    def snapshot_scores(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "last_run_utc": self.last_run_utc,
                "rows": [r.__dict__ for r in self.scores],
                "watchlist_rescue_rows": [r.__dict__ for r in self.watchlist_rescue_rows],
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
                "regime": self.regime.__dict__,
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
                    "stage1_strong_override_count": self.coverage.stage1_strong_override_count,
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
