## What changed in v13.2

- Family profiles now influence **live scoring**, not just metadata.
- Added **family-specific calibration multipliers** on the final decision score.
- Added **family-specific decision thresholds** so stronger families can be slightly loosened and weaker families tightened.
- Weak families can now be **suppressed from signaling** when their holdout lift is poor.
- Limited-data families can still surface as watchlist candidates, but are prevented from being treated as fully actionable.
- Threshold counts and watchlist rescue now respect per-row family thresholds.

## What changed in v13.1

- Added specialist setup families: GAP_TREND, ORB_CONTINUATION, SECTOR_LEADER, PULLBACK_RECLAIM, RS_TREND, and OTHER.
- Added family-aware top-K reranking so RS Rank reflects live Prob 1%, Touch, Path, Stage 1 strength, downside, uncertainty, and learned family edge.
- Added a Setup column to Scanner results and Watchlist rescue.
- Training now records top-K holdout metrics and setup-family profiles into model metadata.

# S&P 500 Prob Scanner — v13.2.0

Production intraday scanner for **same-day S&P 500 long setups**, now with a **market-wide geopolitical regime overlay**.

## What changed in v12.3.1

- Added **RS Rank** to Scanner results: a decision-oriented rank where **#1 is the most promising** setup based on signal status, live Prob 1%, Touch, Path, downside, and uncertainty.
- Default table sort now uses **RS Rank**.
- Added horizontal and vertical scroll support to the major data tables so all columns remain readable.

## What changed in v12.3

- Model-A selection is now tail-aware: candidates are chosen for stronger top-end touch lift and strict-hit behavior, not just generic touch AUC.
- Training is now recency-weighted so newer market behavior matters more than stale samples.
- Strict/worthy positives are emphasized while hard false-positive negatives are upweighted.
- The final touch × path score now gets a strict-outcome calibrator so displayed probabilities better match the actual objective.
- New training env vars were added for recency weighting and strict calibration.

## What changed in v12.1

Version 12.1 fixes the regime-status state machine so the app can no longer quietly report **GREEN** after restart, after-hours, or before a fresh live proxy evaluation has actually completed.

Specifically:
- after market close the regime now reports **CLOSED** and shows the **last live regime** if one exists
- before a valid live proxy window exists the regime now reports **NOT_EVALUATED** instead of defaulting to GREEN
- the app persists the last live GREEN / AMBER / RED decision to disk and restores it safely on restart
- stale persisted states are shown as context only and do **not** apply live score penalties until a fresh evaluation is available
- missing Alpaca data now produces an explicit non-live regime status rather than a misleading normal-state badge

## What changed in v12

Version 12 hardens the operator layer against exactly the kind of environment you flagged: a market where a live geopolitical shock can dominate the next 1–4 hours and make ticker-level technical continuation less reliable.

The app now adds a **top-level regime engine** above the model output.

It uses liquid Alpaca-tradeable proxies to detect broad stress:
- **SPY** for broad equity stress
- **USO** for oil shock
- **VIXY** for volatility shock
- **GLD** for safe-haven demand
- **XLE** relative to SPY for energy-led geopolitical stress

This overlay can:
- reduce live touch probabilities with a regime multiplier
- cap live probabilities in stressed regimes
- raise the touch threshold required for a signal
- raise the minimum path-quality floor
- suppress new signals entirely in **RED** regimes
- apply a lighter penalty to the **Energy** sector

Important: this is a **proxy-based market regime layer**, not a direct news-ingestion engine. It is designed to make the app safer and more honest during headline-sensitive conditions without introducing brittle scraping dependencies.

## Architecture

Two-model decomposition remains intact:

- **Model A** (elastic-net): `P(touch_1pct)` — probability the stock reaches +1% from scan price before close
- **Model B** (LightGBM): `P(acceptable_path | touch)` — probability the path to +1% has MAE >= -0.6%
- **Combined live operator score**: `prob_1 = adjusted_touch * path`

## What the operator now sees

| Column | Source | Description |
|--------|--------|-------------|
| **Touch** | Model A after symbol guardrails **and** regime overlay | Primary live operator signal |
| **Path** | Model B | Path quality estimate |
| **Prob 1%** | Adjusted touch × path | Live regime-adjusted combined score |
| **Signal** | Gating logic | ACTIONABLE / CANDIDATE / blank |

Raw values are still retained in diagnostics and export data:
- `prob_touch_raw`
- `prob_touch_pre_regime`
- `prob_1_raw`
- `prob_1_pre_regime`

## Regime states

### GREEN
Normal market structure.
- no overlay penalty
- normal touch threshold
- normal path floor
- signals allowed

### AMBER
Elevated event risk.
Default behavior:
- touch multiplier: `0.80x`
- probability cap: `70%`
- touch threshold uplift: `1.10x`
- path floor add: `+0.03`
- signals still allowed unless configured otherwise

### RED
Headline-sensitive / market-wide stress regime.
Default behavior:
- touch multiplier: `0.60x`
- probability cap: `62%`
- touch threshold uplift: `1.25x`
- path floor add: `+0.08`
- new signals suppressed by default

### Sector-aware handling
- **Energy** receives a smaller penalty than the rest of the market:
  - AMBER: `0.90x`
  - RED: `0.75x`

## Automatic triggers

The regime layer uses 1-hour proxy moves and stress relationships such as:
- oil shock magnitude
- volatility shock magnitude
- safe-haven strength
- energy leadership relative to SPY
- SPY drawdown over 1 hour and since open

This is intentionally conservative. The goal is not to predict headlines. The goal is to **stop over-trusting technical continuation probabilities when the market is behaving like a macro event tape**.

## Manual override

When you know a headline just hit and want the scanner to react immediately, the app now supports a manual regime override.

Endpoints:
- `GET /api/regime`
- `POST /api/regime/override`
- `POST /api/regime/override/clear`

`POST /api/regime/override` accepts:
- `admin_password`
- `state` = `GREEN`, `AMBER`, or `RED`
- `reason`
- `duration_minutes`

Overrides are persisted to:
- `MODEL_DIR/regime_override.json`

## Existing model behavior retained

The following remain in place:
- decomposed touch × path architecture
- adaptive touch-tail threshold
- tail validation logic
- surfacing cooldown
- unvalidated messy-row suppression
- watchlist rescue lane
- post-close review/export pack

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard |
| `GET /api/status` | System status including live regime state and regime config |
| `GET /api/scores` | Live scored rows |
| `GET /api/regime` | Current regime state |
| `POST /api/regime/override` | Manually force GREEN / AMBER / RED |
| `POST /api/regime/override/clear` | Remove manual override |
| `GET /api/watchlist-rescue` | Rescue watchlist |
| `GET /api/diagnostics` | Diagnostic journal |
| `GET /api/eod-review` | Full-universe post-close review |
| `GET /api/review-export/download` | Download ZIP of named `.txt` review exports |
| `GET /api/training/status` | Training status |
| `POST /train` | Trigger training |

## Key v12 environment variables

```env
REGIME_ENABLED=true
REGIME_OIL_PROXY=USO
REGIME_VOL_PROXY=VIXY
REGIME_SAFE_HAVEN_PROXY=GLD
REGIME_ENERGY_PROXY=XLE

REGIME_AMBER_OIL_MOVE_1H=0.010
REGIME_RED_OIL_MOVE_1H=0.020
REGIME_AMBER_VOL_MOVE_1H=0.030
REGIME_RED_VOL_MOVE_1H=0.060
REGIME_AMBER_SAFE_HAVEN_MOVE_1H=0.004
REGIME_RED_SAFE_HAVEN_MOVE_1H=0.008
REGIME_AMBER_ENERGY_REL_SPY_1H=0.005
REGIME_RED_ENERGY_REL_SPY_1H=0.010
REGIME_AMBER_SPY_DROP_1H=-0.003
REGIME_RED_SPY_DROP_1H=-0.006
REGIME_AMBER_SPY_DROP_SINCE_OPEN=-0.006
REGIME_RED_SPY_DROP_SINCE_OPEN=-0.012

REGIME_AMBER_MULTIPLIER=0.80
REGIME_RED_MULTIPLIER=0.60
REGIME_AMBER_PROB_CAP=0.70
REGIME_RED_PROB_CAP=0.62
REGIME_AMBER_TOUCH_THRESHOLD_MULT=1.10
REGIME_RED_TOUCH_THRESHOLD_MULT=1.25
REGIME_AMBER_PATH_FLOOR_ADD=0.03
REGIME_RED_PATH_FLOOR_ADD=0.08
REGIME_AMBER_SUPPRESS_SIGNALS=false
REGIME_RED_SUPPRESS_SIGNALS=true
REGIME_ENERGY_AMBER_MULTIPLIER=0.90
REGIME_ENERGY_RED_MULTIPLIER=0.75
```

## Operator interpretation in crisis conditions

In calm markets, the scanner output is close to a conventional probability layer.

In stressed geopolitical conditions, treat it as:
- **a regime-adjusted opportunity score with calibration discipline**, not
- a naive claim that the market is still behaving like the training set

That is the point of v12.

