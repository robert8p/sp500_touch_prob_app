# S&P 500 Prob Scanner (FastAPI) — v7

Primary goal: improve probability reliability + scientific auditability **without changing** the definition of Prob 1% / Prob 2%
(labels remain based on future 1-minute **HIGH** to close).

## What’s new in v7

### 1) Time-of-day normalized RVOL (ToD-RVOL)
- Builds per-symbol 5-minute **volume profiles** across the last `TOD_RVOL_LOOKBACK_DAYS` trading days.
- For each symbol and each of the 78 regular-session 5m slots, stores median (and IQR).
- Persists under: `MODEL_DIR/volume_profiles/` (survives restarts on Render persistent disk).
- Runtime feature: `rvol_tod = current_5m_volume / baseline_slot_median_volume`
- If profiles are missing/unavailable, scanner falls back safely and surfaces it in `/api/status`.

### 2) Liquidity / microstructure risk flag (no quotes endpoint)
Adds a **Risk** column and reason-code hints (e.g. `LOW_LIQ`, `WICKY`), using cheap bar-based proxies:
- dollar volume
- range %
- wickiness (wicks / ATR)

No filtering: symbols are **not** excluded, only flagged.

### 3) Coverage diagnostics
`/api/status` includes explicit coverage counts and top skip reasons.
A password-protected endpoint `/api/debug/coverage` returns sample skipped-symbol diagnostics.

## Endpoints
- `GET /` dashboard
- `GET /health` health check
- `GET /api/status` market + alpaca + model/training + coverage diagnostics
- `GET /api/scores` score rows (includes Risk)
- `POST /train` one-click training (builds pt1 + pt2 models)
- `GET /api/training/status`
- `GET /api/debug/coverage?password=...` (protected)

## Environment variables

### Required for live mode
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_DATA_FEED` (ignored if not `sip`; app enforces SIP)
- `TIMEZONE` (default `America/New_York`)

### Scanner
- `SCAN_INTERVAL_MINUTES` (default `5`)
- `MIN_BARS_5M` (default `7`) minimum cached 5m bars to score a symbol

### Training
- `ADMIN_PASSWORD`
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_SYMBOLS` (default `0` = no cap / train on all S&P 500; set e.g. `200` to reduce runtime)

### Storage
- `MODEL_DIR` (default `./runtime/model`; recommend `/var/data/model` on Render)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`)
- `DEBUG_PASSWORD` (optional; if set, protects `/api/debug/coverage`; otherwise uses `ADMIN_PASSWORD`)

### ToD-RVOL
- `TOD_RVOL_LOOKBACK_DAYS` (default `20`)
- `TOD_RVOL_MIN_DAYS` (default `8`) min distinct days required; else profile marked unavailable

### Liquidity risk thresholds
- `LIQ_ROLLING_BARS` (default `12`)
- `LIQ_DVOL_MIN_USD` (default `2000000`)
- `LIQ_RANGE_PCT_MAX` (default `0.012`)
- `LIQ_WICK_ATR_MAX` (default `0.8`)

## Render deployment checklist
- Web Service (single service)
- Dockerfile build
- Health check: `/health`
- Persistent disk: enabled, mounted at `/var/data`
- Recommended env vars:
  - `ALPACA_API_KEY=<value>`
  - `ALPACA_API_SECRET=<value>`
  - `ALPACA_DATA_FEED=sip`
  - `TIMEZONE=America/New_York`
  - `MODEL_DIR=/var/data/model`
  - `SCAN_INTERVAL_MINUTES=5`
  - `MIN_BARS_5M=7`
  - `ADMIN_PASSWORD=<value>`
  - `TRAIN_LOOKBACK_DAYS=60`
  - `TRAIN_MAX_SYMBOLS=0`
  - `TOD_RVOL_LOOKBACK_DAYS=20`
  - `TOD_RVOL_MIN_DAYS=8`

Debug-first deploy:
- `DEMO_MODE=true`
- `DISABLE_SCHEDULER=1`
Then switch to live:
- `DEMO_MODE=false`
- `DISABLE_SCHEDULER=0`

## Notes for reviewers
- Probabilities remain calibrated logistic models (Platt vs isotonic; class_weight none vs balanced; selection by validation Brier).
- If model artifacts are missing or schema mismatched, the app falls back to heuristic and surfaces a warning.
