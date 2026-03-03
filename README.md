# S&P 500 Prob Scanner — v8 (reliability + identification)

Goal: Identify stocks with a **high probability** of reaching **+1% (or +2%)** from scan time **at any point until market close**, using the label:
- `Y=1` if max future **1-minute HIGH** from scan time to close >= (1+X)*P0 else 0 (X=0.01 / 0.02)

## What changed vs v7
### Identification (ranking/top-N)
- Elastic-net logistic regression (saga) with a small hyperparameter grid.
- Adds **auditable interaction features** (momentum×volume, momentum×VWAP, trend×ADX, momentum×time).

### Probability reliability
- **Time-to-close bucketed calibration** (5 buckets).
- **Prior blending (shrinkage)** by bucket: p = α·p_cal + (1−α)·p_prior, α selected by validation Brier.

### Microstructure realism (aligns to HIGH-based label)
Adds numeric microstructure features to the model:
- log rolling median dollar-volume
- rolling median range%
- rolling median wickiness/ATR

### v7 features retained
- ToD-RVOL profiles persisted under `MODEL_DIR/volume_profiles/`
- Liquidity risk flag + reason codes
- Coverage diagnostics in `/api/status` and `/api/debug/coverage` (password protected)
- Demo mode, SIP enforced, DST-safe regular session only, single worker.

## Endpoints
- `GET /` dashboard
- `GET /health`
- `GET /api/status`
- `GET /api/scores`
- `POST /train`
- `GET /api/training/status`
- `GET /api/debug/coverage?password=...`

## Env vars

### Required for live mode
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_DATA_FEED` (ignored if not `sip`; app enforces SIP)
- `TIMEZONE` (default `America/New_York`)

### Scanner
- `SCAN_INTERVAL_MINUTES` (default `5`)
- `MIN_BARS_5M` (default `7`)

### Training
- `ADMIN_PASSWORD`
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_SYMBOLS` (default `0` = all)
- `CALIB_MIN_BUCKET_SAMPLES` (default `200`)
- `ENET_C_VALUES` (default `0.5,1.0`)
- `ENET_L1_VALUES` (default `0.0,0.5`)
- `PRIOR_ALPHA_VALUES` (default `0.6,0.7,0.8,0.9`)

### Storage
- `MODEL_DIR` (default `./runtime/model`; recommend `/var/data/model`)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`)
- `DEBUG_PASSWORD` (optional; else uses ADMIN_PASSWORD)

### ToD-RVOL
- `TOD_RVOL_LOOKBACK_DAYS` (default `20`)
- `TOD_RVOL_MIN_DAYS` (default `8`)

### Liquidity thresholds (risk flag)
- `LIQ_ROLLING_BARS` (default `12`)
- `LIQ_DVOL_MIN_USD` (default `2000000`)
- `LIQ_RANGE_PCT_MAX` (default `0.012`)
- `LIQ_WICK_ATR_MAX` (default `0.8`)

## Render checklist
- Web Service, Dockerfile build
- Health check: `/health`
- Persistent disk mounted at `/var/data`
- Set `MODEL_DIR=/var/data/model`
