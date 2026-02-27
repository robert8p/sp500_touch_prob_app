# S&P 500 Prob Scanner (FastAPI) — v6 reliability upgrade

This build upgrades the **probability reliability** and **cross-symbol robustness**:
- scale-invariant trend (EMA diff normalized)
- improved time-to-close encoding (fraction + log + time-of-day fraction)
- relative strength vs SPY
- OBV slope normalization
- model selection by **Brier score** on held-out days (Platt vs isotonic; balanced vs unweighted)

## Endpoints
- `GET /` dashboard
- `GET /health` health check
- `GET /api/status` market + alpaca + model/training status
- `GET /api/scores` rows with `prob_1` and `prob_2`
- `POST /train` start training (password protected)
- `GET /api/training/status` training job status

## Environment variables

### Required for live mode
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_DATA_FEED` (ignored if not `sip`; app enforces SIP)
- `TIMEZONE` (default `America/New_York`)

### Scanner
- `SCAN_INTERVAL_MINUTES` (default `5`)

### Training
- `ADMIN_PASSWORD`
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_SYMBOLS` (default `0` = no cap / train on all S&P 500; set e.g. `200` to reduce runtime)

### Storage
- `MODEL_DIR` (default `./runtime/model`; recommend `/var/data/model` on Render)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`)

## Render deployment checklist
- Web Service (single service)
- Dockerfile build
- Health check path: `/health`
- Persistent disk: enabled, mounted at `/var/data`
- Env vars:
  - `ALPACA_API_KEY=<value>`
  - `ALPACA_API_SECRET=<value>`
  - `ALPACA_DATA_FEED=sip`
  - `TIMEZONE=America/New_York`
  - `MODEL_DIR=/var/data/model`
  - `SCAN_INTERVAL_MINUTES=5`
  - `ADMIN_PASSWORD=<value>`
  - `TRAIN_LOOKBACK_DAYS=60`
  - `TRAIN_MAX_SYMBOLS=0`

Debug-first deploy:
- `DEMO_MODE=true`
- `DISABLE_SCHEDULER=1`
Then switch to live:
- `DEMO_MODE=false`
- `DISABLE_SCHEDULER=0`

## Note
If you deploy v6 on top of older model artifacts, retrain. v6 adds/changes feature definitions, so old artifacts are treated as incompatible and the app will fall back to heuristic until retrained.
