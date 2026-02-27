# S&P 500 Prob Scanner (FastAPI)

A production-oriented FastAPI web app that scans the S&P 500 during US regular market hours and displays:

- **Prob 1%**: probability of reaching **+1%** from scan time to **today's close**
- **Prob 2%**: probability of reaching **+2%** from scan time to **today's close**

The scanner runs every `SCAN_INTERVAL_MINUTES` aligned to candle close. Models are trained from Alpaca SIP market data and saved to:

- `MODEL_DIR/pt1` (1% model)
- `MODEL_DIR/pt2` (2% model)

## Key behaviors / guarantees

- **Reliability-first:** the dashboard loads and the scanner runs even if training never ran.
- **Heuristic fallback model** is used until trained artifacts exist.
- **SIP feed enforced** for historical + live (the app will always use `sip`; `ALPACA_DATA_FEED` is kept for compatibility but non-`sip` values are ignored).
- **US regular session only**: 09:30–16:00 America/New_York (DST handled).
- **Demo mode**: set `DEMO_MODE=true` and the UI loads without Alpaca keys.
- **One-click training** from the dashboard (`/train`) protected by `ADMIN_PASSWORD`.
- **Scanner always uses the full S&P 500 universe** (including share-class tickers like `BRK.B`). `TRAIN_MAX_SYMBOLS` affects training only.

## Endpoints

- `GET /` dashboard
- `GET /health` health check
- `GET /api/status` status JSON (market + alpaca + model + training)
- `GET /api/scores` rows with `prob_1` and `prob_2`
- `POST /train` start training (password protected)
- `GET /api/training/status` training job status JSON

## Environment variables

### Required for live mode
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_DATA_FEED` (default `sip` — recommended)
- `TIMEZONE` (default `America/New_York`)

### Scanner
- `SCAN_INTERVAL_MINUTES` (default `5`)

### Training
- `ADMIN_PASSWORD`
- `TRAIN_LOOKBACK_DAYS` (default `60`)
- `TRAIN_MAX_SYMBOLS` (default `0` = no cap / train on all S&P 500 symbols). Set to e.g. `200` if you want faster training.

### Storage
- `MODEL_DIR` (default `./runtime/model` — recommend `/var/data/model` on Render persistent disk)

### Debug
- `DEMO_MODE` (default `false`)
- `DISABLE_SCHEDULER` (default `0`; if `1`, the background scheduler does not run but the API still works)

---

## Deploy to Render (single Web Service)

**Known-good checklist (match this exactly):**

1. Create a new **Web Service** in Render (single service).
2. Connect your GitHub repo.
3. **Build method:** Dockerfile.
4. **Instance type:** Standard.
5. **Health check path:** `/health`
6. Enable **Persistent Disk** and mount to: `/var/data`
7. Set these environment variables in Render:

**Live mode**
- `ALPACA_API_KEY=<value>`
- `ALPACA_API_SECRET=<value>`
- `ALPACA_DATA_FEED=sip`
- `TIMEZONE=America/New_York`
- `MODEL_DIR=/var/data/model`
- `SCAN_INTERVAL_MINUTES=5`
- `ADMIN_PASSWORD=<value>`
- `TRAIN_LOOKBACK_DAYS=60`
- `TRAIN_MAX_SYMBOLS=0`

**Debug procedure (prove UI works before live keys)**
1. First deploy with:
   - `DEMO_MODE=true`
   - `DISABLE_SCHEDULER=1`
2. Confirm:
   - `/` loads
   - `/api/status` returns JSON
   - `/api/scores` returns JSON
3. Then set:
   - `DEMO_MODE=false`
   - `DISABLE_SCHEDULER=0`
4. Redeploy.

---

## Notes on probability definition

At each scan time for each symbol:

- Let **P0** be the current price (latest 5-minute bar close at scan time).
- Let **H_future** be the maximum future **1-minute HIGH** from scan time until today’s close.
- Label **Y = 1** if `H_future >= (1+X) * P0`, else 0, where `X` is 0.01 or 0.02.

The displayed probability is a calibrated probability from the trained model, or a stable heuristic fallback if no model artifacts exist.

---

## Local run

```bash
pip install -r requirements.txt
export DEMO_MODE=true
uvicorn app.main:app --reload
```

Open http://localhost:8000

