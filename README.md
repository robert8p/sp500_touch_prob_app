# S&P 500 Prob Scanner — v8.0.2

Goal: identify stocks with a high probability of reaching +1% or +2% from scan time to some point before the close, using:
- `Y=1` if max future **1-minute HIGH** from scan time to close >= `(1+X)*P0` else 0 (X = 0.01 / 0.02)

## Highlights
- Elastic-net logistic regression with auditable interaction features
- Time-to-close bucketed calibration + prior blending
- ToD-RVOL profiles persisted under `MODEL_DIR/volume_profiles/`
- Liquidity / microstructure risk flag
- Coverage diagnostics in `/api/status`
- Training result persistence across restarts
- **v8.0.2 transparency patch**: `risk_reasons` shown separately from directional `reasons`

## Endpoints
- `GET /`
- `GET /health`
- `GET /api/status`
- `GET /api/scores`
- `POST /train`
- `GET /api/training/status`
- `GET /api/debug/coverage?password=...`
