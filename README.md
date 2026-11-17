# MandiSense

Decision support for onion sellers and FPOs (Farmer Producer Organisations): a 7-day price-direction forecast with confidence, sell-now-or-wait advice, a mandi price comparison, and a what-if tool for splitting a large lot across mandis.

Built for the Code-X-Novas Innovathon (Agriculture track).

> **Replay mode.** The app works on real historical Agmarknet prices (2014-2024). You pick a past day, the model sees only data up to that day, and you can then reveal what actually happened. This makes every claim checkable.

## What it does

1. **Forecast**: probability that the modal price 7 days ahead is down (< -3%), flat, or up (> +3%).
2. **Advice**: SELL NOW or WAIT, from expected net money after holding cost, with a confidence label and plain-language reasons.
3. **Compare**: today's prices across 8 nearby mandis.
4. **FPO split what-if**: how a lot of 1-60 tonnes could be shared across up to 3 mandis, net of transport.

## Architecture

```
Mendeley/Agmarknet parquet --> scripts/load_data.py --> Neon PostgreSQL
                                                          |
              scripts/train_forecast.py (sklearn HGB) ----+--> forecasts table
                                                          |
         app/engine.py + app/split.py (pure functions) <--+
                         |
                  FastAPI  /api/v1/*   (app/main.py, service.py, db.py)
                         |
              web/ (vanilla JS + Leaflet map)
```

- **Database**: PostgreSQL (Neon), normalized schema, composite primary keys, CHECK constraints, idempotent upserts.
- **API**: FastAPI with Pydantic models, versioned routes, paging, 404/503 handling. Interactive docs at `/docs`.
- **Engine**: decision logic is plain, unit-tested functions (`tests/`, 16 tests).

## Run locally

```bash
python -m pip install -r requirements.txt
# create .env with:  DATABASE_URL=<your postgres url>
python -m uvicorn app.main:app --reload
# open http://127.0.0.1:8000/   (API docs: /docs)
```

To rebuild from scratch (needs the dataset in `data/raw/`, see Data):
```bash
python -m pip install -r requirements-dev.txt
psql "$DATABASE_URL" -f db/schema.sql -f db/002_forecasts.sql
python scripts/load_data.py
python scripts/train_forecast.py
python -m pytest
```

## Honest results

- **Forecast edge is real but modest.** Walk-forward backtest on 2022, 2023 and 2024 (training stops 7 days before each test year): about **50.8% accuracy vs about 40%** for the best constant guess. It is a lean, not a guarantee. Full numbers in `outputs/backtest_metrics.json`.
- **"Move to another mandi" advice was removed.** Nearby mandis show persistent, noisy price gaps. These are likely grade-mix differences the data cannot separate, so the backtest looked implausible. We show a comparison table instead of advice.
- **Price impact of large lots could not be measured** (coefficient +0.009, R² = 0.002, `outputs/price_impact.json`). So the split tool is a what-if simulator with an explicit, adjustable "illustrative, NOT measured" flexibility assumption. At 0 there is no gain.
- **Cost assumptions** (freight, handling, holding cost) are defaults in `app/engine.py` and are not verified against real rates.

## Data

Daily onion prices from Agmarknet, via the open research dataset by R. Shelar, A. Hajare and S. Mangarule (Mendeley Data, CC BY 4.0). Map tiles: OpenStreetMap contributors, CARTO.

This is decision support built from historical data, not financial advice.
