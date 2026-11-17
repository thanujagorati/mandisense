# MandiSense: Project Documentation

**Track:** Agriculture | **Event:** Code-X-Novas Innovathon
**Team / author:** Gorati Thanuja, GNITS Hyderabad
**Repository:** https://github.com/thanujagorati/mandisense | 

## 1. Problem

Onion prices in Indian mandis can swing sharply within a week. Farmers, FPO (Farmer Producer Organisation) coordinators and small traders usually decide *when* to sell and *where* to send a lot based on gut feeling or word of mouth. A wrong call on a large lot can cost thousands of rupees, and there is no simple tool that shows the odds behind a recommendation.

## 2. Solution

MandiSense is a decision-support tool, aimed at FPO coordinators and extension officers who advise farmers. For a chosen mandi, date and quantity it provides:

1. **A 7-day price-direction forecast**: probability that the price will be down (below -3%), flat, or up (above +3%).
2. **Sell-now-or-wait advice**: computed from expected net money in hand after holding costs, with a confidence label and plain-language reasons.
3. **A mandi price comparison** across 8 nearby mandis.
4. **An FPO lot-split what-if simulator**: how a large lot could be shared across up to 3 mandis, net of transport.

**Replay mode.** The app runs on real historical prices (2014-2024). The user picks a past day; the model sees only data up to that day; the user can then reveal what actually happened. This makes every recommendation verifiable, and it lets us show misses honestly.

## 3. How it works

### 3.1 Data
Daily onion prices from Agmarknet, via the open dataset by R. Shelar, A. Hajare and S. Mangarule (Mendeley Data, CC BY 4.0). We use a cluster of 8 nearby mandis (including Gudimalkapur and Bowenpally in Telangana). Several variety/grade rows per market-day are collapsed into one arrivals-weighted modal price. 21,917 market-day observations are stored.

### 3.2 Database (PostgreSQL on Neon)
Normalized tables: `commodities`, `mandis` (with coordinates), `price_observations` (composite primary key on mandi, commodity, date; CHECK constraint on positive price) and `forecasts`. Loading is idempotent (upserts in a single transaction).

### 3.3 Forecast model
- scikit-learn `HistGradientBoostingClassifier`, three classes (down / flat / up), 7-day horizon, flat = within ±3%.
- Features built from price history only: 1/7/30-day price changes, ratios to 7/30-day moving averages, 7-day volatility, arrivals and their changes, month, day of week, mandi.
- **Walk-forward backtest:** test years 2022, 2023, 2024; training stops 7 days before each test year, so there is no look-ahead leakage.

### 3.4 Decision engine (`app/engine.py`)
Pure, unit-tested functions. Expected 7-day price change is the probability-weighted average of typical moves for each outcome, measured on 2014-2021 data only. Waiting must beat selling now by a 2% margin after a holding cost (0.3% per day). Confidence is *high* at top-class probability ≥ 0.60, *moderate* ≥ 0.45, otherwise *low*.

### 3.5 FPO split simulator (`app/split.py`)
Enumerates ways to split a lot in ~10-quintal chunks across up to 3 mandis (minimum 30 quintals per mandi). Transport cost uses haversine distance × a road factor of 1.3, plus freight and handling. Because extra arrivals may depress a mandi's price, revenue uses an adjustable price-flexibility assumption (default -0.25, capped at a 50% drop).

### 3.6 API and UI
FastAPI service with Pydantic models and versioned `/api/v1` routes (mandis, recommend, compare, split, backtest), paging, and 404/503 handling; a fresh read-only database connection is opened per request because the hosted database idles. A static web UI (vanilla JavaScript + Leaflet map) is served by the same service. Interactive API docs are at `/docs`. 16 automated tests cover the engine and the splitter.

## 4. Results and honest findings

| Finding | Detail |
|---|---|
| Forecast accuracy | About **50.8%** across 2022-2024, versus about **40%** for the best constant guess. A real but modest edge: a lean, not a guarantee. Details in `outputs/backtest_metrics.json`. |
| "Move to another mandi" advice | **Removed.** Nearby mandis show persistent, noisy price gaps that are most likely grade-mix differences the data cannot separate. A backtest of the rule gave implausible results (it fired on about 54% of days), so the product shows a price comparison instead. |
| Price impact of large lots | **Could not be measured** (coefficient +0.009, R² = 0.002). The split tool is therefore a what-if simulator with an explicit assumption. At a flexibility of 0 there is no gain from splitting. |

We chose to report these limits rather than hide them: a tool that advises on money should say how sure it is.

## 5. Limitations

- Forecasts are only available for 2022-2024 replay days; the system does not yet ingest live prices.
- Onion only, 8 mandis.
- Freight, handling and holding-cost values are defaults and have not been checked against real rates.
- Prices are modal prices; grade and quality differences are not modelled.
- Mandi coordinates are city-level approximations.

## 6. Tech stack

Python 3.12, PostgreSQL (Neon), psycopg2, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, Pydantic, Leaflet, pytest, Git/GitHub.

## 7. Run it

See `README.md`. In short: create `.env` with `DATABASE_URL`, run `python -m pip install -r requirements.txt`, then `python -m uvicorn app.main:app --reload` and open http://127.0.0.1:8000/.

## 8. Future work

- Live ingestion of daily Agmarknet prices and a scheduled forecast refresh.
- Telugu and Hindi interface, plus SMS/WhatsApp delivery for coordinators.
- Grade-aware comparison to make "move to another mandi" advice trustworthy.
- Real freight quotes and storage-loss data per FPO.
- More crops (tomato, potato) and more mandis.

## 9. Credits

Price data: R. Shelar, A. Hajare, S. Mangarule, onion dataset 2014-2024 (Mendeley Data, CC BY 4.0), derived from Agmarknet. Map: OpenStreetMap contributors.
