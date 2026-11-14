"""MandiSense API. Run locally from the mandisense folder:

    python -m uvicorn app.main:app --reload

then open http://127.0.0.1:8000/docs
"""
from datetime import date as Date

import psycopg2
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app import service
from app.db import cursor
from app.engine import Forecast, confidence_label

app = FastAPI(
    title="MandiSense API",
    version="1.0.0",
    description=(
        "Decision support for onion sellers: a 7-day price direction forecast with a confidence level, "
        "a sell-now-or-wait recommendation, a mandi price comparison, and an FPO lot-splitting what-if. "
        "Data: Agmarknet prices (via a public research dataset), 2014-2024. Forecasts exist for days in "
        "2022-2024 ('replay mode'). Every assumption is returned with the answer."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.exception_handler(service.NotFound)
async def _not_found(_, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(psycopg2.OperationalError)
async def _db_unavailable(_, exc):
    return JSONResponse(status_code=503, content={
        "detail": "The database is not reachable right now (it may be waking up). Try again in a few seconds."})


# ------------------------------------------------------------------ response models
class MandiOut(BaseModel):
    id: int
    name: str
    state: str
    district: str
    lat: float | None
    lon: float | None
    first_date: Date
    last_date: Date
    days_with_price: int


class PricePoint(BaseModel):
    date: Date
    modal_price_rs_qtl: float
    min_price_rs_qtl: float | None
    max_price_rs_qtl: float | None
    arrivals_tonnes: float | None


class PricePage(BaseModel):
    mandi: str
    total: int
    limit: int
    offset: int
    items: list[PricePoint]


class ForecastOut(BaseModel):
    mandi: str
    forecast_date: Date
    horizon_days: int
    p_down: float
    p_flat: float
    p_up: float
    predicted_class: str
    confidence: str
    note: str


class OptionOut(BaseModel):
    action: str
    mandi: str
    gross_rs: float
    cost_rs: float
    net_rs: float
    note: str


class ReplayOut(BaseModel):
    price_7_days_later: float | None
    change_pct: float | None
    net_if_sold_now: float
    net_if_waited: float | None
    net_following_advice: float | None


class RecommendationOut(BaseModel):
    mandi: str
    date: Date
    quintals: float
    price_today_rs_qtl: float
    action: str
    confidence: str
    reasons: list[str]
    forecast: dict[str, float]
    options: list[OptionOut]
    typical_moves: dict[str, float]
    assumptions: dict
    replay: ReplayOut
    notes: list[str]


class CompareRow(BaseModel):
    mandi: str
    price_3day_avg_rs_qtl: float
    usual_price_vs_home: float
    vs_usual_pct: float
    road_km: float | None
    freight_rs_per_qtl: float | None


class CompareOut(BaseModel):
    home: str
    date: Date
    home_price_rs_qtl: float
    rows: list[CompareRow]
    notes: list[str]


class PlanOut(BaseModel):
    allocation: dict[str, float]
    net_rs: float
    transport_rs: float


class SensitivityOut(BaseModel):
    flexibility: float
    gain_rs: float
    gain_pct: float
    best_plan: dict[str, float]


class OutletOut(BaseModel):
    mandi: str
    usual_daily_arrivals_qtl: float
    road_km: float


class SplitOut(BaseModel):
    home: str
    date: Date
    tonnes: float
    base_price_rs_qtl: float
    flexibility_used: float
    flexibility_note: str
    outlets: list[OutletOut]
    best_single: PlanOut
    best_plan: PlanOut
    gain_rs: float
    gain_pct: float
    sensitivity: list[SensitivityOut]
    notes: list[str]


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["system"])
def health():
    """Checks that the API is up and can reach the database."""
    with cursor() as cur:
        cur.execute("SELECT 1 AS ok")
        cur.fetchone()
    return {"status": "ok", "database": "ok"}


@app.get("/api/v1/mandis", response_model=list[MandiOut], tags=["data"])
def mandis():
    """The mandis in our cluster, with the date range we have prices for."""
    return service.list_mandis()


@app.get("/api/v1/prices", response_model=PricePage, tags=["data"])
def prices(mandi: str, start: Date | None = None, end: Date | None = None,
           limit: int = Query(500, ge=1, le=2000), offset: int = Query(0, ge=0)):
    """Daily modal price (Rs per quintal) and arrivals (tonnes) for one mandi, with paging."""
    m = service.get_mandi(mandi)
    total, items = service.price_series(m["id"], start, end, limit, offset)
    return {"mandi": m["name"], "total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/api/v1/forecast", response_model=ForecastOut, tags=["advice"])
def forecast(mandi: str, date: Date):
    """The model's 7-day outlook (down / flat / up probabilities) as of a past date."""
    m = service.get_mandi(mandi)
    row = service.forecast_row(m["id"], date)
    if row is None:
        raise service.NotFound(f"No forecast for {m['name']} on {date}. Forecasts exist for days in 2022-2024 "
                               "when the mandi reported a price.")
    f = Forecast(row["p_down"], row["p_flat"], row["p_up"])
    return {"mandi": m["name"], "forecast_date": row["forecast_date"], "horizon_days": row["horizon_days"],
            "p_down": row["p_down"], "p_flat": row["p_flat"], "p_up": row["p_up"],
            "predicted_class": row["predicted_class"], "confidence": confidence_label(f),
            "note": "Flat means within +/-3% of today's price. In our backtest, calls with confidence of "
                    "0.60 or more were right about 66% of the time."}


@app.get("/api/v1/recommend", response_model=RecommendationOut, tags=["advice"])
def recommend_endpoint(home: str, date: Date, quintals: float = Query(80, gt=0, le=5000)):
    """Sell now or wait 7 days, with reasons, assumptions, and (replay mode) what really happened."""
    m = service.get_mandi(home)
    return service.recommend_view(m, date, quintals)


@app.get("/api/v1/compare", response_model=CompareOut, tags=["advice"])
def compare(home: str, date: Date):
    """How other mandis' prices compare with their usual gap to your home mandi. Information only."""
    m = service.get_mandi(home)
    return service.compare_table(m, date)


@app.get("/api/v1/split", response_model=SplitOut, tags=["advice"])
def split(home: str, date: Date, tonnes: float = Query(8, gt=0, le=60),
          flexibility: float | None = Query(None, ge=-1.0, le=0.0,
                                            description="Price effect of extra arrivals; 0 means none.")):
    """FPO what-if: is it worth sharing a big lot between mandis? Illustrative, see the notes."""
    m = service.get_mandi(home)
    return service.split_view(m, date, tonnes, flexibility)


@app.get("/api/v1/backtest", tags=["evidence"])
def backtest():
    """The walk-forward test of the forecast: model against two simple baselines, year by year."""
    return service.backtest_metrics()


@app.get("/api/v1/assumptions", tags=["evidence"])
def assumptions():
    """Every adjustable assumption used by the decision and split engines."""
    return service.assumptions_view()
