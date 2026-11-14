"""Data access and calculations behind the API. The routes in main.py stay thin."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date as Date
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.db import cursor
from app.engine import (DEFAULT_MEAN_RETURNS, Assumptions, Forecast, MandiQuote, recommend,
                        road_distance_km)
from app.split import Outlet, SplitAssumptions, optimise

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
CUTOFF = pd.Timestamp("2022-01-01")     # typical moves are measured only before this date
SENSITIVITY = (0.0, -0.10, -0.25, -0.40)


class NotFound(Exception):
    pass


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


# ---------------------------------------------------------------- simple lookups
def get_mandi(name: str) -> dict:
    with cursor() as cur:
        cur.execute("SELECT id, name, state, district, lat, lon FROM mandis WHERE lower(name) = lower(%s)",
                    (name,))
        row = cur.fetchone()
    if not row:
        raise NotFound(f"Unknown mandi '{name}'. See /api/v1/mandis for the list.")
    return dict(row)


def list_mandis() -> list:
    with cursor() as cur:
        cur.execute("""SELECT m.id, m.name, m.state, m.district, m.lat, m.lon,
                              MIN(p.obs_date) AS first_date, MAX(p.obs_date) AS last_date,
                              COUNT(*) AS days_with_price
                       FROM mandis m JOIN price_observations p ON p.mandi_id = m.id
                       GROUP BY m.id ORDER BY m.name""")
        return [dict(r) for r in cur.fetchall()]


def price_series(mandi_id: int, start, end, limit: int, offset: int) -> tuple:
    where = ("WHERE mandi_id = %s AND (%s::date IS NULL OR obs_date >= %s::date) "
             "AND (%s::date IS NULL OR obs_date <= %s::date)")
    params = (mandi_id, start, start, end, end)
    with cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM price_observations {where}", params)
        total = int(cur.fetchone()["n"])
        cur.execute(f"""SELECT obs_date AS date, modal_price_rs_qtl::float AS modal_price_rs_qtl,
                               min_price_rs_qtl::float AS min_price_rs_qtl,
                               max_price_rs_qtl::float AS max_price_rs_qtl,
                               arrivals_tonnes::float AS arrivals_tonnes
                        FROM price_observations {where} ORDER BY obs_date LIMIT %s OFFSET %s""",
                    params + (limit, offset))
        return total, [dict(r) for r in cur.fetchall()]


def forecast_row(mandi_id: int, d: Date):
    with cursor() as cur:
        cur.execute("""SELECT forecast_date, horizon_days, p_down::float AS p_down, p_flat::float AS p_flat,
                              p_up::float AS p_up, predicted_class, model_version
                       FROM forecasts WHERE mandi_id = %s AND forecast_date = %s AND horizon_days = 7
                       ORDER BY model_version DESC LIMIT 1""", (mandi_id, d))
        row = cur.fetchone()
    return dict(row) if row else None


def price_on(mandi_id: int, d: Date, max_age_days: int = 0):
    """Latest price on or before d, no older than max_age_days."""
    with cursor() as cur:
        cur.execute("""SELECT modal_price_rs_qtl::float AS price FROM price_observations
                       WHERE mandi_id = %s AND obs_date <= %s AND obs_date >= %s
                       ORDER BY obs_date DESC LIMIT 1""", (mandi_id, d, d - timedelta(days=max_age_days)))
        row = cur.fetchone()
    return None if row is None else float(row["price"])


def window_prices(d: Date, days: int) -> pd.DataFrame:
    with cursor() as cur:
        cur.execute("""SELECT mandi_id, obs_date AS date, modal_price_rs_qtl::float AS price,
                              arrivals_tonnes::float AS arrivals
                       FROM price_observations WHERE obs_date <= %s AND obs_date >= %s""",
                    (d, d - timedelta(days=days)))
        rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows, columns=["mandi_id", "date", "price", "arrivals"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _all_mandis_frame() -> pd.DataFrame:
    with cursor() as cur:
        cur.execute("SELECT id, name, lat, lon FROM mandis")
        return pd.DataFrame([dict(r) for r in cur.fetchall()], columns=["id", "name", "lat", "lon"])


# ---------------------------------------------------------------- typical moves (cached)
@lru_cache(maxsize=1)
def typical_moves() -> dict:
    """Average 7-day price change when the outcome was down / flat / up, measured on 2014-2021 only."""
    with cursor() as cur:
        cur.execute("""SELECT mandi_id, obs_date AS date, modal_price_rs_qtl::float AS price
                       FROM price_observations WHERE obs_date < %s""", (CUTOFF.date(),))
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return dict(DEFAULT_MEAN_RETURNS)
    prices = pd.DataFrame(rows)
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    old = prices.copy()
    old["target"] = old["date"] + pd.Timedelta(days=7)
    old = old.sort_values("target")
    later = prices.rename(columns={"date": "target_date", "price": "price7"}).sort_values("target_date")
    m = pd.merge_asof(old, later, left_on="target", right_on="target_date", by="mandi_id",
                      tolerance=pd.Timedelta(days=2), direction="backward")
    m["ret"] = m["price7"] / m["price"] - 1
    m = m.dropna(subset=["ret"])
    label = np.where(m["ret"] > 0.03, "up", np.where(m["ret"] < -0.03, "down", "flat"))
    out = m.groupby(label)["ret"].mean().to_dict()
    return {c: float(out.get(c, DEFAULT_MEAN_RETURNS[c])) for c in ("down", "flat", "up")}


# ---------------------------------------------------------------- recommendation (sell now vs wait)
def recommend_view(home: dict, d: Date, quintals: float) -> dict:
    price = price_on(home["id"], d, 0)
    if price is None:
        raise NotFound(f"No price for {home['name']} on {d}.")
    fc = forecast_row(home["id"], d)
    if fc is None:
        raise NotFound(f"No forecast for {home['name']} on {d}. Forecasts exist for days in 2022-2024 "
                       "when the mandi reported a price.")
    forecast = Forecast(fc["p_down"], fc["p_flat"], fc["p_up"])
    moves = typical_moves()
    a = Assumptions()
    home_q = MandiQuote(home["name"], price, _num(home["lat"]), _num(home["lon"]))
    rec = recommend(home_q, [], forecast, quintals, moves, a)     # no MOVE advice: it could not be validated

    sell_net = price * quintals
    later = price_on(home["id"], d + timedelta(days=7), 2)
    replay = {"price_7_days_later": later, "change_pct": None, "net_if_sold_now": sell_net,
              "net_if_waited": None, "net_following_advice": None}
    if later is not None:
        wait_net = later * quintals - a.holding_cost_pct_per_day * a.wait_days * sell_net
        replay.update({"change_pct": later / price - 1, "net_if_waited": wait_net,
                       "net_following_advice": wait_net if rec.action == "WAIT" else sell_net})
    return {
        "mandi": home["name"], "date": d, "quintals": quintals, "price_today_rs_qtl": price,
        "action": rec.action, "confidence": rec.confidence, "reasons": rec.reasons,
        "forecast": {"down": forecast.p_down, "flat": forecast.p_flat, "up": forecast.p_up},
        "options": [{"action": o.action, "mandi": o.mandi, "gross_rs": o.gross_rs, "cost_rs": o.cost_rs,
                     "net_rs": o.net_rs, "note": o.note} for o in rec.options],
        "typical_moves": moves, "assumptions": asdict(a), "replay": replay,
        "notes": ["Replay mode: the forecast came from a model trained only on earlier years.",
                  "Costs and typical moves are assumptions shown above; change them to see the effect."],
    }


# ---------------------------------------------------------------- price comparison table
def compare_table(home: dict, d: Date) -> dict:
    win = window_prices(d, 90)
    ts = pd.Timestamp(d)
    today = win[(win["mandi_id"] == home["id"]) & (win["date"] == ts)]
    if today.empty:
        raise NotFound(f"No price for {home['name']} on {d}.")
    home_price = float(today.iloc[0]["price"])

    before = win[win["date"] < ts]
    wide = before.pivot_table(index="date", columns="mandi_id", values="price")
    ratio = wide.div(wide[home["id"]], axis=0) if home["id"] in wide.columns else None

    a = Assumptions()
    hq = MandiQuote(home["name"], home_price, _num(home["lat"]), _num(home["lon"]))
    rows = []
    for _, m in _all_mandis_frame().iterrows():
        mid = int(m["id"])
        if mid == home["id"] or ratio is None or mid not in ratio.columns:
            continue
        r = ratio[mid].dropna()
        recent = win[(win["mandi_id"] == mid) & (win["date"] <= ts) & (win["date"] > ts - pd.Timedelta(days=3))]
        if len(r) < 30 or recent.empty:
            continue
        usual = float(r.median())
        avg3 = float(recent["price"].mean())
        km = road_distance_km(hq, MandiQuote(m["name"], avg3, _num(m["lat"]), _num(m["lon"])), a)
        rows.append({
            "mandi": m["name"], "price_3day_avg_rs_qtl": avg3, "usual_price_vs_home": usual,
            "vs_usual_pct": avg3 / (usual * home_price) - 1, "road_km": km,
            "freight_rs_per_qtl": None if km is None else a.freight_rs_per_qtl_km * km + a.handling_rs_per_qtl,
        })
    rows.sort(key=lambda x: -x["price_3day_avg_rs_qtl"])
    return {"home": home["name"], "date": d, "home_price_rs_qtl": home_price, "rows": rows,
            "notes": ["Reported prices only. Grades and varieties can differ between mandis.",
                      "'usual_price_vs_home' is the median ratio over the previous 90 days.",
                      "A positive 'vs_usual_pct' means the mandi pays more than its usual gap to your home mandi.",
                      "This is information, not move advice: our backtest could not show that moving pays."]}


# ---------------------------------------------------------------- FPO split what-if
def price_impact_info() -> dict:
    p = OUTPUTS / "price_impact.json"
    if p.exists():
        info = json.loads(p.read_text(encoding="utf-8"))
        used = bool(info.get("estimate_was_used"))
        return {"flexibility_used": float(info["flexibility_used"]), "estimate_was_used": used,
                "estimated_flexibility": info.get("estimated_flexibility"),
                "note": ("estimated from 2014-2021 data" if used else
                         "illustrative assumption, NOT measured: our data showed no detectable effect")}
    return {"flexibility_used": -0.25, "estimate_was_used": False, "estimated_flexibility": None,
            "note": "illustrative assumption, NOT measured"}


def split_view(home: dict, d: Date, tonnes: float, flexibility) -> dict:
    total_qtl = tonnes * 10
    win = window_prices(d, 90)
    ts = pd.Timestamp(d)
    today = win[(win["mandi_id"] == home["id"]) & (win["date"] == ts)]
    if today.empty:
        raise NotFound(f"No price for {home['name']} on {d}.")
    base_price = float(today.iloc[0]["price"])

    a = Assumptions()
    hq = MandiQuote(home["name"], base_price, _num(home["lat"]), _num(home["lon"]))
    prior = win[(win["date"] < ts) & (win["arrivals"] > 0)]
    outlets = []
    for _, m in _all_mandis_frame().iterrows():
        mid = int(m["id"])
        g = prior[prior["mandi_id"] == mid]
        if len(g) < 15:
            continue
        usual_qtl = float(g["arrivals"].median()) * 10          # tonnes -> quintals
        km = 0.0 if mid == home["id"] else road_distance_km(
            hq, MandiQuote(m["name"], base_price, _num(m["lat"]), _num(m["lon"])), a)
        if km is None:
            continue
        outlets.append(Outlet(m["name"], base_price, usual_qtl, km))
    if not any(o.road_km == 0 for o in outlets):
        raise NotFound("Not enough recent arrival data at the home mandi around that date.")

    info = price_impact_info()
    flex = info["flexibility_used"] if flexibility is None else flexibility
    note = info["note"] if flexibility is None else "chosen by the user"

    def run(f: float):
        return optimise(outlets, total_qtl, SplitAssumptions(flexibility=f), a)

    main = run(flex)
    sens = []
    for f in sorted(set(SENSITIVITY + (flex,)), reverse=True):
        r = main if f == flex else run(f)
        sens.append({"flexibility": f, "gain_rs": r.gain_rs, "gain_pct": r.gain_pct,
                     "best_plan": r.best_split.allocation})
    return {
        "home": home["name"], "date": d, "tonnes": tonnes, "base_price_rs_qtl": base_price,
        "flexibility_used": flex, "flexibility_note": note,
        "outlets": [{"mandi": o.name, "usual_daily_arrivals_qtl": o.daily_arrivals_qtl, "road_km": o.road_km}
                    for o in outlets],
        "best_single": {"allocation": main.best_single.allocation, "net_rs": main.best_single.net_rs,
                        "transport_rs": main.best_single.transport_rs},
        "best_plan": {"allocation": main.best_split.allocation, "net_rs": main.best_split.net_rs,
                      "transport_rs": main.best_split.transport_rs},
        "gain_rs": main.gain_rs, "gain_pct": main.gain_pct, "sensitivity": sens,
        "notes": ["What-if simulator. Every mandi is assumed to pay the home mandi's price that day.",
                  "The price effect of a big lot could not be measured in our data, so it is an assumption. "
                  "With no price effect (0.00) there is no gain from splitting.",
                  "Gains come only from volume effects, net of transport."],
    }


# ---------------------------------------------------------------- documents
def backtest_metrics() -> dict:
    p = OUTPUTS / "backtest_metrics.json"
    if not p.exists():
        raise NotFound("Backtest results are not available. Run scripts/train_forecast.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def assumptions_view() -> dict:
    return {"decision": asdict(Assumptions()), "split": asdict(SplitAssumptions()),
            "price_impact": price_impact_info(), "typical_moves_2014_2021": typical_moves()}
