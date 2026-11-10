"""Replay demo: ask for a recommendation on a PAST date, then reveal what really happened.

Examples (run from the mandisense folder):
    python scripts/demo_recommend.py --home Gudimalkapur --date 2024-03-15 --quintals 80
    python scripts/demo_recommend.py --home Bowenpally --date 2023-09-04 --quintals 50

Forecasts exist for days in 2022-2024 when the mandi reported a price. Each one came from a
model that was trained only on earlier years.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from app.engine import (DEFAULT_MEAN_RETURNS, Assumptions, Forecast, MandiQuote, recommend)

load_dotenv()

CUTOFF = pd.Timestamp("2022-01-01")   # typical moves are measured only on data before this date


def fetch_df(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def price_asof(prices: pd.DataFrame, mandi_id: int, date: pd.Timestamp, max_age_days: int):
    s = prices[(prices["mandi_id"] == mandi_id)
               & (prices["date"] <= date)
               & (prices["date"] >= date - pd.Timedelta(days=max_age_days))].sort_values("date")
    return None if s.empty else float(s.iloc[-1]["price"])


def typical_moves(prices: pd.DataFrame) -> dict:
    """Average 7-day price change when the OUTCOME was down / flat / up (down < -3%, up > +3%).

    Measured on 2014-2021 data only, so replaying 2022-2024 never uses information from its own period.
    """
    old = prices[prices["date"] < CUTOFF].copy()
    old["date"] = old["date"].astype("datetime64[ns]")      # same precision on both sides of the join
    old["target"] = old["date"] + pd.Timedelta(days=7)
    old = old.sort_values("target")

    later = prices[prices["date"] < CUTOFF].rename(columns={"date": "target_date", "price": "price7"}).copy()
    later["target_date"] = later["target_date"].astype("datetime64[ns]")
    later = later.sort_values("target_date")

    m = pd.merge_asof(old, later, left_on="target", right_on="target_date", by="mandi_id",
                      tolerance=pd.Timedelta(days=2), direction="backward")
    m["ret"] = m["price7"] / m["price"] - 1
    m = m.dropna(subset=["ret"])
    label = np.where(m["ret"] > 0.03, "up", np.where(m["ret"] < -0.03, "down", "flat"))
    out = m.groupby(label)["ret"].mean().to_dict()
    return {c: float(out.get(c, DEFAULT_MEAN_RETURNS[c])) for c in ("down", "flat", "up")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="mandi name, e.g. Gudimalkapur")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD, a day in 2022-2024")
    ap.add_argument("--quintals", type=float, default=80, help="quantity in quintals (10 quintals = 1 tonne)")
    args = ap.parse_args()
    date = pd.Timestamp(args.date)

    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not found in .env")
    conn = psycopg2.connect(url)
    try:
        mandis = fetch_df(conn, "SELECT id, name, lat, lon FROM mandis")
        prices = fetch_df(conn, "SELECT mandi_id, obs_date AS date, modal_price_rs_qtl::float AS price "
                                "FROM price_observations")
        prices["date"] = pd.to_datetime(prices["date"])
        fc = fetch_df(conn, "SELECT mandi_id, forecast_date AS date, p_down::float AS p_down, "
                            "p_flat::float AS p_flat, p_up::float AS p_up, predicted_class "
                            "FROM forecasts WHERE horizon_days = 7")
        fc["date"] = pd.to_datetime(fc["date"])
    finally:
        conn.close()

    home_row = mandis[mandis["name"].str.lower() == args.home.lower()]
    if home_row.empty:
        raise SystemExit(f"Unknown mandi '{args.home}'. Choose from: {', '.join(mandis['name'])}")
    home_id = int(home_row.iloc[0]["id"])

    f_row = fc[(fc["mandi_id"] == home_id) & (fc["date"] == date)]
    if f_row.empty:
        raise SystemExit(f"No forecast for {args.home} on {args.date}. Forecasts exist only for days in "
                         "2022-2024 when that mandi reported a price. Try another date.")
    f_row = f_row.iloc[0]
    forecast = Forecast(float(f_row["p_down"]), float(f_row["p_flat"]), float(f_row["p_up"]))

    def quote(row, price):
        lat = None if pd.isna(row["lat"]) else float(row["lat"])
        lon = None if pd.isna(row["lon"]) else float(row["lon"])
        return MandiQuote(row["name"], price, lat, lon)

    home_price = price_asof(prices, home_id, date, 0)
    home = quote(home_row.iloc[0], home_price)
    others = []
    for _, r in mandis.iterrows():
        if int(r["id"]) == home_id:
            continue
        p = price_asof(prices, int(r["id"]), date, 3)   # latest price within 3 days
        if p is not None:
            others.append(quote(r, p))

    mean_returns = typical_moves(prices)
    a = Assumptions()
    rec = recommend(home, others, forecast, args.quintals, mean_returns, a)

    print(f"\n=== MandiSense replay: {args.quintals:g} quintals ({args.quintals / 10:g} tonnes) "
          f"at {home.name} on {args.date} ===")
    print("Typical 7-day change by outcome, measured on 2014-2021 data: "
          + ", ".join(f"{k} {v:+.1%}" for k, v in mean_returns.items()))
    print("\nOptions (expected net money in hand):")
    for o in sorted(rec.options, key=lambda x: -x.net_rs):
        print(f"  {o.action:<9} {o.mandi:<18} net Rs {o.net_rs:>10,.0f}   (costs Rs {o.cost_rs:>8,.0f})  {o.note}")
    print(f"\nRECOMMENDATION: {rec.action} -> {rec.mandi}   (confidence: {rec.confidence})")
    for line in rec.reasons:
        print("  -", line)
    print("\nAssumptions you can change: freight Rs "
          f"{a.freight_rs_per_qtl_km}/quintal/km, road factor {a.road_factor}, handling Rs "
          f"{a.handling_rs_per_qtl}/quintal, holding cost {a.holding_cost_pct_per_day:.1%}/day, "
          f"minimum trip {a.min_distance_km:g} km.")

    # Reveal what really happened
    later = price_asof(prices, home_id, date + pd.Timedelta(days=7), 2)
    sell_net = home.price_rs_qtl * args.quintals
    print("\n--- WHAT ACTUALLY HAPPENED ---")
    if later is None:
        print("No price recorded around 7 days later at the home mandi.")
    else:
        wait_net = later * args.quintals - a.holding_cost_pct_per_day * a.wait_days * sell_net
        print(f"Price at {home.name} 7 days later: Rs {later:,.0f} per quintal "
              f"({later / home.price_rs_qtl - 1:+.1%} vs that day).")
        print(f"Selling that day:   Rs {sell_net:,.0f}")
        print(f"Waiting 7 days:     Rs {wait_net:,.0f}  ({wait_net - sell_net:+,.0f} vs selling that day)")
        chosen = next(o for o in rec.options if o.action == rec.action and o.mandi == rec.mandi)
        realised = {"SELL_NOW": sell_net, "WAIT": wait_net}.get(rec.action, chosen.net_rs)
        print(f"Following the advice ({rec.action}): Rs {realised:,.0f}  ({realised - sell_net:+,.0f} vs selling that day)")


if __name__ == "__main__":
    main()
