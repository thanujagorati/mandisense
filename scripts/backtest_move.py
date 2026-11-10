"""Honest test of the MOVE rule: when the engine says MOVE, does the gain really show up?

For every day in 2022-2024 and every home mandi, we ask the engine for advice with a flat forecast
(so WAIT never wins and only MOVE is tested), using only information available that day. When it
says MOVE, we look at what the destination paid the NEXT day, remove its usual gap, subtract
transport, and compare with simply selling at home that day.

Assumption behind the check: onions of home quality fetch the destination's price divided by its
usual ratio. This tests whether UNUSUAL price gaps persist for a day. It cannot tell us anything
about grade differences, which this data does not contain.

Run from the mandisense folder:
    python scripts/backtest_move.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from app.engine import Assumptions, Forecast, MandiQuote, recommend

load_dotenv()

CUTOFF = pd.Timestamp("2022-01-01")
QUINTALS = 80
FLAT = Forecast(0.0, 1.0, 0.0)   # takes WAIT out of the picture


def main() -> None:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT id, name, lat, lon FROM mandis")
    mandis = pd.DataFrame(cur.fetchall(), columns=["id", "name", "lat", "lon"]).set_index("id")
    cur.execute("SELECT mandi_id, obs_date, modal_price_rs_qtl::float FROM price_observations")
    prices = pd.DataFrame(cur.fetchall(), columns=["mandi_id", "date", "price"])
    conn.close()
    prices["date"] = pd.to_datetime(prices["date"])

    wide = prices.pivot_table(index="date", columns="mandi_id", values="price")
    daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    avg3 = daily.rolling(3, min_periods=1).mean()           # 3-day average, today included
    arrival = daily.ffill(limit=3)                          # latest price within 3 days
    a = Assumptions()

    records = []
    test_days = wide.loc["2022-01-01":"2024-12-30"].index
    for home_id in mandis.index:
        # Usual gap = median same-day ratio over the 90 days BEFORE each day (known at decision time).
        ratio_daily = daily.div(daily[home_id], axis=0)
        usual = ratio_daily.rolling(90, min_periods=30).median().shift(1)
        for day in test_days:
            hp = wide.at[day, home_id]
            if pd.isna(hp):
                continue
            ratios = usual.loc[day]
            h = mandis.loc[home_id]
            home_q = MandiQuote(h["name"], float(hp),
                                None if pd.isna(h["lat"]) else float(h["lat"]),
                                None if pd.isna(h["lon"]) else float(h["lon"]))
            others = []
            for mid in mandis.index:
                if mid == home_id or pd.isna(ratios.get(mid)) or pd.isna(avg3.at[day, mid]):
                    continue
                m = mandis.loc[mid]
                others.append(MandiQuote(m["name"], float(avg3.at[day, mid]),
                                         None if pd.isna(m["lat"]) else float(m["lat"]),
                                         None if pd.isna(m["lon"]) else float(m["lon"]),
                                         float(ratios[mid])))
            rec = recommend(home_q, others, FLAT, QUINTALS, assumptions=a)
            records.append({"home": h["name"], "date": day, "action": rec.action})
            if rec.action != "MOVE":
                continue
            dest_id = mandis.index[mandis["name"] == rec.mandi][0]
            chosen = next(o for o in rec.options if o.action == "MOVE" and o.mandi == rec.mandi)
            arr = arrival.at[day + pd.Timedelta(days=1), dest_id]
            if pd.isna(arr):
                records[-1]["gain_pct"] = np.nan
                continue
            realised = arr / ratios[dest_id] * QUINTALS - chosen.cost_rs
            sell = hp * QUINTALS
            # Fairer baseline: sell at HOME the next day (removes the bounce-back of a low quote today).
            home_next = arrival.at[day + pd.Timedelta(days=1), home_id]
            excess = np.nan if pd.isna(home_next) else (realised - home_next * QUINTALS) / sell
            records[-1].update({"dest": rec.mandi, "predicted_pct": (chosen.net_rs - sell) / sell,
                                "gain_pct": (realised - sell) / sell, "excess_pct": excess})

    df = pd.DataFrame(records)
    moves = df[df["action"] == "MOVE"].dropna(subset=["gain_pct"])
    print(f"Decision days tested: {len(df)}   MOVE advised: {int((df['action'] == 'MOVE').sum())} "
          f"({(df['action'] == 'MOVE').mean():.1%} of days)")
    if moves.empty:
        print("The MOVE rule never fired, so there is nothing to test.")
        return
    print(f"\nWhen MOVE was advised ({len(moves)} cases with a next-day price):")
    print(f"  predicted gain, average:      {moves['predicted_pct'].mean():+.1%}")
    print(f"  realised gain, average:       {moves['gain_pct'].mean():+.1%}")
    print(f"  realised gain, median:        {moves['gain_pct'].median():+.1%}")
    print(f"  share with a positive gain:   {(moves['gain_pct'] > 0).mean():.0%}")
    print(f"  share that kept >= +3% gain:  {(moves['gain_pct'] >= 0.03).mean():.0%}")
    ex = moves["excess_pct"].dropna()
    print(f"\nFAIRER TEST: same cases, but compared with selling at HOME the next day ({len(ex)} cases):")
    print(f"  average gain from moving:     {ex.mean():+.1%}")
    print(f"  median gain from moving:      {ex.median():+.1%}")
    print(f"  share with a positive gain:   {(ex > 0).mean():.0%}")
    print(f"  share that kept >= +3% gain:  {(ex >= 0.03).mean():.0%}")
    print("\nMOVE advice by home mandi (count, average realised gain):")
    print(moves.groupby("home")["gain_pct"].agg(["count", "mean"]).round(3).to_string())
    print("\nMOVE advice by destination (count, average realised gain):")
    print(moves.groupby("dest")["gain_pct"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
