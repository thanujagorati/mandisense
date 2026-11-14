"""FPO split demo: should a farmer cooperative share a big lot between mandis?

Examples (run from the mandisense folder):
    python scripts/demo_split.py --home Sadasivpet --date 2024-03-15 --tonnes 8
    python scripts/demo_split.py --home Gudimalkapur --date 2023-09-04 --tonnes 30

Rules of this demo, stated plainly:
  * every mandi is assumed to pay the SAME base price (the home mandi's price that day), because
    our data cannot tell real price gaps from grade differences. Gains come only from volume effects.
  * "usual arrivals" = median daily arrivals over the 90 days before the date.
  * price flexibility comes from outputs/price_impact.json (run estimate_price_impact.py first).
    The result is shown for several flexibility values because that number is an estimate.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from app.engine import Assumptions, MandiQuote, road_distance_km
from app.split import Outlet, SplitAssumptions, optimise

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tonnes", type=float, default=8)
    args = ap.parse_args()
    date = pd.Timestamp(args.date)
    total_qtl = args.tonnes * 10

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT id, name, lat, lon FROM mandis")
    mandis = pd.DataFrame(cur.fetchall(), columns=["id", "name", "lat", "lon"])
    cur.execute("SELECT mandi_id, obs_date, modal_price_rs_qtl::float, arrivals_tonnes::float "
                "FROM price_observations")
    prices = pd.DataFrame(cur.fetchall(), columns=["mandi_id", "date", "price", "arrivals"])
    conn.close()
    prices["date"] = pd.to_datetime(prices["date"])

    home_row = mandis[mandis["name"].str.lower() == args.home.lower()]
    if home_row.empty:
        raise SystemExit(f"Unknown mandi. Choose from: {', '.join(mandis['name'])}")
    home_id = int(home_row.iloc[0]["id"])
    today = prices[(prices["mandi_id"] == home_id) & (prices["date"] == date)]
    if today.empty:
        raise SystemExit(f"No price for {args.home} on {args.date}. Try another date.")
    base_price = float(today.iloc[0]["price"])

    def quote(row):
        lat = None if pd.isna(row["lat"]) else float(row["lat"])
        lon = None if pd.isna(row["lon"]) else float(row["lon"])
        return MandiQuote(row["name"], base_price, lat, lon)

    a = Assumptions()
    home_q = quote(home_row.iloc[0])
    window = prices[(prices["date"] < date) & (prices["date"] >= date - pd.Timedelta(days=90))
                    & (prices["arrivals"] > 0)]
    outlets = []
    for _, r in mandis.iterrows():
        g = window[window["mandi_id"] == int(r["id"])]
        if len(g) < 15:
            continue
        usual_qtl = float(g["arrivals"].median()) * 10          # tonnes -> quintals
        if int(r["id"]) == home_id:
            km = 0.0
        else:
            km = road_distance_km(home_q, quote(r), a)
            if km is None:
                continue
        outlets.append(Outlet(r["name"], base_price, usual_qtl, km))

    if not any(o.road_km == 0 for o in outlets):
        raise SystemExit("Not enough recent arrival data at the home mandi around that date.")

    p = Path("outputs/price_impact.json")
    if p.exists():
        info = json.loads(p.read_text(encoding="utf-8"))
        flex = float(info["flexibility_used"])
        flex_note = ("estimated from 2014-2021 data" if info.get("estimate_was_used")
                     else "illustrative assumption, NOT measured: our data showed no detectable effect")
    else:
        flex, flex_note = -0.25, "illustrative assumption, NOT measured (run estimate_price_impact.py first)"

    print(f"\n=== FPO split: {args.tonnes:g} tonnes ({total_qtl:g} quintals) from {args.home} on {args.date} ===")
    print(f"Base price assumed at every mandi: Rs {base_price:,.0f} per quintal (the home price that day)")
    print(f"Price flexibility used: {flex:+.2f}  ({flex_note})")
    print("\nMandis and their usual daily arrivals (quintals) and road distance:")
    for o in sorted(outlets, key=lambda x: x.road_km):
        print(f"  {o.name:<18} {o.daily_arrivals_qtl:>8,.0f} q/day   {o.road_km:>6.0f} km")

    def show(f: float, label: str) -> None:
        res = optimise(outlets, total_qtl, SplitAssumptions(flexibility=f), a)
        single = ", ".join(f"{k} {v:g}q" for k, v in res.best_single.allocation.items())
        split = ", ".join(f"{k} {v:g}q" for k, v in res.best_split.allocation.items())
        print(f"\n[{label}: flexibility {f:+.2f}]")
        print(f"  Best single mandi: {single}   net Rs {res.best_single.net_rs:,.0f}")
        print(f"  Best plan:         {split}   net Rs {res.best_split.net_rs:,.0f}"
              f"   (transport Rs {res.best_split.transport_rs:,.0f})")
        print(f"  Gain from splitting: Rs {res.gain_rs:,.0f} ({res.gain_pct:+.1%})")

    show(flex, "MAIN CASE")
    print("\nHow much does the answer depend on that uncertain number?")
    print("(flexibility 0.00 means 'no price effect at all', which is what our own data suggested)")
    for f in (0.0, -0.10, -0.25, -0.40):
        show(f, "sensitivity")

    print("\nAssumptions you can change: freight Rs "
          f"{a.freight_rs_per_qtl_km}/quintal/km, road factor {a.road_factor}, handling Rs "
          f"{a.handling_rs_per_qtl}/quintal, smallest load per mandi 30 quintals, at most 3 mandis.")


if __name__ == "__main__":
    main()
