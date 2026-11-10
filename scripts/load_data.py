"""Load the chosen mandi cluster from the onion parquet file into Postgres (Neon).

Run from the mandisense folder:
    python scripts/load_data.py

Safe to re-run: it upserts, so it never creates duplicate rows.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

PARQUET = "data/raw/preprocessed_onion_2014_2024_prices_weather_events.parquet"
SCHEMA = Path("db/schema.sql")

# (state, market) exactly as they appear in the file.
CLUSTER = [
    ("Telangana", "Gudimalkapur"),
    ("Telangana", "Bowenpally"),
    ("Telangana", "Sadasivpet"),
    ("Maharashtra", "Solapur"),
    ("Karnataka", "Hubli (Amaragol)"),
    ("Maharashtra", "Kolhapur"),
    ("Maharashtra", "Pune"),
    ("Maharashtra", "Nasik"),
]

COLS = ["date", "state", "district", "market", "arrivals_tonnes",
        "min_price_rs_qtl", "max_price_rs_qtl", "modal_price_rs_qtl", "lat", "lon"]


def build_market_days() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, columns=COLS)
    wanted = {f"{s}|{m}" for s, m in CLUSTER}
    df = df[(df["state"] + "|" + df["market"]).isin(wanted)]
    df = df[df["modal_price_rs_qtl"] > 0].copy()
    print(f"Rows for our cluster (before collapsing): {len(df)}")

    # A market reports several rows per day (varieties, grades). Collapse to ONE price per
    # market per day: modal price weighted by arrivals; if arrivals are all zero, a plain mean.
    df["arrivals_tonnes"] = df["arrivals_tonnes"].fillna(0)
    df["wp"] = df["modal_price_rs_qtl"] * df["arrivals_tonnes"]
    g = df.groupby(["state", "district", "market", "date"], as_index=False).agg(
        sum_wp=("wp", "sum"),
        sum_w=("arrivals_tonnes", "sum"),
        mean_modal=("modal_price_rs_qtl", "mean"),
        min_price=("min_price_rs_qtl", "min"),
        max_price=("max_price_rs_qtl", "max"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )
    g["modal"] = np.where(g["sum_w"] > 0, g["sum_wp"] / g["sum_w"].where(g["sum_w"] > 0, 1), g["mean_modal"])
    g["arrivals"] = g["sum_w"]
    g["obs_date"] = g["date"].dt.date
    print(f"Market-days after collapsing: {len(g)}")
    return g


def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not found in .env")

    g = build_market_days()

    conn = psycopg2.connect(url)
    try:
        with conn, conn.cursor() as cur:  # 'with conn' = one transaction: all or nothing
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
            cur.execute("SELECT id FROM commodities WHERE name = 'Onion'")
            commodity_id = cur.fetchone()[0]

            # 1) mandis
            mandi_ids = {}
            for (state, district, market), grp in g.groupby(["state", "district", "market"]):
                lat = grp["lat"].mean()
                lon = grp["lon"].mean()
                cur.execute(
                    """INSERT INTO mandis (state, district, name, lat, lon)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (state, district, name)
                       DO UPDATE SET lat = EXCLUDED.lat, lon = EXCLUDED.lon
                       RETURNING id""",
                    (state, district, market,
                     None if pd.isna(lat) else float(lat),
                     None if pd.isna(lon) else float(lon)),
                )
                mandi_ids[(state, district, market)] = cur.fetchone()[0]

            # 2) price observations (bulk upsert)
            rows = []
            for r in g.itertuples(index=False):
                rows.append((
                    mandi_ids[(r.state, r.district, r.market)], commodity_id, r.obs_date,
                    round(float(r.modal), 2),
                    None if pd.isna(r.min_price) else float(r.min_price),
                    None if pd.isna(r.max_price) else float(r.max_price),
                    round(float(r.arrivals), 2),
                ))
            execute_values(
                cur,
                """INSERT INTO price_observations
                   (mandi_id, commodity_id, obs_date, modal_price_rs_qtl,
                    min_price_rs_qtl, max_price_rs_qtl, arrivals_tonnes)
                   VALUES %s
                   ON CONFLICT (mandi_id, commodity_id, obs_date) DO UPDATE SET
                     modal_price_rs_qtl = EXCLUDED.modal_price_rs_qtl,
                     min_price_rs_qtl   = EXCLUDED.min_price_rs_qtl,
                     max_price_rs_qtl   = EXCLUDED.max_price_rs_qtl,
                     arrivals_tonnes    = EXCLUDED.arrivals_tonnes""",
                rows,
                page_size=2000,
            )

            # 3) summary, straight from the database
            cur.execute(
                """SELECT m.name, COUNT(*), MIN(p.obs_date), MAX(p.obs_date)
                   FROM price_observations p JOIN mandis m ON m.id = p.mandi_id
                   GROUP BY m.name ORDER BY COUNT(*) DESC"""
            )
            print("\nIN THE DATABASE NOW (mandi, days, first date, last date):")
            for row in cur.fetchall():
                print("  ", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
