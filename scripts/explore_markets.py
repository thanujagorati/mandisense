"""Find which states and markets have the most complete onion data.

Run from the mandisense folder:
    python scripts/explore_markets.py
"""
import pandas as pd

PATH = "data/raw/preprocessed_onion_2014_2024_prices_weather_events.parquet"
COLS = ["date", "state", "district", "market", "arrivals_tonnes",
        "modal_price_rs_qtl", "lat", "lon"]

df = pd.read_parquet(PATH, columns=COLS)
print("TOTAL ROWS:", len(df))

# 1) Rows per state
print("\nROWS PER STATE (top 12):")
print(df["state"].value_counts().head(12).to_string())

# 2) Collapse to one row per market per day (many varieties/grades per day otherwise)
md = (
    df.dropna(subset=["modal_price_rs_qtl"])
      .groupby(["state", "district", "market", "date"], as_index=False)
      .agg(modal=("modal_price_rs_qtl", "mean"),
           arrivals=("arrivals_tonnes", "sum"),
           lat=("lat", "first"), lon=("lon", "first"))
)
print("\nMARKET-DAY ROWS AFTER COLLAPSING:", len(md))

# 3) Coverage per market: how many days have a price, and how recent
recent = md[md["date"] >= "2022-01-01"]
total_days_recent = (pd.Timestamp("2024-12-31") - pd.Timestamp("2022-01-01")).days + 1
cov = (
    recent.groupby(["state", "district", "market"])
          .agg(days=("date", "nunique"),
               median_arrivals_t=("arrivals", "median"),
               last_date=("date", "max"))
          .reset_index()
)
cov["coverage_pct"] = (cov["days"] / total_days_recent * 100).round(1)

focus_states = ["Telangana", "Andhra Pradesh", "Karnataka", "Maharashtra"]
print("\nBEST-COVERED MARKETS 2022-2024 (focus states, top 25):")
print(cov[cov["state"].isin(focus_states)]
      .sort_values("days", ascending=False).head(25).to_string(index=False))

print("\nTELANGANA MARKETS (2022-2024):")
tg = cov[cov["state"] == "Telangana"].sort_values("days", ascending=False)
print(tg.to_string(index=False) if len(tg) else "No Telangana rows found. Check the state names above.")
