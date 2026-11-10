"""Peek inside the onion dataset: columns, date range, states, missing values.

Run from the mandisense folder:
    python scripts/inspect_data.py
"""
import pandas as pd

PATH = "data/raw/preprocessed_onion_2014_2024_prices_weather_events.parquet"

df = pd.read_parquet(PATH)

print("SHAPE (rows, columns):", df.shape)
print("\nCOLUMNS AND TYPES:")
print(df.dtypes.to_string())

print("\nFIRST 3 ROWS:")
print(df.head(3).T.to_string())

# Try to find the date and place columns without assuming their exact names
date_cols = [c for c in df.columns if "date" in c.lower()]
for c in date_cols:
    print(f"\nDATE RANGE in '{c}':", df[c].min(), "->", df[c].max())

for c in df.columns:
    if c.lower() in ("state", "state_name", "market", "market_name", "variety"):
        print(f"\nDISTINCT VALUES in '{c}':", df[c].nunique())

print("\nMISSING VALUES (top 15 columns):")
print(df.isna().sum().sort_values(ascending=False).head(15).to_string())
