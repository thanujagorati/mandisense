"""Do mandis have persistent price gaps versus each other? (diagnostic, prints tables)

Run from the mandisense folder:
    python scripts/check_price_gaps.py
"""
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute("""SELECT m.name, p.obs_date, p.modal_price_rs_qtl::float
               FROM price_observations p JOIN mandis m ON m.id = p.mandi_id""")
df = pd.DataFrame(cur.fetchall(), columns=["mandi", "date", "price"])
conn.close()

df["date"] = pd.to_datetime(df["date"])
wide = df.pivot_table(index="date", columns="mandi", values="price")

periods = {"2014-2021": ("2014-01-01", "2021-12-31"), "2022-2024": ("2022-01-01", "2024-12-31")}

for base in ["Gudimalkapur", "Bowenpally"]:
    print(f"\n##### PRICE OF EACH MANDI AS A MULTIPLE OF {base} (same day; 1.00 = same price) #####")
    for label, (start, end) in periods.items():
        w = wide.loc[start:end]
        ratio = w.div(w[base], axis=0).drop(columns=[base])
        out = pd.DataFrame({
            "median_ratio": ratio.median(),
            "p25": ratio.quantile(0.25),
            "p75": ratio.quantile(0.75),
            "days_both": ratio.count(),
        }).round(2)
        print(f"\n  {label}:")
        print(out.to_string())
