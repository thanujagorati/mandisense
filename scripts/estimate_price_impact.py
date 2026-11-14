"""Estimate "price flexibility": by what % does a mandi's price tend to move when its arrivals
are 1% above their usual level? (Negative = more supply, lower price.)

Method, in plain words:
  1. For each mandi and day, compare price and arrivals with their own 30-day averages
     (so slow trends and seasons drop out).
  2. Subtract the same-day average across mandis (so region-wide shocks drop out).
  3. Fit one line through what is left. Its slope is the estimate.

Measured on 2014-2021 only. This is an ASSOCIATION, not proof of cause: prices also influence
how much farmers bring. We use it as a labelled estimate and test how sensitive the answer is.

Run from the mandisense folder:
    python scripts/estimate_price_impact.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CUTOFF = pd.Timestamp("2022-01-01")
FALLBACK = -0.25     # illustrative value (NOT a measurement), used only if the estimate looks unreliable


def slope_r2(x, y):
    b = float((x * y).sum() / (x * x).sum())
    r = float(np.corrcoef(x, y)[0, 1])
    return b, r * r


def main() -> None:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""SELECT m.name, p.mandi_id, p.obs_date, p.modal_price_rs_qtl::float,
                          p.arrivals_tonnes::float
                   FROM price_observations p JOIN mandis m ON m.id = p.mandi_id""")
    df = pd.DataFrame(cur.fetchall(), columns=["name", "mandi_id", "date", "price", "arrivals"])
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < CUTOFF]

    parts = []
    for mid, g in df.groupby("mandi_id"):
        g = g.set_index("date").sort_index()
        g = g[g["arrivals"] > 0]
        ma_p = g["price"].rolling("30D", min_periods=10, closed="left").mean()
        ma_a = g["arrivals"].rolling("30D", min_periods=10, closed="left").mean()
        out = pd.DataFrame({"name": g["name"], "mandi_id": mid,
                            "dev_p": np.log(g["price"] / ma_p),
                            "dev_a": np.log(g["arrivals"] / ma_a)}).dropna()
        parts.append(out)
    d = pd.concat(parts).reset_index()

    # Keep days with at least 4 mandis reporting, then remove same-day region-wide movements.
    d = d[d.groupby("date")["mandi_id"].transform("count") >= 4].copy()
    d["dp"] = d["dev_p"] - d.groupby("date")["dev_p"].transform("mean")
    d["da"] = d["dev_a"] - d.groupby("date")["dev_a"].transform("mean")
    for c in ("dp", "da"):                      # tame extreme outliers
        lo, hi = d[c].quantile([0.01, 0.99])
        d[c] = d[c].clip(lo, hi)

    b, r2 = slope_r2(d["da"].values, d["dp"].values)
    print(f"Observations used: {len(d)}")
    print(f"\nOVERALL price flexibility estimate (2014-2021): {b:+.3f}   (R-squared {r2:.3f})")
    print("   meaning: when a mandi's arrivals are 10% above usual, its price is about "
          f"{b * 10:+.1f}% versus usual.")

    d["year"] = d["date"].dt.year
    yearly = {int(y): slope_r2(g["da"].values, g["dp"].values)[0] for y, g in d.groupby("year")}
    print("\nBY YEAR (is it stable?):")
    for y, v in yearly.items():
        print(f"   {y}: {v:+.3f}")
    print("\nBY MANDI:")
    for name, g in d.groupby("name"):
        print(f"   {name:<18} {slope_r2(g['da'].values, g['dp'].values)[0]:+.3f}   ({len(g)} days)")

    negative_years = sum(1 for v in yearly.values() if v < 0)
    reliable = (-0.8 <= b <= -0.05) and negative_years >= max(1, int(0.75 * len(yearly)))
    used = b if reliable else FALLBACK
    reason = ("estimate looks plausible and stable across years" if reliable else
              "estimate looks implausible or unstable, so an illustrative value is used instead (NOT a measurement)")
    print(f"\nVALUE USED BY THE SPLIT ENGINE: {used:+.3f}   ({reason})")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/price_impact.json").write_text(json.dumps({
        "estimated_flexibility": round(b, 4), "r_squared": round(r2, 4),
        "yearly": {str(k): round(v, 4) for k, v in yearly.items()},
        "flexibility_used": round(used, 4), "estimate_was_used": bool(reliable), "reason": reason,
        "measured_on": "2014-2021",
    }, indent=2), encoding="utf-8")
    print("Saved outputs/price_impact.json")


if __name__ == "__main__":
    main()
