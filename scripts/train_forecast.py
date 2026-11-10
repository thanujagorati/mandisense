"""Train the 7-day price-direction forecaster and test it with a walk-forward backtest.

Question the model answers (for each mandi, each day):
    "Will the modal price 7 days from now be DOWN, FLAT or UP versus today?"
    flat = within +/-3% of today's price.

Run from the mandisense folder:
    python scripts/train_forecast.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

HORIZON = 7          # days ahead
THRESH = 0.03        # +/-3% counts as "flat"
CLASSES = ["down", "flat", "up"]   # alphabetical, same order sklearn uses
FOLDS = [2022, 2023, 2024]         # each year is tested using only earlier years
MODEL_VERSION = "hgb-v1"
CONF_CUTOFF = 0.60

FEATURES = ["mandi_id", "price", "pct1", "pct7", "pct30", "ratio_ma7", "ratio_ma30",
            "vol7", "arrivals", "arr_ratio30", "arr_pct7", "month", "dow"]


def label_from_return(r):
    r = np.asarray(r, dtype=float)
    return np.where(r > THRESH, "up", np.where(r < -THRESH, "down", "flat"))


def load_prices(conn) -> pd.DataFrame:
    sql = """SELECT p.mandi_id, p.commodity_id, p.obs_date AS date,
                    p.modal_price_rs_qtl::float AS price,
                    p.arrivals_tonnes::float AS arrivals
             FROM price_observations p
             ORDER BY p.mandi_id, p.obs_date"""
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Features use ONLY information available on day t. The label looks 7 days ahead."""
    parts = []
    for mandi_id, g in prices.groupby("mandi_id"):
        g = g.set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        p = g["price"].reindex(full)
        a = g["arrivals"].reindex(full)
        p5 = p.ffill(limit=5)   # fill short gaps (market closed) for rolling features only
        a5 = a.ffill(limit=5)

        f = pd.DataFrame(index=full)
        f["mandi_id"] = mandi_id
        f["commodity_id"] = g["commodity_id"].iloc[0]
        f["price"] = p
        f["pct1"] = p5 / p5.shift(1) - 1
        f["pct7"] = p5 / p5.shift(7) - 1
        f["pct30"] = p5 / p5.shift(30) - 1
        f["ratio_ma7"] = p5 / p5.rolling(7, min_periods=3).mean()
        f["ratio_ma30"] = p5 / p5.rolling(30, min_periods=10).mean()
        f["vol7"] = p5.rolling(7, min_periods=3).std() / p5.rolling(7, min_periods=3).mean()
        f["arrivals"] = a5
        f["arr_ratio30"] = a5 / a5.rolling(30, min_periods=10).mean()
        f["arr_pct7"] = a5 / a5.shift(7) - 1
        f["month"] = f.index.month
        f["dow"] = f.index.dayofweek
        # Label: price 7 days later (allow it to be up to 2 days stale if that day was a closure)
        f["target_ret"] = p.ffill(limit=2).shift(-HORIZON) / p - 1
        parts.append(f)

    data = pd.concat(parts)
    data.index.name = "date"
    data = data.reset_index().replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["price", "target_ret"]).copy()   # keep days that have a price AND a known outcome
    data["label"] = label_from_return(data["target_ret"])
    return data


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)), 4),
    }


def evaluate(test: pd.DataFrame) -> dict:
    flat = np.array(["flat"] * len(test))
    trend = label_from_return(test["pct7"].fillna(0).values)   # "last week's move continues"
    conf_mask = test["conf"] >= CONF_CUTOFF
    out = {
        "rows": int(len(test)),
        "true_class_share": {c: round(float((test["label"] == c).mean()), 3) for c in CLASSES},
        "model": score(test["label"], test["pred"]),
        "baseline_always_flat": score(test["label"], flat),
        "baseline_trend_continues": score(test["label"], trend),
        "confident_predictions": {
            "cutoff": CONF_CUTOFF,
            "share_of_rows": round(float(conf_mask.mean()), 3),
            "accuracy": round(float(accuracy_score(test.loc[conf_mask, "label"], test.loc[conf_mask, "pred"])), 4)
            if conf_mask.any() else None,
        },
    }
    return out


def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL not found in .env")

    conn = psycopg2.connect(url)
    try:
        prices = load_prices(conn)
        data = build_dataset(prices)
        print(f"Price rows: {len(prices)}   Usable rows (price + known 7-day outcome): {len(data)}")
        print("Label share overall:", data["label"].value_counts(normalize=True).round(3).to_dict())

        feat_idx = [FEATURES.index("mandi_id")]
        tests = []
        results = {"horizon_days": HORIZON, "flat_threshold": THRESH, "model_version": MODEL_VERSION, "folds": {}}

        for year in FOLDS:
            test_start = pd.Timestamp(f"{year}-01-01")
            test_end = pd.Timestamp(f"{year}-12-31")
            # Training stops HORIZON days before the test year, so no training label peeks into it.
            train = data[data["date"] < test_start - pd.Timedelta(days=HORIZON)]
            test = data[(data["date"] >= test_start) & (data["date"] <= test_end)].copy()
            if train.empty or test.empty:
                continue

            model = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=200,
                categorical_features=feat_idx, early_stopping=False, random_state=42,
            )
            model.fit(train[FEATURES], train["label"])
            proba = model.predict_proba(test[FEATURES])
            classes = list(model.classes_)
            for i, c in enumerate(classes):
                test[f"p_{c}"] = proba[:, i]
            for c in CLASSES:                      # in case a class never appeared in training
                if f"p_{c}" not in test:
                    test[f"p_{c}"] = 0.0
            test["pred"] = np.array(classes)[proba.argmax(axis=1)]
            test["conf"] = proba.max(axis=1)

            results["folds"][str(year)] = evaluate(test)
            results["folds"][str(year)]["train_rows"] = int(len(train))
            tests.append(test)

        pooled = pd.concat(tests)
        results["all_folds_pooled"] = evaluate(pooled)

        print("\n===== WALK-FORWARD RESULTS (trained only on the past) =====")
        for k, v in {**results["folds"], "POOLED": results["all_folds_pooled"]}.items():
            print(f"\n{k}: {v['rows']} test rows, true class share {v['true_class_share']}")
            print("   model             ", v["model"])
            print("   always-flat       ", v["baseline_always_flat"])
            print("   trend-continues   ", v["baseline_trend_continues"])
            print("   confident (>=0.6) ", v["confident_predictions"])

        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/backtest_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nSaved outputs/backtest_metrics.json")

        # Store the honest out-of-sample forecasts for the API and the replay demo.
        rows = [
            (int(r.mandi_id), int(r.commodity_id), r.date.date(), HORIZON,
             round(float(r.p_down), 4), round(float(r.p_flat), 4), round(float(r.p_up), 4),
             r.pred, MODEL_VERSION)
            for r in pooled.itertuples(index=False)
        ]
        with conn, conn.cursor() as cur:
            cur.execute(Path("db/002_forecasts.sql").read_text(encoding="utf-8"))
            execute_values(
                cur,
                """INSERT INTO forecasts (mandi_id, commodity_id, forecast_date, horizon_days,
                                          p_down, p_flat, p_up, predicted_class, model_version)
                   VALUES %s
                   ON CONFLICT (mandi_id, commodity_id, forecast_date, horizon_days, model_version)
                   DO UPDATE SET p_down = EXCLUDED.p_down, p_flat = EXCLUDED.p_flat,
                                 p_up = EXCLUDED.p_up, predicted_class = EXCLUDED.predicted_class""",
                rows, page_size=2000,
            )
        print(f"Stored {len(rows)} forecasts in the database.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
