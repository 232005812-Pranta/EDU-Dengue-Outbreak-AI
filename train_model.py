import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from pipeline import FEATURE_COLS, compute_ward_weather_lags, load_raw_data


def train_and_save(split_date_str: str = "2024-01-01", output_dir: str = "models") -> dict:
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_raw_data()
    processed_df = compute_ward_weather_lags(raw_df, drop_na_lags=True)

    split_date = pd.Timestamp(split_date_str)
    train_df = processed_df[processed_df["date"] < split_date].copy()
    test_df = processed_df[processed_df["date"] >= split_date].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df["dengue_cases"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["dengue_cases"]

    # Point forecast model
    point_model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=220,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    point_model.fit(X_train, y_train)

    # 80% Prediction Interval models (10th and 90th percentiles)
    lower_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=160,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    lower_model.fit(X_train, y_train)

    upper_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=160,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    upper_model.fit(X_train, y_train)

    # Chronological validation evaluation
    pred_point = np.clip(point_model.predict(X_test), 0.0, None)
    pred_low = np.minimum(pred_point, np.clip(lower_model.predict(X_test), 0.0, None))
    pred_high = np.maximum(pred_point, np.clip(upper_model.predict(X_test), 0.0, None))

    mae = float(mean_absolute_error(y_test, pred_point))
    rmse = float(root_mean_squared_error(y_test, pred_point))
    r2 = float(r2_score(y_test, pred_point))
    coverage = float(np.mean((y_test >= pred_low) & (y_test <= pred_high)))

    test_eval = test_df[["date", "city", "ward_id", "dengue_cases"]].copy()
    test_eval["predicted_cases"] = np.round(pred_point, 2)
    test_eval["ci_lower"] = np.round(pred_low, 2)
    test_eval["ci_upper"] = np.round(pred_high, 2)
    test_eval["abs_error"] = np.abs(test_eval["dengue_cases"] - test_eval["predicted_cases"])

    city_metrics = []
    for city_name, grp in test_eval.groupby("city"):
        city_metrics.append(
            {
                "city": city_name,
                "mae": round(float(mean_absolute_error(grp["dengue_cases"], grp["predicted_cases"])), 3),
                "rmse": round(float(root_mean_squared_error(grp["dengue_cases"], grp["predicted_cases"])), 3),
                "r2": round(float(r2_score(grp["dengue_cases"], grp["predicted_cases"])), 3),
                "records": int(len(grp)),
            }
        )

    bundle = {
        "point_model": point_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "feature_cols": FEATURE_COLS,
        "split_date": split_date_str,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "interval_coverage_pct": round(coverage * 100.0, 2),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_period": f"{train_df['date'].min().date()} to {train_df['date'].max().date()}",
            "test_period": f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
            "city_metrics": city_metrics,
        },
    }

    bundle_path = os.path.join(output_dir, "dengue_model_bundle.joblib")
    joblib.dump(bundle, bundle_path, compress=3)
    test_eval.to_csv(os.path.join(output_dir, "chronological_validation_results.csv"), index=False)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    elapsed = time.time() - t0
    print(f"Model bundle saved to {bundle_path} in {elapsed:.2f}s")
    print(f"Chronological Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | 80% CI Coverage: {coverage*100:.1f}%")
    return bundle["metrics"]


if __name__ == "__main__":
    train_and_save()