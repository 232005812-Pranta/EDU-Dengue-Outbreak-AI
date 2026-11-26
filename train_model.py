import json
import os
import time
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from pipeline import (
    FEATURE_COLS,
    FORECAST_FEATURE_COLS,
    FORECAST_HORIZONS,
    build_forecast_features,
    build_horizon_dataset,
    compute_ward_weather_lags,
    load_raw_data,
)


def _hgbr(loss: str, quantile=None, max_iter=220, max_depth=6):
    kwargs = dict(loss=loss, max_iter=max_iter, learning_rate=0.05, max_depth=max_depth, random_state=42)
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def train_forecast_models(raw_df: pd.DataFrame, split_date: pd.Timestamp):
    """
    Trains one leakage-free DIRECT model triple (point, q10, q90) per horizon h.
    Inputs = only data observed up to the origin week t; target = cases at t+h.

    Split rule (no temporal leakage):
      train: target_date <  split_date  (so neither inputs nor labels touch the test era)
      test : origin date >= split_date  (forecasts issued inside the unseen 2024-2025 era)
    """
    feat_df = build_forecast_features(raw_df)
    horizon_models, horizon_metrics, eval_frames = {}, [], []

    for h in FORECAST_HORIZONS:
        hd = build_horizon_dataset(feat_df, h)
        train_df = hd[hd["target_date"] < split_date]
        test_df = hd[hd["date"] >= split_date]
        X_tr, y_tr = train_df[FORECAST_FEATURE_COLS], train_df["target_cases"]
        X_te, y_te = test_df[FORECAST_FEATURE_COLS], test_df["target_cases"]

        point = _hgbr("squared_error").fit(X_tr, y_tr)
        lower = _hgbr("quantile", 0.10, 160, 5).fit(X_tr, y_tr)
        upper = _hgbr("quantile", 0.90, 160, 5).fit(X_tr, y_tr)

        p = np.clip(point.predict(X_te), 0.0, None)
        lo = np.minimum(p, np.clip(lower.predict(X_te), 0.0, None))
        hi = np.maximum(p, np.clip(upper.predict(X_te), 0.0, None))

        # Naive persistence baseline: "next weeks look like this week"
        naive = test_df["cases_lag_0w"].to_numpy()

        m = {
            "horizon_weeks": h,
            "mae": round(float(mean_absolute_error(y_te, p)), 4),
            "rmse": round(float(root_mean_squared_error(y_te, p)), 4),
            "r2": round(float(r2_score(y_te, p)), 4),
            "interval_coverage_pct": round(float(np.mean((y_te >= lo) & (y_te <= hi))) * 100.0, 2),
            "naive_persistence_mae": round(float(mean_absolute_error(y_te, naive)), 4),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }
        m["mae_improvement_vs_naive_pct"] = round(
            (1.0 - m["mae"] / max(m["naive_persistence_mae"], 1e-9)) * 100.0, 1
        )
        horizon_metrics.append(m)
        horizon_models[h] = {"point_model": point, "lower_model": lower, "upper_model": upper}

        ev = test_df[["date", "target_date", "city", "ward_id", "target_cases"]].copy()
        ev = ev.rename(columns={"date": "origin_date", "target_cases": "dengue_cases"})
        ev["horizon"] = h
        ev["predicted_cases"] = np.round(p, 2)
        ev["ci_lower"] = np.round(lo, 2)
        ev["ci_upper"] = np.round(hi, 2)
        ev["naive_persistence"] = naive
        eval_frames.append(ev)

    return horizon_models, horizon_metrics, pd.concat(eval_frames, ignore_index=True)



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

    # Final Round: leakage-free direct multi-horizon early-warning models
    horizon_models, horizon_metrics, forecast_eval = train_forecast_models(raw_df, split_date)
    bundle["horizon_models"] = horizon_models
    bundle["forecast_feature_cols"] = FORECAST_FEATURE_COLS
    bundle["forecast_horizons"] = FORECAST_HORIZONS
    bundle["forecast_metrics"] = horizon_metrics
    bundle["sklearn_version"] = sklearn.__version__
    forecast_eval.to_csv(os.path.join(output_dir, "forecast_validation_results.csv"), index=False)
    with open(os.path.join(output_dir, "forecast_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(horizon_metrics, f, indent=2)

    bundle_path = os.path.join(output_dir, "dengue_model_bundle.joblib")
    joblib.dump(bundle, bundle_path, compress=3)
    test_eval.to_csv(os.path.join(output_dir, "chronological_validation_results.csv"), index=False)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    elapsed = time.time() - t0
    print(f"Model bundle saved to {bundle_path} in {elapsed:.2f}s")
    print(f"[Nowcast, same-week weather] Chronological Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | 80% CI Coverage: {coverage*100:.1f}%")
    print("\nLeakage-free early-warning models (inputs known at origin week only):")
    for m in horizon_metrics:
        print(
            f"  h={m['horizon_weeks']}w | MAE {m['mae']:.4f} (naive {m['naive_persistence_mae']:.4f}, "
            f"{m['mae_improvement_vs_naive_pct']:+.1f}%) | RMSE {m['rmse']:.4f} | R2 {m['r2']:.4f} | "
            f"80% CI coverage {m['interval_coverage_pct']:.1f}%"
        )
    return bundle["metrics"]


if __name__ == "__main__":
    train_and_save()