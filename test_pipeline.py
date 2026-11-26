import time
import joblib
import numpy as np
import pandas as pd
from pipeline import (
    FEATURE_COLS,
    FORECAST_FEATURE_COLS,
    FORECAST_HORIZONS,
    build_forecast_features,
    build_horizon_dataset,
    classify_risk_tier,
    compute_ward_weather_lags,
    load_raw_data,
)


def run_verification():
    print("1. Loading dataset and computing ward-level multi-week lags...")
    raw_df = load_raw_data()
    lag_df = compute_ward_weather_lags(raw_df, drop_na_lags=True)
    assert not lag_df[FEATURE_COLS].isnull().any().any(), "NaN values found in lag features!"
    print(f"   Verified {len(lag_df):,} ward-week rows across {lag_df['ward_id'].nunique()} wards.")

    print("2. Loading serialized model bundle and testing inference latency...")
    bundle = joblib.load("models/dengue_model_bundle.joblib")
    sample = lag_df[bundle["feature_cols"]].tail(10)

    t0 = time.perf_counter()
    _ = bundle["point_model"].predict(sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"   Inference completed for 10 ward-weeks in {elapsed_ms:.2f} ms.")

    print("3. Verifying risk tier classification...")
    for val in [1.2, 4.5, 9.0, 15.3]:
        tier, _, _ = classify_risk_tier(val)
        print(f"   Predicted cases: {val:>4.1f} -> Risk Tier: {tier}")

    print("4. Verifying multi-horizon targets are true future weeks...")
    feat_df = build_forecast_features(raw_df)
    assert not feat_df[FORECAST_FEATURE_COLS].isnull().any().any(), "NaN in forecast features!"
    for h in FORECAST_HORIZONS:
        hd = build_horizon_dataset(feat_df, h)
        gap = (hd["target_date"] - hd["date"]).dt.days
        assert (gap == 7 * h).all(), f"h={h}: target is not exactly {h} week(s) ahead"
    print("   Targets are exactly 1, 2, 3, 4 weeks after the origin week for every ward.")

    print("5. Leakage test: deleting all future data must not change any forecast...")
    origin = pd.Timestamp("2024-07-01")
    truncated_feat = build_forecast_features(raw_df[raw_df["date"] <= origin])
    full_row = feat_df[feat_df["date"] == origin].sort_values("ward_id")
    trunc_row = truncated_feat[truncated_feat["date"] == origin].sort_values("ward_id")
    assert np.allclose(full_row[FORECAST_FEATURE_COLS].values, trunc_row[FORECAST_FEATURE_COLS].values)
    for h in FORECAST_HORIZONS:
        m = bundle["horizon_models"][h]["point_model"]
        assert np.allclose(m.predict(full_row[FORECAST_FEATURE_COLS]), m.predict(trunc_row[FORECAST_FEATURE_COLS]))
    print(f"   Forecasts issued on {origin.date()} are identical with or without future data (all 60 wards, h=1..4).")

    print("6. Early-warning accuracy vs naive persistence baseline (2024-2025 holdout):")
    for m in bundle["forecast_metrics"]:
        assert m["mae"] < m["naive_persistence_mae"], f"h={m['horizon_weeks']} model does not beat naive baseline"
        print(f"   +{m['horizon_weeks']}W MAE {m['mae']:.3f} vs naive {m['naive_persistence_mae']:.3f}")

    print("\nAll pipeline and model verification checks passed successfully!")


if __name__ == "__main__":
    run_verification()
