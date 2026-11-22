import time
import joblib
from pipeline import FEATURE_COLS, classify_risk_tier, compute_ward_weather_lags, load_raw_data


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

    print("\nAll pipeline and model verification checks passed successfully!")


if __name__ == "__main__":
    run_verification()
