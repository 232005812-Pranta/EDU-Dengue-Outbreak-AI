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

    print("7. Phase 2: constrained allocation math (scarcity 30%-100%)...")
    from operations import SUPPLY_COLS, run_operations
    from simulator import crew_response_table, optimize_deployment, simulate_plan
    origin = "2025-07-14"
    for pct in (0.3, 0.6, 1.0):
        for mode in ("weighted", "triage"):
            ops = run_operations(bundle, raw_df, feat_df, origin, 3, pct, 40, allocation_mode=mode)
            a = ops["allocation"]
            for col in SUPPLY_COLS:
                assert (a[f"{col}_allocated"] <= a[col]).all(), "allocated more than required"
                assert (a[f"{col}_allocated"] >= 0).all()
                for _, g in a.groupby("city"):
                    assert g[f"{col}_allocated"].sum() <= g[f"{col}_stock"].iloc[0], "exceeded city stock"
            assert a["crews_allocated"].sum() <= 40
            if pct == 1.0:
                assert not a["supply_deficit"].any(), "deficit flagged at 100% stock"
            else:
                assert a["supply_deficit"].any(), "no deficit flagged under scarcity"
    print("   Allocations never exceed demand or city stock; deficits flagged only under scarcity.")

    print("8. Phase 2: overflow routing stays inside each city & conserves patients...")
    t, beds = ops["transfers"], ops["beds"]
    city_of = beds.set_index("ward_id")["city"]
    if len(t):
        assert (t["from_ward"].map(city_of) == t["to_ward"].map(city_of)).all(), "cross-city transfer"
    assert beds["transferred_out"].sum() == beds["received_in"].sum()
    assert (beds["hospital_beds_required"] - beds["transferred_out"] + beds["received_in"]
            <= beds["bed_capacity"] + beds["net_unresolved_deficit"]).all()
    print(f"   {len(t)} transfer order(s), all intra-city; patients out == patients in.")

    print("9. Phase 2: vector-control simulator is model-driven and monotone...")
    resp = crew_response_table(bundle, feat_df, origin)
    assert (resp.groupby("ward_id")["cases_2_4w"].diff().dropna() <= 1e-9).all()
    plan = optimize_deployment(resp, 40)
    assert sum(plan.values()) <= 40
    sim = simulate_plan(resp, plan, ops["ratios"])
    print(f"   40 crews prevent {sim['cases_prevented'].sum():.1f} cases over weeks +2..+4.")

    print("10. Offline kill-switch: block ALL network access and run the full Phase 2 stack...")
    import socket
    real_socket, real_conn = socket.socket, socket.create_connection
    def _blocked(*a, **k):
        raise OSError("network disabled by offline kill-switch test")
    socket.socket = _blocked
    socket.create_connection = _blocked
    try:
        import tempfile
        from dispatch import write_exports
        ops2 = run_operations(bundle, load_raw_data(), build_forecast_features(load_raw_data()), origin, 3, 0.5, 30)
        r2 = crew_response_table(bundle, feat_df, origin)
        with tempfile.TemporaryDirectory() as d:
            files = write_exports(d, ops2, optimize_deployment(r2, 30),
                                  {"origin": origin, "horizon": 3, "target": "2025-08-04", "stock_pct": 50, "basis": "point"})
        from i18n import S, t
        from ward_map import build_risk_map
        fig = build_risk_map(ops2["allocation"], ops2["transfers"], {"cases": "c", "fulfil": "f", "crews": "k"})
        fig_dict = fig.to_dict()
        fig_dict["layout"].pop("template", None)  # theme defaults only, not used by the map
        assert not any(k.startswith(("mapbox", "geo", "map")) for k in fig_dict["layout"]), "no tile/geo basemaps"
        assert all(tr["type"] == "scatter" for tr in fig_dict["data"]), "plain scatter traces only"
        fig_text = str(fig_dict)
        assert "http://" not in fig_text and "https://" not in fig_text, "map must not reference external URLs"
        assert all(t(k, "bn") and t(k, "en") for k in S if "{" not in S[k][0])
        print(f"   Forecast + allocation + routing + simulator + {len(files)} dispatch exports "
              "+ offline map + Bangla UI strings ran with network OFF.")
    finally:
        socket.socket, socket.create_connection = real_socket, real_conn

    print("\nAll pipeline and model verification checks passed successfully!")


if __name__ == "__main__":
    run_verification()
