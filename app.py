import os
import time
import joblib
import sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    FEATURE_COLS,
    FORECAST_HORIZONS,
    build_forecast_features,
    classify_risk_tier,
    load_raw_data,
    recompute_derived_features,
)
from train_model import train_and_save

st.set_page_config(
    page_title="Urban Dengue Outbreak Early Warning AI",
    page_icon="🦟",
    layout="wide",
)


@st.cache_resource
def get_model_bundle():
    bundle_path = os.path.join("models", "dengue_model_bundle.joblib")
    if not os.path.exists(bundle_path):
        train_and_save()
    try:
        bundle = joblib.load(bundle_path)
        stale = "horizon_models" not in bundle or bundle.get("sklearn_version") != sklearn.__version__
    except Exception:
        stale = True
    if stale:  # old preliminary bundle or different scikit-learn version -> retrain (~10s)
        train_and_save()
        bundle = joblib.load(bundle_path)
    return bundle


@st.cache_data
def get_processed_dataset():
    raw = load_raw_data()
    return build_forecast_features(raw)


@st.cache_data
def get_validation_results():
    val_path = os.path.join("models", "forecast_validation_results.csv")
    if not os.path.exists(val_path):
        train_and_save()
    df = pd.read_csv(val_path)
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def predict_rows(bundle: dict, origin_rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Leakage-free h-week-ahead forecast. Each input row is a forecast ORIGIN week t
    (only data observed up to t); output is the prediction for week t+h.
    """
    models = bundle["horizon_models"][horizon]
    X = origin_rows[bundle["forecast_feature_cols"]]
    pt = np.clip(models["point_model"].predict(X), 0.0, None)
    lo = np.minimum(pt, np.clip(models["lower_model"].predict(X), 0.0, None))
    hi = np.maximum(pt, np.clip(models["upper_model"].predict(X), 0.0, None))

    out = origin_rows.copy()
    out["origin_date"] = out["date"]
    out["horizon"] = horizon
    out["target_date"] = out["date"] + pd.Timedelta(weeks=horizon)
    out["predicted_cases"] = np.round(pt, 2)
    out["ci_lower"] = np.round(lo, 2)
    out["ci_upper"] = np.round(hi, 2)
    tiers = [classify_risk_tier(v) for v in pt]
    out["risk_tier"] = [t[0] for t in tiers]
    out["tier_color"] = [t[1] for t in tiers]
    out["recommended_action"] = [t[2] for t in tiers]
    return out


# Load cached artifacts
bundle = get_model_bundle()
df = get_processed_dataset()
val_df = get_validation_results()

st.title("🦟 Urban Dengue Outbreak Early-Warning System (2–4 Week Forecast)")
st.caption(
    "Hyper-local ward risk engine modeling the 2-to-4-week Aedes aegypti breeding cycle, "
    "water-logging vulnerability, and sanitation scores across Bangladesh cities. "
    "Forecasts use ONLY data observed up to the reference week — no future weather or case data."
)

# Observed case lookup (for comparing forecasts against what actually happened)
observed_lookup = load_raw_data().set_index(["ward_id", "date"])["dengue_cases"]

# Sidebar Controls
st.sidebar.header("🎯 Target Area & Forecast Window")
cities = sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", cities, index=0)

city_wards = sorted(df[df["city"] == selected_city]["ward_id"].unique().tolist())
selected_ward = st.sidebar.selectbox("Select Ward ID", city_wards, index=0)

ward_df = df[df["ward_id"] == selected_ward].sort_values("date").reset_index(drop=True)
available_dates = ward_df["date"].dt.strftime("%Y-%m-%d").tolist()
default_date_idx = max(0, len(available_dates) - 8)
selected_date_str = st.sidebar.selectbox(
    "Forecast Issued On (Reference Week)", available_dates, index=default_date_idx,
    help="The last week of observed data. Pick the final week to see a genuine forecast into the unseen future.",
)
forecast_horizon = st.sidebar.slider("Early-Warning Horizon (Weeks Ahead)", 1, 4, 3)

st.sidebar.markdown("---")
st.sidebar.header("🌦️ Weather Scenario Simulator (What-If)")
st.sidebar.caption("Adjust rainfall, temperature, or humidity to simulate climate shocks.")
rain_multiplier = st.sidebar.slider("Recent Rainfall Multiplier (Last 4 Weeks)", 0.5, 2.0, 1.0, 0.1)
temp_delta = st.sidebar.slider("Temperature Shift (°C)", -3.0, 4.0, 0.0, 0.5)
humidity_delta = st.sidebar.slider("Relative Humidity Shift (%)", -15.0, 15.0, 0.0, 1.0)

# Forecast origin row (everything known at the reference week)
ref_idx = ward_df.index[ward_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str][0]
origin_df = ward_df.iloc[[ref_idx]].copy()


def apply_what_if(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for col in ["rainfall_mm", "rainfall_lag_1w", "rainfall_lag_2w", "rainfall_lag_3w", "rainfall_lag_4w"]:
        rows[col] = rows[col] * rain_multiplier
    for col in ["avg_temp_c", "max_temp_c", "min_temp_c", "temp_lag_1w", "temp_lag_2w", "temp_lag_3w", "temp_lag_4w"]:
        rows[col] = rows[col] + temp_delta
    for col in ["relative_humidity_pct", "humidity_lag_1w", "humidity_lag_2w", "humidity_lag_3w", "humidity_lag_4w"]:
        rows[col] = np.clip(rows[col] + humidity_delta, 20.0, 100.0)
    return recompute_derived_features(rows)


origin_df = apply_what_if(origin_df)

# Direct multi-horizon forecast: one dedicated model per horizon, same origin
t_inf0 = time.perf_counter()
pred_window = pd.concat(
    [predict_rows(bundle, origin_df, h) for h in FORECAST_HORIZONS], ignore_index=True
)
inference_ms = (time.perf_counter() - t_inf0) * 1000.0
pred_window["dengue_cases"] = [
    observed_lookup.get((selected_ward, d), np.nan) for d in pred_window["target_date"]
]

target_row = pred_window[pred_window["horizon"] == forecast_horizon].iloc[0]
origin_row = origin_df.iloc[0]
tier_label, tier_color, action_text = classify_risk_tier(target_row["predicted_cases"])

# Top KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label=f"+{forecast_horizon}W Target Week ({target_row['target_date'].strftime('%Y-%m-%d')})",
        value=f"{target_row['predicted_cases']:.1f} cases",
        delta=f"80% CI: [{target_row['ci_lower']:.1f} – {target_row['ci_upper']:.1f}]",
    )
with col2:
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background-color:{tier_color}22;border-left:6px solid {tier_color};'>"
        f"<span style='font-size:13px;color:gray;'>Outbreak Risk Tier</span><br>"
        f"<strong style='font-size:24px;color:{tier_color};'>{tier_label.upper()}</strong></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.metric(
        label="Rainfall, Last 3 Weeks (feeds 2–4W breeding window)",
        value=f"{origin_row['rainfall_0_2w_sum']:.1f} mm",
        delta=f"Cases this week: {int(origin_row['cases_lag_0w'])} | 4W avg {origin_row['cases_4w_mean']:.1f}",
        delta_color="off",
    )
with col4:
    st.metric(
        label="Local Inference Latency",
        value=f"{inference_ms:.2f} ms",
        delta="0 External API Calls",
    )

st.info(f"**Recommended Public Health Action ({selected_ward}):** {action_text}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Ward Forecast & Weather Drivers",
        "🗺️ City-Wide Ward Risk Heatmap",
        "📈 Chronological Validation (MAE / RMSE)",
    ]
)

with tab1:
    left_col, right_col = st.columns([1.3, 1.0])

    with left_col:
        st.subheader(f"Multi-Week Trajectory & Confidence Band — {selected_ward}")
        hist_start = max(0, ref_idx - 12)
        hist_df = ward_df.iloc[hist_start : ref_idx + 1]
        # Anchor the forecast fan at the last observed week
        anchor = pd.DataFrame(
            {
                "target_date": [origin_row["date"]],
                "predicted_cases": [origin_row["cases_lag_0w"]],
                "ci_lower": [origin_row["cases_lag_0w"]],
                "ci_upper": [origin_row["cases_lag_0w"]],
            }
        )
        fan_df = pd.concat([anchor, pred_window], ignore_index=True)

        fig_traj = go.Figure()
        fig_traj.add_trace(
            go.Scatter(
                x=fan_df["target_date"], y=fan_df["ci_upper"], mode="lines",
                line=dict(width=0), showlegend=False, name="Upper 90%",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=fan_df["target_date"], y=fan_df["ci_lower"], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(239, 68, 68, 0.18)",
                name="80% Confidence Interval",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=fan_df["target_date"], y=fan_df["predicted_cases"], mode="lines+markers",
                name="Early-Warning Forecast (1–4W)", line=dict(color="#ef4444", width=3),
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=hist_df["date"], y=hist_df["dengue_cases"], mode="lines+markers",
                name="Observed Cases (known at reference week)", line=dict(color="#3b82f6", width=2),
            )
        )
        future_obs = pred_window.dropna(subset=["dengue_cases"])
        if len(future_obs) and rain_multiplier == 1.0 and temp_delta == 0.0 and humidity_delta == 0.0:
            fig_traj.add_trace(
                go.Scatter(
                    x=future_obs["target_date"], y=future_obs["dengue_cases"], mode="markers",
                    name="Actual Outcome (hidden from model)",
                    marker=dict(color="#3b82f6", size=9, symbol="x"),
                )
            )
        fig_traj.add_vline(x=origin_row["date"], line_dash="dash", line_color="gray")
        fig_traj.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Week",
            yaxis_title="Weekly Dengue Cases",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with right_col:
        st.subheader("Key Outbreak Risk Drivers (Reference Week)")
        city_baseline = df[df["city"] == selected_city]
        drivers = [
            {
                "Driver": "Rainfall Last 3 Weeks (mm)",
                "Ward Value": round(float(origin_row["rainfall_0_2w_sum"]), 1),
                "City Median": round(float(city_baseline["rainfall_0_2w_sum"].median()), 1),
                "Relative Index": float(origin_row["rainfall_0_2w_sum"]) / max(1.0, float(city_baseline["rainfall_0_2w_sum"].median())),
            },
            {
                "Driver": "Recent Cases (4W Mean)",
                "Ward Value": round(float(origin_row["cases_4w_mean"]), 1),
                "City Median": round(float(city_baseline["cases_4w_mean"].median()), 1),
                "Relative Index": float(origin_row["cases_4w_mean"]) / max(1.0, float(city_baseline["cases_4w_mean"].median())),
            },
            {
                "Driver": "Stagnant Breeding Potential",
                "Ward Value": round(float(origin_row["stagnant_breeding_potential"]), 1),
                "City Median": round(float(city_baseline["stagnant_breeding_potential"].median()), 1),
                "Relative Index": float(origin_row["stagnant_breeding_potential"]) / max(1.0, float(city_baseline["stagnant_breeding_potential"].median())),
            },
            {
                "Driver": "Water-Logging Risk Index",
                "Ward Value": round(float(origin_row["water_logging_risk_index"]), 2),
                "City Median": round(float(city_baseline["water_logging_risk_index"].median()), 2),
                "Relative Index": float(origin_row["water_logging_risk_index"]) / max(0.01, float(city_baseline["water_logging_risk_index"].median())),
            },
            {
                "Driver": "2–4W Mean Humidity (%)",
                "Ward Value": round(float(origin_row["humidity_2_4w_mean"]), 1),
                "City Median": round(float(city_baseline["humidity_2_4w_mean"].median()), 1),
                "Relative Index": float(origin_row["humidity_2_4w_mean"]) / max(1.0, float(city_baseline["humidity_2_4w_mean"].median())),
            },
            {
                "Driver": "Poor Sanitation Factor (1 - Score)",
                "Ward Value": round(1.0 - float(origin_row["sanitation_score"]), 2),
                "City Median": round(1.0 - float(city_baseline["sanitation_score"].median()), 2),
                "Relative Index": (1.0 - float(origin_row["sanitation_score"])) / max(0.01, 1.0 - float(city_baseline["sanitation_score"].median())),
            },
            {
                "Driver": "Population Density (per sqkm)",
                "Ward Value": int(origin_row["population_density_sqkm"]),
                "City Median": int(city_baseline["population_density_sqkm"].median()),
                "Relative Index": float(origin_row["population_density_sqkm"]) / max(1.0, float(city_baseline["population_density_sqkm"].median())),
            },
        ]
        driver_df = pd.DataFrame(drivers)
        fig_drv = px.bar(
            driver_df,
            x="Relative Index",
            y="Driver",
            orientation="h",
            text="Ward Value",
            color="Relative Index",
            color_continuous_scale="YlOrRd",
        )
        fig_drv.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="City Baseline (1.0x)")
        fig_drv.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_drv, use_container_width=True)

    st.subheader("Multi-Week Lagged Weather & Prediction Breakdown")
    display_cols = [
        "origin_date",
        "horizon",
        "target_date",
        "predicted_cases",
        "ci_lower",
        "ci_upper",
        "risk_tier",
        "dengue_cases",
        "recommended_action",
    ]
    st.dataframe(
        pred_window[display_cols].rename(columns={"dengue_cases": "actual_cases (if known)"}),
        use_container_width=True,
    )

with tab2:
    st.subheader(f"Interactive Ward-Level Outbreak Risk Heatmap — {selected_city}")
    st.caption(
        f"Predicted weekly Dengue cases per ward, each issued {forecast_horizon} week(s) in advance, "
        "over an 8-week window around the chosen reference date. "
        "Colors use a fixed scale matched to the risk tiers (Medium ≥ 3, High ≥ 7, Critical ≥ 12)."
    )
    city_all = df[df["city"] == selected_city].copy()
    unique_city_dates = sorted(city_all["date"].unique())
    ref_ts = pd.Timestamp(selected_date_str)
    pos = unique_city_dates.index(ref_ts) if ref_ts in unique_city_dates else len(unique_city_dates) - 5
    slice_dates = unique_city_dates[max(0, pos - 4) : min(len(unique_city_dates), pos + 4)]

    heat_sub = city_all[city_all["date"].isin(slice_dates)].copy()
    heat_pred = predict_rows(bundle, heat_sub, forecast_horizon)
    heat_pred["week_str"] = heat_pred["target_date"].dt.strftime("%Y-%m-%d")

    pivot_heat = heat_pred.pivot(index="ward_id", columns="week_str", values="predicted_cases")
    # Fixed color scale aligned with risk tiers (0-3 Low, 3-7 Medium, 7-12 High, 12+ Critical),
    # so colors mean the same thing every week: a dry-season 0.1 stays pale, never dark red.
    HEAT_MAX = 15.0
    tier_colorscale = [
        [0.0, "#f0fdf4"],
        [3.0 / HEAT_MAX, "#fde68a"],
        [7.0 / HEAT_MAX, "#fb923c"],
        [12.0 / HEAT_MAX, "#dc2626"],
        [1.0, "#7f1d1d"],
    ]
    fig_heat = px.imshow(
        pivot_heat,
        labels=dict(x="Target Week", y="Ward ID", color="Predicted Cases"),
        color_continuous_scale=tier_colorscale,
        range_color=[0.0, HEAT_MAX],
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar=dict(
            tickvals=[0, 3, 7, 12, 15],
            ticktext=["0", "3 Medium", "7 High", "12 Critical", "15+"],
        ),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Leakage-Free Early-Warning Validation (Unseen 2024–2025 Seasons)")
    fm = pd.DataFrame(bundle["forecast_metrics"])
    hm = fm[fm["horizon_weeks"] == forecast_horizon].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"+{forecast_horizon}W Test MAE", f"{hm['mae']:.3f} cases",
              delta=f"{hm['mae_improvement_vs_naive_pct']:+.1f}% vs naive", delta_color="off")
    m2.metric(f"+{forecast_horizon}W Test RMSE", f"{hm['rmse']:.3f} cases")
    m3.metric(f"+{forecast_horizon}W Test R²", f"{hm['r2']:.3f}")
    m4.metric("80% CI Empirical Coverage", f"{hm['interval_coverage_pct']:.1f}%")

    st.caption(
        "One dedicated model per horizon. Training uses only (origin, target) pairs whose target week is before "
        "2024-01-01; testing uses forecasts issued from 2024-01-01 onward. Inputs are restricted to data observed "
        "up to the origin week. Naive baseline = 'cases stay the same as this week'."
    )

    c_left, c_right = st.columns([1.4, 1.0])
    with c_left:
        weekly_agg = (
            val_df[(val_df["city"] == selected_city) & (val_df["horizon"] == forecast_horizon)]
            .groupby("target_date")[["dengue_cases", "predicted_cases"]]
            .sum()
            .reset_index()
        )
        fig_val = go.Figure()
        fig_val.add_trace(
            go.Scatter(
                x=weekly_agg["target_date"], y=weekly_agg["dengue_cases"], mode="lines",
                name="Actual City Cases", line=dict(color="#2563eb", width=2),
            )
        )
        fig_val.add_trace(
            go.Scatter(
                x=weekly_agg["target_date"], y=weekly_agg["predicted_cases"], mode="lines",
                name=f"Predicted {forecast_horizon}W in Advance", line=dict(color="#dc2626", width=2, dash="dash"),
            )
        )
        fig_val.update_layout(
            title=f"{selected_city} — Actual vs. {forecast_horizon}-Week-Ahead Forecast (2024–2025)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_val, use_container_width=True)

    with c_right:
        st.markdown("**Accuracy by Forecast Horizon**")
        st.dataframe(
            fm[["horizon_weeks", "mae", "naive_persistence_mae", "mae_improvement_vs_naive_pct",
                "rmse", "r2", "interval_coverage_pct"]],
            use_container_width=True, hide_index=True,
        )
        city_err = (
            val_df[val_df["horizon"] == forecast_horizon]
            .assign(abs_err=lambda d: (d["dengue_cases"] - d["predicted_cases"]).abs())
            .groupby("city")["abs_err"].mean().round(3).reset_index(name=f"MAE (+{forecast_horizon}W)")
        )
        st.markdown(f"**City-Level Error (+{forecast_horizon}W)**")
        st.dataframe(city_err, use_container_width=True, hide_index=True)

    lm = bundle["metrics"]
    st.caption(
        f"For reference — preliminary-round same-week nowcast model (uses the target week's own weather, so not a "
        f"true forecast): MAE {lm['mae']}, R² {lm['r2']}."
    )
