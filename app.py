import os
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    FEATURE_COLS,
    classify_risk_tier,
    compute_ward_weather_lags,
    load_raw_data,
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
    return joblib.load(bundle_path)


@st.cache_data
def get_processed_dataset():
    raw = load_raw_data()
    return compute_ward_weather_lags(raw, drop_na_lags=True)


@st.cache_data
def get_validation_results():
    val_path = os.path.join("models", "chronological_validation_results.csv")
    if not os.path.exists(val_path):
        train_and_save()
    df = pd.read_csv(val_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def predict_rows(bundle: dict, df_rows: pd.DataFrame) -> pd.DataFrame:
    X = df_rows[bundle["feature_cols"]]
    pt = np.clip(bundle["point_model"].predict(X), 0.0, None)
    lo = np.minimum(pt, np.clip(bundle["lower_model"].predict(X), 0.0, None))
    hi = np.maximum(pt, np.clip(bundle["upper_model"].predict(X), 0.0, None))

    out = df_rows.copy()
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
    "water-logging vulnerability, and sanitation scores across Bangladesh cities."
)

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
    "Reference Observation Week", available_dates, index=default_date_idx
)
forecast_horizon = st.sidebar.slider("Early-Warning Horizon (Weeks Ahead)", 1, 4, 3)

st.sidebar.markdown("---")
st.sidebar.header("🌦️ Weather Scenario Simulator (What-If)")
st.sidebar.caption("Adjust rainfall, temperature, or humidity to simulate climate shocks.")
rain_multiplier = st.sidebar.slider("2–4W Lagged Rainfall Multiplier", 0.5, 2.0, 1.0, 0.1)
temp_delta = st.sidebar.slider("Temperature Shift (°C)", -3.0, 4.0, 0.0, 0.5)
humidity_delta = st.sidebar.slider("Relative Humidity Shift (%)", -15.0, 15.0, 0.0, 1.0)

# Extract selected observation and upcoming window
ref_idx = ward_df.index[ward_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str][0]
end_idx = min(len(ward_df) - 1, ref_idx + forecast_horizon)
window_df = ward_df.iloc[ref_idx : end_idx + 1].copy()

# Apply What-If perturbations to weather features & recompute breeding potential
for col in ["rainfall_mm", "rainfall_lag_1w", "rainfall_lag_2w", "rainfall_lag_3w", "rainfall_lag_4w", "rainfall_2_4w_sum"]:
    window_df[col] = window_df[col] * rain_multiplier

for col in ["avg_temp_c", "max_temp_c", "min_temp_c", "temp_lag_1w", "temp_lag_2w", "temp_lag_3w", "temp_lag_4w", "temp_2_4w_mean"]:
    window_df[col] = window_df[col] + temp_delta

for col in ["relative_humidity_pct", "humidity_lag_1w", "humidity_lag_2w", "humidity_lag_3w", "humidity_lag_4w", "humidity_2_4w_mean"]:
    window_df[col] = np.clip(window_df[col] + humidity_delta, 20.0, 100.0)

window_df["stagnant_breeding_potential"] = (
    window_df["rainfall_2_4w_sum"]
    * window_df["water_logging_risk_index"]
    * (1.0 - window_df["sanitation_score"])
)

t_inf0 = time.perf_counter()
pred_window = predict_rows(bundle, window_df)
inference_ms = (time.perf_counter() - t_inf0) * 1000.0

target_row = pred_window.iloc[-1]
tier_label, tier_color, action_text = classify_risk_tier(target_row["predicted_cases"])

# Top KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label=f"Target Week ({target_row['date'].strftime('%Y-%m-%d')})",
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
        label="2–4W Cumulative Rainfall",
        value=f"{target_row['rainfall_2_4w_sum']:.1f} mm",
        delta=f"Humidity: {target_row['humidity_2_4w_mean']:.1f}%",
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
        chart_df = predict_rows(bundle, ward_df.iloc[hist_start:ref_idx].copy())
        combined_df = pd.concat([chart_df, pred_window], ignore_index=True)

        fig_traj = go.Figure()
        fig_traj.add_trace(
            go.Scatter(
                x=combined_df["date"],
                y=combined_df["ci_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper 90%",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=combined_df["date"],
                y=combined_df["ci_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(239, 68, 68, 0.18)",
                name="80% Confidence Interval",
            )
        )
        fig_traj.add_trace(
            go.Scatter(
                x=combined_df["date"],
                y=combined_df["predicted_cases"],
                mode="lines+markers",
                name="Predicted Dengue Cases",
                line=dict(color="#ef4444", width=3),
            )
        )
        if rain_multiplier == 1.0 and temp_delta == 0.0 and humidity_delta == 0.0:
            fig_traj.add_trace(
                go.Scatter(
                    x=combined_df["date"],
                    y=combined_df["dengue_cases"],
                    mode="lines+markers",
                    name="Observed Cases",
                    line=dict(color="#3b82f6", width=2, dash="dot"),
                )
            )
        fig_traj.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Week",
            yaxis_title="Weekly Dengue Cases",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with right_col:
        st.subheader("Key Outbreak Risk Drivers (Target Week)")
        city_baseline = df[df["city"] == selected_city]
        drivers = [
            {
                "Driver": "2–4W Lagged Rainfall (mm)",
                "Ward Value": round(float(target_row["rainfall_2_4w_sum"]), 1),
                "City Median": round(float(city_baseline["rainfall_2_4w_sum"].median()), 1),
                "Relative Index": float(target_row["rainfall_2_4w_sum"]) / max(1.0, float(city_baseline["rainfall_2_4w_sum"].median())),
            },
            {
                "Driver": "Stagnant Breeding Potential",
                "Ward Value": round(float(target_row["stagnant_breeding_potential"]), 1),
                "City Median": round(float(city_baseline["stagnant_breeding_potential"].median()), 1),
                "Relative Index": float(target_row["stagnant_breeding_potential"]) / max(1.0, float(city_baseline["stagnant_breeding_potential"].median())),
            },
            {
                "Driver": "Water-Logging Risk Index",
                "Ward Value": round(float(target_row["water_logging_risk_index"]), 2),
                "City Median": round(float(city_baseline["water_logging_risk_index"].median()), 2),
                "Relative Index": float(target_row["water_logging_risk_index"]) / max(0.01, float(city_baseline["water_logging_risk_index"].median())),
            },
            {
                "Driver": "2–4W Mean Humidity (%)",
                "Ward Value": round(float(target_row["humidity_2_4w_mean"]), 1),
                "City Median": round(float(city_baseline["humidity_2_4w_mean"].median()), 1),
                "Relative Index": float(target_row["humidity_2_4w_mean"]) / max(1.0, float(city_baseline["humidity_2_4w_mean"].median())),
            },
            {
                "Driver": "Poor Sanitation Factor (1 - Score)",
                "Ward Value": round(1.0 - float(target_row["sanitation_score"]), 2),
                "City Median": round(1.0 - float(city_baseline["sanitation_score"].median()), 2),
                "Relative Index": (1.0 - float(target_row["sanitation_score"])) / max(0.01, 1.0 - float(city_baseline["sanitation_score"].median())),
            },
            {
                "Driver": "Population Density (per sqkm)",
                "Ward Value": int(target_row["population_density_sqkm"]),
                "City Median": int(city_baseline["population_density_sqkm"].median()),
                "Relative Index": float(target_row["population_density_sqkm"]) / max(1.0, float(city_baseline["population_density_sqkm"].median())),
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
        "date",
        "rainfall_mm",
        "rainfall_lag_2w",
        "rainfall_lag_3w",
        "rainfall_lag_4w",
        "rainfall_2_4w_sum",
        "avg_temp_c",
        "relative_humidity_pct",
        "predicted_cases",
        "ci_lower",
        "ci_upper",
        "risk_tier",
    ]
    st.dataframe(pred_window[display_cols], use_container_width=True)

with tab2:
    st.subheader(f"Interactive Ward-Level Outbreak Risk Heatmap — {selected_city}")
    st.caption(
        "Displays predicted weekly Dengue cases across all wards in the selected city over an 8-week window around the chosen reference date."
    )
    city_all = df[df["city"] == selected_city].copy()
    unique_city_dates = sorted(city_all["date"].unique())
    ref_ts = pd.Timestamp(selected_date_str)
    pos = unique_city_dates.index(ref_ts) if ref_ts in unique_city_dates else len(unique_city_dates) - 5
    slice_dates = unique_city_dates[max(0, pos - 4) : min(len(unique_city_dates), pos + 4)]

    heat_sub = city_all[city_all["date"].isin(slice_dates)].copy()
    heat_pred = predict_rows(bundle, heat_sub)
    heat_pred["week_str"] = heat_pred["date"].dt.strftime("%Y-%m-%d")

    pivot_heat = heat_pred.pivot(index="ward_id", columns="week_str", values="predicted_cases")
    fig_heat = px.imshow(
        pivot_heat,
        labels=dict(x="Week", y="Ward ID", color="Predicted Cases"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Chronological Outbreak Validation Dashboard (Unseen 2024–2025 Seasons)")
    metrics = bundle["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE (Mean Abs Error)", f"{metrics['mae']} cases")
    m2.metric("Test RMSE", f"{metrics['rmse']} cases")
    m3.metric("Test R² Score", f"{metrics['r2']}")
    m4.metric("80% CI Empirical Coverage", f"{metrics['interval_coverage_pct']}%")

    st.caption(
        f"Strict temporal split — Training Window: {metrics['train_period']} ({metrics['train_rows']:,} rows) | "
        f"Validation Window: {metrics['test_period']} ({metrics['test_rows']:,} rows)."
    )

    c_left, c_right = st.columns([1.4, 1.0])
    with c_left:
        weekly_agg = (
            val_df[val_df["city"] == selected_city]
            .groupby("date")[["dengue_cases", "predicted_cases"]]
            .sum()
            .reset_index()
        )
        fig_val = go.Figure()
        fig_val.add_trace(
            go.Scatter(
                x=weekly_agg["date"],
                y=weekly_agg["dengue_cases"],
                mode="lines",
                name="Actual City Cases",
                line=dict(color="#2563eb", width=2),
            )
        )
        fig_val.add_trace(
            go.Scatter(
                x=weekly_agg["date"],
                y=weekly_agg["predicted_cases"],
                mode="lines",
                name="Predicted City Cases",
                line=dict(color="#dc2626", width=2, dash="dash"),
            )
        )
        fig_val.update_layout(
            title=f"{selected_city} — Total Weekly Actual vs. Predicted Outbreak Curve (2024–2025)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_val, use_container_width=True)

    with c_right:
        st.markdown("**City-Level Validation Error Breakdown**")
        city_err_df = pd.DataFrame(metrics["city_metrics"])
        st.dataframe(city_err_df, use_container_width=True)