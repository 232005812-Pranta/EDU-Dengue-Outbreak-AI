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
from operations import SUPPLY_COLS, run_operations
from simulator import crew_response_table, optimize_deployment, roi_ranking, simulate_plan
from dispatch import build_ward_notices, commissioner_brief_html
from i18n import ACTION_BN, TIER, num, t
from ward_map import build_risk_map

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
def get_raw_dataset():
    return load_raw_data()


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

# Offline language switch (local dictionary only)
L = st.sidebar.radio(t("lang_label"), ["en", "bn"], horizontal=True, key="lang",
                     format_func=lambda x: "English" if x == "en" else "বাংলা")



def lang_radio(container, label, options, name, **kw):
    """Radio whose option labels are translated: keyed per language so the browser always
    redraws the selection, while the chosen value is remembered across language switches."""
    remembered = st.session_state.get(f"_{name}", options[0])
    idx = options.index(remembered) if remembered in options else 0
    value = container.radio(label, options, index=idx, key=f"{name}_{L}", **kw)
    st.session_state[f"_{name}"] = value
    return value


st.title(t("title", L))
st.caption(t("caption", L))

# Observed case lookup (for comparing forecasts against what actually happened)
observed_lookup = load_raw_data().set_index(["ward_id", "date"])["dengue_cases"]

# Sidebar Controls
st.sidebar.header(t("sb_target", L))
cities = sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox(t("city", L), cities, index=0, key="city")

city_wards = sorted(df[df["city"] == selected_city]["ward_id"].unique().tolist())
selected_ward = st.sidebar.selectbox(t("ward", L), city_wards, index=0, key=f"ward_{selected_city}")

ward_df = df[df["ward_id"] == selected_ward].sort_values("date").reset_index(drop=True)
available_dates = ward_df["date"].dt.strftime("%Y-%m-%d").tolist()
# Default to a peak-monsoon week so the operations tabs open on a meaningful scenario
default_date_idx = (available_dates.index("2025-07-14") if "2025-07-14" in available_dates
                    else max(0, len(available_dates) - 8))
selected_date_str = st.sidebar.selectbox(
    t("ref_week", L), available_dates, index=default_date_idx, key="ref_week", help=t("ref_help", L),
)
forecast_horizon = st.sidebar.slider(t("horizon", L), 1, 4, 3, key="horizon")

st.sidebar.markdown("---")
st.sidebar.header(t("sb_weather", L))
st.sidebar.caption(t("sb_weather_cap", L))
rain_multiplier = st.sidebar.slider(t("rain_mult", L), 0.5, 2.0, 1.0, 0.1, key="rain_mult")
temp_delta = st.sidebar.slider(t("temp_shift", L), -3.0, 4.0, 0.0, 0.5, key="temp_shift")
humidity_delta = st.sidebar.slider(t("hum_shift", L), -15.0, 15.0, 0.0, 1.0, key="hum_shift")

st.sidebar.markdown("---")
st.sidebar.header(t("sb_ops", L))
st.sidebar.caption(t("sb_ops_cap", L))
stock_pct = st.sidebar.slider(t("stock", L), 30, 100, 60, 5, key="stock")
total_crews = st.sidebar.slider(t("crews", L), 0, 150, 40, 5, key="crews")
allocation_mode = lang_radio(st.sidebar, t("alloc_rule", L), ["weighted", "triage"], "alloc_mode",
                             horizontal=True, help=t("alloc_help", L))
planning_basis = lang_radio(
    st.sidebar, t("basis", L), ["point", "upper"], "basis", horizontal=True,
    format_func=lambda x: t("basis_point", L) if x == "point" else t("basis_upper", L),
)
capacity_multiplier = st.sidebar.slider(t("cap_mult", L), 0.5, 1.5, 1.0, 0.1, key="cap_mult")
crew_efficacy = st.sidebar.slider(t("efficacy", L), 0.10, 0.40, 0.25, 0.05, key="efficacy")

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
        label=t("kpi_target", L, h=forecast_horizon, d=target_row['target_date'].strftime('%Y-%m-%d')),
        value=t("kpi_cases", L, v=target_row['predicted_cases']),
        delta=f"80% CI: [{target_row['ci_lower']:.1f} – {target_row['ci_upper']:.1f}]",
    )
with col2:
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background-color:{tier_color}22;border-left:6px solid {tier_color};'>"
        f"<span style='font-size:13px;color:gray;'>{t('kpi_tier', L)}</span><br>"
        f"<strong style='font-size:24px;color:{tier_color};'>{TIER[L][tier_label].upper()}</strong></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.metric(
        label=t("kpi_rain", L),
        value=f"{origin_row['rainfall_0_2w_sum']:.1f} mm",
        delta=t("kpi_rain_delta", L, c=int(origin_row['cases_lag_0w']), a=origin_row['cases_4w_mean']),
        delta_color="off",
    )
with col4:
    st.metric(
        label=t("kpi_latency", L),
        value=f"{inference_ms:.2f} ms",
        delta=t("kpi_api", L),
    )

st.info(t("action", L, w=selected_ward, a=ACTION_BN[tier_label] if L == "bn" else action_text))

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [t(f"tab{i}", L) for i in range(1, 9)]
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


# ===========================================================================
# PHASE 2: Operations command system (all offline, deterministic)
# ===========================================================================
raw_df = get_raw_dataset()
ops = run_operations(
    bundle, raw_df, df, selected_date_str, forecast_horizon, stock_pct / 100.0, total_crews,
    planning_basis=planning_basis, capacity_multiplier=capacity_multiplier, allocation_mode=allocation_mode,
)
alloc_df, beds_df, transfers_df, city_df = ops["allocation"], ops["beds"], ops["transfers"], ops["city"]
target_week_str = (pd.Timestamp(selected_date_str) + pd.Timedelta(weeks=forecast_horizon)).strftime("%Y-%m-%d")
response_df = crew_response_table(bundle, df, selected_date_str, max_crews=5, efficacy=crew_efficacy)

with tab4:
    st.subheader(t("alloc_title", L, d=target_week_str, h=forecast_horizon))
    st.caption(t("alloc_cap", L, s=ops['ratios']['saline_bags_needed'], k=ops['ratios']['diagnostic_kits_needed'],
                 b=ops['ratios']['hospital_beds_required'], p=stock_pct, m=allocation_mode))
    k = st.columns(5)
    supply_names = {"saline_bags_needed": t("saline", L), "diagnostic_kits_needed": t("kits", L),
                    "hospital_beds_required": t("beds", L)}
    for i, col in enumerate(SUPPLY_COLS):
        req, al = int(alloc_df[col].sum()), int(alloc_df[f"{col}_allocated"].sum())
        k[i].metric(supply_names[col], f"{al:,} / {req:,}",
                    delta=t("deficit_delta", L, n=f"{al - req:,}") if al < req else t("fully_met", L),
                    delta_color="inverse" if al < req else "normal")
    k[3].metric(t("crews_kpi", L), f"{int(alloc_df['crews_allocated'].sum())} / {int(alloc_df['crews_required'].sum())}")
    k[4].metric(t("deficit_wards", L), int(alloc_df["supply_deficit"].sum()),
                delta=t("hc", L, n=int(alloc_df['critical_deficit'].sum())), delta_color="inverse")

    crit = alloc_df[alloc_df["critical_deficit"]]
    if len(crit):
        st.error(
            t("crit_alert", L, n=len(crit))
            + ", ".join(crit["ward_id"].head(12)) + (" …" if len(crit) > 12 else "")
        )
    elif alloc_df["supply_deficit"].any():
        st.warning(t("partial_alert", L, n=int(alloc_df['supply_deficit'].sum())))
    else:
        st.success(t("ok_alert", L))

    view = alloc_df[alloc_df["city"] == selected_city] if st.toggle(t("city_only", L), value=False, key="city_only") else alloc_df
    show = view[["priority_rank", "city", "ward_id", "risk_tier", "population_density_sqkm", "predicted_cases",
                 "saline_bags_needed", "saline_bags_needed_allocated", "diagnostic_kits_needed",
                 "diagnostic_kits_needed_allocated", "hospital_beds_required", "hospital_beds_required_allocated",
                 "crews_required", "crews_allocated", "fulfillment_pct", "supply_deficit"]].copy()
    show["predicted_cases"] = show["predicted_cases"].round(1)
    st.dataframe(
        show.style.apply(lambda r: ["background-color:#fee2e2" if r["supply_deficit"] else "" for _ in r], axis=1)
        .format(precision=1),
        use_container_width=True, hide_index=True, height=420,
    )
    fig_f = px.bar(alloc_df.sort_values("priority_rank"), x="ward_id", y="fulfillment_pct", color="risk_tier",
                   color_discrete_map={"Low": "#16a34a", "Medium": "#eab308", "High": "#f97316", "Critical": "#dc2626"},
                   labels={"fulfillment_pct": "Supply fulfilment %", "ward_id": "Ward (priority order →)"})
    fig_f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_f, use_container_width=True)
    st.download_button(t("dl_alloc", L), alloc_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"allocation_{selected_date_str}_h{forecast_horizon}.csv", mime="text/csv")

with tab5:
    st.subheader(t("overflow_title", L, d=target_week_str))
    st.caption(t("overflow_cap", L))
    st.dataframe(city_df, use_container_width=True, hide_index=True)
    over = city_df[city_df["unresolved_deficit"] > 0]
    for r in over.itertuples():
        st.error(t("city_over", L, c=r.city, n=int(r.unresolved_deficit)))
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        bc = beds_df[beds_df["city"] == selected_city]
        fig_b = go.Figure()
        fig_b.add_bar(x=bc["ward_id"], y=bc["bed_capacity"], name="Local bed capacity", marker_color="#94a3b8")
        fig_b.add_bar(x=bc["ward_id"], y=bc["hospital_beds_required"], name="Beds required", marker_color="#ef4444")
        fig_b.add_bar(x=bc["ward_id"], y=bc["beds_occupied_after_routing"], name="Occupied after routing", marker_color="#2563eb")
        fig_b.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10),
                            title=f"{selected_city}: demand vs capacity per ward")
        st.plotly_chart(fig_b, use_container_width=True)
    with c2:
        st.markdown(f"**{t('transfers', L, c=selected_city)}**")
        tc = transfers_df[transfers_df["city"] == selected_city]
        if len(tc):
            st.dataframe(tc, use_container_width=True, hide_index=True)
        else:
            st.info(t("no_transfer", L))
    st.markdown(f"**{t('ledger', L)}**")
    st.dataframe(beds_df[beds_df["city"] == selected_city][[
        "ward_id", "risk_tier", "hospital_beds_required", "bed_capacity", "local_deficit", "local_headroom",
        "transferred_out", "received_in", "net_unresolved_deficit", "utilization_pct"]],
        use_container_width=True, hide_index=True)

with tab6:
    st.subheader(t("sim_title", L))
    st.caption(
        f"Each crew deployed at {selected_date_str} cuts the ward's effective breeding rainfall by "
        f"{int(crew_efficacy * 100)}% (diminishing returns); the SAME trained ML models are re-run for weeks +2, +3, +4."
    )
    mode_sim = lang_radio(st, t("plan_mode", L), ["auto", "manual"], "sim_mode", horizontal=True,
                          format_func=lambda x: t(x, L))
    if mode_sim == "manual":
        choices = st.multiselect(t("sim_wards", L), sorted(alloc_df["ward_id"]), key="sim_wards",
                                 default=list(roi_ranking(response_df)["ward_id"].head(5)))
        per = st.slider(t("sim_per", L), 1, 5, 2, key="sim_per")
        plan = {w: per for w in choices}
    else:
        plan = optimize_deployment(response_df, total_crews)
    sim_df = simulate_plan(response_df, plan, ops["ratios"])
    deployed = sim_df[sim_df["crews_deployed"] > 0]
    m = st.columns(4)
    m[0].metric(t("crews_deployed", L), int(deployed["crews_deployed"].sum()))
    m[1].metric(t("prevented", L), f"{deployed['cases_prevented'].sum():.1f}")
    m[2].metric(t("beds_freed", L), f"{deployed['beds_freed'].sum():.1f}")
    m[3].metric(t("saline_saved", L), f"{deployed['saline_saved'].sum():.0f}")
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(f"**{t('plan_table', L)}**")
        st.dataframe(deployed, use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"**{t('roi_table', L)}**")
        st.dataframe(roi_ranking(response_df).head(15), use_container_width=True, hide_index=True)
    ward_curve = response_df[response_df["ward_id"] == selected_ward]
    fig_c = go.Figure()
    for h in [2, 3, 4]:
        fig_c.add_scatter(x=ward_curve["crews"], y=ward_curve[f"cases_h{h}"], mode="lines+markers", name=f"Week +{h}")
    fig_c.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Crews deployed",
                        yaxis_title="Predicted cases", title=f"{selected_ward}: model response to crews")
    st.plotly_chart(fig_c, use_container_width=True)

with tab7:
    st.subheader(t("dispatch_title", L))
    crews_plan = optimize_deployment(response_df, total_crews)
    notices = build_ward_notices(alloc_df, beds_df, transfers_df, crews_plan)
    city_notices = [n for n in notices if n["city"] == selected_city]
    ward_pick = st.selectbox(t("ward", L), [n["ward_id"] for n in city_notices], key="dispatch_ward")
    n = next(x for x in city_notices if x["ward_id"] == ward_pick)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📱 SMS (English)**")
        st.code(n["sms_en"], language=None)
        st.markdown("**📱 SMS (বাংলা)**")
        st.code(n["sms_bn"], language=None)
    with c2:
        st.markdown("**💬 WhatsApp**")
        st.code(n["whatsapp"], language=None)
    meta = {"origin": selected_date_str, "horizon": forecast_horizon, "target": target_week_str,
            "stock_pct": stock_pct, "basis": "point forecast" if planning_basis == "point" else "90th percentile"}
    brief = commissioner_brief_html(selected_city, alloc_df, beds_df, transfers_df, city_df, crews_plan, meta)
    st.download_button(t("brief_dl", L), brief.encode("utf-8"),
                       file_name=f"commissioner_brief_{selected_city.replace(' ', '_')}.html", mime="text/html")
    all_sms = "\n\n".join(x["sms_en"] + "\n" + x["sms_bn"] for x in city_notices)
    st.download_button(t("sms_dl", L), all_sms.encode("utf-8"),
                       file_name=f"sms_{selected_city.replace(' ', '_')}.txt", mime="text/plain")
    with st.expander(t("brief_preview", L)):
        st.components.v1.html(brief, height=600, scrolling=True)

with tab8:
    st.subheader(t("map_title", L, d=target_week_str, h=forecast_horizon))
    st.caption(t("map_cap", L))
    scope = lang_radio(st, t("map_scope", L), ["all", "city"], "map_scope", horizontal=True,
                       format_func=lambda x: t("all_cities", L) if x == "all" else t("sel_city", L))
    map_labels = {"cases": t("m_cases", L), "fulfil": t("m_fulfil", L), "crews": t("m_crews", L)}
    fig_map = build_risk_map(alloc_df, transfers_df, map_labels, tier_names=TIER[L],
                             city_filter=selected_city if scope == "city" else None)
    st.plotly_chart(fig_map, use_container_width=True, config={"displaylogo": False})
