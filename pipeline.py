import os
import glob
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "rainfall_lag_1w",
    "rainfall_lag_2w",
    "rainfall_lag_3w",
    "rainfall_lag_4w",
    "temp_lag_1w",
    "temp_lag_2w",
    "temp_lag_3w",
    "temp_lag_4w",
    "humidity_lag_1w",
    "humidity_lag_2w",
    "humidity_lag_3w",
    "humidity_lag_4w",
    "rainfall_2_4w_sum",
    "temp_2_4w_mean",
    "humidity_2_4w_mean",
    "sin_week",
    "cos_week",
    "stagnant_breeding_potential",
]


def resolve_dataset_path(default_path: str = "bangladesh_dengue_large_dataset.csv") -> str:
    """Finds the Dengue dataset CSV automatically in the project directory."""
    candidates = [
        default_path,
        "hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    csv_files = glob.glob("*dengue*.csv")
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(
        "Dataset CSV not found. Place 'bangladesh_dengue_large_dataset.csv' in the root folder."
    )


def load_raw_data(csv_path=None) -> pd.DataFrame:
    path = resolve_dataset_path(csv_path) if csv_path else resolve_dataset_path()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    return df


def compute_ward_weather_lags(df: pd.DataFrame, drop_na_lags: bool = True) -> pd.DataFrame:
    """
    Computes grouped multi-week weather lag features per ward_id to capture
    the 2-to-4-week Aedes aegypti mosquito breeding & viral incubation cycle.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ward_id", "date"]).reset_index(drop=True)

    for lag in [1, 2, 3, 4]:
        data[f"rainfall_lag_{lag}w"] = data.groupby("ward_id")["rainfall_mm"].shift(lag)
        data[f"temp_lag_{lag}w"] = data.groupby("ward_id")["avg_temp_c"].shift(lag)
        data[f"humidity_lag_{lag}w"] = data.groupby("ward_id")["relative_humidity_pct"].shift(lag)

    # Critical 2-to-4-week breeding window aggregates
    data["rainfall_2_4w_sum"] = (
        data["rainfall_lag_2w"] + data["rainfall_lag_3w"] + data["rainfall_lag_4w"]
    )
    data["temp_2_4w_mean"] = (
        data["temp_lag_2w"] + data["temp_lag_3w"] + data["temp_lag_4w"]
    ) / 3.0
    data["humidity_2_4w_mean"] = (
        data["humidity_lag_2w"] + data["humidity_lag_3w"] + data["humidity_lag_4w"]
    ) / 3.0

    # Cyclical seasonality encoding
    week_of_year = data["date"].dt.isocalendar().week.astype(int)
    data["sin_week"] = np.sin(2.0 * np.pi * week_of_year / 52.0)
    data["cos_week"] = np.cos(2.0 * np.pi * week_of_year / 52.0)

    # Hyper-local stagnant water breeding interaction index
    data["stagnant_breeding_potential"] = (
        data["rainfall_2_4w_sum"]
        * data["water_logging_risk_index"]
        * (1.0 - data["sanitation_score"])
    )

    if drop_na_lags:
        data = data.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return data


# ---------------------------------------------------------------------------
# Final Round: leakage-free DIRECT multi-horizon early-warning features
# ---------------------------------------------------------------------------
# At a forecast origin week t, a health officer only knows data observed up to
# and including week t. Each horizon h (1..4 weeks ahead) gets its own model
# that maps origin-week information -> dengue cases in week t+h. No future
# weather or future case counts are ever used as inputs.

FORECAST_HORIZONS = [1, 2, 3, 4]

# Autoregressive surveillance features (past observed case counts, ward-level)
CASE_LAG_COLS = [
    "cases_lag_0w",       # cases observed in the origin week itself
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_4w_mean",      # mean of weeks t-3..t
    "cases_4w_max",
    "cases_trend_2w",     # cases_t - cases_{t-2}: growth signal
]

# Rainfall from weeks t-2..t becomes the 2-4w breeding window of target t+h
RECENT_RAIN_COLS = [
    "rainfall_0_2w_sum",
    "recent_breeding_potential",
]

FORECAST_FEATURE_COLS = FEATURE_COLS + RECENT_RAIN_COLS + CASE_LAG_COLS


def compute_case_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ward-isolated autoregressive case features (past/current week only)."""
    data = df.copy()
    g = data.groupby("ward_id")["dengue_cases"]
    data["cases_lag_0w"] = data["dengue_cases"].astype(float)
    for lag in [1, 2, 3]:
        data[f"cases_lag_{lag}w"] = g.shift(lag)
    roll = g.rolling(4, min_periods=4)
    data["cases_4w_mean"] = roll.mean().reset_index(level=0, drop=True)
    data["cases_4w_max"] = roll.max().reset_index(level=0, drop=True)
    data["cases_trend_2w"] = data["cases_lag_0w"] - data["cases_lag_2w"]

    data["rainfall_0_2w_sum"] = (
        data["rainfall_mm"] + data["rainfall_lag_1w"] + data["rainfall_lag_2w"]
    )
    data["recent_breeding_potential"] = (
        data["rainfall_0_2w_sum"]
        * data["water_logging_risk_index"]
        * (1.0 - data["sanitation_score"])
    )
    return data


def recompute_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Recomputes aggregate/interaction features after What-If edits to raw lags."""
    out = data.copy()
    out["rainfall_2_4w_sum"] = out["rainfall_lag_2w"] + out["rainfall_lag_3w"] + out["rainfall_lag_4w"]
    out["rainfall_0_2w_sum"] = out["rainfall_mm"] + out["rainfall_lag_1w"] + out["rainfall_lag_2w"]
    out["temp_2_4w_mean"] = (out["temp_lag_2w"] + out["temp_lag_3w"] + out["temp_lag_4w"]) / 3.0
    out["humidity_2_4w_mean"] = (
        out["humidity_lag_2w"] + out["humidity_lag_3w"] + out["humidity_lag_4w"]
    ) / 3.0
    vuln = out["water_logging_risk_index"] * (1.0 - out["sanitation_score"])
    out["stagnant_breeding_potential"] = out["rainfall_2_4w_sum"] * vuln
    out["recent_breeding_potential"] = out["rainfall_0_2w_sum"] * vuln
    return out


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Origin-week feature table (one row per ward-week = forecast origin t).
    Rows are kept even when the future target is unknown, so the latest week
    can produce a genuine forward-looking forecast.
    """
    data = compute_ward_weather_lags(df, drop_na_lags=False)
    data = compute_case_lag_features(data)
    data = data.dropna(subset=FORECAST_FEATURE_COLS).reset_index(drop=True)
    return data


def build_horizon_dataset(feat_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Attaches the h-week-ahead target to each origin row:
        target_cases = dengue_cases at week t+h (same ward)
        target_date  = date of week t+h
    Rows whose target lies beyond the dataset end are dropped (training only).
    """
    data = feat_df.sort_values(["ward_id", "date"]).copy()
    g = data.groupby("ward_id")
    data["target_cases"] = g["dengue_cases"].shift(-horizon)
    data["target_date"] = g["date"].shift(-horizon)
    data["horizon"] = horizon
    return data.dropna(subset=["target_cases", "target_date"]).reset_index(drop=True)


def classify_risk_tier(predicted_cases: float):
    """
    Maps predicted weekly ward dengue cases into actionable public health tiers.
    Returns: (tier_label, badge_color, recommended_action)
    """
    if predicted_cases < 3.0:
        return (
            "Low",
            "#16a34a",
            "Routine vector surveillance & community awareness.",
        )
    elif predicted_cases < 7.0:
        return (
            "Medium",
            "#eab308",
            "Targeted larvicide application & stagnant drain clearance.",
        )
    elif predicted_cases < 12.0:
        return (
            "High",
            "#f97316",
            "Urgent thermal fogging, hotspot containment & local clinic alert.",
        )
    else:
        return (
            "Critical",
            "#dc2626",
            "Emergency ward surge readiness, intensive vector control & hospital bed allocation.",
        )