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