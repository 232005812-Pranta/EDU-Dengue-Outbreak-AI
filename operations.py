"""
Phase 2 operations layer: turns leakage-free ward forecasts into supply demand,
constrained (scarcity-aware) allocations and intra-city hospital overflow routing.

Everything here is deterministic, local and offline (pandas/numpy only).
"""
import math

import numpy as np
import pandas as pd

from pipeline import classify_risk_tier

SPLIT_DATE = pd.Timestamp("2024-01-01")

SUPPLY_COLS = ["saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required"]
SUPPLY_LABELS = {
    "saline_bags_needed": "Saline bags",
    "diagnostic_kits_needed": "Diagnostic kits",
    "hospital_beds_required": "Hospital beds",
}

# Fallback per-case ratios, used ONLY if the dataset lacks the supply columns.
# (Values learned from the onsite dataset; kept so the engine never fails offline.)
DEFAULT_SUPPLY_RATIOS = {
    "saline_bags_needed": 2.385,
    "diagnostic_kits_needed": 1.165,
    "hospital_beds_required": 0.297,
}

TIER_WEIGHT = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}
# Spraying crews a ward needs at each risk tier (vector-control SOP style rule).
CREW_NEED = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


# ---------------------------------------------------------------------------
# 1. Data-derived planning parameters
# ---------------------------------------------------------------------------
def learn_supply_ratios(raw_df: pd.DataFrame, split_date: pd.Timestamp = SPLIT_DATE) -> dict:
    """
    Learns supplies-per-case ratios from the TRAINING period only (no test leakage),
    via least squares through the origin: ratio = sum(cases*supply) / sum(cases^2).
    """
    train = raw_df[raw_df["date"] < split_date]
    cases = train["dengue_cases"].to_numpy(dtype=float)
    ratios = {}
    for col in SUPPLY_COLS:
        if col in train.columns and (cases ** 2).sum() > 0:
            y = train[col].to_numpy(dtype=float)
            ratios[col] = float((cases * y).sum() / (cases ** 2).sum())
        else:
            ratios[col] = DEFAULT_SUPPLY_RATIOS[col]
    return ratios


def estimate_bed_capacity(
    raw_df: pd.DataFrame,
    split_date: pd.Timestamp = SPLIT_DATE,
    quantile: float = 0.75,
    min_beds: int = 1,
) -> pd.Series:
    """
    The dataset has no bed-capacity column, so capacity is derived transparently:
      1) City pool size = sum over the city's wards of the `quantile` (default 75th pct)
         of weekly beds required in past monsoon seasons (Jun-Oct, training years only)
         -> each city is staffed for a 'normal busy monsoon week', not for outbreak peaks.
      2) That city pool is distributed across wards in proportion to population density
         (hospitals follow population, not mosquito breeding risk).
    So water-logged, high-risk wards can exceed local beds while denser low-risk wards
    keep headroom - exactly the situation that needs intra-city overflow routing.
    """
    train = raw_df[(raw_df["date"] < split_date) & (raw_df["date"].dt.month.between(6, 10))]
    if "hospital_beds_required" in train.columns:
        beds = train["hospital_beds_required"]
    else:
        beds = pd.Series(
            _round_half_up(train["dengue_cases"] * DEFAULT_SUPPLY_RATIOS["hospital_beds_required"]),
            index=train.index,
        )
    ward_q = beds.groupby(train["ward_id"]).quantile(quantile)
    wards = raw_df.drop_duplicates("ward_id").set_index("ward_id")[["city", "population_density_sqkm"]]
    wards["ward_q"] = ward_q
    cap = {}
    for _, grp in wards.groupby("city"):
        pool = float(grp["ward_q"].sum())
        share = grp["population_density_sqkm"] / grp["population_density_sqkm"].sum()
        for ward_id, s in share.items():
            cap[ward_id] = max(min_beds, int(round(pool * s)))
    return pd.Series(cap, name="bed_capacity").astype(int)


# ---------------------------------------------------------------------------
# 2. City-wide (all 60 wards) leakage-free forecast for one origin week
# ---------------------------------------------------------------------------
def predict_horizon(bundle: dict, origin_rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    models = bundle["horizon_models"][horizon]
    X = origin_rows[bundle["forecast_feature_cols"]]
    pt = np.clip(models["point_model"].predict(X), 0.0, None)
    lo = np.minimum(pt, np.clip(models["lower_model"].predict(X), 0.0, None))
    hi = np.maximum(pt, np.clip(models["upper_model"].predict(X), 0.0, None))
    out = origin_rows[["date", "city", "ward_id", "population_density_sqkm",
                       "water_logging_risk_index", "sanitation_score"]].copy()
    out["origin_date"] = out["date"]
    out["horizon"] = horizon
    out["target_date"] = out["date"] + pd.Timedelta(weeks=horizon)
    out["predicted_cases"] = pt
    out["ci_lower"] = lo
    out["ci_upper"] = hi
    return out.drop(columns=["date"]).reset_index(drop=True)


def origin_rows_for(feat_df: pd.DataFrame, origin_date) -> pd.DataFrame:
    rows = feat_df[feat_df["date"] == pd.Timestamp(origin_date)]
    if rows.empty:
        raise ValueError(f"No feature rows for origin week {origin_date}")
    return rows.sort_values(["city", "ward_id"]).reset_index(drop=True)


def forecast_all_wards(bundle, feat_df, origin_date, horizon, planning_basis="point") -> pd.DataFrame:
    """
    planning_basis: 'point' = expected cases, 'upper' = 90th-percentile (safety-stock) plan.
    """
    fc = predict_horizon(bundle, origin_rows_for(feat_df, origin_date), horizon)
    fc["planning_cases"] = fc["ci_upper"] if planning_basis == "upper" else fc["predicted_cases"]
    tiers = [classify_risk_tier(v) for v in fc["predicted_cases"]]
    fc["risk_tier"] = [t[0] for t in tiers]
    fc["recommended_action"] = [t[2] for t in tiers]
    return fc


# ---------------------------------------------------------------------------
# 3. Demand + constrained allocation (Requirement 1)
# ---------------------------------------------------------------------------
def compute_demand(fc: pd.DataFrame, ratios: dict) -> pd.DataFrame:
    df = fc.copy()
    for col in SUPPLY_COLS:
        df[col] = _round_half_up(df["planning_cases"] * ratios[col])
    df["crews_required"] = df["risk_tier"].map(CREW_NEED).astype(int)
    density_norm = df["population_density_sqkm"] / df["population_density_sqkm"].median()
    df["priority_score"] = (
        df["risk_tier"].map(TIER_WEIGHT) * density_norm * (1.0 + df["planning_cases"] / 10.0)
    ).round(3)
    return df


def _greedy_fill(order_idx, demand: pd.Series, stock: int) -> pd.Series:
    alloc = pd.Series(0, index=demand.index, dtype=int)
    remaining = int(stock)
    for i in order_idx:
        give = min(int(demand.at[i]), remaining)
        alloc.at[i] = give
        remaining -= give
        if remaining <= 0:
            break
    return alloc


def _weighted_fill(demand: pd.Series, weight: pd.Series, stock: int) -> pd.Series:
    """
    Priority-weighted water-filling: stock is shared in proportion to priority x demand,
    no ward gets more than it needs, and any surplus is re-shared among the rest.
    Integer units via largest-remainder rounding. Total allocated == min(stock, total demand).
    """
    demand = demand.astype(float)
    alloc = pd.Series(0.0, index=demand.index)
    remaining = float(min(stock, demand.sum()))
    active = demand[demand > 0].index
    while remaining > 1e-9 and len(active):
        w = (weight[active] * demand[active]).clip(lower=1e-9)
        share = remaining * w / w.sum()
        room = demand[active] - alloc[active]
        give = np.minimum(share, room)
        alloc[active] += give
        remaining -= give.sum()
        active = room[room - give > 1e-9].index
        if (share <= room).all():
            break
    base = np.floor(alloc).astype(int)
    leftover = int(round(min(stock, demand.sum()))) - int(base.sum())
    frac = (alloc - base).sort_values(ascending=False)
    for i in frac.index[: max(leftover, 0)]:
        if base[i] < demand[i]:
            base[i] += 1
    return base.astype(int)


def allocate_constrained(
    demand_df: pd.DataFrame,
    stock_pct: float,
    total_crews: int,
    pool_by: str = "city",
    mode: str = "weighted",
) -> pd.DataFrame:
    """
    Scarcity-aware allocation.
    - Each city's regional depot holds stock_pct x (that city's total demand) of every supply.
    - mode="weighted" (default): priority-weighted water-filling - every ward in need gets a
      share, high-risk / high-density wards get proportionally more (higher fulfilment %).
    - mode="triage": strict priority order - a ward is fully served before the next one.
    - priority_score = risk-tier weight x density (vs median) x (1 + cases/10).
    - Spraying crews (one region-wide pool) go to wards by the same priority up to crews_required.
    Every ward whose allocation < requirement is flagged with a supply-deficit warning.
    """
    df = demand_df.copy()
    order = df.sort_values(["priority_score", "planning_cases"], ascending=False).index
    groups = df.groupby(pool_by).groups if pool_by else {"ALL": df.index}

    for col in SUPPLY_COLS:
        alloc = pd.Series(0, index=df.index, dtype=int)
        df[f"{col}_stock"] = 0
        for _, idx in groups.items():
            idx = pd.Index(idx)
            stock = int(math.floor(stock_pct * df.loc[idx, col].sum()))
            df.loc[idx, f"{col}_stock"] = stock
            if mode == "triage":
                pool_order = [i for i in order if i in idx]
                alloc.loc[idx] = _greedy_fill(pool_order, df.loc[idx, col], stock).loc[idx]
            else:
                alloc.loc[idx] = _weighted_fill(df.loc[idx, col], df.loc[idx, "priority_score"], stock)
        df[f"{col}_allocated"] = alloc
        df[f"{col}_deficit"] = df[col] - alloc

    df["crews_allocated"] = _greedy_fill(order, df["crews_required"], int(total_crews))
    df["crews_deficit"] = df["crews_required"] - df["crews_allocated"]

    req_total = df[SUPPLY_COLS].sum(axis=1)
    alloc_total = df[[f"{c}_allocated" for c in SUPPLY_COLS]].sum(axis=1)
    df["fulfillment_pct"] = np.where(req_total > 0, 100.0 * alloc_total / req_total.replace(0, 1), 100.0).round(1)
    df["supply_deficit"] = df[[f"{c}_deficit" for c in SUPPLY_COLS]].sum(axis=1) > 0
    df["crew_shortfall"] = df["crews_deficit"] > 0
    df["critical_deficit"] = df["supply_deficit"] & df["risk_tier"].isin(["High", "Critical"])
    df["priority_rank"] = df["priority_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("priority_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Intra-city hospital overflow routing (Requirement 2)
# ---------------------------------------------------------------------------
def route_overflow(df: pd.DataFrame, bed_capacity: pd.Series, capacity_multiplier: float = 1.0):
    """
    Within each city pool: wards whose beds_required exceed local capacity send overflow
    patients to the wards with the most free beds (lowest utilisation) in the SAME city.
    Largest deficits are routed first; transfers never cross city boundaries.
    Returns (ward_table, transfers, city_summary).
    """
    w = df[["city", "ward_id", "risk_tier", "predicted_cases", "hospital_beds_required"]].copy()
    w["bed_capacity"] = (
        w["ward_id"].map(bed_capacity).fillna(2).astype(float) * capacity_multiplier
    ).round().clip(lower=1).astype(int)
    w["local_deficit"] = (w["hospital_beds_required"] - w["bed_capacity"]).clip(lower=0)
    w["local_headroom"] = (w["bed_capacity"] - w["hospital_beds_required"]).clip(lower=0)
    w["transferred_out"] = 0
    w["received_in"] = 0

    transfers = []
    for city, grp in w.groupby("city"):
        need = grp["local_deficit"].copy()
        spare = grp["local_headroom"].copy()
        for src in need.sort_values(ascending=False).index:
            while need.at[src] > 0 and spare.sum() > 0:
                # receiver = ward with the most free beds (ties: lower predicted burden)
                cands = grp.loc[spare[spare > 0].index]
                dst = cands.assign(sp=spare[cands.index]).sort_values(
                    ["sp", "predicted_cases"], ascending=[False, True]).index[0]
                moved = int(min(need.at[src], spare.at[dst]))
                need.at[src] -= moved
                spare.at[dst] -= moved
                w.at[src, "transferred_out"] += moved
                w.at[dst, "received_in"] += moved
                transfers.append({
                    "city": city,
                    "from_ward": w.at[src, "ward_id"],
                    "to_ward": w.at[dst, "ward_id"],
                    "patients": moved,
                })

    w["beds_occupied_after_routing"] = (
        w["hospital_beds_required"] - w["transferred_out"] + w["received_in"]
    ).clip(upper=w["bed_capacity"])
    w["net_unresolved_deficit"] = (w["local_deficit"] - w["transferred_out"]).clip(lower=0)
    w["utilization_pct"] = (100.0 * w["hospital_beds_required"] / w["bed_capacity"]).round(1)

    city = w.groupby("city").agg(
        wards=("ward_id", "count"),
        beds_required=("hospital_beds_required", "sum"),
        bed_capacity=("bed_capacity", "sum"),
        patients_rerouted=("transferred_out", "sum"),
        unresolved_deficit=("net_unresolved_deficit", "sum"),
    ).reset_index()
    city["city_utilization_pct"] = (100.0 * city["beds_required"] / city["bed_capacity"]).round(1)
    city["remaining_headroom"] = (city["bed_capacity"] - city["beds_required"]).clip(lower=0)
    city["status"] = np.where(
        city["unresolved_deficit"] > 0, "CITY OVER CAPACITY",
        np.where(city["patients_rerouted"] > 0, "Balanced via routing", "Within capacity"),
    )
    return w, pd.DataFrame(transfers, columns=["city", "from_ward", "to_ward", "patients"]), city


def run_operations(bundle, raw_df, feat_df, origin_date, horizon, stock_pct, total_crews,
                   planning_basis="point", capacity_multiplier=1.0, capacity_quantile=0.75,
                   allocation_mode="weighted"):
    """One call that produces the full operational picture (used by app, exports and tests)."""
    ratios = learn_supply_ratios(raw_df)
    fc = forecast_all_wards(bundle, feat_df, origin_date, horizon, planning_basis)
    alloc = allocate_constrained(compute_demand(fc, ratios), stock_pct, total_crews, mode=allocation_mode)
    wards, transfers, city = route_overflow(alloc, estimate_bed_capacity(raw_df, quantile=capacity_quantile), capacity_multiplier)
    return {"ratios": ratios, "allocation": alloc, "beds": wards, "transfers": transfers, "city": city}
