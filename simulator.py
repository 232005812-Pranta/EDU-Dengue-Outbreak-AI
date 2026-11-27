"""
Phase 2 Requirement 3: model-driven "What-If" vector-control impact simulator.

Mechanism (counterfactual, not a fixed % rule):
  Larviciding/fogging destroys breeding in standing water, so a crew deployed at the
  forecast origin week reduces the ward's *effective breeding rainfall*. We scale the
  rainfall features the model sees (rainfall_mm, rainfall_lag_1..4w) by (1 - r(n)) where
      r(n) = 1 - (1 - efficacy) ** n          (diminishing returns per extra crew)
  recompute the derived breeding indices (stagnant/recent_breeding_potential, 0-2w and
  2-4w rainfall sums), and re-run the SAME trained horizon models for weeks +2, +3, +4.
  cases_prevented = baseline forecast - counterfactual forecast (floored at 0).
Horizon +1 is left unchanged: infections showing up next week already happened.
"""
import numpy as np
import pandas as pd

from operations import origin_rows_for, predict_horizon
from pipeline import recompute_derived_features

SIM_HORIZONS = [2, 3, 4]
RAIN_COLS = ["rainfall_mm", "rainfall_lag_1w", "rainfall_lag_2w", "rainfall_lag_3w", "rainfall_lag_4w"]


def breeding_reduction(n_crews, efficacy: float):
    return 1.0 - (1.0 - efficacy) ** np.asarray(n_crews, dtype=float)


def apply_intervention(origin_rows: pd.DataFrame, n_crews, efficacy: float) -> pd.DataFrame:
    rows = origin_rows.copy()
    keep = 1.0 - breeding_reduction(n_crews, efficacy)
    for col in RAIN_COLS:
        rows[col] = rows[col] * keep
    return recompute_derived_features(rows)


def crew_response_table(bundle, feat_df, origin_date, max_crews: int = 5, efficacy: float = 0.25) -> pd.DataFrame:
    """
    For every ward and every crew count 0..max_crews, the model-predicted cases for
    weeks +2..+4 after deployment. One batched model call per horizon (fast, offline).
    """
    base = origin_rows_for(feat_df, origin_date)
    stacked = []
    for n in range(max_crews + 1):
        rows = apply_intervention(base, n, efficacy)
        rows["crews"] = n
        stacked.append(rows)
    stacked = pd.concat(stacked, ignore_index=True)

    out = stacked[["city", "ward_id", "crews"]].copy()
    for h in SIM_HORIZONS:
        out[f"cases_h{h}"] = predict_horizon(bundle, stacked, h)["predicted_cases"].to_numpy()
    out["cases_2_4w"] = out[[f"cases_h{h}" for h in SIM_HORIZONS]].sum(axis=1)
    # Enforce monotone non-increasing response in crews (tree models can wiggle slightly)
    out = out.sort_values(["ward_id", "crews"])
    out["cases_2_4w"] = out.groupby("ward_id")["cases_2_4w"].cummin()
    base0 = out[out["crews"] == 0].set_index("ward_id")["cases_2_4w"]
    out["cases_prevented"] = (out["ward_id"].map(base0) - out["cases_2_4w"]).clip(lower=0)
    return out.reset_index(drop=True)


def simulate_plan(response: pd.DataFrame, crews_per_ward: dict, ratios: dict) -> pd.DataFrame:
    """Impact of a specific deployment plan {ward_id: crews}."""
    base = response[response["crews"] == 0].set_index("ward_id")
    rows = []
    for ward_id, b in base.iterrows():
        n = int(crews_per_ward.get(ward_id, 0))
        n = min(n, int(response["crews"].max()))
        cf = response[(response["ward_id"] == ward_id) & (response["crews"] == n)].iloc[0]
        prevented = float(cf["cases_prevented"])
        rows.append({
            "city": b["city"],
            "ward_id": ward_id,
            "crews_deployed": n,
            "baseline_cases_2_4w": round(float(b["cases_2_4w"]), 2),
            "projected_cases_2_4w": round(float(cf["cases_2_4w"]), 2),
            "cases_prevented": round(prevented, 2),
            "beds_freed": round(prevented * ratios["hospital_beds_required"], 2),
            "saline_saved": round(prevented * ratios["saline_bags_needed"], 2),
            "roi_cases_per_crew": round(prevented / n, 2) if n > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("roi_cases_per_crew", ascending=False).reset_index(drop=True)


def optimize_deployment(response: pd.DataFrame, total_crews: int) -> dict:
    """
    Greedy marginal-ROI optimiser: assign crews one at a time to the ward where the NEXT
    crew prevents the most additional cases (optimal for concave/diminishing-return curves).
    """
    curve = response.pivot(index="ward_id", columns="crews", values="cases_prevented")
    max_n = int(curve.columns.max())
    plan = {w: 0 for w in curve.index}
    for _ in range(int(total_crews)):
        best_w, best_gain = None, 0.0
        for w, n in plan.items():
            if n >= max_n:
                continue
            gain = curve.at[w, n + 1] - curve.at[w, n]
            if gain > best_gain + 1e-9:
                best_w, best_gain = w, gain
        if best_w is None:
            break
        plan[best_w] += 1
    return {w: n for w, n in plan.items() if n > 0}


def roi_ranking(response: pd.DataFrame) -> pd.DataFrame:
    """Ward ranking by first-crew ROI (cases prevented over weeks +2..+4 by one crew)."""
    one = response[response["crews"] == 1][["city", "ward_id", "cases_prevented"]]
    base = response[response["crews"] == 0].set_index("ward_id")["cases_2_4w"]
    one = one.assign(baseline_cases_2_4w=one["ward_id"].map(base).round(2),
                     roi_first_crew=one["cases_prevented"].round(2))
    return one.drop(columns="cases_prevented").sort_values("roi_first_crew", ascending=False).reset_index(drop=True)
