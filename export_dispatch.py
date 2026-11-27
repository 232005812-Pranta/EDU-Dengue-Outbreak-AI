"""
Generates the sample dispatch exports in ./exports (SMS, WhatsApp, printable HTML briefs, CSVs).
Runs 100% offline.

Usage:
    python export_dispatch.py
    python export_dispatch.py --origin 2025-07-14 --horizon 3 --stock 60 --crews 40
"""
import argparse
import json
import os

import joblib
import pandas as pd

from operations import learn_supply_ratios, run_operations
from pipeline import build_forecast_features, load_raw_data
from simulator import crew_response_table, optimize_deployment, simulate_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2025-07-14", help="forecast origin week (YYYY-MM-DD, a Monday)")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--stock", type=int, default=60, help="available regional stock, %% of demand (30-100)")
    ap.add_argument("--crews", type=int, default=40, help="total spraying crews")
    ap.add_argument("--out", default="exports")
    args = ap.parse_args()

    bundle = joblib.load(os.path.join("models", "dengue_model_bundle.joblib"))
    raw = load_raw_data()
    feat = build_forecast_features(raw)

    ops = run_operations(bundle, raw, feat, args.origin, args.horizon, args.stock / 100.0, args.crews)
    response = crew_response_table(bundle, feat, args.origin)
    plan = optimize_deployment(response, args.crews)
    sim = simulate_plan(response, plan, learn_supply_ratios(raw))

    from dispatch import write_exports
    target = pd.Timestamp(args.origin) + pd.Timedelta(weeks=args.horizon)
    meta = {"origin": args.origin, "horizon": args.horizon, "target": target.strftime("%Y-%m-%d"),
            "stock_pct": args.stock, "basis": "point forecast"}
    files = write_exports(args.out, ops, plan, meta)
    sim.to_csv(os.path.join(args.out, "vector_control_plan.csv"), index=False)
    with open(os.path.join(args.out, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "supply_ratios": ops["ratios"],
                   "cases_prevented_2_4w": round(float(sim["cases_prevented"].sum()), 2)}, f, indent=2)
    files += ["vector_control_plan.csv", "run_metadata.json"]
    print(f"Wrote {len(files)} files to ./{args.out}: " + ", ".join(files))


if __name__ == "__main__":
    main()
