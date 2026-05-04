#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare analytic rho against recomputed learned probabilities (raw + calibrated).

Inputs:
  - data/theory/analytic_rho_reference.parquet
  - data/artifacts/ml/final/rho_hat_windows_raw.parquet
  - data/artifacts/ml/final/rho_hat_windows_calibrated.parquet

Outputs:
  - reports/final/rho_recomputed_agg_period_veh.csv
  - reports/final/rho_compare_recomputed_global.csv
  - reports/final/rho_compare_recomputed_by_period.csv
  - reports/final/rho_compare_recomputed_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    err = d["y_pred"] - d["y_true"]
    return {
        "n": float(len(d)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "corr": float(d["y_true"].corr(d["y_pred"])) if len(d) > 1 else np.nan,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare analytic rho vs recomputed learned rho")
    p.add_argument("--analytic", default="data/theory/analytic_rho_reference.parquet")
    p.add_argument("--raw", default="data/artifacts/ml/final/rho_hat_windows_raw.parquet")
    p.add_argument("--cal", default="data/artifacts/ml/final/rho_hat_windows_calibrated.parquet")
    p.add_argument("--out-dir", default="reports/final")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = pd.read_parquet(args.analytic)[["period", "vehID", "rho"]].rename(columns={"rho": "rho_analytic"})
    r = pd.read_parquet(args.raw)[["period", "vehID", "rho_hat"]].rename(columns={"rho_hat": "rho_raw"})
    c = pd.read_parquet(args.cal)[["period", "vehID", "rho_hat"]].rename(columns={"rho_hat": "rho_cal"})

    for df in (a, r, c):
        df["period"] = df["period"].astype(str)
        df["vehID"] = df["vehID"].astype(str)

    # Aggregate learned outputs at analytic granularity (period, vehID)
    agg = (
        r.merge(c, on=["period", "vehID"], how="inner")
         .groupby(["period", "vehID"], as_index=False)
         .agg(
             rho_raw_mean=("rho_raw", "mean"),
             rho_raw_max=("rho_raw", "max"),
             rho_cal_mean=("rho_cal", "mean"),
             rho_cal_max=("rho_cal", "max"),
         )
    )

    joined = a.merge(agg, on=["period", "vehID"], how="inner")

    variants = ["rho_raw_mean", "rho_raw_max", "rho_cal_mean", "rho_cal_max"]

    global_rows = []
    for v in variants:
        global_rows.append({"variant": v, **metrics(joined["rho_analytic"], joined[v])})
    global_df = pd.DataFrame(global_rows).sort_values("mae").reset_index(drop=True)

    by_period_rows = []
    for (period, grp) in joined.groupby("period"):
        for v in variants:
            by_period_rows.append({"period": period, "variant": v, **metrics(grp["rho_analytic"], grp[v])})
    by_period_df = pd.DataFrame(by_period_rows).sort_values(["period", "variant"]).reset_index(drop=True)

    # Export
    (out_dir / "rho_recomputed_agg_period_veh.csv").write_text(agg.to_csv(index=False), encoding="utf-8")
    (out_dir / "rho_compare_recomputed_global.csv").write_text(global_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "rho_compare_recomputed_by_period.csv").write_text(by_period_df.to_csv(index=False), encoding="utf-8")

    lines = []
    lines.append("=" * 72)
    lines.append("RHO COMPARISON (RECOMPUTED): ANALYTIC VS LEARNED")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"analytic rows: {len(a):,}")
    lines.append(f"agg learned rows (period,vehID): {len(agg):,}")
    lines.append(f"joined rows: {len(joined):,}")
    lines.append("")
    lines.append("GLOBAL")
    lines.append(global_df.to_string(index=False))
    lines.append("")
    lines.append("BY PERIOD")
    lines.append(by_period_df.to_string(index=False))
    lines.append("")
    lines.append("BEST VARIANT (global by MAE)")
    lines.append(str(global_df.iloc[0].to_dict()))
    lines.append("")
    lines.append("=" * 72)
    report = "\n".join(lines)
    (out_dir / "rho_compare_recomputed_report.txt").write_text(report, encoding="utf-8")

    print("[OK] comparison recomputed complete")
    print(global_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
