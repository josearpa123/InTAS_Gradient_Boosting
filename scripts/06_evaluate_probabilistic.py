#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate probabilistic validity metrics for analytic vs learned rho variants.

This script computes calibration-oriented metrics using analytic rho as
reference probability at the same aggregation grain (period, vehID).

Outputs:
  - probabilistic_validity_global.csv
  - probabilistic_validity_bins.csv
  - probabilistic_validity_report.txt
  - probabilistic_calibration_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate probabilistic validity metrics")
    p.add_argument("--analytic", default="data/theory/analytic_rho_reference.parquet")
    p.add_argument("--raw", default="data/artifacts/ml/final/rho_hat_windows_raw.parquet")
    p.add_argument("--cal", default="data/artifacts/ml/final/rho_hat_windows_calibrated.parquet")
    p.add_argument("--out-dir", default="reports/final")
    p.add_argument("--bins", type=int, default=10)
    return p.parse_args()


def clamp01(s: pd.Series) -> pd.Series:
    return s.astype(float).clip(0.0, 1.0)


def brier_score_soft(y_ref: pd.Series, p: pd.Series) -> float:
    d = pd.DataFrame({"y_ref": y_ref, "p": p}).dropna()
    if d.empty:
        return float("nan")
    return float(np.mean((d["p"] - d["y_ref"]) ** 2))


def ece_soft(y_ref: pd.Series, p: pd.Series, n_bins: int = 10) -> float:
    d = pd.DataFrame({"y_ref": y_ref, "p": p}).dropna()
    if d.empty:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(d)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (d["p"] >= lo) & (d["p"] <= hi)
        else:
            m = (d["p"] >= lo) & (d["p"] < hi)
        b = d.loc[m]
        if b.empty:
            continue
        conf = float(b["p"].mean())
        acc = float(b["y_ref"].mean())
        ece += (len(b) / n) * abs(acc - conf)
    return float(ece)


def calibration_bins(y_ref: pd.Series, p: pd.Series, variant: str, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y_ref": y_ref, "p": p}).dropna()
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows: list[dict[str, float | int | str]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (d["p"] >= lo) & (d["p"] <= hi)
        else:
            m = (d["p"] >= lo) & (d["p"] < hi)
        b = d.loc[m]
        rows.append(
            {
                "variant": variant,
                "bin": i,
                "bin_left": float(lo),
                "bin_right": float(hi),
                "n": int(len(b)),
                "pred_mean": float(b["p"].mean()) if len(b) else np.nan,
                "ref_mean": float(b["y_ref"].mean()) if len(b) else np.nan,
            }
        )

    return pd.DataFrame(rows)


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
    joined["rho_analytic"] = clamp01(joined["rho_analytic"])

    variants = ["rho_raw_mean", "rho_raw_max", "rho_cal_mean", "rho_cal_max"]
    global_rows: list[dict[str, float | str]] = []
    bins_rows: list[pd.DataFrame] = []

    for v in variants:
        p = clamp01(joined[v])
        global_rows.append(
            {
                "variant": v,
                "n": float(len(joined)),
                "brier_soft": brier_score_soft(joined["rho_analytic"], p),
                "ece_soft": ece_soft(joined["rho_analytic"], p, n_bins=args.bins),
            }
        )
        bins_rows.append(calibration_bins(joined["rho_analytic"], p, variant=v, n_bins=args.bins))

    global_df = pd.DataFrame(global_rows).sort_values("brier_soft").reset_index(drop=True)
    bins_df = pd.concat(bins_rows, ignore_index=True)

    (out_dir / "probabilistic_validity_global.csv").write_text(global_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "probabilistic_validity_bins.csv").write_text(bins_df.to_csv(index=False), encoding="utf-8")

    # Calibration curve figure
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.3, label="y = x")
    for v in variants:
        b = bins_df[(bins_df["variant"] == v) & (bins_df["n"] > 0)].copy()
        if b.empty:
            continue
        plt.plot(b["pred_mean"], b["ref_mean"], marker="o", linewidth=1.6, label=v)

    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Reference probability (analytic rho mean)")
    plt.title("Calibration Curves (Soft Reference)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "probabilistic_calibration_curve.png", dpi=180)
    plt.close()

    best = global_df.iloc[0].to_dict() if not global_df.empty else {}
    lines = [
        "=" * 72,
        "PROBABILISTIC VALIDITY (SOFT REFERENCE)",
        "=" * 72,
        "",
        "Reference: analytic rho at (period, vehID)",
        f"Rows evaluated: {len(joined):,}",
        f"Bins for ECE/calibration: {args.bins}",
        "",
        "GLOBAL METRICS",
        global_df.to_string(index=False),
        "",
        "BEST VARIANT (by Brier)",
        str(best),
        "",
        "NOTE:",
        "- These are soft calibration metrics using analytic rho as probability reference.",
        "- For strict probabilistic validation, compute Brier/ECE against observed binary events.",
        "",
        "=" * 72,
    ]
    report = "\n".join(lines)
    (out_dir / "probabilistic_validity_report.txt").write_text(report, encoding="utf-8")

    print("[OK] probabilistic validity evaluation complete")
    print(global_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
