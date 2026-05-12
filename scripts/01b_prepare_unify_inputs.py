#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare integration inputs for metrics unification.

This script bridges:
1) OMNeT KPI extraction outputs -> data/kpi/summary_kpis_avg.csv
2) SUMO silver exposure -> data/mobility_metrics.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def default_scenario_mapping() -> dict[tuple[str, str], str]:
    return {
        ("HC1_5pct", "nearest"): "SingleConnection-CBR-DL",
        ("HC1_5pct", "ceiling"): "SingleConnection-CBR-UL",
        ("HC1_10pct", "nearest"): "DoubleConnection-CBR-DL",
        ("HC1_10pct", "ceiling"): "DoubleConnection-CBR-UL",
        ("HC2_5pct", "nearest"): "SplitBearer-CBR-DL",
        ("HC2_5pct", "ceiling"): "DualConnectivity",
        ("HC2_10pct", "nearest"): "DoubleConnection-CBR-DL",
        ("HC2_10pct", "ceiling"): "DoubleConnection-CBR-UL",
        ("HC3_5pct", "nearest"): "SingleConnection-CBR-DL",
        ("HC3_5pct", "ceiling"): "SingleConnection-CBR-UL",
        ("HC3_10pct", "nearest"): "SplitBearer-CBR-DL",
        ("HC3_10pct", "ceiling"): "DualConnectivity",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KPI summary and mobility metrics for step 02")
    p.add_argument("--omnet-raw", default="reports/final/objetivo2/kpis_omnet_raw.csv")
    p.add_argument("--cell-exposure", default="data/silver/cell_exposure.parquet")
    p.add_argument("--route-label", default="data/silver/route_label.parquet")
    p.add_argument("--kpi-summary-out", default="data/kpi/summary_kpis_avg.csv")
    p.add_argument("--mobility-out", default="data/mobility_metrics.parquet")
    p.add_argument("--scenario-source", default="baseline", choices=["baseline", "learned", "all"])
    return p.parse_args()


def build_kpi_summary(raw_csv: Path, output_csv: Path, scenario_source: str) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing OMNeT raw KPI CSV: {raw_csv}")

    raw = pd.read_csv(raw_csv)
    required = {"period", "policy", "scenario"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"OMNeT raw KPI file missing columns: {sorted(missing)}")

    if scenario_source in {"baseline", "learned"}:
        raw = raw[raw["scenario"].astype(str) == scenario_source].copy()
    if raw.empty:
        raise ValueError("No OMNeT KPI rows available after scenario filtering")

    mapping = default_scenario_mapping()
    raw["period"] = raw["period"].astype(str)
    raw["policy"] = raw["policy"].astype(str)
    raw["scenario"] = raw.apply(
        lambda r: mapping.get((r["period"], r["policy"]), str(r["scenario"])),
        axis=1,
    )

    candidate_metrics = [
        "cdr_global",
        "cdr_proxy_global",
        "throughput_global",
        "throughput_generated_global",
        "sinr_global",
        "cqi_global",
        "handover_proxy",
        "packet_loss_rate",
        "throughput_per_cell_mean",
        "packet_loss_per_cell_mean",
        "num_cells_sampled",
    ]
    metric_cols = [c for c in candidate_metrics if c in raw.columns]
    if not metric_cols:
        raise ValueError("No supported KPI metric columns found in OMNeT raw CSV")

    long_df = raw[["scenario"] + metric_cols].melt(
        id_vars=["scenario"], value_vars=metric_cols, var_name="kpi", value_name="value"
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        raise ValueError("KPI summary is empty after numeric cleanup")

    summary = (
        long_df.groupby(["scenario", "kpi"], as_index=False)["value"]
        .agg(count="count", avg="mean", sum="sum", min="min", max="max")
        .sort_values(["scenario", "kpi"])
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def build_mobility_metrics(cell_exposure_parquet: Path, route_label_parquet: Path, output_parquet: Path) -> pd.DataFrame:
    if not cell_exposure_parquet.exists():
        raise FileNotFoundError(f"Missing cell exposure parquet: {cell_exposure_parquet}")
    if not route_label_parquet.exists():
        raise FileNotFoundError(f"Missing route label parquet: {route_label_parquet}")

    exposure = pd.read_parquet(cell_exposure_parquet)
    labels = pd.read_parquet(route_label_parquet)

    needed_exposure = {"run_id", "vehID", "cell_id", "time_s"}
    missing_exposure = needed_exposure.difference(exposure.columns)
    if missing_exposure:
        raise ValueError(f"cell_exposure missing columns: {sorted(missing_exposure)}")

    needed_labels = {"run_id", "vehID"}
    missing_labels = needed_labels.difference(labels.columns)
    if missing_labels:
        raise ValueError(f"route_label missing columns: {sorted(missing_labels)}")

    meta_cols = [c for c in ["run_id", "vehID", "period", "policy", "rep"] if c in labels.columns]
    meta = labels[meta_cols].drop_duplicates(subset=["run_id", "vehID"])

    df = exposure.merge(meta, on=["run_id", "vehID"], how="left")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce").fillna(0.0)
    df["cell_id"] = df["cell_id"].astype(str)

    group_cols = [c for c in ["run_id", "period", "policy", "rep", "cell_id"] if c in df.columns]
    if "run_id" not in group_cols or "cell_id" not in group_cols:
        raise ValueError("Unable to build mobility metrics: missing run_id/cell_id grouping keys")

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            duration_in_cell_s=("time_s", "sum"),
            samples_in_cell=("time_s", "count"),
            unique_vehicles=("vehID", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(output_parquet, index=False)
    return agg


def _raw_csv_is_empty(path: Path) -> bool:
    """Return True if the CSV file is missing, empty, or has 0 data rows."""
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
        return df.empty
    except Exception:
        return True


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.omnet_raw)
    cell_exposure = Path(args.cell_exposure)
    route_label = Path(args.route_label)
    kpi_out = Path(args.kpi_summary_out)
    mobility_out = Path(args.mobility_out)

    # Graceful fallback: if OMNeT raw KPIs are empty (no OMNeT run) but
    # pre-computed seed outputs already exist, reuse them as-is.
    if _raw_csv_is_empty(raw_csv) and kpi_out.exists() and mobility_out.exists():
        print("[WARN] OMNeT raw KPI file is empty (OMNeT not executed).")
        print("[WARN] Reusing existing seed outputs for KPI summary and mobility metrics.")
        print(f"  kpi_summary : {kpi_out}")
        print(f"  mobility    : {mobility_out}")
        return 0

    summary = build_kpi_summary(raw_csv, kpi_out, args.scenario_source)
    mobility = build_mobility_metrics(cell_exposure, route_label, mobility_out)

    print("=" * 72)
    print("Preparación de insumos para unificación completada")
    print(f"KPI summary   : {kpi_out} (rows={len(summary):,})")
    print(f"Mobility table: {mobility_out} (rows={len(mobility):,})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
