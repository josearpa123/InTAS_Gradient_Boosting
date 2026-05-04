#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2.2 - Build unified metrics table (mobility + OMNeT++ KPIs).

Inputs:
  - data/mobility_metrics.parquet
  - data/kpi/summary_kpis_avg.csv

Output:
  - data/unified_metrics.parquet

Notes:
  - KPI table is converted from long to wide using avg/sum/min/max/count per KPI.
  - Scenario mapping can be overridden by CSV if needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def default_scenario_mapping() -> dict[tuple[str, str], str]:
    """Default mapping (period, policy) -> OMNeT++ scenario name.

    This mapping can be replaced with --mapping-csv when a project-specific
    one-to-one calibration is available.
    """
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


def load_mapping(mapping_csv: str | None) -> dict[tuple[str, str], str]:
    if not mapping_csv:
        return default_scenario_mapping()

    m = pd.read_csv(mapping_csv)
    needed = {"period", "policy", "scenario"}
    missing = needed.difference(m.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {sorted(missing)}")

    return {(str(r.period), str(r.policy)): str(r.scenario) for _, r in m.iterrows()}


def kpi_long_to_wide(kpi_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long KPI summary to wide table by scenario."""
    required = {"scenario", "kpi", "count", "avg", "sum", "min", "max"}
    missing = required.difference(kpi_long.columns)
    if missing:
        raise ValueError(f"KPI summary missing columns: {sorted(missing)}")

    rows = []
    for scenario, grp in kpi_long.groupby("scenario", sort=True):
        out = {"scenario": scenario}
        for _, rec in grp.iterrows():
            kpi = str(rec["kpi"])
            out[f"network_{kpi}_count"] = rec["count"]
            out[f"network_{kpi}_avg"] = rec["avg"]
            out[f"network_{kpi}_sum"] = rec["sum"]
            out[f"network_{kpi}_min"] = rec["min"]
            out[f"network_{kpi}_max"] = rec["max"]
        rows.append(out)

    wide = pd.DataFrame(rows)
    cols = ["scenario"] + sorted([c for c in wide.columns if c != "scenario"])
    return wide[cols]


def build_unified_table(
    mobility_path: str,
    kpi_summary_path: str,
    output_path: str,
    mapping_csv: str | None = None,
) -> pd.DataFrame:
    mobility = pd.read_parquet(mobility_path)
    kpi_long = pd.read_csv(kpi_summary_path)

    mapping = load_mapping(mapping_csv)

    mobility = mobility.copy()
    mobility["period"] = mobility["period"].astype(str)
    mobility["policy"] = mobility["policy"].astype(str)
    mobility["scenario"] = mobility.apply(
        lambda r: mapping.get((r["period"], r["policy"]), "unknown"), axis=1
    )

    kpi_wide = kpi_long_to_wide(kpi_long)

    unified = mobility.merge(kpi_wide, on="scenario", how="left")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    unified.to_parquet(output_path, index=False)

    return unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join mobility metrics with KPI summaries")
    parser.add_argument(
        "--mobility",
        default="data/mobility_metrics.parquet",
        help="Mobility metrics parquet",
    )
    parser.add_argument(
        "--kpi-summary",
        default="data/kpi/summary_kpis_avg.csv",
        help="Long-format KPI summary CSV",
    )
    parser.add_argument(
        "--output",
        default="data/unified_metrics.parquet",
        help="Unified metrics output parquet",
    )
    parser.add_argument(
        "--mapping-csv",
        default=None,
        help="Optional CSV with columns: period,policy,scenario",
    )
    parser.add_argument(
        "--allow-precomputed-output",
        action="store_true",
        default=True,
        help=(
            "If inputs are missing but --output already exists, keep the existing file "
            "and continue (default: enabled)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mobility_path = Path(args.mobility)
    kpi_path = Path(args.kpi_summary)
    output_path = Path(args.output)

    print("=" * 72)
    print("Phase 2.2 - Metrics Unification")
    print("=" * 72)
    print(f"Mobility input: {mobility_path.resolve()}")
    print(f"KPI input     : {kpi_path.resolve()}")
    if args.mapping_csv:
        print(f"Mapping CSV   : {Path(args.mapping_csv).resolve()}")
    print(f"Output        : {output_path.resolve()}")

    if (not mobility_path.exists() or not kpi_path.exists()) and args.allow_precomputed_output:
        if output_path.exists():
            pre = pd.read_parquet(output_path)
            print("\n[WARN] Missing fresh inputs for unification.")
            print("[WARN] Reusing precomputed unified metrics file.")
            print(f"Rows: {len(pre):,}")
            print(f"Columns: {len(pre.columns):,}")
            print("=" * 72)
            return 0

    unified = build_unified_table(
        mobility_path=args.mobility,
        kpi_summary_path=args.kpi_summary,
        output_path=args.output,
        mapping_csv=args.mapping_csv,
    )

    print("\nSummary")
    print(f"Rows: {len(unified):,}")
    print(f"Columns: {len(unified.columns):,}")
    print(f"Runs: {unified['run_id'].nunique():,}")
    print(f"Scenarios mapped: {unified['scenario'].nunique():,}")
    unknown = int((unified["scenario"] == "unknown").sum())
    print(f"Rows with scenario='unknown': {unknown:,}")
    print("=" * 72)
    print("[OK] unified_metrics.parquet generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
