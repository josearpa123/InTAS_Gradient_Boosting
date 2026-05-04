#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract OMNeT++ KPIs for Objetivo 2.

Supported layouts:
1. Paired runs for impact comparison:
   data/omnet_results/{baseline,learned}/<run_id>/{results.sca,results.vec}
2. Legacy flat inventory:
   data/omnet_sca/*.sca

Outputs:
  - reports/final/objetivo2/kpis_omnet_raw.csv
  - reports/final/objetivo2/kpis_omnet_by_cell.csv
  - reports/final/objetivo2/kpis_omnet_inventory_report.txt
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data/omnet_results"
DEFAULT_LEGACY_DIR = REPO_ROOT / "data/omnet_sca"
DEFAULT_OUT_DIR = REPO_ROOT / "reports/final/objetivo2"

CELL_RE = re.compile(r"(masterEnb\d+|secondaryGnb\d+)")
RUN_RE_LIST = [
    re.compile(r"^(HC[123]_\d+pct)__([a-zA-Z0-9]+)__rep(\d+)$"),
    re.compile(r"^run_(HC[123])_(\d+)_([a-zA-Z0-9]+)_(\d+)(?:_.*)?$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OMNeT KPIs for Objective 2")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--legacy-dir", default=str(DEFAULT_LEGACY_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_run_components(run_id: str) -> dict[str, object]:
    for pattern in RUN_RE_LIST:
        match = pattern.match(run_id)
        if not match:
            continue
        if run_id.startswith("run_"):
            hc = match.group(1)
            vfh = int(match.group(2))
            policy = match.group(3)
            rep = int(match.group(4))
            return {
                "period": f"{hc}_{vfh}pct",
                "hc": hc,
                "vfh": vfh,
                "policy": policy,
                "rep": rep,
            }
        period = match.group(1)
        policy = match.group(2)
        rep = int(match.group(3))
        hc, vfh_txt = period.split("_", 1)
        return {
            "period": period,
            "hc": hc,
            "vfh": int(vfh_txt.replace("pct", "")),
            "policy": policy,
            "rep": rep,
        }
    return {
        "period": None,
        "hc": None,
        "vfh": None,
        "policy": None,
        "rep": None,
    }


def read_sca_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("scalar "):
                continue
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            module = parts[1]
            metric_full = parts[2]
            value = safe_float(parts[3])
            if ":" in metric_full:
                name, attr = metric_full.split(":", 1)
            else:
                name, attr = metric_full, None
            rows.append(
                {
                    "module": module,
                    "name": name,
                    "attr": attr,
                    "value": value,
                }
            )
    # Keep a stable schema even when no scalar rows are found.
    return pd.DataFrame(rows, columns=["module", "name", "attr", "value"])


def metric_value(
    df: pd.DataFrame,
    name: str,
    attr: str | None = None,
    module_regex: str | None = None,
    agg: str = "mean",
) -> float:
    required_cols = {"name", "attr", "module", "value"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        return float("nan")
    sub = df[df["name"] == name].copy()
    if sub.empty:
        return float("nan")
    if attr is not None:
        sub = sub[sub["attr"] == attr]
        if sub.empty:
            return float("nan")
    if module_regex:
        pattern = re.compile(module_regex, flags=re.IGNORECASE)
        mask = sub["module"].astype(str).map(lambda value: bool(pattern.search(value)))
        sub = sub.loc[mask.astype(bool)]
        if sub.empty:
            return float("nan")
    values = pd.to_numeric(sub["value"], errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    if agg == "sum":
        return float(values.sum())
    return float(values.mean())


def first_non_nan(values: list[float]) -> float:
    for value in values:
        if not math.isnan(value):
            return value
    return float("nan")


def explicit_cdr_value(df: pd.DataFrame) -> tuple[float, str]:
    cdr_names = [
        "cdr",
        "callDropRate",
        "callDropRatio",
        "dropRate",
        "sessionDropRate",
        "hoFailureRate",
        "handoverFailureRate",
    ]
    attrs = [None, "mean", "value", "simu5g_rateavg"]
    for name in cdr_names:
        for attr in attrs:
            val = metric_value(df, name=name, attr=attr)
            if not math.isnan(val):
                label = f"explicit:{name}" if attr is None else f"explicit:{name}:{attr}"
                return float(val), label

    # Fallback heuristic: any scalar name that looks like CDR/call-drop.
    if {"name", "value"}.issubset(df.columns):
        mask = df["name"].astype(str).str.contains(r"cdr|call.?drop|drop.?rate", case=False, regex=True)
        sub = pd.to_numeric(df.loc[mask, "value"], errors="coerce").dropna()
        if len(sub):
            return float(sub.mean()), "explicit:heuristic-pattern"

    return float("nan"), "missing"


def cell_kpis(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "throughput_dl": ("macCellThroughputDl", "simu5g_rateavg"),
        "throughput_ul": ("macCellThroughputUl", "simu5g_rateavg"),
        "packet_loss_dl": ("macCellPacketLossDl", "mean"),
        "packet_loss_ul": ("macCellPacketLossUl", "mean"),
        "prb_usage_dl": ("avgServedBlocksDl", "mean"),
        "prb_usage_ul": ("avgServedBlocksUl", "mean"),
    }
    modules = sorted({m for m in df["module"].dropna().astype(str) if CELL_RE.search(m)})
    for module in modules:
        cell_match = CELL_RE.search(module)
        if not cell_match:
            continue
        cell_id = cell_match.group(1)
        row: dict[str, object] = {"cell_id": cell_id, "module": module}
        has_signal = False
        for out_name, (metric_name, attr) in metrics.items():
            value = metric_value(df, metric_name, attr=attr, module_regex=re.escape(module))
            row[out_name] = value
            has_signal = has_signal or not math.isnan(value)
        if has_signal:
            rows.append(row)
    return pd.DataFrame(rows)


def derive_global_metrics(df: pd.DataFrame) -> dict[str, float | str]:
    generated_mean = metric_value(df, "cbrGeneratedThroughput", attr="simu5g_rateavg", agg="mean")
    received_mean = metric_value(df, "cbrReceivedThroughput", attr="simu5g_rateavg", agg="mean")
    generated_sum = metric_value(df, "cbrGeneratedThroughput", attr="simu5g_rateavg", agg="sum")
    received_sum = metric_value(df, "cbrReceivedThroughput", attr="simu5g_rateavg", agg="sum")
    sinr = metric_value(df, "rcvdSinrDl", attr="mean", module_regex=r"ue\[\d+\]\.cellularNic\.(channelModel|nrChannelModel)")
    cqi = metric_value(df, "averageCqiDl", attr="mean", module_regex=r"ue\[\d+\]\.cellularNic\.(phy|nrPhy)")
    packet_loss = metric_value(df, "macCellPacketLossDl", attr="mean", module_regex=r"(masterEnb|secondaryGnb)\d+\.cellularNic\.mac")
    handover_proxy = metric_value(df, "handoverLatency", attr="mean")

    cdr_explicit, cdr_source = explicit_cdr_value(df)
    if math.isnan(generated_sum) or generated_sum == 0 or math.isnan(received_sum):
        cdr_proxy = float("nan")
    else:
        cdr_proxy = float(max(0.0, 1.0 - (received_sum / generated_sum)))

    cdr_global = cdr_explicit
    if math.isnan(cdr_global):
        cdr_global = cdr_proxy
        if not math.isnan(cdr_proxy):
            cdr_source = "proxy:1-received/generated"

    return {
        "cdr_global": cdr_global,
        "cdr_source": cdr_source,
        "cdr_proxy_global": cdr_proxy,
        "throughput_global": received_mean,
        "throughput_generated_global": generated_mean,
        "sinr_global": sinr,
        "cqi_global": cqi,
        "handover_count": float("nan"),
        "handover_proxy": handover_proxy,
        "packet_loss_rate": packet_loss,
    }


def scenario_from_flat_name(name: str) -> str:
    lowered = name.lower()
    if "baseline" in lowered:
        return "baseline"
    if "learned" in lowered or "rhohat" in lowered or "rho_hat" in lowered:
        return "learned"
    return "inventory_only"


def extract_paired_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    by_cell_rows: list[dict[str, object]] = []
    if not results_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for scenario_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        scenario = scenario_dir.name
        for run_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir()]):
            sca_candidates = sorted(run_dir.glob("*.sca"))
            if not sca_candidates:
                continue
            sca_path = sca_candidates[0]
            df = read_sca_rows(sca_path)
            cell_df = cell_kpis(df)
            components = parse_run_components(run_dir.name)
            record = {
                "scenario": scenario,
                "run_id": run_dir.name,
                **components,
                **derive_global_metrics(df),
                "source_path": str(sca_path),
            }
            if not cell_df.empty:
                record["throughput_per_cell_mean"] = float(pd.to_numeric(cell_df["throughput_dl"], errors="coerce").dropna().mean())
                record["packet_loss_per_cell_mean"] = float(pd.to_numeric(cell_df["packet_loss_dl"], errors="coerce").dropna().mean())
                record["num_cells_sampled"] = int(cell_df["cell_id"].nunique())
                for _, row in cell_df.iterrows():
                    by_cell_rows.append(
                        {
                            "scenario": scenario,
                            "run_id": run_dir.name,
                            **components,
                            **row.to_dict(),
                        }
                    )
            else:
                record["throughput_per_cell_mean"] = float("nan")
                record["packet_loss_per_cell_mean"] = float("nan")
                record["num_cells_sampled"] = 0
            raw_rows.append(record)
    return pd.DataFrame(raw_rows), pd.DataFrame(by_cell_rows)


def extract_flat_inventory(legacy_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    by_cell_rows: list[dict[str, object]] = []
    if not legacy_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for sca_path in sorted(legacy_dir.glob("*.sca")):
        scenario_name = sca_path.stem
        scenario = scenario_from_flat_name(scenario_name)
        df = read_sca_rows(sca_path)
        cell_df = cell_kpis(df)
        record = {
            "scenario": scenario,
            "run_id": scenario_name,
            "legacy_scenario_name": scenario_name,
            **parse_run_components(scenario_name),
            **derive_global_metrics(df),
            "source_path": str(sca_path),
        }
        if not cell_df.empty:
            record["throughput_per_cell_mean"] = float(pd.to_numeric(cell_df["throughput_dl"], errors="coerce").dropna().mean())
            record["packet_loss_per_cell_mean"] = float(pd.to_numeric(cell_df["packet_loss_dl"], errors="coerce").dropna().mean())
            record["num_cells_sampled"] = int(cell_df["cell_id"].nunique())
            for _, row in cell_df.iterrows():
                by_cell_rows.append(
                    {
                        "scenario": scenario,
                        "run_id": scenario_name,
                        "legacy_scenario_name": scenario_name,
                        **row.to_dict(),
                    }
                )
        else:
            record["throughput_per_cell_mean"] = float("nan")
            record["packet_loss_per_cell_mean"] = float("nan")
            record["num_cells_sampled"] = 0
        raw_rows.append(record)
    return pd.DataFrame(raw_rows), pd.DataFrame(by_cell_rows)


def write_inventory_report(raw_df: pd.DataFrame, by_cell_df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("OMNET KPI EXTRACTION INVENTORY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"raw rows: {len(raw_df):,}")
    lines.append(f"by-cell rows: {len(by_cell_df):,}")
    lines.append(f"scenario labels: {sorted(raw_df['scenario'].dropna().astype(str).unique().tolist()) if len(raw_df) else []}")
    lines.append("")
    if len(raw_df):
        lines.append("GLOBAL KPI COVERAGE")
        for column in [
            "cdr_global",
            "cdr_proxy_global",
            "throughput_global",
            "sinr_global",
            "cqi_global",
            "packet_loss_rate",
            "throughput_per_cell_mean",
        ]:
            non_null = int(pd.to_numeric(raw_df[column], errors="coerce").notna().sum()) if column in raw_df.columns else 0
            lines.append(f"- {column}: {non_null}/{len(raw_df)} non-null")
        if "cdr_source" in raw_df.columns:
            lines.append("")
            lines.append("CDR SOURCE BREAKDOWN")
            cdr_source_counts = raw_df["cdr_source"].fillna("missing").astype(str).value_counts().to_dict()
            for name, count in sorted(cdr_source_counts.items(), key=lambda item: item[0]):
                lines.append(f"- {name}: {count}")
        lines.append("")
        if not {"baseline", "learned"}.issubset(set(raw_df["scenario"].dropna().astype(str).unique())):
            lines.append("PAIRWISE COMPARISON STATUS")
            lines.append("- Missing baseline/learned paired OMNeT runs in current workspace.")
            lines.append("- Comparison scripts will emit a diagnostic report instead of fake deltas.")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    legacy_dir = Path(args.legacy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paired_raw, paired_cell = extract_paired_results(results_dir)
    if len(paired_raw):
        raw_df, by_cell_df = paired_raw, paired_cell
        source_mode = "paired-results"
    else:
        raw_df, by_cell_df = extract_flat_inventory(legacy_dir)
        source_mode = "legacy-flat"

    raw_path = out_dir / "kpis_omnet_raw.csv"
    by_cell_path = out_dir / "kpis_omnet_by_cell.csv"
    report_path = out_dir / "kpis_omnet_inventory_report.txt"

    raw_df.to_csv(raw_path, index=False)
    by_cell_df.to_csv(by_cell_path, index=False)
    write_inventory_report(raw_df, by_cell_df, report_path)

    print("=" * 72)
    print("EXTRAYENDO KPIs DE OMNeT++")
    print("=" * 72)
    print(f"source mode: {source_mode}")
    print(f"raw rows: {len(raw_df):,}")
    print(f"by-cell rows: {len(by_cell_df):,}")
    if len(raw_df):
        print(f"scenario labels: {sorted(raw_df['scenario'].dropna().astype(str).unique().tolist())}")
    print(f"saved: {raw_path}")
    print(f"saved: {by_cell_path}")
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
