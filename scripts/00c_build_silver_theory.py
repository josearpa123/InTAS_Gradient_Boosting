#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build silver datasets and analytic rho reference from bronze + SUMO route definitions."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cell exposure, route labels/events, analytic rho")
    p.add_argument("--bronze-fcd-dir", default="data/bronze/fcd")
    p.add_argument("--cells-poly", default="scenarios/sumo/network/cells.poly.xml")
    p.add_argument("--routes-dir", default="scenarios/sumo/routes")
    p.add_argument("--fcd-cells-dir", default="data/silver/fcd_cells")
    p.add_argument("--cell-exposure-out", default="data/silver/cell_exposure.parquet")
    p.add_argument("--route-label-out", default="data/silver/route_label.parquet")
    p.add_argument("--route-events-out", default="data/silver/route_events.parquet")
    p.add_argument("--analytic-out", default="data/theory/analytic_rho_reference.parquet")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--assign-cells-script", default="scripts/helpers/fcd_assign_cells.py")
    return p.parse_args()


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    # HC1_5pct__nearest__rep00
    period, policy, rep_text = run_id.split("__")
    rep = int(rep_text.replace("rep", ""))
    return period, policy, rep


def parse_route_candidates(route_file: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    root = ET.parse(route_file).getroot()
    for veh in root.findall(".//vehicle"):
        veh_id = veh.get("id")
        if not veh_id:
            continue
        rd = veh.find("routeDistribution")
        if rd is None:
            continue
        routes = rd.findall("route")
        if not routes:
            continue
        cand: list[dict[str, object]] = []
        for r in routes:
            edges = (r.get("edges") or "").split()
            prob = float(r.get("probability") or 0.0)
            cost = float(r.get("cost") or 0.0)
            cand.append({"edges": edges, "probability": prob, "cost": cost})
        chosen_idx = int(rd.get("last") or 0)
        if chosen_idx < 0 or chosen_idx >= len(cand):
            chosen_idx = 0
        baseline_idx = max(range(len(cand)), key=lambda i: float(cand[i]["probability"]))
        out[veh_id] = {
            "candidates": cand,
            "chosen_idx": chosen_idx,
            "baseline_idx": baseline_idx,
        }
    return out


def extract_actual_sequences(fcd: pd.DataFrame) -> dict[str, dict[str, list]]:
    d = fcd[["vehID", "t", "edge_id"]].dropna(subset=["vehID", "t", "edge_id"]).copy()
    d["vehID"] = d["vehID"].astype(str)
    d["edge_id"] = d["edge_id"].astype(str)
    d["t"] = pd.to_numeric(d["t"], errors="coerce")
    d = d.dropna(subset=["t"]).sort_values(["vehID", "t"])
    first_row_or_edge_change = d["vehID"].ne(d["vehID"].shift()) | d["edge_id"].ne(d["edge_id"].shift())
    changes = d.loc[first_row_or_edge_change, ["vehID", "t", "edge_id"]]
    grouped = changes.groupby("vehID", sort=False).agg({"t": list, "edge_id": list})
    return {
        veh_id: {"times": vals["t"], "edges": vals["edge_id"]}
        for veh_id, vals in grouped.to_dict(orient="index").items()
    }


def first_divergence(planned: list[str], actual: list[str]) -> tuple[int | None, str | None, str | None]:
    n = min(len(planned), len(actual))
    for i in range(n):
        if planned[i] != actual[i]:
            return i, planned[i], actual[i]
    if len(actual) > len(planned):
        return len(planned), None, actual[len(planned)]
    return None, None, None


def ensure_fcd_cells(args: argparse.Namespace, run_ids: list[str]) -> None:
    import subprocess

    assign_script = Path(args.assign_cells_script).resolve()
    if not assign_script.exists():
        raise FileNotFoundError(f"Missing assign cells script: {assign_script}")

    for run_id in run_ids:
        out = Path(args.fcd_cells_dir) / f"{run_id}.parquet"
        if out.exists():
            continue
        cmd = [
            args.python_bin,
            str(assign_script),
            "--run_id",
            run_id,
            "--cells_poly",
            str(Path(args.cells_poly).resolve()),
            "--bronze_dir",
            str(Path(args.bronze_fcd_dir).resolve()),
            "--out_dir",
            str(Path(args.fcd_cells_dir).resolve()),
        ]
        subprocess.run(cmd, check=True)


def build_cell_exposure(fcd_cells_dir: Path, run_ids: list[str]) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        p = fcd_cells_dir / f"{run_id}.parquet"
        if not p.exists():
            continue
        period, policy, rep = parse_run_id(run_id)
        df = pd.read_parquet(p, columns=["vehID", "cell_id", "t"])
        df = df.dropna(subset=["vehID", "cell_id", "t"]).copy()
        if df.empty:
            continue
        df["vehID"] = df["vehID"].astype(str)
        df["cell_id"] = df["cell_id"].astype(str)
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        g = (
            df.groupby(["vehID", "cell_id"], as_index=False)
            .agg(
                t_first=("t", "min"),
                t_last=("t", "max"),
                samples=("t", "count"),
            )
            .sort_values(["vehID", "cell_id"])
        )
        g["time_s"] = g["samples"].astype(float)
        g.insert(0, "run_id", run_id)
        g.insert(1, "period", period)
        g.insert(2, "policy", policy)
        g.insert(3, "rep", rep)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_labels_events_and_analytic(bronze_fcd_dir: Path, routes_dir: Path, run_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_cache: dict[str, dict[str, dict[str, object]]] = {}
    labels: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for run_id in run_ids:
        period, policy, rep = parse_run_id(run_id)
        route_path = routes_dir / f"routes_{period}.rou.xml"
        if period not in route_cache:
            if not route_path.exists():
                raise FileNotFoundError(f"Missing route file: {route_path}")
            route_cache[period] = parse_route_candidates(route_path)
        route_defs = route_cache[period]

        fcd_path = bronze_fcd_dir / f"{run_id}.parquet"
        if not fcd_path.exists():
            continue
        fcd = pd.read_parquet(fcd_path, columns=["vehID", "t", "edge_id"])
        actual_seq = extract_actual_sequences(fcd)

        vehicle_ids = sorted(set(route_defs.keys()).union(actual_seq.keys()))
        for veh_id in vehicle_ids:
            meta = route_defs.get(veh_id)
            actual = actual_seq.get(veh_id, {"times": [], "edges": []})
            if meta is None:
                labels.append(
                    {
                        "run_id": run_id,
                        "period": period,
                        "policy": policy,
                        "rep": rep,
                        "vehID": veh_id,
                        "label": 0,
                        "n_candidates": 0,
                        "baseline_mode": "missing_route_distribution",
                        "baseline_prob": 0.0,
                        "baseline_cost": 0.0,
                        "chosen_prob": 0.0,
                        "chosen_cost": 0.0,
                        "matched": "missing_route_distribution",
                    }
                )
                continue

            candidates = meta["candidates"]
            chosen = candidates[int(meta["chosen_idx"])]
            baseline = candidates[int(meta["baseline_idx"])]

            planned_edges = list(chosen["edges"])
            actual_edges = list(actual["edges"])
            idx_div, edge_planned, edge_actual = first_divergence(planned_edges, actual_edges)
            has_div = idx_div is not None

            labels.append(
                {
                    "run_id": run_id,
                    "period": period,
                    "policy": policy,
                    "rep": rep,
                    "vehID": veh_id,
                    "label": int(has_div),
                    "n_candidates": int(len(candidates)),
                    "baseline_mode": "max_prob",
                    "baseline_prob": float(baseline["probability"]),
                    "baseline_cost": float(baseline["cost"]),
                    "chosen_prob": float(chosen["probability"]),
                    "chosen_cost": float(chosen["cost"]),
                    "matched": "exact" if actual_edges else "no_fcd",
                }
            )

            if has_div:
                t_ref = float(actual["times"][idx_div]) if idx_div < len(actual["times"]) else float(actual["times"][-1]) if actual["times"] else 0.0
                events.append(
                    {
                        "run_id": run_id,
                        "period": period,
                        "policy": policy,
                        "rep": rep,
                        "vehID": veh_id,
                        "t_ref": t_ref,
                        "idx_diverge": int(idx_div),
                        "edge_planned": edge_planned,
                        "edge_actual": edge_actual,
                    }
                )

    label_df = pd.DataFrame(labels)
    events_df = pd.DataFrame(events)
    analytic = (
        label_df.groupby(["period", "vehID"], as_index=False)["label"]
        .mean()
        .rename(columns={"label": "rho"})
    )
    return label_df, events_df, analytic


def main() -> int:
    args = parse_args()
    bronze_fcd_dir = Path(args.bronze_fcd_dir).resolve()
    routes_dir = Path(args.routes_dir).resolve()
    fcd_cells_dir = Path(args.fcd_cells_dir).resolve()

    run_ids = sorted([p.stem for p in bronze_fcd_dir.glob("*.parquet") if "__rep" in p.stem])
    if not run_ids:
        raise FileNotFoundError(f"No bronze FCD parquet files found in {bronze_fcd_dir}")

    ensure_fcd_cells(args, run_ids)
    exposure = build_cell_exposure(fcd_cells_dir, run_ids)
    labels, events, analytic = build_labels_events_and_analytic(bronze_fcd_dir, routes_dir, run_ids)

    cell_exposure_out = Path(args.cell_exposure_out)
    route_label_out = Path(args.route_label_out)
    route_events_out = Path(args.route_events_out)
    analytic_out = Path(args.analytic_out)
    for p in [cell_exposure_out, route_label_out, route_events_out, analytic_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    exposure.to_parquet(cell_exposure_out, index=False)
    labels.to_parquet(route_label_out, index=False)
    events.to_parquet(route_events_out, index=False)
    analytic.to_parquet(analytic_out, index=False)

    summary = {
        "runs": len(run_ids),
        "cell_exposure_rows": int(len(exposure)),
        "route_label_rows": int(len(labels)),
        "route_events_rows": int(len(events)),
        "analytic_rows": int(len(analytic)),
        "positive_rate": float(labels["label"].mean()) if len(labels) else 0.0,
    }
    out_summary = Path("reports/final/silver_theory_build_summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Built silver/theory artifacts: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
