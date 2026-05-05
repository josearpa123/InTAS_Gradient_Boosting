#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run SUMO batch simulations for InTAS periods/vfh/policies/repetitions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SUMO batch and persist raw XML outputs")
    p.add_argument("--sumo-bin", default="sumo")
    p.add_argument("--scenario-dir", default="scenarios/sumo")
    p.add_argument("--runs-dir", default="sim/runs")
    p.add_argument("--rep-from", type=int, default=0)
    p.add_argument("--rep-to", type=int, default=14)
    p.add_argument("--periods", nargs="+", default=["HC1", "HC2", "HC3"])
    p.add_argument("--vfhs", nargs="+", default=["5", "10"])
    p.add_argument("--policies", nargs="+", default=["nearest", "ceiling"])
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def sumocfg_name(period: str, vfh: str) -> str:
    return f"{period.lower()}_{vfh}pct.sumocfg"


def run_one(args: argparse.Namespace, run_id: str, sumocfg: Path, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fcd = out_dir / "fcd.xml.gz"
    tripinfo = out_dir / "tripinfo.xml.gz"
    vehroute = out_dir / "vehroute.xml"
    edgedata = out_dir / "edgedata.xml.gz"
    lanedata = out_dir / "lanedata.xml.gz"
    lanechanges = out_dir / "lanechanges.xml.gz"

    if args.skip_existing and fcd.exists() and tripinfo.exists() and vehroute.exists() and edgedata.exists():
        return

    cmd = [
        args.sumo_bin,
        "-c",
        str(sumocfg),
        "--seed",
        str(seed),
        "--fcd-output",
        str(fcd),
        "--tripinfo-output",
        str(tripinfo),
        "--vehroute-output",
        str(vehroute),
        "--edgedata-output",
        str(edgedata),
        "--lanedata-output",
        str(lanedata),
        "--lanechange-output",
        str(lanechanges),
    ]
    log_path = out_dir / "sumo.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    args = parse_args()
    scenario_dir = Path(args.scenario_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    total = 0
    for period in args.periods:
        for vfh in args.vfhs:
            cfg = scenario_dir / sumocfg_name(period, vfh)
            if not cfg.exists():
                raise FileNotFoundError(f"Missing SUMO config: {cfg}")
            for policy_idx, policy in enumerate(args.policies):
                for rep in range(args.rep_from, args.rep_to + 1):
                    total += 1
                    run_id = f"{period}_{vfh}pct__{policy}__rep{rep:02d}"
                    out_dir = runs_dir / run_id
                    seed = 100000 + rep + policy_idx * 1000 + int(vfh) * 10000 + (1 if period == "HC1" else 2 if period == "HC2" else 3) * 100
                    run_one(args, run_id, cfg, out_dir, seed)
                    manifest = {
                        "run_id": run_id,
                        "period": f"{period}_{vfh}pct",
                        "policy": policy,
                        "rep": rep,
                        "seed_sumo": seed,
                        "sumocfg": str(cfg),
                        "outdir": str(out_dir),
                    }
                    (out_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    records.append(manifest)

    summary = {
        "runs_dir": str(runs_dir),
        "total_runs": total,
        "records": records,
    }
    out_summary = Path("reports/final/sumo_batch_summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] SUMO batch completed. runs={total} summary={out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
