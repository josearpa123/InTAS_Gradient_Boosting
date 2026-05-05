#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract SUMO XML outputs into bronze parquet datasets for all runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch bronze extraction from sim/runs")
    p.add_argument("--runs-dir", default="sim/runs")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--extract-script", default="scripts/helpers/extract_bronze.py")
    p.add_argument("--bronze-out", default="data/bronze")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    extract_script = Path(args.extract_script).resolve()
    bronze_out = Path(args.bronze_out).resolve()

    if not extract_script.exists():
        raise FileNotFoundError(f"Missing extractor script: {extract_script}")

    if not runs_dir.exists():
        print(f"[WARN] runs_dir not found: {runs_dir}")
        print("[WARN] Skipping bronze extraction batch.")
        return 0

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    ok = 0
    fail = 0
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        cmd = [
            args.python_bin,
            str(extract_script),
            "--run_id",
            run_id,
            "--runs_dir",
            str(runs_dir),
            "--out",
            str(bronze_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rec = {
            "run_id": run_id,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-5000:],
            "stderr_tail": proc.stderr[-5000:],
        }
        rows.append(rec)
        if proc.returncode == 0:
            ok += 1
        else:
            fail += 1

    summary = {
        "runs_dir": str(runs_dir),
        "bronze_out": str(bronze_out),
        "total_runs": len(run_dirs),
        "ok_runs": ok,
        "failed_runs": fail,
        "rows": rows,
    }
    out_summary = Path("reports/final/bronze_batch_summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Bronze extraction finished. ok={ok} fail={fail} summary={out_summary}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
