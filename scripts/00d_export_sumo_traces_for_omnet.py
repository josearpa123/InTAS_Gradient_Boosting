#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00d_export_sumo_traces_for_omnet.py — Integración secuencial SUMO → OMNeT++

Convierte las trazas FCD de SUMO (datos de movilidad vehicular) al formato de
trazas de movilidad de INET/OMNeT++ (TraceFileMobility), completando así el
acoplamiento offline SUMO↔OMNeT++.

Arquitectura de co-simulación utilizada (acoplamiento secuencial):
  1. SUMO genera trazas FCD con posiciones, velocidades y aceleraciones reales
     de los vehículos para cada condición experimental (HC1-HC3 × 5%/10%).
  2. Este script convierte esas trazas al formato que OMNeT++ puede leer con
     el módulo TraceFileMobility de INET.
  3. OMNeT++ ejecuta la capa 5G usando los movimientos reales de SUMO, en
     lugar de LinearMobility (trayectoria recta artificial).

Esto es equivalente a la integración Veins pero en modo offline (secuencial),
lo que permite reproducibilidad sin requerir TraCI en tiempo real.

Formato de salida por corrida (INET TraceFileMobility):
  <t_s>  <x_m>  <y_m>
  Una línea por timestep, coordenadas en metros relativas al origen de celda.

Entradas:
  data/bronze/fcd/                   (parquets FCD por run_id, del paso 00B)

Salidas:
  data/sumo_traces_omnet/<run_id>/<vehID>.trace
  data/sumo_traces_omnet/trace_index.json
  reports/final/sumo_traces_omnet_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COORD_COLS_CANDIDATES = [
    ("x", "y"),
    ("pos_x", "pos_y"),
    ("lon", "lat"),
    ("x_m", "y_m"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SUMO FCD parquets to INET TraceFileMobility traces"
    )
    p.add_argument("--fcd-dir", default="data/bronze/fcd",
                   help="Directory with per-run FCD parquets (output of 00b)")
    p.add_argument("--cells-poly", default="scenarios/sumo/network/cells.poly.xml",
                   help="SUMO cells polygon (for coordinate origin)")
    p.add_argument("--out-dir", default="data/sumo_traces_omnet",
                   help="Output directory for trace files")
    p.add_argument("--summary", default="reports/final/sumo_traces_omnet_summary.json")
    p.add_argument("--max-vehicles", type=int, default=5,
                   help="Max vehicles per run to export (first N by vehID, for demo runs)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Timestep of FCD data in seconds (default: 0.1 s)")
    return p.parse_args()


def detect_coord_cols(df: pd.DataFrame) -> tuple[str, str] | None:
    for cx, cy in COORD_COLS_CANDIDATES:
        if cx in df.columns and cy in df.columns:
            return cx, cy
    return None


def get_origin(cells_poly: Path) -> tuple[float, float]:
    """Read first cell centroid as coordinate origin for OMNeT++ canvas."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(cells_poly).getroot()
        for poly in root.findall(".//poly"):
            shape = poly.get("shape", "")
            if shape:
                pts = [tuple(map(float, pt.split(","))) for pt in shape.split()]
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    return float(np.mean(xs)), float(np.mean(ys))
    except Exception:
        pass
    return 0.0, 0.0


def write_trace(veh_df: pd.DataFrame, cx: str, cy: str,
                ox: float, oy: float, out_path: Path, dt: float) -> int:
    """Write INET TraceFileMobility format: time(s)  x(m)  y(m)."""
    veh_df = veh_df.sort_values("timestep_time" if "timestep_time" in veh_df.columns else cx)
    lines = []
    for i, row in enumerate(veh_df.itertuples(index=False)):
        t = getattr(row, "timestep_time", i * dt)
        x = getattr(row, cx, 0.0) - ox
        y = getattr(row, cy, 0.0) - oy
        lines.append(f"{float(t):.3f}\t{float(x):.3f}\t{float(y):.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    args = parse_args()
    fcd_dir = Path(args.fcd_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ox, oy = get_origin(Path(args.cells_poly))
    print(f"[INFO] Coordinate origin: ox={ox:.1f}, oy={oy:.1f}")

    if not fcd_dir.exists():
        print(f"[WARN] FCD directory not found: {fcd_dir}")
        print("[WARN] Run 00a + 00b first to generate FCD data, or set INTAS_RUN_SUMO=1")
        summary_path.write_text(
            json.dumps({"status": "skipped_no_fcd", "fcd_dir": str(fcd_dir)}, indent=2),
            encoding="utf-8",
        )
        return 0

    parquet_files = sorted(fcd_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(fcd_dir.rglob("fcd*.parquet"))

    if not parquet_files:
        print(f"[WARN] No FCD parquets found in {fcd_dir}. Skipping trace export.")
        summary_path.write_text(
            json.dumps({"status": "no_parquets", "fcd_dir": str(fcd_dir)}, indent=2),
            encoding="utf-8",
        )
        return 0

    trace_index: dict[str, list[str]] = {}
    total_traces = 0
    run_summaries = []

    for pq in parquet_files:
        run_id = pq.stem.replace("fcd_", "").replace("_fcd", "")
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[WARN] Cannot read {pq}: {e}")
            continue

        coord = detect_coord_cols(df)
        if coord is None:
            print(f"[WARN] No coordinate columns found in {pq} — skipping")
            continue
        cx, cy = coord

        veh_col = next((c for c in ["vehID", "vehicle_id", "id"] if c in df.columns), None)
        if veh_col is None:
            print(f"[WARN] No vehicle ID column in {pq} — skipping")
            continue

        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        vehs = sorted(df[veh_col].unique())[: args.max_vehicles]
        run_traces = []

        for veh in vehs:
            vdf = df[df[veh_col] == veh].copy()
            if len(vdf) < 2:
                continue
            safe_name = str(veh).replace("/", "_").replace(":", "_")
            trace_path = run_dir / f"{safe_name}.trace"
            n = write_trace(vdf, cx, cy, ox, oy, trace_path, args.dt)
            run_traces.append(str(trace_path.relative_to(out_dir.parent)))
            total_traces += 1

        trace_index[run_id] = run_traces
        run_summaries.append({
            "run_id": run_id,
            "n_vehicles_exported": len(run_traces),
            "source": str(pq),
        })
        print(f"[OK] {run_id}: {len(run_traces)} vehicle traces")

    index_path = out_dir / "trace_index.json"
    index_path.write_text(
        json.dumps({"origin_x": ox, "origin_y": oy,
                    "coord_unit": "meters", "time_unit": "seconds",
                    "inet_module": "TraceFileMobility",
                    "runs": trace_index}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "status": "ok",
        "fcd_dir": str(fcd_dir),
        "out_dir": str(out_dir),
        "total_runs": len(run_summaries),
        "total_traces": total_traces,
        "max_vehicles_per_run": args.max_vehicles,
        "runs": run_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Exported {total_traces} traces across {len(run_summaries)} runs")
    print(f"[OK] Trace index: {index_path}")
    print(f"[OK] Summary: {summary_path}")
    print()
    print("NEXT STEP:")
    print("  These trace files can be used with INET TraceFileMobility.")
    print("  In omnetpp.ini:")
    print("    *.ue[*].mobility.typename = \"TraceFileMobility\"")
    print(f"   *.ue[*].mobility.traceFile = \"{out_dir}/<run_id>/<vehID>.trace\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
