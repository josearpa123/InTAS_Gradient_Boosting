#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05b_inject_rho_omnet.py — Reinyección de ρ̂ aprendido en OMNeT++

Lee rho_hat_windows_calibrated.parquet y genera un "plan de inyección" que
asigna a cada corrida OMNeT++ la configuración que corresponde a la política
dictada por el estimador ML:

  - Si mean(ρ̂) ≥ umbral (por defecto 0.5):
        → DoubleConnection-CBR-DL  (doble conectividad proactiva)
          Interpretación: el modelo predice alta probabilidad de desvío;
          se activa handover anticipado a la celda techo (ceiling policy).

  - Si mean(ρ̂) < umbral:
        → SingleConnection-CBR-DL  (conexión única / política baseline nearest)
          Interpretación: el modelo predice baja probabilidad de desvío;
          no se altera la conexión primaria.

El plan se guarda como JSON y como CSV para auditoría.
El script 00_run_omnet_obj2_batch.py lo consume con --injection-plan.

Salidas:
  data/artifacts/ml/injection/injection_plan.json
  data/artifacts/ml/injection/injection_plan.csv
  data/artifacts/ml/injection/injection_summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_CONF = "SingleConnection-CBR-DL"
LEARNED_CONF = "DoubleConnection-CBR-DL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-run OMNeT++ injection plan from calibrated rho_hat"
    )
    p.add_argument(
        "--cal",
        default="data/artifacts/ml/final/rho_hat_windows_calibrated.parquet",
        help="Calibrated rho_hat parquet (output of 05_train_ml_model.py)",
    )
    p.add_argument(
        "--out-dir",
        default="data/artifacts/ml/injection",
        help="Output directory for injection plan files",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ρ̂ threshold above which DoubleConnection is selected (default: 0.5)",
    )
    p.add_argument(
        "--baseline-conf",
        default=BASELINE_CONF,
        help="OMNeT++ config name for baseline (low rho_hat)",
    )
    p.add_argument(
        "--learned-conf",
        default=LEARNED_CONF,
        help="OMNeT++ config name for learned (high rho_hat)",
    )
    return p.parse_args()


def infer_run_key(row: pd.Series) -> str:
    """Build a run_id key from period + policy + rep columns (if present)."""
    parts = []
    for col in ("period", "policy", "rep"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return "__".join(parts) if parts else str(row.get("run_id", "unknown"))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.read_parquet(args.cal)

    if "rho_hat" not in cal.columns:
        raise SystemExit("[ERROR] La columna 'rho_hat' no existe en el parquet de entrada.")

    # Determinar columnas de agrupación disponibles
    group_cols = [c for c in ["run_id", "period", "policy", "rep"] if c in cal.columns]
    if not group_cols:
        raise SystemExit(
            "[ERROR] No se encontraron columnas de agrupación "
            "(run_id, period, policy, rep) en el parquet."
        )

    agg = (
        cal.groupby(group_cols, as_index=False)
        .agg(
            rho_hat_mean=("rho_hat", "mean"),
            rho_hat_max=("rho_hat", "max"),
            rho_hat_p75=("rho_hat", lambda s: float(np.percentile(s, 75))),
            n_windows=("rho_hat", "count"),
        )
    )

    # Decisión de configuración basada en rho_hat_mean vs umbral
    agg["selected_conf"] = np.where(
        agg["rho_hat_mean"] >= args.threshold,
        args.learned_conf,
        args.baseline_conf,
    )
    agg["rho_above_threshold"] = agg["rho_hat_mean"] >= args.threshold

    # Construir plan como dict {run_id_str → conf_name}
    plan: dict[str, str] = {}
    for _, row in agg.iterrows():
        if "run_id" in row.index:
            key = str(row["run_id"])
        else:
            key = infer_run_key(row)
        plan[key] = str(row["selected_conf"])

    # Estadísticas del plan
    n_total = len(agg)
    n_learned = int((agg["selected_conf"] == args.learned_conf).sum())
    n_baseline = n_total - n_learned
    mean_rho = float(agg["rho_hat_mean"].mean())

    # Guardar plan JSON (consumido por 00_run_omnet_obj2_batch.py)
    plan_payload = {
        "threshold": args.threshold,
        "baseline_conf": args.baseline_conf,
        "learned_conf": args.learned_conf,
        "n_runs": n_total,
        "n_learned": n_learned,
        "n_baseline": n_baseline,
        "global_mean_rho_hat": mean_rho,
        "plan": plan,
    }
    plan_path = out_dir / "injection_plan.json"
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Guardar CSV para auditoría
    csv_path = out_dir / "injection_plan.csv"
    agg.to_csv(csv_path, index=False)

    # Resumen legible
    lines = [
        "=" * 72,
        "PLAN DE INYECCIÓN DE ρ̂ EN OMNeT++",
        "=" * 72,
        "",
        f"Umbral de decisión : {args.threshold}",
        f"Config baseline    : {args.baseline_conf}",
        f"Config learned     : {args.learned_conf}",
        "",
        f"Total de corridas  : {n_total}",
        f"→ Learned (ρ̂ ≥ {args.threshold}): {n_learned}  ({100*n_learned/max(n_total,1):.1f}%)",
        f"→ Baseline (ρ̂ < {args.threshold}): {n_baseline}  ({100*n_baseline/max(n_total,1):.1f}%)",
        f"ρ̂ medio global    : {mean_rho:.4f}",
        "",
        "INTERPRETACIÓN:",
        "  Cuando ρ̂ ≥ umbral el estimador ML predice que el vehículo",
        "  desviará su ruta y se activa doble conectividad proactiva",
        "  (DoubleConnection-CBR-DL → política ceiling anticipada).",
        "  Cuando ρ̂ < umbral se mantiene la conexión única (baseline).",
        "",
        f"Plan guardado en   : {plan_path}",
        f"CSV de auditoría   : {csv_path}",
        "=" * 72,
    ]
    summary = "\n".join(lines)
    (out_dir / "injection_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
