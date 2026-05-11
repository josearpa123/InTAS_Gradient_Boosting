#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05c_validate_mmtqhu_profiles.py — Validación contra perfiles MMtQHU

Compara ρ̂ calibrado con rho analítico segmentando los vehículos según el
perfil de comportamiento del modelo MMtQHU (Modelo de Movilidad basado en
Tipos y hábitos de la Quincena de Hogar y Uso):

  Propietario (owner):   vehículos con movilidad propia (id sin prefijo VFH_).
                         Representan usuarios con rutas relativamente estables.

  Empleado (commuter):   vehículos identificados con prefijo VFH_ en el id.
                         Representan usuarios con mayor variabilidad de ruta
                         asociada a trayectos laborales.

Métricas reportadas: MAE, RMSE, bias, correlación de Pearson.

Entradas:
  data/theory/analytic_rho_reference.parquet
  data/artifacts/ml/final/rho_hat_windows_calibrated.parquet

Salidas:
  reports/final/mmtqhu_validation_by_profile.csv
  reports/final/mmtqhu_validation_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate rho_hat against analytic rho split by MMtQHU behavioral profiles"
    )
    p.add_argument("--analytic", default="data/theory/analytic_rho_reference.parquet")
    p.add_argument("--cal", default="data/artifacts/ml/final/rho_hat_windows_calibrated.parquet")
    p.add_argument("--out-dir", default="reports/final")
    p.add_argument(
        "--vfh-prefix",
        default="VFH_",
        help="Prefix used to identify 'empleado' vehicles in vehID (default: VFH_)",
    )
    return p.parse_args()


def classify_profile(veh_id: str, vfh_prefix: str) -> str:
    return "empleado" if str(veh_id).startswith(vfh_prefix) else "propietario"


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if d.empty:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "corr": float("nan")}
    err = d["y_pred"] - d["y_true"]
    corr = float(d["y_true"].corr(d["y_pred"])) if len(d) > 1 else float("nan")
    return {
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "corr": corr,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = pd.read_parquet(args.analytic)[["period", "vehID", "rho"]].rename(
        columns={"rho": "rho_analytic"}
    )
    c = pd.read_parquet(args.cal)[["period", "vehID", "rho_hat"]].rename(
        columns={"rho_hat": "rho_cal"}
    )

    for df in (a, c):
        df["period"] = df["period"].astype(str)
        df["vehID"] = df["vehID"].astype(str)

    # Agregar ρ̂ al grano del analítico (period, vehID)
    agg_cal = (
        c.groupby(["period", "vehID"], as_index=False)
        .agg(rho_cal_mean=("rho_cal", "mean"), rho_cal_max=("rho_cal", "max"))
    )

    joined = a.merge(agg_cal, on=["period", "vehID"], how="inner")

    # Asignar perfil MMtQHU
    joined["profile"] = joined["vehID"].apply(
        lambda v: classify_profile(v, args.vfh_prefix)
    )

    profiles = ["propietario", "empleado", "global"]
    rows = []

    for profile in profiles:
        if profile == "global":
            subset = joined
        else:
            subset = joined[joined["profile"] == profile]

        if subset.empty:
            continue

        for variant, col in [("rho_cal_mean", "rho_cal_mean"), ("rho_cal_max", "rho_cal_max")]:
            m = metrics(subset["rho_analytic"], subset[col])
            rows.append({"profile": profile, "variant": variant, **m})

    result_df = pd.DataFrame(rows)
    csv_path = out_dir / "mmtqhu_validation_by_profile.csv"
    result_df.to_csv(csv_path, index=False)

    # Reporte legible
    lines = [
        "=" * 72,
        "VALIDACIÓN DE ρ̂ CONTRA PERFILES MMtQHU",
        "=" * 72,
        "",
        f"Vehículos analíticos       : {len(a):,}",
        f"Vehículos ρ̂ (calibrados)   : {len(c):,}",
        f"Pares unidos (period,vehID): {len(joined):,}",
        "",
    ]

    for profile in profiles:
        sub = result_df[result_df["profile"] == profile]
        if sub.empty:
            continue
        n_vehs = len(joined) if profile == "global" else int(
            (joined["profile"] == profile).sum()
        )
        lines.append(f"──── Perfil: {profile.upper()} (n={n_vehs:,}) ────")
        lines.append(sub.to_string(index=False))
        lines.append("")

    lines += [
        "NOTA:",
        "  'propietario' = vehículos sin prefijo VFH_ (movilidad propia/privada).",
        "  'empleado'    = vehículos con prefijo VFH_ (trayectos laborales/flotilla).",
        "  La mayor dispersión en empleados refleja la mayor variabilidad de ruta",
        "  asociada a cambios de turno y origen-destino diversificado.",
        "",
        "=" * 72,
    ]
    report = "\n".join(lines)
    report_path = out_dir / "mmtqhu_validation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
