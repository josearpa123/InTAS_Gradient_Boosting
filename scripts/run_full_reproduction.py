import subprocess
import os
import sys
from pathlib import Path

def run_step(name, command):
    print(f"\n{'='*70}")
    print(f" [PASO {name[:2]}] EJECUTANDO: {name[3:]}")
    print(f"{'='*70}")
    # Usamos python3 para asegurar compatibilidad en Linux/Docker
    cmd = command.replace("python ", "python3 ")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR CRÍTICO] El paso '{name}' falló con código {result.returncode}.")
        sys.exit(1)

def should_skip(outputs, force_rebuild):
    if force_rebuild:
        return False
    if not outputs:
        return False
    return all(Path(p).exists() for p in outputs)

def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    force_rebuild = os.environ.get("INTAS_FORCE_REBUILD", "0") == "1"

    print("#"*70)
    print(" REPRODUCCIÓN TOTAL: SISTEMA INTAS + ML (Versión Tesis 2026)")
    print("#"*70)
    if not force_rebuild:
        print(" Modo incremental: se omiten pasos con salidas existentes.")
        print("#"*70)

    steps = [
        (
            "01 Extracción de KPIs OMNeT++ (PRBs/SINR)",
            "python scripts/01_extract_omnet_kpis.py",
            ["reports/final/objetivo2/kpis_omnet_inventory_report.txt"],
        ),
        (
            "02 Unificación de Métricas (180 Corridas)",
            "python scripts/02_unify_metrics.py",
            ["data/unified_metrics.parquet"],
        ),
        (
            "03 Construcción de Dataset Gold (Windowing)",
            "python scripts/03_build_gold_windows.py",
            ["data/dataset_windows.parquet"],
        ),
        (
            "04 Refino de Tabla de ML",
            "python scripts/04_build_ml_table.py",
            ["data/ml_table.parquet"],
        ),
        (
            "05 Entrenamiento CatBoost + Calibración Isotonic",
            "python scripts/05_train_ml_model.py",
            [
                "data/models/catboost_gbdt.cbm",
                "data/models/isotonic.joblib",
                "data/artifacts/ml/final/rho_hat_windows_raw.parquet",
                "data/artifacts/ml/final/rho_hat_windows_calibrated.parquet",
                "reports/ml/report_catboost_isotonic.json",
            ],
        ),
        (
            "06 Evaluación de Validez Probabilística (Brier/ECE)",
            "python scripts/06_evaluate_probabilistic.py",
            ["reports/final/probabilistic_validity_global.csv"],
        ),
        (
            "07 Comparación rho_hat vs rho_analítica",
            "python scripts/07_compare_analytic_vs_learned.py",
            ["reports/final/rho_compare_recomputed_global.csv"],
        ),
        (
            "08 Generación de Figuras de Tesis (PDF/SVG)",
            "python scripts/08_generate_figures.py",
            ["reports/final/thesis_figures/figures_manifest.md"],
        ),
    ]

    for name, cmd, outputs in steps:
        if should_skip(outputs, force_rebuild):
            print(f"[SKIP] {name} (salidas existentes)")
            continue
        run_step(name, cmd)

    print("\n" + "!"*70)
    print(" ¡PROCESO COMPLETADO!")
    print(" Todos los artefactos de la tesis han sido regenerados.")
    print(" Ubicación de figuras: reports/final/thesis_figures/")
    print("!"*70)

if __name__ == "__main__":
    main()
