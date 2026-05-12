import subprocess
import os
import sys
from pathlib import Path

def run_step(name, command):
    print(f"\n{'='*70}")
    print(f" [PASO {name[:2]}] EJECUTANDO: {name[3:]}")
    print(f"{'='*70}")
    # Usar el mismo intérprete que lanzó el orquestador (respeta venv activo)
    cmd = command.replace("python ", sys.executable + " ")
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
    run_sumo = os.environ.get("INTAS_RUN_SUMO", "0") == "1"
    run_omnet = os.environ.get("INTAS_RUN_OMNET", "0") == "1"
    omnet_rep_from = os.environ.get("INTAS_OMNET_REP_FROM", "0")
    omnet_rep_to = os.environ.get("INTAS_OMNET_REP_TO", "14")

    print("#"*70)
    print(" REPRODUCCIÓN TOTAL: SISTEMA INTAS + ML (Versión Tesis 2026)")
    print("#"*70)
    if not force_rebuild:
        print(" Modo incremental: se omiten pasos con salidas existentes.")
    if run_sumo:
        print(" Modo SUMO activo: se generan corridas y datos bronze/silver/theory.")
    if run_omnet:
        print(f" Modo OMNeT activo: repeticiones {omnet_rep_from}..{omnet_rep_to}.")
    print("#"*70)

    steps = [
        (
            "00A Corridas SUMO batch (raw XML)",
            (
                "python scripts/00a_run_sumo_batch.py "
                f"--rep-from {omnet_rep_from} --rep-to {omnet_rep_to}"
            ),
            ["reports/final/sumo_batch_summary.json"],
        ),
        (
            "00B Extracción Bronze desde corridas SUMO",
            "python scripts/00b_extract_bronze_batch.py",
            ["reports/final/bronze_batch_summary.json"],
        ),
        (
            "00C Construcción Silver + Referencia Analítica",
            "python scripts/00c_build_silver_theory.py",
            [
                "data/silver/cell_exposure.parquet",
                "data/silver/route_label.parquet",
                "data/silver/route_events.parquet",
                "data/theory/analytic_rho_reference.parquet",
            ],
        ),
        (
            "00D Exportar trazas SUMO→OMNeT++ (integración secuencial)",
            "python scripts/00d_export_sumo_traces_for_omnet.py",
            ["reports/final/sumo_traces_omnet_summary.json"],
        ),
        (
            "00 Corridas OMNeT++ baseline/learned (Objetivo 2)",
            (
                "python scripts/00_run_omnet_obj2_batch.py "
                f"--rep-from {omnet_rep_from} --rep-to {omnet_rep_to}"
            ),
            ["reports/final/omnet_batch_summary.json"],
        ),
        (
            "01 Extracción de KPIs OMNeT++ (PRBs/SINR)",
            "python scripts/01_extract_omnet_kpis.py",
            ["reports/final/objetivo2/kpis_omnet_inventory_report.txt"],
        ),
        (
            "01B Preparación de insumos SUMO+OMNeT para unificación",
            "python scripts/01b_prepare_unify_inputs.py",
            [
                "data/kpi/summary_kpis_avg.csv",
                "data/mobility_metrics.parquet",
            ],
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
            "05 Entrenamiento CatBoost + XGBoost + Calibración Isotonic",
            "python scripts/05_train_ml_model.py",
            [
                "data/models/catboost_gbdt.cbm",
                "data/models/isotonic.joblib",
                "data/models/xgboost_ref.json",
                "data/artifacts/ml/final/rho_hat_windows_raw.parquet",
                "data/artifacts/ml/final/rho_hat_windows_calibrated.parquet",
                "reports/ml/report_catboost_isotonic.json",
            ],
        ),
        (
            "05B Reinyección de rho_hat en OMNeT++ (plan de inyección)",
            "python scripts/05b_inject_rho_omnet.py",
            ["data/artifacts/ml/injection/injection_plan.json"],
        ),
        (
            "05C Validación contra perfiles MMtQHU (propietario/empleado)",
            "python scripts/05c_validate_mmtqhu_profiles.py",
            ["reports/final/mmtqhu_validation_by_profile.csv"],
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
        (
            "09 OMNeT++ con reinyección de rho_hat (ceteris paribus ML)",
            (
                "python scripts/00_run_omnet_obj2_batch.py "
                "--injection-plan data/artifacts/ml/injection/injection_plan.json "
                "--results-root data/omnet_results_injected "
                "--summary reports/final/omnet_injected_summary.json "
                f"--rep-from {omnet_rep_from} --rep-to {omnet_rep_to}"
            ),
            ["reports/final/omnet_injected_summary.json"],
        ),
    ]

    for name, cmd, outputs in steps:
        if (name.startswith("00A ") or name.startswith("00B ") or name.startswith("00C ") or name.startswith("00D ")) and not run_sumo:
            print(f"[SKIP] {name} (activar con INTAS_RUN_SUMO=1)")
            continue
        if (name.startswith("00 ") or name.startswith("09 ")) and not run_omnet:
            print(f"[SKIP] {name} (activar con INTAS_RUN_OMNET=1)")
            continue
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
