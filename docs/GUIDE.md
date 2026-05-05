# GUÍA DE EJECUCIÓN REPRODUCIBLE (InTAS_PRODUCCION_READY)

## 1) Objetivo operativo
Este paquete permite reproducir el pipeline de datos + ML para generar:
- Probabilidades `rho_hat` (raw y calibradas).
- Métricas comparativas contra la referencia analítica.
- Figuras finales de tesis.

## 2) Estructura base usada por scripts
- Entradas: `data/`
- Modelos: `data/models/`
- Reportes: `reports/`
- Scripts: `scripts/`

## 3) Ejecución end-to-end (local)
Desde la raíz del proyecto:
```bash
. .venv/bin/activate  # si ya existe
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_reproduction.py
```

## 4) Salidas principales
- `reports/final/rho_compare_recomputed_global.csv`
- `reports/final/probabilistic_validity_global.csv`
- `reports/final/thesis_figures/`
- `reports/ml/report_catboost_isotonic.json`

## 5) Ejecución por etapas
```bash
python scripts/00a_run_sumo_batch.py
python scripts/00b_extract_bronze_batch.py
python scripts/00c_build_silver_theory.py
python scripts/01_extract_omnet_kpis.py
python scripts/01b_prepare_unify_inputs.py
python scripts/02_unify_metrics.py
python scripts/03_build_gold_windows.py
python scripts/04_build_ml_table.py
python scripts/05_train_ml_model.py
python scripts/06_evaluate_probabilistic.py
python scripts/07_compare_analytic_vs_learned.py
python scripts/08_generate_figures.py
```

Con corridas OMNeT++ incluidas en el orquestador:
```bash
INTAS_RUN_OMNET=1 python scripts/run_full_reproduction.py
```

Desde cero con generación de datasets de movilidad:
```bash
INTAS_RUN_SUMO=1 python scripts/run_full_reproduction.py
```

Modo full integrado (SUMO + OMNeT + ML):
```bash
INTAS_RUN_SUMO=1 INTAS_RUN_OMNET=1 python scripts/run_full_reproduction.py
```

## 6) Alcance de reproducibilidad
- El flujo principal usa datos ya empaquetados en `data/`.
- La extracción OMNeT (`scripts/01_extract_omnet_kpis.py`) y unificación completa (`scripts/02_unify_metrics.py`) se ejecutan con entradas disponibles; si no hay insumos de simulación crudos, el pipeline reutiliza artefactos precomputados.
- Para forzar reconstrucción total de etapas (sin modo incremental): `INTAS_FORCE_REBUILD=1 python scripts/run_full_reproduction.py`.
