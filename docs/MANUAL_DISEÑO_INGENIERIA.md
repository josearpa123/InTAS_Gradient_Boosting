# MANUAL DE INGENIERÍA Y OPERACIÓN (VERSIÓN REPRODUCIBLE)

## Arquitectura de ejecución
El sistema se organiza en un pipeline secuencial:
1. Corridas SUMO batch (XML crudo).
2. Extracción Bronze (XML -> parquet).
3. Construcción Silver + referencia analítica.
4. Extracción OMNeT (si hay insumos crudos).
5. Preparación de insumos de unificación (KPI summary + mobility metrics).
6. Unificación de métricas.
7. Construcción de ventanas (dataset Gold).
8. Construcción de tabla ML.
9. Entrenamiento CatBoost + calibración isotónica.
10. Validación probabilística.
11. Comparación analítica vs aprendida.
12. Generación de figuras.

## Contrato de rutas del paquete
- Datos de entrada y artefactos intermedios: `data/`
- Reportes finales: `reports/final/`
- Figuras tesis: `reports/final/thesis_figures/`

## Operación local recomendada
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_reproduction.py
```

Para incluir corridas OMNeT++ desde el mismo flujo:
```bash
INTAS_RUN_OMNET=1 python scripts/run_full_reproduction.py
```

Para generar datasets de movilidad desde SUMO:
```bash
INTAS_RUN_SUMO=1 python scripts/run_full_reproduction.py
```

Para flujo completo integrado:
```bash
INTAS_RUN_SUMO=1 INTAS_RUN_OMNET=1 python scripts/run_full_reproduction.py
```

## Operación con Docker
```bash
docker build -t intas-tesis .
docker run --name intas-container intas-tesis
docker cp intas-container:/app/reports/final ./resultados_tesis
```

## Trazabilidad y reproducibilidad
- El entrenamiento genera `report_catboost_isotonic.json` con hiperparámetros y métricas.
- Las probabilidades `rho_hat` raw/calibradas se guardan en `data/artifacts/ml/final/`.
- Las figuras finales se generan a partir de reportes calculados, sin métricas hardcodeadas.
- El orquestador usa modo incremental por defecto (omite pasos con artefactos ya existentes).
