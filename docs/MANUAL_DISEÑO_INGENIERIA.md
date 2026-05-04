# MANUAL DE INGENIERÍA Y OPERACIÓN (VERSIÓN REPRODUCIBLE)

## Arquitectura de ejecución
El sistema se organiza en un pipeline secuencial:
1. Extracción OMNeT (si hay insumos crudos).
2. Unificación de métricas.
3. Construcción de ventanas (dataset Gold).
4. Construcción de tabla ML.
5. Entrenamiento CatBoost + calibración isotónica.
6. Validación probabilística.
7. Comparación analítica vs aprendida.
8. Generación de figuras.

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
