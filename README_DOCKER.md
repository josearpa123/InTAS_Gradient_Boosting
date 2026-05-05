# REPRODUCIBILIDAD CON DOCKER: InTAS + ML

Este paquete está preparado para ejecutar el pipeline de datos + ML dentro de contenedor, incluyendo el bloque analítico (`rho`) para comparación y calibración.

## Requisitos
- Docker instalado.

## 1. Construir la imagen
Desde `InTAS_PRODUCCION_READY`:
```bash
docker build -t intas-tesis .
```

## 2. Ejecutar el pipeline completo
```bash
docker run --name intas-container intas-tesis
```

## 2.1 Ejecutar regeneración de datos desde SUMO
```bash
docker run --name intas-container-sumo -e INTAS_RUN_SUMO=1 intas-tesis
```

## 2.2 Ejecutar flujo full (SUMO + OMNeT + ML)
```bash
docker run --name intas-container-full -e INTAS_RUN_SUMO=1 -e INTAS_RUN_OMNET=1 intas-tesis
```

## 3. Extraer resultados
```bash
docker cp intas-container:/app/reports/final ./resultados_tesis
```

## Qué produce
- `reports/final/rho_compare_recomputed_global.csv`
- `reports/final/probabilistic_validity_global.csv`
- `reports/final/thesis_figures/` (PNG/PDF/SVG y manifest)
- `reports/ml/report_catboost_isotonic.json`

## Nota de alcance
- El modo por defecto ejecuta ML + comparación analítica usando los artefactos disponibles.
- `INTAS_RUN_SUMO=1` sí está soportado en el contenedor (requiere recursos de cómputo adecuados).
- `INTAS_RUN_OMNET=1` requiere librerías/modelos OMNeT/INET/Simu5G/Artery compilados dentro del contenedor; el Dockerfile actual no compila ese stack automáticamente.
