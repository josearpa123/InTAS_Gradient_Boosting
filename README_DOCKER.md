# REPRODUCIBILIDAD CON DOCKER: InTAS + ML

Este paquete está preparado para ejecutar el pipeline de datos y ML con artefactos incluidos en `data/`.

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

## 3. Extraer resultados
```bash
docker cp intas-container:/app/reports/final ./resultados_tesis
```

## Qué produce
- `reports/final/rho_compare_recomputed_global.csv`
- `reports/final/probabilistic_validity_global.csv`
- `reports/final/thesis_figures/` (PNG/PDF/SVG y manifest)

## Nota de alcance
- El pipeline reproducible incluido usa los artefactos preempaquetados de `data/`.
- La re-simulación OMNeT++ desde `.sca/.vec` no forma parte del flujo por defecto si esos archivos no están presentes.
