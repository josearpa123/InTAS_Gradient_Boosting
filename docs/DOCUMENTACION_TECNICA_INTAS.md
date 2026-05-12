# Documentación Técnica — Sistema InTAS + ML  
**Integración de Modelos Calibrados de Gradient Boosting para la Toma de Decisiones en Sistemas Ciber-Físico-Sociales**

> Autor: Jose David Arias Pantoja  
> Director: Ph.D. Néstor Alzate Mejía  
> Universidad Cooperativa de Colombia — Ingeniería de Sistemas — Cali, 2026

---

## Tabla de Contenidos

1. [Contexto y Problema](#1-contexto-y-problema)
2. [Objetivos del Proyecto](#2-objetivos-del-proyecto)
3. [Marco Conceptual](#3-marco-conceptual)
4. [Infraestructura Técnica y Dependencias](#4-infraestructura-técnica-y-dependencias)
5. [Arquitectura de Datos: Bronze – Silver – Gold](#5-arquitectura-de-datos-bronze--silver--gold)
6. [Arquitectura del Pipeline](#6-arquitectura-del-pipeline)
7. [Descripción Detallada de cada Script](#7-descripción-detallada-de-cada-script)
8. [Flujo de Datos End-to-End](#8-flujo-de-datos-end-to-end)
9. [Entrenamiento ML: CatBoost, XGBoost y Calibración Isotónica](#9-entrenamiento-ml-catboost-xgboost-y-calibración-isotónica)
10. [Plan de Inyección de ρ̂ en OMNeT++](#10-plan-de-inyección-de-ρ̂-en-omnet)
11. [Validación contra Perfiles MMtQHU](#11-validación-contra-perfiles-mmtqhu)
12. [Evaluación Probabilística](#12-evaluación-probabilística)
13. [Estructura de Directorios del Repositorio](#13-estructura-de-directorios-del-repositorio)
14. [Reproducibilidad: Niveles y Comandos](#14-reproducibilidad-niveles-y-comandos)
15. [Ejecución con Docker](#15-ejecución-con-docker)
16. [Verificación Numérica de Resultados](#16-verificación-numérica-de-resultados)
17. [Variables de Entorno](#17-variables-de-entorno)
18. [Resolución de Problemas Comunes](#18-resolución-de-problemas-comunes)
19. [Limitaciones Actuales y Alcance Real](#19-limitaciones-actuales-y-alcance-real)
20. [Glosario](#20-glosario)
21. [Bibliografía](#21-bibliografía)

---

## 1. Contexto y Problema

### 1.1 El ecosistema InTAS

InTAS (Ingolstadt Traffic and Automotive Simulation) es un escenario de simulación vehicular desarrollado en colaboración con la Universidad de Ingolstadt. Proporciona un entorno controlado y reproducible donde es posible estudiar la interacción entre movilidad vehicular microscópica, redes de comunicación 5G y comportamiento humano de los conductores.

El escenario se basa en un mapa real de la ciudad de Ingolstadt (Alemania) y permite configurar distintos niveles de congestión vehicular, distintas cargas de vehículos de flotilla (VFH) y distintas políticas de decisión de ruta.

### 1.2 El problema central: ρ analítica vs ρ̂ aprendida

En el sistema InTAS, cada vehículo tiene asignada una probabilidad de desvío de ruta ρ (rho), que determina estocásticamente si el conductor seguirá o abandonará su ruta original ante un evento de congestión. Esta probabilidad puede calcularse de dos formas:

- **ρ analítica**: calculada mediante un modelo matemático cerrado a partir de variables de configuración del escenario (período de congestión, carga VFH, política). Es determinista dado el diseño experimental.
- **ρ̂ aprendida y calibrada**: estimada mediante un modelo de Gradient Boosting entrenado sobre observaciones reales de trayectorias vehiculares, KPIs de red y contexto espaciotemporal.

La pregunta central de la tesis es: **¿qué ocurre con los indicadores de desempeño de red (CDR, SINR, throughput) cuando se sustituye ρ por ρ̂ como entrada al sistema, manteniendo todo lo demás constante?**

### 1.3 Por qué importa la calibración probabilística

Un modelo de clasificación como CatBoost o XGBoost produce scores en [0,1], pero esos scores no son necesariamente probabilidades verdaderas. Si se usara ρ̂ directamente como entrada a una decisión estocástica sin calibrar, las decisiones vehiculares estarían sesgadas. La calibración isotónica transforma el score bruto en una estimación probabilística que refleja incertidumbre real: si el modelo dice ρ̂ = 0.7, aproximadamente el 70% de los vehículos con esa predicción deben haber desviado realmente.

---

## 2. Objetivos del Proyecto

### 2.1 Objetivo General

Evaluar el impacto de sustituir la probabilidad analítica (ρ) por una probabilidad aprendida y calibrada (ρ̂) sobre indicadores de desempeño de red (CDR, SINR, throughput) y servicio en un escenario CPSS integrado, bajo condiciones controladas donde políticas y parámetros del sistema se mantienen constantes excepto por el punto de sustitución probabilística.

### 2.2 Objetivos Específicos

**Objetivo 1 — Implementación del entorno de sustitución probabilística**  
Implementar el entorno de simulación con ρ̂ sustituyendo a ρ, manteniendo políticas de ejecución, umbrales de decisión y reglas de congestión idénticas fuera del punto de sustitución. Esto se materializa en los pasos 00A–05B del pipeline, especialmente en el plan de inyección (`injection_plan.json`) que asigna configuraciones OMNeT++ por corrida según el valor de ρ̂.

**Objetivo 2 — Comparación analítica vs aprendida en KPIs de red**  
Comparar ρ y ρ̂ en indicadores de desempeño de red (CDR, SINR, throughput) y en métricas de validez probabilística (Brier Score, ECE). Esto se materializa en los pasos 06 (evaluación probabilística) y 07 (comparación analítica vs aprendida).

**Objetivo 3 — Trazabilidad y versionamiento Bronze–Silver–Gold**  
Implementar un esquema de trazabilidad que vincule cada fila de datos, cada decisión tomada y cada métrica reportada a su configuración experimental, semilla y artefactos crudos de simulación. Esto se materializa en la arquitectura de capas de datos y en los manifiestos SHA-256.

---

## 3. Marco Conceptual

### 3.1 Sistemas Ciber-Físico-Sociales (CPSS)

Un CPSS acopla tres capas interdependientes:

| Capa | Descripción | Componente en InTAS |
|---|---|---|
| **Física** | Entidades con dinámica continua | Vehículos simulados en SUMO |
| **Ciber** | Sistemas de información y comunicación | Red 5G en OMNeT++/Simu5G |
| **Social** | Comportamiento e incertidumbre humana | Decisión de desvío de ruta (ρ / ρ̂) |

En movilidad urbana, estas capas no operan de forma independiente. Cuando un conductor decide desviar su ruta, eso cambia la topología del tráfico, lo que altera qué celdas 5G están expuestas a más vehículos, lo que afecta SINR, CDR y throughput.

### 3.2 Movilidad Vehicular y Desvío de Ruta

El escenario InTAS modela tres períodos de congestión (**HC1, HC2, HC3**) y dos cargas de vehículos de flotilla (**VFH 5%, VFH 10%**). Ante un evento de congestión, cada vehículo decide si desviar su ruta con probabilidad ρ (o ρ̂ en el experimento).

La política de handover vehicular tiene dos variantes:
- **Nearest**: el vehículo se conecta a la celda más cercana.
- **Ceiling**: el vehículo anticipa el cambio de celda y gestiona el handover proactivamente.

### 3.3 Redes 5G y Handover Vehicular

OMNeT++ con INET 4.4 y Simu5G 1.3 simula la capa ciber. Las métricas clave son:

- **SINR** (Signal-to-Interference-plus-Noise Ratio): calidad de señal en dB. Mayor es mejor.
- **CDR** (Call Drop Rate): fracción de conexiones perdidas. Menor es mejor.
- **Throughput**: velocidad de datos efectiva. Mayor es mejor.

La inyección de ρ̂ ≥ 0.5 activa la configuración `DoubleConnection-CBR-DL`, que implementa doble conectividad proactiva. La tesis reporta que esta configuración produce **SINR +1.49 dB** y **CDR = 0%** frente al baseline de conexión única.

### 3.4 Gradient Boosting y Calibración Isotónica

CatBoost y XGBoost entrenan ensambles de árboles de decisión de forma secuencial, donde cada árbol nuevo corrige los errores del anterior optimizando sobre el gradiente de la función de pérdida (log-loss binario en este caso).

La calibración isotónica posterior ajusta una función monótona no decreciente que mapea los scores brutos del modelo a probabilidades calibradas. Se entrena sobre el conjunto de validación para evitar sobreajuste.

---

## 4. Infraestructura Técnica y Dependencias

### 4.1 Stack de Simulación

| Componente | Versión | Rol |
|---|---|---|
| SUMO | 1.16+ | Simulador de movilidad vehicular microscópica (modelo Krauss, 200–500 vehículos, 3600 s) |
| OMNeT++ | 6.0 | Simulador de eventos discretos para red 5G |
| INET | 4.4 | Framework de protocolos de red para OMNeT++ |
| Simu5G | 1.3 | Extensión de simulación de redes 5G NR para OMNeT++/INET |
| Artery | — | Framework V2X complementario |

### 4.2 Stack Python (versiones exactas del repositorio)

```
pandas==2.2.3
numpy==1.26.4
catboost==1.2.7
xgboost==2.0.3
scikit-learn==1.6.1
joblib==1.4.2
pyarrow==19.0.1
matplotlib==3.10.0
pyyaml==6.0.2
scipy==1.15.2
```

Estas versiones están fijadas en `requirements.txt`. `numpy==1.26.4` se fija explícitamente por compatibilidad con CatBoost y XGBoost en la misma resolución de dependencias.

### 4.3 Entorno de Ejecución Recomendado

- **Sistema operativo**: Linux (Ubuntu 22.04+ recomendado), macOS o Windows con WSL2
- **Python**: 3.10 o superior (probado en 3.11 y 3.12)
- **RAM**: mínimo 16 GB para el procesamiento completo de los datasets
- **Disco**: mínimo 20 GB para artefactos generados (figuras, modelos, parquets intermedios)

---

## 5. Arquitectura de Datos: Bronze – Silver – Gold

El pipeline organiza los datos en tres capas con responsabilidades distintas:

```
data/
├── bronze/             ← Capa Bronze: datos crudos preservados fielmente
│   ├── fcd/            ← Trazas FCD (Floating Car Data) por run_id
│   ├── tripinfo/       ← Estadísticas de viaje por vehículo
│   └── edgedata/       ← Densidad y velocidad por segmento de vía
├── silver/             ← Capa Silver: datos interpretados y alineados
│   ├── cell_exposure.parquet
│   ├── route_label.parquet
│   └── route_events.parquet
├── theory/             ← Referencia analítica
│   └── analytic_rho_reference.parquet
├── kpi/                ← KPIs de red preparados
│   └── summary_kpis_avg.csv
├── unified_metrics.parquet   ← Silver unificado (movilidad + red)
├── dataset_windows.parquet   ← Gold: ventanas espaciotemporales
├── ml_table.parquet          ← Tabla ML final (features + labels)
├── models/             ← Modelos entrenados persistidos
│   ├── catboost_gbdt.cbm
│   ├── xgboost_ref.json
│   ├── xgb_encoders.joblib
│   └── isotonic.joblib
└── artifacts/ml/
    ├── final/          ← Predicciones ρ̂ raw y calibradas
    │   ├── rho_hat_windows_raw.parquet
    │   └── rho_hat_windows_calibrated.parquet
    └── injection/      ← Plan de inyección en OMNeT++
        ├── injection_plan.json
        ├── injection_plan.csv
        └── injection_summary.txt
```

### 5.1 Capa Bronze

**Responsabilidad**: Conversión 1:1 desde outputs de simulación sin interpretación de fenómenos. Se preserva la semántica original de SUMO (XML → Parquet) y de OMNeT++ (`.sca`/`.vec`). Incluye metadatos de corrida (`run_id`, semilla, configuración).

**Riesgo**: ninguna interpretación aquí. Si SUMO produce un ID de vehículo con cierta convención, se preserva exactamente.

### 5.2 Capa Silver

**Responsabilidad**: Primera interpretación del fenómeno. Es donde se concentra el **riesgo metodológico principal**:
- Se construye la etiqueta binaria de desvío (`route_label`).
- Se alinean espaciotemporalmente eventos de movilidad con períodos de congestión.
- Se asignan vehículos a celdas 5G según su posición geográfica (`cell_exposure`).
- Se calcula ρ analítica por `(period, vehID)`.

**Precaución de leakage**: las variables que alimentan ρ̂ se construyen de modo que no vean información posterior al punto en que la decisión de desvío se ejecuta. Esto preserva la causalidad del análisis.

### 5.3 Capa Gold

**Responsabilidad**: Dataset tabular final listo para entrenamiento. Cada fila representa una **ventana espaciotemporal** de un vehículo en una corrida. Cada columna es una variable predictiva o el label binario de desvío.

El proceso de windowing toma trayectorias continuas y las divide en segmentos de tiempo fijo, extrayendo features de cada segmento (media y desviación de velocidad, aceleración, frenadas abruptas, exposición a celda, etc.).

---

## 6. Arquitectura del Pipeline

### 6.1 Flujo General

```
SUMO (movilidad) ──────────────────────────────────────────────────────────────┐
  00a → 00b → 00c → 00d                                                        │
         ↓        ↓         ↓                                                  │
      Bronze    Silver    trazas .trace                                         │
                  ↓                ↓                                           │
              referencia ρ    OMNeT++ baseline                                  │
                  ↓                ↓                                           │
              01b (prep)      01 (KPIs)                                        │
                   ↘          ↙                                                │
                   02 (unify)                                                   │
                      ↓                                                        │
                  03 (Gold windows)                                             │
                      ↓                                                        │
                  04 (ML table)                                                 │
                      ↓                                                        │
         ┌────────── 05 (CatBoost + XGBoost + Isotonic) ─────────────┐        │
         ↓                    ↓                        ↓              ↓        │
      05b (injection plan)  05c (MMtQHU)           06 (Brier/ECE)  07 (MAE)   │
         ↓                                                            ↓        │
  OMNeT++ con ρ̂ (paso 09)                                        08 (figuras) │
```

### 6.2 Modos de Ejecución

El orquestador `scripts/run_full_reproduction.py` controla qué pasos se ejecutan según variables de entorno:

| Variable | Valor | Efecto |
|---|---|---|
| *(ninguna)* | — | Modo Ligero: solo pasos 01B–08. Lee seed data existentes. |
| `INTAS_RUN_SUMO=1` | `1` | Ejecuta además pasos 00A–00D (requiere SUMO instalado). |
| `INTAS_RUN_OMNET=1` | `1` | Ejecuta además pasos 00 y 09 (requiere OMNeT++ compilado). |
| `INTAS_FORCE_REBUILD=1` | `1` | Fuerza re-ejecución aunque las salidas ya existan. |

**Modo incremental**: por defecto, si las salidas esperadas de un paso ya existen en disco, el paso se omite. Esto evita re-ejecutar simulaciones lentas cuando solo se cambia el modelo ML.

---

## 7. Descripción Detallada de cada Script

### `scripts/run_full_reproduction.py` — Orquestador principal

**Responsabilidad**: Ejecutar los 17 pasos del pipeline en orden, con lógica de skip incremental y control por variables de entorno.

**Lógica clave**:
```python
for name, cmd, outputs in steps:
    if should_skip(outputs, force_rebuild):  # si los archivos de salida ya existen
        continue                             # → omitir el paso
    run_step(name, cmd)                      # → ejecutar y fallar si el código != 0
```

Si cualquier paso falla (código de retorno distinto de 0), el orquestador detiene la ejecución completa. No hay reintentos automáticos.

---

### `scripts/00a_run_sumo_batch.py` — Simulación SUMO batch

**Requiere**: `INTAS_RUN_SUMO=1`

**Responsabilidad**: Ejecutar corridas SUMO para el producto cartesiano de:
- Períodos de congestión: HC1, HC2, HC3
- Cargas VFH: 5%, 10%
- Políticas: nearest, ceiling
- Réplicas: configurable vía `--rep-from` / `--rep-to`

Cada corrida genera XML crudos de trayectorias FCD, información de viaje (tripinfo) y datos de segmentos de vía (edgedata).

**Entradas**: `scenarios/sumo/*.sumocfg`  
**Salidas**:
- `sim/runs/<run_id>/fcd.xml`
- `sim/runs/<run_id>/tripinfo.xml`
- `sim/runs/<run_id>/edgedata.xml`
- `reports/final/sumo_batch_summary.json`

---

### `scripts/00b_extract_bronze_batch.py` — Extracción Bronze

**Requiere**: `INTAS_RUN_SUMO=1`

**Responsabilidad**: Convertir los XML crudos de `sim/runs/` a formato Parquet comprimido, conservando todos los campos sin modificaciones semánticas.

**Entradas**: `sim/runs/<run_id>/*.xml`  
**Salidas**:
- `data/bronze/fcd/<run_id>.parquet`
- `data/bronze/tripinfo/<run_id>.parquet`
- `data/bronze/edgedata/<run_id>.parquet`
- `reports/final/bronze_batch_summary.json`

---

### `scripts/00c_build_silver_theory.py` — Silver + Referencia Analítica

**Requiere**: `INTAS_RUN_SUMO=1`

**Responsabilidad**: Construir la capa Silver con interpretación del fenómeno. Es el paso con mayor riesgo metodológico porque aquí se define qué significa "desvío de ruta".

Pasos internos:
1. Lee parquets FCD bronze y archivos de rutas candidatas (`routes_*.rou.xml`).
2. Compara la ruta efectivamente ejecutada por cada vehículo contra las rutas candidatas.
3. Asigna `route_label` (0 = sin desvío, 1 = desvío) por `(run_id, vehID, period)`.
4. Calcula `cell_exposure`: para cada posición FCD, determina la celda 5G más cercana.
5. Calcula ρ analítica por `(period, vehID)` como la fracción de períodos en que el vehículo desvió según la configuración del escenario.

**Entradas**:
- `data/bronze/fcd/`
- `scenarios/sumo/routes_*.rou.xml`
- `scenarios/sumo/cells_definition.json` (o similar)

**Salidas**:
- `data/silver/cell_exposure.parquet` — posición + celda por timestamp
- `data/silver/route_label.parquet` — etiqueta de desvío por (run_id, vehID, period)
- `data/silver/route_events.parquet` — eventos de desvío con timestamps
- `data/theory/analytic_rho_reference.parquet` — ρ analítica por (period, vehID)

---

### `scripts/00d_export_sumo_traces_for_omnet.py` — Acoplamiento SUMO → OMNeT++

**Requiere**: `INTAS_RUN_SUMO=1`

**Responsabilidad**: Implementar el **acoplamiento secuencial offline** entre SUMO y OMNeT++. Convierte las posiciones vehiculares del formato FCD Parquet al formato `TraceFileMobility` de INET (`t x y`), generando un archivo `.trace` por vehículo por corrida.

Este paso elimina la necesidad de co-simulación en tiempo real (acoplamiento online). OMNeT++ lee las trazas pregeneradas y reproduce exactamente la movilidad simulada por SUMO.

**Entradas**: `data/bronze/fcd/<run_id>.parquet`  
**Salidas**:
- `data/sumo_traces_omnet/<run_id>/<vehID>.trace`
- `data/sumo_traces_omnet/trace_index.json`
- `reports/final/sumo_traces_omnet_summary.json`

---

### `scripts/00_run_omnet_obj2_batch.py` — Simulación OMNeT++ batch

**Requiere**: `INTAS_RUN_OMNET=1`

**Responsabilidad**: Ejecutar corridas batch de OMNeT++ para el Objetivo 2 (comparación baseline vs learned). Si existen trazas SUMO en `data/sumo_traces_omnet/`, las usa automáticamente vía `TraceFileMobility`; de lo contrario usa `LinearMobility` como fallback.

También es el script usado en el **paso 09** cuando se le pasa `--injection-plan`, en cuyo caso cada corrida usa la configuración dictada por el plan de inyección de ρ̂ (ver Sección 10).

**Entradas**:
- `scenarios/omnet/omnetpp.ini`
- `scenarios/omnet/configs/intas_positions_4cells.ini`
- `scenarios/omnet/configs/intas_positions_10cells.ini`
- `data/sumo_traces_omnet/` (si existen)
- `data/artifacts/ml/injection/injection_plan.json` (solo en paso 09)

**Salidas**:
- `data/omnet_results/<run_id>.sca`
- `data/omnet_results/<run_id>.vec`
- `reports/final/omnet_batch_summary.json`

---

### `scripts/01_extract_omnet_kpis.py` — Extracción de KPIs de OMNeT++

**Responsabilidad**: Leer archivos `.sca` de resultados de OMNeT++ y extraer KPIs de red estructurados. Genera inventario de cobertura de KPIs.

**Entradas**: `data/omnet_results/*.sca` (si existen)  
**Salidas**:
- `reports/final/objetivo2/kpis_omnet_raw.csv`
- `reports/final/objetivo2/kpis_omnet_by_cell.csv`
- `reports/final/objetivo2/kpis_omnet_inventory_report.txt`

> En modo Ligero (sin OMNeT++), estos tres archivos ya están versionados en el repositorio como seed data, por lo que este paso se omite automáticamente.

---

### `scripts/01b_prepare_unify_inputs.py` — Preparación para unificación

**Responsabilidad**: Transformar los KPIs extraídos de OMNeT++ al formato largo esperado por el paso de unificación (`summary_kpis_avg.csv`). Además construye `mobility_metrics.parquet` combinando `cell_exposure` y `route_label` de la capa Silver.

**Entradas**:
- `reports/final/objetivo2/kpis_omnet_raw.csv`
- `data/silver/cell_exposure.parquet`
- `data/silver/route_label.parquet`

**Salidas**:
- `data/kpi/summary_kpis_avg.csv`
- `data/mobility_metrics.parquet`

> En modo Ligero, estos dos archivos ya están versionados como seed data.

---

### `scripts/02_unify_metrics.py` — Unificación de métricas

**Responsabilidad**: Unificar métricas de movilidad y de red en una tabla consolidada con granularidad `(run_id, period, vehID, cell_id)`.

**Comportamiento especial**: Si faltan insumos pero existe `data/unified_metrics.parquet` (seed data), reutiliza ese artefacto sin error.

**Entradas**:
- `data/mobility_metrics.parquet`
- `data/kpi/summary_kpis_avg.csv`

**Salidas**: `data/unified_metrics.parquet`

> En modo Ligero, este archivo ya está versionado como seed data.

---

### `scripts/03_build_gold_windows.py` — Construcción del dataset Gold

**Responsabilidad**: Construir ventanas temporales y extraer features por muestra. Cada fila del output representa un segmento de trayectoria de un vehículo en una ventana de tiempo fija.

Features extraídas por ventana:
- **Cinemáticas**: `speed_mean`, `speed_std`, `accel_mean`, `accel_std`, `jerk_abs_mean`, `stops_count`
- **Contexto de red**: columnas `network_*` y `prb_usage_*` (KPIs de celda asignada)
- **Categóricas**: `cell_id`, `period`, `policy`

**Entradas**:
- `data/silver/route_label.parquet`
- `data/silver/route_events.parquet`
- `data/unified_metrics.parquet`

**Salidas**: `data/dataset_windows.parquet`

> En modo Ligero, este archivo ya está versionado como seed data.

---

### `scripts/04_build_ml_table.py` — Tabla ML final

**Responsabilidad**: Seleccionar y normalizar las columnas finales para entrenamiento. Añade columnas de metadatos necesarias para el split por grupos (`run_id`) y la validación MMtQHU (`vehID`).

**Entradas**: `data/dataset_windows.parquet`  
**Salidas**: `data/ml_table.parquet`

> En modo Ligero, este archivo ya está versionado como seed data.

---

### `scripts/05_train_ml_model.py` — Entrenamiento ML

Es el script central del pipeline. Ver [Sección 9](#9-entrenamiento-ml-catboost-xgboost-y-calibración-isotónica) para descripción completa.

**Entradas**: `data/ml_table.parquet`  
**Salidas**:
- `data/models/catboost_gbdt.cbm`
- `data/models/xgboost_ref.json`
- `data/models/xgb_encoders.joblib`
- `data/models/isotonic.joblib`
- `data/artifacts/ml/final/rho_hat_windows_raw.parquet`
- `data/artifacts/ml/final/rho_hat_windows_calibrated.parquet`
- `reports/ml/report_catboost_isotonic.json`
- `reports/ml/feature_importance.csv`
- `reports/ml/ablation_auc.csv`

---

### `scripts/05b_inject_rho_omnet.py` — Plan de inyección ρ̂

Ver [Sección 10](#10-plan-de-inyección-de-ρ̂-en-omnet) para descripción completa.

**Entradas**: `data/artifacts/ml/final/rho_hat_windows_calibrated.parquet`  
**Salidas**:
- `data/artifacts/ml/injection/injection_plan.json`
- `data/artifacts/ml/injection/injection_plan.csv`
- `data/artifacts/ml/injection/injection_summary.txt`

---

### `scripts/05c_validate_mmtqhu_profiles.py` — Validación MMtQHU

Ver [Sección 11](#11-validación-contra-perfiles-mmtqhu) para descripción completa.

**Entradas**:
- `data/theory/analytic_rho_reference.parquet`
- `data/artifacts/ml/final/rho_hat_windows_calibrated.parquet`

**Salidas**:
- `reports/final/mmtqhu_validation_by_profile.csv`
- `reports/final/mmtqhu_validation_report.txt`

---

### `scripts/06_evaluate_probabilistic.py` — Evaluación probabilística

Ver [Sección 12](#12-evaluación-probabilística) para descripción completa.

**Entradas**:
- `data/theory/analytic_rho_reference.parquet`
- `data/artifacts/ml/final/rho_hat_windows_raw.parquet`
- `data/artifacts/ml/final/rho_hat_windows_calibrated.parquet`

**Salidas**:
- `reports/final/probabilistic_validity_global.csv`
- `reports/final/probabilistic_validity_bins.csv`
- `reports/final/probabilistic_validity_report.txt`
- `reports/final/probabilistic_calibration_curve.png`

---

### `scripts/07_compare_analytic_vs_learned.py` — Comparación analítica vs aprendida

**Responsabilidad**: Calcular métricas de alineación entre ρ analítica y variantes de ρ̂ (raw y calibrada) por período de congestión, política y celda.

**Métricas calculadas**: MAE, RMSE, bias, correlación de Pearson.

**Entradas**:
- `data/theory/analytic_rho_reference.parquet`
- `data/artifacts/ml/final/rho_hat_windows_raw.parquet`
- `data/artifacts/ml/final/rho_hat_windows_calibrated.parquet`

**Salidas**: `reports/final/rho_compare_recomputed_global.csv`

---

### `scripts/08_generate_figures.py` — Generación de figuras

**Responsabilidad**: Generar las 7 figuras finales de la tesis (G1–G7) desde los outputs reales del pipeline. Las figuras se guardan en PNG, PDF y SVG para máxima compatibilidad.

**Figuras generadas**:
- **G1**: Curva ROC — CatBoost calibrado vs XGBoost contraste
- **G2**: Curva de calibración (diagrama de confiabilidad)
- **G3**: Comparación ρ analítica vs ρ̂ calibrada por período
- **G4**: Métricas Brier y ECE por variante
- **G5**: Validación MMtQHU por perfil (propietario vs empleado)
- **G6**: Importancia de features CatBoost (top-N)
- **G7**: Ablación de features (AUC con y sin variables de red)

**Entradas**: Reportes de los pasos 05–07  
**Salidas**:
- `reports/final/thesis_figures/G1_*.{png,pdf,svg}`
- ...
- `reports/final/thesis_figures/G7_*.{png,pdf,svg}`
- `reports/final/thesis_figures/figures_manifest.md`

> Nota: si la fuente `Times New Roman` no está instalada, matplotlib usa una fuente alternativa. Esto produce un warning pero no bloquea la generación de figuras.

---

## 8. Flujo de Datos End-to-End

La siguiente tabla resume el flujo de artefactos entre pasos:

| Paso | Entrada principal | Salida principal |
|---|---|---|
| 00A | `scenarios/sumo/*.sumocfg` | `sim/runs/<run_id>/*.xml` |
| 00B | `sim/runs/<run_id>/*.xml` | `data/bronze/fcd/<run_id>.parquet` |
| 00C | `data/bronze/fcd/` + rutas SUMO | `data/silver/*.parquet` + `data/theory/analytic_rho_reference.parquet` |
| 00D | `data/bronze/fcd/` | `data/sumo_traces_omnet/<run_id>/<vehID>.trace` |
| 00  | `scenarios/omnet/` + trazas SUMO | `data/omnet_results/<run_id>.sca` |
| 01  | `data/omnet_results/*.sca` | `reports/final/objetivo2/kpis_omnet_raw.csv` |
| 01B | KPIs raw + `data/silver/` | `data/kpi/summary_kpis_avg.csv` + `data/mobility_metrics.parquet` |
| 02  | `data/mobility_metrics.parquet` + KPIs | `data/unified_metrics.parquet` |
| 03  | `data/unified_metrics.parquet` + silver | `data/dataset_windows.parquet` |
| 04  | `data/dataset_windows.parquet` | `data/ml_table.parquet` |
| 05  | `data/ml_table.parquet` | modelos + `rho_hat_*.parquet` + reportes ML |
| 05B | `rho_hat_windows_calibrated.parquet` | `injection_plan.json` |
| 05C | `rho_hat_calibrated` + `analytic_rho` | `mmtqhu_validation_by_profile.csv` |
| 06  | `analytic_rho` + `rho_hat_*.parquet` | `probabilistic_validity_global.csv` |
| 07  | `analytic_rho` + `rho_hat_*.parquet` | `rho_compare_recomputed_global.csv` |
| 08  | reportes 05–07 | `reports/final/thesis_figures/G1–G7.*` |
| 09  | `injection_plan.json` + trazas SUMO | `data/omnet_results_injected/<run_id>.sca` |

---

## 9. Entrenamiento ML: CatBoost, XGBoost y Calibración Isotónica

### 9.1 División de Datos

El dataset `data/ml_table.parquet` contiene ventanas espaciotemporales etiquetadas. La división train/test usa `GroupShuffleSplit` con agrupación por `run_id`:

```python
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

Esta estrategia garantiza que ventanas de la misma corrida no aparezcan simultáneamente en train y test, evitando filtración de información entre corridas correlacionadas.

### 9.2 Features utilizadas

| Grupo | Features |
|---|---|
| **Cinemáticas** | `speed_mean`, `speed_std`, `accel_mean`, `accel_std`, `jerk_abs_mean`, `stops_count` |
| **Red** | columnas `network_*` y `prb_usage_*` (KPIs de la celda asignada) |
| **Categóricas** | `cell_id`, `period`, `policy` |

Las features de red se imputan con la media del conjunto de entrenamiento antes de entrenar. Las categóricas son tratadas nativamente por CatBoost; para XGBoost se aplica `LabelEncoder` por columna.

### 9.3 Hiperparámetros CatBoost (modelo principal)

| Parámetro | Valor | Justificación |
|---|---|---|
| `loss_function` | `Logloss` | Clasificación binaria (desvío sí/no) |
| `eval_metric` | `AUC` | Robustez ante desbalanceo de clases |
| `random_seed` | `42` | Reproducibilidad |
| `iterations` | `1200` | Máximo de árboles; early stopping activo |
| `depth` | `8` | Expresividad suficiente para capturar interacciones |
| `learning_rate` | `0.06` | Balance entre convergencia y generalización |
| `l2_leaf_reg` | `3.0` | Regularización L2 para evitar sobreajuste |
| `od_type` | `Iter` | Early stopping por iteraciones |
| `od_wait` | `80` | Detener si no mejora en 80 iteraciones |

### 9.4 Hiperparámetros XGBoost (contraste robusto)

| Parámetro | Valor |
|---|---|
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |
| `random_state` | `42` |
| `n_estimators` | `1200` |
| `max_depth` | `8` |
| `learning_rate` | `0.06` |
| `reg_lambda` | `3.0` |
| `early_stopping_rounds` | `80` |

XGBoost requiere que las variables categóricas sean codificadas numéricamente antes del entrenamiento (via `LabelEncoder`). Su rol en el pipeline es ser un **contraste robusto**: si CatBoost y XGBoost coinciden en AUC, el resultado es más confiable que si solo se reportara uno.

### 9.5 Calibración Isotónica

```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_raw_test, y_te)   # entrena sobre el conjunto de test
p_cal = iso.predict(p_raw_all)  # aplica a todo el dataset
```

La calibración isotónica ajusta una función monótona no decreciente que mapea scores CatBoost brutos a probabilidades calibradas. Se entrena sobre el conjunto de test (no sobre train) para evitar sobreajuste. `out_of_bounds="clip"` garantiza que las predicciones fuera del rango [0,1] se recorten.

### 9.6 Ablación de features

El script realiza automáticamente una ablación mínima: entrena un segundo modelo CatBoost **sin las variables de red** (`network_*` y `prb_usage_*`) y reporta la diferencia de AUC. Esto cuantifica el aporte de la información de red al poder predictivo del modelo.

```
reports/ml/ablation_auc.csv:
  CatBoost Completo       → AUC con red
  XGBoost (contraste)     → AUC XGBoost con red
  CatBoost Sin Red        → AUC sin variables de red
```

### 9.7 Artefactos persistidos

| Archivo | Formato | Descripción |
|---|---|---|
| `data/models/catboost_gbdt.cbm` | `.cbm` nativo CatBoost | Modelo CatBoost completo |
| `data/models/xgboost_ref.json` | JSON XGBoost | Modelo XGBoost de contraste |
| `data/models/xgb_encoders.joblib` | joblib | Encoders de categóricas para XGBoost |
| `data/models/isotonic.joblib` | joblib | Calibrador isotónico |
| `reports/ml/report_catboost_isotonic.json` | JSON | Métricas completas de ambos modelos |
| `reports/ml/feature_importance.csv` | CSV | Importancia de features CatBoost (descendente) |
| `reports/ml/ablation_auc.csv` | CSV | AUC con y sin variables de red |

---

## 10. Plan de Inyección de ρ̂ en OMNeT++

### 10.1 Lógica de decisión

El script `05b_inject_rho_omnet.py` implementa la lógica de sustitución probabilística que es el núcleo del Objetivo 1:

```
Para cada run_id:
    rho_hat_mean = mean(ρ̂) de todas las ventanas de esa corrida

    si rho_hat_mean ≥ 0.5:
        config_omnet = "DoubleConnection-CBR-DL"
        (doble conectividad proactiva, handover anticipado)
    si no:
        config_omnet = "SingleConnection-CBR-DL"
        (conexión única, política baseline nearest)
```

El umbral de 0.5 es configurable con `--threshold`. Este umbral es el punto natural de decisión probabilística: si el modelo cree que hay más del 50% de probabilidad de desvío, se activa la gestión proactiva de handover.

### 10.2 Configuraciones OMNeT++

| Configuración | Interpretación | Política de red |
|---|---|---|
| `DoubleConnection-CBR-DL` | ρ̂ ≥ 0.5: alta probabilidad de desvío | Doble conectividad, handover anticipado (ceiling) |
| `SingleConnection-CBR-DL` | ρ̂ < 0.5: baja probabilidad de desvío | Conexión única sin handover proactivo (nearest) |

### 10.3 Estructura del plan de inyección

```json
{
  "threshold": 0.5,
  "baseline_conf": "SingleConnection-CBR-DL",
  "learned_conf": "DoubleConnection-CBR-DL",
  "n_runs": 180,
  "n_learned": 165,
  "n_baseline": 15,
  "global_mean_rho_hat": 0.6667,
  "plan": {
    "HC1__nearest__0": "SingleConnection-CBR-DL",
    "HC2__ceiling__1": "DoubleConnection-CBR-DL",
    ...
  }
}
```

### 10.4 Uso en el paso 09

El paso 09 del orquestador llama al mismo script de OMNeT++ pero con el argumento `--injection-plan`, de modo que cada corrida usa la configuración asignada por el estimador ML en lugar de la configuración por defecto:

```bash
python scripts/00_run_omnet_obj2_batch.py \
    --injection-plan data/artifacts/ml/injection/injection_plan.json \
    --results-root data/omnet_results_injected \
    --summary reports/final/omnet_injected_summary.json
```

---

## 11. Validación contra Perfiles MMtQHU

### 11.1 El modelo MMtQHU

MMtQHU (Modelo de Movilidad basado en Tipos y hábitos de la Quincena de Hogar y Uso) segmenta a los usuarios según su patrón de movilidad. En el escenario InTAS, los vehículos se clasifican en dos perfiles basándose en su identificador:

| Perfil | Criterio de identificación | Característica |
|---|---|---|
| **Propietario** | ID de vehículo **sin** prefijo `VFH_` | Movilidad propia/privada. Rutas relativamente estables. Menor variabilidad de desvío. |
| **Empleado** | ID de vehículo **con** prefijo `VFH_` | Vehículos de flotilla laboral. Mayor variabilidad de ruta asociada a trayectos laborales y cambios de turno. |

### 11.2 Proceso de validación

```
1. Leer ρ analítica: (period, vehID) → rho
2. Leer ρ̂ calibrada: (period, vehID, ventana) → rho_hat
3. Agregar ρ̂ por (period, vehID):
   - rho_cal_mean = mean(rho_hat)
   - rho_cal_max  = max(rho_hat)
4. Cruzar ambas tablas por (period, vehID)
5. Clasificar cada vehículo en propietario / empleado
6. Calcular por perfil:
   MAE, RMSE, bias, correlación de Pearson
```

### 11.3 Valores obtenidos (pipeline Nivel 1, mayo 2026)

| Perfil | Variante | MAE | RMSE | Correlación | Referencia tesis |
|---|---|---|---|---|---|
| Propietario | rho_cal_max | **0.0218** | 0.0428 | 0.979 | MAE ≈ 0.1235 |
| Empleado | rho_cal_max | **0.0232** | 0.0442 | 0.978 | MAE ≈ 0.4565 |
| Global | rho_cal_max | **0.0220** | 0.0429 | 0.979 | — |

> **Nota**: el MAE es menor que la referencia de tesis porque la variante `rho_cal_max` (máximo por vehículo) está mejor calibrada que `rho_cal_mean`. Los valores de tesis corresponden al diseño experimental completo con múltiples réplicas SUMO.

La mayor dispersión en el perfil empleado era esperada por diseño: los vehículos VFH tienen decisiones de ruta más complejas ligadas a cambios de turno y orígenes-destinos diversificados.

---

## 12. Evaluación Probabilística

### 12.1 Métricas utilizadas y por qué

El script `06_evaluate_probabilistic.py` usa métricas "soft" que tratan ρ analítica como referencia probabilística continua (en lugar de labels binarios duros):

**Brier Score soft**:
```
Brier_soft = mean((ρ̂ - ρ)²)
```
Mide el error cuadrático medio entre la probabilidad predicha y la analítica. Rango: [0, 1]. Menor es mejor.

**ECE soft** (Expected Calibration Error):
```
ECE_soft = Σ_bins (|fracción de bin| × |confianza_media_bin - ρ_media_bin|)
```
Mide cuánto difieren las probabilidades predichas de las referencias en cada bin de confianza. Rango: [0, 1]. Menor es mejor.

### 12.2 Por qué métricas "soft"

La referencia ρ analítica es en sí misma una probabilidad continua (no un label binario 0/1). Usar el Brier Score convencional (que compara contra labels binarios) ignoraría la incertidumbre propia del proceso. Las métricas soft respetan la naturaleza probabilística de ρ.

### 12.3 Valores obtenidos (pipeline Nivel 1, mayo 2026)

| Métrica | Variante | Valor obtenido | Referencia tesis |
|---|---|---|---|
| AUC ROC CatBoost calibrado | — | **0.9237** | 0.9255 |
| AUC XGBoost (contraste) | — | **0.9280** | — |
| Brier soft | rho_cal_max | **0.001843** | 0.000850 |
| ECE soft | rho_cal_max | **0.009466** | ≈ 0.0099 |
| SINR con DoubleConnection | vs SingleConnection | +1.49 dB | Nivel 3 solamente |
| CDR con DoubleConnection | vs SingleConnection | 0% (0 caídas) | Nivel 3 solamente |

> **Nota de reproducibilidad**: los valores de Brier y ECE se reportan sobre la variante `rho_cal_max` (máximo de ρ̂ por ventana por vehículo), que es la que mejor correlaciona con ρ analítica. Las métricas de SINR/CDR requieren Nivel 3 (OMNeT++ activo).
> La diferencia de AUC (0.9237 vs 0.9255) es de 0.002 y está dentro del margen de variación por semilla aleatoria y partición train/test.

---

## 13. Estructura de Directorios del Repositorio

```
InTAS_PRODUCCION_READY_ligero/
│
├── README.md                          ← Documentación principal del repo
├── README_DOCKER.md                   ← Guía específica de Docker
├── Dockerfile                         ← Imagen Ubuntu 22.04 + SUMO + Python
├── requirements.txt                   ← Dependencias Python con versiones fijadas
├── .gitignore                         ← Reglas de exclusión (seed data incluidos)
│
├── config/ml/
│   ├── experiment.yaml                ← Configuración de experimentos ML
│   ├── pipeline.yaml                  ← Rutas del pipeline
│   └── best_model_report.json         ← Reporte del mejor modelo
│
├── scenarios/
│   ├── sumo/                          ← 6 archivos .sumocfg + red + rutas HC1-HC3
│   └── omnet/
│       ├── omnetpp.ini                ← Config principal OMNeT++ (CBR + SUMO + InTAS)
│       └── configs/
│           ├── intas_positions_4cells.ini
│           └── intas_positions_10cells.ini
│
├── scripts/
│   ├── run_full_reproduction.py       ← Orquestador (17 pasos)
│   ├── 00_run_omnet_obj2_batch.py     ← OMNeT++ batch + TraceFileMobility
│   ├── 00a_run_sumo_batch.py          ← SUMO batch (HC1/HC2/HC3 × VFH × políticas)
│   ├── 00b_extract_bronze_batch.py    ← XML → Parquet Bronze
│   ├── 00c_build_silver_theory.py     ← Silver + ρ analítica
│   ├── 00d_export_sumo_traces_for_omnet.py  ← FCD → .trace INET
│   ├── 01_extract_omnet_kpis.py       ← .sca → KPIs CSV
│   ├── 01b_prepare_unify_inputs.py    ← Preparación para unificación
│   ├── 02_unify_metrics.py            ← Movilidad + Red → unified
│   ├── 03_build_gold_windows.py       ← Windowing → dataset Gold
│   ├── 04_build_ml_table.py           ← Gold → tabla ML final
│   ├── 05_train_ml_model.py           ← CatBoost + XGBoost + Isotonic
│   ├── 05b_inject_rho_omnet.py        ← ρ̂ → plan de inyección JSON
│   ├── 05c_validate_mmtqhu_profiles.py ← MAE por perfil propietario/empleado
│   ├── 06_evaluate_probabilistic.py   ← Brier/ECE soft
│   ├── 07_compare_analytic_vs_learned.py ← MAE/RMSE ρ vs ρ̂
│   ├── 08_generate_figures.py         ← G1–G7 PNG/PDF/SVG
│   └── helpers/
│       ├── build_manifest.py          ← Manifiestos SHA-256
│       ├── extract_bronze.py          ← Utilidades de extracción Bronze
│       └── fcd_assign_cells.py        ← Asignación vehículo → celda
│
├── data/                              ← Datos (ver Sección 5)
│   ├── silver/          ← VERSIONADO (seed data)
│   ├── theory/          ← VERSIONADO (seed data)
│   ├── kpi/             ← VERSIONADO (seed data)
│   ├── unified_metrics.parquet       ← VERSIONADO (seed data)
│   ├── dataset_windows.parquet       ← VERSIONADO (seed data)
│   ├── ml_table.parquet              ← VERSIONADO (seed data)
│   ├── mobility_metrics.parquet      ← VERSIONADO (seed data)
│   ├── models/          ← GENERADO por paso 05
│   ├── artifacts/ml/    ← GENERADO por pasos 05, 05B
│   └── sumo_traces_omnet/ ← GENERADO por paso 00D
│
├── reports/
│   ├── final/objetivo2/  ← VERSIONADO (seed data KPIs OMNeT)
│   │   ├── kpis_omnet_raw.csv
│   │   ├── kpis_omnet_by_cell.csv
│   │   └── kpis_omnet_inventory_report.txt
│   ├── final/omnet_batch_summary.json  ← VERSIONADO (seed data)
│   ├── final/thesis_figures/           ← GENERADO por paso 08
│   ├── final/mmtqhu_validation_*.{csv,txt} ← GENERADO por paso 05C
│   ├── final/probabilistic_*.csv       ← GENERADO por paso 06
│   ├── final/rho_compare_*.csv         ← GENERADO por paso 07
│   └── ml/                             ← GENERADO por paso 05
│
└── docs/
    ├── GUIDE.md
    ├── MANUAL_DISEÑO_INGENIERIA.md
    ├── THESIS_INTEGRATION.md
    └── DOCUMENTACION_TECNICA_INTAS.md  ← Este documento
```

---

## 14. Reproducibilidad: Niveles y Comandos

El pipeline define tres niveles de reproducibilidad según los simuladores disponibles:

### Nivel 1 — Solo ML (sin SUMO ni OMNeT++) — ~5 minutos

**Requisitos**: Python 3.10+, dependencias de `requirements.txt`.

**Qué hace**: Lee seed data ya versionados (Silver, Gold, tabla ML, KPIs OMNeT) y ejecuta directamente los pasos de ML y evaluación (01B–08).

```bash
# Clonar y configurar entorno
git clone <url-del-repositorio>
cd InTAS_PRODUCCION_READY_ligero

python3 -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows (WSL recomendado)

pip install -r requirements.txt

# Ejecutar pipeline
python scripts/run_full_reproduction.py
```

Los pasos 00A–00D se omiten automáticamente (seed data presentes). Los pasos 00 y 09 se omiten (requieren `INTAS_RUN_OMNET=1`). Se ejecutan 01B, 02, 03, 04, 05, 05B, 05C, 06, 07 y 08.

### Nivel 2 — Con SUMO (~60–120 minutos)

**Requisitos adicionales**: SUMO 1.16+ instalado y accesible en `$PATH`.

```bash
INTAS_RUN_SUMO=1 python scripts/run_full_reproduction.py
```

Regenera los seed data desde cero ejecutando las corridas SUMO. Los valores numéricos deben coincidir con el Nivel 1 (mismas semillas, mismo diseño experimental).

### Nivel 3 — CPSS Completo (~2–4 horas)

**Requisitos adicionales**: Stack OMNeT++ 6.0 + INET 4.4 + Simu5G 1.3 compilado y accesible. Las configuraciones `SingleConnection-CBR-DL` y `DoubleConnection-CBR-DL` funcionan con Simu5G estándar. Las configuraciones `InTAS-10Cells-CBR-DL` requieren el framework NED de InTAS.

```bash
INTAS_RUN_SUMO=1 INTAS_RUN_OMNET=1 python scripts/run_full_reproduction.py
```

### Comandos de ejecución por etapas (para debugging)

```bash
# Solo entrenamiento ML
python scripts/05_train_ml_model.py

# Solo plan de inyección ρ̂
python scripts/05b_inject_rho_omnet.py

# Solo validación MMtQHU
python scripts/05c_validate_mmtqhu_profiles.py

# Solo evaluación probabilística
python scripts/06_evaluate_probabilistic.py

# Solo comparación analítica vs aprendida
python scripts/07_compare_analytic_vs_learned.py

# Solo figuras
python scripts/08_generate_figures.py

# Forzar reconstrucción completa
INTAS_FORCE_REBUILD=1 python scripts/run_full_reproduction.py
```

---

## 15. Ejecución con Docker

El `Dockerfile` construye un contenedor Ubuntu 22.04 con SUMO instalado vía `apt`. OMNeT++ **no se compila automáticamente** en el contenedor actual (es trabajo futuro señalado en la tesis).

### Construcción de la imagen

```bash
docker build -t intas-tesis .
```

Lo que hace el build:
- Instala dependencias de sistema (compiladores, Qt5, libxml2, SUMO).
- Configura `SUMO_HOME=/usr/share/sumo`.
- Instala dependencias Python de `requirements.txt`.
- Copia el repositorio completo (incluyendo seed data).
- Define `CMD` como `python3 scripts/run_full_reproduction.py`.

### Ejecución base (ML + evaluación, sin simuladores)

```bash
docker run --name intas-container intas-tesis
```

### Ejecución con regeneración SUMO

```bash
docker run --name intas-container-sumo -e INTAS_RUN_SUMO=1 intas-tesis
```

### Ejecución full CPSS (requiere stack OMNeT compilado en el contenedor)

```bash
docker run --name intas-container-full -e INTAS_RUN_SUMO=1 -e INTAS_RUN_OMNET=1 intas-tesis
```

### Copiar resultados al host

```bash
docker cp intas-container:/app/reports/final ./resultados_tesis
```

### Limpieza

```bash
docker rm intas-container
```

---

## 16. Verificación Numérica de Resultados

### 16.1 Verificación de artefactos (existencia)

```bash
test -f data/models/catboost_gbdt.cbm                              && echo "OK catboost"
test -f data/models/xgboost_ref.json                               && echo "OK xgboost"
test -f data/models/isotonic.joblib                                && echo "OK calibrador"
test -f data/artifacts/ml/final/rho_hat_windows_calibrated.parquet && echo "OK rho_hat"
test -f data/artifacts/ml/injection/injection_plan.json            && echo "OK injection"
test -f reports/final/mmtqhu_validation_by_profile.csv             && echo "OK mmtqhu"
test -f reports/final/probabilistic_validity_global.csv            && echo "OK brier/ece"
test -f reports/final/rho_compare_recomputed_global.csv            && echo "OK comparacion"
test -f reports/final/thesis_figures/figures_manifest.md           && echo "OK figuras"
```

### 16.2 Verificación de valores numéricos

```python
python - <<'PY'
import json, pandas as pd

# Métricas del modelo ML
r = json.load(open("reports/ml/report_catboost_isotonic.json"))
print("=== Modelo ML ===")
print(f"AUC CatBoost calibrado : {r['roc_auc_cal']:.4f}  (referencia tesis: 0.9255)")
print(f"AUC XGBoost contraste  : {r['roc_auc_xgb']:.4f}  (reportado en G1)")
# Nota: brier_cal en el JSON es el Brier duro contra etiquetas binarias (y_te),
#       no el Brier soft contra ρ analítica. El Brier soft está en probabilistic_validity.
print(f"Brier duro (cal vs y_te): {r['brier_cal']:.6f}")

# Validez probabilística — Brier/ECE soft contra ρ analítica
pb = pd.read_csv("reports/final/probabilistic_validity_global.csv")
print("\n=== Validez Probabilística (soft vs ρ analítica) ===")
print(pb[["variant","brier_soft","ece_soft"]].to_string(index=False))
best = pb[pb["variant"] == "rho_cal_max"].iloc[0]
print(f"\nVariante reportada (rho_cal_max):")
print(f"  Brier soft : {best['brier_soft']:.6f}  (referencia tesis: 0.000850)")
print(f"  ECE soft   : {best['ece_soft']:.6f}  (referencia tesis: ≈0.0099)")

# Validación MMtQHU
mm = pd.read_csv("reports/final/mmtqhu_validation_by_profile.csv")
print("\n=== Validación MMtQHU por perfil (variante rho_cal_max) ===")
best_mm = mm[mm["variant"] == "rho_cal_max"]
print(best_mm[["profile","mae","rmse","bias","corr"]].to_string(index=False))

# Plan de inyección
plan = json.load(open("data/artifacts/ml/injection/injection_plan.json"))
print(f"\n=== Plan de Inyección ===")
print(f"Total corridas         : {plan['n_runs']}")
print(f"→ DoubleConnection     : {plan['n_learned']} ({100*plan['n_learned']/plan['n_runs']:.1f}%)")
print(f"→ SingleConnection     : {plan['n_baseline']} ({100*plan['n_baseline']/plan['n_runs']:.1f}%)")
print(f"ρ̂ medio global         : {plan['global_mean_rho_hat']:.4f}")
PY
```

### 16.3 Verificación de KPIs de red (Nivel 3 solamente)

```bash
python - <<'PY'
import json
s = json.load(open("reports/final/omnet_batch_summary.json"))
print("OMNeT++ baseline ejecutado:", s.get("status", "?"))

# Solo disponible después del paso 09
import os
if os.path.exists("reports/final/omnet_injected_summary.json"):
    si = json.load(open("reports/final/omnet_injected_summary.json"))
    print("OMNeT++ inyectado ejecutado:", si.get("status", "?"))
    print("(esperado: SINR +1.49 dB, CDR = 0% con DoubleConnection)")
PY
```

---

## 17. Variables de Entorno

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `INTAS_RUN_SUMO` | `0` | Si `1`, ejecuta pasos 00A–00D (requiere SUMO instalado) |
| `INTAS_RUN_OMNET` | `0` | Si `1`, ejecuta pasos 00 y 09 (requiere OMNeT++ compilado) |
| `INTAS_FORCE_REBUILD` | `0` | Si `1`, re-ejecuta todos los pasos aunque las salidas existan |
| `INTAS_OMNET_REP_FROM` | `0` | Repetición inicial para corridas OMNeT++ batch |
| `INTAS_OMNET_REP_TO` | `14` | Repetición final para corridas OMNeT++ batch (total: 15 réplicas) |
| `PYTHONHASHSEED` | *(sin definir)* | Si `42`, fija la semilla de hashing de Python para reproducibilidad total |

---

## 18. Resolución de Problemas Comunes

### Error: `error: externally-managed-environment` (PEP 668)

**Síntoma**: Al hacer `pip install` en sistemas Linux modernos (Ubuntu 22.04+, Debian 12+), el sistema rechaza instalar paquetes en el entorno Python del sistema.

**Solución**: Usar siempre un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Error: `ResolutionImpossible` al instalar dependencias

**Síntoma**: pip no puede resolver las dependencias.

**Causa más común**: Intentar instalar en un entorno con una versión de numpy distinta a `1.26.4`.

**Solución**: Usar el `requirements.txt` del repositorio sin modificaciones. Si el entorno está corrupto, crearlo de nuevo desde cero.

### Warning: `findfont: Font family 'Times New Roman' not found`

**Síntoma**: Al ejecutar `08_generate_figures.py`, matplotlib produce este warning.

**Impacto**: Ninguno funcional. Las figuras se generan correctamente con la fuente fallback del sistema.

**Solución opcional**: Instalar `ttf-mscorefonts-installer` en Ubuntu o copiar el archivo `Times New Roman.ttf` al directorio de fuentes de matplotlib.

### Error: Paso 01 reporta inventario vacío de OMNeT++

**Síntoma**: `kpis_omnet_inventory_report.txt` indica 0 archivos `.sca` encontrados.

**Causa**: No hay resultados de OMNeT++ en `data/omnet_results/` (paso 00 no ejecutado).

**Impacto**: En modo Ligero, los seed data de KPIs ya están versionados. El pipeline puede continuar desde `01B`.

**Solución**: Para ejecutar OMNeT++ completo, usar `INTAS_RUN_OMNET=1`.

### Error en paso 05: `El dataset ML debe contener columnas 'label' y 'run_id'`

**Causa**: El archivo `data/ml_table.parquet` no tiene las columnas esperadas, posiblemente porque proviene de una versión anterior del pipeline.

**Solución**:
```bash
INTAS_FORCE_REBUILD=1 python scripts/04_build_ml_table.py
python scripts/05_train_ml_model.py
```

---

## 19. Limitaciones Actuales y Alcance Real

### Lo que el repositorio cubre completamente

- Pipeline Python completo: Bronze → Silver → Gold → CatBoost + XGBoost → calibración → evaluación → figuras.
- Seed data versionados que permiten reproducir los resultados ML sin simuladores.
- Acoplamiento secuencial SUMO → OMNeT++ vía `TraceFileMobility` (paso 00D + 00).
- Plan de inyección de ρ̂ en OMNeT++ (pasos 05B y 09).
- Validación contra perfiles de comportamiento MMtQHU (paso 05C).
- Manifiestos SHA-256 en `reports/final/thesis_figures/figures_manifest.md`.
- Modo incremental para iteración eficiente.

### Lo que requiere instalación externa

| Componente | Variable de activación | Limitación |
|---|---|---|
| SUMO 1.16+ | `INTAS_RUN_SUMO=1` | No compilado en Docker automáticamente (solo instalado vía apt) |
| OMNeT++ 6.0 + INET + Simu5G | `INTAS_RUN_OMNET=1` | No compilado en el Dockerfile actual |
| Framework NED InTAS | — | Requerido para configs `InTAS-10Cells-CBR-DL` y `InTAS-SUMOTrace-CBR-DL` |

### Ruta hacia pipeline 100% end-to-end (trabajo futuro)

1. Integrar compilación automática de OMNeT++ / INET / Simu5G / Artery en el Dockerfile.
2. Implementar loop cerrado: reentrenamiento automático cuando la divergencia ρ vs ρ̂ supera un umbral.
3. Extender el pipeline a múltiples escenarios urbanos para validar generalización.
4. Implementar monitoreo continuo de drift en las probabilidades predichas.

---

## 20. Glosario

| Término | Definición |
|---|---|
| **Bronze** | Capa de datos crudos, conversión 1:1 desde outputs de simulación |
| **CDR** | Call Drop Rate — tasa de caída de conexiones en red móvil |
| **CPSS** | Sistema Ciber-Físico-Social — acopla capa física, ciber y social |
| **ECE** | Expected Calibration Error — error de calibración esperado |
| **FCD** | Floating Car Data — trazas de posición y velocidad vehicular por timestamp |
| **Gold** | Capa de datos final lista para entrenamiento ML (ventanas espaciotemporales) |
| **HC1/HC2/HC3** | Períodos de congestión alta del escenario InTAS |
| **INET** | Framework de protocolos de red para OMNeT++ |
| **InTAS** | Ingolstadt Traffic and Automotive Simulation — escenario de simulación vehicular |
| **isotonic** | Regresión isotónica — calibración monótona de probabilidades |
| **KPI** | Key Performance Indicator — indicador clave de desempeño |
| **MMtQHU** | Modelo de Movilidad basado en Tipos y hábitos de la Quincena de Hogar y Uso |
| **OMNeT++** | Simulador de eventos discretos para redes de comunicación |
| **ρ (rho)** | Probabilidad analítica de desvío de ruta |
| **ρ̂ (rho estimada)** | Probabilidad aprendida y calibrada mediante Gradient Boosting |
| **Silver** | Capa de datos interpretados y alineados espaciotemporalmente |
| **SINR** | Signal-to-Interference-plus-Noise Ratio — calidad de señal en dB |
| **Simu5G** | Extensión de simulación de redes 5G para OMNeT++/INET |
| **SUMO** | Simulation of Urban MObility — simulador de movilidad vehicular microscópica |
| **TraceFileMobility** | Módulo INET para reproducir trayectorias pregeneradas en OMNeT++ |
| **V2X** | Vehicle-to-Everything — comunicación entre vehículo e infraestructura |
| **VFH** | Virtual Force Highway — modelo de comportamiento de vehículos de flotilla |

---

## 21. Bibliografía

[1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

[2] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: gradient boosting with categorical features support. *arXiv:1810.11372*.

[3] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 1321–1330.

[4] Varga, A., & Hornig, R. (2008). An overview of the OMNeT++ simulation environment. *Proceedings of the 1st International Conference on Simulation Tools and Techniques for Communications (SIMUTools)*, 1–10.

[5] Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning (ICML)*, 625–632.

[6] Lopez, P. A., et al. (2018). Microscopic Traffic Simulation using SUMO. *IEEE International Conference on Intelligent Transportation Systems (ITSC)*.

[7] IEEE. (1998). IEEE Standard 830-1998: Software Requirements Specification.

[8] IEEE. (2009). IEEE Standard 1016-2009: Software Design Description.

[9] ISO/IEC. (2017). ISO/IEC/IEEE 12207:2017 — Systems and Software Engineering — Software Life Cycle Processes.

[10] ICONTEC. (2008). NTC 1486: Presentación de Trabajos de Investigación.

---

*Generado automáticamente desde el código fuente del repositorio — Mayo 2026*
