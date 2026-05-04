# OBJETIVO 2 - TEXTO DE INTEGRACION A TESIS
## Impacto de la politica aprendida (rho_hat) sobre KPIs de red en OMNeT++

Fecha de corte: 2026-03-16
Muestra analizada: 183 pares alineados (baseline vs learned)
Cobertura objetivo: HC1/HC2/HC3, VFH 5% y 10%, politicas nearest/ceiling, repeticiones 1..14

## 1. Texto sugerido para capitulo de resultados

Con el fin de evaluar el impacto de la politica aprendida (rho_hat) sobre indicadores de calidad de red, se ejecuto una comparacion pareada baseline vs learned sobre 183 corridas de OMNeT++, manteniendo constante la configuracion experimental por periodo, nivel de demanda y repeticion. El analisis se realizo sobre CDR, throughput global y SINR, complementado con metricas por celda para capturar cambios de calidad de enlace en el plano radio.

Los resultados agregados muestran que CDR y throughput global permanecen invariantes entre baseline y learned, mientras que la SINR presenta una mejora media de +1.489 unidades en la condicion learned. Esta mejora es consistente en todos los periodos (HC1, HC2 y HC3) y en ambas politicas de asignacion evaluadas (nearest y ceiling), con mayor magnitud en HC3 y en la politica ceiling.

Desde una perspectiva de interpretacion, los resultados indican que la politica aprendida no degrada el rendimiento agregado del trafico de aplicacion, y simultaneamente mejora la calidad de canal observada (SINR), lo cual sugiere una asignacion de recursos radio mas favorable. Adicionalmente, el throughput por celda incrementa de forma sistematica en learned frente a baseline en todos los grupos evaluados.

## 2. Texto sugerido para discusion estadistica

El contraste pareado reporta significancia estadistica para SINR (p = 2.923e-22), lo que respalda que la mejora observada no se explica por variacion aleatoria en la muestra. En CDR y throughput global no se reporta p-valor interpretable (p = NaN) debido a ausencia de variabilidad efectiva en las series comparadas (valores practicamente constantes), por lo que se concluye equivalencia empirica en esas metricas bajo el escenario evaluado.

En terminos practicos, la evidencia sostiene que rho_hat aporta una ganancia de calidad de enlace sin penalizar la capacidad global ni la entrega de trafico a nivel agregado.

## 3. Parrafo de conclusion listo para tesis

En el Objetivo 2 se verifico que la calibracion aprendida rho_hat mejora de manera robusta la calidad de enlace en la red celular (SINR), manteniendo estable el desempeno agregado en CDR y throughput. Por tanto, los resultados confirman un impacto positivo y no regresivo de la estrategia learned respecto al baseline, aportando evidencia experimental de su conveniencia para escenarios de alta demanda y heterogeneidad espacial.

## 4. Amenazas a la validez y alcance

- CDR y throughput global presentan dinamica saturada en esta configuracion, por lo que el efecto principal se manifiesta en SINR y throughput por celda.
- La inferencia esta acotada al espacio experimental cubierto (HC1-3, VFH 5/10, nearest/ceiling, rep 1..14).
- Se recomienda extender con otras semillas y perfiles de carga para fortalecer validez externa.

## 5. Checklist reproducible (comandos)

Ejecutar desde la raiz de `InTAS_PRODUCCION_READY`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_reproduction.py
```

## 6. Evidencia y artefactos resultantes

- CSV de comparacion detallada: kpi_comparison_detailed.csv
- Resumen por periodo: kpi_by_period.csv
- Resumen por politica: kpi_by_policy.csv
- Figura principal de impacto: fig_o2_kpi_impact.png
- Boxplot comparativo: fig_o2_boxplot.png
- Reporte final consolidado: OBJETIVO_2_FINAL_REPORT.md
