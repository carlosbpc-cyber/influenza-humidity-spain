# Humidity anomalies, influenza timing and an age-structured regional model for Spain

Code and data for the manuscript:

> *Year-to-year humidity anomalies, not geographical gradients, modestly modulate influenza timing in Europe:
> multi-country and regional analyses with an age-structured transmission model, 2010–2027.*
> Báguena Pérez-Crespo C, et al. (submitted to *Eurosurveillance*, 2026).

The repository reproduces every analysis, table and figure of the revised manuscript and of
Supplementary Material S1, starting from open data. Code comments and variable names are in Spanish;
this README and all outputs used in the paper are in English.

## Requirements

Python ≥ 3.10 and the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

All inputs needed by the main pipeline are included in `datos/` and `datos_nuevos/`, so nothing needs to be
downloaded to reproduce the results. `descargar_datos.py` re-downloads the climate, mobility and tourism
inputs from their original sources (see *Data* below).

## Quick start

```bash
bash run_all.sh            # full pipeline, ~1–2 h on a laptop (calibrations dominate)
bash run_all.sh --quick    # skips the calibrations and uses the stored parameters, ~5 min
```

Results are written to `resultados/` and figures to `figures/` (PNG, PDF and SVG). Stored copies of all
outputs are already included, so each step can also be run on its own.

## Pipeline and correspondence with the manuscript

Run from the repository root, in this order:

| Step | Script | Produces | Manuscript |
|---|---|---|---|
| 0 | `descargar_datos.py` (optional) | `datos_nuevos/` | Table 1 |
| 1 | `analisis_europa_largo.py` | `resultados/europa_largo_efectos.json` | Results (Europe), Fig. 1, Table S6 |
| 2 | `analisis_svge_regional.py` | `resultados/svge_regional_efectos.json` | Results (Spanish regions), Fig. 2, Table S7 |
| 3 | `humedad_ventanas.py` | `resultados/humedad_ventanas_*.json`, `*_ventanas_tabla.csv` | Humidity windows, leave-one-season-out and placebo tests; Figs. 1–2, Tables S6–S7 |
| 4 | `calibrar_v4.py` | `resultados/parametros_v4.json` | Calibration, emulation of κ and θ; Table S4 |
| 5 | `calibrar_ihr_v4.py` | `resultados/ihr_v4.json` | Infection–hospitalisation ratios |
| 6 | `validacion_loso.py` | `resultados/validacion_loso*.csv`, `parametros_loso.json` | Out-of-sample validation; Table 3, Table S10 |
| 7 | `prediccion_inicio.py` | `resultados/prediccion_inicio*.csv/json` | Out-of-sample prediction updated at epidemic onset; Table 3, Fig. 3, Table S10 |
| 8 | `validacion_hosp.py` | `resultados/validacion_hospitalizaciones.csv` | Out-of-sample hospitalisations; Table S11 |
| 9 | `enkf_v4.py`, `enkf_loso.py` | `resultados/enkf_*` | Real-time updating (EAKF); Table S5 |
| 10 | `escenarios_v4.py` | `resultados/resultados_v4.json`, `resultados_ccaa_v4.csv` | Projections 2026/27; Table 2, Fig. 4 |
| 11 | `figuras_revision.py` | `figures/figure1–4.*` | Figures 1–4 |

Supplementary analyses that need inputs not redistributed here (see *Data*):

| Script | Input to add | Manuscript |
|---|---|---|
| `analisis_antigenico.py` | `datos/antigenico/north-america_summarized_mean_seasonal_distances_h3n2_ha_21y.tsv` (Perofsky et al., *eLife* 2024, data repository) | A(H3N2) antigenic advance; Table S8 |
| `analisis_hemisferio_sur.py` | `datos/flunet/VIW_FNT.csv` (WHO FluNet export) | Southern-hemisphere season; Table S8 |

Their stored outputs (`resultados/antigenico_europa.json`, `resultados/hemisferio_sur.json`) are included.

### Model modules

| Module | Content |
|---|---|
| `modelo_gripe_ccaa_v2.py` | 17 regions, population, MITMA mobility mixing matrix, FRONTUR arrivals, ERVISS European positivity |
| `modelo_v3.py` | Seasonal cycle, specific humidity from ERA5 dew point and pressure, daily humidity anomalies |
| `datos_v4.py` | Age groups (0–14, 15–64, ≥65), Prem et al. contact matrices, school calendars, vaccination coverage and effectiveness |
| `modelo_v4.py` | Age-structured SEIR metapopulation model, surveillance proxy and hospitalisations |
| `calibrar.py` | SiVIRA calibration seasons 2022/23–2025/26 (national influenza proxy, climate, European positivity) |
| `datos_sivira_regional.py` | Regional influenza proxy (ARI rate × regional positivity) for 2024/25 and 2025/26 |

### Options (environment variables)

Defaults reproduce the revised manuscript.

| Variable | Default | Meaning |
|---|---|---|
| `VENTANA_MESES` | `10,11,12` | Months of the humidity anomaly used to emulate κ and in scenarios (`11,12,1` reproduces the original submission) |
| `EFECTO_ANOM_SEM` | `1.0` | Target delay (weeks per g/kg) for the emulation of κ |
| `DQ_ESCENARIO` | `0.6` | Humidity anomaly (g/kg) of the humid and dry scenarios |
| `MAXITER` | `2500` | Nelder–Mead iterations in `validacion_loso.py` |
| `IHR_FIJO` | unset | `1` ignores `resultados/ihr_v4.json` and uses default IHRs |

Sensitivity analysis of the supplement (κ emulated with the November–January window and 1.98 weeks per g/kg):
`VENTANA_MESES=11,12,1 EFECTO_ANOM_SEM=1.98 python3 calibrar_v4.py` (run in a copy of the repository, as it
overwrites `resultados/parametros_v4.json`).

## Data

| Folder / file | Content | Source | Terms |
|---|---|---|---|
| `datos_nuevos/clima/` | Daily ERA5 temperature, dew point and surface pressure at regional and European capitals | Copernicus ERA5 via Open-Meteo | CC BY 4.0 |
| `datos/clima_ccaa_2010_2019.json` | First ERA5 climate series used in the original submission | Copernicus ERA5 | CC BY 4.0 |
| `datos_nuevos/movilidad_ccaa_mitma.csv` | Trips between regions (4 days, 2024), aggregated from municipal data | Ministry of Transport (MITMA) open mobility data | CC BY 4.0 |
| `datos_nuevos/frontur_10823.json`, `poblacion_2853.json` | International tourist arrivals by region; population | INE (FRONTUR table 10823; table 2853) | INE reuse terms (attribution) |
| `datos/erviss/` | European sentinel positivity, ILI/ARI rates and subtypes | ECDC ERVISS (GitHub) | ECDC reuse policy (attribution) |
| `datos/europa_subtipos.csv` | Type/subtype shares for Spain | ERVISS and SiVIRA reports | as sources |
| `datos/sivira_*.json`, `datos/sivira_edad/` | National and regional influenza proxy and hospitalisation proxy by age | SiVIRA weekly reports, Instituto de Salud Carlos III | public reports |
| `datos/svge_redes_2010_2019.txt` | Regional peak weeks by sentinel network, 2010/11–2018/19 | SVGE annual reports (ISCIII) | public reports |
| `datos/contactos_esp_3edades.json` | Contact matrices aggregated to three age groups | Prem et al., *PLoS Comput Biol* 2021 | CC BY 4.0 |
| `datos/poblacion_edad_ccaa.csv` | Population shares by age and region | Ministry of Health vaccination denominators (compiled by Datadista) | as source |
| `datos/antigenico/nextstrain_ctiter_6y.json` | A(H3N2), A(H1N1)pdm09 and B/Victoria cTiter summaries | Nextstrain | CC BY 4.0 |
| `resultados/europa_largo_tabla.csv` | Country-season analysis table (22 EU/EEA countries) | Derived from WHO FluNet and ERA5 as described in S1.7 | derived data |
| `resultados/svge_regional_tabla.csv` | Network-season analysis table (Spain) | Derived from SVGE reports and ERA5 | derived data |

Raw WHO FluNet exports are not redistributed; the derived country-season table is included.
`datos/contactos_esp_3edades.json` and `datos/poblacion_edad_ccaa.csv` are rebuilt by `datos_v4.py` only if
deleted; that requires the original matrices and denominators (set `PREM_MATRICES_DIR` and
`DATADISTA_COVID_DIR`, and install the `rdata` package).

## Licence

Code: MIT licence (see `LICENSE`). Data files keep the terms of their original sources listed above; please cite
them when reusing the data.

## Citation

See `CITATION.cff`. Please cite the article and the archived version of this repository (Zenodo DOI).
