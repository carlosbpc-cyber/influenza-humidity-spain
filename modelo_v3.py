"""
Reconstrucción (octubre 2026) de las funciones de v3 que usa el modelo v4: ciclo estacional común,
humedad específica diaria por comunidad (ERA5) y su anomalía, y proporción de gripe B por temporada en España.
"""
import glob, json, os, warnings
import numpy as np, pandas as pd
import modelo_gripe_ccaa_v2 as v2

COD = v2.COD
FB_ES = {2022: 0.284, 2023: 0.026, 2024: 0.333, 2025: 0.005, 2026: 0.10}   # FluNet/ERVISS España (manuscrito)
CLIMA_REF = ("2015-07-01", "2025-06-30")                                   # periodo de referencia de la anomalía


def ciclo(fechas, d0):
    """c(t) = ½[1 + cos(2π(t − d0)/365)], t = día del año."""
    doy = pd.DatetimeIndex(fechas).dayofyear.values.astype(float)
    return 0.5 * (1 + np.cos(2 * np.pi * (doy - d0) / 365))


def q_especifica(td_c, p_hpa):
    """Humedad específica (kg/kg) a partir del punto de rocío (°C) y la presión en superficie (hPa)."""
    e = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    return 0.622 * e / (p_hpa - 0.378 * e)


def cargar_clima(prefijo=""):
    """Serie diaria de humedad específica (kg/kg) y temperatura (°C) por lugar, de datos_nuevos/clima."""
    q, T = {}, {}
    for f in sorted(glob.glob(os.path.join(v2.NUEVOS, "clima", f"{prefijo}*.json"))):
        cod = os.path.basename(f).rsplit("_", 2)[0]
        if not prefijo and cod.startswith("EU_"):
            continue
        d = json.load(open(f))
        idx = pd.to_datetime(d["time"])
        td = np.array(d["dew_point_2m_mean"], float); p = np.array(d["surface_pressure_mean"], float)
        q.setdefault(cod, []).append(pd.Series(q_especifica(td, p), index=idx))
        T.setdefault(cod, []).append(pd.Series(np.array(d["temperature_2m_mean"], float), index=idx))
    Q = pd.DataFrame({k: pd.concat(v).sort_index().groupby(level=0).mean() for k, v in q.items()})
    TT = pd.DataFrame({k: pd.concat(v).sort_index().groupby(level=0).mean() for k, v in T.items()})
    return Q, TT


try:
    QDIA, TDIA = cargar_clima()
    QDIA = QDIA.reindex(columns=COD)
    CLIMA_OK = QDIA.notna().any().all()
except Exception:
    QDIA = TDIA = None; CLIMA_OK = False
if not CLIMA_OK:
    warnings.warn("Sin clima ERA5 en datos_nuevos/clima: anomalías de humedad = 0 (provisional).")


def climatologia_semanal(Q, ref=CLIMA_REF):
    """Media por semana ISO del año en el periodo de referencia."""
    r = Q.loc[ref[0]:ref[1]]
    return r.groupby(r.index.isocalendar().week.values).mean()


def anomalia_q(fechas, q=None):
    """Anomalía diaria de humedad específica (kg/kg) [días, n]: valor del día menos la media 2015-2025 de la misma
    semana del año. `q` se acepta por compatibilidad; se usa la serie ERA5 cargada."""
    f = pd.DatetimeIndex(fechas)
    if not CLIMA_OK:
        return np.zeros((len(f), len(COD)))
    clim = climatologia_semanal(QDIA)
    obs = QDIA.reindex(f)
    sem = f.isocalendar().week.values
    an = obs.values - clim.reindex(sem).values
    return np.nan_to_num(an, nan=0.0)


def q_temporada(fechas):
    f = pd.DatetimeIndex(fechas)
    if not CLIMA_OK:
        return np.full((len(f), len(COD)), np.nan)
    return QDIA.reindex(f).values
