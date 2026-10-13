"""
Reconstrucción (octubre 2026) del módulo base del modelo por CCAA: geografía, población, movilidad entre
comunidades, llegadas internacionales, positividad europea y utilidades.

El módulo original no se conservó. Se ha reconstruido a partir de lo que usan los scripts v4 y de los datos
abiertos descritos en el manuscrito (Tabla 1):
  - Población por comunidad: la misma que figura en resultados/resultados_ccaa_v4.csv (pob_M).
  - Movilidad: viajes diarios entre comunidades (MITMA, teléfonos móviles, 4 días de 2024) -> M_MEZCLA.
  - Llegadas: turistas internacionales por comunidad de destino (INE FRONTUR, media diaria de los últimos
    24 meses publicados) -> LLEGADAS.
  - Positividad europea: centinela de atención primaria, UE/EEE (ECDC ERVISS) -> prev_europa().
Si los ficheros de datos_nuevos/ aún no existen, se usan valores provisionales y se avisa.
"""
import json, os, warnings
import numpy as np, pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
NUEVOS = os.environ.get("DATOS_NUEVOS", os.path.join(AQUI, "datos_nuevos"))

COD = ["AND", "ARA", "AST", "BAL", "CAN", "CNT", "CYL", "CLM", "CAT", "VAL", "EXT", "GAL", "MAD", "MUR", "NAV", "PVA", "RIO"]
NOMBRE = ["Andalucía", "Aragón", "Asturias", "Illes Balears", "Canarias", "Cantabria", "Castilla y León",
          "Castilla-La Mancha", "Cataluña", "C. Valenciana", "Extremadura", "Galicia", "Madrid", "Murcia", "Navarra",
          "País Vasco", "La Rioja"]
n = len(COD)
POB = np.array([8.63, 1.35, 1.01, 1.23, 2.24, 0.59, 2.39, 2.10, 8.12, 5.32, 1.05, 2.70, 7.00, 1.57, 0.68, 2.23, 0.32])
N = POB * 1e6
PROVISIONAL = []

# ---------------------------------------------------------------- movilidad entre comunidades
_F_MOV = os.path.join(NUEVOS, "movilidad_ccaa_mitma.csv")


def _mezcla():
    if os.path.exists(_F_MOV):
        d = pd.read_csv(_F_MOV)
        V = d.groupby(["origen", "destino"]).viajes.sum().unstack().reindex(index=COD, columns=COD).fillna(0)
        V = V / d.dia.nunique()                                   # viajes medios por día
        off = V.values.copy(); np.fill_diagonal(off, 0)
        # fracción del tiempo que los residentes de i pasan en j: cada visita genera dos viajes (ida y vuelta)
        # y ocupa ~8 de 24 horas -> viajes / (2 x 3 x población). El parámetro mov (=4) escala la mezcla.
        off = off / (6 * N[:, None])
    else:
        PROVISIONAL.append("movilidad (modelo de gravedad)")
        # provisional: modelo de gravedad con distancia entre capitales (solo para probar el código)
        xy = np.array([[37.39, -5.98], [41.65, -0.89], [43.36, -5.85], [39.57, 2.65], [28.12, -15.43], [43.46, -3.81],
                       [41.65, -4.72], [39.86, -4.02], [41.39, 2.16], [39.47, -0.38], [38.92, -6.34], [42.88, -8.55],
                       [40.42, -3.70], [37.99, -1.13], [42.81, -1.65], [42.85, -2.67], [42.47, -2.45]])
        dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) * 100 + 50
        off = (N[None] / dist ** 2); np.fill_diagonal(off, 0)
        off = off / off.sum(1, keepdims=True) * 0.004
    return off + np.diag(1 - off.sum(1))


M_MEZCLA = _mezcla()

# ---------------------------------------------------------------- llegadas internacionales
_F_FR = os.path.join(NUEVOS, "frontur_10823.json")
_INE = {"Andalucía": "AND", "Balears, Illes": "BAL", "Canarias": "CAN", "Cataluña": "CAT",
        "Comunitat Valenciana": "VAL", "Madrid, Comunidad de": "MAD"}


def _llegadas():
    """Turistas internacionales por día (media de los últimos 24 meses de FRONTUR, tabla INE 10823).
    FRONTUR solo desglosa seis comunidades; el resto ("Otras Comunidades Autónomas") se reparte por población."""
    if os.path.exists(_F_FR):
        J = json.load(open(_F_FR, encoding="utf-8"))
        out, otras = {}, None
        for s in J:
            nom = s["Nombre"]
            if not nom.endswith("Turista. Dato base. "):
                continue
            region = nom.split(". Turista")[0]
            v = [x["Valor"] for x in s["Data"] if x.get("Valor") is not None][:24]
            if not v:
                continue
            diario = float(np.mean(v)) * 12 / 365
            if region in _INE:
                out[_INE[region]] = diario
            elif region.startswith("Otras"):
                otras = diario
        if len(out) == 6 and otras is not None:
            resto = [c for c in COD if c not in out]
            w = np.array([POB[COD.index(c)] for c in resto]); w = w / w.sum()
            out.update({c: otras * wi for c, wi in zip(resto, w)})
            return np.array([out[c] for c in COD])
        warnings.warn("FRONTUR: formato no reconocido; se usan valores provisionales")
    PROVISIONAL.append("llegadas (FRONTUR aproximado)")
    anual = dict(CAT=19.0, BAL=15.1, CAN=17.7, AND=13.2, VAL=10.9, MAD=8.4, PVA=1.8, GAL=1.6, MUR=1.3, CYL=1.0,
                 ARA=0.6, CNT=0.5, AST=0.5, NAV=0.4, CLM=0.3, EXT=0.3, RIO=0.1)   # millones/año, orden de magnitud
    return np.array([anual[c] * 1e6 / 365 for c in COD])


LLEGADAS = _llegadas()
P65 = None   # se rellena en datos_v4 (fracción >= 65 por comunidad)

# ---------------------------------------------------------------- positividad europea (ERVISS)
_F_ERV = os.path.join(AQUI, "datos", "erviss", "sentinelTestsDetectionsPositivity.csv")


def _serie_europa():
    d = pd.read_csv(_F_ERV)
    d = d[(d.countryname == "EU/EEA") & (d.pathogen == "Influenza") & (d.indicator == "positivity") & (d.age == "total")]
    d = d[d.pathogensubtype.eq("total") | d.pathogensubtype.isna()]
    f = [pd.Timestamp.fromisocalendar(int(s[:4]), int(s[6:]), 4) for s in d.yearweek]   # jueves de cada semana ISO
    return pd.Series(d.value.values / 100, index=f).sort_index().groupby(level=0).mean()


EU = _serie_europa()


def prev_europa(fechas, temporada=None, desplaz=0, mult=1.0):
    """Positividad de gripe en la UE/EEE (fracción) para cada día de `fechas`.
    temporada=None -> curva media por semana de temporada (2022/23-2025/26), para proyecciones.
    desplaz: días de adelanto (<0) o retraso (>0) de la curva; mult: factor multiplicativo."""
    f = pd.DatetimeIndex(fechas) - pd.Timedelta(days=desplaz)
    if temporada is None:
        ini = {y: pd.Timestamp.fromisocalendar(y, 40, 1) for y in (2022, 2023, 2024, 2025)}
        curvas = []
        for y, t0 in ini.items():
            s = EU[(EU.index >= t0) & (EU.index < t0 + pd.Timedelta(days=364))]
            curvas.append(pd.Series(s.values, index=(s.index - t0).days))
        media = pd.concat(curvas, axis=1).mean(1).sort_index()
        t0 = pd.Timestamp.fromisocalendar(pd.DatetimeIndex(fechas)[0].isocalendar()[0], 40, 1)
        x = (f - t0).days.values
        v = np.interp(x, media.index.values, media.values)
    else:
        x = (f - pd.Timestamp("2000-01-01")).days.values
        xp = (EU.index - pd.Timestamp("2000-01-01")).days.values
        v = np.interp(x, xp, EU.values)
    return np.clip(v * mult, 0, 1)


def a_semanal(x, fechas):
    """Suma diaria -> semanal (semanas de 7 días desde fechas[0]). x: [días, ...]."""
    x = np.asarray(x); k = len(x) // 7
    sem = x[:k * 7].reshape(k, 7, *x.shape[1:]).sum(1)
    idx = pd.DatetimeIndex(fechas)[:k * 7:7]
    return pd.DataFrame(sem.reshape(k, -1), index=idx)


if PROVISIONAL:
    warnings.warn("Datos provisionales en uso: " + ", ".join(PROVISIONAL) + ". Ejecutar descargar_datos.py.")
