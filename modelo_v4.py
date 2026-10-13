"""
Modelo de gripe por CCAA, versión 4: v3 + estructura por edades.

Novedades respecto a v3
  - Tres grupos de edad (0-14, 15-64, 65+) en cada comunidad, con matrices de contacto sintéticas de España
    (Prem et al. 2021) separadas por ámbito: hogar, escuela, trabajo y otros.
  - Calendario escolar por comunidad: en vacaciones desaparecen los contactos escolares; en Navidad
    el trabajo baja al 60 % y los contactos de hogar y ocio se multiplican por un factor estimado (reuniones).
  - Vacunación real por temporada, comunidad y edad (mayores, 6-59 meses, 15-64) y efectividad por temporada
    según los subtipos circulantes.
  - Hospitalizaciones por edad.
Se conserva: ciclo estacional común, anomalía climática (kappa), efecto del tipo B (theta), movilidad e importación.

    lambda_{i,a}(t) = beta(t) * sum_j M_ij * sum_b C_{i,ab}(t) * I_{j,b} / N_{j,b}
    beta(t) = gamma * R0_i(t) / rho(C_lectivo)       (R0 conserva su significado de v3)
"""
import json, os
import numpy as np, pandas as pd
import modelo_gripe_ccaa_v2 as v2
import modelo_v3 as v3
import datos_v4 as dv

AQ = v2.AQUI
COD, n, POB = v2.COD, v2.n, v2.POB
NA = dv.NA                                    # [n, 3]
N = NA.sum(1)
FRAC = dv.FRAC
P = dict(R0min=1.2, R0max=1.78, d0=42.5, kappa=-18.0, theta=0.035, sigma=1.0, gamma=1 / 2.5, rep=0.077, eps=0.03,
         mov=4.0, social_navidad=1.0, IHR=[0.00092, 0.00085, 0.0159])   # IHR calibrada con el proxy de hospitalización de SiVIRA por edad (calibrar_ihr_v4.py)
CAL = os.path.join(AQ, "resultados", "parametros_v4.json")
if os.path.exists(CAL):
    P.update(json.load(open(CAL))["comunes"])
_IHR = os.path.join(AQ, "resultados", "ihr_v4.json")
if os.path.exists(_IHR) and os.environ.get("IHR_FIJO") != "1":
    P["IHR"] = json.load(open(_IHR))["IHR"]           # revisión: IHR de la última calibración (calibrar_ihr_v4.py)

# matrices de contacto por comunidad: contactos con el grupo b escalados por la estructura de edad local
_NAC = (NA.sum(0) / NA.sum())
# v4.0 escalaba los contactos por la estructura de edad local; los datos regionales (SVGE 2010-2019 y SiVIRA con
# positividad regional) no apoyan el retraso de las comunidades envejecidas que eso produce, así que en v4.1 se usa la
# matriz nacional en todas las comunidades (la edad local solo afecta a vacunación y hospitalizaciones).
CONTACTOS_LOCALES = os.environ.get("CONTACTOS_LOCALES", "0") == "1"
_ESC = FRAC / _NAC if CONTACTOS_LOCALES else np.ones_like(FRAC)   # [n, 3]
C_AMB = {k: dv.CONT[k][None, :, :] * _ESC[:, None, :] for k in dv.CONT}   # [n, 3, 3]
C_LECT_NAC = sum(dv.CONT.values())
RHO = float(np.max(np.abs(np.linalg.eigvals(C_LECT_NAC))))


def matrices_dia(fechas, p):
    """C[d, i, a, b] para cada día según vacaciones escolares y Navidad."""
    cerr, nav = dv.vacaciones(fechas)
    base = C_AMB["home"] + C_AMB["others"]
    esc, trab = C_AMB["school"], C_AMB["work"]
    ft = np.where(nav, 0.6, 1.0)                       # menos trabajo en Navidad (fijo)
    fs = np.where(nav, p["social_navidad"], 1.0)       # reuniones familiares y sociales en Navidad (estimado)
    return fs[:, None, None, None] * base[None] + (~cerr)[:, :, None, None] * esc[None] + ft[:, None, None, None] * trab[None]


def simular(fechas, dq, eu, S0, I0_frac, fB=0.0, p=None, anio=None, mult_R=1.0, dt=0.5, inmunes=None):
    """Devuelve incidencia diaria [dias, n, 3] y Rt [dias, n]."""
    p = {**P, **(p or {})}
    anio = anio or fechas[0].year
    vac = dv.inmunes_vacuna(anio, fB) if inmunes is None else inmunes
    S = NA * np.clip(S0 - vac, 0.05, 1)
    I = NA * np.asarray(I0_frac).reshape(-1, 1) * np.ones((1, 3)); E = I * p["gamma"] / p["sigma"]; S = S - I - E
    off = v2.M_MEZCLA - np.diag(np.diag(v2.M_MEZCLA)); Mx = off * p["mov"] + np.diag(1 - off.sum(1) * p["mov"])
    Cd = matrices_dia(fechas, p); c = v3.ciclo(fechas, p["d0"])
    dias = len(fechas); inc = np.zeros((dias, n, 3)); Rt = np.zeros((dias, n))
    ext = v2.LLEGADAS * p["eps"]
    for d in range(dias):
        R0 = (p["R0min"] + (p["R0max"] - p["R0min"]) * c[d]) * np.exp(p["kappa"] * dq[d]) * (1 - p["theta"] * fB) * mult_R
        beta = R0 * p["gamma"] / RHO                                         # [n]
        # Rt aproximado: radio espectral de la NGM local con susceptibles actuales
        if d % 7 == 0:
            for i in range(n):
                K = (S[i] / NA[i])[:, None] * Cd[d, i] * beta[i] / p["gamma"]
                Rt[d:d + 7, i] = np.max(np.abs(np.linalg.eigvals(K)))
        for _ in range(int(round(1 / dt))):
            prev = Mx @ (I / NA)                                             # [n, 3] prevalencia efectiva
            lam = beta[:, None] * np.einsum("iab,ib->ia", Cd[d], prev)
            nE = lam * S * dt; nI = p["sigma"] * E * dt; nR = p["gamma"] * I * dt
            S -= nE; E += nE - nI; I += nI - nR
            I[:, 1] += ext * eu[d] * dt
            inc[d] += nI
    return inc, Rt


def proxy_semanal(inc, fechas, rep):
    """Proxy SiVIRA por 100.000/semana y comunidad (todas las edades)."""
    return v2.a_semanal(inc.sum(2), fechas) / (N / 1e5) * rep


def hospitalizaciones(inc, p=None):
    p = {**P, **(p or {})}
    return (inc.sum(0) * np.array(p["IHR"])[None, :])                      # [n, 3]


if __name__ == "__main__":
    import time
    f = pd.date_range("2025-09-29", periods=35 * 7, freq="D")
    eu = v2.prev_europa(f, None); dq = np.zeros((len(f), n))
    t = time.time(); inc, Rt = simular(f, dq, eu, 0.83, np.full(n, 2e-5), 0.1, anio=2025)
    print("seg", round(time.time() - t, 2), "ataque por edad %", (inc.sum((0, 1)) / NA.sum(0) * 100).round(1), "total", round(inc.sum() / N.sum() * 100, 1))
    print("rho", round(RHO, 2), "Rt max", Rt.max(0).round(2))
