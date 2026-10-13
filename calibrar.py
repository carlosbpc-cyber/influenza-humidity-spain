"""
Reconstrucción (octubre 2026) de los contextos de calibración: una entrada por temporada 2022/23-2025/26 con
fechas diarias (desde la semana 40), humedad, positividad europea y proxy observado de SiVIRA.

Observado: proxy nacional de gripe en atención primaria (tasa IRA x positividad, /100.000/semana), tal y como se
usó en el manuscrito (datos/sivira_nacional.json). Las series regionales de 2022/23 y 2023/24 no se conservaron:
`obs` repite la serie nacional en todas las comunidades (solo afecta al arranque de la epidemia en cada región).
"""
import json, os
import numpy as np, pandas as pd
import modelo_v4 as m
import modelo_v3 as m3
import modelo_gripe_ccaa_v2 as v2

SEM = 33
_S = json.load(open(os.path.join(v2.AQUI, "datos", "sivira_nacional.json"), encoding="utf-8"))
TEMPS = [t for t in _S if t[:2] == "20"]
_PR = json.load(open(os.path.join(v2.AQUI, "datos", "sivira_proxy_regional.json")))
W = v2.POB / v2.POB.sum()


def nat(x):
    """Agregado nacional (media ponderada por población) de una serie semanal regional [semanas, n]."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x
    w = np.where(np.isnan(x), 0, W[None, :])
    return np.nansum(x * W[None, :], 1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)


CTX = []
for t in TEMPS:
    ini = pd.Timestamp(_S[t]["inicio"])
    f = pd.date_range(ini, periods=SEM * 7 + 14, freq="D")
    o = np.array(_S[t]["obs"], float)[:SEM]
    obs = np.repeat(o[:, None], v2.n, 1)
    CTX.append(dict(t=t, fechas=f, q=m3.q_temporada(f), eu=v2.prev_europa(f, t), obs=obs))
