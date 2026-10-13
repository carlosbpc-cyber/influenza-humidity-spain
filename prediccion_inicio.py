"""
Predicción fuera de muestra condicionada al inicio de la temporada (revisión, figura 3).

Para cada temporada t: parámetros calibrados sin t (parametros_loso.json). La única cantidad que se
re-estima es la susceptibilidad inicial S0, ajustada a las semanas observadas hasta el corte; el resto
de la temporada se predice. Así se reproduce lo que sabría un epidemiólogo al emitir el pronóstico.
Corte principal: semana en que el proxy nacional supera 60/100.000 por segunda semana consecutiva
(umbral epidémico), con un mínimo de 4 semanas de datos. Sensibilidad: corte fijo a 1 de diciembre (semana 48).
Salida: resultados/prediccion_inicio.csv, resultados/prediccion_inicio_curvas.json
"""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import calibrar as cal, calibrar_v4 as c4, modelo_v4 as m
from validacion_loso import metricas

L = json.load(open("resultados/parametros_loso.json"))
UMBRAL = 60


def corte_umbral(o, minimo=4):
    for k in range(1, len(o)):
        if o[k] > UMBRAL and o[k - 1] > UMBRAL:
            return max(k, minimo - 1)
    return len(o) - 1


def ajustar_S0(i, p, o, k):
    f = lambda S0: np.mean((np.log1p(c4.correr(i, S0, p)[0][:k + 1]) - np.log1p(o[:k + 1])) ** 2)
    r = minimize_scalar(f, bounds=(0.6, 0.95), method="bounded", options=dict(xatol=1e-4))
    return float(r.x)


fil, curvas = [], {}
for i, t in enumerate(cal.TEMPS):
    p = {**m.P, **L[t]["comunes"]}; o = cal.nat(cal.CTX[i]["obs"])
    s0, _ = c4.correr(i, L[t]["S0_prediccion"], p)
    curvas[t] = dict(obs=o.tolist(), sin_datos=s0.tolist())
    for nom, k in [("inicio (umbral)", corte_umbral(o)), ("1 de diciembre", 8)]:
        S0 = ajustar_S0(i, p, o, k); s, _ = c4.correr(i, S0, p)
        todo = metricas(o, s); post = metricas(o[k + 1:], s[k + 1:])
        fil.append(dict(temporada=t, corte=nom, semana_corte=k, S0=S0, S0_sin_datos=L[t]["S0_prediccion"],
                        r=todo["r"], r_resto=post["r"], err_pico=todo["err_pico"], err_mediana=todo["err_mediana"],
                        err_altura=todo["err_altura"], pico_ya_observado=bool(np.argmax(o) <= k)))
        curvas[t][nom] = dict(k=k, S0=S0, mod=s.tolist())
F = pd.DataFrame(fil); F.to_csv("resultados/prediccion_inicio.csv", index=False)
json.dump(curvas, open("resultados/prediccion_inicio_curvas.json", "w"))
pd.set_option("display.width", 200); print(F.round(3).to_string())
F["abs_pico"] = F.err_pico.abs(); F["abs_med"] = F.err_mediana.abs(); F["abs_alt"] = F.err_altura.abs()
print(F.groupby("corte")[["r", "r_resto", "abs_pico", "abs_med", "abs_alt"]].mean().round(2))

# Regional: semana mediana con la predicción actualizada al inicio
from validacion_loso import semana_mediana
reg = []
for i, t in enumerate(cal.TEMPS):
    if t not in c4.PR: continue
    p = {**m.P, **L[t]["comunes"]}; S0 = curvas[t]["inicio (umbral)"]["S0"]
    _, inc = c4.correr(i, S0, p); ob = np.array(c4.PR[t]["obs"], float)
    regm = c4.v2a(inc.sum(2), cal.CTX[i]["fechas"]) / (m.N / 1e5) * p["rep"]
    for j, c in enumerate(m.COD):
        if c4.PR[t]["fuente"].get(c) != "regional" or np.nansum(ob[:, j]) <= 0: continue
        reg.append(dict(temporada=t, cod=c, med_obs=semana_mediana(ob[:, j]), med_inicio=semana_mediana(regm[:, j])))
R = pd.DataFrame(reg); R.to_csv("resultados/prediccion_inicio_regional.csv", index=False)
R["e"] = (R.med_inicio - R.med_obs).abs()
print("Regional actualizado al inicio:", R.groupby("temporada").e.mean().round(2).to_dict(), "media", round(R.e.mean(), 2), "<=2:", f"{(R.e <= 2).mean():.0%}")
