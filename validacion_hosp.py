"""
Revisión, punto 3: contraste de las hospitalizaciones del modelo con datos observados.

  a) Fuera de muestra con SiVIRA: para cada temporada 2023/24-2025/26, la hospitalización por infección (IHR) por
     edad se estima con las OTRAS dos temporadas y se aplica a las infecciones que el modelo predice para la
     temporada excluida (calibrado sin ella, resultados/parametros_loso.json). Se compara con el proxy de
     hospitalización de SiVIRA (IRAG x positividad, /100.000) acumulado por edad y con la semana de su pico.
  b) Contraste externo: hospitalizaciones totales por gripe en 2024/25 frente a la estimación del ISCIII
     (Aguilar Figueroa et al. 2025, ref. 1: más de 33.000).
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import calibrar_v4 as c4, calibrar as cal, modelo_v4 as m

E = json.load(open("datos/sivira_edad/proxy_edad_raw.json"))
LOSO = json.load(open("resultados/parametros_loso.json"))
J = json.load(open("resultados/parametros_v4.json"))
REP = {"<1": [0.33, 0, 0], "1-4": [1.45, 0, 0], "5-19": [4.9, 2.5, 0], "20-59": [0, 24.5, 0], "60-69": [0, 2.95, 2.55],
       "70-79": [0, 0, 4.0], "≥80": [0, 0, 2.95]}
INF = {"2023-2024": "202440", "2024-2025": "202539", "2025-2026": "202620"}


def obs_semanal(rep, k):
    """Proxy semanal por 100.000 en 0-14, 15-64 y 65+ [33, 3]."""
    num = np.zeros((33, 3)); den = np.zeros(3)
    for s in E[rep][k]:
        g, v = s.split(":"); x = np.array([float(z) for z in v.split()])[:33]
        num += x[:, None] * np.array(REP[g])[None]; den += np.array(REP[g])
    return num / den


def infecciones(t, fuera):
    i = cal.TEMPS.index(t)
    if fuera:
        p = LOSO[t]["comunes"]; S0 = LOSO[t]["S0_prediccion"]
    else:
        p = J["comunes"]; S0 = J["S0_temporadas"][t]
    _, inc = c4.correr(i, S0, {**m.P, **p})
    sem = m.v2.a_semanal(inc.reshape(len(inc), -1), cal.CTX[i]["fechas"]).values.reshape(-1, m.n, 3)[:33]
    return sem.sum(1) / m.NA.sum(0) * 1e5            # infecciones semanales por 100.000 por edad [33, 3]


filas = []
inf_in = {t: infecciones(t, False) for t in INF}
ihr = {t: obs_semanal(r, "hosp").sum(0) / inf_in[t].sum(0) for t, r in INF.items()}
for t, r in INF.items():
    otras = [k for k in INF if k != t]
    ihr_cv = np.mean([ihr[k] for k in otras], 0)
    inf_out = infecciones(t, True)
    pred = inf_out * ihr_cv[None]                     # hospitalizaciones semanales por 100.000 [33, 3]
    obs = obs_semanal(r, "hosp")
    tot_o, tot_p = obs.sum(1), pred.sum(1)
    pk_o, pk_p = int(np.argmax(tot_o)), int(np.argmax(tot_p)) + 1     # +1 semana de demora infección-ingreso
    for a, g in enumerate(["0-14", "15-64", "65+"]):
        filas.append(dict(temporada=t, edad=g, obs_100k=round(obs[:, a].sum(), 1), pred_100k=round(pred[:, a].sum(), 1),
                          razon=round(pred[:, a].sum() / obs[:, a].sum(), 2)))
    filas.append(dict(temporada=t, edad="total (semana pico)", obs_100k=pk_o, pred_100k=pk_p, razon=pk_p - pk_o))
    # hospitalizaciones absolutas predichas fuera de muestra
    n_hosp = (pred.sum(0) / 1e5 * m.NA.sum(0)).sum()
    filas.append(dict(temporada=t, edad="hospitalizaciones totales (n)", obs_100k=np.nan, pred_100k=round(n_hosp), razon=np.nan))
R = pd.DataFrame(filas); R.to_csv("resultados/validacion_hospitalizaciones.csv", index=False)
print(R.to_string())
print("Referencia externa 2024/25 (ISCIII, ref. 1): > 33.000 hospitalizaciones por gripe")
