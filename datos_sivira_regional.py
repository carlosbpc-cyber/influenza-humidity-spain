"""Positividad semanal por CCAA (2024/25 y 2025/26) a partir de los acumulados de los informes semanales de SiVIRA,
y proxy regional = tasa IRA de la comunidad x positividad de la comunidad (si hay muestras suficientes)."""
import json, numpy as np, pandas as pd
import calibrar as cal
ABR = {"ANDALU": "AND", "ARAGON": "ARA", "ASTURI": "AST", "BALEAR": "BAL", "CANARI": "CAN", "CANTAB": "CNT", "CLAMAN": "CLM",
       "CYLEON": "CYL", "CATALU": "CAT", "CVALEN": "VAL", "EXTREM": "EXT", "GALICI": "GAL", "MADRID": "MAD", "MURCIA": "MUR",
       "NAVARR": "NAV", "PVASCO": "PVA", "LARIOJ": "RIO", "NACION": "ESP"}
fil = []
for l in open("datos/sivira_edad/positividad_ccaa_acumulada.txt"):
    if l.startswith("#") or "|" not in l: continue
    rep, resto = l.strip().split("|")
    for it in resto.split(";"):
        k, v = it.split(":"); m, dd = v.split(",")
        if k in ABR and m != "-":
            fil.append(dict(rep=rep, cod=ABR[k], m=float(m), d=float(dd)))
A = pd.DataFrame(fil)
A["anio"] = A.rep.str[:4].astype(int); A["semana"] = A.rep.str[4:].astype(int)
A["temp"] = np.where(A.semana >= 40, A.anio, A.anio - 1)
# el informe de la semana W publica datos hasta la semana W-1
A["fecha"] = [pd.Timestamp.fromisocalendar(a, s, 1) - pd.Timedelta(days=7) for a, s in zip(A.anio, A.semana)]
salida = {}
for t in (2024, 2025):
    S = A[A.temp == t].pivot_table(index="fecha", columns="cod", values=["m", "d"]).sort_index()
    ini = pd.Timestamp.fromisocalendar(t, 40, 1)
    idx = pd.date_range(ini, periods=cal.SEM, freq="7D")
    M = S["m"].reindex(idx).ffill().fillna(0); Dd = S["d"].reindex(idx).ffill().fillna(0)
    M = M.cummax(); Dd = Dd.cummax()                      # acumulados no decrecientes (revisiones)
    dm = M.diff().fillna(M.iloc[0]); dd = Dd.diff().fillna(Dd.iloc[0])
    # suavizado: ventana centrada de 3 semanas
    pos = (dd.rolling(3, center=True, min_periods=1).sum() / dm.rolling(3, center=True, min_periods=1).sum().replace(0, np.nan))
    nmues = dm.rolling(3, center=True, min_periods=1).sum()
    salida[t] = dict(pos=pos, n=nmues)
TEMP = {2024: 2, 2025: 3}                                    # índice en cal.CTX
prox = {}
for t, i in TEMP.items():
    ctx = cal.CTX[i]; sv = cal.SV[ctx["t"]]
    pos_nac = np.array([float(x) for x in sv["pos"].split()])[:cal.SEM] / 100
    P = salida[t]["pos"]; Nn = salida[t]["n"]
    obs = np.full((cal.SEM, len(cal.m.COD)), np.nan); fuente = {}
    for j, c in enumerate(cal.m.COD):
        s = sv[c].split()
        if not s: continue
        ira = np.array([float(x) for x in s][:cal.SEM]); k = len(ira)
        if c in P and (Nn[c].iloc[:20] >= 15).mean() > 0.6:
            pr = P[c].values[:k]; pr = np.where(np.isnan(pr) | (Nn[c].values[:k] < 15), pos_nac[:k], pr)
            fuente[c] = "regional"
        else:
            pr = pos_nac[:k]; fuente[c] = "nacional"
        obs[:k, j] = ira * pr
    prox[ctx["t"]] = dict(obs=np.round(obs, 1).tolist(), fuente=fuente)
    print(ctx["t"], "positividad regional en:", [c for c, v in fuente.items() if v == "regional"])
json.dump(prox, open("datos/sivira_proxy_regional.json", "w"))
