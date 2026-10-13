"""Semana pico por red centinela (SVGE 2010/11-2018/19, 8 temporadas, 14-15 comunidades): ¿qué explica las diferencias regionales?"""
import re, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
import datos_v4 as dv, modelo_gripe_ccaa_v2 as v2
MAP = {"Andalucía": "AND", "Aragón": "ARA", "Asturias": "AST", "Baleares": "BAL", "Canarias": "CAN", "Cantabria": "CNT",
       "Castilla La Mancha": "CLM", "Castilla y León": "CYL", "Cataluña": "CAT", "Comunitat Valenciana": "VAL", "Extremadura": "EXT",
       "Madrid": "MAD", "Navarra": "NAV", "País Vasco": "PVA", "La Rioja": "RIO"}
fil, temp = [], None
for l in open("datos/svge_redes_2010_2019.txt", encoding="utf-8"):
    l = l.strip()
    if l.startswith("## "): temp = int(l[3:]); continue
    if not l or l.startswith("#") or l.startswith("Global"): continue
    nom = next((k for k in MAP if l.startswith(k)), None)
    if not nom: continue
    m = re.match(re.escape(nom) + r"\s+(\d{1,2})/(\d{4})\s+(\S+)", l)
    w, y, tasa = int(m.group(1)), int(m.group(2)), m.group(3)
    sem = (pd.Timestamp.fromisocalendar(y, w, 1) - pd.Timestamp.fromisocalendar(temp, 40, 1)).days // 7   # semanas desde la 40
    fil.append(dict(cod=MAP[nom], temp=temp, sem_pico=sem, tasa=float(tasa.replace(",", ".")) if tasa != "NA" else np.nan,
                    B=int(bool(re.search(r"\bB\b", l.split()[-1]) and l.split()[-1].startswith("B")))))
d = pd.DataFrame(fil)
CLI = json.load(open("datos/clima_ccaa_2010_2019.json"))["datos"]
cl = pd.DataFrame([dict(cod=k, temp=int(s.split(":")[0]), q=float(s.split(":")[1].split(",")[0]), T=float(s.split(",")[1]))
                   for k, v in CLI.items() for s in v.split()])
d = d.merge(cl, on=["cod", "temp"])
d["q_anom"] = d.q - d.groupby("cod").q.transform("mean"); d["T_anom"] = d["T"] - d.groupby("cod")["T"].transform("mean")
d["q_media"] = d.groupby("cod").q.transform("mean")
E = dv.EDAD; d["p65"] = d.cod.map(E.p65) * 100; d["p0_14"] = d.cod.map(E.p0_14) * 100
d["llegadas"] = d.cod.map(dict(zip(v2.COD, v2.LLEGADAS / v2.N * 1e3)))   # turistas diarios por 1.000 hab
d["isla"] = d.cod.isin(["CAN", "BAL"]).astype(int)
d.to_csv("resultados/svge_regional_tabla.csv", index=False)
print(len(d), "red-temporadas;", d.cod.nunique(), "redes;", d.temp.nunique(), "temporadas")
cl_ = lambda dd: dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.temp)[0]})
res = {}
def corre(nom, f, dd=d):
    m = smf.ols(f, dd).fit(**cl_(dd)); ks = [k for k in m.params.index if not k.startswith("C(") and k != "Intercept"]
    res[nom] = {k: [round(m.params[k], 2), round(m.conf_int().loc[k, 0], 2), round(m.conf_int().loc[k, 1], 2), round(m.pvalues[k], 3)] for k in ks}
    res[nom]["n"] = int(m.nobs); res[nom]["R2"] = round(m.rsquared, 2); print(nom, res[nom])
corre("solo temporada", "sem_pico ~ C(temp)")
corre("entre comunidades: edad", "sem_pico ~ p65 + C(temp)")
corre("entre comunidades: niños", "sem_pico ~ p0_14 + C(temp)")
corre("entre comunidades: humedad media", "sem_pico ~ q_media + C(temp)")
corre("entre comunidades: llegadas e islas", "sem_pico ~ np.log(llegadas) + isla + C(temp)")
corre("entre comunidades: todo", "sem_pico ~ p65 + q_media + np.log(llegadas) + C(temp)")
corre("dentro: anomalía humedad", "sem_pico ~ q_anom + C(temp) + C(cod)")
corre("dentro: anomalía temperatura", "sem_pico ~ T_anom + C(temp) + C(cod)")
r2c = smf.ols("sem_pico ~ C(temp) + C(cod)", d).fit().rsquared; res["R2 temporada+comunidad"] = round(r2c, 2)
print("R2 temporada + comunidad", round(r2c, 2), " DE semana pico dentro de temporada:", round(d.groupby("temp").sem_pico.std().mean(), 2))
res["DE_intra_temporada"] = round(d.groupby("temp").sem_pico.std().mean(), 2)
json.dump(res, open("resultados/svge_regional_efectos.json", "w"), indent=1, ensure_ascii=False)
