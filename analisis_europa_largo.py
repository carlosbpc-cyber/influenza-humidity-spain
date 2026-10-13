"""Análisis multinivel 22 países x 13 temporadas (FluNet): tabla de efectos para el manuscrito."""
import re, pandas as pd, numpy as np, statsmodels.formula.api as smf, json
d = pd.read_csv("resultados/europa_largo_tabla.csv")
d = d.dropna(subset=["mediana", "pico", "q_nj_anom", "b"])
d["b10"] = d.b; d["h3"] = d.h3
out = {}
def fit(f, y="mediana"):
    vs = [v for v in re.findall(r"[a-z_0-9]+", f) if v in d.columns] + [y]
    dd = d.dropna(subset=vs)
    m = smf.ols(f"{y} ~ " + f, dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.temp)[0]})
    return m
for y in ["mediana", "pico"]:
    for nom, f in {"anomalia_q": "q_nj_anom + b + C(c)", "anomalia_t": "t_nj_anom + b + C(c)",
                   "conjunto": "q_nj_anom + t_nj_anom + b + C(c)", "h3": "q_nj_anom + b + h3 + C(c)",
                   "entre_paises": "q_nj_media + q_nj_anom + b + C(temp)",
                   "inmunidad": "q_nj_anom + b + prev_same + C(c)", "ambos_FE": "q_nj_anom + b + C(c) + C(temp)"}.items():
        m = fit(f, y)
        out[f"{y}:{nom}"] = {k: [round(m.params[k], 2), round(m.conf_int().loc[k, 0], 2), round(m.conf_int().loc[k, 1], 2), round(m.pvalues[k], 4)]
                             for k in m.params.index if not k.startswith("C(") and k != "Intercept"}
        out[f"{y}:{nom}"]["n"] = int(m.nobs)
# varianza explicada por temporada
for y in ["mediana", "pico"]:
    out[f"R2_temporada_{y}"] = round(smf.ols(f"{y} ~ C(temp)", d).fit().rsquared, 3)
    out[f"R2_temporada_pais_{y}"] = round(smf.ols(f"{y} ~ C(temp)+C(c)", d).fit().rsquared, 3)
out["n"] = len(d); out["paises"] = d.c.nunique(); out["temporadas"] = d.temp.nunique()
out["r_q_t"] = round(d[["q_nj_anom", "t_nj_anom"]].corr().iloc[0, 1], 2)
L = pd.read_csv("resultados/europa_largo_loso.csv"); out["loso"] = L.groupby("modelo").err.mean().round(2).to_dict()
json.dump(out, open("resultados/europa_largo_efectos.json", "w"), indent=1, ensure_ascii=False)
for k, v in out.items(): print(k, v)
