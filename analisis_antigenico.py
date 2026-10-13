"""Índice antigénico A(H3N2) (Perofsky et al. eLife 2024; distancias HA estación a estación) y momento/intensidad en Europa 2010-2019."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, json
D = pd.read_csv("datos/antigenico/north-america_summarized_mean_seasonal_distances_h3n2_ha_21y.tsv", sep="\t")
D["y"] = D.current_season_start.str[:4].astype(int); D["S"] = D.other_season_start.str[:4].astype(int)
A = D[D.S - D.y == 1].pivot_table(index="S", columns="distance_map", values="mean")
A = (A - A.loc[2010:2019].mean()) / A.loc[2010:2019].std()     # z dentro del periodo de análisis
d = pd.read_csv("resultados/europa_largo_tabla.csv").dropna(subset=["mediana", "q_nj_anom", "b", "h3"])
d = d[d.temp <= 2019].copy()
for k in ["wolf", "koel", "titer_tree_model"]: d["ag_" + k] = d.temp.map(A[k]); d["agx_" + k] = d["ag_" + k] * d.h3
cl = lambda dd: dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.temp)[0]})
out = {}
for y in ["mediana", "pico", "lpos"]:
    for k in ["wolf", "koel", "titer_tree_model"]:
        # (a) interacción con la proporción H3, con efectos fijos de país y temporada
        m = smf.ols(f"{y} ~ h3 + agx_{k} + q_nj_anom + b + C(c) + C(temp)", d).fit(**cl(d))
        # (b) efecto principal ponderado por H3, solo efectos fijos de país (10 temporadas)
        m2 = smf.ols(f"{y} ~ agx_{k} + h3 + q_nj_anom + b + C(c)", d).fit(**cl(d))
        out[f"{y}:{k}"] = dict(FE_ambos=[round(m.params[f"agx_{k}"], 2), round(m.pvalues[f"agx_{k}"], 3)],
                               FE_pais=[round(m2.params[f"agx_{k}"], 2), round(m2.pvalues[f"agx_{k}"], 3)])
        print(y, k, out[f"{y}:{k}"])
# nivel temporada: ¿explica el índice el efecto común de temporada?
fe = smf.ols("mediana ~ q_nj_anom + b + C(c) + C(temp)", d).fit().params.filter(like="C(temp)")
fe.index = [int(i.split("T.")[1].rstrip("]")) for i in fe.index]
s = pd.DataFrame(dict(fe=fe)); s["h3"] = d.groupby("temp").h3.mean(); s["wolf"] = A.wolf; s["wolfx"] = s.wolf * s.h3
print(s.round(2)); print("corr efecto temporada vs wolf*H3:", round(s[["fe", "wolfx"]].corr().iloc[0, 1], 2), " vs wolf:", round(s[["fe", "wolf"]].corr().iloc[0, 1], 2))
json.dump(out, open("resultados/antigenico_europa.json", "w"), indent=1)
