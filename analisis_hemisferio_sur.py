"""¿Anticipa la temporada del hemisferio sur (abr-sep del año Y) la temporada europea Y/Y+1? FluNet, 13 temporadas."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, json
from scipy import stats
F = pd.read_csv("datos/flunet/VIW_FNT.csv", low_memory=False,
                usecols=["COUNTRY_CODE", "ISO_YEAR", "ISO_WEEK", "SPEC_PROCESSED_NB", "INF_ALL", "AH3", "AH1N12009", "INF_A", "INF_B", "ORIGIN_SOURCE"])
SUR = ["AUS", "CHL", "ZAF", "ARG", "NZL"]
s = F[F.COUNTRY_CODE.isin(SUR) & F.ISO_WEEK.between(14, 39)].groupby(["COUNTRY_CODE", "ISO_YEAR", "ISO_WEEK"]).sum(numeric_only=True).reset_index()
fil = []
for (c, y), g in s.groupby(["COUNTRY_CODE", "ISO_YEAR"]):
    g = g.sort_values("ISO_WEEK"); pos = g.INF_ALL.fillna(0)
    if pos.sum() < 100: continue
    cum = pos.cumsum(); med = g.ISO_WEEK.values[np.searchsorted(cum, cum.iloc[-1] / 2)]
    tests = g.SPEC_PROCESSED_NB.sum()
    sub = g[["AH3", "AH1N12009", "INF_B"]].sum(); tot = sub.sum()
    fil.append(dict(c=c, temp=y, pos=pos.sum(), positividad=pos.sum() / tests if tests > 0 else np.nan, med_sur=med,
                    h3_sur=sub.AH3 / tot if tot else np.nan, b_sur=sub.INF_B / tot if tot else np.nan))
S = pd.DataFrame(fil)
S["lpos_rel"] = np.log(S.pos) - S.groupby("c").pos.transform(lambda x: np.log(x).mean())
S["med_rel"] = S.med_sur - S.groupby("c").med_sur.transform("mean")
sur = S.groupby("temp")[["lpos_rel", "med_rel", "h3_sur", "b_sur", "positividad"]].mean()
d = pd.read_csv("resultados/europa_largo_tabla.csv").dropna(subset=["mediana", "q_nj_anom", "b"])
m = smf.ols("mediana ~ q_nj_anom + b + C(c) + C(temp)", d).fit()
fe = m.params.filter(like="C(temp)"); fe.index = [int(i.split("T.")[1].rstrip("]")) for i in fe.index]
fe = fe - fe.mean()
d["lpos_rel"] = d.lpos - d.groupby("c").lpos.transform("mean")
eu = pd.DataFrame(dict(fe_mediana=fe, tam_rel=d.groupby("temp").lpos_rel.mean(), h3_eu=d.groupby("temp").h3.mean(), b_eu=d.groupby("temp").b.mean()))
X = eu.join(sur, how="inner"); print(X.round(2).to_string())
res = {}
for a, b_ in [("fe_mediana", "med_rel"), ("fe_mediana", "lpos_rel"), ("tam_rel", "lpos_rel"), ("h3_eu", "h3_sur"), ("b_eu", "b_sur"), ("fe_mediana", "h3_sur")]:
    r, p = stats.spearmanr(X[a], X[b_], nan_policy="omit"); res[f"{a}~{b_}"] = [round(r, 2), round(p, 3)]; print(a, b_, round(r, 2), round(p, 3))
json.dump(dict(n_temporadas=len(X), correlaciones_spearman=res), open("resultados/hemisferio_sur.json", "w"), indent=1)
# controlar tendencia secular (más tests con los años) y periodo pos-COVID
X = X.dropna(subset=["fe_mediana"]).copy(); X["anio"] = X.index; X["post"] = (X.index >= 2022).astype(int)
for a, b_ in [("tam_rel", "lpos_rel"), ("fe_mediana", "lpos_rel"), ("fe_mediana", "med_rel")]:
    ra = smf.ols(f"{a} ~ anio + post", X).fit().resid; rb = smf.ols(f"{b_} ~ anio + post", X).fit().resid
    r, p = stats.spearmanr(ra, rb); res[f"parcial {a}~{b_}"] = [round(r, 2), round(p, 3)]; print("parcial", a, b_, round(r, 2), round(p, 3))
    r, p = stats.spearmanr(X.loc[X.post == 0, a], X.loc[X.post == 0, b_]); res[f"pre2020 {a}~{b_}"] = [round(r, 2), round(p, 3)]; print("solo 2010-2019", a, b_, round(r, 2), round(p, 3))
json.dump(dict(n_temporadas=len(X), correlaciones_spearman=res), open("resultados/hemisferio_sur.json", "w"), indent=1)
