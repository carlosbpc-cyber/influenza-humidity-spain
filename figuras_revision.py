"""Figuras 1-4 revisadas (octubre 2026): ventana de humedad oct-dic, validación fuera de muestra y escenarios ±0,6 g/kg."""
import json, os, numpy as np, pandas as pd, matplotlib, warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
AZ, NA_, GR, RO, VE = "#1f5f8b", "#d9822b", "#7a7a7a", "#b33a3a", "#2a9d8f"
os.makedirs("figures", exist_ok=True)
EN = {"AND": "Andalusia", "ARA": "Aragon", "AST": "Asturias", "BAL": "Balearic Isl.", "CAN": "Canary Isl.", "CNT": "Cantabria",
      "CYL": "Castile-León", "CLM": "Castile-La Mancha", "CAT": "Catalonia", "VAL": "Valencia", "EXT": "Extremadura", "GAL": "Galicia",
      "MAD": "Madrid", "MUR": "Murcia", "NAV": "Navarre", "PVA": "Basque C.", "RIO": "La Rioja"}


def guardar(fig, nombre):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"figures/{nombre}.{ext}", dpi=300, bbox_inches="tight")


def fe(d, y, x, lugar, extra=""):
    dd = d.dropna(subset=[y, x])
    m = smf.ols(f"{y} ~ {x}{extra} + C({lugar}) + C(temp)", dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.temp)[0]})
    return [m.params[x], m.conf_int().loc[x, 0], m.conf_int().loc[x, 1], m.pvalues[x]], m


def bosque(a, filas, xlim, xlabel, texto_x):
    for i, (lab, (est, lo, hi, p)) in enumerate(filas[::-1]):
        c = RO if p < 0.05 else GR
        a.plot([lo, hi], [i, i], color=c, lw=2); a.plot(est, i, "o", color=c, ms=6)
        a.text(texto_x, i, f"{est:+.2f} ({lo:+.2f} to {hi:+.2f})", va="center", fontsize=7)
    a.axvline(0, color="k", lw=.8); a.set_yticks(range(len(filas))); a.set_yticklabels([f[0] for f in filas[::-1]], fontsize=7.5)
    a.set_xlim(*xlim); a.set_xlabel(xlabel)


def parcial(a, d, y, x, lugar, extra, b, ci, n, xlabel, ylabel, titulo, etiqueta):
    dd = d.dropna(subset=[y, x])
    ry = smf.ols(f"{y} ~ C({lugar}) + C(temp){extra}", dd).fit().resid
    rx = smf.ols(f"{x} ~ C({lugar}) + C(temp){extra}", dd).fit().resid
    a.scatter(rx, ry, s=10, color=AZ, alpha=.55, lw=0)
    xs = np.linspace(rx.min(), rx.max(), 10); a.plot(xs, b * xs, color=RO, lw=2)
    a.set_xlabel(xlabel); a.set_ylabel(ylabel); a.set_title(titulo, loc="left", fontweight="bold")
    a.text(.97, .03, f"{b:+.2f} weeks per g/kg\n(95% CI {ci[0]:.2f} to {ci[1]:.2f})\nn = {n} {etiqueta}", transform=a.transAxes,
           va="bottom", ha="right", fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8))


# ---------------- Figura 1: Europa
d = pd.read_csv("resultados/europa_ventanas_tabla.csv").dropna(subset=["mediana", "b"])
E = json.load(open("resultados/europa_largo_efectos.json"))
v = {k: fe(d, "mediana", k, "c", " + b")[0] for k in ["a_nov_ene", "a_oct_dic", "a_pre_inicio", "a_pre_pico"]}
bB = fe(d, "mediana", "b", "c", " + a_oct_dic")[0]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.05, 1.45, 0.8]))
parcial(ax[0], d, "mediana", "a_oct_dic", "c", " + b", v["a_oct_dic"][0], v["a_oct_dic"][1:3], int(d.a_oct_dic.notna().sum()),
        "Specific humidity anomaly, Oct–Dec (g/kg)\n(residual after country, season and type B)", "Median epidemic week (residual, weeks)",
        "A  Pre-epidemic humidity anomaly", "country-seasons")
filas = [("Humidity anomaly, Oct–Dec (per g/kg)", v["a_oct_dic"]), ("Humidity anomaly, 4 wk before onset", v["a_pre_inicio"]),
         ("Humidity anomaly, 4 wk before median wk", v["a_pre_pico"]), ("Humidity anomaly, Nov–Jan", v["a_nov_ene"]),
         ("Temperature anomaly, Nov–Jan (per °C)*", E["mediana:anomalia_t"]["t_nj_anom"]),
         ("Mean humidity between countries (per g/kg)", E["mediana:entre_paises"]["q_nj_media"]), ("Type B share (0→100%)", bB),
         ("A(H3) share (0→100%)*", E["mediana:h3"]["h3"])]
bosque(ax[1], filas, (-5.5, 8.2), "Shift in median epidemic week (weeks; + = later)", 5.2)
ax[1].set_title("B  Covariate effects (95% CI)", loc="left", fontweight="bold")
r2s, r2sc = E["R2_temporada_mediana"], E["R2_temporada_pais_mediana"]
ax[2].bar([0, 1], [r2s * 100, (r2sc - r2s) * 100], color=[AZ, NA_], width=.6)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["Season\n(common to\nall countries)", "Country\n(added)"], fontsize=8)
for i, val in enumerate([r2s * 100, (r2sc - r2s) * 100]): ax[2].text(i, val + 1, f"{val:.0f}%", ha="center")
ax[2].set_ylim(0, 75); ax[2].set_ylabel("Variance of median week explained (%)"); ax[2].set_title("C  What drives timing", loc="left", fontweight="bold")
fig.text(.01, -.04, "Season-clustered 95% CIs; models with country and season fixed effects except * (country fixed effects only). Humidity anomalies of windows anchored to onset or median\n"
         "week are daily anomalies against the same calendar week. Data: WHO FluNet 2010/11–2019/20 and 2022/23–2024/25, 22 EU/EEA countries; ERA5 via Open-Meteo at capitals.", fontsize=7, color="#444")
fig.tight_layout(); guardar(fig, "figure1_europe")

# ---------------- Figura 2: España
d = pd.read_csv("resultados/svge_ventanas_tabla.csv"); S = json.load(open("resultados/svge_regional_efectos.json"))
w = {k: fe(d, "sem_pico", k, "cod")[0] for k in ["a_nov_ene", "a_oct_dic", "a_pre_pico"]}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), gridspec_kw=dict(width_ratios=[1.05, 1.0, 1.45]))
parcial(ax[0], d, "sem_pico", "a_oct_dic", "cod", "", w["a_oct_dic"][0], w["a_oct_dic"][1:3], len(d),
        "Specific humidity anomaly, Oct–Dec (g/kg)\n(residual after network and season effects)", "Peak week (residual, weeks)",
        "A  Pre-epidemic humidity anomaly", "network-seasons")
fe_ = smf.ols("sem_pico ~ C(temp) + C(cod) - 1", d).fit().params.filter(like="C(cod)"); fe_.index = [i.split("[")[1].rstrip("]").replace("T.", "") for i in fe_.index]
fe_ = fe_ - fe_.mean(); g = d.groupby("cod").agg(q=("q_media", "first")).loc[fe_.index]
ax[1].scatter(g.q, fe_, color=AZ, s=25)
for c in fe_.index: ax[1].annotate(c, (g.q[c], fe_[c]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax[1].axhline(0, color=GR, lw=.6); ax[1].set_xlabel("Mean winter specific humidity (g/kg)"); ax[1].set_ylabel("Mean peak delay vs other regions (weeks)")
ax[1].set_title("B  No humidity gradient between regions", loc="left", fontweight="bold")
filas = [("Humidity anomaly, Oct–Dec (per g/kg)", w["a_oct_dic"]), ("Humidity anomaly, 4 wk before peak", w["a_pre_pico"]),
         ("Humidity anomaly, Nov–Jan", w["a_nov_ene"]), ("Temperature anomaly, Nov–Jan (per °C)", S["dentro: anomalía temperatura"]["T_anom"]),
         ("Mean humidity (per g/kg)", S["entre comunidades: humedad media"]["q_media"]),
         ("Population ≥65 y (per % point)", S["entre comunidades: edad"]["p65"]), ("Population 0–14 y (per % point)", S["entre comunidades: niños"]["p0_14"])]
bosque(ax[2], filas, (-4.5, 6.6), "Shift in peak week (weeks; + = later)", 3.4)
ax[2].set_title("C  Effects (95% CI)", loc="left", fontweight="bold")
fig.text(.01, -.04, "Data: SVGE sentinel networks, 2010/11–2016/17 and 2018/19 (ISCIII annual reports); ERA5 via Open-Meteo at regional capitals. Anomaly models with season and network\n"
         "fixed effects; between-region models with season fixed effects. Season-clustered 95% CIs.", fontsize=7, color="#444")
fig.tight_layout(); guardar(fig, "figure2_spain_regions")

# ---------------- Figura 3: ajuste y validación fuera de muestra
import calibrar as cal, calibrar_v4 as c4, modelo_v4 as m
J = json.load(open("resultados/parametros_v4.json")); L = json.load(open("resultados/parametros_loso.json"))
V = pd.read_csv("resultados/validacion_loso.csv"); PI = pd.read_csv("resultados/prediccion_inicio.csv"); CI = json.load(open("resultados/prediccion_inicio_curvas.json")); R = pd.read_csv("resultados/validacion_loso_regional.csv")
fig = plt.figure(figsize=(12, 6)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
for k, t in enumerate(cal.TEMPS):
    a = fig.add_subplot(gs[0, k]); f = cal.CTX[k]["fechas"][:cal.SEM * 7:7]
    o = cal.nat(cal.CTX[k]["obs"]); s_in = J["validacion"][k]["mod"]
    s_out, _ = c4.correr(k, L[t]["S0_prediccion"], {**m.P, **L[t]["comunes"]})
    ref = np.mean([cal.nat(cal.CTX[j]["obs"]) for j in range(4) if j != k], 0)
    a.plot(f, o, "o-", color="k", ms=3, lw=1, label="Observed")
    a.plot(f, s_in, color=GR, lw=1.5, label="Fitted to all seasons")
    a.plot(f, ref, color=NA_, lw=1.1, ls="--", label="Mean of other seasons")
    a.plot(f, s_out, color=AZ, lw=1.1, ls=":", label="Out of sample, no data from the season")
    ci = CI[t]["inicio (umbral)"]; a.plot(f, ci["mod"], color=AZ, lw=2.2, label="Out of sample, updated at onset")
    a.axvline(f[ci["k"]], color=AZ, lw=.7, alpha=.5); a.text(f[ci["k"]], a.get_ylim()[1] * .97, " onset", fontsize=6.5, color=AZ, va="top")
    r_in = V[(V.temporada == t) & (V.modelo.str.startswith("dentro"))].r.iloc[0]
    r_ci = PI[(PI.temporada == t) & (PI.corte == "inicio (umbral)")].r.iloc[0]
    a.set_title(f"{t.replace('-', '/')}", loc="left", fontsize=9, fontweight="bold")
    a.text(.98, .97, f"fit r = {r_in:.2f}\nforecast r = {r_ci:.2f}", transform=a.transAxes, ha="right", va="top", fontsize=7.5)
    a.tick_params(axis="x", labelrotation=45, labelsize=7); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    if k == 0: a.set_ylabel("Influenza proxy per 100,000/week")
    if k == 0: fig.legend(*a.get_legend_handles_labels(), fontsize=7, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(.5, 1.03))
for k, (t, g) in enumerate(R.groupby("temporada")):
    a = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
    jx = np.random.default_rng(1).uniform(-.18, .18, len(g))
    a.scatter(g.med_obs + jx, g.med_mod, color=AZ, s=30, label=f"Out of sample, no data from the season: mean error {(g.med_mod - g.med_obs).abs().mean():.1f} wk")
    a.scatter(g.med_obs + jx, g.med_ref, color=NA_, marker="s", s=22, alpha=.8, label=f"Mean of other seasons: {(g.med_ref - g.med_obs).abs().mean():.1f} wk")
    lo, hi = g[["med_obs", "med_mod", "med_ref"]].min().min() - 1, g[["med_obs", "med_mod", "med_ref"]].max().max() + 1
    a.plot([lo, hi], [lo, hi], color="k", lw=.6); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("Observed median week (weeks from week 40)"); a.set_ylabel("Predicted median week")
    a.set_title(f"Regional timing (regional positivity), {t.replace('-', '/')}", loc="left", fontsize=9, fontweight="bold")
    a.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(rect=(0, 0, 1, .96)); guardar(fig, "figure3_validation")

# ---------------- Figura 4: escenarios
R4 = json.load(open("resultados/resultados_v4.json")); T = pd.read_csv("resultados/resultados_ccaa_v4.csv")
NOM = dict(base="Normal winter", templado="Humid Oct–Dec (+0.6 g/kg)", frio="Dry Oct–Dec (−0.6 g/kg)", medit="Warm Mediterranean", nino="Strong El Niño",
           tipoB="Type B season", deriva="Antigenic drift", peor="Worst case", vac_ninos="Child vaccination ↑", vac_mayores="Older-adult vaccination ↑")
COL = dict(base="k", templado="#4a90c2", frio="#1f3f6b", medit=VE, nino=NA_, tipoB="#8e6bb3", deriva=RO, peor="#6b0f0f", vac_ninos="#6aa84f", vac_mayores="#b5a642")
cods = list(T.cod.unique()); W = T[T.escenario == "base"].set_index("cod").pob_M; W = W / W.sum()
fig, ax = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1.35, 1, 0.9]))
for k, e in R4["escenarios"].items():
    f = pd.to_datetime(e["semanas"]); nat = (pd.DataFrame(e["ili"])[cods] * W.values).sum(axis=1)
    ax[0].plot(f, nat, color=COL[k], lw=2.2 if k in ("base", "peor") else 1.3, ls="--" if k in ("tipoB", "nino", "vac_ninos", "vac_mayores") else "-", label=NOM[k])
ax[0].axhline(60, color=GR, ls=":", lw=1); ax[0].set_ylabel("Influenza proxy per 100,000/week (Spain)")
ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); ax[0].set_xlim(pd.Timestamp("2026-10-01"), pd.Timestamp("2027-04-15"))
ax[0].legend(fontsize=6.3, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12)); ax[0].set_title("A  National curves, 2026/27", loc="left", fontweight="bold")
b = T[T.escenario == "base"].sort_values("indice_riesgo"); p_ = T[T.escenario == "peor"].set_index("cod").loc[b.cod]
y = np.arange(len(b)); ax[1].barh(y, b.indice_riesgo, color=AZ, height=.6, label="Normal winter")
ax[1].plot(p_.indice_riesgo, y, "|", color=RO, ms=9, mew=2, label="Worst case")
ax[1].set_yticks(y); ax[1].set_yticklabels([EN[c] for c in b.cod], fontsize=7); ax[1].set_xlim(0, 100)
ax[1].set_xlabel("Regional risk index (0–100)"); ax[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
ax[1].set_title("B  Risk index by region", loc="left", fontweight="bold")
orden = ["base", "frio", "tipoB", "deriva", "peor", "vac_ninos", "vac_mayores"]
H = T.groupby("escenario")[["hosp_0_14", "hosp_15_64", "hosp_65"]].sum().loc[orden] / 1000
left = np.zeros(len(orden))
for col, c, lab in [("hosp_65", RO, "≥65 y"), ("hosp_15_64", NA_, "15–64 y"), ("hosp_0_14", AZ, "0–14 y")]:
    ax[2].barh(range(len(orden)), H[col], left=left, color=c, label=lab, height=.6); left += H[col].values
for i, val in enumerate(left): ax[2].text(val + .5, i, f"{val:.1f}", va="center", fontsize=7)
ax[2].set_yticks(range(len(orden))); ax[2].set_yticklabels([NOM[k] for k in orden], fontsize=7); ax[2].invert_yaxis()
ax[2].set_xlabel("Hospitalisations (thousands)"); ax[2].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3); ax[2].set_xlim(0, 58)
ax[2].set_title("C  Hospitalisations by age", loc="left", fontweight="bold")
fig.tight_layout(); guardar(fig, "figure4_scenarios")
print({k: [round(x, 2) for x in val] for k, val in v.items()}, {k: [round(x, 2) for x in val] for k, val in w.items()}, [round(x, 2) for x in bB])
