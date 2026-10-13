"""Proyecciones 2026/27 con el modelo v4 (por edades)."""
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import modelo_v4 as m
import modelo_gripe_ccaa_v2 as v2
import datos_v4 as dv

J = json.load(open(os.path.join(m.AQ, "resultados", "parametros_v4.json")))
p = {**m.P, **J["comunes"]}; S0m = float(np.mean(list(J["S0_temporadas"].values())))
INICIO = pd.Timestamp.fromisocalendar(2026, 40, 1)
F = pd.date_range(INICIO, periods=35 * 7, freq="D")
VENT = np.array([f.month in tuple(int(k) for k in os.environ.get("VENTANA_MESES", "10,11,12").split(",")) for f in F])
DQ_ESC = float(os.environ.get("DQ_ESCENARIO", "0.6"))   # g/kg de los escenarios climáticos (revisión: p5-p95 observado)
MEDIT = np.array([0.5 if c == "AND" else 1.0 if c in ("CAT", "VAL", "MUR", "BAL") else 0.0 for c in m.COD])
I0 = np.full(m.n, 261 * 0.082 / p["rep"] / 7 * 2.5 / 1e5)
UMBRAL = 60.0
VAC_BASE = dv.inmunes_vacuna(2026, 0.10)


def dq_const(g_kg, peso=None):
    dq = np.zeros((len(F), m.n)); w = np.ones(m.n) if peso is None else peso
    dq[VENT] = g_kg / 1000 * w; return dq


def vac_infantil(cob_0_4, cob_5_14):
    """Vacunación infantil reforzada: cobertura 6-59 meses y 5-14 años (efectividad de la temporada)."""
    v = VAC_BASE.copy(); ve = dv.efectividad(2026, 0.10); f04 = dv.EDAD["p0_4_en_0_14"].values
    v[:, 0] = (cob_0_4 * f04 + cob_5_14 * (1 - f04)) * ve
    return v


def dS_deriva(objetivo_sem=2.1):
    """Aumento de susceptibilidad que adelanta 2,1 semanas la semana mediana (efecto europeo de +1 DE de
    distancia antigénica A(H3N2) con dominio de H3; Perofsky et al. 2024, distancias HI)."""
    def med(dS):
        inc, _ = m.simular(F, dq_const(0), v2.prev_europa(F, None), S0m + dS, I0, 0.10, p, anio=2026, inmunes=VAC_BASE)
        cum = np.cumsum(inc.sum((1, 2))); return np.searchsorted(cum, cum[-1] / 2)
    base = med(0.0)
    for dS in np.arange(0.0, 0.051, 0.0025):
        if base - med(dS) >= objetivo_sem * 7: return round(float(dS), 4)
    return 0.05


DS = dS_deriva()
ESC = {
    "base":      dict(nombre="Invierno normal", dq=dq_const(0)),
    "templado":  dict(nombre="Invierno templado y húmedo", dq=dq_const(+DQ_ESC)),
    "frio":      dict(nombre="Invierno frío y seco", dq=dq_const(-DQ_ESC)),
    "medit":     dict(nombre="Mediterráneo cálido (costa)", dq=dq_const(+0.6, MEDIT)),
    "nino":      dict(nombre="El Niño fuerte (importación)", dq=dq_const(0), eu_desplaz=-21, eu_mult=1.5),
    "tipoB":     dict(nombre="Temporada de tipo B", dq=dq_const(0), fB=0.8),
    "deriva":    dict(nombre="Deriva antigénica A(H3N2) (+1 DE)", dq=dq_const(0), dS=DS),
    "peor":      dict(nombre="Peor caso (deriva + El Niño + frío)", dq=dq_const(-DQ_ESC), dS=DS, eu_desplaz=-21, eu_mult=1.5),
    "vac_ninos": dict(nombre="Vacunación infantil reforzada (70 % 6-59 m, 40 % 5-14 a)", dq=dq_const(0), vac=vac_infantil(0.70, 0.40)),
    "vac_mayores": dict(nombre="Mayores al 75 %", dq=dq_const(0), vac65=0.75),
}


def correr(e):
    eu = v2.prev_europa(F, None, e.get("eu_desplaz", 0), e.get("eu_mult", 1.0))
    fB = e.get("fB", 0.10)
    vac = e.get("vac", dv.inmunes_vacuna(2026, fB))
    if "vac65" in e: vac = vac.copy(); vac[:, 2] = e["vac65"] * dv.efectividad(2026, fB) * 0.8
    return m.simular(F, e["dq"], eu, S0m + e.get("dS", 0), I0, fB, p, anio=2026, inmunes=vac)


def metricas(inc, Rt):
    proxy = m.proxy_semanal(inc, F, p["rep"]); proxy.columns = range(m.n)
    ataque_edad = inc.sum(0) / m.NA                                  # [n, 3]
    ataque = inc.sum((0, 2)) / m.N
    hosp_edad = m.hospitalizaciones(inc, p)                          # [n, 3] número
    hosp = hosp_edad.sum(1) / m.N * 1e5
    pico = proxy.max().values; sobre = proxy.gt(UMBRAL)
    ini = [sobre[c].idxmax() if sobre[c].any() else pd.NaT for c in range(m.n)]
    dias = np.array([(d - INICIO).days if pd.notna(d) else 245 for d in ini])
    idx = 100 * (0.35 * np.clip(pico / 700, 0, 1) + 0.40 * np.clip(hosp / 120, 0, 1) + 0.25 * np.clip(1 - (dias - 31) / 151, 0, 1))
    Rtmax = Rt.max(0)
    t = pd.DataFrame(dict(cod=m.COD, nombre=v2.NOMBRE, pob_M=m.POB, Rt_max=Rtmax.round(2),
                          p_arraigo_1intro=np.where(Rtmax > 1, 1 - 1 / Rtmax, 0).round(2),
                          inicio_epidemico=[d.strftime("%Y-%m-%d") if pd.notna(d) else None for d in ini],
                          semana_pico=[proxy.index[int(np.argmax(proxy[c].values))].strftime("%Y-%m-%d") for c in range(m.n)],
                          pico_ILI_100k=pico.round(0), tasa_ataque_pct=(ataque * 100).round(1),
                          ataque_0_14=(ataque_edad[:, 0] * 100).round(1), ataque_15_64=(ataque_edad[:, 1] * 100).round(1),
                          ataque_65=(ataque_edad[:, 2] * 100).round(1),
                          hosp_100k=hosp.round(0), hosp_0_14=hosp_edad[:, 0].round(0), hosp_15_64=hosp_edad[:, 1].round(0),
                          hosp_65=hosp_edad[:, 2].round(0), indice_riesgo=idx.round(0)))
    return t, proxy


if __name__ == "__main__":
    out = {"escenarios": {}, "umbral": UMBRAL, "parametros": J, "dS_deriva": DS}
    tabs = []; W = m.POB / m.POB.sum()
    for k, e in ESC.items():
        inc, Rt = correr(e); t, proxy = metricas(inc, Rt)
        proxy.columns = m.COD
        out["escenarios"][k] = dict(nombre=e["nombre"], semanas=[d.strftime("%Y-%m-%d") for d in proxy.index],
                                    ili={c: proxy[c].round(1).tolist() for c in m.COD}, tabla=t.to_dict("records"))
        t.insert(0, "escenario", k); tabs.append(t)
    res = pd.concat(tabs); res.to_csv(os.path.join(m.AQ, "resultados", "resultados_ccaa_v4.csv"), index=False)
    json.dump(out, open(os.path.join(m.AQ, "resultados", "resultados_v4.json"), "w"), ensure_ascii=False)
    print("dS deriva", DS)
    for k in ESC:
        r = res[res.escenario == k]; e = out["escenarios"][k]
        nat = (pd.DataFrame(e["ili"])[m.COD] * W).sum(axis=1); i = int(nat.idxmax())
        pobA = m.NA.sum(0)
        print(f"{k:11s} pico {e['semanas'][i]} {nat.max():5.0f}  ataque {(r.tasa_ataque_pct.values*W).sum():5.1f}% "
              f"(0-14 {(r.ataque_0_14.values*m.NA[:,0]).sum()/pobA[0]:4.1f} | 15-64 {(r.ataque_15_64.values*m.NA[:,1]).sum()/pobA[1]:4.1f} | 65+ {(r.ataque_65.values*m.NA[:,2]).sum()/pobA[2]:4.1f})  "
              f"hosp {r.hosp_0_14.sum()+r.hosp_15_64.sum()+r.hosp_65.sum():7.0f} (65+ {r.hosp_65.sum():6.0f})  inicio {r.inicio_epidemico.min()}..{r.inicio_epidemico.max()}  idx {r.indice_riesgo.min():.0f}-{r.indice_riesgo.max():.0f}")
    print(res[res.escenario == "base"].sort_values("indice_riesgo", ascending=False)[["cod", "inicio_epidemico", "semana_pico", "tasa_ataque_pct", "hosp_100k", "indice_riesgo"]].to_string(index=False))
