"""
Revisión, puntos 2 y 4: ventana temporal de la humedad y utilidad predictiva.

Exposiciones (anomalía de humedad específica, g/kg):
  nov-ene     media nov-ene menos la media de ese lugar en todas las temporadas (definición del manuscrito)
  oct-dic     igual, octubre-diciembre (anterior al pico en casi todas las temporadas)
  pre-inicio  4 semanas antes del inicio de la epidemia (solo Europa: el SVGE regional no publica inicio)
  pre-pico    4 semanas antes de la semana pico (Europa: semana mediana; España: semana pico)
Las ventanas ancladas a la epidemia usan la anomalía de cada día respecto a la media 2010-2025 de la misma semana
del año, para que una ventana más tardía (más fría y seca) no se confunda con un efecto.

Modelos: semana ~ anomalía (+ tipo B en Europa) + efectos fijos de lugar y temporada; IC por conglomerados de
temporada. Predicción: validación dejando fuera cada temporada (media del lugar como referencia) con IC por
remuestreo de temporadas y un control placebo (anomalías de otra temporada del mismo lugar).
"""
import json, os, sys
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import modelo_v3 as m3

rng = np.random.default_rng(7)
SALIDA = {}


def anomalias_diarias(Q):
    """Q: humedad diaria (kg/kg) por lugar -> anomalía diaria (g/kg) respecto a la media de la misma semana ISO."""
    sem = Q.index.isocalendar().week.values
    clim = Q.groupby(sem).mean()
    return (Q - clim.reindex(sem).values) * 1000


def media_ventana(S, lugar, ini, fin):
    if lugar not in S.columns:
        return np.nan
    x = S.loc[ini:fin, lugar]
    return float(x.mean()) if x.notna().sum() >= 0.8 * ((pd.Timestamp(fin) - pd.Timestamp(ini)).days + 1) else np.nan


def semana_a_fecha(temp, idx):
    return pd.Timestamp.fromisocalendar(int(temp), 40, 1) + pd.Timedelta(weeks=int(idx))


def exposiciones(d, lugar_col, temp_col, Q, ancla_inicio=None, ancla_pico=None):
    A = anomalias_diarias(Q)
    Qg = Q * 1000
    filas = []
    for _, r in d.iterrows():
        l, t = r[lugar_col], int(r[temp_col])
        e = {}
        e["q_nov_ene"] = media_ventana(Qg, l, f"{t}-11-01", f"{t + 1}-01-31")
        e["q_oct_dic"] = media_ventana(Qg, l, f"{t}-10-01", f"{t}-12-31")
        if ancla_inicio is not None and pd.notna(r[ancla_inicio]):
            f = semana_a_fecha(t, r[ancla_inicio]); e["a_pre_inicio"] = media_ventana(A, l, f - pd.Timedelta(days=28), f - pd.Timedelta(days=1))
        if ancla_pico is not None and pd.notna(r[ancla_pico]):
            f = semana_a_fecha(t, r[ancla_pico]); e["a_pre_pico"] = media_ventana(A, l, f - pd.Timedelta(days=28), f - pd.Timedelta(days=1))
        filas.append(e)
    E = pd.DataFrame(filas, index=d.index)
    for v in ["q_nov_ene", "q_oct_dic"]:
        E[v.replace("q_", "a_")] = E[v] - E.groupby(d[lugar_col])[v].transform("mean")
    return E


def efecto(d, y, x, lugar, extra=""):
    dd = d.dropna(subset=[y, x])
    f = f"{y} ~ {x}{extra} + C({lugar}) + C(temp)"
    m = smf.ols(f, dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.temp)[0]})
    return dict(beta=round(m.params[x], 2), lo=round(m.conf_int().loc[x, 0], 2), hi=round(m.conf_int().loc[x, 1], 2),
                p=round(m.pvalues[x], 4), n=int(m.nobs))


def loso(d, y, lugar, x=None, extra=""):
    err = np.full(len(d), np.nan)
    for t in d.temp.unique():
        tr, te = d[d.temp != t], (d.temp == t).values
        f = f"{y} ~ C({lugar})" + (f" + {x}" if x else "") + extra
        m = smf.ols(f, tr).fit()
        ok = te & d[lugar].isin(tr[lugar].unique()).values
        err[ok] = np.abs(d.loc[ok, y].values - m.predict(d.loc[ok]).values)
    return err


def utilidad(d, y, lugar, x, extra="", nplacebo=200):
    d = d.dropna(subset=[y, x]).reset_index(drop=True)
    e0, e1 = loso(d, y, lugar, None, extra), loso(d, y, lugar, x, extra)
    ok = ~np.isnan(e0) & ~np.isnan(e1); diff = (e1 - e0)[ok]; seas = d.temp.values[ok]; us = np.unique(seas)
    bs = [np.mean(np.concatenate([diff[seas == s] for s in rng.choice(us, len(us))])) for _ in range(3000)]
    pl = []
    for _ in range(nplacebo):
        dp = d.copy(); dp["xp"] = dp.groupby(lugar)[x].transform(lambda s: rng.permutation(s.values))
        pl.append(np.nanmean(loso(dp, y, lugar, "xp", extra)[ok]) - np.nanmean(e0[ok]))
    pl = np.array(pl)
    return dict(mae_base=round(np.mean(e0[ok]), 2), mae_humedad=round(np.mean(e1[ok]), 2),
                diferencia=round(np.mean(diff), 2), lo=round(np.percentile(bs, 2.5), 2), hi=round(np.percentile(bs, 97.5), 2),
                placebo_media=round(pl.mean(), 2), p_placebo=round((1 + np.sum(pl <= np.mean(diff))) / (len(pl) + 1), 3))


def europa():
    d = pd.read_csv("resultados/europa_largo_tabla.csv").dropna(subset=["mediana", "b"]).reset_index(drop=True)
    Q, _ = m3.cargar_clima("EU_")
    Q.columns = [c.replace("EU_", "") for c in Q.columns]
    E = exposiciones(d, "c", "temp", Q, ancla_inicio="inicio15", ancla_pico="mediana")
    d = pd.concat([d, E], axis=1)
    chk = d[["q_nj", "q_nov_ene", "q_od", "q_oct_dic"]].corr().round(3)
    out = dict(comprobacion_nov_ene=float(chk.loc["q_nj", "q_nov_ene"]), comprobacion_oct_dic=float(chk.loc["q_od", "q_oct_dic"]),
               de_anomalia=dict(nov_ene=round(d.a_nov_ene.std(), 2), oct_dic=round(d.a_oct_dic.std(), 2)),
               p5_p95_nov_ene=d.a_nov_ene.quantile([.05, .95]).round(2).tolist())
    for x in ["a_nov_ene", "a_oct_dic", "a_pre_inicio", "a_pre_pico"]:
        out[x] = dict(efecto=efecto(d, "mediana", x, "c", " + b"), utilidad=utilidad(d, "mediana", "c", x))
    d.to_csv("resultados/europa_ventanas_tabla.csv", index=False)
    return out


def espana():
    d = pd.read_csv("resultados/svge_regional_tabla.csv").reset_index(drop=True)
    Q = m3.QDIA
    E = exposiciones(d, "cod", "temp", Q, ancla_pico="sem_pico")
    d = pd.concat([d, E], axis=1)
    out = dict(comprobacion_nov_ene=float(d[["q", "q_nov_ene"]].corr().iloc[0, 1].round(3)),
               de_anomalia=dict(nov_ene=round(d.a_nov_ene.std(), 2), oct_dic=round(d.a_oct_dic.std(), 2)),
               p5_p95_nov_ene=d.a_nov_ene.quantile([.05, .95]).round(2).tolist())
    for x in ["a_nov_ene", "a_oct_dic", "a_pre_pico"]:
        out[x] = dict(efecto=efecto(d, "sem_pico", x, "cod"), utilidad=utilidad(d, "sem_pico", "cod", x))
    d.to_csv("resultados/svge_ventanas_tabla.csv", index=False)
    return out


if __name__ == "__main__":
    res = dict(europa=europa(), espana=espana())
    json.dump(res, open("resultados/humedad_ventanas.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
