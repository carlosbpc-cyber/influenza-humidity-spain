"""
Asimilación semanal de datos (filtro de Kalman por conjuntos, EAKF; Shaman y Karspeck 2012) sobre el modelo v4.

Para cada temporada 2022/23-2025/26:
  - Conjunto de 200 simulaciones con parámetros inciertos (susceptibilidad inicial, R0max, d0, notificación,
    factor social de Navidad). La susceptibilidad a priori NO usa la estimada para esa temporada.
  - Cada semana se asimila el dato nacional de SiVIRA (proxy de gripe) y se actualizan estados y parámetros.
  - Desde cada semana se predice el resto de la temporada; se mide el error en la semana y la altura del pico
    según cuántas semanas faltaban para el pico observado.
  Comparación: el mismo conjunto sin asimilar datos ("sin actualizar").
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import modelo_v4 as m
import modelo_gripe_ccaa_v2 as v2
import calibrar as cal
import calibrar_v4 as c4

rng = np.random.default_rng(2026)
J = json.load(open("resultados/parametros_v4.json")); P0 = {**m.P, **J["comunes"]}
E = 200
NA = m.NA; n = m.n
off = v2.M_MEZCLA - np.diag(np.diag(v2.M_MEZCLA)); MX = off * P0["mov"] + np.diag(1 - off.sum(1) * P0["mov"])
NOMBRES_PAR = ["S0", "R0max", "d0", "rep", "social"]


def preparar(ctx):
    f = ctx["fechas"]; cerr, nav = m.dv.vacaciones(f)
    B = m.C_AMB["home"] + m.C_AMB["others"]
    R = (~cerr)[:, :, None, None] * m.C_AMB["school"][None] + np.where(nav, 0.6, 1.0)[:, None, None, None] * m.C_AMB["work"][None]
    doy = np.array([x.dayofyear for x in f], float)
    return dict(B=B, R=R, nav=nav, doy=doy)


def inicial(ctx, i_temp, S0_prior):
    par = np.column_stack([
        rng.uniform(*S0_prior, E),
        rng.normal(P0["R0max"], 0.08, E),
        rng.normal(P0["d0"], 12, E),
        P0["rep"] * np.exp(rng.normal(0, 0.25, E)),
        rng.normal(P0["social_navidad"], 0.15, E)])
    anio = int(ctx["t"][:4]); vac = m.dv.inmunes_vacuna(anio)
    I0f = np.nan_to_num(np.nanmean(ctx["obs"][:2], axis=0), nan=1.0) / 7 * 2.5 / 1e5
    I = NA[None] * (I0f[None, :, None] / par[:, 3, None, None]) * np.exp(rng.normal(0, 0.3, (E, 1, 1)))
    S = NA[None] * np.clip(par[:, 0, None, None] - vac[None], 0.05, 1) - I * (1 + P0["gamma"] / P0["sigma"])
    Ee = I * P0["gamma"] / P0["sigma"]
    return dict(S=S, E=Ee, I=I), par


def avanzar_semana(st, par, pre, ctx, k, dq, fB):
    """Integra 7 días desde el día 7k; devuelve incidencia semanal [E, n, 3]."""
    S, Ee, I = st["S"].copy(), st["E"].copy(), st["I"].copy(); inc = np.zeros_like(S)
    g, s = P0["gamma"], P0["sigma"]
    for d in range(7 * k, min(7 * k + 7, len(ctx["fechas"]))):
        c = (1 + np.cos(2 * np.pi * (pre["doy"][d] - par[:, 2]) / 365)) / 2
        R0 = (P0["R0min"] + (par[:, 1, None] - P0["R0min"]) * c[:, None]) * np.exp(P0["kappa"] * dq[d])[None] * (1 - P0["theta"] * fB)
        beta = R0 * g / m.RHO                                                     # [E, n]
        fs = np.where(pre["nav"][d], par[:, 4], 1.0)
        for _ in range(2):
            prev = np.einsum("ij,ejb->eib", MX, I / NA[None])
            lam = beta[:, :, None] * (fs[:, None, None] * np.einsum("iab,eib->eia", pre["B"], prev) + np.einsum("iab,eib->eia", pre["R"][d], prev))
            nE = lam * S * 0.5; nI = s * Ee * 0.5; nR = g * I * 0.5
            S -= nE; Ee += nE - nI; I += nI - nR
            I[:, :, 1] += v2.LLEGADAS[None] * ctx["eu"][d] * P0["eps"] * 0.5
            inc += nI
    return dict(S=np.maximum(S, 0), E=np.maximum(Ee, 0), I=np.maximum(I, 0)), inc


def obs_modelo(inc, par):
    return inc.sum((1, 2)) / NA.sum() * 1e5 * par[:, 3]                         # proxy nacional por 100.000


def eaKF(st, par, h, y, oev):
    """Actualización EAKF de estados y parámetros con la observación y."""
    hm, hv = h.mean(), h.var()
    if hv <= 0: return st, par
    pv = 1 / (1 / hv + 1 / oev); pm = pv * (hm / hv + y / oev)
    alpha = np.sqrt(oev / (oev + hv)); dh = pm + alpha * (h - hm) - h
    def upd(X):
        Xf = X.reshape(E, -1); cov = ((Xf - Xf.mean(0)) * (h - hm)[:, None]).mean(0)
        return (Xf + np.outer(dh, cov / hv)).reshape(X.shape)
    st = {k: np.maximum(upd(v), 0) for k, v in st.items()}
    par = upd(par)
    par[:, 0] = np.clip(par[:, 0], 0.4, 0.99); par[:, 1] = np.clip(par[:, 1], 1.25, 3.0)
    par[:, 3] = np.clip(par[:, 3], 0.01, 0.5); par[:, 4] = np.clip(par[:, 4], 0.5, 2.5)
    return st, par


def inflar(st, par, lam=1.03):
    out = {}
    for k, v in st.items():
        mu = v.mean(0, keepdims=True); out[k] = np.maximum(mu + lam * (v - mu), 0)
    mu = par.mean(0, keepdims=True); return out, mu + lam * (par - mu)


def pronosticar(st, par, pre, ctx, k0, dq, fB, nsem):
    tray = []
    s = {kk: v.copy() for kk, v in st.items()}
    for k in range(k0, nsem):
        s, inc = avanzar_semana(s, par, pre, ctx, k, dq, fB); tray.append(obs_modelo(inc, par))
    return np.array(tray)                                                       # [semanas, E]


def temporada(i):
    ctx = cal.CTX[i]; pre = preparar(ctx); dq = c4.DQ[i]; fB = c4.FB[i]
    otros = [v for j, v in enumerate(J["S0_temporadas"].values()) if j != i]
    S0p = (min(otros) - 0.05, min(max(otros) + 0.03, 0.97))
    y = cal.nat(ctx["obs"]); nsem = len(y); pico = int(np.nanargmax(y)); alt = np.nanmax(y)
    res = []
    for modo in ["asimilando", "sin actualizar"]:
        st, par = inicial(ctx, i, S0p)
        for k in range(nsem):
            if k >= 3 and k <= pico + 2:
                tr = pronosticar(st, par, pre, ctx, k, dq, fB, nsem).mean(1)
                ppk = k + int(np.argmax(tr)) if tr.max() > np.nanmax(y[:k]) else int(np.nanargmax(y[:k]))
                palt = max(tr.max(), np.nanmax(y[:k]))
                res.append(dict(temporada=ctx["t"], modo=modo, semana=k, antelacion=pico - k,
                                err_sem=ppk - pico, err_alt=(palt - alt) / alt))
            st, inc = avanzar_semana(st, par, pre, ctx, k, dq, fB)
            if modo == "asimilando" and not np.isnan(y[k]):
                h = obs_modelo(inc, par); oev = max(25.0, (0.15 * y[k]) ** 2)
                # se actualiza a la vez la incidencia de la semana (para coherencia) y los estados
                st, par = eaKF(st, par, h, y[k], oev); st, par = inflar(st, par)
    return res, dict(temporada=ctx["t"], S0_post=float(par[:, 0].mean()))


if __name__ == "__main__":
    filas = []
    for i in range(4):
        r, info = temporada(i); filas += r; print(info, flush=True)
    R = pd.DataFrame(filas); R.to_csv("resultados/enkf_v4_pronosticos.csv", index=False)
    R["abs_sem"] = R.err_sem.abs(); R["abs_alt"] = R.err_alt.abs()
    R["grupo"] = pd.cut(R.antelacion, [-3, 0, 2, 4, 30], labels=["pico o después", "1-2 sem antes", "3-4 sem antes", "≥5 sem antes"])
    T = R.groupby(["grupo", "modo"]).agg(err_sem_medio=("abs_sem", "mean"), pico_1sem=("abs_sem", lambda x: (x <= 1).mean()),
                                        err_alt_mediano=("abs_alt", "median"), n=("abs_sem", "size")).round(2)
    print(T.to_string()); T.to_csv("resultados/enkf_v4_resumen.csv")
