"""
Validación fuera de muestra del modelo v4 (punto 1 de la revisión).

Para cada temporada t de 2022/23-2025/26:
  1. Se calibra el modelo con las otras tres (misma función de pérdida y optimizador que calibrar_v4.py;
     kappa y theta se mantienen fijos porque proceden de datos independientes, 2010-2019).
  2. Se predice t sin usar ningún dato de t: susceptibilidad inicial = media de las tres temporadas de
     calibración (igual que en la proyección 2026/27), clima y tipo B observados de t.
  3. Se compara con el observado y con una referencia sencilla: la curva media de las otras tres temporadas.
Métricas: correlación, error en la semana del pico, en la semana mediana y en la altura del pico; y para
2024/25 y 2025/26, error regional en la semana mediana con positividad regional.

Salida: resultados/validacion_loso.csv, resultados/validacion_loso_regional.csv, resultados/parametros_loso.json
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
import calibrar as cal
import calibrar_v4 as c4
import modelo_v4 as m

PAR = os.environ.get("PARAMETROS", "resultados/parametros_v4.json")
J = json.load(open(PAR)); FIJOS = {k: J["comunes"][k] for k in ("R0min", "mov", "kappa", "theta")}
TEMPS = cal.TEMPS


def perdida_sub(x, idx):
    R0max, d0, rep, nav = x[0], x[1], np.exp(x[2]), x[3]; S0s = x[4:]
    if not (1.5 < R0max < 6 and -60 < d0 < 90 and 0.5 < nav < 2.5 and all(0.3 < s < 0.95 for s in S0s)):
        return 1e6
    p = dict(FIJOS, R0max=R0max, d0=d0, rep=rep, eps=c4.EPS_FIJO, social_navidad=nav)
    L = 0.0
    for i, S0 in zip(idx, S0s):
        s, inc = c4.correr(i, S0, p); o = cal.nat(cal.CTX[i]["obs"])
        L += np.mean((np.log1p(s) - np.log1p(o)) ** 2)
        t = cal.CTX[i]["t"]
        if c4.PESO_REG > 0 and t in c4.PR:
            reg = c4.v2a(inc.sum(2), cal.CTX[i]["fechas"]) / (m.N / 1e5) * p["rep"]
            ob = np.array(c4.PR[t]["obs"], float); js = [j for j, c in enumerate(m.COD) if c4.PR[t]["fuente"].get(c) == "regional"]
            on = ob[:, js] / np.nanmean(ob[:, js], 0); mn = reg[:, js] / reg[:, js].mean(0)
            L += c4.PESO_REG * np.nanmean((np.log1p(mn) - np.log1p(on)) ** 2)
    return L


def calibrar_sub(idx, x0):
    best = None
    for k in range(2):                               # dos arranques para evitar mínimos locales
        r = minimize(perdida_sub, x0, args=(idx,), method="Nelder-Mead",
                     options=dict(maxiter=int(os.environ.get("MAXITER", 2500)), xatol=1e-3, fatol=1e-5, adaptive=True))
        if best is None or r.fun < best.fun:
            best = r
        x0 = r.x
    return best


def semana_mediana(y):
    c = np.cumsum(np.nan_to_num(y)); return int(np.searchsorted(c, c[-1] / 2))


def metricas(o, s):
    return dict(r=float(np.corrcoef(o, s)[0, 1]), err_pico=int(np.argmax(s) - np.argmax(o)),
                err_mediana=semana_mediana(s) - semana_mediana(o),
                err_altura=float((np.max(s) - np.max(o)) / np.max(o)),
                rmse_log=float(np.sqrt(np.mean((np.log1p(s) - np.log1p(o)) ** 2))))


def main():
    S0_all = list(J["S0_temporadas"].values())
    fil, reg_fil, pars = [], [], {}
    for i, t in enumerate(TEMPS):
        idx = [j for j in range(len(TEMPS)) if j != i]
        x0 = np.array([J["comunes"]["R0max"], J["comunes"]["d0"], np.log(J["comunes"]["rep"]), J["comunes"]["social_navidad"]]
                      + [S0_all[j] for j in idx])
        r = calibrar_sub(idx, x0)
        p = dict(FIJOS, R0max=float(r.x[0]), d0=float(r.x[1]), rep=float(np.exp(r.x[2])), eps=c4.EPS_FIJO,
                 social_navidad=float(r.x[3]))
        S0_pred = float(np.mean(r.x[4:]))
        pars[t] = dict(comunes=p, S0_calibracion=dict(zip([TEMPS[j] for j in idx], map(float, r.x[4:]))),
                       S0_prediccion=S0_pred, perdida=float(r.fun))
        o = cal.nat(cal.CTX[i]["obs"])
        s, inc = c4.correr(i, S0_pred, p)
        ref = np.mean([cal.nat(cal.CTX[j]["obs"]) for j in idx], 0)          # curva media de las otras temporadas
        # dentro de muestra: modelo calibrado con las cuatro temporadas y su S0 propio
        s_in, _ = c4.correr(i, S0_all[i], {**J["comunes"]})
        for nom, y in [("dentro de muestra (calibrado con las 4 temporadas)", s_in), ("fuera de muestra", s), ("curva media de otras temporadas", ref)]:
            fil.append(dict(temporada=t, modelo=nom, **metricas(o, y)))
        if t in c4.PR:
            ob = np.array(c4.PR[t]["obs"], float)
            regm = c4.v2a(inc.sum(2), cal.CTX[i]["fechas"]) / (m.N / 1e5) * p["rep"]
            for j, c in enumerate(m.COD):
                if c4.PR[t]["fuente"].get(c) != "regional" or np.nansum(ob[:, j]) <= 0:
                    continue
                reg_fil.append(dict(temporada=t, cod=c, med_obs=semana_mediana(ob[:, j]), med_mod=semana_mediana(regm[:, j]),
                                    med_ref=semana_mediana(np.mean([cal.nat(cal.CTX[k]["obs"]) for k in idx], 0)),
                                    r=float(np.corrcoef(np.nan_to_num(ob[:, j]), regm[:, j])[0, 1])))
        print(t, "S0 pred", round(S0_pred, 3), {k: round(v, 3) for k, v in p.items() if isinstance(v, float)}, flush=True)
    F = pd.DataFrame(fil); F.to_csv("resultados/validacion_loso.csv", index=False)
    R = pd.DataFrame(reg_fil); R.to_csv("resultados/validacion_loso_regional.csv", index=False)
    json.dump(pars, open("resultados/parametros_loso.json", "w"), indent=1)
    F["abs_pico"] = F.err_pico.abs(); F["abs_med"] = F.err_mediana.abs(); F["abs_alt"] = F.err_altura.abs()
    print(F.round(2).to_string())
    print(F.groupby("modelo")[["r", "abs_pico", "abs_med", "abs_alt", "rmse_log"]].mean().round(2).to_string())
    if len(R):
        R["err_mod"] = (R.med_mod - R.med_obs).abs(); R["err_ref"] = (R.med_ref - R.med_obs).abs()
        print("Regional, error medio en la semana mediana: modelo", round(R.err_mod.mean(), 2), "| referencia", round(R.err_ref.mean(), 2),
              "| dentro de 2 semanas:", f"{(R.err_mod <= 2).mean():.0%}", "n =", len(R))


if __name__ == "__main__":
    main()
