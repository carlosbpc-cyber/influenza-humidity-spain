"""Calibración del modelo v4 (por edades): kappa y theta por emulación de los efectos europeos; resto con SiVIRA."""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize, brentq
import modelo_v4 as m
import modelo_v3 as m3
import calibrar as cal          # contextos SiVIRA (fechas, q real, Europa, observado)

EFECTO_ANOM = float(os.environ.get("EFECTO_ANOM_SEM", "1.0")) * 7   # semanas de retraso por +1 g/kg (Europa 1,35; España SVGE 1,98)
EFECTO_B = 2.7 * 7      # días de retraso de la semana mediana cuando el 100 % es tipo B
CTX = cal.CTX
EPS_FIJO = 0.03   # ~1,2 % de viajeros infecciosos con positividad europea del 40 % (supuesto; no identificable)
import json as _json, os as _os
PR = _json.load(open(_os.path.join(m.AQ, "datos", "sivira_proxy_regional.json")))
PESO_REG = 0.5   # peso de la forma de las curvas regionales con positividad propia (2024/25 y 2025/26)
DQ = [m3.anomalia_q(c["fechas"], c["q"]) for c in CTX]
FB = [m3.FB_ES[int(c["t"][:4])] for c in CTX]


def nat_sem(inc, fechas, rep):
    return cal.nat(v2a(inc, fechas) / (m.N / 1e5) * rep)


def v2a(inc, fechas):
    return m.v2.a_semanal(inc, fechas).values[:cal.SEM]


def correr(i, S0, p):
    c = CTX[i]
    I0 = np.nan_to_num(np.nanmean(c["obs"][:2], axis=0), nan=1.0) / p["rep"] / 7 * 2.5 / 1e5
    inc, _ = m.simular(c["fechas"], DQ[i], c["eu"], S0, I0, FB[i], p, anio=int(c["t"][:4]))
    return cal.nat(v2a(inc.sum(2), c["fechas"]) / (m.N / 1e5) * p["rep"]), inc


def perdida(x, fijos):
    R0max, d0, rep, eps = x[0], x[1], np.exp(x[2]), EPS_FIJO; S0s = x[4:]
    if not (1.5 < R0max < 6 and -60 < d0 < 90 and 0.5 < x[3] < 2.5 and all(0.3 < s < 0.95 for s in S0s)): return 1e6
    p = dict(fijos, R0max=R0max, d0=d0, rep=rep, eps=eps, social_navidad=x[3])
    L = 0.0
    for i, S0 in enumerate(S0s):
        s, inc = correr(i, S0, p); o = cal.nat(CTX[i]["obs"])
        L += np.mean((np.log1p(s) - np.log1p(o)) ** 2)
        t = CTX[i]["t"]
        if PESO_REG > 0 and t in PR:                                  # forma de las curvas regionales (positividad propia)
            reg = v2a(inc.sum(2), CTX[i]["fechas"]) / (m.N / 1e5) * p["rep"]
            ob = np.array(PR[t]["obs"], float); js = [j for j, c in enumerate(m.COD) if PR[t]["fuente"].get(c) == "regional"]
            on = ob[:, js] / np.nanmean(ob[:, js], 0); mn = reg[:, js] / reg[:, js].mean(0)
            L += PESO_REG * np.nanmean((np.log1p(mn) - np.log1p(on)) ** 2)
    return L


def calibrar(fijos, x0):
    r = minimize(perdida, x0, args=(fijos,), method="Nelder-Mead", options=dict(maxiter=1500, xatol=1e-3, fatol=1e-5, adaptive=True))
    return r


def dia_pico(p, S0, dq_const=0.0, fB=0.0):
    """Simulación de una temporada tipo (fechas de 2024/25, clima normal + anomalía constante en la ventana)."""
    c = CTX[2]; f = c["fechas"]
    dq = np.zeros((len(f), m.n))
    meses = tuple(int(k) for k in os.environ.get("VENTANA_MESES", "10,11,12").split(","))   # revisión: 10,11,12
    ventana = np.array([(x.month in meses) for x in f])
    dq[ventana] = dq_const
    I0 = np.full(m.n, 20 / p["rep"] / 7 * 2.5 / 1e5)
    inc, _ = m.simular(f, dq, c["eu"], S0, I0, fB, p, anio=2024)
    inc = inc.sum(2)
    cum = np.cumsum(inc.sum(1)); return int(np.searchsorted(cum, cum[-1] / 2))   # día mediano de la temporada


def emular(p, S0):
    base = dia_pico(p, S0)
    ks = np.arange(0, -151, -1.5)
    dk = [abs(dia_pico(dict(p, kappa=k), S0, dq_const=0.001) - base - EFECTO_ANOM) for k in ks]
    ths = np.arange(0, 0.40, 0.005)
    dt = [abs(dia_pico(dict(p, theta=t), S0, fB=1.0) - base - EFECTO_B) for t in ths]
    return float(ks[int(np.argmin(dk))]), float(ths[int(np.argmin(dt))])


if __name__ == "__main__":
    fijos = dict(R0min=1.2, mov=4.0, kappa=-18.0, theta=0.035, social_navidad=0.6)
    x0 = np.array([1.7, 50.0, np.log(0.077), 1.2, 0.80, 0.89, 0.78, 0.87])   # x[3]: factor de contactos laborales en Navidad
    for ronda in range(4):
        r = calibrar(fijos, x0); x0 = r.x
        p = dict(fijos, R0max=r.x[0], d0=r.x[1], rep=float(np.exp(r.x[2])), eps=EPS_FIJO, social_navidad=float(r.x[3]))
        S0m = float(np.mean(r.x[4:]))
        try:
            k, th = emular(p, S0m)
        except ValueError as e:
            print("emulación fallida", e); break
        print(f"ronda {ronda}: pérdida {r.fun:.3f}  R0max {r.x[0]:.2f} d0 {r.x[1]:.1f} rep {np.exp(r.x[2]):.3f} "
              f"navidad {r.x[3]:.2f} S0 {np.round(r.x[4:], 3)}  -> kappa {k:.0f} theta {th:.3f}", flush=True)
        if abs(k - fijos["kappa"]) < 10 and abs(th - fijos["theta"]) < 0.01: break
        fijos.update(kappa=float(k), theta=float(th))
    r = calibrar(fijos, x0)
    p = dict(fijos, R0max=float(r.x[0]), d0=float(r.x[1]), rep=float(np.exp(r.x[2])), eps=EPS_FIJO, social_navidad=float(r.x[3]))
    val = []
    for i, S0 in enumerate(r.x[4:]):
        s, _ = correr(i, S0, p); o = cal.nat(CTX[i]["obs"])
        val.append(dict(temporada=CTX[i]["t"], inicio=CTX[i]["fechas"][0].strftime("%Y-%m-%d"),
                        obs=np.round(o, 1).tolist(), mod=np.round(s, 1).tolist(),
                        r=float(np.corrcoef(o, s)[0, 1]), pico_obs=int(o.argmax()), pico_mod=int(s.argmax())))
        print(CTX[i]["t"], "r", round(val[-1]["r"], 2), "pico obs/mod", val[-1]["pico_obs"], val[-1]["pico_mod"], "alt", round(o.max()), round(s.max()))
    json.dump(dict(comunes=p, S0_temporadas=dict(zip(cal.TEMPS, map(float, r.x[4:]))), perdida=float(r.fun),
                   fB_temporadas=dict(zip(cal.TEMPS, FB)), validacion=val),
              open(os.path.join(m.AQ, "resultados", "parametros_v4.json"), "w"), indent=1)
