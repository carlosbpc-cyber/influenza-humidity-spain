"""
Entradas del modelo v4 (por edades):
  - Población por grupo de edad (0-14, 15-64, 65+) y comunidad.
  - Matrices de contacto sintéticas para España (Prem et al. 2021) por ámbito: hogar, escuela, trabajo, otros.
  - Calendario escolar por comunidad (aproximado por reglas, ver CALENDARIO_NOTAS).
  - Vacunación por temporada, comunidad y edad; efectividad vacunal por temporada según subtipos.
"""
import os, json
import numpy as np, pandas as pd
import modelo_gripe_ccaa_v2 as v2

AQ = v2.AQUI
COD, POB = v2.COD, v2.POB
EDADES = ["0-14", "15-64", "65+"]

# ---------------------------------------------------------------- población por edad
# Denominadores de la campaña de vacunación COVID-19 (Ministerio de Sanidad, recopilados por Datadista, 2021-22):
# población por grupos 12-19, 20-29, ..., 70-79, >=80 y 5-11 por comunidad. 0-4 = 5-11 * 5/7 * 0,80 (natalidad decreciente,
# ajustado a ~1,87 millones en España). 0-14 = 0-4 + 5-11 + 3/8 de 12-19; 65+ = 70-79 + >=80 + mitad de 60-69.
_NOMBRES_DATADISTA = {"AND": "Andalucía", "ARA": "Aragón", "AST": "Asturias", "BAL": "Baleares", "CAN": "Canarias",
                      "CNT": "Cantabria", "CYL": "Castilla y León", "CLM": "Castilla La Mancha", "CAT": "Cataluña",
                      "VAL": "C. Valenciana", "EXT": "Extremadura", "GAL": "Galicia", "MAD": "Madrid", "MUR": "Murcia",
                      "NAV": "Navarra", "PVA": "País Vasco", "RIO": "La Rioja"}
ARCH_EDAD = os.path.join(AQ, "datos", "poblacion_edad_ccaa.csv")


def _construir_edades():
    base = os.environ.get("DATADISTA_COVID_DIR", "ext/datasets/COVID 19/")  # https://github.com/datadista/datasets
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    d = pd.read_csv(base + "ccaa_vacunas_grupos_etarios_1_dosis.csv"); d = d[d["Fecha publicación"] == d["Fecha publicación"].max()]
    g = ["≥80 años", "70-79 años", "60-69 años", "50-59 años", "40-49 años", "30-39 años", "20-29 años", "12-19 años"]
    P = pd.DataFrame({x: (num(d["Vacunados " + x]) / (num(d["% " + x]) / 100)).values for x in g}, index=d.CCAA.values)
    p = pd.read_csv(base + "ccaa_vacunas_pediatricas_5_11.csv", usecols=[0, 1, 2], on_bad_lines="skip")
    p = p[p["Fecha publicación"] == "2022-02-17"].set_index("CCAA"); P["5-11"] = num(p["Población 5-11 años"]).reindex(P.index)
    P["0-4"] = P["5-11"] * 5 / 7 * 0.80
    fil = []
    for c in COD:
        r = P.loc[_NOMBRES_DATADISTA[c]]
        a = r["0-4"] + r["5-11"] + 3 / 8 * r["12-19 años"]
        v = r["70-79 años"] + r["≥80 años"] + 0.5 * r["60-69 años"]
        t = r[["0-4", "5-11"] + g].sum()
        fil.append(dict(cod=c, p0_14=a / t, p15_64=(t - a - v) / t, p65=v / t, p0_4_en_0_14=r["0-4"] / a))
    pd.DataFrame(fil).round(4).to_csv(ARCH_EDAD, index=False)


if not os.path.exists(ARCH_EDAD):
    _construir_edades()
EDAD = pd.read_csv(ARCH_EDAD).set_index("cod").loc[COD]
FRAC = EDAD[["p0_14", "p15_64", "p65"]].values               # [n, 3]
NA = (POB * 1e6)[:, None] * FRAC                            # población [n, 3]

# ---------------------------------------------------------------- matrices de contacto
ARCH_C = os.path.join(AQ, "datos", "contactos_esp_3edades.json")


def _construir_contactos():
    ruta = os.environ.get("PREM_MATRICES_DIR", "ext/synthetic-contact-matrices/output/syntheticmatrices/")  # https://github.com/kieshaprem/synthetic-contact-matrices
    # pesos de población nacional por grupo quinquenal (0-4 ... 75+), de los mismos denominadores
    w = np.array([1.87, 2.35, 2.45, 2.49, 2.45, 2.48, 2.80, 3.14, 3.91, 3.91, 3.65, 3.48, 2.95, 2.56, 2.15, 4.80])  # millones, 2021-22
    grupos = [range(0, 3), range(3, 13), range(13, 16)]
    out = {}
    import rdata
    for amb in ["home", "school", "work", "others"]:
        M = np.asarray(rdata.read_rda(ruta + f"contact_{amb}.rdata")[f"contact_{amb}"]["ESP"], float)  # M[i,j]: contactos de i con j
        C = np.zeros((3, 3))
        for I, gi in enumerate(grupos):
            for J, gj in enumerate(grupos):
                C[I, J] = sum(w[i] * M[i, j] for i in gi for j in gj) / sum(w[i] for i in gi)
        out[amb] = C.round(4).tolist()
    json.dump(dict(fuente="Prem et al. 2021, matrices sintéticas, España; agregadas a 0-14/15-64/65+", **out), open(ARCH_C, "w"), indent=1)


if not os.path.exists(ARCH_C):
    _construir_contactos()
_CJ = json.load(open(ARCH_C))
CONT = {k: np.array(_CJ[k]) for k in ["home", "school", "work", "others"]}

# ---------------------------------------------------------------- calendario escolar
CALENDARIO_NOTAS = ("Reglas aproximadas: Navidad 22 dic-7 ene en todas; puente de la Constitución 6-8 dic en todas; "
                    "Semana Santa: 10 días (sábado previo al Domingo de Ramos a Lunes de Pascua) salvo CAT, VAL, BAL, PVA y NAV "
                    "(Jueves Santo a lunes de la semana de Pascua+1); Carnaval: CNT semana completa (calendario bimestral), "
                    "CAN 5 días, GAL y EXT lunes-miércoles + fin de semana, CYL lunes-martes; otoño: CNT semana alrededor del 1 nov.")
PASCUA = {2023: "2023-04-09", 2024: "2024-03-31", 2025: "2025-04-20", 2026: "2026-04-05", 2027: "2027-03-28"}
CARNAVAL_LUNES = {2023: "2023-02-20", 2024: "2024-02-12", 2025: "2025-03-03", 2026: "2026-02-16", 2027: "2027-02-08"}


def vacaciones(fechas):
    """Devuelve (escuela_cerrada [dias, n] bool, navidad [dias] bool)."""
    f = pd.DatetimeIndex(fechas); cerr = np.zeros((len(f), len(COD)), bool)
    nav = np.array([(x.month == 12 and x.day >= 22) or (x.month == 1 and x.day <= 7) for x in f])
    puente = np.array([x.month == 12 and 6 <= x.day <= 8 for x in f])
    cerr |= (nav | puente)[:, None]
    for y in sorted(set(f.year)):
        if y not in PASCUA: continue
        P = pd.Timestamp(PASCUA[y]); L = pd.Timestamp(CARNAVAL_LUNES[y])
        for i, c in enumerate(COD):
            ini, fin = (P - pd.Timedelta(days=3), P + pd.Timedelta(days=8)) if c in ("CAT", "VAL", "BAL", "PVA", "NAV") \
                else (P - pd.Timedelta(days=8), P + pd.Timedelta(days=1))
            cerr[(f >= ini) & (f <= fin), i] = True
            car = {"CNT": (-2, 6), "CAN": (-2, 4), "GAL": (-2, 2), "EXT": (-2, 2), "CYL": (-2, 1)}.get(c)
            if car: cerr[(f >= L + pd.Timedelta(days=car[0])) & (f <= L + pd.Timedelta(days=car[1])), i] = True
    i = COD.index("CNT")
    for y in sorted(set(f.year)):
        cerr[(f >= pd.Timestamp(f"{y}-10-29")) & (f <= pd.Timestamp(f"{y}-11-06")), i] = True
    return cerr, nav


# ---------------------------------------------------------------- vacunación
# Mayores de 65 (gripómetro Sanofi y Ministerio): nacional 2022/23 67,3 %; 2023/24 64,4 %; 2024/25 64,9 %; 2025/26 58,5 %.
# Comunidades con dato publicado (diferencia con la media nacional del mismo año aplicada a todas las temporadas).
COB65_NAC = {2022: 0.673, 2023: 0.644, 2024: 0.649, 2025: 0.585, 2026: 0.585}
COB65_DIF = {"GAL": +0.10, "RIO": +0.10, "NAV": +0.08, "CYL": +0.06, "PVA": +0.06, "AST": +0.07, "CLM": +0.06, "AND": +0.035}
# 6-59 meses: 2022/23 solo GAL 51 %, AND 42 %, MUR 44 % (resto ~3 %); 2023/24 nacional 36 %; 2024/25 45,5 %; 2025/26 52,4 %
# (GAL 71 %, AND 65 %, AST 63 %, RIO 56 %).
COB_INF = {2022: dict(_=0.03, GAL=0.513, AND=0.422, MUR=0.443),
           2023: dict(_=0.33, GAL=0.60, AND=0.50, MUR=0.50),
           2024: dict(_=0.42, GAL=0.65, AND=0.58, MUR=0.55, AST=0.55),
           2025: dict(_=0.48, GAL=0.711, AND=0.646, AST=0.626, RIO=0.564, MUR=0.60),
           2026: dict(_=0.48, GAL=0.711, AND=0.646, AST=0.626, RIO=0.564, MUR=0.60)}
COB_15_64 = 0.12          # grupos de riesgo, 60-64 años (35 %), sanitarios
# Efectividad frente a gripe atendida en primaria (VEBIS/I-MOVE; Madrid 2025/26): A(H1N1) 0,46; A(H3N2) 0,36 (0,45 en 2025/26); B 0,70
VE_SUB = dict(h1=0.46, h3=0.36, b=0.70)
SUB = pd.read_csv(os.path.join(AQ, "datos", "europa_subtipos.csv"))
SUB_ES = {int(r.temp): dict(h1=r.h1, h3=r.h3, b=r.b) for _, r in SUB[SUB.countryname == "Spain"].iterrows()}


def efectividad(anio, fB=None):
    s = SUB_ES.get(anio, dict(h1=0.45, h3=0.45, b=0.10)).copy()
    if fB is not None:
        a = s["h1"] + s["h3"] or 1; s = dict(h1=(1 - fB) * s["h1"] / a, h3=(1 - fB) * s["h3"] / a, b=fB)
    ve = dict(VE_SUB); ve["h3"] = 0.45 if anio == 2025 else ve["h3"]
    return sum(s[k] * ve[k] for k in s) / sum(s.values())


def inmunes_vacuna(anio, fB=None):
    """Fracción protegida por la vacuna al inicio de la temporada [n, 3]."""
    ve = efectividad(anio, fB); ci = COB_INF[min(anio, 2026)]
    out = np.zeros((len(COD), 3))
    for i, c in enumerate(COD):
        out[i, 0] = ci.get(c, ci["_"]) * EDAD.loc[c, "p0_4_en_0_14"] * ve
        out[i, 1] = COB_15_64 * ve
        out[i, 2] = min(COB65_NAC[min(anio, 2026)] + COB65_DIF.get(c, 0), 0.9) * ve * 0.8   # menor respuesta en mayores
    return out


if __name__ == "__main__":
    print(EDAD.round(3).to_string())
    print("P65 v2 vs nuevo:", np.round(v2.P65, 1), np.round(FRAC[:, 2] * 100, 1))
    for k, v in CONT.items(): print(k, np.round(v, 2).tolist())
    c, nav = vacaciones(pd.date_range("2025-09-29", "2026-05-31"))
    print("días sin escuela por CCAA 2025/26:", dict(zip(COD, c.sum(0))))
    for y in (2022, 2023, 2024, 2025, 2026): print(y, "VE", round(efectividad(y), 2), "inmunes 65+ (AND,GAL,MAD)", inmunes_vacuna(y)[[0, 11, 12], 2].round(2), "0-14", inmunes_vacuna(y)[[0, 11, 12], 0].round(3))
