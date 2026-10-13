"""Calibra la hospitalización por infección (IHR) por edad con el proxy de hospitalización de SiVIRA por edad
(IRAG x positividad, 2023/24-2025/26) y las infecciones por edad del modelo v4.1 en esas temporadas."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import calibrar_v4 as c4, modelo_v4 as m
J = json.load(open("resultados/parametros_v4.json")); p = J["comunes"]
E = json.load(open("datos/sivira_edad/proxy_edad_raw.json"))
POBG = {"<1": 0.33, "1-4": 1.45, "5-19": 7.4, "20-59": 24.5, "60-69": 5.5, "70-79": 4.0, "≥80": 2.95}   # millones (2021-22)
# reparto de cada grupo de SiVIRA en nuestros tres grupos (millones)
REP = {"<1": [0.33, 0, 0], "1-4": [1.45, 0, 0], "5-19": [4.9, 2.5, 0], "20-59": [0, 24.5, 0], "60-69": [0, 2.95, 2.55],
       "70-79": [0, 0, 4.0], "≥80": [0, 0, 2.95]}
def obs_3(rep, k):
    tasa = {}
    for s in E[rep][k]:
        g, v = s.split(":"); tasa[g] = np.array([float(z) for z in v.split()])[:33].sum()
    num = np.zeros(3); den = np.zeros(3)
    for g, w in REP.items(): num += np.array(w) * tasa[g]; den += np.array(w)
    return num / den                       # por 100.000
fil = {}
for rep, i in [("202440", 1), ("202539", 2), ("202620", 3)]:
    S0 = list(J["S0_temporadas"].values())[i]
    _, inc = c4.correr(i, S0, p)
    inf = inc.sum((0, 1)) / m.NA.sum(0) * 1e5         # infecciones por 100.000 por edad
    fil[c4.CTX[i]["t"]] = dict(hosp_obs=obs_3(rep, "hosp").round(1).tolist(), ap_obs=obs_3(rep, "ap").round(0).tolist(),
                              inf_mod=inf.round(0).tolist(), ihr=(obs_3(rep, "hosp") / inf).round(5).tolist(),
                              consulta_por_infeccion=(obs_3(rep, "ap") / (inf * p["rep"])).round(2).tolist())
    print(c4.CTX[i]["t"], fil[c4.CTX[i]["t"]])
ihr = np.mean([v["ihr"] for v in fil.values()], 0)
print("IHR medio por edad (0-14, 15-64, 65+):", ihr.round(5))
json.dump(dict(temporadas=fil, IHR=ihr.round(5).tolist(),
               nota="IHR = proxy de hospitalización SiVIRA acumulado / infecciones del modelo; consulta_por_infeccion = proxy AP observado / (infecciones x fracción de notificación), >1 indica más consulta o más infección que el modelo en ese grupo"),
          open("resultados/ihr_v4.json", "w"), indent=1, ensure_ascii=False)
