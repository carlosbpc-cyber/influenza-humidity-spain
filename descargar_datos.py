#!/usr/bin/env python3
"""
Descarga los datos abiertos que faltan para reconstruir el modelo de gripe por CCAA.
Ejecutar una vez desde la raíz del repositorio (necesita `curl`):

    python3 descargar_datos.py

Solo usa la biblioteca estándar de Python y `curl`. Se puede interrumpir y
volver a lanzar: lo ya descargado no se repite. Deja todo en la subcarpeta "datos_nuevos".

  1. Clima diario ERA5 (Open-Meteo): temperatura, punto de rocío y presión en superficie
     - 17 capitales autonómicas: octubre-febrero 2010/11-2014/15 y septiembre-mayo 2015/16-2026/27
     - 22 capitales europeas, octubre-febrero, 2010/11-2024/25
     La API gratuita limita el volumen por hora; el script hace pausas solas (duración total ~1-3 h).
  2. Viajes entre municipios por teléfono móvil (MITMA, estudios básicos v2), 4 días de 2024,
     agregados a matriz entre comunidades (se descarga ~0,5 GB y no se guarda el fichero original).
  3. INE: turistas internacionales por comunidad de destino (FRONTUR, tabla 10823) y población.
"""
import csv, gzip, io, json, os, subprocess, sys, time
from datetime import date

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "datos_nuevos")
os.makedirs(os.path.join(OUT, "clima"), exist_ok=True)


def curl(url, destino=None, intentos=4):
    for k in range(intentos):
        args = ["curl", "-sS", "-L", "--fail-with-body", "--max-time", "600", url]
        if destino:
            args[1:1] = ["-o", destino]
        r = subprocess.run(args, capture_output=True)
        if r.returncode == 0:
            return r.stdout
        msg = (r.stdout or b"")[:300].decode(errors="ignore") + (r.stderr or b"")[:300].decode(errors="ignore")
        if "limit" in msg.lower() or "429" in msg:
            espera = 600 if "hour" in msg.lower() else 65
            print(f"   límite de la API: espero {espera} s ({msg.strip()[:120]})", flush=True)
            time.sleep(espera)
        else:
            print(f"   error ({k + 1}/{intentos}): {msg.strip()[:200]}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError("No se pudo descargar: " + url)


# ------------------------------------------------------------------ 1. clima
ESPANA = {  # capital autonómica: lat, lon
    "AND": (37.389, -5.984), "ARA": (41.649, -0.889), "AST": (43.362, -5.849), "BAL": (39.570, 2.650),
    "CAN": (28.291, -16.000), "CNT": (43.462, -3.810), "CYL": (41.652, -4.724), "CLM": (39.857, -4.024),
    "CAT": (41.389, 2.159), "VAL": (39.470, -0.376), "EXT": (38.917, -6.344), "GAL": (42.881, -8.546),
    "MAD": (40.417, -3.704), "MUR": (37.992, -1.131), "NAV": (42.812, -1.645), "PVA": (42.847, -2.672),
    "RIO": (42.465, -2.445)}
ESPANA["CAN"] = (28.124, -15.430)   # Las Palmas de Gran Canaria (la capitalidad es compartida con Santa Cruz)
EUROPA = {
    "AT": (48.208, 16.373), "BE": (50.850, 4.352), "CH": (46.948, 7.447), "CY": (35.186, 33.382),
    "CZ": (50.076, 14.438), "DE": (52.520, 13.405), "DK": (55.676, 12.568), "EE": (59.437, 24.754),
    "ES": (40.417, -3.704), "FI": (60.170, 24.938), "FR": (48.857, 2.352), "GR": (37.984, 23.728),
    "HU": (47.498, 19.040), "IE": (53.350, -6.260), "LU": (49.611, 6.130), "NL": (52.370, 4.895),
    "NO": (59.914, 10.752), "PL": (52.230, 21.012), "PT": (38.722, -9.139), "SE": (59.329, 18.069),
    "SI": (46.056, 14.506), "SK": (48.149, 17.107)}
VARS = "temperature_2m_mean,dew_point_2m_mean,surface_pressure_mean"
_peso = []   # (tiempo, peso) para no pasar de ~4.500 llamadas ponderadas por hora


def frenar(peso):
    ahora = time.time()
    _peso[:] = [(t, w) for t, w in _peso if ahora - t < 3600]
    while sum(w for _, w in _peso) + peso > 4500 or sum(w for t, w in _peso if ahora - t < 60) + peso > 500:
        time.sleep(20); ahora = time.time()
        _peso[:] = [(t, w) for t, w in _peso if ahora - t < 3600]
    _peso.append((time.time(), peso))


def clima(cod, lat, lon, ini, fin):
    f = os.path.join(OUT, "clima", f"{cod}_{ini}_{fin}.json")
    if os.path.exists(f) and os.path.getsize(f) > 1000:
        return
    dias = (date.fromisoformat(fin) - date.fromisoformat(ini)).days + 1
    frenar(max(1.0, dias / 14))
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={ini}&end_date={fin}&daily={VARS}&models=era5&timezone=UTC")
    txt = curl(url)
    d = json.loads(txt)
    if "daily" not in d:
        raise RuntimeError(f"Respuesta inesperada para {cod}: {str(d)[:300]}")
    json.dump(d["daily"], open(f, "w"))


def bloque_clima():
    hoy = date.today()
    tareas = []
    for y in range(2010, 2027):                      # España: oct-feb (2010-2014); 20 sep-31 may (2015 en adelante)
        ini = date(y, 10, 1) if y < 2015 else date(y, 9, 20)
        fin = date(y + 1, 2, 28) if y < 2015 else date(y + 1, 5, 31)
        fin = min(fin, date.fromordinal(hoy.toordinal() - 6))
        if ini < fin:
            tareas += [(c, *ESPANA[c], ini.isoformat(), fin.isoformat()) for c in ESPANA]
    for y in range(2010, 2025):                      # Europa: 1 oct (y) - 28 feb (y+1)
        tareas += [("EU_" + c, *EUROPA[c], date(y, 10, 1).isoformat(), date(y + 1, 2, 28).isoformat()) for c in EUROPA]
    hechas = 0
    for t in tareas:
        clima(*t)
        hechas += 1
        if hechas % 25 == 0 or hechas == len(tareas):
            print(f"   clima: {hechas}/{len(tareas)} bloques", flush=True)


# ------------------------------------------------------------------ 2. movilidad MITMA
PROV_CCAA = {"04": "AND", "11": "AND", "14": "AND", "18": "AND", "21": "AND", "23": "AND", "29": "AND", "41": "AND",
             "22": "ARA", "44": "ARA", "50": "ARA", "33": "AST", "07": "BAL", "35": "CAN", "38": "CAN", "39": "CNT",
             "05": "CYL", "09": "CYL", "24": "CYL", "34": "CYL", "37": "CYL", "40": "CYL", "42": "CYL", "47": "CYL",
             "49": "CYL", "02": "CLM", "13": "CLM", "16": "CLM", "19": "CLM", "45": "CLM", "08": "CAT", "17": "CAT",
             "25": "CAT", "43": "CAT", "03": "VAL", "12": "VAL", "46": "VAL", "06": "EXT", "10": "EXT", "15": "GAL",
             "27": "GAL", "32": "GAL", "36": "GAL", "28": "MAD", "30": "MUR", "31": "NAV", "01": "PVA", "20": "PVA",
             "48": "PVA", "26": "RIO"}
DIAS_MITMA = ["2024-02-13", "2024-02-15", "2024-02-17", "2024-11-20"]


def bloque_movilidad():
    f_out = os.path.join(OUT, "movilidad_ccaa_mitma.csv")
    if os.path.exists(f_out):
        return
    tot = {}
    for dia in DIAS_MITMA:
        y, mth, d = dia.split("-")
        url = (f"https://movilidad-opendata.mitma.es/estudios_basicos/por-municipios/viajes/ficheros-diarios/"
               f"{y}-{mth}/{y}{mth}{d}_Viajes_municipios.csv.gz")
        tmp = os.path.join(OUT, "_tmp_mitma.csv.gz")
        print(f"   movilidad {dia} (~125 MB)...", flush=True)
        curl(url, tmp)
        with gzip.open(tmp, "rt", encoding="utf-8", errors="ignore") as fh:
            rd = csv.reader(fh, delimiter="|")
            cab = next(rd)
            io_, id_, iv = cab.index("origen"), cab.index("destino"), cab.index("viajes")
            for r in rd:
                o, de = r[io_][:2], r[id_][:2]
                if o in PROV_CCAA and de in PROV_CCAA:
                    k = (dia, PROV_CCAA[o], PROV_CCAA[de])
                    try:
                        tot[k] = tot.get(k, 0.0) + float(r[iv])
                    except ValueError:
                        pass
        os.remove(tmp)
    with open(f_out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dia", "origen", "destino", "viajes"])
        for (dia, o, de), v in sorted(tot.items()):
            w.writerow([dia, o, de, round(v, 1)])


# ------------------------------------------------------------------ 3. INE
def bloque_ine():
    for nombre, url in [("frontur_10823.json", "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/10823?nult=36"),
                        ("poblacion_2853.json", "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/2853?nult=2")]:
        f = os.path.join(OUT, nombre)
        if not os.path.exists(f):
            curl(url, f)


if __name__ == "__main__":
    t0 = time.time()
    for nombre, fn in [("INE (FRONTUR y población)", bloque_ine), ("movilidad MITMA", bloque_movilidad),
                       ("clima ERA5", bloque_clima)]:
        print(f"== {nombre}", flush=True)
        try:
            fn()
            print(f"   hecho ({(time.time() - t0) / 60:.0f} min)", flush=True)
        except Exception as e:
            print(f"   FALLÓ: {e}", flush=True)
    print("Terminado. Los datos están en:", OUT)
