#!/usr/bin/env bash
# Reproduces all analyses of the revised manuscript. Usage: bash run_all.sh [--quick]
#   --quick  skips the calibrations (steps 4-6 and 9) and uses the stored parameters in resultados/
set -euo pipefail
cd "$(dirname "$0")"
QUICK=${1:-}
run() { echo; echo "=== $*"; python3 "$@"; }

run analisis_europa_largo.py
run analisis_svge_regional.py
run humedad_ventanas.py
if [ "$QUICK" != "--quick" ]; then
  run calibrar_v4.py
  run calibrar_ihr_v4.py
  run validacion_loso.py
  run enkf_v4.py
  run enkf_loso.py
fi
run prediccion_inicio.py
run validacion_hosp.py
run escenarios_v4.py
run figuras_revision.py
echo; echo "Done. Results in resultados/, figures in figures/."
