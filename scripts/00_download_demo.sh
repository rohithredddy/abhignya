#!/usr/bin/env bash
# Downloads the open MIMIC-IV Clinical Database Demo (ODbL) tables used by CertiVOI
# into data/raw/mimic-iv-demo. Uses wget if available, otherwise curl.
set -euo pipefail
DEST="data/raw/mimic-iv-demo"
BASE="https://physionet.org/files/mimic-iv-demo/2.2"
TABLES="hosp/patients hosp/admissions hosp/labevents hosp/d_labitems icu/icustays"
mkdir -p "$DEST/hosp" "$DEST/icu"
for t in $TABLES; do
  if command -v wget >/dev/null 2>&1; then
    wget -q -N -c -O "$DEST/$t.csv.gz" "$BASE/$t.csv.gz"
  else
    curl -sSfL --retry 3 -o "$DEST/$t.csv.gz" "$BASE/$t.csv.gz"
  fi
  echo "downloaded $t"
done
echo "Demo downloaded to $DEST"
