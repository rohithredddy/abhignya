#!/usr/bin/env bash
# Downloads the open MIMIC-IV Clinical Database Demo (ODbL) into data/raw/mimic-iv-demo
set -euo pipefail
DEST="data/raw/mimic-iv-demo"
mkdir -p "$DEST"
wget -r -N -c -np -nH --cut-dirs=3 -P "$DEST" https://physionet.org/files/mimic-iv-demo/2.2/
echo "Demo downloaded to $DEST"
