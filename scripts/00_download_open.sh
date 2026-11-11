#!/usr/bin/env bash
# Download open-access datasets (no credentialing). Usage:
#   bash scripts/00_download_open.sh challenge2019   # PhysioNet/CinC 2019, CC BY 4.0, ~40k files
#   bash scripts/00_download_open.sh eicu_demo       # eICU-CRD demo v2.0.1, ODbL (2 tables)
# Re-run to resume: existing files are skipped. Keep concurrency low (PhysioNet rate-limits).
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  challenge2019)
    base=https://physionet.org/files/challenge-2019/1.0.0/training
    for s in A B; do
      d=data/raw/challenge2019/training_set$s
      mkdir -p "$d"
      curl -sSL "$base/training_set$s/" | grep -o 'href="p[0-9]*\.psv"' | sed 's/href="//;s/"//' \
        | while read -r f; do [ -s "$d/$f" ] || printf 'url = "%s"\noutput = "%s"\n' \
            "$base/training_set$s/$f" "$d/$f"; done > "/tmp/certivoi_cfg_$s.txt"
      curl -sSL --parallel --parallel-max 4 --retry 8 --retry-delay 15 --retry-all-errors \
        --max-time 60 -K "/tmp/certivoi_cfg_$s.txt" || true
      echo "set $s: $(ls "$d" | wc -l) files"
    done ;;
  eicu_demo)
    d=data/raw/eicu-demo
    mkdir -p "$d"
    for t in patient lab; do
      curl -sSL --retry 5 --retry-all-errors -o "$d/$t.csv.gz" \
        "https://physionet.org/files/eicu-crd-demo/2.0.1/$t.csv.gz?download"
    done
    gzip -t "$d/patient.csv.gz" "$d/lab.csv.gz" && echo "eICU demo tables OK" ;;
  *) echo "usage: $0 challenge2019|eicu_demo" >&2; exit 2 ;;
esac
