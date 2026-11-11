# Full MIMIC-IV run (run locally; CONTRIBUTING.md §4).
# Prereq: MIMIC-IV hosp/ and icu/ folders under data/raw/mimiciv/ (credentialed PhysioNet access).
# Usage (PowerShell, repo root):  .\scripts\run_full.ps1          # smoke test, then full run
#                                 .\scripts\run_full.ps1 -SkipSmoke
param([switch]$SkipSmoke)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$ov = "configs/full.yaml"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $t = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name" -ForegroundColor Red; exit 1 }
    Write-Host ("done in {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
}

foreach ($d in "data/raw/mimiciv/hosp", "data/raw/mimiciv/icu") {
    if (-not (Test-Path $d)) { Write-Host "Missing $d - download MIMIC-IV first." -ForegroundColor Red; exit 1 }
}

Step "1. Verify itemids/units (inspect results/verify_items/items.csv if this fails)" {
    & $py scripts/run.py verify --override $ov }
if (-not $SkipSmoke) {
    Step "2. Smoke test on 5% of patients" { & $py scripts/run.py all --override $ov --sample-frac 0.05 }
}
Step "3. Full run (E1-E7)" { & $py scripts/run.py all --override $ov }
Step "4. Paper figures and tables" { & $py scripts/make_figures.py }

$zip = "results_full_{0:yyyyMMdd}.zip" -f (Get-Date)
Compress-Archive -Path results\* -DestinationPath $zip -Force
Write-Host "`nAll done. Share ONLY results/ (or $zip) - aggregates, small cells suppressed." -ForegroundColor Green
Write-Host "Never share anything under data/." -ForegroundColor Yellow
