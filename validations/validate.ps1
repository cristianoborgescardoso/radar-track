param(
    [double]$Start = 360.0,
    [double]$Duration = 60.0,
    [switch]$AllDetections
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$ScriptsDir = Join-Path $PSScriptRoot "scripts"
$OutDir = Join-Path $PSScriptRoot "outputs"
New-Item -ItemType Directory -Force $OutDir | Out-Null

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "External Real-World Validation (Fortem R20 / AERPAW Opt2 Corridor)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$ConvertArgs = @("--start", $Start, "--duration", $Duration)
if (-not $AllDetections) {
    $ConvertArgs += "--airborne-only"
}
& $Python (Join-Path $ScriptsDir "convert_fortem.py") @ConvertArgs
if ($LASTEXITCODE -ne 0) {
    throw "Fortem R20 data conversion failed (convert_fortem.py)."
}

$ValidationPlots = Join-Path $PSScriptRoot "data\aerpaw_opt2_subset\plots.json"
$ValidationTracks = Join-Path $OutDir "tracks.json"
$ValidationConfig = Join-Path $PSScriptRoot "fortem_r20.properties"

& (Join-Path $ProjectRoot "tracker.ps1") $ValidationPlots $ValidationTracks --config $ValidationConfig
if ($LASTEXITCODE -ne 0) {
    throw "Execution of tracker.ps1 on Fortem R20 validation data failed."
}

& $Python (Join-Path $ScriptsDir "evaluate_fortem.py")
if ($LASTEXITCODE -ne 0) {
    throw "Comparative evaluation failed (evaluate_fortem.py)."
}

Write-Host "==================================================================" -ForegroundColor Green
Write-Host "Validation completed successfully! Output: validations/outputs/" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
