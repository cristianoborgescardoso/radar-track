param(
    [Parameter(Position = 0)]
    [string]$InputFile = "plots.json",

    [Parameter(Position = 1)]
    [string]$OutputFile = "tracks.json",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# Create isolated virtual environment and install dependencies automatically if missing
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "[setup] Creating isolated virtual environment in $VenvDir..." -ForegroundColor Cyan
    python -m venv $VenvDir
    if (Test-Path -LiteralPath $VenvPip) {
        & $VenvPip install --upgrade pip -q
        & $VenvPip install -q -r (Join-Path $ScriptDir "requirements.txt")
    }
}

$PythonBin = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

& $PythonBin (Join-Path $ScriptDir "radar_tracker\tracker.py") $InputFile $OutputFile @ExtraArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
