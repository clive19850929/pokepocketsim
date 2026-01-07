$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "[venv] Activating $venvActivate"
    . $venvActivate
} else {
    Write-Host "[venv] .venv not found. Create one per README.md before running this script."
}

$pyInfo = python -c "import sys; print(f'{sys.executable} (Python {sys.version.split()[0]})')"
Write-Host "[python] $pyInfo"

python -m pip install -U pip pip-tools

Push-Location $repoRoot
pip-compile requirements-torch-cpu.in -o requirements-torch-cpu.txt
pip-compile requirements.in -o requirements.txt -c requirements-torch-cpu.txt
Pop-Location
