$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (!(Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run scripts\setup_env.ps1 first."
}

& $VenvPython --version
& $VenvPython -m pip --version
& $VenvPython -m header_ai_train.cli --version

Write-Host "Environment verification completed."
