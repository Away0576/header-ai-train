param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonLauncher = "py"

Write-Host "Project root: $ProjectRoot"
Write-Host "Required Python: $PythonVersion"

try {
    & $PythonLauncher "-$PythonVersion" --version
} catch {
    throw "Python $PythonVersion is not available. Install Python $PythonVersion first, then rerun this script."
}

if (!(Test-Path -LiteralPath $VenvPath)) {
    Write-Host "Creating virtual environment: $VenvPath"
    & $PythonLauncher "-$PythonVersion" -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment python not found: $VenvPython"
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Verifying CLI..."
& $VenvPython -m header_ai_train.cli --version

Write-Host "Environment setup completed."
