# PowerShell helper to run Streamlit using the repository virtualenv
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-Not (Test-Path $python)) {
    Write-Error "Could not find virtualenv python at $python. Activate your venv first or ensure .venv exists."
    exit 1
}
& $python -m streamlit run "$PSScriptRoot\dashboard\app.py"
