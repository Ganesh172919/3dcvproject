param(
    [ValidateRange(1, 65535)][int]$Port = 8010,
    [ValidateSet('cpu', 'directml', 'cuda')][string]$Device = 'cpu',
    [switch]$Install,
    [switch]$InstallModels,
    [switch]$InstallTripoSR,
    [switch]$Seed
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $environment = if ($Device -eq 'cpu') { '.venv' } else { ".venv-$Device" }
    $requirements = if ($Device -eq 'cpu') { 'requirements.txt' } else { "requirements-$Device.txt" }
    $python = Join-Path $PSScriptRoot "$environment\Scripts\python.exe"
    if ($PSBoundParameters.ContainsKey('Device')) { $env:RECON_DEVICE = $Device }
    if ($Install -or -not (Test-Path -LiteralPath $python)) {
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            if (-not (Test-Path -LiteralPath $python)) {
                uv venv --python 3.12 $environment
                if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
            }
            uv pip install --python $python -r $requirements
        } else {
            if (-not (Test-Path -LiteralPath $python)) {
                py -3.12 -m venv $environment
                if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 or uv, then run this script again.' }
            }
            & $python -m pip install -r $requirements
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not install Python dependencies.' }
    }

    if ($Install -or $InstallModels -or $InstallTripoSR -or -not (Test-Path 'engine/models/u2netp.onnx') -or
        -not (Test-Path 'engine/models/depth_anything_v2_small_quantized.onnx')) {
        $modelArgs = @('scripts/fetch_models.py')
        if ($Install -or $InstallModels) { $modelArgs += '--all' }
        elseif ($InstallTripoSR) { $modelArgs += '--triposr' }
        & $python @modelArgs
        if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Check the download error above and rerun.' }
    }
    if ($Seed) {
        & $python -m app.seed
        if ($LASTEXITCODE -ne 0) { throw 'Demo account setup failed.' }
    }
    if (-not $env:APP_URL) { $env:APP_URL = "http://127.0.0.1:$Port" }
    Write-Host "3DFUN AI is starting at http://127.0.0.1:$Port"
    & $python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
