param(
    [ValidateRange(1, 65535)][int]$Port = 8010,
    [ValidateSet('cpu', 'directml', 'cuda')][string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run.ps1') -Port $Port -Device $Device
