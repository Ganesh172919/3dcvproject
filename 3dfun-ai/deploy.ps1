# 3DFUN AI Standalone Deployment Script (PowerShell)
param(
    [int]$Port = 8010,
    [string]$Device = "cpu"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   3DFUN AI Standalone Studio Deployment (Windows)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "[1/3] Building & Launching via Docker Compose..." -ForegroundColor Yellow
    docker compose up -d --build
    Write-Host "Studio running at: http://localhost:$Port" -ForegroundColor Green
} else {
    Write-Host "[1/3] Docker not detected. Falling back to local Python venv..." -ForegroundColor Yellow
    if (!(Test-Path .venv)) {
        Write-Host "Creating Python virtual environment..."
        python -m venv .venv
        .\.venv\Scripts\python.exe -m pip install --upgrade pip
        .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    }
    Write-Host "[2/3] Seeding database..." -ForegroundColor Yellow
    .\.venv\Scripts\python.exe -m app.seed

    Write-Host "[3/3] Starting Uvicorn Server on port $Port..." -ForegroundColor Green
    .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
}
