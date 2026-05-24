#!/usr/bin/env pwsh
# =============================================================================
# run_benchmarks.ps1 - Automate CPU vs GPU timing comparison
#
# Usage (from project root):
#   .\benchmarks\run_benchmarks.ps1
#   .\benchmarks\run_benchmarks.ps1 -BuildDir "build" -Samples 64
# =============================================================================

param(
    [string] $BuildDir = "build",
    [int]    $Samples  = 128,
    [int]    $Width    = 800,
    [int]    $Height   = 600
)

$BinDir = Join-Path $BuildDir "bin"
$CPU    = Join-Path $BinDir "cpu_raytracer.exe"
$GPU    = Join-Path $BinDir "cuda_raytracer.exe"

function Check-Exists {
    param($path)
    if (-not (Test-Path $path)) {
        Write-Error "Binary not found: $path - run 'cmake --build $BuildDir --config Release' first."
        exit 1
    }
}

Check-Exists $CPU
Check-Exists $GPU

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CUDA Ray Tracer - Benchmark Suite"     -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Resolution : ${Width} x ${Height}"
Write-Host "  GPU samples: $Samples spp"
Write-Host ""

# -- CPU baseline --------------------------------------------------------------
Write-Host "[ 1/2 ] Running CPU ray tracer (4 spp)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $CPU 2>&1
$sw.Stop()
$CpuMs = $sw.ElapsedMilliseconds
Write-Host "  Wall time: $CpuMs ms" -ForegroundColor Green

Write-Host ""

# -- GPU render ----------------------------------------------------------------
Write-Host "[ 2/2 ] Running CUDA ray tracer ($Samples spp)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $GPU --samples $Samples --width $Width --height $Height 2>&1
$sw.Stop()
$GpuMs = $sw.ElapsedMilliseconds
Write-Host "  Wall time: $GpuMs ms" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Results"                               -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ("  CPU (4 spp)          : {0,8} ms" -f $CpuMs)
Write-Host ("  GPU ($Samples spp) : {0,8} ms" -f $GpuMs)

if ($GpuMs -gt 0 -and $CpuMs -gt 0) {
    # Normalise to same SPP for a fair speedup ratio
    $GpuEquiv = $GpuMs * 4.0 / $Samples   # GPU time if it had run at 4 spp
    $Speedup  = [math]::Round($CpuMs / $GpuEquiv, 1)
    Write-Host ("  Speedup (normalised) : {0,8}x" -f $Speedup) -ForegroundColor Magenta
}

Write-Host ""
Write-Host "Renders written to: renders/" -ForegroundColor Green
Write-Host "  baseline_render.ppm  - CPU output"
Write-Host "  gpu_render.ppm       - GPU output"
Write-Host ""
