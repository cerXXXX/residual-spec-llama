# no20_contention.ps1 - hypothesis No. 20: how much a PCIe upload like the stock
# expert upload slows CPU work that a HybriMoE split would run at the same time.
#
# Tiel entirely on the CPU (-ngl 0): generation there is DDR4-bound, the worst
# case for contention. Each row is one llama bench process; rows with a load run
# bench\h2d_load.exe (pageable = driver memcpy through staging, like the expert
# buffer; pinned = DMA only) for the whole process, then stop it.
#
#   powershell -File bench\no20_contention.ps1 -Out bench-logs\no20-contention
#   ... -Rows none-1,pageableB-1,pinnedB-1   (suffix B: the load's host thread
#   sleeps in cudaStreamSynchronize instead of spinning on a core)
#   -HideGpu   CUDA_VISIBLE_DEVICES=-1 for llama bench only: a pure CPU process
#   -Test pp   -p 64 -n 0 (more compute per byte) instead of tg (-p 0 -n 32, DDR4-bound)
#   -Mask AAA  the same six cores without logical CPU 0 (interrupts and DPCs, No. 48)
#   -Plain     -t 6 only: no -C/--cpu-strict/--prio/--poll
#   -BufMiB N  source buffer of the load (default 4096)
param([Parameter(Mandatory = $true)][string]$Out,
      [string[]]$Rows = @("none-1", "pageable-1", "pinned-1", "none-2", "pageable-2", "pinned-2"),
      [switch]$HideGpu,
      [string]$Test = "tg",
      [string]$Mask = "555",
      [switch]$Plain,
      [int]$BufMiB = 4096)

$ws = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $ws "bin\tiel-fork-no17-a6eeaa342\llama.exe"
$model = Join-Path $ws "models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
New-Item -ItemType Directory -Force $Out | Out-Null
$env:GGML_OP_OFFLOAD_MIN_BATCH = "128"

$Rows = $Rows | ForEach-Object { $_ -split "," }  # powershell -File passes "a,b,c" as one string
$pre = $(if ($BufMiB -ne 4096) { "b$BufMiB-" } else { "" }) + $(if ($Plain) { "plain-" } else { "" }) + $(if ($Mask -ne "555") { "m$Mask-" } else { "" }) + $(if ($HideGpu) { "nogpu-" } else { "" }) + $(if ($Test -eq "pp") { "pp-" } else { "" })
foreach ($row in $Rows) {
    $load = $row.Split("-")[0]
    $h = $null
    if ($load -ne "none") {
        $pin = if ($load.StartsWith("pinned")) { "1" } else { "0" }
        $blk = if ($load.EndsWith("B")) { "1" } else { "0" }
        $h = Start-Process -FilePath (Join-Path $ws "bench\h2d_load.exe") -ArgumentList "600", "2", $pin, $blk, $BufMiB `
            -RedirectStandardOutput (Join-Path $Out "$pre$row.h2d.txt") -NoNewWindow -PassThru
        Start-Sleep 3
    }
    $args_ = @("bench", "-m", $model, "-ngl", "0", "-t", "6", "-C", $Mask, "--cpu-strict", "1", "--prio", "2",
               "--poll", "50", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-r", "3")
    if ($Plain) { $args_ = @("bench", "-m", $model, "-ngl", "0", "-t", "6", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-r", "3") }
    $args_ += if ($Test -eq "pp") { @("-p", "64", "-n", "0") } else { @("-p", "0", "-n", "32") }
    Write-Host "[$pre$row] $bin $($args_ -join ' ')"
    if ($HideGpu) { $env:CUDA_VISIBLE_DEVICES = "-1" }
    & $bin @args_ 2>&1 | Out-File -FilePath (Join-Path $Out "$pre$row.log") -Encoding utf8
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    if ($h) { Stop-Process -Id $h.Id -Force -ErrorAction SilentlyContinue }
    Select-String -Path (Join-Path $Out "$pre$row.log") -Pattern "tg32|pp64" | ForEach-Object { Write-Host "[$pre$row] $($_.Line)" }
}
