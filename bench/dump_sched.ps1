# Dump the decode graph of the scheduler.
#
# GGML_SCHED_DEBUG prints the reserve and prefill graphs too, and without an
# actual generation request those are all you get -- so this starts a server,
# sends one short request, and the decode graph is the LAST graph in the log.
# Feed the log to sched_xfer.py.
#
#   .\dump_sched.ps1 -Locality 0 -Out sched_off.log
#   .\dump_sched.ps1 -Locality 1 -Out sched_on.log
#   python sched_xfer.py sched_off.log sched_on.log
#
# -Locality drives GGML_SCHED_SRC_LOCALITY, which only the own build in
# bin/b10826-srclocality understands; the pinned builds ignore it.

param(
    [string]$Locality = "0",
    [string]$Out      = "sched.log",
    [int]   $Port     = 9099,
    [string]$LlamaBin = "D:\localllamacpp\bin\b10826-srclocality\llama.exe",
    [string]$Model    = "D:\localllamacpp\models\gemma-4-26B-A4B-it-qat\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    [switch]$Tiel,                 # Tiel working flags (and model) instead of Gemma's
    [string]$Prefetch = ""         # LLAMA_MOE_PREFETCH for the Tiel fork (hypothesis No. 4)
)

$env:GGML_SCHED_DEBUG        = "2"
$env:GGML_SCHED_SRC_LOCALITY = $Locality

# -lv 10 is what lets GGML_LOG_DEBUG through; `bench` has no such flag, only `serve`
$serveArgs = @("serve","-m",$Model,"-c","16384","-ngl","99","-fa","on","-b","2048","-ub","2048",
               "-ctk","f16","-ctv","f16","-t","6","-tb","6","-C","555","--cpu-strict","1",
               "--n-cpu-moe","23","--host","127.0.0.1","--port","$Port","-lv","10","-np","1")
if ($Tiel) {
    $Model = "D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
    $serveArgs = @("serve","-m",$Model,"-c","131072","-ngl","99","-fa","on","-b","4096","-ub","4096",
                   "-ctk","q8_0","-ctv","q8_0","-t","6","-tb","6","-C","555","--cpu-strict","1","--prio","2","--poll","50",
                   "--n-cpu-moe","40","--load-mode","none","--host","127.0.0.1","--port","$Port","-lv","10","-np","1")
    $env:GGML_OP_OFFLOAD_MIN_BATCH = "128"
}
if ($Prefetch) { $env:LLAMA_MOE_PREFETCH = $Prefetch } else { [Environment]::SetEnvironmentVariable("LLAMA_MOE_PREFETCH", $null, "Process") }

$p = Start-Process -FilePath $LlamaBin -ArgumentList $serveArgs `
                   -RedirectStandardOutput "$Out.out" -RedirectStandardError "$Out" `
                   -PassThru -NoNewWindow

$ok = $false
for ($i = 0; $i -lt 180; $i++) {
    Start-Sleep -Seconds 2
    try {
        if ((Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) {
            $ok = $true; break
        }
    } catch {}
}
if (-not $ok) { Write-Output "server did not come up"; Stop-Process -Id $p.Id -Force; exit 1 }

$body = @{ prompt = "The capital of France is"; n_predict = 8; temperature = 0; seed = 1234 } | ConvertTo-Json
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/completion" -Method Post `
                              -Body $body -ContentType "application/json" -TimeoutSec 120 -UseBasicParsing
    Write-Output ("completion ok: " + $resp.Content.Substring(0, [Math]::Min(160, $resp.Content.Length)))
} catch { Write-Output "completion failed: $_" }

Start-Sleep -Seconds 2
Stop-Process -Id $p.Id -Force
Write-Output "log: $Out"
