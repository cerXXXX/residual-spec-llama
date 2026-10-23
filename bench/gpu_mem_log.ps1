# gpu_mem_log.ps1 - per-process GPU memory of llama.exe, from the WDDM counters.
#
# nvidia-smi reports only dedicated VRAM, so an overcommitted card looks fine
# there while WDDM quietly backs part of the allocation with system memory.
# "\GPU Process Memory(*)\Shared Usage" is that part (hypothesis No. 24). One
# CSV row per sample: time, pid, dedicated MiB, shared MiB, non-local MiB.
#
#   powershell -File bench\gpu_mem_log.ps1 -Out bench-logs\x\gpumem.csv [-IntervalSec 2]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [double]$IntervalSec = 2
)

"time,pid,dedicated_mib,shared_mib,nonlocal_mib" | Out-File -FilePath $Out -Encoding utf8
$paths = "\GPU Process Memory(*)\Dedicated Usage", "\GPU Process Memory(*)\Shared Usage",
         "\GPU Process Memory(*)\Non Local Usage"
while ($true) {
    # llama.exe of the fork, llama-b10826.exe of the pinned build
    $pids = @(Get-Process llama* -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    if ($pids.Count -gt 0) {
        try {
            $s = (Get-Counter $paths -ErrorAction Stop).CounterSamples
        } catch { Start-Sleep -Seconds $IntervalSec; continue }
        $t = Get-Date -Format "HH:mm:ss"
        foreach ($p in $pids) {
            $mine = $s | Where-Object { $_.InstanceName -like "pid_${p}_*" }
            if (-not $mine) { continue }
            # a process has one instance per adapter; the NVIDIA one is the one with dedicated memory
            $byInst = $mine | Group-Object InstanceName
            foreach ($g in $byInst) {
                $d = ($g.Group | Where-Object { $_.Path -like "*dedicated usage" }).CookedValue
                $sh = ($g.Group | Where-Object { $_.Path -like "*shared usage" }).CookedValue
                $nl = ($g.Group | Where-Object { $_.Path -like "*non local usage" }).CookedValue
                if ($d -lt 64MB -and $sh -lt 64MB) { continue }
                "$t,$p,$([math]::Round($d / 1MB)),$([math]::Round($sh / 1MB)),$([math]::Round($nl / 1MB))" |
                    Out-File -FilePath $Out -Append -Encoding utf8
            }
        }
    }
    Start-Sleep -Seconds $IntervalSec
}
