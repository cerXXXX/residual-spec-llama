# host_mem_log.ps1 - system commit and llama.exe private bytes, once per interval.
#
# An MTP server on the fork fails with "bad allocation" / ggml_init returning
# NULL (hypothesis No. 24): host memory, not VRAM. On WDDM an overcommitted
# card is backed by system memory, so the two are not independent. One CSV row
# per sample: time, committed GiB, commit limit GiB, available MiB, and the
# private bytes / working set of every llama* process.
#
#   powershell -File bench\host_mem_log.ps1 -Out bench-logs\x\hostmem.csv [-IntervalSec 2]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [double]$IntervalSec = 2
)

"time,committed_gib,limit_gib,available_mib,pid,private_gib,ws_gib" | Out-File -FilePath $Out -Encoding utf8
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $limit = $os.TotalVirtualMemorySize / 1MB          # KiB -> GiB
    $committed = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
    $avail = $os.FreePhysicalMemory / 1KB              # KiB -> MiB
    $t = Get-Date -Format "HH:mm:ss"
    $procs = @(Get-Process llama* -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) {
        "$t,$([math]::Round($committed, 2)),$([math]::Round($limit, 2)),$([math]::Round($avail)),,," |
            Out-File -FilePath $Out -Append -Encoding utf8
    }
    foreach ($p in $procs) {
        "$t,$([math]::Round($committed, 2)),$([math]::Round($limit, 2)),$([math]::Round($avail)),$($p.Id)," +
        "$([math]::Round($p.PrivateMemorySize64 / 1GB, 2)),$([math]::Round($p.WorkingSet64 / 1GB, 2))" |
            Out-File -FilePath $Out -Append -Encoding utf8
    }
    Start-Sleep -Seconds $IntervalSec
}
