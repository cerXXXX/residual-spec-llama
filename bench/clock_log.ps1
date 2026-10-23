# clock_log.ps1 - CPU clock and GPU power state, once per interval (hypothesis No. 20).
# A PCIe upload slowed pure-CPU llama bench 2.5-3.6x; this tells a power/clock
# limit (laptop CPU+GPU share a budget) from DDR4 contention.
#
#   powershell -File bench\clock_log.ps1 -Out <csv> [-Seconds 120]
param([Parameter(Mandatory = $true)][string]$Out, [int]$Seconds = 120)

"time,cpu_perf_pct,cpu_mhz,cpu_util_pct,gpu_pstate,gpu_power_w,gpu_util_pct,gpu_mem_mhz" | Out-File $Out -Encoding utf8
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
    $p = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
    $g = (nvidia-smi --query-gpu=pstate,power.draw,utilization.gpu,clocks.mem --format=csv,noheader,nounits) -replace " ", ""
    "$(Get-Date -Format HH:mm:ss),$($p.PercentProcessorPerformance),$($p.ProcessorFrequency),$($p.PercentProcessorTime),$g" |
        Out-File $Out -Append -Encoding utf8
    Start-Sleep -Milliseconds 1000
}
