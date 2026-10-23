# Per-logical-CPU interrupt / DPC / busy time, one row per CPU per second.
# Used for hypothesis No. 48: where do interrupts land while TG runs.
# WMI class, not Get-Counter: counter paths are localized, and a child
# powershell on this (Russian) Windows does not resolve the English ones.
# usage: powershell -File bench\cpu_irq_log.ps1 -Out <csv> -Seconds <n>
param([string]$Out, [int]$Seconds = 120)
'sample,cpu,interrupt,dpc,busy' | Out-File $Out -Encoding utf8
for ($i = 0; $i -lt $Seconds; $i++) {
    $rows = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor
    $lines = foreach ($x in $rows) { "$i,$($x.Name),$($x.PercentInterruptTime),$($x.PercentDPCTime),$($x.PercentProcessorTime)" }
    $lines | Out-File $Out -Append -Encoding utf8
    Start-Sleep -Milliseconds 1000
}
