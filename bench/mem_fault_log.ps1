# mem_fault_log.ps1 - available RAM, page faults and hard faults (pages read from
# disk), system-wide and for llama*, once a second (hypothesis No. 20: why the CPU
# is slower while a PCIe load runs). WMI classes, not Get-Counter: counter names
# are localised on this machine (No. 48).
#
#   powershell -File bench\mem_fault_log.ps1 -Out <csv> [-Seconds 300]
param([Parameter(Mandatory = $true)][string]$Out, [int]$Seconds = 300)

"time,avail_mib,sys_faults_s,sys_pages_in_s,sys_trans_faults_s,llama_ws_mib,llama_faults_s,h2d_ws_mib" |
    Out-File $Out -Encoding utf8
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
    $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $procs = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -Filter "Name LIKE 'llama%' OR Name LIKE 'h2d_load%'"
    $l = $procs | Where-Object { $_.Name -like "llama*" } | Select-Object -First 1
    $h = $procs | Where-Object { $_.Name -like "h2d_load*" } | Select-Object -First 1
    $lws = if ($l) { [math]::Round($l.WorkingSet / 1MB) } else { "" }
    $lpf = if ($l) { $l.PageFaultsPersec } else { "" }
    $hws = if ($h) { [math]::Round($h.WorkingSet / 1MB) } else { "" }
    "$(Get-Date -Format HH:mm:ss),$($m.AvailableMBytes),$($m.PageFaultsPersec),$($m.PagesInputPersec),$($m.TransitionFaultsPersec),$lws,$lpf,$hws" |
        Out-File $Out -Append -Encoding utf8
    Start-Sleep -Milliseconds 1000
}
