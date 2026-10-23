# Hypothesis No. 4: llama bench A/B matrix for the early-router L3 prefetch patch
# in the Tiel fork (LLAMA_MOE_PREFETCH*, src/llama-moe-prefetch.cpp). One binary,
# configurations differ only by environment; "off" opens and closes the run.
# Usage: powershell -File bench\moe_prefetch_matrix.ps1 <bin dir> <log file> [reps]
param([string]$Bin, [string]$Log, [int]$Reps = 3)
$m = "D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
$a = "bench -m $m -ngl 99 -ncmoe 40 -t 6 -C 555 --cpu-strict 1 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -lm none -p 0 -n 128 -r $Reps".Split(' ')
$keys = "LLAMA_MOE_PREFETCH","LLAMA_MOE_PREFETCH_MODE","LLAMA_MOE_PREFETCH_PRIO","LLAMA_MOE_PREFETCH_CPUS","LLAMA_MOE_PREFETCH_SPIN_US","LLAMA_MOE_PREFETCH_STOP","LLAMA_MOE_PREFETCH_STATS","LLAMA_MOE_PREFETCH_WAIT"
$cfg = @(
  @{n="off";              e=@{}},
  @{n="timing-only";      e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_MODE="3"}},
  @{n="cv-159";           e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="1,5,9"}},
  @{n="addr-159";         e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="1,5,9"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="addr-357911";      e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="3,5,7,9,11"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="addr-59";          e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="5,9"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="addr-3711";        e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="3,7,11"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="addr-all6";        e=@{LLAMA_MOE_PREFETCH="4"; LLAMA_MOE_PREFETCH_CPUS="1,3,5,7,9,11"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="addr-357911-m5";   e=@{LLAMA_MOE_PREFETCH="5"; LLAMA_MOE_PREFETCH_CPUS="3,5,7,9,11"; LLAMA_MOE_PREFETCH_WAIT="1"}},
  @{n="off-end";          e=@{}})
if ($env:MATRIX_ONLY) { $cfg = $cfg | Where-Object { $env:MATRIX_ONLY.Split(',') -contains $_.n } }
Set-Location $Bin
"# $(Get-Date -Format s) bin=$Bin reps=$Reps" | Out-File -Encoding utf8 $Log
foreach ($c in $cfg) {
  foreach ($k in $keys) { [Environment]::SetEnvironmentVariable($k, $null, "Process") }
  [Environment]::SetEnvironmentVariable("LLAMA_MOE_PREFETCH_STATS", "1", "Process")
  foreach ($k in $c.e.Keys) { [Environment]::SetEnvironmentVariable($k, $c.e[$k], "Process") }
  $o = & .\llama.exe @a 2>&1 | ForEach-Object { "$_" }
  $tg = (($o | Select-String "tg128") | Select-Object -Last 1).Line -replace '.*tg128 \|\s*','' -replace '\s*\|\s*$',''
  $st = (($o | Select-String "moe_prefetch: \d+ layers") | Select-Object -Last 1).Line
  $line = "{0,-18} {1,-16} {2}" -f $c.n, $tg, $st
  $line; $line | Out-File -Append -Encoding utf8 $Log
}
