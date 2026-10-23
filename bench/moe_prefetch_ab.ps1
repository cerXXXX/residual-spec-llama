# Hypothesis No. 4: interleaved llama bench A/B for the early-router L3 prefetch
# (one fork binary, LLAMA_MOE_PREFETCH unset vs set, no stats so no mark nodes).
# Each config is "name=m:cpus:wait[:stop[:mode]]" ("off" = unset); rounds cycle through all.
# Usage: powershell -File bench\moe_prefetch_ab.ps1 <bin dir> <log> [rounds=4] [reps=5] [configs]
param([string]$Bin, [string]$Log, [int]$Rounds = 4, [int]$Reps = 5,
      [string]$Configs = "off;m4-159=4:1,5,9:0")
$model = "D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
$a = "bench -m $model -ngl 99 -ncmoe 40 -t 6 -C 555 --cpu-strict 1 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -lm none -p 0 -n 128 -r $Reps".Split(' ')
$keys = "LLAMA_MOE_PREFETCH","LLAMA_MOE_PREFETCH_CPUS","LLAMA_MOE_PREFETCH_WAIT","LLAMA_MOE_PREFETCH_STATS","LLAMA_MOE_PREFETCH_STOP","LLAMA_MOE_PREFETCH_MODE"
Set-Location $Bin
"# $(Get-Date -Format s) bin=$Bin rounds=$Rounds reps=$Reps configs=$Configs" | Out-File -Encoding utf8 $Log
"# llama.exe " + ($a -join ' ') | Out-File -Append -Encoding utf8 $Log
for ($i = 1; $i -le $Rounds; $i++) {
  foreach ($c in $Configs.Split(';')) {
    foreach ($k in $keys) { [Environment]::SetEnvironmentVariable($k, $null, "Process") }
    $name = $c
    if ($c -ne "off") {
      $name, $spec = $c.Split('=')
      $p = $spec.Split(':')
      $env:LLAMA_MOE_PREFETCH = $p[0]; $env:LLAMA_MOE_PREFETCH_CPUS = $p[1]; $env:LLAMA_MOE_PREFETCH_WAIT = $p[2]
      if ($p.Count -gt 3) { $env:LLAMA_MOE_PREFETCH_STOP = $p[3] }; if ($p.Count -gt 4) { $env:LLAMA_MOE_PREFETCH_MODE = $p[4] }
    }
    $o = & .\llama.exe @a 2>&1 | ForEach-Object { "$_" }
    $tg = (($o | Select-String "tg128") | Select-Object -Last 1).Line -replace '.*tg128 \|\s*','' -replace '\s*\|\s*$',''
    $line = "{0} {1,-10} {2}" -f $i, $name, $tg
    $line; $line | Out-File -Append -Encoding utf8 $Log
  }
}
