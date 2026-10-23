# Nsight Compute profile of the vec FA decode kernel (No. 17), d = 65536, q8_0 KV.
# ncu: UTF-8 copy in D:\tmp\ncu-utf8 (Cyrillic Documents path). TEMP/TMP must be ASCII,
# otherwise the target-side profiler dies with "bad conversion". Reports go to
# D:\tmp\ncu-rep; read them with bench\ncu_export.py (the full raw page also dies).
# -Clock none = real clocks; ncu's default 'base' locks clocks and doubles kernel times.
param([string[]]$Groups = @('0','8'), [string]$Clock = 'base')
$env:TEMP = 'D:\tmp\ncutemp'; $env:TMP = 'D:\tmp\ncutemp'
New-Item -ItemType Directory -Force D:\tmp\ncutemp, D:\tmp\ncu-rep | Out-Null
$ncu = 'D:\tmp\ncu-utf8\target\windows-desktop-win7-x64\ncu.exe'
$bin = 'D:\localllamacpp\bin\tiel-fork-no17-a6eeaa342\llama.exe'
$model = 'D:\localllamacpp\models\Tiel-Coder-35B-A3B-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf'
$d = 'D:\localllamacpp\bench-logs\no17-fa-q8'
$metrics = 'gpu__time_duration.sum,dram__bytes_read.sum,lts__t_sectors_op_read.sum,lts__t_sector_op_read_hit_rate.pct,sm__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed'
foreach ($g in $Groups) {
  $env:GGML_CUDA_FA_VEC_GQA = $g
  $sfx = $(if ($Clock -eq 'none') { '-clk' } else { '' }); $out = "$d\ncu-gqa$g-q8_0-d65536$sfx.txt"
  $a = @('--clock-control',$Clock,'--kernel-name','regex:flash_attn_ext_vec','--launch-skip','20','--launch-count','10','--metrics',$metrics,'-f','-o',"D:\tmp\ncu-rep\gqa$g-q8_0-d65536$sfx",
         $bin,'bench','-m',$model,'-ngl','99','-ncmoe','40','-t','6','-C','555','--cpu-strict','1','-fa','on','-ctk','q8_0','-ctv','q8_0',
         '-b','2048','-ub','2048','-lm','none','-p','0','-n','4','-d','65536','-r','1')
  "[+] Env: GGML_CUDA_FA_VEC_GQA=$g" | Out-File -Encoding utf8 $out
  "[+] Executing: $ncu $($a -join ' ')" | Out-File -Append -Encoding utf8 $out
  $p = Start-Process -FilePath $ncu -ArgumentList $a -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$out.stdout" -RedirectStandardError "$out.stderr"
  Get-Content "$out.stdout" | Out-File -Append -Encoding utf8 $out
  "---- stderr ----" | Out-File -Append -Encoding utf8 $out
  Get-Content "$out.stderr" | Out-File -Append -Encoding utf8 $out
  "exit $($p.ExitCode)" | Out-File -Append -Encoding utf8 $out
  Remove-Item "$out.stdout","$out.stderr"
}
