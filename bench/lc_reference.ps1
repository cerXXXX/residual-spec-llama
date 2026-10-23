# Hypothesis No. 12: reference answers (bench/tiel_reference.py, 8 prompts, greedy) for the layer-centric fork.
#   a  pf3, -b 4096       - patch inert with the env vars unset: must match fork-pf3 bit for bit
#   b  pf3, -b 32768      - the larger logical batch alone
#   c  pf3 + layer-centric + lazy replan, -b 32768
# Usage: powershell -File bench\lc_reference.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$bin = 'bin\tiel-fork-lc-5152b82df\llama.exe'
$b32 = @('--llama-arg=-b', '--llama-arg=32768')
$runs = @(
    @{ label = 'lc-a-pf3-b4k';     env = @{ GGML_SCHED_PREFETCH = '3' };                                                          extra = @() },
    @{ label = 'lc-b-pf3-b32k';    env = @{ GGML_SCHED_PREFETCH = '3' };                                                          extra = $b32 },
    @{ label = 'lc-c-lc1lazy-b32k'; env = @{ GGML_SCHED_PREFETCH = '3'; LLAMA_LAYER_CENTRIC = '1'; GGML_SCHED_LAZY_REPLAN = '1' }; extra = $b32 }
)
foreach ($r in $runs) {
    foreach ($k in $r.env.Keys) { Set-Item "Env:$k" -Value $r.env[$k] }
    $args_ = @('bench\tiel_reference.py', 'run', '--label', $r.label, '--llama-bin', $bin)
    if ($r.extra.Count -gt 0) { $args_ += '--extra'; $args_ += $r.extra }
    & python @args_
    foreach ($k in $r.env.Keys) { Set-Item "Env:$k" -Value $null }
}
foreach ($pair in @(@('fork-pf3', 'lc-a-pf3-b4k'), @('fork-stock', 'lc-b-pf3-b32k'), @('lc-b-pf3-b32k', 'lc-c-lc1lazy-b32k'), @('fork-stock', 'lc-c-lc1lazy-b32k'))) {
    "== $($pair[0]) vs $($pair[1])"
    & python bench\tiel_reference.py compare $pair[0] $pair[1]
}
