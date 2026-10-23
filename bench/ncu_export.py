"""Export selected metrics from .ncu-rep files (full raw page dies with 'bad conversion').

Usage: python bench/ncu_export.py <report.ncu-rep> [...]
Prints medians per metric and writes <report>.metrics.csv next to the report.
"""
import csv, io, os, subprocess, sys, statistics

NCU = r"D:\tmp\ncu-utf8\target\windows-desktop-win7-x64\ncu.exe"
METRICS = [
    "gpu__time_duration.sum", "dram__bytes_read.sum", "lts__t_sectors_op_read.sum",
    "lts__t_sector_op_read_hit_rate.pct", "sm__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_wait.ratio",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread", "launch__occupancy_limit_registers",
    "sm__maximum_warps_per_active_cycle_pct", "launch__grid_size", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]
env = dict(os.environ, TEMP=r"D:\tmp\ncutemp", TMP=r"D:\tmp\ncutemp")

def export(rep):
    out = subprocess.run([NCU, "--import", rep, "--csv", "--page", "raw", 
                          "--metrics", ",".join(METRICS)], capture_output=True, env=env)
    if out.returncode:
        sys.exit(f"{rep}: {out.stdout[-300:]!r} {out.stderr[-300:]!r}")
    rows = list(csv.reader(io.StringIO(out.stdout.decode("utf-8"))))
    hdr, units, data = rows[0], rows[1], rows[2:]
    return hdr, units, data

for rep in sys.argv[1:]:
    hdr, units, data = export(rep)
    print(f"== {os.path.basename(rep)}: {len(data)} kernels, {data[0][hdr.index('Kernel Name')][:60]}")
    for m in METRICS:
        i = hdr.index(m)
        v = [float(r[i].replace(",", "")) for r in data]
        print(f"  {m:70s} {statistics.median(v):16.4f} {units[i]:10s} (min {min(v):.4g} max {max(v):.4g})")
    with open(rep.replace(".ncu-rep", ".metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerow(units); w.writerows(data)
