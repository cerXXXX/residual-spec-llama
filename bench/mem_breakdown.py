"""mem_breakdown.py - what the Tiel server's private bytes are made of (No. 31).

Starts one server through start_server.py (sweep_bench.Server), takes a
proc_commit snapshot of llama.exe at fixed stages and classifies the private
allocations:

  experts  - the single largest allocation (CPU model buffer);
  idle     - allocations with nothing resident. On WDDM cudaMalloc commits a
             host backing store of the same size in 64 MiB private chunks that
             the CPU never touches (checked with cudart: 2048 MiB cudaMalloc =
             +2048 MiB private, 32 x 64 MiB, 0 resident);
  resident - everything else (heap, pinned CUDA_Host buffers, prompt cache).

    python bench/mem_breakdown.py --logdir bench-logs\\no31-mem --label off [start_server args...]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_bench as sb
import proc_commit as pc
import psutil

MIB = 2**20


def llama_pid(srv):
    for c in psutil.Process(srv.proc.pid).children(recursive=True):
        if c.name().lower().startswith("llama"):
            return c.pid
    return None


def classify(s):
    regs = s["regions"]
    out = {"private_usage": s["private_usage_mib"], "ws": s["ws_mib"],
           "in_va": s["private_va_mib"], "unmapped": s["unmapped_mib"]}
    big = regs[0] if regs else {"mib": 0, "resident_mib": 0}
    out["experts"] = big["mib"]
    out["experts_resident"] = big["resident_mib"]
    idle = [r for r in regs[1:] if r["resident_mib"] is not None and r["resident_mib"] < 0.05 * r["mib"]]
    out["idle"] = sum(r["mib"] for r in idle)
    out["idle_n"] = len(idle)
    out["idle_sizes"] = sorted({round(r["mib"], 1) for r in idle})[-8:]
    out["resident"] = s["private_va_mib"] - out["experts"] - out["idle"]
    out["resident_top"] = [(round(r["mib"], 1), round(r["resident_mib"] or 0, 1))
                           for r in regs[1:] if r not in idle][:10]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--stages", default="idle,short,prefill,other")
    a, launch = ap.parse_known_args()
    os.makedirs(a.logdir, exist_ok=True)

    prompts = sb.load_prompts()
    srv = sb.Server(launch, os.path.join(a.logdir, a.label + ".log"))
    print("cmd:", " ".join(srv.start()), flush=True)
    stages = {}
    try:
        srv.wait_ready()
        pid = llama_pid(srv)
        print("llama pid", pid, flush=True)

        def snap(name):
            time.sleep(2)
            s = pc.snapshot(pid, with_resident=True, min_resident_mib=1)
            c = classify(s)
            c["vram_mib"] = sb.vram_used_mib()
            c["log_line"] = len(srv.lines)
            stages[name] = {"summary": c, "regions": s["regions"][:60]}
            print(f"[{name}] private {c['private_usage']:.0f}  ws {c['ws']:.0f}  experts {c['experts']:.0f}"
                  f" (res {c['experts_resident']:.0f})  idle {c['idle']:.0f} in {c['idle_n']}"
                  f"  resident-other {c['resident']:.0f}  unmapped {c['unmapped']:.0f}  vram {c['vram_mib']}",
                  flush=True)
            print(f"        idle sizes {c['idle_sizes']}  resident top {c['resident_top']}", flush=True)

        for st in a.stages.split(","):
            if st == "short":
                sb.chat(prompts["tg"], 1)
            elif st == "prefill":
                sb.chat(prompts["prefill"], 1)
            elif st == "other":
                sb.chat(prompts["copy"][0], 1)
            elif st == "other2":
                sb.chat(prompts["free"][0], 1)
            elif st == "mid":
                # ~900 new tokens: one ubatch below GGML_SCHED_PREFETCH_MIN, the hybrid MoE range (No. 20)
                src = open(os.path.join(sb.WS, "prompts", "prompt_real.txt"), encoding="utf-8", errors="replace").read()
                sb.chat([{"role": "user", "content": src[:3000] + "\nOK?"}], 1)
            snap(st)
        stages["executing"] = srv.executing_line()
    finally:
        srv.stop()
        with open(os.path.join(a.logdir, a.label + ".json"), "w", encoding="utf-8") as f:
            json.dump(stages, f, indent=1)


if __name__ == "__main__":
    main()
