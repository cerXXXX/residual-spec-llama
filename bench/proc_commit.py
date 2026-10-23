"""proc_commit.py - where a process's private bytes (commit charge) live.

Hypothesis No. 31 side finding: the Tiel server holds ~8 GiB of private bytes
above the 19.3 GiB of experts. This walks the address space with VirtualQueryEx
and splits the commit charge into:

  - private committed regions (MEM_PRIVATE): heap, VirtualAlloc, pinned host
    buffers; each one reported with its resident part (QueryWorkingSetEx);
  - committed copy-on-write / writable image and mapped pages;
  - the rest: PrivateUsage minus everything visible in the address space.
    On WDDM the CUDA driver charges device allocations (cudaMalloc) to the
    process commit without mapping them, so this residue is VRAM.

    python bench/proc_commit.py <pid> [--top 15] [--json out.json]
    python bench/proc_commit.py --name llama --watch 2 --csv out.csv
"""
import argparse, ctypes, ctypes.wintypes as wt, json, sys, time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEM_COMMIT, MEM_PRIVATE, MEM_MAPPED, MEM_IMAGE = 0x1000, 0x20000, 0x40000, 0x1000000
PAGE_WRITECOPY, PAGE_EXECUTE_WRITECOPY = 0x08, 0x80
MIB = 2**20


class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wt.DWORD), ("PartitionId", wt.WORD),
                ("RegionSize", ctypes.c_size_t), ("State", wt.DWORD),
                ("Protect", wt.DWORD), ("Type", wt.DWORD)]


class PMCEX(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


class WSEX(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("VirtualAttributes", ctypes.c_size_t)]


k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t
k32.OpenProcess.restype = wt.HANDLE
psapi.QueryWorkingSetEx.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]


def resident_bytes(h, base, size, page=4096):
    """Resident part of a region, sampled in chunks of 64K pages."""
    n_pages = size // page
    total = 0
    chunk = 65536
    for start in range(0, n_pages, chunk):
        n = min(chunk, n_pages - start)
        arr = (WSEX * n)()
        for i in range(n):
            arr[i].VirtualAddress = base + (start + i) * page
        if not psapi.QueryWorkingSetEx(h, arr, ctypes.sizeof(arr)):
            return None
        total += sum(1 for e in arr if e.VirtualAttributes & 1)
    return total * page


def snapshot(pid, with_resident=True, min_resident_mib=64):
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        raise OSError(ctypes.get_last_error())
    try:
        pm = PMCEX(); pm.cb = ctypes.sizeof(PMCEX)
        psapi.GetProcessMemoryInfo(h, ctypes.byref(pm), pm.cb)
        mbi = MBI(); addr = 0
        allocs = {}          # AllocationBase -> committed private bytes
        cow = 0
        while k32.VirtualQueryEx(h, addr, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            base = mbi.BaseAddress or 0
            if mbi.State == MEM_COMMIT:
                if mbi.Type == MEM_PRIVATE:
                    a = allocs.setdefault(mbi.AllocationBase or 0, [0, []])
                    a[0] += mbi.RegionSize
                    a[1].append((base, mbi.RegionSize))
                elif mbi.Protect & 0xFF in (PAGE_WRITECOPY, PAGE_EXECUTE_WRITECOPY):
                    cow += mbi.RegionSize
            addr = base + mbi.RegionSize
            if addr >= 0x7FFFFFFFFFFF:
                break
        priv_va = sum(v[0] for v in allocs.values())
        regions = []
        for ab, (sz, parts) in sorted(allocs.items(), key=lambda kv: -kv[1][0]):
            res = None
            if with_resident and sz >= min_resident_mib * MIB:
                res = sum(resident_bytes(h, b, s) or 0 for b, s in parts)
            regions.append({"base": hex(ab), "mib": sz / MIB,
                            "resident_mib": None if res is None else res / MIB})
        return {"time": time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}", "pid": pid,
                "private_usage_mib": pm.PrivateUsage / MIB, "ws_mib": pm.WorkingSetSize / MIB,
                "private_va_mib": priv_va / MIB, "cow_mib": cow / MIB,
                "unmapped_mib": (pm.PrivateUsage - priv_va) / MIB,
                "n_private_allocs": len(allocs), "regions": regions}
    finally:
        k32.CloseHandle(h)


def find_pids(name):
    import psutil
    return [p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith(name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", nargs="?", type=int)
    ap.add_argument("--name", default=None, help="process name prefix instead of pid")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    ap.add_argument("--watch", type=float, default=0, help="sample every N s (no resident walk)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.watch:
        out = open(a.csv, "w", encoding="utf-8") if a.csv else sys.stdout
        out.write("time,pid,private_usage_mib,ws_mib,private_va_mib,unmapped_mib,largest_mib,big_allocs_mib\n")
        while True:
            pids = [a.pid] if a.pid else find_pids(a.name or "llama")
            for pid in pids:
                try:
                    s = snapshot(pid, with_resident=False)
                except OSError:
                    continue
                big = s["regions"][0]["mib"] if s["regions"] else 0
                out.write(f'{s["time"]},{pid},{s["private_usage_mib"]:.0f},{s["ws_mib"]:.0f},'
                          f'{s["private_va_mib"]:.0f},{s["unmapped_mib"]:.0f},{big:.0f},'
                          + " ".join(f'{r["mib"]:.0f}' for r in s["regions"] if r["mib"] >= 200) + "\n")
            out.flush()
            time.sleep(a.watch)

    pid = a.pid or find_pids(a.name or "llama")[0]
    s = snapshot(pid)
    print(f'pid {pid}: PrivateUsage {s["private_usage_mib"]:.0f} MiB, WS {s["ws_mib"]:.0f} MiB')
    print(f'  private committed in address space {s["private_va_mib"]:.0f} MiB '
          f'in {s["n_private_allocs"]} allocations; copy-on-write {s["cow_mib"]:.0f} MiB')
    print(f'  not in address space (driver-charged) {s["unmapped_mib"]:.0f} MiB')
    print(f'  largest private allocations:')
    for r in s["regions"][:a.top]:
        res = "" if r["resident_mib"] is None else f'  resident {r["resident_mib"]:9.1f}'
        print(f'    {r["base"]:>16}  {r["mib"]:9.1f} MiB{res}')
    rest = sum(r["mib"] for r in s["regions"][a.top:])
    print(f'    {"the rest":>16}  {rest:9.1f} MiB')
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    main()
