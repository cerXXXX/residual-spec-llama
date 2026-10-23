"""Summarise the last GGML_SCHED_DEBUG graph in a log.

Counts splits, backend changes between adjacent splits, and the bytes of the
split input tensors -- i.e. what physically crosses the backend boundary.
"""
import re
import sys
from collections import Counter

SPLIT = re.compile(r"##\s*SPLIT\s*#(\d+):\s*(\S+)\s*#\s*(\d+)\s*inputs\s*:?(.*)")
INPUT = re.compile(r"\[([^\[\]()]*?)\s*\(\s*([0-9]+)([KM])\)\s*\]")


def parse(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    # the logger may break one split header across lines; rejoin so that the
    # inputs of a split stay on its header line
    text = re.sub(r"\]\s*\n\s*\[", "] [", text)

    graphs = []
    cur = None
    for line in text.splitlines():
        m = SPLIT.search(line)
        if not m:
            continue
        idx, backend, n_inputs, rest = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if idx == 0 or cur is None:
            cur = []
            graphs.append(cur)
        inputs = [(name, int(size) * (1024 if unit == "K" else 1024 * 1024))
                  for name, size, unit in INPUT.findall(rest)]
        cur.append((backend, n_inputs, inputs))
    return graphs


def report(graph, label):
    changes = 0
    xfer = 0
    xfer_all = 0
    by_name = Counter()
    bytes_by_name = Counter()
    prev = None
    for backend, n_inputs, inputs in graph:
        total = sum(sz for _, sz in inputs)
        xfer_all += total
        if prev is not None and backend != prev:
            changes += 1
            xfer += total
            for name, sz in inputs:
                key = re.sub(r"\d+$", "N", name)
                by_name[key] += 1
                bytes_by_name[key] += sz
        prev = backend
    print(f"{label}: splits={len(graph)} backend_changes={changes} "
          f"xfer_at_changes={xfer/1e6:.3f} MB xfer_all_splits={xfer_all/1e6:.3f} MB")
    for key, cnt in bytes_by_name.most_common(8):
        print(f"    {key:24s} n={by_name[key]:4d}  {bytes_by_name[key]/1024:9.1f} KB")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        graphs = parse(path)
        if not graphs:
            print(f"{path}: no sched dump found")
            continue
        print(f"{path}: {len(graphs)} graphs")
        report(graphs[-1], "  last (decode)")
