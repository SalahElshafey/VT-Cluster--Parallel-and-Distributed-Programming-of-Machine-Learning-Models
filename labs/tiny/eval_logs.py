#!/usr/bin/env python3
"""
Evaluate a multi-node DDP run from SLURM logs produced by launch_tiny.sh.

Inputs it expects:
  - ~/slurm_logs/preflight.<JOBID>.txt
  - ~/slurm_logs/*.<JOBID>.*.out  (per-node/per-task SLURM outputs)

What it reports:
  - Rendezvous summary (nnodes, nproc_per_node, rdzv endpoint) + consistency check
  - Ranks seen: min/max rank, WORLD_SIZE candidates, missing ranks (if any)
  - Per-node sanity banner versions (PY, torch, transformers, numpy, datasets)
  - Preflight visibility per node (proj=ok/pkgs=ok) and staging
  - Failure signatures (Traceback, ERROR, RuntimeError, OSError, port collisions, timeouts)

Usage:
  python eval_ddp_logs.py --job 123456
  python eval_ddp_logs.py --job $SLURM_JOB_ID
"""

import argparse, glob, os, re, sys
from collections import defaultdict, Counter

RANK_RE = re.compile(r"\[RANK\s+(\d+)\]\s+WORLD_SIZE=(\d+)")
RDZV_RE = re.compile(r"torchrun:\s+nnodes=(\d+)\s+nproc_per_node=(\d+)\s+node_rank=(\d+)\s+rdzv=([^\s]+)")
SANITY_RE = re.compile(
    r"NODE\s+(\S+)\s+OK\s+->\s+PY\s+(\S+)\s+torch\s+(\S+)\s+tfm\s+(\S+)\s+numpy\s+(\S+)\s+datasets\s+(\S+)\s+root\s+(\S+)"
)
FAIL_PAT = re.compile(r"(Traceback|ERROR|RuntimeError|OSError|Address already in use|Timed out)", re.IGNORECASE)

def read_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[[could not read {path}: {e}]]"

def parse_preflight(path: str):
    status = {}
    if not os.path.exists(path):
        return status
    for line in read_text(path).splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        host = parts[0]
        proj = "missing"; pkgs = "missing"
        for p in parts[1:]:
            if p.startswith("proj="): proj = p.split("=",1)[1]
            if p.startswith("pkgs="): pkgs = p.split("=",1)[1]
        status[host] = (proj, pkgs)
    return status

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="SLURM job id, e.g. 123456")
    ap.add_argument("--logs-dir", default=os.path.expanduser("~/slurm_logs"))
    args = ap.parse_args()

    preflight = os.path.join(args.logs_dir, f"preflight.{args.job}.txt")
    outs = sorted(glob.glob(os.path.join(args.logs_dir, f"*.{args.job}.*.out")))
    if not outs:
        print(f"[!] No log files found matching {args.logs_dir}/*.{args.job}.*.out", file=sys.stderr)
        sys.exit(2)

    # --- gather
    ranks_seen = set()
    world_sizes = Counter()
    missing_rank_check_ws = set()
    failures = []
    rdzv_records = []
    sanity = {}  # host -> tuple( py, torch, tfm, numpy, datasets, root )
    file_fail_counts = Counter()

    for path in outs:
        txt = read_text(path)

        for m in RANK_RE.finditer(txt):
            r = int(m.group(1)); ws = int(m.group(2))
            ranks_seen.add(r); world_sizes[ws] += 1
            missing_rank_check_ws.add(ws)

        for m in RDZV_RE.finditer(txt):
            rdzv_records.append((
                int(m.group(1)),   # nnodes
                int(m.group(2)),   # nproc_per_node
                int(m.group(3)),   # node_rank (per-file)
                m.group(4),        # endpoint host:port
                os.path.basename(path)
            ))

        for m in SANITY_RE.finditer(txt):
            host, py, torchv, tfmv, npv, dsv, root = m.groups()
            sanity[host] = (py, torchv, tfmv, npv, dsv, root)

        # failures
        m_iter = list(FAIL_PAT.finditer(txt))
        if m_iter:
            file_fail_counts[os.path.basename(path)] = len(m_iter)
            failures.extend([f"{os.path.basename(path)}: {m.group(0)}" for m in m_iter])

    # --- preflight
    pf = parse_preflight(preflight)

    # --- summary
    print("="*72)
    print(f" EVALUATION REPORT FOR JOB {args.job}")
    print("="*72)

    # Rendezvous summary
    print("\n[RDZV] Rendezvous records (unique):")
    uniq_rdzv = set((nn, npn, ep) for nn, npn, _nr, ep, _src in rdzv_records)
    for nn, npn, ep in sorted(uniq_rdzv):
        print(f"  - nnodes={nn}  nproc_per_node={npn}  endpoint={ep}")
    if len(uniq_rdzv) == 0:
        print("  (!) No rendezvous banner found. Check launcher output.")
    elif len(uniq_rdzv) > 1:
        print("  (!) Inconsistent rendezvous configs detected across logs.")

    # Ranks & world size
    if ranks_seen:
        print("\n[RANKS] Rank coverage:")
        print(f"  - ranks seen: min={min(ranks_seen)}  max={max(ranks_seen)}  count={len(ranks_seen)}")
        ws_list = sorted(world_sizes.items())
        print(f"  - WORLD_SIZE candidates (value -> observations): {ws_list}")
        if len(missing_rank_check_ws) == 1:
            ws = next(iter(missing_rank_check_ws))
            expected = set(range(ws))
            missing = sorted(expected - ranks_seen)
            if missing:
                print(f"  (!) Missing ranks: {missing[:15]}{' ...' if len(missing)>15 else ''}")
            elif max(ranks_seen) != ws-1:
                print(f"  (!) Max rank {max(ranks_seen)} != WORLD_SIZE-1 ({ws-1})")
            else:
                print("  ✓ All ranks accounted for (max == WORLD_SIZE-1).")
        else:
            print("  (!) Multiple WORLD_SIZE values observed; check consistency.")
    else:
        print("\n[RANKS] (!) No '[RANK r] WORLD_SIZE=w' lines found.")

    # Sanity banners table (versions)
    if sanity:
        print("\n[SANITY] Per-node versions (from sanity probe):")
        hdr = f"{'HOST':<16} {'PY':<6} {'torch':<8} {'tfm':<8} {'numpy':<8} {'datasets':<10} root"
        print("  " + hdr)
        print("  " + "-"*len(hdr))
        for host in sorted(sanity):
            py, tv, tfmv, npv, dsv, root = sanity[host]
            print(f"  {host:<16} {py:<6} {tv:<8} {tfmv:<8} {npv:<8} {dsv:<10} {root}")
        # flag version drifts
        def drift(field_idx):
            vals = Counter(v[field_idx] for v in sanity.values())
            return len(vals) > 1, vals
        for name, idx in [("PY",0),("torch",1),("tfm",2),("numpy",3),("datasets",4)]:
            drifted, vals = drift(idx)
            if drifted:
                print(f"  (!) Version drift in {name}: {dict(vals)}")
    else:
        print("\n[SANITY] (!) No sanity banners found.")

    # Preflight results
    if pf:
        ok = sum(1 for v in pf.values() if v == ("ok","ok"))
        print("\n[PREFLIGHT] Project/pkgs visibility by node:")
        for host in sorted(pf):
            proj, pkgs = pf[host]
            tag = "OK" if (proj, pkgs) == ("ok","ok") else "STAGED/MISSING"
            print(f"  {host:<16} proj={proj:<7} pkgs={pkgs:<7}  -> {tag}")
        print(f"  Summary: {ok}/{len(pf)} nodes had both proj & pkgs visible.")
    else:
        print("\n[PREFLIGHT] (!) No preflight file found.")

    # Failures
    if failures:
        print("\n[FAILURES] Signatures found:")
        # show top offending files
        worst = file_fail_counts.most_common(5)
        for f, c in worst:
            print(f"  - {f}: {c} hits")
        # show a few examples
        for line in failures[:8]:
            print(f"    e.g., {line}")
        if len(failures) > 8:
            print(f"    ... (+{len(failures)-8} more)")
    else:
        print("\n[FAILURES] ✓ No failure signatures detected.")

    # Verdict (basic)
    print("\n[VERDICT]")
    verdict_ok = True
    if not uniq_rdzv: verdict_ok = False
    if not ranks_seen: verdict_ok = False
    if len(missing_rank_check_ws) != 1: verdict_ok = False
    else:
        ws = next(iter(missing_rank_check_ws))
        if (set(range(ws)) - ranks_seen): verdict_ok = False
    if failures: verdict_ok = False
    print("  ✓ PASS" if verdict_ok else "  ✗ CHECK LOGS (see sections above)")

if __name__ == "__main__":
    main()
