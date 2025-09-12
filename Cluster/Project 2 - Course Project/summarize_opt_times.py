#!/usr/bin/env python
import sys, os, re, json, glob

def parse_training_seconds(logdir: str) -> float:
    secs = []
    for p in glob.glob(os.path.join(logdir, "timing_rank*.log")):
        try:
            for line in open(p, "r", encoding="utf-8", errors="ignore"):
                m = re.search(r"\[Training\]\s+([\d\.]+)\s+sec", line)
                if m: secs.append(float(m.group(1)))
        except Exception:
            pass
    return max(secs) if secs else float("nan")

def read_meta(logdir: str):
    for name in ("meta.final.json", "meta.json"):
        mp = os.path.join(logdir, name)
        if os.path.exists(mp):
            try:
                return json.load(open(mp, "r", encoding="utf-8"))
            except Exception:
                pass
    return {}

def read_wall(logdir: str):
    wp = os.path.join(logdir, "wallclock_seconds.txt")
    if os.path.exists(wp):
        try:
            return int(open(wp, "r", encoding="utf-8").read().strip())
        except Exception:
            return None
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_opt_times.py logs/<JOBID> [logs/<JOBID> ...]")
        sys.exit(1)

    print("OPT-2.7B LoRA — PP + ZeRO-1 Training Times")
    print("-" * 70)
    for ld in sys.argv[1:]:
        meta = read_meta(ld)
        tsec = parse_training_seconds(ld)
        wall = read_wall(ld)
        nnodes = meta.get("nnodes", "?")
        job_id = meta.get("job_id", os.path.basename(ld))
        label = f"N={nnodes!s:>2}  job={job_id}"
        right = f"train={tsec:.2f}s" if tsec == tsec else "train=NA"
        if wall is not None:
            right += f"  wall={wall}s"
        print(f"{label:<40} {right}")
