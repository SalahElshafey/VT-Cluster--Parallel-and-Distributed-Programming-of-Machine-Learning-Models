#!/usr/bin/env python
import sys, os, re, json, glob

def parse_training_seconds(logdir: str) -> float:
    secs = []
    for p in glob.glob(os.path.join(logdir, "timing_rank*.log")):
        for line in open(p, "r", encoding="utf-8", errors="ignore"):
            m = re.search(r"\[Training\]\s+([\d\.]+)\s+sec", line)
            if m: secs.append(float(m.group(1)))
    return max(secs) if secs else float("nan")

def read_meta(logdir: str):
    mp = os.path.join(logdir, "meta.json")
    return json.load(open(mp)) if os.path.exists(mp) else {}

def read_wall(logdir: str):
    wp = os.path.join(logdir, "wallclock_seconds.txt")
    return int(open(wp).read().strip()) if os.path.exists(wp) else None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_medium_times.py logs/<JOBID> [logs/<JOBID> ...]")
        sys.exit(1)

    print("DistilGPT-2 LoRA — DDP Training Times (medium):")
    print("-" * 70)
    for ld in sys.argv[1:]:
        meta = read_meta(ld)
        tsec = parse_training_seconds(ld)
        wall = read_wall(ld)
        label = f"N={meta.get('nnodes','?'):>2}  dataset={meta.get('dataset','?'):>6}  job={meta.get('job_id','?')}"
        right = f"train={tsec:.2f}s"
        if wall is not None:
            right += f"  wall={wall}s"
        print(f"{label:<40} {right}")
