"""
auto_run.py
===========
Auto-scheduled offline-to-online RL experiments with dynamic resource detection.

Detects free GPU memory and CPU cores, then dispatches experiments in parallel
to available slots. Each experiment gets pinned to a specific GPU via
CUDA_VISIBLE_DEVICES.

Experiments:
  1. Offline-only evaluation (1 run, 10 episodes)
  2. Online SAC (3 seeds) — lower-bound baseline
  3. WSRL (3 seeds) — policy-dominant offline-to-online
  4. RLPD (0.50, 3 seeds) — data-dominant offline-to-online
  5. RLPD ablation (0.25 & 0.75, 3 seeds each) — offline ratio sweep

After all experiments finish, runs analyze_results.py for the summary table.

Usage:
    cd offline-sumo/
    python auto_run.py                        # auto-detect resources, full run
    python auto_run.py --n_epochs 100          # shorter run for testing
    python auto_run.py --mem_per_job 3000      # override memory estimate (MB)
    python auto_run.py --only wsrl,rlpd        # only run specific methods
"""

import os, sys, time, argparse, subprocess, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)


# ── Auto-detect SUMO_HOME ───────────────────────────────────────────────────
def detect_sumo_home():
    """Find SUMO_HOME: existing env var > pip eclipse-sumo > /usr/share/sumo."""
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    # Try pip-installed eclipse-sumo: site-packages/sumo/
    try:
        import sumolib
        # .../site-packages/sumolib → .../site-packages/sumo/ (shares "tools" dir)
        site = os.path.dirname(os.path.dirname(sumolib.__file__))
        pip_sumo = os.path.join(site, "sumo")
        if os.path.isdir(os.path.join(pip_sumo, "tools")):
            return pip_sumo
    except Exception:
        pass
    # System install fallback
    for p in ("/usr/share/sumo", "/usr/local/share/sumo"):
        if os.path.isdir(os.path.join(p, "tools")):
            return p
    return None

_SUMO_HOME = detect_sumo_home()
if _SUMO_HOME:
    os.environ["SUMO_HOME"] = _SUMO_HOME
    print(f"SUMO_HOME = {_SUMO_HOME}")
else:
    print("WARNING: could not auto-detect SUMO_HOME. Set it manually:")
    print("  export SUMO_HOME=/path/to/sumo   # dir with tools/ inside")

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",     type=int, default=300)
parser.add_argument("--seeds",        type=str, default="42,123,456")
parser.add_argument("--mem_per_job",  type=int, default=2500,
                    help="estimated GPU memory per job in MB")
parser.add_argument("--cpu_per_job",  type=int, default=2)
parser.add_argument("--min_cpu_free", type=int, default=2,
                    help="reserve this many CPU cores for system")
parser.add_argument("--only",         type=str, default="all",
                    help="comma list of: eval,online,wsrl,rlpd,ablation  (or 'all')")
parser.add_argument("--python",       type=str, default="python",
                    help="python executable (use 'conda run -n ENV python' if needed)")
parser.add_argument("--dry_run",      action="store_true", help="print plan, don't execute")
args = parser.parse_args()

SEEDS = [int(s) for s in args.seeds.split(",")]
SELECTED = set(args.only.split(",")) if args.only != "all" else {"eval", "bc", "resac", "online", "wsrl", "rlpd", "ablation"}


# ── Resource detection ──────────────────────────────────────────────────────
def get_gpu_free_memory():
    """Returns [(gpu_id, free_mb), ...] or [] if no GPU / nvidia-smi missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5).decode()
        return [(int(p[0].strip()), int(p[1].strip()))
                for p in (l.split(',') for l in out.strip().split('\n')) if len(p) == 2]
    except Exception:
        return []


def compute_slots():
    """Compute parallel job slots, one entry per slot = gpu_id (or None for CPU)."""
    n_cpu = os.cpu_count() or 1
    cpu_slots_max = max(1, (n_cpu - args.min_cpu_free) // args.cpu_per_job)

    gpus = get_gpu_free_memory()
    gpu_assignments = []
    for gpu_id, free_mb in gpus:
        n_slots = free_mb // args.mem_per_job
        if n_slots > 0:
            gpu_assignments.extend([gpu_id] * n_slots)
            print(f"  GPU {gpu_id}: {free_mb} MB free → {n_slots} slot(s)")

    if gpu_assignments:
        n = min(len(gpu_assignments), cpu_slots_max)
        return gpu_assignments[:n], "cuda"
    else:
        print(f"  No GPU available → CPU mode, {cpu_slots_max} slot(s)")
        return [None] * cpu_slots_max, "cpu"


# ── Experiment list ─────────────────────────────────────────────────────────
def build_experiments():
    exps = []
    py = args.python.split()

    if "eval" in SELECTED:
        exps.append(("eval_offline",
                    py + ["eval_offline.py", "--n_eval", "10"]))

    if "bc" in SELECTED:
        for s in SEEDS:
            exps.append((f"bc_seed{s}",
                        py + ["train_bc.py",
                              "--seed", str(s), "--n_steps", "100000",
                              "--n_eval", "10"]))

    if "resac" in SELECTED:
        for s in SEEDS:
            exps.append((f"resac_offline_seed{s}",
                        py + ["train_resac_offline.py",
                              "--seed", str(s), "--n_steps", "100000",
                              "--n_eval", "10"]))

    if "online" in SELECTED:
        for s in SEEDS:
            exps.append((f"online_seed{s}",
                        py + ["train_online_only.py",
                              "--seed", str(s), "--n_epochs", str(args.n_epochs)]))

    if "wsrl" in SELECTED:
        for s in SEEDS:
            exps.append((f"wsrl_seed{s}",
                        py + ["train_wsrl.py",
                              "--seed", str(s), "--n_epochs", str(args.n_epochs)]))

    if "rlpd" in SELECTED:
        for s in SEEDS:
            exps.append((f"rlpd_seed{s}",
                        py + ["train_rlpd.py",
                              "--seed", str(s), "--n_epochs", str(args.n_epochs),
                              "--offline_ratio", "0.5"]))

    if "ablation" in SELECTED:
        for ratio in [0.25, 0.75]:
            for s in SEEDS:
                exps.append((f"rlpd_ratio{ratio}_seed{s}",
                            py + ["train_rlpd.py",
                                  "--seed", str(s), "--n_epochs", str(args.n_epochs),
                                  "--offline_ratio", str(ratio)]))

    return exps


# ── Dispatcher ──────────────────────────────────────────────────────────────
def dispatch(slots, device_type, experiments, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    queue = list(experiments)
    active = {}  # slot_idx -> (name, Popen, log_file, start_time)
    total = len(experiments)
    done = 0
    failed = []
    ts_fmt = lambda s: f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"
    t_start = time.time()

    print(f"\nDispatching {total} experiments to {len(slots)} slot(s) [{device_type}]")
    print(f"Logs: {log_dir}/\n")

    while queue or active:
        # Fill empty slots
        for i, gpu_id in enumerate(slots):
            if i in active or not queue:
                continue
            name, cmd = queue.pop(0)
            env = os.environ.copy()
            if device_type == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                cmd = cmd + ["--device", "cuda"]
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
                cmd = cmd + ["--device", "cpu"]
            log_path = os.path.join(log_dir, f"{name}.log")
            log_f = open(log_path, "w")
            log_f.write(f"# Command: {' '.join(cmd)}\n")
            log_f.write(f"# GPU: {gpu_id}, Time: {datetime.datetime.now()}\n\n")
            log_f.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            active[i] = (name, proc, log_f, time.time())
            dev = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
            print(f"  [{ts_fmt(time.time()-t_start)}] [slot {i}/{dev}] START {name}  "
                  f"(queue: {len(queue)}, active: {len(active)}, done: {done}/{total})")

        # Wait & check
        time.sleep(10)
        to_remove = []
        for i, (name, proc, log_f, t0) in active.items():
            rc = proc.poll()
            if rc is not None:
                log_f.close()
                dur = time.time() - t0
                if rc == 0:
                    print(f"  [{ts_fmt(time.time()-t_start)}] [slot {i}] DONE  {name}  "
                          f"({ts_fmt(dur)})  {done+1}/{total}")
                else:
                    print(f"  [{ts_fmt(time.time()-t_start)}] [slot {i}] FAIL  {name}  "
                          f"(exit {rc}, {ts_fmt(dur)})")
                    failed.append(name)
                done += 1
                to_remove.append(i)
        for i in to_remove:
            del active[i]

    print(f"\n{'='*60}")
    print(f"All done in {ts_fmt(time.time()-t_start)}. "
          f"Success: {total-len(failed)}/{total}, Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")
    return len(failed) == 0


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" OFFLINE-TO-ONLINE RL — AUTO-SCHEDULED EXPERIMENTS")
    print("=" * 60)
    print(f"Working directory: {_HERE}")
    print(f"n_epochs: {args.n_epochs}, seeds: {SEEDS}")
    print(f"Selected: {sorted(SELECTED)}")

    print("\nDetecting resources...")
    slots, device_type = compute_slots()

    experiments = build_experiments()
    print(f"\nTotal experiments: {len(experiments)}")
    for name, _ in experiments:
        print(f"  - {name}")

    if args.dry_run:
        print("\n[DRY RUN] exiting without executing")
        return

    log_dir = os.path.join(_HERE, "experiment_output", "auto_run_logs")
    success = dispatch(slots, device_type, experiments, log_dir)

    if success and "eval" in SELECTED:
        print("\nRunning analysis...")
        subprocess.run(args.python.split() + ["analyze_results.py"],
                      cwd=_HERE)


if __name__ == "__main__":
    main()
