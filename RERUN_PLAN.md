# Re-run Plan — sampler `max_traj_events` truncation fix

## Bug summary

**Root cause** (confirmed via raw libsumo probe + bridge probe + H2O+ source comparison):

`utils/bus_sampler.py` `BusEvalSampler.sample` and `BusStepSampler.sample` capped each
episode at a fixed number of decision events (`max_traj_events=200` for eval,
`100` for online training rollouts). H2O+'s `bus_h2o/train_sim.py:449` and
`collect_data_sumo.py:187` (the source these were ported from) use `while not
done:` — i.e., run until the simulator's natural termination
(`getMinExpectedNumber() <= 0` or `steps >= max_steps=18000`). The truncation
was introduced in the offline-sumo port and was not present upstream.

**Effect on episode length** (1 NoHold episode, measured):

| metric                             | old (max_traj=200)       | patched (uncapped, until env.done) |
|-----------------------------------:|:-------------------------|:-----------------------------------|
| sim_t per episode                  | ~2,000 s (≈ 33 min)      | ~18,000 s (= 5 h)                  |
| decision events per episode        | ≤ 200                    | ~5,000 (NoHold; varies by method)  |
| SUMO `<person>` events per episode | ~108                     | ~6,300 (4,599 completed + 1,732 pending) |
| wallclock per episode              | ~3-8 s (14-way parallel) | **493 s single-thread**            |
| mean_return per episode (NoHold)   | -11,951                  | -1,684,876                         |

The wallclock blow-up is dominated by the per-step cost of simulating all
~15K SUMO `<person>` entities (vs ~108 in the truncated regime), not just the
9× sim_t factor.

## Confirmed: data integrity — what does NOT need to be regenerated

- **Offline dataset `merged_all_v2.h5` (3.1M transitions)**:
  - Collected by H2O+'s `bus_h2o/collect_data_sumo.py` which does NOT use
    the truncated sampler. Inspected: `predicted boarding queue` feature
    has range [0, 31] with 27% nonzero, headway features in normal range,
    rewards span [-2142, 0]. Dataset is faithful.
  - **No need to recollect.**

- **Trained weights of pure-offline methods** (training is on HDF5 batches,
  does not call SUMO):
  - BC, CQL (×3 alpha), H2O+ offline, RE-SAC offline, all 6 RE-SAC ablations
    (no L1, no LCB, no reg, twinQ only, twinQ+LCB, noBC).
  - **No need to retrain.**

## Re-evaluation plan (Phase 1 — start now)

Reasoning: every *Best* / *Final* number in the paper currently comes from
200-event truncated episodes. With the fix, eval episodes run to sim_t=18000
on the full passenger volume; numbers will change in absolute scale, but the
inter-method ranking should be preserved (all methods evaluated under the same
new protocol).

| group | cells | ckpts × 10 ep | est wallclock (14-worker, cpu-only) |
|---|---|---:|---:|
| Offline (BC, CQL ×3, H2O+) | 5 | 150 ckpts × 10 = 1,500 ep | ~14.6 hr |
| RE-SAC main + ablations    | 7 | 270 ckpts × 10 = 2,700 ep | ~26.4 hr |
| Online SAC                 | 1 | 40 × 10 = 400 ep          | ~3.9 hr  |
| WSRL twin-Q                | 1 | 30 × 10 = 300 ep          | ~2.9 hr  |
| RLPD twin-Q (×3 ratios)    | 3 | 66 × 10 = 660 ep          | ~6.4 hr  |
| WSRL-E10 / RLPD-E10        | 2 | 36 × 10 = 360 ep          | ~3.5 hr  |
| RE-SAC noBC                | 1 | 30 × 10 = 300 ep          | ~2.9 hr  |
| **Total**                  | **20** | **5,720 ep**          | **~56 hr (single-batch 14-way)** |

This Phase-1 work can run concurrently with the main GPU cluster (it's
SUMO/CPU-only, `--vram 0`); the bottleneck is CPU cores per node.

Per-method submission policy (one task per checkpoint):
- `--cwd /home/erzhu419/offline-sumo`
- `--vram 0` (SUMO CPU-only)
- `--cpu 1 --ram-mb 2500` (each SUMO worker is single-threaded; small RAM)
- `--signature offline-sumo/eval-v2/<method>` (history accumulates per cell)

## Re-training plan (Phase 2 — after Phase 1 results in)

Online and offline-to-online methods used `BusStepSampler` with
`max_traj_events=100`, so each "rollout" was sim_t ≈ 1,000 s (≈ 17 min) of
the 5-hour episode. After the fix, rollouts run to env.done (5 h) per call,
so the same wall-clock training budget gathers different transitions and
the policies will need re-training to match.

| method | seeds × epochs | est per-seed wallclock |
|---|---|---:|
| Online SAC                  | 5 × 600 ep | (need to remeasure; old 4 hr/seed) |
| WSRL                        | 5 × 300 ep | "                                  |
| WSRL-E10                    | 3 × 300 ep | "                                  |
| RLPD ($\rho \in \{0.25,0.50,0.75\}$) | (5+3+3) × 300 ep | " |
| RLPD-E10 ($\rho{=}0.5$)     | 3 × 300 ep | "                                  |
| **Total**                   | **27 × 300-600 epoch runs** | unknown — measure after first complete |

Each online epoch involves one rollout + N gradient updates. Old rollout
≈ 17 min sim → wallclock < 1 min (libsumo fast). New rollout = 5 h sim →
needs measurement. Could be ≥ 5× slower per rollout. Total training cost
likely ≥ 5× current; rough order: **~20-50 GPU-days**.

## Schedule strategy (per scheduler skill)

**Phase 1 (re-eval) — fire now:**
- Submit all 572 eval tasks under signature `offline-sumo/eval-v2/<method>`
- `--vram 0`, `--cpu 1`, `--ram-mb 2500`
- Dispatch once; let the watcher pace as CPU frees on each node
- Watcher every 60s will pick up stragglers
- `wait-for --signature 'offline-sumo/eval-v2/*'` in `Bash run_in_background`
  → `task-notification` fires when all done

**Phase 2 (re-train) — after Phase 1 dashboard built:**
- Wait for GPU saturation to ease (currently 5/5 GPUs at 100% util)
- Submit by method; one signature per (method, seed)
- Use small step count first (e.g. 50 epochs) to remeasure wallclock per
  rollout under the new sim_t=18000 regime; extrapolate full cost.
- Decide on parallelism per node based on remeasurement.

## Paper update plan (after re-runs)

1. Replace every Best / Final number in Tables II–IV with re-evaluated values.
2. Hierarchical bootstrap intervals re-derived from new per-episode returns.
3. Operational table re-computed on full sim_t=18000 episodes.
4. §III protocol text: change "each episode = 18000 sim steps" to actually
   match the run (now actually 18000 sim steps), and remove any reference
   to "200 decision events" if present.
5. Append to revision notes: explanation of the bug + that ranking was
   already internally consistent under truncation.

## Files modified for the fix

- `utils/bus_sampler.py` (BusEvalSampler.sample, BusStepSampler.sample): replaced
  hard `for ev_idx in range(max_traj_events)` with `while not env.done`
  (with safety cap = 50× legacy default).
- `env/sumo_env/rl_bridge.py` (passenger trip-time tracking added in
  round-4 work; will be useful again once the eval fix is verified).
- `env/envs/sumo_gym_env.py` (`get_passenger_stats()` accessor added).

