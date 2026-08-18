# run-001 — agglomeration sweep at 128³

19 runs, Perlmutter, 1 node. Fixed: `nosync=1`, `recreate_linop=0`,
`bottom_solver=bicgstab`, `max_coarsening_level` default.

## Validity

All checks pass: `MG levels : 6` (5 for the no-agg arm) confirms the AMReX
min-width patch is active; device counts 1/2/4 match rank counts; **9 iterations
in every one of the 19 runs**, so all comparisons are like-for-like.

Note the AMReX version string still reads `26.08-22-g189d80c7d704`, identical to
run-000 — the patch is applied *uncommitted* on Perlmutter. The MG level count,
not the version string, is the provenance signal.

## ⚠️ The `Bottom` timer is meaningless under `nosync=1`

`MLMG: Timers: ... Bottom = X` reports 6–8 ms in these logs. That is **not** the
bottom-solve cost. Measured on one GPU with `nosync=0`, where timers are
meaningful:

| | Solve (ms) | Bottom (ms) |
|---|---|---|
| `nosync=0`, bicgstab | 31.53 | **1.32** |
| `nosync=0`, smoother | 32.57 | **2.39** |
| `nosync=1`, bicgstab | 24.67 | 10.08 ← artifact |
| `nosync=1`, smoother | 24.69 | 0.65 |

BiCGStab's dot products force a device→host sync, so the bottom timer absorbs
all work queued by the preceding V-cycle levels — the same pipeline-drain effect
`norminf` showed in run-000's profile. The proof is that total solve time is
identical (24.67 vs 24.69) while the two "Bottom" figures differ 15×.

BiCGStab converges in **1–2 iterations** on the coarsest grid, and is genuinely
*cheaper* than `smoother` (1.32 vs 2.39 ms). It is not a problem here.

**Consequence:** any `Iter − Bottom` decomposition of these logs is invalid.
Only total solve time is trustworthy under `nosync=1`.

## Results (total solve, ms)

| config | hierarchy | 1 GPU | 2 GPU | 4 GPU |
|---|---|---|---|---|
| **Arm A — agglomeration on** | | | | |
| `agg_grid_size=8` | agg @ L4/6 | | 31.76 | 37.46 |
| `agg_grid_size=16` | agg @ L3/6 | | 28.93 | 32.33 |
| default (`=32`) | agg @ L2/6 | 14.82 | 26.26 | 28.37 |
| `agg_grid_size=64` | agg @ L1/6 | | 23.77 | 24.30 |
| **Arm B — agglomeration off** | | | | |
| `consolidation=0` | none, 5 lev | | 23.45 | 29.60 |
| `con_strategy=1` | con @ L4/5 | | 18.29 | 26.33 |
| `con_strategy=2` | con @ L4/5 | | 18.19 | 26.39 |
| `con_strategy=3` (default) | con @ L4/5 | | **18.07** | 26.28 |
| `con_ratio=4` | con @ L4/5 | | 18.19 | **22.54** |

### Arm A: agglomerate as early as possible

Strictly monotone at both rank counts — no interior optimum. `agg_grid_size=64`
(agglomerate at MG level 1) beats the default by **9.5%** at 2 GPUs and **14.3%**
at 4 GPUs.

This is the opposite of the run-000 hypothesis, which expected delaying
agglomeration to pay by avoiding `ParallelCopy`. The mechanism is the reverse:
once agglomerated, every deeper level is *local* and does no MPI at all.
Delaying it keeps tiny boxes spread across ranks, so each level pays cross-rank
`FillBoundary` on minuscule messages. At these sizes the redistribution is cheap
and the distributed halo exchange is not — latency, not bandwidth.

### Arm B: turning agglomeration off is better still

Best at both rank counts, and it beats the AMReX default configuration by
**31%** (2 GPU) and **21%** (4 GPU).

- `con_strategy` is irrelevant — 1/2/3 spread under 1%. Which ranks own the
  coarse boxes does not matter at this scale.
- `con_ratio=4` matters at 4 ranks (22.54 vs 26.28, **−14%**): it consolidates
  4→1 in a single step instead of 4→2→1. No benefit at 2 ranks, where ratio 2
  already reaches one rank in one step.
- Consolidation is clearly worth having: `consolidation=0` is the worst
  non-pathological config in the arm (23.45 / 29.60).

**Confound:** the no-agg arm also has 5 MG levels rather than 6 (coarsest 8³ vs
4³), so "agglomeration off" and "one fewer level" cannot be separated from this
data alone.

## Scaling: still negative

Best 4-GPU config (22.54 ms) versus 1 GPU (14.82 ms) = **1.52× slower**.
Best 2-GPU (18.07) = 1.22× slower. Tuning recovered a large fraction of the
overhead but did not change the qualitative answer at 128³: extra GPUs buy
capacity, not speed.

## Unresolved: run-001 default vs run-000

| | 1 GPU | 2 GPU | 4 GPU |
|---|---|---|---|
| run-000 (7 lev, smoother) | 14.84 | 20.51 | 22.60 |
| run-001 default (6 lev, bicgstab) | 14.81 | 26.26 | 28.37 |
| run-001 **best** | 14.81 | **18.07** | **22.54** |

The *default* regressed ~25% at 2 and 4 GPUs while 1 GPU was unchanged. The
bottom solver is ruled out (identical totals at 1 GPU, see above), which points
at the min-width patch — but that patch *helped* at 1 GPU, so the story is not
clean. Needs a controlled A/B with the patch reverted.

The *tuned* configs did not regress: run-001's best beats run-000 at 2 GPUs and
matches it at 4.

## Recommendations

1. **Use `agglomeration=0` with `con_ratio=4`** at 4 GPUs, `con_ratio=2` at 2.
   This is the best measured configuration and is 21–31% better than AMReX
   defaults at this problem size.
2. If agglomeration must stay on, **set `agg_grid_size=64`** — never leave it at
   the default 32, which agglomerates too late.
3. **Never read the `Bottom` timer, or any per-function TinyProfiler number,
   from a `nosync=1` run.** Re-run with `nosync=0` for attribution.

## Next

- Controlled A/B on the min-width patch to close the run-000 gap.
- Separate the two confounded variables in Arm B: run `agglomeration=0` with
  `max_coarsening_level` forced to give 6 levels, to isolate "no agglomeration"
  from "one fewer level".
- The 1-GPU number (14.82 ms) has not moved across either run. At 128³ that is
  the floor being chased, and it is set by launch/halo overhead — the structural
  item from run-000 (~103 `FillBoundary` launches per V-cycle against ~32
  `Fsmooth`, with zero real communication) remains untouched.
