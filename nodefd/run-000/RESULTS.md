# run-000 — MLMG + MLEBNodeFDLaplacian on Perlmutter

First data set from the `nodefd` benchmark. Purpose: measure `MLMG::setNoGpuSync`
and the cost of WarpX's per-call linop reconstruction, on real hardware.

## Configuration

| | |
|---|---|
| Machine | Perlmutter, 1 node, A100 40GB (`gpu&hbm40g`) |
| AMReX | `26.08-22-g189d80c7d704` |
| Problem | 128³, fully periodic, no EB, single level |
| Decomposition | `amrex::decompose(domain, NProcs())` |
| MG levels | 7 (all rank counts) |
| Bottom solver | `smoother` (see "Caveats") |
| Tolerance | `reltol=1e-11`, `abstol=0` |
| Repeats | 1 warmup + 5 timed solves |

Boxes per rank count: 1 GPU → `(128,128,128)`; 2 GPU → `(128,128,64)`;
4 GPU → `(128,64,64)`. All extents powers of 2, all coarsenable — no MG collapse.

**Validity checks that passed.** All 12 runs printed "Fixing GPU assignment for
Perlmutter according to heuristics", and `CUDA initialized with N devices`
reported 1/2/4 matching the rank count — so every rank got a distinct,
NUMA-affine GPU. Iteration count was **9 in every run**, and the final residual
was bit-identical between `nosync=0` and `nosync=1` at each rank count
(6.677e-13 / 6.918e-13 / 7.100e-13 for 1/2/4 GPUs; it varies with rank count
only because the decomposition changes the reduction order).

## Headline numbers

Solve time in ms, `recreate_linop=0`:

| GPUs | nosync=0 | nosync=1 | nosync gain | scaling vs 1 GPU (nosync=1) |
|---|---|---|---|---|
| 1 | 26.77 | 14.85 | **1.80×** | — |
| 2 | 30.62 | 20.53 | 1.49× | **0.72× (slower)** |
| 4 | 33.46 | 22.63 | 1.48× | **0.66× (slower)** |

Two findings:

1. **`nosync` is a large, free win** — up to 1.80× on one GPU, ~1.48× on four,
   with provably identical results. Notably larger than the 1.26× measured
   locally on an RTX 5070 at the same size.

2. **Strong scaling is negative.** Four A100s solve this problem **1.52× slower**
   than one (1.25× slower with `nosync=0`). 128³ does not come close to
   occupying even one A100, so extra ranks add communication and shrink kernels
   without removing enough work.

   **128³ is a common production size for WarpX users, so this is a user-facing
   result, not a benchmark artifact.** At this size the honest advice is that
   extra GPUs buy capacity, not speed — and the optimization target is launch
   and communication overhead at fixed size, not kernel throughput. This
   investigation is scoped to 128³ for that reason.

## Where the time goes

### 1 GPU, `nosync=0` (per-function timings are meaningful here)

| Function | NCalls | Excl. time (s) | % |
|---|---|---|---|
| `FillBoundary_nowait()` | 4645 | 0.0476 | **29.1** |
| `MLEBNodeFDLaplacian::Fsmooth()` | 1440 | 0.0434 | 26.5 |
| `MLEBNodeFDLaplacian::Fapply()` | 320 | 0.0063 | 3.9 |
| `MLEBNodeFDLaplacian::interpolation()` | 270 | 0.0059 | 3.6 |
| `FabArray::setVal()` | 455 | 0.0050 | 3.1 |
| `MLNodeLinOp::applyBC()` | 3155 | 0.0049 | 3.0 |

On a single GPU there is **no neighbor exchange at all**, so that 29% is pure
halo bookkeeping and kernel-launch overhead. Per V-cycle (45 V-cycles = 5 solves
× 9 iterations) that is ~103 `FillBoundary` and ~32 `Fsmooth` launches, most of
them on tiny coarse levels. Classic latency-bound multigrid — which is exactly
why `nosync` pays so well here.

### 4 GPUs, `nosync=0`

| Function | NCalls | min / avg / max (s) | % |
|---|---|---|---|
| `FabArray::ParallelCopy_finish()` | 450 | 0.0029 / 0.0554 / 0.0731 | **34.1** |
| `FillBoundary_nowait()` | 3868 | 0.0371 / 0.0442 / 0.0656 | 30.5 |
| `FillBoundary_finish()` | 3868 | 0.0269 / 0.0283 / 0.0306 | 14.3 |
| `MLEBNodeFDLaplacian::Fsmooth()` | 1170 | 0.0101 / 0.0150 / 0.0293 | **13.7** |
| `FabArray::ParallelCopy_nowait()` | 450 | 0.0008 / 0.0022 / 0.0062 | 2.9 |

**~82% communication, ~14% actual smoothing.** That is the whole explanation for
the negative scaling.

`ParallelCopy` is the **agglomeration/consolidation redistribution** — confirmed
by a local 4-rank CPU test where `agglomeration=0 consolidation=0` removed it
from the profile entirely (MG levels 7→6, still 9 iterations). Its 25× min/max
spread across ranks (2.9 ms vs 73.1 ms) is the load imbalance from coarse levels
collapsing onto a subset of ranks while the others wait.

## Cost of rebuilding the linop each solve

`recreate_linop=1` reproduces what WarpX does — a fresh `MLEBNodeFDLaplacian`
**and** a fresh `MLMG` on every `computePhi` call. Extra total time (build +
solve) versus reusing both:

| | 1 GPU | 2 GPU | 4 GPU |
|---|---|---|---|
| nosync=1 | +0.80 ms (5.4%) | +1.27 ms (6.2%) | +1.35 ms (6.0%) |
| nosync=0 | +0.85 ms (3.2%) | +1.31 ms (4.3%) | +1.09 ms (3.3%) |

Only ~0.31 ms of that is the `define()` itself. **The rest lands inside
`solve()`**, because a fresh `MLMG` re-runs the full `prepareForSolve`
allocation path every call instead of amortizing it across solves.

→ Caching the linop + MLMG across steps in WarpX is worth roughly 5%, and most
of that win comes from reusing **MLMG**, not the linop.

## Caveats

- **`nosync=1` per-function profiles are not attributable.** With `nosync=1` and
  `tiny_profiler.device_synchronize_around_region=0`, `Fsmooth` appears to drop
  43.4 → 6.7 ms while `norminf` balloons to 47.0 ms (51%). That is an artifact:
  launches return immediately, and `norminf`'s device→host reduction is the
  first forced sync, so it absorbs all the drained work. **Only the wall-clock
  per-solve times are trustworthy under `nosync=1`** — those are what the tables
  above use. Do not turn `device_synchronize_around_region=1` on to "fix" this:
  it reinstates exactly the syncs `nosync` removes and makes the flag look
  worthless.

- **`bottom_solver=smoother`, not the AMReX/WarpX default (`bicgstab`).** This
  all-periodic operator is singular, but `MLEBNodeFDLaplacian` hard-codes
  `isSingular()`/`isBottomSingular()` to `false`, so MLMG never projects the
  constant mode out. BiCGStab and CG diverge to resid/bnorm ≈ 1e20 within ~13
  V-cycles; `smoother` is stable. Possible upstream AMReX issue:
  `MLNodeLinOp::buildMasks` *does* correctly compute `m_is_bottom_singular=true`
  here, and the `final` override shadows it — which also makes the
  `AMREX_ASSERT(!isBottomSingular())` at `AMReX_MLEBNodeFDLaplacian.cpp:306`
  vacuous.

  **Mechanism (corrected — an earlier note here blamed the Krylov bottom solve
  breaking down; that is wrong).** With `bottom_verbose=1`, BiCGStab is seen to
  *succeed*, meeting its 1e-4 tolerance, for the first eight V-cycles. What
  diverges is its input: the bottom-level initial error grows geometrically
  across V-cycles (8.55e-19 → 3.58e-14 → 1.91e-10 → 6.91e-06 → 1.11e-04 → 9.34 →
  … → 3.0e+15). `MLMG: Bottom solve failed.` (`AMReX_MLMG.H:2066`) appears only
  at V-cycle 9, long after the run is unrecoverable — it is a **symptom, not the
  trigger**.

  The trigger is the **7th MG level specifically**: 128³ coarsens
  128→64→32→16→8→4→**2**, and on a 2-cell periodic nodal grid the FD Laplacian is
  almost entirely nullspace, so the Krylov solve returns an arbitrarily scaled
  constant that is then amplified by each successive V-cycle. Measured
  threshold, `bottom_solver=bicgstab`, 4 ranks:

  | `max_coarsening_level` | MG levels | result |
  |---|---|---|
  | 0–5 | 1–6 | converges, 8–9 iterations |
  | 6, 30 | 7 | **diverges** |

  **Resolved for future runs** by the `mg_domain_min_width = 4` change to the
  local AMReX checkout (see "AMReX: deferred fix and interim local change"),
  which removes the degenerate level so the stock `bicgstab` works. The
  `smoother` workaround applies to run-000 only.

- **Did the `smoother` workaround distort the headline result?** (Local-machine
  check, indicative only — not valid performance data.)
  At `max_coarsening_level=4` on a local GPU, the nosync gain is 1.27× with
  `smoother` (30.93 → 24.26 ms) versus 1.23× with `bicgstab` (29.81 → 24.15 ms),
  both at 9 iterations. This was worth checking because BiCGStab's dot products
  are device→host reductions, i.e. exactly the syncs under study — but they are
  too small a share of the total to move the conclusion.

- **`agg_grid_size` defaults differ between CPU and GPU builds** (8 vs 32),
  producing structurally different MG hierarchies. The local CPU check above saw
  270 `ParallelCopy` calls where the GPU run had 450. Pin it explicitly before
  comparing CPU and GPU profiles.

## Follow-up experiment: truncating the MG hierarchy

> **⚠️ ALL TIMINGS IN THIS SECTION ARE FROM THE LOCAL DEV MACHINE (single RTX
> 5070) AND ARE NOT VALID PERFORMANCE DATA.** This machine has one GPU, so it
> cannot measure multi-GPU behaviour at all, and its single-GPU ratios do not
> transfer to A100. The numbers below were used to justify the
> `mg_domain_min_width = 4` AMReX change and to bound the MG level cost; the
> level-cost bound (~0.54 ms) was later measured on Perlmutter at **1.95 ms at
> 1 GPU and ~8 ms at 2-4 GPUs** — wrong by 4-15x. See `../run-002/RESULTS.md`
> for authoritative numbers. Retained only as a record of what was done.

At 128³ every MG level costs a roughly fixed number of kernel launches no matter
how small its grid is, so the deepest levels may be pure overhead. Swept
`max_coarsening_level` on a local RTX 5070, 1 GPU, `nosync=1`,
`recreate_linop=0`, 3 repeats (run-to-run spread <0.1 ms):

| `max_coarsening_level` | MG levels | solve (ms) | iterations |
|---|---|---|---|
| 30 (default) | 7 | 25.15 | 9 |
| 5 | 6 | 24.73 | 9 |
| **4** | **5** | **24.32** | **9** |
| 3 | 4 | 31.62 | 12 |
| 2 | 3 | 120.5 | 47 |
| 1 | 2 | 495.9 | 191 |

The two deepest levels are worth **3.3%** on a single GPU — real, reproducible,
and free (iteration count unchanged at 9). Below 5 levels the coarse-grid
correction degrades and iteration count rises sharply, so 5 is the floor.

Repeating with the stock `bicgstab` bottom solver, which the truncation makes
usable again (local GPU, 1 rank; mcl=30 row still uses `smoother` because
`bicgstab` diverges at 7 levels):

| MCL | MG levels | nosync=0 (ms) | nosync=1 (ms) | nosync gain |
|---|---|---|---|---|
| **4** | 5 | **29.75** | **24.16** | 1.23× |
| 5 | 6 | 31.52 | 24.70 | 1.28× |
| 30 | 7 | 34.19 | 25.12 | 1.36× |

Two of these three points are now handled by the `mg_domain_min_width = 4` patch
described below, which makes **6 levels** the default and restores `bicgstab`.
Relative to that new baseline, setting `max_coarsening_level=4` explicitly buys
one further level of truncation — another ~2.2% on GPU (24.70 → 24.16 ms at
`nosync=1`) at unchanged iteration count, with two coarsening steps of margin
from the degenerate level instead of one. Worth carrying as a variant in
run-001 rather than assuming it: the deep levels are also where agglomeration
lives, so at 4 GPUs its value may be much larger than the 1-GPU number suggests.

**`nosync` and hierarchy truncation are not additive.** The nosync gain falls
from 1.36× at 7 levels to 1.23× at 5, because both attack the same cost —
per-level launch and sync overhead on tiny coarse grids. With `nosync=0`,
truncating 7→5 levels is worth 13%; with `nosync=1` (what WarpX does) it is
worth only 3.8%. Expect the same non-additivity between `max_coarsening_level`
and the agglomeration knobs at 4 GPUs, since removing deep levels is exactly
what removes the agglomerated ones.

This is a modest win on its own, but it is the most promising lever at 4 GPUs:
those same deep levels are where agglomeration happens, so truncating there may
remove the 34% `ParallelCopy` entirely rather than just saving launches.
**That is the experiment to run next.** It cannot be answered locally — the
machine has one GPU, and the CPU build cannot stand in because its
`agg_grid_size` default (8 vs 32) builds a different hierarchy.

## AMReX: deferred fix and interim local change

### Deferred — `MLEBNodeFDLaplacian` is wrong for singular problems

`MLEBNodeFDLaplacian` hard-codes both `isSingular()` and `isBottomSingular()` to
`false` as `final` overrides (`AMReX_MLEBNodeFDLaplacian.H:171-172`). This
shadows `MLNodeLinOp::buildMasks`, which *does* correctly compute
`m_is_bottom_singular = true` for an all-periodic covered domain
(`AMReX_MLNodeLinOp.cpp:295-301`). Consequences:

- MLMG never calls `makeSolvable()`, so the constant mode is never projected out
  at any level or in the bottom solve.
- `AMREX_ASSERT(!isBottomSingular())` at `AMReX_MLEBNodeFDLaplacian.cpp:306` is
  vacuous — it calls the `final` override, not the base member it was evidently
  meant to check.

The proper fix is to let the base-class computation through so `makeSolvable`
runs. **Deferred** — it is a real AMReX bug but not blocking this investigation,
which works around it as described below.

### Interim — `mg_domain_min_width = 4` in the local AMReX checkout

From run-001 onward, `mg_domain_min_width` is changed from 2 to 4 in the AMReX
source used for these runs (`AMReX_MLLinOp.H:859`, with the existing EB-triggered
bump to 4 at `:1105` as precedent). Rationale and measurements are in the
follow-up experiment above; briefly:

- It removes the degenerate 2-cell coarsest level, where a periodic nodal
  stencil collapses because a node's two neighbours are the same node. That
  level is what breaks `bicgstab` — on CPU as well as GPU.
- With it, the stock AMReX/WarpX default `bicgstab` bottom solver works, so the
  `bottom_solver=smoother` workaround used in run-000 is no longer needed.
- Local-machine measurement (single GPU, **not valid performance data**)
  suggested ~1.7%/~7.8%. The real effect, measured on Perlmutter in run-002, is
  **13% at 1 GPU and 28-33% at 2-4 GPUs** — much larger. The robustness
  argument (removing a near-degenerate coarsest grid) was the sound part of the
  justification; the local timings were not.

**Provenance warning when comparing runs.** run-000 was taken with **stock**
AMReX `26.08-22-g189d80c7d704`: `mg_domain_min_width = 2`, **7 MG levels** at
128³, and `bottom_solver=smoother`. Runs from run-001 on will default to **6 MG
levels** and `bicgstab`. Do not compare timings across that boundary without
accounting for both changes — from the table above, the level count alone moves
GPU solve time by 1.7-7.8% depending on `nosync`.

## Next steps

All at fixed 128³.

1. **`max_coarsening_level` sweep at 2 and 4 GPUs** — values 30/5/4/3, crossed
   with `nosync=0/1`. Hypothesis: truncation removes the agglomerated levels and
   with them the 34% `ParallelCopy`, so the win should be much larger than the
   3.3% seen at 1 GPU. Watch iteration count, which must stay at 9.
2. **Agglomeration knobs at 4 GPUs** — `agglomeration=0`, and tuned
   `agg_grid_size`/`con_grid_size`. Verified locally that `agglomeration=0` runs
   correctly (MG 7→6, still 9 iterations). Attacks the same 34% by a different
   route; worth having both numbers to see which mechanism actually pays.
3. **Reduce launch count per V-cycle.** At 1 GPU, ~103 `FillBoundary` launches
   per V-cycle against ~32 `Fsmooth`, with *zero* real communication, is the
   floor on what tuning can achieve — getting past it means fusing or
   eliminating coarse-level halo exchanges in AMReX itself. This is the
   structural fix behind findings 1 and 2.

## Raw data

Solve/build times are the mean over the 5 timed solves, max across ranks.

| run | build (ms) | solve (ms) | iters | final residual |
|---|---|---|---|---|
| g1-ns0-re0 | 0.001 | 26.77 | 9 | 6.67688127e-13 |
| g1-ns0-re1 | 0.271 | 27.35 | 9 | 6.67688127e-13 |
| g1-ns1-re0 | 0.001 | 14.85 | 9 | 6.67688127e-13 |
| g1-ns1-re1 | 0.278 | 15.37 | 9 | 6.67688127e-13 |
| g2-ns0-re0 | 0.002 | 30.62 | 9 | 6.917799666e-13 |
| g2-ns0-re1 | 0.307 | 31.62 | 9 | 6.917799666e-13 |
| g2-ns1-re0 | 0.002 | 20.53 | 9 | 6.917799666e-13 |
| g2-ns1-re1 | 0.308 | 21.50 | 9 | 6.917799666e-13 |
| g4-ns0-re0 | 0.002 | 33.46 | 9 | 7.099876242e-13 |
| g4-ns0-re1 | 0.309 | 34.24 | 9 | 7.099876242e-13 |
| g4-ns1-re0 | 0.002 | 22.63 | 9 | 7.099876242e-13 |
| g4-ns1-re1 | 0.321 | 23.66 | 9 | 7.099876242e-13 |

`ns0`/`ns1` = `nosync=0`/`1`; `re0`/`re1` = `recreate_linop=0`/`1`.
