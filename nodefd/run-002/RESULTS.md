# run-002 — level-count-matched: agglomeration vs consolidation

20 runs (2 repeats × 10 configs), Perlmutter, 1 node, 128³.
Fixed: `nosync=1`, `recreate_linop=0`, `bottom_solver=bicgstab`, `nsolves=20`.

## Validity

All pass. MG level counts correct (6 for `agg6`/`mcl6`, 5 for the rest), device
counts match rank counts, **9 iterations in all 20 runs**, and repeat spread is
0.1–1.1% — small against the effects being measured.

Critically, the matched pair held in the actual runs: `agg5` and `con5` report
**identical ranks per level** (4,4,1,1,1 at 4 GPUs; 2,2,1,1,1 at 2 GPUs).

## Headline: run-001's conclusion is overturned

**At matched level count, agglomeration beats consolidation.**

| | `agg5` | `con5` | winner |
|---|---|---|---|
| 2 GPUs | **17.64** | 18.12 | agglomeration +2.6% |
| 4 GPUs | **20.27** | 22.34 | agglomeration +9.3% |

run-001 concluded that no-agglomeration + consolidation beat agglomeration by
21–31%. That was **entirely the MG level count**, not the redistribution
mechanism. With levels controlled, the mechanism favours agglomeration — the
opposite sign.

The run-001 recommendation ("use `agglomeration=0` with `con_ratio=4`") is
**withdrawn**. See run-001/RESULTS.md, annotated as superseded.

## What actually matters: the level count

| | 6 levels | 5 levels | saving |
|---|---|---|---|
| 1 GPU | 14.84 | 12.89 | −13.1% (1.95 ms) |
| 2 GPUs | 26.33 | 17.64 | −33.0% (8.69 ms) |
| 4 GPUs | 28.30 | 20.27 | −28.4% (8.03 ms) |

run-001 bounded this at ~0.54 ms from a local RTX 5070 at 1 GPU. That estimate
was wrong by **4× at 1 GPU and 15× at multi-GPU** — the extrapolation from
different hardware and rank count did not hold at all.

### Why one 4³ level on one rank costs 8 ms

Profiles at 4 GPUs, totals over 20 solves:

| | `agg6` (6 lev) | `agg5` (5 lev) |
|---|---|---|
| `ParallelCopy_finish` | 1440 calls, **12.13 ms/solve** (36%) | 1080 calls, 4.21 ms/solve (17%) |
| `amrex::Dot()` | **1340 calls** (12.5%) | **100 calls** (2.0%) |
| `FabArray::norminf()` | 935 calls | 315 calls |
| `Fsmooth` | 3600 calls (2.8%) | 2880 calls (3.0%) |

The `ParallelCopy` delta (12.13 − 4.21 = 7.92 ms) accounts for essentially the
whole 8.03 ms gap. Two mechanisms combine:

1. **The bottom solver has to work on 4³ but not on 8³.** `Dot()` — BiCGStab's
   inner product, a global reduction — is called **1340 vs 100 times**, i.e.
   ~1.9 BiCGStab iterations per V-cycle versus ~0.14. The 4³ periodic grid is
   near-degenerate; the 8³ one converges essentially on entry. This is the
   singular-operator problem again: 4 cells is better than 2, still bad.
2. **Ranks 1–3 block while rank 0 grinds the extra serial level.**
   `ParallelCopy_finish` spans min 0.0088 s / max 0.2426 s across ranks — a 27×
   spread. That is waiting, not transferring.

This explains the 1-GPU vs multi-GPU asymmetry: at 1 GPU the extra level costs
1.95 ms of real serial work with nobody waiting; at 2–4 GPUs the same level
costs ~8 ms because three GPUs idle through it and the reductions become MPI
allreduces.

## Best configuration

`agg_grid_size=64 max_coarsening_level=4` — agglomerate at MG level 1, 5 levels.

| | default | best | gain |
|---|---|---|---|
| 1 GPU | 14.90 | **12.89** | **13.5%** |
| 2 GPUs | 26.28 | **15.40** | **41.4%** |
| 4 GPUs | 28.35 | **16.49** | **41.8%** |

(1-GPU "best" is `mcl=4`; agglomeration knobs are no-ops at one rank.)

Its profile shows `Dot()` gone from the top entries entirely — the bottom solve
is effectively free — and `FillBoundary` 35% cheaper than `agg5` because level 1
runs on one rank instead of four.

Full ranking at 4 GPUs: `agg64x5` 16.52 < `agg5` 20.27 < `con5` 22.34 <
`agg6` 28.30 (default).

## Scaling

Still negative, but much improved: best 4-GPU (16.52) vs best 1-GPU (12.89) =
**1.28× slower**, down from 1.52× in run-001. Two GPUs (15.45) are 1.20× slower
than one. Extra GPUs still buy capacity, not speed, at 128³.

## Note on `max_coarsening_level`

run-001 deliberately excluded `mcl` from the sweep on solver-stability grounds —
more coarsening levels generally being preferable. That caution is reasonable as
a default policy, but for this problem the data argues the other way:

- Truncating to 5 levels is worth 13–33%.
- Iteration count stayed at **9 in every one of the 20 runs**, across all rank
  counts and both repeats. No convergence degradation was observed at all.
- The mechanism is understood — it removes a near-degenerate coarsest grid that
  the bottom solver struggles on — so the gain is not accidental.

The floor still applies: at 4 levels iterations rose to 12 and time regressed
(run-000). Five levels is the optimum here, not a general recommendation.

## Recommendations

1. **`agg_grid_size=64` + `max_coarsening_level=4`** — 41% faster than AMReX
   defaults at 2 and 4 GPUs, 13.5% at 1 GPU, iteration count unchanged.
2. **Keep agglomeration on.** run-001's advice to disable it was an artefact of
   uncontrolled level count.
3. Do not extrapolate timing ratios across hardware or rank count — the 0.54 ms
   bound from a local GPU was off by up to 15×.

## Next

- The dominant remaining cost is `FillBoundary` (27% + 20% in the best config)
  with **zero real communication at 1 GPU**. That is the structural item from
  run-000 and is still untouched: ~103 halo exchanges per V-cycle against ~32
  smooths. It needs an AMReX-side change, not a knob.
- The unexplained run-000 vs run-001 default regression is now moot — run-002's
  best beats every earlier configuration at every rank count.
- Worth checking whether the bottom-solver cost on coarse periodic grids can be
  removed outright by fixing `MLEBNodeFDLaplacian::isSingular()` (the deferred
  AMReX bug), which would let `makeSolvable` project out the constant mode.
