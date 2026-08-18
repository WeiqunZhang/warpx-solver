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
   than one (1.25× slower with `nosync=0`). This is not a configuration error —
   128³ is far too small to occupy even one A100, so extra ranks add
   communication and shrink kernels without removing enough work.

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
  constant mode out of the bottom solve. BiCGStab and CG diverge to
  resid/bnorm ≈ 1e20 within ~13 V-cycles. The bottom solve is <1% of total, so
  this barely perturbs what is being measured. Possible upstream AMReX issue:
  `MLNodeLinOp::buildMasks` *does* correctly compute `m_is_bottom_singular=true`
  here, and the `final` override shadows it — which also makes the
  `AMREX_ASSERT(!isBottomSingular())` at `AMReX_MLEBNodeFDLaplacian.cpp:306`
  vacuous.

- **`agg_grid_size` defaults differ between CPU and GPU builds** (8 vs 32),
  producing structurally different MG hierarchies. The local CPU check above saw
  270 `ParallelCopy` calls where the GPU run had 450. Pin it explicitly before
  comparing CPU and GPU profiles.

## Next steps

1. **`n_cell=256` and `512`.** 128³ measures overhead, not the solver. 512³ puts
   ~4.3 GB/rank on 4 GPUs and would actually reach the bandwidth-bound regime —
   the test of whether `nosync` still pays and whether scaling turns positive.
2. **Agglomeration knobs at 4 GPUs.** With `ParallelCopy` at 34% and badly
   imbalanced, `agglomeration=0` or tuned `agg_grid_size`/`con_grid_size` is the
   obvious lever. Verified locally that `agglomeration=0` runs correctly.
3. **Reduce launch count per V-cycle.** At 1 GPU, ~103 `FillBoundary` launches
   per V-cycle against ~32 `Fsmooth` is the real target; fusing or eliminating
   coarse-level halo exchanges would attack the dominant cost directly.

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
