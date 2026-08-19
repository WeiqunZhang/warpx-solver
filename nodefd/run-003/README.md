# run-003 — `max_coarsening_level` × `agg_grid_size`, Dirichlet in x

## Two changes since run-002

Both affect comparability with earlier runs; do not compare timings across this
boundary without accounting for them.

### 1. The problem is no longer fully periodic

`is_periodic = 0 1 1` — homogeneous Dirichlet in x, periodic in y and z.

A fully periodic domain makes this operator **singular**, and because
`MLEBNodeFDLaplacian` hard-codes `isSingular()`/`isBottomSingular()` to `false`
(the deferred AMReX bug), MLMG never projects the constant mode out. run-002
showed the consequence: on the 4³ coarsest grid BiCGStab was called on to do
real work — `amrex::Dot()` 1340 times versus 100 at 8³ — and at 7 levels it
diverged outright.

Dirichlet in one direction removes the nullspace entirely. Verified locally:
**BiCGStab is now stable at every coarsening depth, including 7 levels**, where
the periodic problem blew up to resid/bnorm ≈ 1e20.

This matters for interpreting run-002: its headline finding that 5 levels beat
6 by 28–33% was very likely an artefact of that pathology, not a property of
multigrid. run-003 re-tests the question on a well-posed problem.

### 2. The AMReX `mg_min_width = 4` patch is reverted

Stock 2/2 restored (commit `0aa26b6dd5` reverted). That patch existed only to
dodge the singular 2-cell level, which change (1) eliminates. Reverting:

- lets the sweep reach **7 MG levels** instead of capping at 6, so it can
  actually test whether more coarsening is better;
- makes the results applicable to **upstream AMReX defaults** rather than to a
  locally modified build. `mg_box_min_width` also gates when agglomeration
  triggers, so with the patch in place the `agg_grid_size` axis would have been
  measured on non-stock behaviour.

## Prerequisites on Perlmutter — both mandatory

1. **Revert commit `0aa26b6dd5`** in the Perlmutter AMReX checkout. (It was
   applied there uncommitted, so the version string never reflected it —
   check the source, not `git describe`.)
2. **Rebuild the executable.**

**Validation:** the `mcl=30` runs must report `MG levels : 7`. If they say 6,
the patch is still in and every point in the sweep is shifted by one level.

## The sweep

| `max_coarsening_level` | 30 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|
| MG levels | **7** | 6 | 5 | 4 | 3 |
| iterations (local check) | 8 | 8 | 8 | 7 | 6 |

| `agg_grid_size` | -1 (default 32) | 16 | 64 |
|---|---|---|---|
| agglomeration starts at | MG level 2 | MG level 3 | MG level 1 |

Full 5 × 3 cross product at 2 and 4 GPUs; `mcl` only at 1 GPU, where
agglomeration is a no-op. Two repeats, `nsolves=20`, `nwarmup=3`.
**70 runs, ~18 minutes.**

All 15 combinations were verified locally to converge — structure and
convergence only. No timings were taken locally; this machine has one GPU and
cannot measure multi-process performance.

## Reading the results

- **The `mcl` axis is the headline.** run-002 said fewer levels is much better;
  that was measured on a singular problem where the coarse grids were
  pathological. If the ordering reverses here, the earlier conclusion was an
  artefact and the conventional expectation (more coarsening is better) holds
  for a well-posed problem.
- **The sweep is not iteration-matched.** Iteration count falls with level
  count (8,8,8,7,6), because a larger coarsest grid means the bottom solve does
  more of the work per V-cycle. Total time-to-converge is the right metric —
  but report iterations alongside, since a config that wins on time by doing
  more bottom work is a different trade than one that wins per iteration.
- **`agg_grid_size`** should be re-checked, not assumed: run-002 found
  agglomerating as early as possible (`64`) monotonically best, but that was
  with `mg_box_min_width=4` and a singular operator.
- Expect the 1-GPU column to isolate pure level-count cost, with no
  redistribution and no MPI at all.

## Standing rules

- `nosync=1`, `recreate_linop=0` — established, no longer under test.
- Ignore the `Bottom` timer and every per-function TinyProfiler number: these
  are `nosync=1` runs, where those figures are pipeline-drain artefacts. Total
  solve time only. For attribution, re-run a single config with `nosync=0`.
