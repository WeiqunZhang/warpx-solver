# run-001 — agglomeration sweep at 128³

Follow-up to run-000, which found that at 4 GPUs ~82% of solve time is
communication, with `FabArray::ParallelCopy` alone at 34% and badly imbalanced
across ranks (2.9 ms min vs 73.1 ms max). That `ParallelCopy` is the
agglomeration redistribution. This run sweeps it.

## Prerequisites

1. **The AMReX min-width patch must be present** — commit `mg min width = 4`
   (`0aa26b6dd5`), which sets `mg_domain_min_width = mg_box_min_width = 4` under
   `#ifdef AMREX_USE_GPU`. Confirm the Perlmutter AMReX checkout has it.
2. **Rebuild the executable after that patch** — it changes the MG hierarchy.
3. Sanity check: every `.ou` should report `MG levels : 6` (5 for the
   `agglomeration=0` arm). If you see 7, the patch is missing and the
   `bicgstab` bottom solver will diverge.

## Fixed settings

`nosync=1`, `recreate_linop=0` — standing defaults from here on; both are
established results from run-000 and are no longer under test.

`max_coarsening_level` is **not** swept. More coarsening levels are preferred
for solver stability, and the 3.3% it offered at 1 GPU is not worth the risk.

`linop_verbose=2` is on for every run, so each log records
`agglomerated AMR level 0 starting at MG level N of M`. Without it the results
are uninterpretable.

## What is actually being varied

Consolidation is **inert while agglomeration is on**: a covered domain takes the
agglomeration branch in `MLLinOpT::defineGrids`, so `con_*` never runs. It only
becomes the active mechanism when `agglomeration=0`. Hence two arms.

### Arm A — agglomeration on, vary where it starts

`agg_grid_size` controls the start level (measured, 4 ranks, 128³):

| `agg_grid_size` | starts at | levels agglomerated | file |
|---|---|---|---|
| 8 | MG level 4 of 6 | 2 | `run-gN-agg8.ou` |
| 16 | MG level 3 of 6 | 3 | `run-gN-agg16.ou` |
| -1 → 32 (default) | MG level 2 of 6 | 4 | `run-gN-base.ou` |
| 64 | MG level 1 of 6 | 5 | `run-gN-agg64.ou` |

The trade-off: agglomerating later keeps more levels distributed (more
cross-rank `FillBoundary`, tiny boxes per rank) but does fewer `ParallelCopy`
redistributions. 64 and 128 both land on level 1, so 128 is omitted.

### Arm B — agglomeration off, consolidation active

| config | file |
|---|---|
| `agglomeration=0 consolidation=0` — no redistribution at all | `run-gN-noagg-nocon.ou` |
| `agglomeration=0 con_strategy=1` — group by rank index | `run-gN-noagg-cs1.ou` |
| `agglomeration=0 con_strategy=2` — modulo | `run-gN-noagg-cs2.ou` |
| `agglomeration=0 con_strategy=3` — SFC reorder (default) | `run-gN-noagg-cs3.ou` |
| `agglomeration=0 con_ratio=4` — halve the consolidation steps | `run-gN-noagg-cr4.ou` |

`con_strategy` changes *which ranks* own coarse boxes, i.e. the communication
pattern, without changing the start level — all three report
`consolidated starting at MG level 4 of 5`. The differences will show up in
`ParallelCopy` cost and in its min/max spread across ranks, not in the log
header.

## Layout

19 runs: 1 GPU baseline only (no redistribution exists at 1 rank), then 9
configs each at 2 and 4 GPUs. ~7 minutes wall, dominated by `srun` startup.

All configs were dry-run locally (4 ranks on 1 GPU) before submission: all
converge in 9 iterations, none abort.

## What to look for

- `ParallelCopy_finish` share and its min/max spread in the `solve` region
  table, versus the 34% / 25× baseline from run-000.
- Whether Arm A has an interior optimum, or whether the extremes win.
- Whether any config makes 4 GPUs faster than 1 — run-000 had 4 GPUs 1.52×
  *slower* than 1 at this size. That is the bar.
- Iteration count must stay at 9 everywhere; if it moves, the comparison is
  invalid.
