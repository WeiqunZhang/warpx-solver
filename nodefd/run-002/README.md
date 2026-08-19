# run-002 — level-count-matched: agglomeration vs consolidation

## The question

run-001 found no-agglomeration + consolidation beats agglomeration by 21% at
4 GPUs and 31% at 2. But the two configurations differed in **two** ways:

1. the redistribution mechanism, and
2. the MG level count (5 vs 6).

A rank-matched comparison in run-001 bounded the level term at ~0.5 ms against
gaps of 5.8-8.2 ms, suggesting the mechanism dominates — but that bound came
from a local RTX 5070 at 1 GPU, not an A100 at 2/4 GPUs. This run removes the
confound outright.

## The matched pair

Setting `max_coarsening_level=4` gives agglomeration the same 5-level hierarchy
that no-agglomeration produces naturally. Verified at 4 ranks:

| lev | domain | **B**: agg, `mcl=4` | **C**: noagg, `con_ratio=4` |
|---|---|---|---|
| 0 | 128³ | 4 boxes / 4 ranks | 4 boxes / 4 ranks |
| 1 | 64³ | 4 boxes / 4 ranks | 4 boxes / 4 ranks |
| 2 | 32³ | **1 box** / 1 rank (rank 0) | **4 boxes** / 1 rank (rank 0) |
| 3 | 16³ | **1 box** / 1 rank | **4 boxes** / 1 rank |
| 4 | 8³ | **1 box** / 1 rank | **4 boxes** / 1 rank |

Identical level count, domain per level, ranks per level, and collapse point.
The sole difference is the number of boxes on the owning rank — i.e. exactly the
redistribution mechanism:

- **Agglomeration** rebuilds the BoxArray into one box, so its `ParallelCopy` is
  a general box-intersection gather.
- **Consolidation** preserves box shapes and changes only ownership — a straight
  point-to-point move.

Both converge in 9 iterations.

## Configurations

| tag | config | levels | purpose |
|---|---|---|---|
| `agg6` | defaults | 6 | run-001 baseline, reproducibility check |
| `agg5` | `max_coarsening_level=4` | 5 | **matched to `con5`** |
| `con5` | `agglomeration=0 con_ratio=4` | 5 | **matched to `agg5`** |
| `agg64x5` | `agg_grid_size=64 max_coarsening_level=4` | 5 | collapse-early, at matched levels |

Run at 2 and 4 GPUs. At 1 GPU there is no redistribution at all, so only the
level count is varied (`mcl6` vs `mcl5`) — that isolates the extra-level cost on
A100 hardware, replacing the RTX 5070 bound.

**Two repeats of everything**, `nsolves=20`, `nwarmup=3`. The effects being
separated are ~0.5 ms against ~25 ms totals, so run-to-run spread has to be
measured rather than assumed.

20 runs, ~6 minutes wall.

## Reading the results

- **`agg5` vs `con5` is the headline.** Any difference is now attributable to
  the redistribution mechanism alone. If the ~20-31% gap survives at matched
  levels, agglomeration's box-intersection gather is the cause and the AMReX
  default is simply the wrong choice at this problem size. If the gap collapses,
  run-001's result was mostly the extra level and the recommendation changes.
- **`agg6` vs `agg5`** gives the level cost at 2/4 GPUs directly.
- **`g1-mcl6` vs `g1-mcl5`** gives it with zero redistribution — the cleanest
  measurement of the extra level.
- **`agg64x5` vs `agg5`** tests whether "collapse to one rank as early as
  possible" still holds once level count is controlled.

## Validity checks before trusting anything

- `MG levels : 6` for `agg6`; `MG levels : 5` for `agg5`, `con5`, `agg64x5`.
  Wrong counts mean the min-width patch is missing or `mcl` did not bind.
- **9 iterations in every run.** Any deviation invalidates the comparison.
- Device count matches rank count (1/2/4).
- The `MG hierarchy` block in each log should match the table above.
- Ignore the `Bottom` timer and all per-function TinyProfiler numbers — these
  are `nosync=1` runs, where those figures are pipeline-drain artifacts. Total
  solve time only.
