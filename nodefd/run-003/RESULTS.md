# run-003 — `max_coarsening_level` × `agg_grid_size`, Dirichlet in x

70 runs (2 repeats × 35 configs), Perlmutter, 1 node, 128³, `is_periodic = 0 1 1`.
Fixed: `nosync=1`, `recreate_linop=0`, `bottom_solver=bicgstab`, `nsolves=20`, `nwarmup=3`.

## Validity

All 70 runs completed and converged. Both prerequisites were met.

- **`mcl=30` reports `MG levels : 7`** — the `mg_min_width=4` patch was reverted
  and the executable rebuilt. The sweep is on stock AMReX (2/2).
- `periodicity : 0 1 1` in all 70 runs; ranks == devices == requested GPUs in all.
- Levels track `mcl` exactly as predicted: `mcl` 30/5/4/3/2 → 7/6/5/4/3 levels.
- Iterations match the local prediction exactly: **8, 8, 8, 7, 6** for 7/6/5/4/3
  levels, identical at 1, 2 and 4 GPUs, both repeats, and constant across all 23
  solves within every run.
- **BiCGStab is stable everywhere, including 7 levels.** Worst final residual over
  all 70 runs is `8.8e-12`. run-002's periodic 7-level case diverged to ~1e20.
- Repeat spread: median 0.25%, max 1.62%. Negligible against the effects below.

One degenerate cell to note: at `mcl=2` (3 levels) with `ags=16`, agglomeration
**never triggers** — ranks/level is 2,2,2 and 4,4,4, with no `defineGrids()`
redistribution line. That cell is a "no redistribution at all" baseline, not an
`ags=16` measurement. It is also the slowest 3-level config, consistent with the
rest of the `ags` axis.

## Headline: the run-002 ordering survives, its magnitude does not

The README asked whether "fewer levels is better" would reverse on a well-posed
problem. **It does not reverse.** Time falls monotonically from 7 → 4 levels at
every rank count and every `agg_grid_size` — 7 of 7 monotone sequences, no
exceptions. But the effect is much smaller than run-002 measured, and the
optimum moved one level shallower.

| 6 → 5 levels, default `agg_grid_size` | run-002 (periodic) | run-003 (Dirichlet) |
|---|---|---|
| 1 GPU | −13.1% | −7.1% |
| 2 GPUs | −33.0% | −6.8% |
| 4 GPUs | −28.4% | −5.5% |

(Matched to run-002's configuration, which used the default `agg_grid_size`. At
`ags=64` the run-003 saving is −6.9% at both 2 and 4 GPUs.)

The isolation is clean. At 4 GPUs, `agg_grid_size` default, **identical MG
hierarchy** (6 levels, 4³ coarsest, ranks/level 4,4,1,1,1,1), changing only the
boundary condition:

| | run-002 periodic | run-003 Dirichlet |
|---|---|---|
| time | 28.35 ms | **18.58 ms** |
| iterations | 9 | 8 |
| `amrex::Dot()` per solve (rank 0) | **268** | **16** |
| ms/iteration | 3.15 | 2.32 |

At 5 levels (8³ coarsest), where run-002's bottom solve was already healthy
(`Dot()` 20/solve), the per-iteration cost is **2.24 vs 2.18 ms — a 2.7%
difference**. Once the bottom solve is well-posed, the boundary condition costs
nothing.

**So run-002's 28–33% multi-GPU cliff at 6 levels was the singular-operator
pathology, essentially in full.** The residual level-count effect on a well-posed
problem is ~7% per level. run-002's headline number is an artefact; its ordering
and its recommendation direction are not.

## The level-count axis is a U, and the minimum is 4 levels

`ags=64` (best) at each rank count, ms/solve:

| levels | coarsest | `mcl` | 1 GPU | 2 GPUs | 4 GPUs | iters |
|---|---|---|---|---|---|---|
| 7 | 2³ | 30 | 14.35 | 17.00 | 17.74 | 8 |
| 6 | 4³ | 5 | 12.31 | 14.39 | 15.24 | 8 |
| 5 | 8³ | 4 | 11.43 | 13.39 | 14.20 | 8 |
| **4** | **16³** | **3** | **9.74** | **11.05** | **11.83** | **7** |
| 3 | 32³ | 2 | 10.41 | 11.47 | 12.04 | 6 |

Four levels wins at every rank count. The turn at 3 levels is real but the
optimum is flat: 4 vs 3 levels is 6.4% at 1 GPU, 3.6% at 2, and only **1.7% at
4 GPUs** — still ~7× the repeat spread, but not a wide margin.

The U-shape is explained by the bottom solve. BiCGStab inner products on the
coarsest grid, per V-cycle on the rank that owns it:

| levels | coarsest | `Dot()` / V-cycle | ≈ BiCGStab iters / V-cycle |
|---|---|---|---|
| 7 | 2³ | 8.0 | ~2.0 |
| 6 | 4³ | 2.0 | ~0.5 |
| 5 | 8³ | 2.0 | ~0.5 |
| 4 | 16³ | 3.4 | ~0.9 |
| 3 | 32³ | **17.3** | **~4.3** |

Two costs trade off. Removing a coarse level saves one level of traversal
(~4 smooths and ~11 halo exchanges per V-cycle, mostly latency); but it enlarges
the coarsest grid, and past 16³ BiCGStab work explodes — 5× more inner products
at 32³ than at 16³. That is what turns the curve at 3 levels, and it costs more
than the one saved iteration (6 vs 7) buys back.

The 2³ level at the other end is worth nothing at all: 7 levels needs the same 8
iterations as 6, and costs an extra 2.0–2.6 ms. Note the bottom solve is *also*
slightly unhappy at 2³ (8 `Dot()` per V-cycle vs 2 at 4³) — a residue of the
tiny-grid regime, though nothing like the periodic pathology.

## `agg_grid_size=64` wins everywhere, and it is the bigger lever

**`ags=64` < `ags=default(32)` < `ags=16` in all 10 (rank count × level count)
combinations — 10/10, no exceptions.** run-002's finding is confirmed on stock
AMReX with a well-posed operator.

ms/solve at 4 GPUs:

| levels | `ags=64` | `ags=def` | `ags=16` | 64 vs default |
|---|---|---|---|---|
| 7 | **17.74** | 21.01 | 24.59 | −15.6% |
| 6 | **15.24** | 18.56 | 22.29 | −17.9% |
| 5 | **14.20** | 17.53 | 21.23 | −19.0% |
| 4 | **11.83** | 14.79 | 17.95 | −20.0% |
| 3 | **12.04** | 14.52 | 18.09 | −17.1% |

At 2 GPUs the same ordering holds, worth 10–14%.

The mechanism is visible in the call counts, and it is not what the names
suggest. Across the `ags` axis at fixed level count, **every counter is
identical** — `FillBoundary_nowait` 650/solve, `Fsmooth` 192, `applyBC` 449,
`Dot()` 16 — *except* `ParallelCopy`, which goes **up** as you agglomerate
earlier:

| 4 GPUs, 7 levels | `ags=64` | `ags=def` | `ags=16` |
|---|---|---|---|
| distributed levels | L0 | L0–L1 | L0–L2 |
| `ParallelCopy_finish` / solve | **96** | 80 | 64 |
| time | **17.74** | 21.01 | 24.59 |

The count follows `2 × (agglomerated levels) × iterations` exactly, in all 15
configs. So the winning configuration does the *most* redistribution copying and
is still fastest by 28%. The redistribution is not the cost; **being distributed
is**. Moving a level onto one rank does not change how many halo exchanges it
does — it changes each one from an MPI exchange into a local copy.

The price of leaving one more level distributed is close to constant, and does
not scale with the grid it is on:

| level moved to distributed | 2 GPUs | 4 GPUs |
|---|---|---|
| L1 (64³) | 1.5–2.2 ms | 2.5–3.3 ms |
| L2 (32³) | 2.3–2.6 ms | 3.2–3.7 ms |

A 32³ level costs as much as a 64³ one — eight times less data for the same
price. That is latency and synchronisation, not bandwidth, and it is why the
answer is to agglomerate as early as the API allows.

Between the two axes at 4 GPUs, the contributions are comparable: level count
(7→4 at `ags=64`) is −33.3%, agglomeration (`ags` 16→64 at 4 levels) is −34.1%.

## Best configuration

**`max_coarsening_level=3 agg_grid_size=64`** — 4 MG levels, 16³ coarsest,
everything below level 0 on one rank.

| | AMReX default | best | gain |
|---|---|---|---|
| 1 GPU | 14.35 | **9.74** | **32.2%** |
| 2 GPUs | 18.93 | **11.05** | **41.6%** |
| 4 GPUs | 21.01 | **11.83** | **43.7%** |

(Default = `mcl=30`, default `agg_grid_size`. At 1 GPU the `ags` knob is a no-op.)

This is one level shallower than run-002's recommendation. On this problem,
run-002's `mcl=4`+`ags=64` gives 14.20 ms at 4 GPUs; `mcl=3` gives 11.83 —
**16.7% better**. Iteration count drops 8 → 7 at the same time, so this one wins
on both time and iterations.

Worst config measured is `mcl=30 ags=16` at 4 GPUs, 24.59 ms — 2.08× the best.

## Scaling

Still negative, and still the headline structural problem.

| | best config | vs 1 GPU |
|---|---|---|
| 1 GPU | 9.74 ms | 1.00× |
| 2 GPUs | 11.05 ms | 1.14× |
| 4 GPUs | 11.83 ms | 1.22× |

Improved from run-002 (1.20× / 1.28×), but the direction has not changed: at
128³ extra GPUs buy capacity, not speed. The best configuration is the one that
uses the GPUs least — only the finest level runs distributed.

## Recommendations

1. **`max_coarsening_level=3 agg_grid_size=64`** — 32–44% faster than AMReX
   defaults, and fewer iterations than the default as well.
2. **Agglomerate as early as possible.** Confirmed on stock AMReX with a
   well-posed operator, 10/10 combinations. The default `agg_grid_size` leaves
   one level too many distributed and costs ~3 ms per rank-count doubling.
3. **Do not carry run-002's level-count magnitudes forward.** The 28–33%
   multi-GPU figure was the singular-operator bottom solve. On a well-posed
   problem the level-count effect is ~7% per level down to 4 levels, and the
   optimum is 4 levels, not 5.
4. **Do not coarsen below 16³ on this problem.** At 32³ coarsest the BiCGStab
   bottom solve does 5× the inner products and gives the time back.

## Next

- **`FillBoundary` remains the structural item, untouched since run-000.** Even
  in the best config it is ~10.4 halo exchanges per level per V-cycle against 4
  smooths — 292 calls per solve at 4 levels. Every `ags` result above is a
  statement about how expensive those exchanges are when they cross ranks; the
  count itself has never moved. This needs an AMReX-side change, not a knob.
- **Attribution needs an `nosync=0` run.** Every per-function *time* here is a
  pipeline-drain artefact and was not used; the analysis above rests entirely on
  call counts and total solve time. A single `nosync=0` run at `mcl=3 ags=64`
  and at `mcl=3 ags=16` would confirm the latency interpretation of the ~3 ms
  per-distributed-level cost directly.
- **The `agg_grid_size` axis is truncated at its optimum.** 64 is the largest
  value tested and it won everywhere, monotonically. Values above 64 cannot
  agglomerate earlier than level 1 on this hierarchy, so 64 is likely already
  saturated — worth one confirming point at 128 rather than an axis.
- **The 3-vs-4 level margin is thin at 4 GPUs (1.7%).** If the production
  problem is larger than 128³ the coarsest grid at fixed `mcl` grows with it and
  the bottom-solve wall moves; `mcl` should be re-tuned at the production size
  rather than carried over as `3`.
- `MLEBNodeFDLaplacian::isSingular()` (the deferred AMReX bug) is now quantified:
  on the affected 6-level periodic case it cost 34% of solve time and 17× the
  bottom-solver inner products. That is the size of the prize if it is ever fixed
  — but only for problems that are actually singular, which this one no longer is.
