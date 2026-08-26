# run-004 — Nsight Systems profile of the run-003 optimum, 1 GPU

`max_coarsening_level=3 agg_grid_size=64`, 1 rank, 128³, 4 MG levels, 7 iterations.
6 runs on Perlmutter; `nosync=1`, `recreate_linop=0`, `nsolves=20`, `nwarmup=3`.

## Validity — read this before the numbers

Four things went differently than planned. None invalidates the profile; two
change how it must be read.

### 1. The baseline does not reproduce run-003, and the reason is the hardware

| | run-003 `g1-mcl3` | run-004 `base` |
|---|---|---|
| ms/solve | 9.735 | **9.176 / 9.172** |
| GPU | A100-SXM4 **40 GB** | A100-SXM4 **80 GB** |
| AMReX | `26.08-22-g189d80c7d704` | `26.08-24-g92c458707217` |

The README's validation criterion said a mismatch means the environment moved.
It did — but cleanly. The two AMReX commits are `0aa26b6dd5 "mg min width = 4"`
and `92c4587072 "revert last commit"`: **the source diff between the two
revisions is empty.** MG hierarchy, iteration count (7) and final residual
(`3.15736326e-12`) are identical to run-003. So the only variable is the GPU.

That accident is worth more than the run it disturbed. A100 40 GB → 80 GB is
HBM2 → HBM2e, **1555 → 2039 GB/s, +31% memory bandwidth**, for **+6.1% solve
speed**. A bandwidth-bound solve would have moved most of the way to +31%. This
is the first direct measurement of the run-003 latency-bound inference, and it
came from a hardware swap nobody planned.

Cross-run timing comparisons with run-000..003 are now invalid without
accounting for this. Comparisons *within* run-004 are unaffected.

### 2. Report generation failed on Perlmutter; the data was recovered locally

Every nsys run wrote a `.qdstrm` but no `.nsys-rep`:

```
Import Failed ... CreateFileException
[errinfo_errno] = 524, "Unknown error 524"
[file] = /global/u2/w/wqz/mygitrepo/warpx-solver/nodefd/run-004/nsys-nosync1-r1.nsys-rep
```

errno 524 is `ENOTSUPP`. The importer's file-creation path is not supported on
NERSC **global home** (`/global/u2`). The intermediate `.qdstrm` went to
`/tmp/nsys-wqz/` and was fine — only the final write to home failed. Phase 3
then found no `.nsys-rep` and silently did nothing.

All three traces were converted locally without loss:
`QdstrmImporter --input-file X.qdstrm`. **Next run: write reports to
`$PSCRATCH`, not `$HOME`.**

### 3. GPU metrics failed — not the permission problem that was anticipated

```
Ampere GA100 | NVIDIA A100-SXM4-80GB PCI[0000:c1:00.0] - Already under profiling
```
— all four GPUs. Not `ERR_NVGPUCTRPERM`. Something already holds the profiling
context on every GPU of the node; the most likely candidate is NERSC's
node-health GPU monitoring (DCGM), which is known to occupy it. That is a
question for NERSC, not a script bug. The availability probe added to `run.sh`
is what made this diagnosable rather than a bare usage error.

### 4. The `--sample=cpu` arm was a null result, and the cause was our own flags

56 samples in the 206 ms window, none symbolicated. But the interesting part is
why the data was missing at all: **nsys traces `cuda,nvtx,osrt,opengl` by
default, and `--trace=cuda,nvtx` removed `osrt`.** `osrt_sum` on the trace
returns "does not contain OS Runtime trace data" for that reason alone. The
separate `--sample=cpu` run existed to recover host-side attribution that the
default collection would have supplied for free.

So the collect line was over-specified and net-negative: it dropped useful data
and added a run to compensate. Plain `nsys profile -o <name> <exe> <inputs>`
would have produced strictly more than what was collected here. The only flag
that turned out to be *necessary* is an `-o` under `$PSCRATCH` (see 2 above).
`--capture-range` was redundant with filtering on `REG::solve` at stats time.

None of this changes a number in this report — the GPU timeline and the NVTX
projection are complete and self-consistent — but it is the lesson for run-005.

Both `nosync=1` repeats agree to **0.01%** (10.327 / 10.326 ms/solve).

### 5. nsys overhead is not negligible

| | ms/solve |
|---|---|
| no nsys | 9.174 |
| under nsys | 10.273 |

**+12.0%**, i.e. **+1.40 µs per kernel launch** across 786 launches/solve. That
number is itself a result: adding 1.4 µs to each launch costs 12% of the solve.
Absolute GPU times below carry this; proportions do not.

## Where the 9.2 ms actually goes

First real attribution in the series. `REG::solve`, 20 solves, per solve:

| | ms/solve | share |
|---|---|---|
| wall (under nsys) | 10.327 | 100% |
| **GPU busy (kernels)** | **8.066** | **78.1%** |
| GPU idle | 2.261 | 21.9% |

786 kernel launches per solve. By kernel:

| kernel | % GPU | ms/solve | calls/solve | med µs |
|---|---|---|---|---|
| `Fsmooth` | 41.7 | 3.367 | 168 | 7.23 |
| `Array4CopyTag` (FillBoundary copies) | 15.3 | 1.232 | 301 | 3.58 |
| `Fapply` | 7.9 | 0.636 | 41 | 6.56 |
| `interpolation` | 6.7 | 0.542 | 21 | 9.12 |
| `solutionResidual` | 5.6 | 0.448 | 8 | 55.97 |
| `setVal` | 4.4 | 0.355 | 58 | 2.59 |
| `ReduceOpMax` (norminf) | 4.2 | 0.339 | 28 | 4.58 |
| `Xpay` | 3.3 | 0.267 | 21 | 5.44 |
| `restriction` | 2.4 | 0.193 | 21 | 5.44 |
| `ReduceOpSum` (Dot) | 1.8 | 0.148 | 24 | 5.86 |

And by AMReX range, GPU time projected onto NVTX — the number the TinyProfiler
tables could never give:

| range | GPU ms/solve | % of window | host ms/solve | host/GPU |
|---|---|---|---|---|
| `MLMG::solve()` | 10.229 | 99.1 | 10.272 | 1.00 |
| `Fsmooth` | 4.429 | 42.9 | 2.368 | **0.53** |
| `actualBottomSolve` | 1.549 | 15.0 | 2.974 | 1.92 |
| `FillBoundary_nowait` | 1.288 | 12.5 | 1.926 | 1.50 |
| `applyBC` | 0.914 | 8.8 | 1.828 | 2.00 |
| `Fapply` | 0.636 | 6.2 | 0.241 | 0.38 |
| `norminf` | 0.554 | 5.4 | **4.392** | **7.93** |

The `host/GPU` column is the whole `nosync=1` story in one place. Under 1, the
host has run ahead and the GPU work outlives the range (`Fsmooth` 0.53,
`Fapply` 0.38). Over 1, the host is blocked (`norminf` **7.93** — 4.4 ms/solve
of host time for 0.55 ms of GPU work). The host races ahead queueing smooths,
then stalls in the convergence check waiting for the queue to drain. Host-side
CUDA API confirms it: `cudaStreamSynchronize` **4.272 ms/solve** (71 calls,
60 µs each) against `cudaLaunchKernel` 3.110 ms/solve (786 calls, 3.96 µs each).

## The finding: the solve is two different problems at once

Bucketing kernels by grid size gives the MG level (`Fsmooth` confirms the
mapping: 8386 blocks = 129³ nodes = L0, 1073 = 65³ = L1, 141 = 33³ = L2):

| level | calls/solve | GPU ms/solve | % GPU | launch cost ms/solve | **launch/exec** |
|---|---|---|---|---|---|
| L0 128³ | 113 | 4.820 | 59.8 | 0.447 | **0.09** |
| L1 64³ | 104 | 0.985 | 12.2 | 0.412 | 0.42 |
| L2 32³ | 202 | 0.911 | 11.3 | 0.800 | 0.88 |
| L3 16³ + bottom | 367 | 1.350 | 16.7 | 1.453 | **1.08** |

**The two ends of the V-cycle are limited by opposite things.**

**The fine level is bandwidth-bound.** L0 is 59.8% of GPU time in 14% of the
launches. `Fsmooth` on L0 runs 2.15M threads in 48.97 µs; at 24–40 B/node of
traffic that is **1050–1750 GB/s, or 52–86% of the 80 GB A100's 2039 GB/s
peak**. There is little headroom here, and the 40→80 GB result agrees: the part
that responds to bandwidth is only ~60% of the GPU time, so +31% bandwidth
bought +6% overall.

**The coarse levels are launch-bound.** L2+L3 are **72% of all launches for 28%
of the GPU time**, and at L3 **launching costs more than executing** (1.08×).
Across the whole solve, **46% of kernels are shorter than their own 3.96 µs
launch cost**, and **47% of launches are too small to fill the A100's 108 SMs**
(≤108 blocks) while contributing 16.7% of GPU time. Median kernel: 4.32 µs.

This is the mechanism behind run-003's central result. run-003 found that
removing a coarse level saved a near-constant ~2 ms *regardless of whether the
level was 64³ or 32³ — eight times less data for the same price* — and inferred
latency. That inference is now measured: the coarse levels cost launches, not
bytes, and launches do not shrink with the grid. It also explains why the
optimum is 4 levels rather than more.

## Answers to the questions run-004 asked

1. **Where do the 9.2 ms go?** 78% GPU-busy; `Fsmooth` 42%, FillBoundary copies
   15%, bottom solve 15%. Answered.
2. **Latency- or bandwidth-bound?** Both, in different places — fine level at
   52–86% of peak bandwidth, coarse levels launch-bound. The single most
   convincing number is unplanned: +31% hardware bandwidth → +6% solve.
3. **What is `FillBoundary` doing at 1 rank?** `Array4CopyTag` local copy
   kernels: **301 launches/solve, 1.232 ms, 15.3% of GPU time, median 3.58 µs —
   below the 3.96 µs launch cost. 59% of them are launch-limited.** The
   structural item from run-000 is now quantified, and it is a launch-count
   problem, not a data-movement one.
4. **Do coarse levels cost more than launch overhead?** No — at L3 they cost
   *less*: 1.453 ms of launch for 1.350 ms of execution.
5. **GPU-idle fraction?** 21.9% (2.261 ms/solve).
6. **What does `nosync=1` buy?** Not measured — arm cut before submission. The
   `host/GPU` column shows the mechanism it exploits.

The host-side question (2c's purpose) was answered by `cuda_api_sum` from the
ordinary runs, not by the sampling arm that was added for it.

## Recommendations

1. **Stop tuning level count; start reducing launch count.** The `mcl` axis is
   exhausted — run-003 found the optimum and run-004 explains it. The remaining
   coarse-level cost is 569 launches/solve for 2.26 ms, over half of it launch
   overhead. Fusing or batching coarse-level kernels, or CUDA-graph capture of
   the V-cycle, attacks the actual limiter. A graph replay would remove most of
   the 3.11 ms/solve of `cudaLaunchKernel` time.
2. **`FillBoundary` is the single best target**, confirming run-000 through
   run-003 with numbers for the first time: 301 launches/solve at 1 rank with
   zero MPI, most of them shorter than their launch cost. This is the AMReX-side
   change the last four analyses have pointed at.
3. **Do not expect much from faster memory.** +31% bandwidth bought +6%.
4. **Write nsys reports to `$PSCRATCH`.** Global home cannot create `.nsys-rep`
   (errno 524).
5. **Collect with the default trace.** `nsys profile -o $PSCRATCH/<name> <exe>
   <inputs>` is the whole recipe. Restricting `--trace` cost this run its
   host-side data; filtering to `REG::solve` belongs at `nsys stats` time, where
   it is free and reversible.

## Next

- **CUDA graphs are the obvious experiment.** 786 launches/solve at 3.96 µs is
  3.11 ms/solve of pure launch cost against 8.07 ms of GPU work. AMReX has graph
  support; capturing one V-cycle would test the ceiling directly. This is the
  first run in the series to make that the clear next move rather than a knob.
- **Re-run GPU metrics** once the "Already under profiling" holder is
  identified. It would confirm the 52–86% bandwidth estimate, which currently
  rests on an assumed byte-per-node count — the one soft number in this report.
- **The 80 GB result deserves a deliberate run**, not an accident: `mcl` sweep
  at 128³ on 80 GB hardware to check the optimum has not moved now that the
  fine-level cost has dropped ~6% while coarse-level launch cost has not.
- A `nosync=0` trace would still cleanly separate queueing from execution, if
  the `host/GPU` projection ever proves ambiguous.

## Standing rules

- Total solve time and call counts remain the only trustworthy TinyProfiler
  outputs. Everything per-function in this report comes from the nsys hardware
  timeline or the NVTX projection.
- No timings from the local box. The recovery conversion done there is a
  format transformation, not a measurement.
