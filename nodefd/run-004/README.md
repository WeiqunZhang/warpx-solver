# run-004 — Nsight Systems profile of the run-003 optimum, 1 GPU

## Why

run-003 established the best configuration on this problem —
`max_coarsening_level=3 agg_grid_size=64`, 4 MG levels, **9.735 ms/solve at
1 GPU** — but could not say where that time goes.

Every per-function *time* in run-000..003 was discarded under the standing rule,
and correctly so: `nosync=1` leaves work queued on the stream, so TinyProfiler's
host-side timers measure launch, not completion. All four analyses so far rest
on **call counts and total solve time only**. That is enough to rank
configurations and it is not enough to say what to fix.

Nsight Systems reads the CUDA hardware timeline, which is ground truth
regardless of `nosync`. The `nvtx_gpu_proj_sum` report then projects the AMReX
NVTX ranges onto that timeline, giving per-function **GPU** attribution without
having to abandon `nosync=1`.

This is the attribution run run-003's Next section asked for, done a better way
than the `nosync=0` re-run proposed there. Hence no `nosync=0` arm: the
projection makes one unnecessary, and it would profile a configuration nobody
runs.

## Configuration under study

`max_coarsening_level=3 agg_grid_size=64`, 1 rank, 128³, `is_periodic = 0 1 1`.
`nosync=1`, `recreate_linop=0`, `nsolves=20`, `nwarmup=3` — identical to
run-003's `g1-mcl3` arm.

`agg_grid_size=64` is a **no-op at one rank** — verified locally, it produces an
identical hierarchy to the default (`consolidated AMR level 0 starting at MG
level 3 of 4`). It is set explicitly to pin the configuration on record; the
profile is about `mcl=3`.

The allocation shape (4 tasks / 4 GPUs, `srun -n 1`) deliberately matches
run-003 rather than requesting a single GPU, so the same GPU-visibility
heuristics and affinity apply and the timings stay comparable.

## The runs

| tag | nsys | `nosync` | `mcl` | purpose |
|---|---|---|---|---|
| `base-r1`, `base-r2` | — | 1 | 3 | reproduce run-003; reference for nsys overhead |
| `nsys-nosync1-r1/r2` | yes | 1 | 3 | **primary** — production timeline |
| `nsys-cpusample` | yes | 1 | 3 | host-side sampling — is the CPU the bottleneck? |
| `nsys-gpumetrics` | yes | 1 | 3 | SM occupancy + DRAM throughput |

6 runs, all 1 GPU, a few seconds of solve each. **One solver configuration
throughout** — `mcl=3`, `ags=64`, `nosync=1`, set once at the top of `run.sh` as
`MCL` / `AGS`. The rows differ only in what the collector records, never in what
is solved.

Two contrast runs were considered and cut: `mcl=30`, because run-003 already
ranked the level counts and what is missing is attribution *within* the winning
config; and `nosync=0`, because the NVTX projection removes the reason to want
it. Nothing here profiles a configuration that would not be run in production.

## Questions this should answer

1. **Where do the 9.7 ms go?** Per kernel (`cuda_gpu_kern_sum`) and per AMReX
   range (`nvtx_gpu_proj_sum`). First real attribution in the series.
2. **Latency-bound or bandwidth-bound?** run-003 inferred latency from the
   per-distributed-level cost being flat at ~3 ms whether the level was 64³ or
   32³ — eight times less data for the same price. That was an inference from
   totals. `cuda_kern_exec_sum` splits launch / queue / execute directly, and
   the GPU-metrics run gives achieved DRAM throughput and SM occupancy.
3. **What is `FillBoundary` actually doing at 1 rank?** It is the structural item
   since run-000 — ~10.4 halo exchanges per level per V-cycle against 4 smooths,
   292 calls per solve at 4 levels, with **zero MPI traffic at one rank**. If it
   is expensive here, it is packing kernels and launch overhead, and the trace
   will name them.
4. **Do the coarse levels cost anything but launch overhead?**
   `cuda_gpu_kern_gb_sum` gives grid/block per kernel; the 16³ and 32³ levels
   should show launches occupying a handful of SMs.
5. **What fraction of the solve is GPU-idle?** Total kernel time against the
   `REG::solve` range span.

## Mechanics

NVTX needs no code change. `GNUmakefile` has `TINY_PROFILE = TRUE`, and under
CUDA TinyProfiler pushes an `nvtxRangePush` for every `BL_PROFILE`
(`AMReX_TinyProfiler.cpp:167`). `BL_PROFILE_REGION("solve")` appears as the
range **`REG::solve`**, and every stats report is filtered on it, so setup,
warmup and teardown are excluded. Unfiltered totals are also written to
`*.stats-unfiltered`; the difference is the setup + warmup cost.

`--capture-range=cudaProfilerApi` works because AMReX calls `cudaProfilerStart()`
in `Device::Initialize()` and `cudaProfilerStop()` in `Device::Finalize()`
(`AMReX_GpuDevice.cpp:457,467`). The capture therefore spans AMReX's lifetime
and drops MPI and CUDA context setup.

MPI tracing is deliberately **not** enabled: at one rank there is no MPI traffic
to see, and `--mpi-impl=mpich` could not be validated off Perlmutter.

Traces are all collected before any post-processing, so a wall-clock overrun
during stats generation still leaves every `.nsys-rep` on disk — `nsys stats`
can be re-run on a login node.

## Prerequisites

Reservation is `hackathon_day2` (run-003 used `hackathon_day1`).

`nsys` is on PATH by default on Perlmutter — no module load. Its version is
recorded in `nsys-version.txt` regardless: report format and the
`--gpu-metrics-devices` spelling are both version-dependent, and run-003 showed
what a silent tool change costs when results are read back months later.

**Version skew:** Perlmutter has nsys **2026.2.1**; the flags and the eight
report names below were validated against **2026.1.3** locally. A minor bump
should not move any of them, but that could not be verified off-site, and the
failure mode is quiet — nsys writes `ERROR: Report 'x' could not be found.` into
the output and **still exits 0**, so a renamed report would vanish leaving no
error and no missing file. Phase 3 therefore counts the reports it actually got
and warns on a shortfall; if that fires, `nsys stats --help-reports` on a login
node lists what 2026.2 does have.

Validated locally before submission (structure only — this box has one GPU and
its timings mean nothing): the full script runs end to end against this exact
binary and all 8 stats reports generate cleanly under
`--filter-nvtx=REG::solve`.

## Reading the results

- **Validation first.** `run-base-r1.ou` must reproduce run-003's **9.735
  ms/solve**, `MG levels : 4`, and **7 iterations**. If it does not, the
  environment has moved since run-003 and nothing here is comparable to it.
- **Then nsys overhead.** `nsys-nosync1` against `base`. Every kernel-level
  number is measured under the collector; if the collector costs more than a few
  percent, that caveat attaches to all of it.
- **`nvtx_gpu_proj_sum` is the report to read first**, not `nvtx_pushpop_sum`.
  The push/pop numbers are host range times and carry exactly the `nosync=1`
  distortion that made the TinyProfiler tables unusable. The projection is GPU
  time.
- The `Bottom` timer and the TinyProfiler tables in the `.ou` files remain
  unusable for the same reason as always. They are left in only because the
  driver prints them.

## Standing rules

- `nosync=1`, `recreate_linop=0` — established, not under test.
- `tiny_profiler.device_synchronize_around_region = 0`. Leave it there: setting
  it puts back exactly the syncs `nosync=1` removes. nsys does not need it.
- No timings from the local box, ever. One GPU, wrong architecture.
