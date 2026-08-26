#!/bin/bash
#SBATCH --account=ntrain6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug
#SBATCH --reservation=hackathon_day2

# run-004: Nsight Systems profile of the run-003 optimum, 1 GPU.
#
# Config under study:  max_coarsening_level=3 agg_grid_size=64, 1 rank.
#   = 4 MG levels, 16^3 coarsest, 7 iterations, 9.735 ms/solve in run-003.
#
# WHY nsys.  Every per-function TIME in run-000..003 was unusable: nosync=1
# leaves work queued on the stream, so TinyProfiler's host-side timers measure
# launch, not completion, and the standing rule has been "total solve time
# only".  That left the central question of run-003 unanswered -- where do the
# 9.7 ms actually go?  nsys reads the CUDA hardware timeline, which is ground
# truth regardless of nosync, and `nvtx_gpu_proj_sum` projects the AMReX NVTX
# ranges onto it, so we get per-function GPU attribution WITHOUT having to give
# up nosync=1.  That is why there is no nosync=0 arm here: the projection makes
# one unnecessary, and a nosync=0 run would profile a configuration nobody runs.
#
# The allocation shape (4 tasks / 4 GPUs, srun -n 1) deliberately matches
# run-003 so the 1-GPU numbers stay comparable -- same GPU-visibility
# heuristics, same affinity.
#
# NVTX comes free: GNUmakefile has TINY_PROFILE=TRUE, and under CUDA
# TinyProfiler pushes an nvtxRangePush for every BL_PROFILE
# (AMReX_TinyProfiler.cpp:167).  BL_PROFILE_REGION("solve") shows up as the
# range `REG::solve`, which is what every stats report below is filtered on --
# so warmup, setup and teardown are excluded without touching main.cpp.
#
# Verified locally before submission (structure only -- this box has one GPU
# and its timings mean nothing):
#   - all nsys flags below parse and run against this exact binary;
#   - `--capture-range=cudaProfilerApi` works: AMReX calls cudaProfilerStart()
#     in Device::Initialize() and cudaProfilerStop() in Device::Finalize()
#     (AMReX_GpuDevice.cpp:457,467), so the capture spans AMReX's lifetime and
#     drops MPI/CUDA context setup;
#   - `REG::solve` and `REG::warmup` appear as NVTX ranges, and all 8 stats
#     reports below accept --filter-nvtx=REG::solve;
#   - at 1 rank, agg_grid_size=64 gives an IDENTICAL hierarchy to the default
#     ("consolidated AMR level 0 starting at MG level 3 of 4").  It is a no-op
#     here, set explicitly only to pin the config on record.
#
# ==> VALIDATION: run-base-r1.ou must reproduce run-003's 9.735 ms/solve and
#     "MG levels : 4" with 7 iterations.  If it does not, the environment has
#     moved since run-003 and nothing below is comparable to it.

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"

EXE=../nodefd3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=../inputs

# ---- the config under study -----------------------------------------------
MCL=3            # max_coarsening_level -> MCL+1 MG levels
AGS=64           # agg_grid_size (a no-op at 1 rank; set to pin it on record)

# The rest is identical to run-003's 1-GPU arm.
COMMON="nosync=1 recreate_linop=0 nsolves=20 nwarmup=3 linop_verbose=2 \
        max_coarsening_level=${MCL} agg_grid_size=${AGS}"

# nsys is on PATH by default on Perlmutter -- no module load needed.  Record
# the version anyway: report format and the --gpu-metrics-devices spelling are
# both version-dependent, and run-003 showed how much a silent tool change can
# cost when reading results months later.
nsys --version | tee nsys-version.txt

# Common nsys options.  Deliberate choices:
#   --trace=cuda,nvtx  : no MPI trace.  At 1 rank there is no MPI traffic to
#                        see, and --mpi-impl=mpich could not be validated off
#                        Perlmutter.
#   --capture-range    : start at AMReX init, end at AMReX finalize.
NSYS="nsys profile --force-overwrite=true --trace=cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop"

# ===========================================================================
# PHASE 1 -- baseline, NO nsys.
# Two jobs: reproduce run-003's 9.735 ms, and give the reference against which
# nsys's own overhead is measured.  Do not skip this: if nsys perturbs the run
# materially, every number in phase 2 needs that caveat attached.
# ===========================================================================
for REP in 1 2; do
    srun -n 1 ${EXE} ${INPUTS} ${COMMON} >& run-base-r${REP}.ou
done

# ===========================================================================
# PHASE 2 -- trace collection.
# All traces are collected BEFORE any post-processing, so that a wall-clock
# overrun during stats generation still leaves every .nsys-rep on disk.
# `nsys stats` can always be re-run later on a login node.
# ===========================================================================

# 2a. The primary measurement: production config, minimal collector overhead.
#     Two repeats, matching phase 1, so the overhead figure has a spread.
for REP in 1 2; do
    srun -n 1 ${NSYS} --sample=none --cpuctxsw=none -o nsys-nosync1-r${REP} \
         ${EXE} ${INPUTS} ${COMMON} >& run-nsys-nosync1-r${REP}.ou
done

# 2b. Host-side sampling.  run-003 inferred that this problem is latency-bound,
#     not bandwidth-bound.  If the host cannot issue launches fast enough to
#     keep the queue full, the stall is on the CPU and this is where it shows.
srun -n 1 ${NSYS} --sample=cpu --cpuctxsw=process-tree -o nsys-cpusample \
     ${EXE} ${INPUTS} ${COMMON} >& run-nsys-cpusample.ou

# 2c. GPU metrics: SM occupancy and DRAM throughput, i.e. the direct test of
#     latency-bound vs bandwidth-bound.
#     LAST and non-fatal on purpose.  This needs GPU performance counters to be
#     readable by unprivileged users; it fails locally with ERR_NVGPUCTRPERM and
#     could not be validated off Perlmutter.  If it fails, phases 1-2b are
#     unaffected -- check run-nsys-gpumetrics.ou before assuming the data exists.
#     The probe below runs first and records, on the compute node itself,
#     whether the GPUs are usable for metrics.  Its output is the diagnostic:
#     "Insufficient privilege / ERR_NVGPUCTRPERM" means counters are locked
#     down and the profile that follows cannot succeed -- that is a site
#     configuration question for NERSC, not a bug in this script.
echo "--- gpu-metrics availability probe ---" > run-nsys-gpumetrics.ou
srun -n 1 nsys profile --gpu-metrics-devices=help >> run-nsys-gpumetrics.ou 2>&1 \
  || true
echo "--- gpu-metrics profile attempt ---" >> run-nsys-gpumetrics.ou

srun -n 1 ${NSYS} --sample=none --cpuctxsw=none \
     --gpu-metrics-devices=cuda-visible -o nsys-gpumetrics \
     ${EXE} ${INPUTS} ${COMMON} >> run-nsys-gpumetrics.ou 2>&1 \
  || echo "WARNING: gpu-metrics run failed; see the probe output above. \
Phases 1-2b are unaffected." >> run-nsys-gpumetrics.ou

# ===========================================================================
# PHASE 3 -- stats extraction, filtered to the timed region.
#
# --filter-nvtx=REG::solve restricts every report to BL_PROFILE_REGION("solve"),
# so warmup, grid setup and linop construction are excluded.
#
# The reports, and what each is for:
#   cuda_gpu_kern_sum    which kernels own the GPU time
#   cuda_gpu_kern_gb_sum grid/block per kernel -- exposes the tiny coarse-level
#                        launches that occupy a handful of SMs
#   cuda_gpu_sum         kernels + memops together
#   cuda_gpu_mem_time_sum  memops alone (H2D/D2H/D2D)
#   nvtx_gpu_proj_sum    ** the key one ** GPU time projected onto AMReX ranges;
#                        this is what replaces the unusable TinyProfiler times
#   cuda_kern_exec_sum   launch / queue / exec split.  Q >> K means the GPU is
#                        waiting on launches, i.e. latency-bound
#   cuda_api_sum         host-side CUDA API cost
#   nvtx_pushpop_sum     host range time, for comparison against the projection
# ===========================================================================
REPORTS="-r cuda_gpu_kern_sum -r cuda_gpu_kern_gb_sum -r cuda_gpu_sum \
         -r cuda_gpu_mem_time_sum -r nvtx_gpu_proj_sum -r cuda_kern_exec_sum \
         -r cuda_api_sum -r nvtx_pushpop_sum"

# NOTE on tool version: these flags and report names were validated against nsys
# 2026.1.3; Perlmutter runs 2026.2.1.  A minor bump should not move any of them,
# but that was not verifiable off-site, and the failure mode is quiet: nsys
# writes "ERROR: Report 'x' could not be found." into the output and STILL EXITS
# 0.  A renamed report would therefore vanish with no error and no missing file.
# Hence the explicit count below -- verified against a deliberately bogus report
# name.  NREPORTS must match the number of -r flags in REPORTS.
NREPORTS=8

for REPFILE in *.nsys-rep; do
    [ -e "${REPFILE}" ] || continue
    BASE=${REPFILE%.nsys-rep}
    echo "=== stats: ${BASE} (filtered to REG::solve) ==="
    nsys stats --force-export=true --filter-nvtx=REG::solve ${REPORTS} \
         "${REPFILE}" >& "${BASE}.stats" \
      || echo "WARNING: stats call failed for ${BASE}; .nsys-rep is still on disk"

    NGOT=$(grep -c '^ \*\*' "${BASE}.stats" || true)
    if [ "${NGOT}" -ne "${NREPORTS}" ]; then
        echo "WARNING: ${BASE}.stats has ${NGOT}/${NREPORTS} reports." \
             "Report names may have moved in nsys 2026.2 -- see below."
        grep -n 'could not be found' "${BASE}.stats" || true
    fi
    # Unfiltered totals too -- the difference against the filtered numbers is
    # the setup + warmup cost, which is worth having rather than discarding.
    nsys stats --report cuda_gpu_kern_sum --report nvtx_pushpop_sum \
         "${REPFILE}" >& "${BASE}.stats-unfiltered" \
      || true
done

# `nsys --version` is already in nsys-version.txt; if any WARNING above fired,
# `nsys stats --help-reports` on a login node lists what this version does have.


# Trim the SQLite exports; they are large and regenerable from the .nsys-rep.
rm -f ./*.sqlite

echo "run-004 complete."
ls -la
