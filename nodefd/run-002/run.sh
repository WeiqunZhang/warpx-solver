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
#SBATCH --reservation=hackathon_day1

# run-002: level-count-matched comparison of agglomeration vs consolidation.
#
# run-001 found no-agglomeration + consolidation beats agglomeration by 21-31%,
# but the two differed in MG level count (5 vs 6) as well as mechanism.  This
# run removes that confound: agg with max_coarsening_level=4 has EXACTLY the
# same hierarchy as noagg+consolidation -- same levels, same domains, same
# ranks per level, same collapse point (L2) -- differing only in the number of
# boxes on the owning rank (1 vs N), i.e. purely the redistribution mechanism.
#
# Verified at 4 ranks before submission:
#   agg   mcl=4 : lev 0-1  4 boxes / 4 ranks ; lev 2-4  1 box  / 1 rank (rank 0)
#   con   cr4   : lev 0-1  4 boxes / 4 ranks ; lev 2-4  4 boxes/ 1 rank (rank 0)
# Both converge in 9 iterations.
#
# Requires the AMReX min-width patch (GPU: mg_domain_min_width =
# mg_box_min_width = 4).  Check every .ou reports "MG levels : 6" for the mcl=30
# configs and "MG levels : 5" for the mcl=4 / noagg configs.
#
# Standing settings: nosync=1, recreate_linop=0 (also the ../inputs defaults;
# repeated on the command line for provenance in the logs).

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"
EXE=../nodefd3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=../inputs
COMMON="nosync=1 recreate_linop=0 nsolves=20 nwarmup=3 linop_verbose=2"

# Two repeats of everything: the effects being separated are ~0.5 ms against
# ~25 ms totals, so run-to-run spread must be bounded, not assumed.
for REP in 1 2; do

  # ---- 1 GPU: pure MG level cost, no redistribution of any kind ----------
  # Isolates the "extra level" term on A100.  run-001 could only bound it at
  # ~0.54 ms from a local RTX 5070.
  srun -n 1 ${EXE} ${INPUTS} ${COMMON} \
       >& run-g1-mcl6-r${REP}.ou
  srun -n 1 ${EXE} ${INPUTS} ${COMMON} max_coarsening_level=4 \
       >& run-g1-mcl5-r${REP}.ou

  for NGPU in 2 4; do

    # ---- A: agglomeration, 6 levels (run-001 baseline) -------------------
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} \
         >& run-g${NGPU}-agg6-r${REP}.ou

    # ---- B: agglomeration, 5 levels  <-- MATCHED to C --------------------
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} max_coarsening_level=4 \
         >& run-g${NGPU}-agg5-r${REP}.ou

    # ---- C: no agglomeration + consolidation, 5 levels  <-- MATCHED to B -
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agglomeration=0 con_ratio=4 \
         >& run-g${NGPU}-con5-r${REP}.ou

    # ---- E: collapse-early agglomeration at matched level count ----------
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agg_grid_size=64 max_coarsening_level=4 \
         >& run-g${NGPU}-agg64x5-r${REP}.ou

  done
done
