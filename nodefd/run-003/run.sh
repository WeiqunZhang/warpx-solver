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

# run-003: max_coarsening_level x agg_grid_size sweep, Dirichlet in x.
#
# CHANGED SINCE run-002 -- both matter for comparison:
#
#  1. The problem is no longer fully periodic.  is_periodic = 0 1 1, i.e.
#     homogeneous Dirichlet in x, periodic in y and z.  This removes the
#     constant nullspace, so MLEBNodeFDLaplacian's hard-coded
#     isSingular()==false no longer matters and the bottom solve stops being
#     pathological on coarse grids.
#
#  2. The AMReX mg_min_width=4 patch is REVERTED -- stock 2/2.  It existed only
#     to dodge the singular 2-cell level, which (1) has eliminated.  Reverting
#     lets the sweep reach 7 MG levels and makes the results applicable to
#     upstream AMReX rather than to a modified build.
#
# ==> PREREQUISITES on Perlmutter, both mandatory:
#       a) revert commit 0aa26b6dd5 ("mg min width = 4") in the AMReX checkout
#       b) REBUILD the executable afterwards
#     Check: the mcl=30 runs must report "MG levels : 7".  If they say 6, the
#     patch is still in and the whole sweep is shifted by one level.
#
# Verified locally before submission (structure/convergence only, no timings --
# this box has one GPU): all 15 (mcl, ags) combinations converge, 7/6/5/4/3
# levels for mcl 30/5/4/3/2, agglomeration starting at L2/L3/L1 for
# ags -1/16/64.  bicgstab is stable at every depth including 7 levels.
#
# NOTE: iteration count varies with level count (8,8,8,7,6 for 7,6,5,4,3
# levels), so this sweep is NOT iteration-matched.  That is inherent -- fewer
# levels means the bottom solve does more of the work per V-cycle.  Total
# time-to-converge is the metric; report iteration counts alongside it.
#
# Standing settings: nosync=1, recreate_linop=0, bottom_solver=bicgstab (stock).

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"
EXE=../nodefd3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=../inputs
COMMON="nosync=1 recreate_linop=0 nsolves=20 nwarmup=3 linop_verbose=2"

# mcl 30 5 4 3 2  ->  7 6 5 4 3 MG levels
# ags -1 16 64    ->  agglomerate at MG level 2 / 3 / 1
MCLS="30 5 4 3 2"
AGSS="-1 16 64"

for REP in 1 2; do

  # ---- 1 GPU: mcl only; agglomeration is a no-op at one rank -------------
  for MCL in ${MCLS}; do
    srun -n 1 ${EXE} ${INPUTS} ${COMMON} max_coarsening_level=${MCL} \
         >& run-g1-mcl${MCL}-r${REP}.ou
  done

  # ---- 2 and 4 GPUs: full cross product ----------------------------------
  for NGPU in 2 4; do
    for MCL in ${MCLS}; do
      for AGS in ${AGSS}; do
        TAG=$( [ ${AGS} -lt 0 ] && echo def || echo ${AGS} )
        srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} \
             max_coarsening_level=${MCL} agg_grid_size=${AGS} \
             >& run-g${NGPU}-mcl${MCL}-ags${TAG}-r${REP}.ou
      done
    done
  done

done
