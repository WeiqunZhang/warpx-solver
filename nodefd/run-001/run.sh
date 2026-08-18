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

# run-001: agglomeration sweep at fixed 128^3.
#
# Requires the AMReX min-width patch (commit "mg min width = 4"): on GPU,
# mg_domain_min_width = mg_box_min_width = 4, giving 6 MG levels at 128^3 and
# letting the stock bicgstab bottom solver work.  Verify each .ou reports
# "MG levels : 6" before trusting the numbers.
#
# Standing settings (not swept): nosync=1, recreate_linop=0.  Both come from
# ../inputs; they are repeated on the command line only for provenance in the logs.
#
# max_coarsening_level is deliberately NOT swept -- more coarsening levels are
# preferred for solver stability.
#
# linop_verbose=2 records "agglomerated AMR level 0 starting at MG level N of M"
# in every log, which is what makes the results interpretable.

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"
EXE=../nodefd3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=../inputs
COMMON="nosync=1 recreate_linop=0 nsolves=10 nwarmup=2 linop_verbose=2"

# Arm A -- agglomeration ON, vary where it starts.
#   agg_grid_size:  8 -> MG lvl 4,  16 -> lvl 3,  32(default) -> lvl 2,  64 -> lvl 1
# Arm B -- agglomeration OFF, consolidation becomes the active mechanism.
#   (con_* are inert while agglomeration=1, so they are only swept here.)

for NGPU in 1 2 4; do

  # --- baseline: stock defaults -------------------------------------------
  srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} \
       >& run-g${NGPU}-base.ou

  # 1 GPU has no redistribution at all; baseline is the only meaningful point.
  if [ ${NGPU} -eq 1 ]; then continue; fi

  # --- Arm A: agglomeration on, sweep start level -------------------------
  for AGS in 8 16 64; do
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agg_grid_size=${AGS} \
         >& run-g${NGPU}-agg${AGS}.ou
  done

  # --- Arm B: agglomeration off, consolidation active ---------------------
  srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agglomeration=0 consolidation=0 \
       >& run-g${NGPU}-noagg-nocon.ou

  for CS in 1 2 3; do
    srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agglomeration=0 con_strategy=${CS} \
         >& run-g${NGPU}-noagg-cs${CS}.ou
  done

  srun -n ${NGPU} ${EXE} ${INPUTS} ${COMMON} agglomeration=0 con_ratio=4 \
       >& run-g${NGPU}-noagg-cr4.ou

done
