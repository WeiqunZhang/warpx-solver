#!/bin/bash
#SBATCH --account=m4546
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"
EXE=./cylinder3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=./inputs

NGPU=1
srun -n ${NGPU} ${EXE} ${INPUTS} amr.n_cell="256 256 128" > run-${NGPU}.ou

NGPU=4
srun -n ${NGPU} ${EXE} ${INPUTS} amr.n_cell="512 512 128" > run-${NGPU}.ou
