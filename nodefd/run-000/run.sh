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

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"
EXE=../nodefd3d.gnu.TPROF.MPI.CUDA.ex
INPUTS=../inputs

NGPU=1
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=0 >& run-g${NGPU}-ns0-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=1 >& run-g${NGPU}-ns0-re1.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=0 >& run-g${NGPU}-ns1-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=1 >& run-g${NGPU}-ns1-re1.ou

NGPU=2
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=0 >& run-g${NGPU}-ns0-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=1 >& run-g${NGPU}-ns0-re1.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=0 >& run-g${NGPU}-ns1-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=1 >& run-g${NGPU}-ns1-re1.ou

NGPU=4
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=0 >& run-g${NGPU}-ns0-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=0 recreate_linop=1 >& run-g${NGPU}-ns0-re1.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=0 >& run-g${NGPU}-ns1-re0.ou
srun -n ${NGPU} ${EXE} ${INPUTS} nosync=1 recreate_linop=1 >& run-g${NGPU}-ns1-re1.ou
