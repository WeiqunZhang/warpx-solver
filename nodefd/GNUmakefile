# Standalone MLMG + MLEBNodeFDLaplacian benchmark.
#
# The exported AMREX_HOME in this shell may point at a stale path, and "?="
# will NOT override an exported environment variable.  Pass it explicitly:
#
#   make AMREX_HOME=/home/wqzhang/git/amrex -j16
#
AMREX_HOME ?= ../amrex

DEBUG        = FALSE
DIM          = 3
COMP         = gnu
PRECISION    = DOUBLE

USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE

# CUDA_ARCH is deliberately NOT set here -- it would be silently overridden.
# Tools/GNUMake/sites/Make.unknown:96-101 auto-detects it from
# `nvidia-smi --query-gpu=compute_cap` (120 on this box), and
# sites/Make.nersc:71 forces 80 on Perlmutter.  Both use plain `=` and are
# included at Make.defs:1066-1071, i.e. after everything above.  To force a
# value, put it on the make command line or export AMREX_CUDA_ARCH.

USE_EB       = FALSE
USE_HYPRE    = FALSE

# MLEBNodeFDLaplacian lives behind USE_LINEAR_SOLVERS_EM.  INCFLO pulls in
# cell-centered solvers we do not use; turning it off cuts compile time.
USE_LINEAR_SOLVERS_EM     = TRUE
USE_LINEAR_SOLVERS_INCFLO = FALSE

# TINY_PROFILE also gives NVTX ranges automatically under USE_CUDA.
TINY_PROFILE = TRUE
BL_NO_FORT   = TRUE

EBASE        = nodefd

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

# LinearSolvers/MLMG/Make.package pulls in Src/Boundary itself.
Pdirs := Base Boundary LinearSolvers/MLMG
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
