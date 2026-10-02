# Standalone 3D EB/MLMG test for the rlehe_cylinder WarpX case.
#
# Pass AMREX_HOME explicitly if the environment points elsewhere:
#
#   make AMREX_HOME=/home/wqzhang/git/amrex -j16
#
AMREX_HOME ?= ../../amrex

DEBUG        = FALSE
DIM          = 3
COMP         = gnu
PRECISION    = DOUBLE

USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE

USE_EB       = TRUE
USE_HYPRE    = FALSE
USE_PETSC    = FALSE

# MLEBNodeFDLaplacian is part of the EM linear-solver package.
USE_LINEAR_SOLVERS_EM     = TRUE
USE_LINEAR_SOLVERS_INCFLO = FALSE

TINY_PROFILE = TRUE
BL_NO_FORT   = TRUE

EBASE        = cylinder

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs := Base Boundary AmrCore EB LinearSolvers/MLMG
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
