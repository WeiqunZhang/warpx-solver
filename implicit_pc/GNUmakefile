# WarpX implicit PC test: GMRES + AlgMG vs PETSc (ASM/LU/ILU) on WarpX's PC matrix.
#
# AMREX_HOME is set with '=' because the shell may export another tree; override
# it on the command line:  make AMREX_HOME=/path/to/amrex -j16
#
AMREX_HOME = ../../amg-amrex

DEBUG        = FALSE
DIM          = 3
COMP         = gnu
PRECISION    = DOUBLE

USE_MPI      = TRUE
USE_OMP      = FALSE
USE_CUDA     = FALSE

TINY_PROFILE = TRUE
BL_NO_FORT   = TRUE

USE_LINEAR_SOLVERS_EM     = FALSE
USE_LINEAR_SOLVERS_INCFLO = FALSE

# PETSc for the comparison (not AMReX's USE_PETSC interface).
USE_PETSC_TEST = TRUE
ifeq ($(USE_CUDA),TRUE)
  PETSC_DIR_TEST ?= $(HOME)/opt/petsc-3.26.0-cuda
else
  PETSC_DIR_TEST ?= $(HOME)/opt/petsc
endif

EBASE        = implicit_pc

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

ifeq ($(USE_PETSC_TEST),TRUE)
  DEFINES += -DUSE_PETSC_TEST
  INCLUDE_LOCATIONS += $(PETSC_DIR_TEST)/include
  LIBRARY_LOCATIONS += $(PETSC_DIR_TEST)/lib
  ifeq ($(USE_CUDA),TRUE)
    LIBRARIES += -Xlinker -rpath -Xlinker $(PETSC_DIR_TEST)/lib -lpetsc
  else
    LIBRARIES += -Wl,-rpath,$(PETSC_DIR_TEST)/lib -lpetsc
  endif
endif

Pdirs := Base LinearSolvers
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
