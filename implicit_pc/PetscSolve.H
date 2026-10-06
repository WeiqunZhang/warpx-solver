// PETSc KSP path mirroring WarpX_PETSc.cpp: GMRES on a shell operator, right
// preconditioned by a PETSc PC built from the assembled MatrixPC arrays.
#ifndef WPC_PETSC_SOLVE_H_
#define WPC_PETSC_SOLVE_H_

#include "WarpXMatrix.H"

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

#include <memory>
#include <string>

namespace wpc {

void petsc_initialize ();
void petsc_finalize ();

class PetscKSP
{
public:
    //! A is the operator applied by the shell matrix; it can be reset per solve.
    PetscKSP (amrex::Long nlocal, amrex::Long nglobal, int restart, int verbose, bool coo);
    ~PetscKSP ();
    PetscKSP (PetscKSP const&) = delete;
    PetscKSP& operator= (PetscKSP const&) = delete;

    void setOperator (amrex::SpMatrix<amrex::Real> const* A);
    //! WarpX's assemblePCMatrix (host MatSetValues), or device COO if coo.
    void assemble (MatrixPC const& P);
    //! KSPSetUp + KSPSetUpOnBlocks (PC setup and sub-block factorization).
    void setup ();
    void solve (amrex::AlgVector<amrex::Real>& x, amrex::AlgVector<amrex::Real> const& b,
                amrex::Real rtol, amrex::Real atol, int maxit);

    [[nodiscard]] int numIters () const;
    [[nodiscard]] std::string const& reason () const;
    [[nodiscard]] std::string const& pcDescription () const;

    struct Impl;
private:
    std::unique_ptr<Impl> m_impl;
};

}

#endif
