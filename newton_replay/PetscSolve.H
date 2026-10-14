// PETSc KSP path mirroring WarpX_PETSc.cpp: GMRES on a shell operator, right
// preconditioned by ASM + LU built from the dumped PC matrix.
#ifndef NR_PETSC_SOLVE_H_
#define NR_PETSC_SOLVE_H_

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace wpc {

//! Local rows of a CSR matrix on the host, global column indices
struct HostCSR
{
    amrex::Long row_begin = 0;           // global index of the first local row
    std::vector<std::int64_t> ptr;       // nlocal+1, starting at 0
    std::vector<std::int64_t> col;
    std::vector<amrex::Real> val;
};

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
    //! WarpX's assemblePCMatrix (host MatSetValues), or COO if coo.
    void assemble (HostCSR const& P);
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
