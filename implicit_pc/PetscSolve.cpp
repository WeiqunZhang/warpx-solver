#include "PetscSolve.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_SpMV.H>

#include <petscksp.h>
#if defined(AMREX_USE_CUDA)
#  include <petscdevice_cuda.h>
#elif defined(AMREX_USE_HIP)
#  include <petscdevice_hip.h>
#endif

#include <vector>

using namespace amrex;

namespace wpc {

void petsc_initialize ()
{
    // Raw PETSc options, e.g. petsc.options = "-ksp_monitor -log_view". Inserted
    // before PetscInitialize so that initialization options also work.
    ParmParse pp("petsc");
    std::string opts;
    if (pp.query("options", opts) && !opts.empty()) {
        PetscCallAbort(MPI_COMM_WORLD, PetscOptionsInsertString(nullptr, opts.c_str()));
    }
    PETSC_COMM_WORLD = ParallelDescriptor::Communicator();
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(nullptr, nullptr, nullptr, nullptr));
}

void petsc_finalize ()
{
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
}

struct PetscKSP::Impl
{
    Long nlocal = 0;
    Long nglobal = 0;
    bool coo = false;
    Mat A = nullptr; // shell
    Mat P = nullptr;
    Vec x = nullptr;
    Vec b = nullptr;
    KSP ksp = nullptr;
    SpMatrix<Real> const* Aop = nullptr;
    AlgVector<Real> xv, yv;
    int niters = 0;
    std::string reason;
    std::string pcdesc;
    // COO pattern
    PetscCount ncoo = -1;
    Gpu::DeviceVector<int> coo_off;
    Gpu::DeviceVector<Real> coo_val;
};

namespace {

// Vec device/host array access
PetscErrorCode getRead (Vec v, PetscScalar const** p)
{
    PetscFunctionBeginUser;
#if defined(AMREX_USE_CUDA)
    PetscCall(VecCUDAGetArrayRead(v, p));
#elif defined(AMREX_USE_HIP)
    PetscCall(VecHIPGetArrayRead(v, p));
#else
    PetscCall(VecGetArrayRead(v, p));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode restoreRead (Vec v, PetscScalar const** p)
{
    PetscFunctionBeginUser;
#if defined(AMREX_USE_CUDA)
    PetscCall(VecCUDARestoreArrayRead(v, p));
#elif defined(AMREX_USE_HIP)
    PetscCall(VecHIPRestoreArrayRead(v, p));
#else
    PetscCall(VecRestoreArrayRead(v, p));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode getWrite (Vec v, PetscScalar** p)
{
    PetscFunctionBeginUser;
#if defined(AMREX_USE_CUDA)
    PetscCall(VecCUDAGetArrayWrite(v, p));
#elif defined(AMREX_USE_HIP)
    PetscCall(VecHIPGetArrayWrite(v, p));
#else
    PetscCall(VecGetArrayWrite(v, p));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode restoreWrite (Vec v, PetscScalar** p)
{
    PetscFunctionBeginUser;
#if defined(AMREX_USE_CUDA)
    PetscCall(VecCUDARestoreArrayWrite(v, p));
#elif defined(AMREX_USE_HIP)
    PetscCall(VecHIPRestoreArrayWrite(v, p));
#else
    PetscCall(VecRestoreArrayWrite(v, p));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Stand-in for WarpX's JFNK shell (JacobianFunctionMF::apply): y = A x.
PetscErrorCode applyMatOp (Mat A, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    BL_PROFILE("PETSc::applyMatOp");
    PetscKSP::Impl* ctx = nullptr;
    PetscCall(MatShellGetContext(A, &ctx));
    PetscScalar const* px = nullptr;
    PetscScalar* py = nullptr;
    PetscCall(getRead(x, &px));
    PetscCall(getWrite(y, &py));
    Gpu::Device::synchronize(); // PETSc and AMReX use different streams
    auto const n = ctx->nlocal;
    Gpu::copyAsync(Gpu::deviceToDevice, px, px+n, ctx->xv.data());
    SpMV(ctx->yv, *ctx->Aop, ctx->xv);
    Gpu::copyAsync(Gpu::deviceToDevice, ctx->yv.data(), ctx->yv.data()+n, py);
    Gpu::Device::synchronize();
    PetscCall(restoreWrite(y, &py));
    PetscCall(restoreRead(x, &px));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// PETScSolver_impl::setOptions (WarpX_PETSc.cpp:239-312)
void setOptions ()
{
    PetscBool pc_specified;
    PetscOptionsHasName( NULL, NULL, "-pc_type", &pc_specified );
    if (pc_specified) { return; }

    const amrex::ParmParse pp_pc("pc_petsc");

    std::string pctype = "asm"; // default
    {
        std::string val = pctype;
        pp_pc.query("type", val);
        PetscOptionsSetValue( NULL, "-pc_type", val.c_str());
        pctype = val;
    }

    if (pctype == "asm") {
        {
            int val = 0;
            pp_pc.query("asm_overlap", val);
            std::string valstr = std::to_string(val);
            PetscOptionsSetValue( NULL, "-pc_asm_overlap", valstr.c_str());
        }
        std::string subpctype = "ilu"; // default
        {
            std::string val = subpctype;
            pp_pc.query("sub_type", val);
            PetscOptionsSetValue( NULL, "-sub_pc_type", val.c_str());
            subpctype = val;
        }
        if (subpctype == "ilu") {
            int val = 2;
            pp_pc.query("ilu_factor_levels", val);
            std::string valstr = std::to_string(val);
            PetscOptionsSetValue( NULL, "-sub_pc_factor_levels", valstr.c_str());
        }
    } else if (pctype == "hypre") {
        std::string hyprepctype = "euclid"; // default
        {
            std::string val = hyprepctype;
            pp_pc.query("hypre_type", val);
            PetscOptionsSetValue( NULL, "-pc_hypre_type", val.c_str());
            hyprepctype = val;
        }
        if (hyprepctype == "euclid") {
            int val = 2;
            pp_pc.query("euclid_factor_levels", val);
            std::string valstr = std::to_string(val);
            PetscOptionsSetValue( NULL, "-pc_hypre_euclid_level", valstr.c_str());
        }
    }
}

}

// KSP_impl::createObjects (WarpX_PETSc.cpp:399-496)
PetscKSP::PetscKSP (Long nlocal, Long nglobal, int restart, int verbose, bool coo)
    : m_impl(std::make_unique<Impl>())
{
    auto& d = *m_impl;
    d.nlocal = nlocal;
    d.nglobal = nglobal;
    d.coo = coo;
    auto const comm = PETSC_COMM_WORLD;

    PetscCallAbort(comm, VecCreate(comm, &d.x));
#if defined(AMREX_USE_CUDA)
    PetscCallAbort(comm, VecSetType(d.x, VECCUDA));
#elif defined(AMREX_USE_HIP)
    PetscCallAbort(comm, VecSetType(d.x, VECHIP));
#else
    PetscCallAbort(comm, VecSetType(d.x, VECSTANDARD));
#endif
    PetscCallAbort(comm, VecSetSizes(d.x, PetscInt(nlocal), PetscInt(nglobal)));
    PetscCallAbort(comm, VecSetFromOptions(d.x));
    PetscCallAbort(comm, VecDuplicate(d.x, &d.b));

    PetscCallAbort(comm, MatCreateShell(comm, PetscInt(nlocal), PetscInt(nlocal),
                                        PetscInt(nglobal), PetscInt(nglobal), &d, &d.A));
    PetscCallAbort(comm, MatShellSetOperation(d.A, MATOP_MULT, (void (*)(void))applyMatOp));
#if defined(AMREX_USE_CUDA)
    PetscCallAbort(comm, MatShellSetVecType(d.A, VECCUDA));
#elif defined(AMREX_USE_HIP)
    PetscCallAbort(comm, MatShellSetVecType(d.A, VECHIP));
#endif
    PetscCallAbort(comm, MatSetUp(d.A));

    PetscCallAbort(comm, KSPCreate(comm, &d.ksp));
    PC pc;
    PetscCallAbort(comm, KSPGetPC(d.ksp, &pc));
    PetscCallAbort(comm, KSPSetPCSide(d.ksp, PC_RIGHT));
    setOptions();
    PetscCallAbort(comm, PCSetFromOptions(pc));

    PetscCallAbort(comm, MatCreate(comm, &d.P));
    PetscCallAbort(comm, MatSetSizes(d.P, PetscInt(nlocal), PetscInt(nlocal), PETSC_DETERMINE, PETSC_DETERMINE));
#if defined(AMREX_USE_CUDA)
    PetscCallAbort(comm, MatSetType(d.P, MATAIJCUSPARSE));
#elif defined(AMREX_USE_HIP)
    PetscCallAbort(comm, MatSetType(d.P, MATAIJHIPSPARSE));
#else
    PetscCallAbort(comm, MatSetType(d.P, MATAIJ));
    if (!coo) {
        PetscCallAbort(comm, MatMPIAIJSetPreallocation(d.P, 1, NULL, 1, NULL));
    }
#endif
    if (!coo) {
        PetscCallAbort(comm, MatSetOption(d.P, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
        PetscCallAbort(comm, MatSetOption(d.P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        PetscCallAbort(comm, MatSetUp(d.P));
    }
    PetscCallAbort(comm, MatSetFromOptions(d.P));
    PetscCallAbort(comm, KSPSetOperators(d.ksp, d.A, d.P));
    PetscCallAbort(comm, KSPGMRESSetRestart(d.ksp, restart));
    PetscCallAbort(comm, KSPSetNormType(d.ksp, KSP_NORM_UNPRECONDITIONED));
    if (verbose > 1) {
        PetscCallAbort(comm, PetscOptionsSetValue(NULL, "-ksp_monitor_true_residual", NULL));
    }
    PetscCallAbort(comm, KSPSetFromOptions(d.ksp));

    PCType pctype;
    PetscCallAbort(comm, PCGetType(pc, &pctype));
    d.pcdesc = pctype ? pctype : "none";
    {
        char const* subtype = nullptr;
        PetscBool has = PETSC_FALSE;
        char buf[64];
        PetscCallAbort(comm, PetscOptionsGetString(NULL, NULL, "-sub_pc_type", buf, sizeof(buf), &has));
        if (has) { subtype = buf; d.pcdesc += std::string("+") + subtype; }
        PetscInt lev = 0;
        PetscCallAbort(comm, PetscOptionsGetInt(NULL, NULL, "-sub_pc_factor_levels", &lev, &has));
        if (has && subtype && std::string(subtype) == "ilu") { d.pcdesc += "(" + std::to_string(lev) + ")"; }
        PetscInt ov = 0;
        PetscCallAbort(comm, PetscOptionsGetInt(NULL, NULL, "-pc_asm_overlap", &ov, &has));
        if (has) { d.pcdesc += ",ov" + std::to_string(ov); }
    }
    d.pcdesc += coo ? ",coo" : ",setvalues";
}

PetscKSP::~PetscKSP ()
{
    auto& d = *m_impl;
    KSPDestroy(&d.ksp);
    MatDestroy(&d.P);
    MatDestroy(&d.A);
    VecDestroy(&d.x);
    VecDestroy(&d.b);
}

void PetscKSP::setOperator (SpMatrix<Real> const* A)
{
    auto& d = *m_impl;
    d.Aop = A;
    d.xv.define(A->partition());
    d.yv.define(A->partition());
}

// PETScSolver_impl::assemblePCMatrix (WarpX_PETSc.cpp:336-385)
void PetscKSP::assemble (MatrixPC const& pcmat)
{
    BL_PROFILE("PetscKSP::assemble");
    auto& d = *m_impl;
    auto const comm = PETSC_COMM_WORLD;
    PetscInt const n = pcmat.m_ndofs_l;
    PetscInt const ncols_max = pcmat.m_pc_mat_nnz;
    AMREX_ALWAYS_ASSERT(n == d.nlocal);

    if (!d.coo) {
        std::vector<int> h_r_indices_g(pcmat.m_r_indices_g.size());
        std::vector<int> h_n_nz_cols(pcmat.m_num_nz.size());
        std::vector<int> h_c_indices_g(pcmat.m_c_indices_g.size());
        std::vector<amrex::Real> h_a_ij(pcmat.m_a_ij.size());
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          pcmat.m_r_indices_g.begin(), pcmat.m_r_indices_g.end(),
                          h_r_indices_g.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          pcmat.m_num_nz.begin(), pcmat.m_num_nz.end(),
                          h_n_nz_cols.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          pcmat.m_c_indices_g.begin(), pcmat.m_c_indices_g.end(),
                          h_c_indices_g.begin() );
        amrex::Gpu::copy( amrex::Gpu::deviceToHost,
                          pcmat.m_a_ij.begin(), pcmat.m_a_ij.end(),
                          h_a_ij.begin() );
        for (int i = 0; i < n; i++) {
            PetscCallAbort(comm, MatSetValues( d.P,
                                               1,
                                               &h_r_indices_g[i],
                                               h_n_nz_cols[i],
                                               &h_c_indices_g[i*ncols_max],
                                               &h_a_ij[i*ncols_max],
                                               INSERT_VALUES ));
        }
        PetscCallAbort(comm, MatAssemblyBegin(d.P, MAT_FINAL_ASSEMBLY));
        PetscCallAbort(comm, MatAssemblyEnd(d.P, MAT_FINAL_ASSEMBLY));
    } else {
        if (d.ncoo < 0) {
            // Pattern from the first assembly; it is the same every Newton iteration.
            std::vector<int> h_r(n), h_nz(n), h_c(pcmat.m_c_indices_g.size());
            Gpu::copy(Gpu::deviceToHost, pcmat.m_r_indices_g.begin(), pcmat.m_r_indices_g.end(), h_r.begin());
            Gpu::copy(Gpu::deviceToHost, pcmat.m_num_nz.begin(), pcmat.m_num_nz.end(), h_nz.begin());
            Gpu::copy(Gpu::deviceToHost, pcmat.m_c_indices_g.begin(), pcmat.m_c_indices_g.end(), h_c.begin());
            std::vector<int> h_off(n+1, 0);
            for (int i = 0; i < n; ++i) { h_off[i+1] = h_off[i] + h_nz[i]; }
            d.ncoo = h_off[n];
            std::vector<PetscInt> ci(d.ncoo), cj(d.ncoo);
            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < h_nz[i]; ++k) {
                    ci[h_off[i]+k] = h_r[i];
                    cj[h_off[i]+k] = h_c[i*ncols_max+k];
                }
            }
            PetscCallAbort(comm, MatSetPreallocationCOO(d.P, d.ncoo, ci.data(), cj.data()));
            d.coo_off.resize(n+1);
            Gpu::copy(Gpu::hostToDevice, h_off.begin(), h_off.end(), d.coo_off.begin());
            d.coo_val.resize(d.ncoo);
        }
        auto const* off = d.coo_off.data();
        auto const* nz = pcmat.m_num_nz.data();
        auto const* a = pcmat.m_a_ij.data();
        auto* v = d.coo_val.data();
        ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
        {
            for (int k = 0; k < nz[i]; ++k) { v[off[i]+k] = a[i*ncols_max+k]; }
        });
        Gpu::streamSynchronize();
        PetscCallAbort(comm, MatSetValuesCOO(d.P, d.coo_val.data(), INSERT_VALUES));
    }
}

void PetscKSP::setup ()
{
    BL_PROFILE("PetscKSP::setup");
    auto& d = *m_impl;
    // PCSetUp redoes the setup when P changed; KSPSolve would otherwise do it.
    PC pc;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(d.ksp, &pc));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(d.ksp));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetUp(pc));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUpOnBlocks(d.ksp));
    Gpu::Device::synchronize();
}

// KSP_impl::solve (WarpX_PETSc.cpp:532-560)
void PetscKSP::solve (AlgVector<Real>& a_x, AlgVector<Real> const& a_b,
                      Real rtol, Real atol, int maxit)
{
    BL_PROFILE("PetscKSP::solve");
    auto& d = *m_impl;
    auto const comm = PETSC_COMM_WORLD;
    auto const n = d.nlocal;
    {
        PetscScalar* p = nullptr;
        PetscCallAbort(comm, getWrite(d.b, &p));
        Gpu::copyAsync(Gpu::deviceToDevice, a_b.data(), a_b.data()+n, p);
        Gpu::Device::synchronize();
        PetscCallAbort(comm, restoreWrite(d.b, &p));
    }
    PetscCallAbort(comm, VecSet(d.x, 0.0));
    PetscCallAbort(comm, KSPSetTolerances(d.ksp, rtol, atol, PETSC_CURRENT, maxit));
    PetscCallAbort(comm, KSPSolve(d.ksp, d.b, d.x));
    {
        PetscScalar const* p = nullptr;
        PetscCallAbort(comm, getRead(d.x, &p));
        Gpu::Device::synchronize();
        Gpu::copyAsync(Gpu::deviceToDevice, p, p+n, a_x.data());
        Gpu::Device::synchronize();
        PetscCallAbort(comm, restoreRead(d.x, &p));
    }
    PetscInt its;
    PetscCallAbort(comm, KSPGetIterationNumber(d.ksp, &its));
    d.niters = int(its);
    char const* conv_reason;
    PetscCallAbort(comm, KSPGetConvergedReasonString(d.ksp, &conv_reason));
    d.reason = conv_reason;
}

int PetscKSP::numIters () const { return m_impl->niters; }
std::string const& PetscKSP::reason () const { return m_impl->reason; }
std::string const& PetscKSP::pcDescription () const { return m_impl->pcdesc; }

}
