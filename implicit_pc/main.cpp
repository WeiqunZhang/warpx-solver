// WarpX theta-implicit PC test: the matrix WarpX's MatrixPC hands to PETSc
// (I + alpha curl curl + mass matrices on the Yee grid), solved with GMRES
// preconditioned by AlgMG or by PETSc (ASM/LU/ILU, as in WarpX).
//
// Mimics the linear solves inside WarpX's Newton loop: nnewton times, the
// mass matrices change (same pattern), the PC matrix is reassembled, the PC
// is set up again, and GMRES solves with the operator A. A is the same
// matrix with wider mass-matrix stencils (width_op), standing in for WarpX's
// matrix-free Jacobian.

#include "WarpXMatrix.H"
#include "MassMatrix.H"
#ifdef USE_PETSC_TEST
#include "PetscSolve.H"
#endif

#include <AMReX.H>
#include <AMReX_AlgMG.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_SpMV.H>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

using namespace amrex;
using namespace wpc;

namespace {

struct Params
{
#if (AMREX_SPACEDIM == 3)
    IntVect n_cell{64,64,64};
    Array<Real,AMREX_SPACEDIM> prob_lo{0,0,0};
    Array<Real,AMREX_SPACEDIM> prob_hi{4.4e-3,4.4e-3,4.4e-3};
    Vector<std::string> bc_lo{"pec","periodic","periodic"};
    Vector<std::string> bc_hi{"pec","periodic","periodic"};
#else
    IntVect n_cell{512,512};
    Array<Real,AMREX_SPACEDIM> prob_lo{0,0};
    Array<Real,AMREX_SPACEDIM> prob_hi{3.52e-2,3.52e-2};
    Vector<std::string> bc_lo{"pec","periodic"};
    Vector<std::string> bc_hi{"pec","periodic"};
#endif
    IntVect max_grid_size{32};

    Real dt = 1.e-12;
    Real theta = 0.5;

    int shape = 2;
    int width_pc = 1;
    int width_op = 2;
    Real newton_density_step = 0.05;
    MassMatrixParams mm;

    std::string solver = "both";
    int nnewton = 3;
    int check = 1;

    Real rtol = 1.e-4;
    Real atol = 0;
    int max_iter = 1000;
    int restart = 30;
    int verbose = 0;

    int coo_assembly = 0;

    void read ()
    {
        {
            ParmParse pp("geometry");
            Vector<int> v;
            if (pp.queryarr("n_cell", v)) {
                AMREX_ALWAYS_ASSERT(int(v.size()) == AMREX_SPACEDIM);
                n_cell = IntVect(v.data());
            }
            Vector<int> m;
            if (pp.queryarr("max_grid_size", m)) {
                max_grid_size = (m.size() == 1) ? IntVect(m[0]) : IntVect(m.data());
            }
            Vector<Real> lo, hi;
            if (pp.queryarr("prob_lo", lo)) { std::copy(lo.begin(), lo.end(), prob_lo.begin()); }
            if (pp.queryarr("prob_hi", hi)) { std::copy(hi.begin(), hi.end(), prob_hi.begin()); }
        }
        {
            ParmParse pp("bc");
            pp.queryarr("lo", bc_lo);
            pp.queryarr("hi", bc_hi);
            AMREX_ALWAYS_ASSERT(int(bc_lo.size()) == AMREX_SPACEDIM && int(bc_hi.size()) == AMREX_SPACEDIM);
        }
        {
            ParmParse pp("implicit");
            pp.query("dt", dt);
            pp.query("theta", theta);
        }
        {
            ParmParse pp("mm");
            pp.query("shape", shape);
            AMREX_ALWAYS_ASSERT(shape == 2);
            pp.query("width_pc", width_pc);
            pp.query("width_op", width_op);
            pp.query("density(x,y,z)", mm.density);
            pp.query("noise", mm.noise);
            pp.query("seed", mm.seed);
            pp.query("newton_density_step", newton_density_step);
            Vector<Real> b0;
            if (pp.queryarr("B0", b0)) { std::copy(b0.begin(), b0.end(), mm.B0.begin()); }
            Vector<std::string> names{"electrons"};
            pp.queryarr("species", names);
            for (auto const& name : names) {
                Species s;
                s.name = name;
                if (name == "deuterium") { s.charge = 1; s.mass = 3670.48; }
                ParmParse pps("mm."+name);
                pps.query("charge", s.charge);
                pps.query("mass", s.mass);
                pps.query("density_scale", s.density_scale);
                mm.species.push_back(s);
            }
        }
        {
            ParmParse pp;
            pp.query("solver", solver);
            pp.query("nnewton", nnewton);
            pp.query("check", check);
        }
        {
            ParmParse pp("gmres");
            pp.query("rtol", rtol);
            pp.query("atol", atol);
            pp.query("max_iter", max_iter);
            pp.query("restart", restart);
            pp.query("verbose", verbose);
        }
        {
            ParmParse pp("petsc");
            pp.query("coo_assembly", coo_assembly);
        }
    }
};

FieldBoundaryType parseBC (std::string const& s)
{
    if (s == "periodic") { return FieldBoundaryType::Periodic; }
    if (s == "pec") { return FieldBoundaryType::PEC; }
    if (s == "pmc") { return FieldBoundaryType::PMC; }
    amrex::Abort("Unknown bc " + s);
    return FieldBoundaryType::PEC;
}

void applyAlgMGOptions (AlgMG<Real>& amg)
{
    ParmParse pp("algmg");
    int i;
    Real r;
    std::string s;
    if (pp.query("verbose", i)) { amg.setVerbose(i); }
    if (pp.query("max_levels", i)) { amg.setMaxLevels(i); }
    if (pp.query("max_coarse_size", i)) { amg.setMaxCoarseSize(i); }
    if (pp.query("theta", r)) { amg.setStrongThreshold(r); }
    if (pp.query("agg_levels", i)) { amg.setAggressiveNumLevels(i); }
    if (pp.query("agg_direct", i)) { amg.setAggressiveDirectInterp(i != 0); }
    if (pp.query("interp", s)) { amg.setInterpType(getEnumCaseInsensitive<AlgMGInterpType>(s)); }
    if (pp.query("pmax", i)) { amg.setPMaxElmts(i); }
    if (pp.query("trunc", r)) { amg.setTruncFactor(r); }
    if (pp.query("smoother", s)) { amg.setSmoother(getEnumCaseInsensitive<AlgMGSmoother>(s)); }
    if (pp.query("relax_weight", r)) { amg.setRelaxWeight(r); }
    if (pp.query("cheby_degree", i)) { amg.setChebyshevDegree(i); }
    if (pp.query("cheby_ratio", r)) { amg.setChebyshevRatio(r); }
    if (pp.query("nu1", i)) { amg.setPreSmooth(i); }
    if (pp.query("nu2", i)) { amg.setPostSmooth(i); }
    if (pp.query("nu_bottom", i)) { amg.setBottomSmooth(i); }
    if (pp.query("bottom", s)) { amg.setBottomSolver(getEnumCaseInsensitive<AlgMGBottomSolver>(s)); }
    if (pp.query("bottom_tol", r)) { amg.setBottomTol(r); }
}

double timer ()
{
    Gpu::Device::synchronize();
    ParallelDescriptor::Barrier();
    return amrex::second();
}

//! MatrixPC's padded arrays -> SpMatrix (padding dropped as invalid entries).
void toSpMatrix (MatrixPC const& m, AlgPartition const& part, SpMatrix<Real>& A)
{
    BL_PROFILE("toSpMatrix");
    Long const n = m.m_ndofs_l;
    int const nc = m.m_pc_mat_nnz;
    Gpu::DeviceVector<Long> cols(n*nc);
    Gpu::DeviceVector<Long> offs(n+1);
    Gpu::DeviceVector<Real> vals(n*nc);
    auto* pc = cols.data();
    auto* po = offs.data();
    auto* pv = vals.data();
    auto const* nz = m.m_num_nz.data();
    auto const* ci = m.m_c_indices_g.data();
    auto const* a = m.m_a_ij.data();
    ParallelFor(n+1, [=] AMREX_GPU_DEVICE (Long i)
    {
        po[i] = i*nc;
        if (i < n) {
            for (int k = 0; k < nc; ++k) {
                bool const valid = k < nz[i];
                pc[i*nc+k] = valid ? Long(ci[i*nc+k]) : Long(-1);
                pv[i*nc+k] = valid ? a[i*nc+k] : Real(0);
            }
        }
    });
    Gpu::streamSynchronize();
    A.define(part, pv, pc, n*nc, po, CsrSorted{false}, CsrValid{false});
}

void randomVector (AlgVector<Real>& v, int seed)
{
    auto const ib = v.globalBegin();
    auto* p = v.data();
    auto const s = std::uint64_t(seed);
    ParallelFor(v.numLocalRows(), [=] AMREX_GPU_DEVICE (Long i)
    {
        p[i] = hash_uniform(std::uint64_t(ib+i), s*0x9e3779b97f4a7c15ULL + 12345);
    });
    Gpu::streamSynchronize();
}

Real trueRelResidual (SpMatrix<Real> const& A, AlgVector<Real> const& x, AlgVector<Real> const& b)
{
    AlgVector<Real> r(A.partition());
    SpMV(r, A, x);
    Axpy(r, Real(-1), b);
    return r.norm2() / b.norm2();
}

// ---- self-checks ----

void vecToFields (AlgVector<Real> const& x, WarpXSolverDOF const& dofs,
                  Array<MultiFab,3>& E, Periodicity const& period)
{
    auto const* px = x.data();
    for (int dir = 0; dir < 3; ++dir) {
        E[dir].setVal(0.0);
        auto const& ea = E[dir].arrays();
        auto const& da = dofs.m_array[dir]->const_arrays();
        ParallelFor(E[dir], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            int const lid = da[b](i,j,k,0);
            if (lid >= 0) { ea[b](i,j,k) = px[lid]; }
        });
        E[dir].FillBoundaryAndSync(period);
    }
}

void fieldsToVec (Array<MultiFab,3> const& F, WarpXSolverDOF const& dofs, AlgVector<Real>& y)
{
    auto* py = y.data();
    for (int dir = 0; dir < 3; ++dir) {
        auto const& fa = F[dir].const_arrays();
        auto const& da = dofs.m_array[dir]->const_arrays();
        ParallelFor(F[dir], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            int const lid = da[b](i,j,k,0);
            if (lid >= 0) { py[lid] = fa[b](i,j,k); }
        });
    }
    Gpu::streamSynchronize();
}

//! Matrix-free I + alpha curl(curl E) + M E with two discrete curls; PEC
//! zeroes the derivative terms normal to the wall at tangential wall points
//! (what WarpX's masks do). PEC and periodic only.
void applyReference (Array<MultiFab,3>& out, Array<MultiFab,3> const& E,
                     Array<MultiFab,3> const& sigma,
                     GpuArray<GpuArray<int,3>,3> const& MM_ncomp,
                     Real alpha, Geometry const& geom,
                     Array<FieldBoundaryType,AMREX_SPACEDIM> const& bclo,
                     Array<FieldBoundaryType,AMREX_SPACEDIM> const& bchi)
{
    BoxArray const cba = amrex::convert(E[0].boxArray(), IntVect(0));
    auto const& dm = E[0].DistributionMap();
    auto const dxi = geom.InvCellSizeArray();
    Box const ndomain = amrex::surroundingNodes(geom.Domain());
    GpuArray<int,AMREX_SPACEDIM> pec_lo{}, pec_hi{};
    for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) {
        pec_lo[sd] = (bclo[sd] == FieldBoundaryType::PEC) ? ndomain.smallEnd(sd) : std::numeric_limits<int>::lowest();
        pec_hi[sd] = (bchi[sd] == FieldBoundaryType::PEC) ? ndomain.bigEnd(sd) : std::numeric_limits<int>::max();
    }
    // Mask for the derivative along sd at index ii (nodal in sd).
    auto mk = [=] AMREX_GPU_DEVICE (int sd, int ii) -> Real {
        return (ii == pec_lo[sd] || ii == pec_hi[sd]) ? 0.0_rt : 1.0_rt;
    };

    Array<MultiFab,3> B;
#if (AMREX_SPACEDIM == 3)
    IntVect const bt[3] = {IntVect(1,0,0), IntVect(0,1,0), IntVect(0,0,1)};
#else
    IntVect const bt[3] = {IntVect(1,0), IntVect(0,0), IntVect(0,1)};
#endif
    for (int d = 0; d < 3; ++d) {
        B[d].define(amrex::convert(cba, bt[d]), dm, 1, 1);
    }
    for (MFIter mfi(B[0]); mfi.isValid(); ++mfi) {
        auto const ex = E[0].const_array(mfi);
        auto const ey = E[1].const_array(mfi);
        auto const ez = E[2].const_array(mfi);
        auto const bx = B[0].array(mfi);
        auto const by = B[1].array(mfi);
        auto const bz = B[2].array(mfi);
#if (AMREX_SPACEDIM == 3)
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bx(i,j,k) = (ez(i,j+1,k)-ez(i,j,k))*dxi[1] - (ey(i,j,k+1)-ey(i,j,k))*dxi[2];
        });
        ParallelFor(B[1][mfi].box(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            by(i,j,k) = (ex(i,j,k+1)-ex(i,j,k))*dxi[2] - (ez(i+1,j,k)-ez(i,j,k))*dxi[0];
        });
        ParallelFor(B[2][mfi].box(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bz(i,j,k) = (ey(i+1,j,k)-ey(i,j,k))*dxi[0] - (ex(i,j+1,k)-ex(i,j,k))*dxi[1];
        });
#else
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bx(i,j,k) = -(ey(i,j+1,k)-ey(i,j,k))*dxi[1];
        });
        ParallelFor(B[1][mfi].box(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            by(i,j,k) = (ex(i,j+1,k)-ex(i,j,k))*dxi[1] - (ez(i+1,j,k)-ez(i,j,k))*dxi[0];
        });
        ParallelFor(B[2][mfi].box(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bz(i,j,k) = (ey(i+1,j,k)-ey(i,j,k))*dxi[0];
        });
#endif
    }

    for (int dir = 0; dir < 3; ++dir) {
        out[dir].define(E[dir].boxArray(), dm, 1, 0);
        auto const nc = MM_ncomp[dir];
        for (MFIter mfi(out[dir]); mfi.isValid(); ++mfi) {
            Box const vbx = mfi.validbox();
            Box const full_bx = E[dir][mfi].box();
            auto const o = out[dir].array(mfi);
            auto const e = E[dir].const_array(mfi);
            auto const s = sigma[dir].const_array(mfi);
            auto const bx = B[0].const_array(mfi);
            auto const by = B[1].const_array(mfi);
            auto const bz = B[2].const_array(mfi);
            ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real cc = 0;
#if (AMREX_SPACEDIM == 3)
                if (dir == 0) {
                    cc = mk(1,j)*(bz(i,j,k)-bz(i,j-1,k))*dxi[1] - mk(2,k)*(by(i,j,k)-by(i,j,k-1))*dxi[2];
                } else if (dir == 1) {
                    cc = mk(2,k)*(bx(i,j,k)-bx(i,j,k-1))*dxi[2] - mk(0,i)*(bz(i,j,k)-bz(i-1,j,k))*dxi[0];
                } else {
                    cc = mk(0,i)*(by(i,j,k)-by(i-1,j,k))*dxi[0] - mk(1,j)*(bx(i,j,k)-bx(i,j-1,k))*dxi[1];
                }
#else
                if (dir == 0) {
                    cc = -mk(1,j)*(by(i,j,k)-by(i,j-1,k))*dxi[1];
                } else if (dir == 1) {
                    cc = mk(1,j)*(bx(i,j,k)-bx(i,j-1,k))*dxi[1] - mk(0,i)*(bz(i,j,k)-bz(i-1,j,k))*dxi[0];
                } else {
                    cc = mk(0,i)*(by(i,j,k)-by(i-1,j,k))*dxi[0];
                }
#endif
                Real m = 0;
                int mm_comp = 0;
                for (int c2 = 0; c2 < nc[2]; ++c2) {
                for (int c1 = 0; c1 < nc[1]; ++c1) {
                for (int c0 = 0; c0 < nc[0]; ++c0) {
                    IntVect const iv(AMREX_D_DECL(i + c0-(nc[0]-1)/2,
                                                  j + c1-(nc[1]-1)/2,
                                                  k + c2-(nc[2]-1)/2));
                    if (full_bx.contains(iv)) { m += s(i,j,k,mm_comp)*e(iv); }
                    ++mm_comp;
                }}}
                o(i,j,k) = e(i,j,k) + alpha*cc + m;
            });
        }
    }
    Gpu::streamSynchronize();
}

void runChecks (MatrixPC const& mat, Array<MultiFab,3> const& sigma,
                GpuArray<GpuArray<int,3>,3> const& MM_ncomp, char const* name,
                WarpXSolverDOF const& dofs, AlgPartition const& part,
                Array<MultiFab,3>& E, Geometry const& geom, Real alpha,
                Array<FieldBoundaryType,AMREX_SPACEDIM> const& bclo,
                Array<FieldBoundaryType,AMREX_SPACEDIM> const& bchi)
{
    SpMatrix<Real> A;
    toSpMatrix(mat, part, A);
    AlgVector<Real> x(part), y(part), Ax(part), Ay(part), ref(part);
    randomVector(x, 777);
    randomVector(y, 999);
    SpMV(Ax, A, x);
    SpMV(Ay, A, y);
    Real const xAy = Dot(x, Ay);
    Real const yAx = Dot(y, Ax);
    amrex::Print() << "  check " << name << ": nnz/row max " << mat.m_pc_mat_nnz
                   << ", total nnz " << [&] { Long n = A.numLocalNonZeros();
                                              ParallelDescriptor::ReduceLongSum(n); return n; }()
                   << ", |x'Ay - y'Ax|/|x'Ay| = " << std::abs(xAy-yAx)/std::abs(xAy) << "\n";

    bool pec_or_periodic = true;
    for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) {
        if (bclo[sd] == FieldBoundaryType::PMC || bchi[sd] == FieldBoundaryType::PMC) { pec_or_periodic = false; }
    }
    if (!pec_or_periodic) {
        amrex::Print() << "  check " << name << ": matrix-free comparison skipped (PMC)\n";
        return;
    }
    vecToFields(x, dofs, E, geom.periodicity());
    Array<MultiFab,3> out;
    applyReference(out, E, sigma, MM_ncomp, alpha, geom, bclo, bchi);
    fieldsToVec(out, dofs, ref);
    Real const refnorm = ref.norminf();
    Axpy(ref, Real(-1), Ax);
    Real const err = ref.norminf() / refnorm;
    amrex::Print() << "  check " << name << ": |A x - ref|_inf / |ref|_inf = " << err
                   << (err < 1.e-12 ? "  PASS" : "  FAIL") << "\n";
}

struct Record
{
    std::string solver;
    int newton = 0;
    double assemble = 0; // conversion to the solver's matrix
    double setup = 0;
    double solve = 0;
    int iters = 0;
    Real relres = 0;
};

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");
        Params p;
        p.read();

        bool const do_algmg = (p.solver == "algmg" || p.solver == "both");
        bool const do_petsc = (p.solver == "petsc" || p.solver == "both");
#ifdef USE_PETSC_TEST
        if (do_petsc) { petsc_initialize(); }
#else
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!do_petsc, "Built without PETSc (USE_PETSC_TEST=FALSE)");
#endif
        {
            Array<FieldBoundaryType,AMREX_SPACEDIM> bclo, bchi;
            Array<int,AMREX_SPACEDIM> is_periodic{};
            for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) {
                bclo[sd] = parseBC(p.bc_lo[sd]);
                bchi[sd] = parseBC(p.bc_hi[sd]);
                AMREX_ALWAYS_ASSERT((bclo[sd] == FieldBoundaryType::Periodic) ==
                                    (bchi[sd] == FieldBoundaryType::Periodic));
                is_periodic[sd] = (bclo[sd] == FieldBoundaryType::Periodic);
            }
            Box const domain(IntVect(0), p.n_cell-1);
            Geometry const geom(domain, RealBox(p.prob_lo, p.prob_hi), CoordSys::cartesian, is_periodic);
            BoxArray ba(domain);
            ba.maxSize(p.max_grid_size);
            DistributionMapping const dm(ba);

            int const ng = std::max({2, p.width_pc, p.width_op});
            Array<MultiFab,3> E;
            for (int dir = 0; dir < 3; ++dir) {
                E[dir].define(amrex::convert(ba, EIndexType(dir)), dm, 1, ng);
            }

            WarpXSolverDOF dofs;
            dofs.Define(E, geom.periodicity());

            Array<MultiFab,3> curl2_BC_mask;
            InitializeCurlCurlBCMasks(curl2_BC_mask, E, geom, bclo, bchi);

            Vector<Long> rows(ParallelDescriptor::NProcs()+1, 0);
            {
                Vector<Long> nl(ParallelDescriptor::NProcs());
#ifdef AMREX_USE_MPI
                MPI_Allgather(&dofs.m_nDoFs_l, 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                              nl.data(), 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                              ParallelDescriptor::Communicator());
#else
                nl[0] = dofs.m_nDoFs_l;
#endif
                for (int i = 0; i < int(nl.size()); ++i) { rows[i+1] = rows[i] + nl[i]; }
            }
            AlgPartition const part(rows);

            Real const thetaDt = p.theta*p.dt;
            Real const alpha = (thetaDt*PhysConst::c)*(thetaDt*PhysConst::c);
            auto const nc_pc = MassMatricesNComp(p.width_pc, p.shape);
            auto const nc_op = MassMatricesNComp(p.width_op, p.shape);

            {
                auto const dxi = geom.InvCellSizeArray();
                amrex::Print() << "Domain " << domain << ", " << ba.size() << " boxes, "
                               << ParallelDescriptor::NProcs() << " ranks\n"
                               << "alpha/dx^2 =";
                for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) { amrex::Print() << " " << alpha*dxi[sd]*dxi[sd]; }
                amrex::Print() << ", width_pc " << p.width_pc << ", width_op " << p.width_op
                               << ", noise " << p.mm.noise << "\n";
            }

            MatrixPC Ppc, Aop;
            Array<MultiFab,3> sig_pc, sig_op;
            for (auto* m : {&Ppc, &Aop}) {
                m->m_dofs = &dofs;
                m->m_geom = geom;
                m->m_curl2_BC_mask = &curl2_BC_mask;
                m->m_thetaDt = thetaDt;
            }
            Ppc.m_bcoefs = &sig_pc;
            Ppc.m_MM_ncomp = nc_pc;
            Aop.m_bcoefs = &sig_op;
            Aop.m_MM_ncomp = nc_op;

#ifdef USE_PETSC_TEST
            std::unique_ptr<PetscKSP> petsc;
#endif
            Vector<Record> records;
            Vector<double> t_mm, t_asm;

            for (int k = 0; k < p.nnewton; ++k) {
                Real const dfac = 1.0_rt + p.newton_density_step*Real(k);
                int const seed = p.mm.seed + k;

                double t0 = timer();
                FillMassMatrices(sig_pc, nc_pc, E, geom, p.mm, p.dt, p.theta, dfac, seed);
                FillMassMatrices(sig_op, nc_op, E, geom, p.mm, p.dt, p.theta, dfac, seed);
                double t1 = timer();
                Ppc.Update(); // MatrixPC::Assemble, the same for both solvers
                double t2 = timer();
                Aop.Update();
                t_mm.push_back(t1-t0);
                t_asm.push_back(t2-t1);

                SpMatrix<Real> A;
                toSpMatrix(Aop, part, A);
                AlgVector<Real> b(part);
                randomVector(b, 100+k);

                if (k == 0) {
                    Real smax = 0;
                    for (int dir = 0; dir < 3; ++dir) {
                        int const ctr = (nc_pc[dir][0]*nc_pc[dir][1]*nc_pc[dir][2]-1)/2;
                        smax = std::max(smax, sig_pc[dir].max(ctr));
                    }
                    amrex::Print() << "max mass-matrix center = " << smax
                                   << ", rows " << dofs.m_nDoFs_g
                                   << ", nnz/row max: PC " << Ppc.m_pc_mat_nnz
                                   << ", op " << Aop.m_pc_mat_nnz << "\n";
                    if (p.check) {
                        runChecks(Ppc, sig_pc, nc_pc, "PC", dofs, part, E, geom, alpha, bclo, bchi);
                        runChecks(Aop, sig_op, nc_op, "op", dofs, part, E, geom, alpha, bclo, bchi);
                    }
                }

                if (do_algmg) {
                    BL_PROFILE_REGION("AlgMG");
                    Record rec{"AlgMG", k};
                    double ta = timer();
                    SpMatrix<Real> Psp;
                    toSpMatrix(Ppc, part, Psp);
                    double tb = timer();
                    AlgMG<Real> amg(Psp);
                    applyAlgMGOptions(amg);
                    amg.setup();
                    double tc = timer();
                    GMRES_MV<Real> gm(&A);
                    gm.setPrecond([&] (AlgVector<Real>& x, AlgVector<Real> const& r) { amg.precond(x, r); });
                    gm.getGMRES().setRestartLength(p.restart);
                    gm.getGMRES().setMaxIters(p.max_iter);
                    gm.setVerbose(p.verbose);
                    AlgVector<Real> x(part);
                    x.setVal(0);
                    gm.solve(x, b, p.rtol, p.atol);
                    double td = timer();
                    rec.assemble = tb-ta;
                    rec.setup = tc-tb;
                    rec.solve = td-tc;
                    rec.iters = gm.getGMRES().getNumIters();
                    rec.relres = trueRelResidual(A, x, b);
                    if (k == 0) {
                        Long nnz = 0, nnz0 = Psp.numLocalNonZeros();
                        for (int lev = 0; lev < amg.numLevels(); ++lev) {
                            nnz += amg.getMatrix(lev).numLocalNonZeros();
                        }
                        ParallelDescriptor::ReduceLongSum(nnz);
                        ParallelDescriptor::ReduceLongSum(nnz0);
                        amrex::Print() << "AlgMG: " << amg.numLevels() << " levels, operator complexity "
                                       << double(nnz)/double(nnz0) << "\n";
                    }
                    records.push_back(rec);
                }

#ifdef USE_PETSC_TEST
                if (do_petsc) {
                    BL_PROFILE_REGION("PETSc");
                    if (!petsc) {
                        petsc = std::make_unique<PetscKSP>(dofs.m_nDoFs_l, dofs.m_nDoFs_g, p.restart,
                                                           p.verbose, p.coo_assembly != 0);
                    }
                    Record rec{"PETSc", k};
                    petsc->setOperator(&A);
                    double ta = timer();
                    petsc->assemble(Ppc);
                    double tb = timer();
                    petsc->setup();
                    double tc = timer();
                    AlgVector<Real> x(part);
                    petsc->solve(x, b, p.rtol, p.atol, p.max_iter);
                    double td = timer();
                    rec.assemble = tb-ta;
                    rec.setup = tc-tb;
                    rec.solve = td-tc;
                    rec.iters = petsc->numIters();
                    rec.relres = trueRelResidual(A, x, b);
                    if (k == 0) {
                        amrex::Print() << "PETSc PC: " << petsc->pcDescription() << "\n";
                    }
                    if (petsc->reason().find("CONVERGED") == std::string::npos) {
                        amrex::Print() << "PETSc KSP: " << petsc->reason() << "\n";
                    }
                    records.push_back(rec);
                }
#endif
            }

            amrex::Print() << "\nnewton  mass+MatrixPC::Assemble (common)\n";
            for (int k = 0; k < p.nnewton; ++k) {
                amrex::Print() << "  " << k << "     " << t_mm[k] << " + " << t_asm[k] << "\n";
            }
            amrex::Print() << "\nsolver newton   convert     setup     solve     total  iters   rel.res\n";
            for (auto const& r : records) {
                amrex::Print() << std::left << std::setw(6) << r.solver << std::right
                               << std::setw(7) << r.newton << std::fixed << std::setprecision(4)
                               << std::setw(10) << r.assemble << std::setw(10) << r.setup
                               << std::setw(10) << r.solve
                               << std::setw(10) << r.assemble+r.setup+r.solve
                               << std::setw(7) << r.iters << std::scientific << std::setprecision(2)
                               << std::setw(10) << r.relres << std::defaultfloat << "\n";
            }
        }
#ifdef USE_PETSC_TEST
        if (do_petsc) { petsc_finalize(); }
#endif
    }
    amrex::Finalize();
}
