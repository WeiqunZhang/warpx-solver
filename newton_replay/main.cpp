// Replay the linear systems of WarpX's Newton solver outside WarpX.
//
// WarpX (with newton.dump_linear_systems = 1) writes, per Newton iteration,
// the Jacobian A and the preconditioner matrix P as CSR, the right-hand side
// b and its own solution x. Here each system is solved with amrex::GMRES
// preconditioned by AlgMG on P (or on A), or with PETSc's KSP and PC as in
// WarpX's pc_petsc (ASM + LU/ILU, ...), so solver settings can be iterated on
// in seconds instead of rebuilding and rerunning WarpX.
//
// File layouts: vectors are int64 n, then n doubles. CSR is int64 n, int64
// nnz, int64 row_ptr[n+1], int64 col[nnz], double val[nnz]. Rows are split
// evenly over the MPI ranks.

#include <AMReX.H>
#include <AMReX_AlgMG.H>
#include <AMReX_AlgVector.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_SpMV.H>
#include <AMReX_SpMatrix.H>

#ifdef USE_PETSC_TEST
#include "PetscSolve.H"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <string>
#include <vector>

using namespace amrex;

#ifndef USE_PETSC_TEST
namespace wpc {
struct HostCSR
{
    Long row_begin = 0;
    std::vector<std::int64_t> ptr, col;
    std::vector<Real> val;
};
}
#endif

namespace {

double timer ()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <class T>
void readRaw (std::ifstream& ifs, T* p, std::int64_t n)
{
    ifs.read(reinterpret_cast<char*>(p), std::streamsize(n*std::int64_t(sizeof(T))));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ifs.good(), "short read");
}

std::map<std::string,std::string> readInfo (std::string const& fname)
{
    std::map<std::string,std::string> m;
    std::ifstream ifs(fname);
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string k, v;
        if (iss >> k >> v) { m[k] = v; }
    }
    return m;
}

//! Rows [lo,hi) of the whole vector on this rank
void readVector (std::string const& fname, AlgVector<Real>& v)
{
    std::ifstream ifs(fname, std::ios::binary);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ifs.good(), "cannot open " + fname);
    std::int64_t n;
    readRaw(ifs, &n, 1);
    AMREX_ALWAYS_ASSERT(n == v.partition().numGlobalRows());
    Long const lo = v.globalBegin();
    Long const nl = v.numLocalRows();
    ifs.seekg(std::streamoff(lo*Long(sizeof(Real))), std::ios::cur);
    std::vector<Real> h(nl);
    readRaw(ifs, h.data(), nl);
    Gpu::copy(Gpu::hostToDevice, h.begin(), h.end(), v.data());
}

//! Local rows of a CSR file into a SpMatrix; the host copy is returned too
std::int64_t readMatrix (std::string const& fname, AlgPartition const& part, SpMatrix<Real>& A,
                         wpc::HostCSR* host = nullptr)
{
    std::ifstream ifs(fname, std::ios::binary);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ifs.good(), "cannot open " + fname);
    std::int64_t n, nnz;
    readRaw(ifs, &n, 1);
    readRaw(ifs, &nnz, 1);
    AMREX_ALWAYS_ASSERT(n == part.numGlobalRows());
    int const rank = ParallelDescriptor::MyProc();
    Long const lo = part[rank];
    Long const hi = part[rank+1];
    Long const nl = hi - lo;
    std::vector<std::int64_t> ptr(nl+1);
    ifs.seekg(std::streamoff(lo*8), std::ios::cur);
    readRaw(ifs, ptr.data(), nl+1);
    std::int64_t const e0 = ptr[0], e1 = ptr[nl];
    Long const nnzl = e1 - e0;
    std::vector<std::int64_t> col(nnzl);
    std::vector<Real> val(nnzl);
    // skip the rest of row_ptr, then the first e0 columns
    ifs.seekg(std::streamoff((n - hi)*8 + e0*8), std::ios::cur);
    readRaw(ifs, col.data(), nnzl);
    ifs.seekg(std::streamoff((nnz - e1)*8 + e0*std::int64_t(sizeof(Real))), std::ios::cur);
    readRaw(ifs, val.data(), nnzl);
    for (auto& p : ptr) { p -= e0; }
    if (host) {
        host->row_begin = lo;
        host->ptr = ptr;
        host->col = col;
        host->val = val;
    }

    Gpu::DeviceVector<Long> dptr(nl+1), dcol(nnzl);
    Gpu::DeviceVector<Real> dval(nnzl);
    Gpu::copy(Gpu::hostToDevice, ptr.begin(), ptr.end(), dptr.begin());
    Gpu::copy(Gpu::hostToDevice, col.begin(), col.end(), dcol.begin());
    Gpu::copy(Gpu::hostToDevice, val.begin(), val.end(), dval.begin());
    Gpu::streamSynchronize();
    A.define(part, dval.data(), dcol.data(), nnzl, dptr.data(), CsrSorted{true}, CsrValid{true});
    return nnz;
}

void applyAlgMGOptions (AlgMG<Real>& amg)
{
    ParmParse pp("algmg");
    int i; Real r; std::string s;
    if (pp.query("verbose", i)) { amg.setVerbose(i); }
    if (pp.query("max_levels", i)) { amg.setMaxLevels(i); }
    if (pp.query("max_coarse_size", i)) { amg.setMaxCoarseSize(i); }
    if (pp.query("strong_threshold", r)) { amg.setStrongThreshold(r); }
    if (pp.query("agg_levels", i)) { amg.setAggressiveNumLevels(i); }
    if (pp.query("agg_direct", i)) { amg.setAggressiveDirectInterp(i != 0); }
    if (pp.query("interp", s)) { amg.setInterpType(getEnumCaseInsensitive<AlgMGInterpType>(s)); }
    if (pp.query("pmax", i)) { amg.setPMaxElmts(i); }
    if (pp.query("trunc", r)) { amg.setTruncFactor(r); }
    if (pp.query("smoother", s)) { amg.setSmoother(getEnumCaseInsensitive<AlgMGSmoother>(s)); }
    if (pp.query("relax_weight", r)) { amg.setRelaxWeight(r); }
    if (pp.query("chebyshev_degree", i)) { amg.setChebyshevDegree(i); }
    if (pp.query("chebyshev_ratio", r)) { amg.setChebyshevRatio(r); }
    if (pp.query("nu1", i)) { amg.setPreSmooth(i); }
    if (pp.query("nu2", i)) { amg.setPostSmooth(i); }
    if (pp.query("nu_bottom", i)) { amg.setBottomSmooth(i); }
    if (pp.query("bottom", s)) { amg.setBottomSolver(getEnumCaseInsensitive<AlgMGBottomSolver>(s)); }
    if (pp.query("bottom_tol", r)) { amg.setBottomTol(r); }
}

Real relResidual (SpMatrix<Real> const& A, AlgVector<Real> const& x, AlgVector<Real> const& b)
{
    AlgVector<Real> r(A.partition());
    computeResidual(r, A, x, b);
    return r.norm2() / b.norm2();
}

Real relDiff (AlgVector<Real> const& x, AlgVector<Real> const& y)
{
    AlgVector<Real> d(x.partition());
    d.copy(x);
    Axpy(d, Real(-1), y);
    return d.norm2() / y.norm2();
}

struct Record
{
    std::string name;
    int warpx_iters = 0;
    int iters = 0;
    double setup = 0, solve = 0;
    Real relres = 0, relres_warpx = 0, diff = 0;
    int levels = 0;
    double complexity = 0;
};

}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");
        ParmParse pp;
        std::string dump_dir = "linsys";
        pp.query("dump_dir", dump_dir);
        std::vector<std::string> systems;
        pp.queryarr("systems", systems);
        if (systems.empty()) {
            for (auto const& e : std::filesystem::directory_iterator(dump_dir)) {
                if (e.is_directory() && std::filesystem::exists(e.path() / "A.bin")) {
                    systems.push_back(e.path().filename().string());
                }
            }
            std::sort(systems.begin(), systems.end());
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!systems.empty(), "no systems found in " + dump_dir);

        std::string pc = "algmg";     // algmg | algmg_on_jacobian | none | petsc
        int pc_max_iter = 1;          // V-cycles per application (pc_algmg.max_iter)
        int nrepeat = 1;
        pp.query("pc", pc);
        pp.query("pc_max_iter", pc_max_iter);
        pp.query("nrepeat", nrepeat);

        Real rtol = -1, atol = 0;
        int max_iter = 1000, restart = 30, verbose = 0;
        {
            ParmParse ppg("gmres");
            ppg.query("rtol", rtol);
            ppg.query("atol", atol);
            ppg.query("max_iter", max_iter);
            ppg.query("restart", restart);
            ppg.query("verbose", verbose);
        }

#ifdef USE_PETSC_TEST
        std::unique_ptr<wpc::PetscKSP> petsc;
        int petsc_coo = 0;
        ParmParse("petsc").query("coo_assembly", petsc_coo);
        if (pc == "petsc") { wpc::petsc_initialize(); }
#else
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pc != "petsc", "Built without PETSc (USE_PETSC_TEST=FALSE)");
#endif

        Vector<Record> records;
        for (auto const& sys : systems) {
            std::string const dir = dump_dir + "/" + sys;
            auto info = readInfo(dir + "/info.txt");
            Long const n = std::stol(info.at("n"));
            Real const sys_rtol = (rtol > 0) ? rtol : Real(std::stod(info.at("linsol_rtol")));

            // WarpX's DOF partition when replayed on the same number of ranks
            // (the ASM subdomains are the per-rank row blocks); even split otherwise.
            int const nprocs = ParallelDescriptor::NProcs();
            Vector<Long> rows(nprocs+1);
            bool same_partition = false;
            {
                std::vector<Long> ro;
                std::ifstream ifs(dir + "/info.txt");
                std::string line;
                while (std::getline(ifs, line)) {
                    if (line.rfind("row_offsets", 0) == 0) {
                        std::istringstream iss(line.substr(11));
                        Long v;
                        while (iss >> v) { ro.push_back(v); }
                    }
                }
                if (int(ro.size()) == nprocs+1) {
                    for (int r = 0; r <= nprocs; ++r) { rows[r] = ro[r]; }
                    same_partition = true;
                } else {
                    for (int r = 0; r <= nprocs; ++r) { rows[r] = (n*r)/nprocs; }
                }
            }
            AlgPartition const part(rows);

            SpMatrix<Real> A, P;
            wpc::HostCSR Phost;
            double t0 = timer();
            readMatrix(dir + "/A.bin", part, A);
            if (pc == "algmg" || pc == "petsc") { readMatrix(dir + "/P.bin", part, P, &Phost); }
            AlgVector<Real> b(part), xref(part), x(part);
            readVector(dir + "/b.bin", b);
            readVector(dir + "/x.bin", xref);
            double t1 = timer();

            Record rec;
            rec.name = sys;
            rec.warpx_iters = std::stoi(info.at("linsol_iters"));
            rec.relres_warpx = relResidual(A, xref, b);

            if (verbose > 0) {
                amrex::Print() << sys << ": n = " << n << ", read " << t1-t0 << " s, rtol " << sys_rtol
                               << (same_partition ? ", WarpX partition" : ", even partition")
                               << ", WarpX " << rec.warpx_iters << " iterations, |b-Ax|/|b| = "
                               << rec.relres_warpx << "\n";
            }

#ifdef USE_PETSC_TEST
            if (pc == "petsc") {
                if (!petsc) {
                    petsc = std::make_unique<wpc::PetscKSP>(part.numLocalRows(), n, restart,
                                                            verbose, petsc_coo != 0);
                    amrex::Print() << "PETSc PC: " << petsc->pcDescription() << "\n";
                }
                for (int rep = 0; rep < nrepeat; ++rep) {
                    petsc->setOperator(&A);
                    double ta = timer();
                    petsc->assemble(Phost);
                    petsc->setup();
                    double tb = timer();
                    petsc->solve(x, b, sys_rtol, atol, max_iter);
                    double tc = timer();
                    rec.setup = tb-ta;
                    rec.solve = tc-tb;
                    rec.iters = petsc->numIters();
                    rec.relres = relResidual(A, x, b);
                    rec.diff = relDiff(x, xref);
                    if (petsc->reason().find("CONVERGED") == std::string::npos) {
                        amrex::Print() << sys << ": PETSc KSP " << petsc->reason() << "\n";
                    }
                }
                records.push_back(rec);
                continue;
            }
#endif
            for (int rep = 0; rep < nrepeat; ++rep) {
                std::unique_ptr<AlgMG<Real>> amg;
                double ta = timer();
                if (pc != "none") {
                    amg = std::make_unique<AlgMG<Real>>((pc == "algmg") ? P : A);
                    applyAlgMGOptions(*amg);
                    amg->setFixedIter(pc_max_iter);
                    amg->setup();
                }
                double tb = timer();

                GMRES_MV<Real> gm(&A);
                if (amg) {
                    gm.setPrecond([&] (AlgVector<Real>& z, AlgVector<Real> const& r) {
                        if (pc_max_iter == 1) {
                            amg->precond(z, r);
                        } else {
                            z.setVal(0);
                            amg->solve(z, r);
                        }
                    });
                }
                gm.getGMRES().setRestartLength(restart);
                gm.getGMRES().setMaxIters(max_iter);
                gm.setVerbose(verbose);
                x.setVal(0);
                gm.solve(x, b, sys_rtol, atol);
                double tc = timer();

                rec.setup = tb-ta;
                rec.solve = tc-tb;
                rec.iters = gm.getGMRES().getNumIters();
                rec.relres = relResidual(A, x, b);
                rec.diff = relDiff(x, xref);
                if (amg) {
                    Long nnz = 0, nnz0 = amg->getMatrix(0).numLocalNonZeros();
                    for (int lev = 0; lev < amg->numLevels(); ++lev) {
                        nnz += amg->getMatrix(lev).numLocalNonZeros();
                    }
                    ParallelDescriptor::ReduceLongSum(nnz);
                    ParallelDescriptor::ReduceLongSum(nnz0);
                    rec.levels = amg->numLevels();
                    rec.complexity = double(nnz)/double(nnz0);
                }
            }
            records.push_back(rec);
        }

        amrex::Print() << "\npc = " << pc << ", gmres rtol = " << (rtol > 0 ? std::to_string(rtol) : std::string("from dump"))
                       << ", " << ParallelDescriptor::NProcs() << " ranks\n";
        amrex::Print() << std::left << std::setw(18) << "system" << std::right
                       << std::setw(7) << "warpx" << std::setw(7) << "iters"
                       << std::setw(10) << "setup" << std::setw(10) << "solve"
                       << std::setw(11) << "relres" << std::setw(11) << "relres_wx"
                       << std::setw(11) << "|x-xwx|" << std::setw(5) << "lev" << std::setw(7) << "cmplx\n";
        int tot_wx = 0, tot = 0;
        double tot_setup = 0, tot_solve = 0;
        for (auto const& r : records) {
            amrex::Print() << std::left << std::setw(18) << r.name << std::right
                           << std::setw(7) << r.warpx_iters << std::setw(7) << r.iters
                           << std::fixed << std::setprecision(4)
                           << std::setw(10) << r.setup << std::setw(10) << r.solve
                           << std::scientific << std::setprecision(2)
                           << std::setw(11) << r.relres << std::setw(11) << r.relres_warpx
                           << std::setw(11) << r.diff
                           << std::setw(5) << r.levels << std::fixed << std::setw(7) << r.complexity
                           << std::defaultfloat << "\n";
            tot_wx += r.warpx_iters; tot += r.iters; tot_setup += r.setup; tot_solve += r.solve;
        }
        amrex::Print() << std::left << std::setw(18) << "total" << std::right
                       << std::setw(7) << tot_wx << std::setw(7) << tot
                       << std::fixed << std::setprecision(4)
                       << std::setw(10) << tot_setup << std::setw(10) << tot_solve
                       << std::defaultfloat << "\n";
#ifdef USE_PETSC_TEST
        petsc.reset();
        if (pc == "petsc") { wpc::petsc_finalize(); }
#endif
    }
    amrex::Finalize();
}
