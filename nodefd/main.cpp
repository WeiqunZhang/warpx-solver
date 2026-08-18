//
// Standalone 3D benchmark for amrex::MLMG driving amrex::MLEBNodeFDLaplacian.
//
// This mimics WarpX's electrostatic Poisson solve
// (WarpX/Source/ablastr/fields/PoissonSolver.H) closely enough to be a useful
// optimization target, but with no EB, a fully periodic domain, and a single
// AMR level.
//
// Note on the periodic problem: an all-periodic Laplacian with no Dirichlet
// boundary is singular, but MLEBNodeFDLaplacian hard-codes isSingular() to
// false, so MLMG never projects out the constant mode.  We therefore use an
// exactly zero-mean RHS (and subtract the residual mean to round-off) so the
// system is consistent.  If that still misbehaves, set fixed_iter > 0.
//

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>

using namespace amrex;

namespace {

struct TestParams
{
    int n_cell = 128;
    int nboxes = -1;          // -1 -> ParallelDescriptor::NProcs()
    int nsolves = 5;
    int nwarmup = 1;
    int nosync = 1;           // MLMG::setNoGpuSync -- the flag under study
    int recreate_linop = 1;   // rebuild the linop inside the timed loop (WarpX-like)

    int verbose = 1;
    int bottom_verbose = 0;
    int max_iter = 200;
    int fixed_iter = 0;
    Real reltol = Real(1.e-11);
    Real abstol = Real(0.0);
    // NOT the AMReX/WarpX default (bicgstab).  On this all-periodic problem the
    // operator is singular, but MLEBNodeFDLaplacian hard-codes isSingular() to
    // false, so MLMG never projects the constant mode out of the bottom solve.
    // BiCGStab and CG then blow up (resid/bnorm -> 1e20 within ~13 V-cycles);
    // the smoother bottom solve is stable and converges in ~9 iterations.
    // The bottom solve is well under 1% of the total, so this barely perturbs
    // what we are actually measuring.
    std::string bottom_solver{"smoother"};

    int max_coarsening_level = 30;
    int agglomeration = 1;
    int consolidation = 1;
    int agg_grid_size = -1;
    int con_grid_size = -1;
    int semicoarsening = 0;

    int pre_smooth = 2;
    int post_smooth = 2;
    int final_smooth = 8;
    int bottom_smooth = 0;

    Real sigma = Real(1.0);

    void read ()
    {
        ParmParse pp;
        pp.query("n_cell", n_cell);
        pp.query("nboxes", nboxes);
        pp.query("nsolves", nsolves);
        pp.query("nwarmup", nwarmup);
        pp.query("nosync", nosync);
        pp.query("recreate_linop", recreate_linop);

        pp.query("verbose", verbose);
        pp.query("bottom_verbose", bottom_verbose);
        pp.query("max_iter", max_iter);
        pp.query("fixed_iter", fixed_iter);
        pp.query("reltol", reltol);
        pp.query("abstol", abstol);
        pp.query("bottom_solver", bottom_solver);

        pp.query("max_coarsening_level", max_coarsening_level);
        pp.query("agglomeration", agglomeration);
        pp.query("consolidation", consolidation);
        pp.query("agg_grid_size", agg_grid_size);
        pp.query("con_grid_size", con_grid_size);
        pp.query("semicoarsening", semicoarsening);

        pp.query("pre_smooth", pre_smooth);
        pp.query("post_smooth", post_smooth);
        pp.query("final_smooth", final_smooth);
        pp.query("bottom_smooth", bottom_smooth);

        pp.query("sigma", sigma);

        if (nboxes <= 0) { nboxes = ParallelDescriptor::NProcs(); }
    }
};

MLMG::BottomSolver
parseBottomSolver (std::string const& s)
{
    if (s.empty() || s == "default") { return MLMG::BottomSolver::Default; }
    if (s == "bicgstab") { return MLMG::BottomSolver::bicgstab; }
    if (s == "cg")       { return MLMG::BottomSolver::cg; }
    if (s == "bicgcg")   { return MLMG::BottomSolver::bicgcg; }
    if (s == "cgbicg")   { return MLMG::BottomSolver::cgbicg; }
    if (s == "smoother") { return MLMG::BottomSolver::smoother; }
    amrex::Abort("Unknown bottom_solver: " + s
                 + " (use default/bicgstab/cg/bicgcg/cgbicg/smoother)");
    return MLMG::BottomSolver::Default;
}

// Smooth, periodic, zero-mean RHS.
void
initRhs (MultiFab& rhs, Geometry const& geom)
{
    BL_PROFILE("initRhs");

    const auto problo = geom.ProbLoArray();
    const auto dx     = geom.CellSizeArray();
    constexpr Real two_pi = Real(2.0*3.1415926535897932);
    const GpuArray<Real,AMREX_SPACEDIM> k
        {AMREX_D_DECL(two_pi/geom.ProbLength(0),
                      two_pi/geom.ProbLength(1),
                      two_pi/geom.ProbLength(2))};

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        auto const& a = rhs.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
        {
            const Real x = problo[0] + Real(i)*dx[0];
            const Real y = problo[1] + Real(j)*dx[1];
            const Real z = problo[2] + Real(kk)*dx[2];
            a(i,j,kk) = std::sin(k[0]*x) * std::sin(k[1]*y) * std::sin(k[2]*z);
        });
    }

    // Remove any round-off level mean so the singular periodic system stays
    // consistent.  sum_unique counts nodes shared by several boxes only once.
    const Real sum = rhs.sum_unique(0, false, geom.periodicity());
    const auto nnodes = static_cast<Real>(geom.Domain().d_numPts());
    rhs.plus(-sum/nnodes, 0, 1);
}

void
printBoxInfo (BoxArray const& ba)
{
    IntVect mn(std::numeric_limits<int>::max());
    IntVect mx(std::numeric_limits<int>::lowest());
    for (int i = 0, n = static_cast<int>(ba.size()); i < n; ++i) {
        IntVect const len = ba[i].length();
        mn.min(len);
        mx.max(len);
    }
    amrex::Print() << "  BoxArray      : " << ba.size() << " boxes, min size "
                   << mn << ", max size " << mx << "\n";
}

void
main_main ()
{
    BL_PROFILE("main_main");

    TestParams p;
    p.read();

    Box domain(IntVect(0), IntVect(p.n_cell-1));
    RealBox rb({AMREX_D_DECL(Real(0.),Real(0.),Real(0.))},
               {AMREX_D_DECL(Real(1.),Real(1.),Real(1.))});
    Array<int,AMREX_SPACEDIM> const is_periodic{AMREX_D_DECL(1,1,1)};
    Geometry geom(domain, rb, CoordSys::cartesian, is_periodic);

    BoxArray ba;
    DistributionMapping dm;
    {
        BL_PROFILE("setup-grids");
        ba = amrex::decompose(domain, p.nboxes);
        dm.define(ba);
    }

    amrex::Print() << "\nMLEBNodeFDLaplacian benchmark\n"
                   << "  domain        : " << domain << " (fully periodic, no EB)\n"
                   << "  MPI ranks     : " << ParallelDescriptor::NProcs() << "\n"
                   << "  nboxes        : " << p.nboxes << "\n";
    printBoxInfo(ba);
    amrex::Print() << "  nosync        : " << p.nosync << "\n"
                   << "  recreate_linop: " << p.recreate_linop << "\n"
                   << "  warmup/timed  : " << p.nwarmup << " / " << p.nsolves << "\n\n";

    BoxArray const nba = amrex::convert(ba, IntVect::TheNodeVector());
    MultiFab phi(nba, dm, 1, 1);  // 1 ghost => MLMG aliases it, no internal copy
    MultiFab rhs(nba, dm, 1, 0);

    initRhs(rhs, geom);

    Array<LinOpBCType,AMREX_SPACEDIM> const lobc
        {AMREX_D_DECL(LinOpBCType::Periodic,LinOpBCType::Periodic,LinOpBCType::Periodic)};
    Array<LinOpBCType,AMREX_SPACEDIM> const hibc = lobc;

    std::unique_ptr<MLEBNodeFDLaplacian> linop;
    std::unique_ptr<MLMG> mlmg;

    auto build_solver = [&] ()
    {
        BL_PROFILE("build-solver");

        LPInfo info;
        info.setMaxCoarseningLevel(p.max_coarsening_level);
        info.setAgglomeration(p.agglomeration);
        info.setConsolidation(p.consolidation);
        info.setSemicoarsening(p.semicoarsening);
        if (p.agg_grid_size > 0) { info.setAgglomerationGridSize(p.agg_grid_size); }
        if (p.con_grid_size > 0) { info.setConsolidationGridSize(p.con_grid_size); }

        mlmg.reset();  // MLMG holds a reference to the linop; destroy it first
        linop = std::make_unique<MLEBNodeFDLaplacian>();
        linop->define({geom}, {ba}, {dm}, info);
        linop->setSigma({AMREX_D_DECL(p.sigma,p.sigma,p.sigma)});
        linop->setDomainBC(lobc, hibc);

        mlmg = std::make_unique<MLMG>(*linop);
        mlmg->setVerbose(p.verbose);
        mlmg->setBottomVerbose(p.bottom_verbose);
        mlmg->setMaxIter(p.max_iter);
        mlmg->setConvergenceNormType(MLMGNormType::greater);  // what WarpX uses
        mlmg->setNoGpuSync(p.nosync);
        mlmg->setPreSmooth(p.pre_smooth);
        mlmg->setPostSmooth(p.post_smooth);
        mlmg->setFinalSmooth(p.final_smooth);
        mlmg->setBottomSmooth(p.bottom_smooth);
        mlmg->setBottomSolver(parseBottomSolver(p.bottom_solver));
        if (p.fixed_iter > 0) { mlmg->setFixedIter(p.fixed_iter); }
    };

    if (!p.recreate_linop) { build_solver(); }

    Vector<Real> t_build, t_solve;
    int niter_last = -1;
    Real resid_last = Real(-1.0);
    bool reported = false;

    auto one_solve = [&] (bool timed)
    {
        // MLMG aliases phi in place (it has exactly 1 ghost), so without this
        // reset every solve after the first would start already converged and
        // do zero V-cycles.
        phi.setVal(Real(0.0));
        rhs.OverrideSync(geom.periodicity());  // WarpX does this before each solve

        Gpu::streamSynchronize();
        ParallelDescriptor::Barrier();
        const Real t0 = amrex::second();

        if (p.recreate_linop) { build_solver(); }

        Gpu::streamSynchronize();
        const Real t1 = amrex::second();

        if (!reported) {
            reported = true;
            const int nmglev = linop->NMGLevels(0);
            amrex::Print() << "  MG levels     : " << nmglev << "\n";
            if (nmglev <= 2) {
                amrex::Print()
                    << "\n  *** WARNING: only " << nmglev << " MG level(s). ***\n"
                    << "  amrex::decompose makes boxes as cubic as possible even when that\n"
                    << "  means odd cell counts, so the boxes above are not coarsenable and\n"
                    << "  multigrid has no coarse-grid correction -- the solve degenerates to\n"
                    << "  plain smoothing and will crawl or fail to converge.  Choose nboxes\n"
                    << "  so that it divides " << p.n_cell << " evenly in each direction\n"
                    << "  (powers of 2 are safe), or raise n_cell.\n\n";
            } else {
                amrex::Print() << "\n";
            }
        }

        mlmg->solve({&phi}, {&rhs}, p.reltol, p.abstol);

        // MLMG with nosync leaves work queued on the stream; sync so the timer
        // measures completion, not launch.
        Gpu::streamSynchronize();
        const Real t2 = amrex::second();

        niter_last = mlmg->getNumIters();
        resid_last = mlmg->getFinalResidual();

        if (timed) {
            t_build.push_back(t1-t0);
            t_solve.push_back(t2-t1);
        }
    };

    // Each BL_PROFILE_REGION needs its own scope: the macro always names its
    // object tiny_profile_region_vname, so two in one scope fail to compile.
    {
        BL_PROFILE_REGION("warmup");
        for (int i = 0; i < p.nwarmup; ++i) { one_solve(false); }
    }
    {
        BL_PROFILE_REGION("solve");
        for (int i = 0; i < p.nsolves; ++i) { one_solve(true); }
    }

    if (t_solve.empty()) { return; }

    // Report the max over ranks -- that is the wall time that matters.
    ParallelDescriptor::ReduceRealMax(t_build.dataPtr(), static_cast<int>(t_build.size()),
                                      ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealMax(t_solve.dataPtr(), static_cast<int>(t_solve.size()),
                                      ParallelDescriptor::IOProcessorNumber());

    const auto n = static_cast<Real>(t_solve.size());
    const Real bsum = std::accumulate(t_build.begin(), t_build.end(), Real(0.0));
    const Real ssum = std::accumulate(t_solve.begin(), t_solve.end(), Real(0.0));

    amrex::Print() << "\n--- timings over " << t_solve.size()
                   << " solves (max over ranks) ---\n";
    for (int i = 0, m = static_cast<int>(t_solve.size()); i < m; ++i) {
        amrex::Print() << "  solve " << i << " :  build " << t_build[i]
                       << " s,  solve " << t_solve[i] << " s\n";
    }
    amrex::Print() << "  mean    :  build " << bsum/n << " s,  solve " << ssum/n << " s\n"
                   << "  min/max solve : "
                   << *std::min_element(t_solve.begin(), t_solve.end()) << " / "
                   << *std::max_element(t_solve.begin(), t_solve.end()) << " s\n"
                   << "  MLMG iterations: " << niter_last
                   << ",  final residual: " << resid_last << "\n\n";
}

} // anonymous namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");
        main_main();
    }
    amrex::Finalize();
}
