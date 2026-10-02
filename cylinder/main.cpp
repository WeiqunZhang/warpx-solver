// Standalone 3D MLMG test for the rlehe_cylinder WarpX case.
//
// Builds the same linear system as WarpX's lab-frame electrostatic solve
// (ablastr::fields::computePhi with EB) for inputs_D, without WarpX.
// rho = 0 and phi = 0 initially; the EB Dirichlet potential drives the solve.

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_RealBox.H>

#include <algorithm>
#include <limits>
#include <string>

using namespace amrex;

namespace {

struct Params
{
    Array<int,3> n_cell{1024, 1024, 256};
    IntVect max_grid_size{256, 256, 64};

    Array<Real,3> prob_lo{-0.51_rt, -0.51_rt, -2.e-3_rt};
    Array<Real,3> prob_hi{ 0.51_rt,  0.51_rt, 50.e-3_rt};

    std::string eb_potential{" -40e3*(z>16.5e-3) - 41e3*(z>5.50e-3)*(z<16.5e-3)"};
    Real time = 0.0_rt;

    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 200;
    Real reltol = 1.e-6_rt;
    Real abstol = 0.0_rt;
    Real bottom_reltol = 3.e-4_rt;
    int final_smooth = 8;

    int nsolves = 1;
    int reset_phi = 1;

    void read ()
    {
        {
            ParmParse pp("amr");
            Vector<int> v(n_cell.begin(), n_cell.end());
            if (pp.queryarr("n_cell", v, 0, 3)) { std::copy(v.begin(), v.end(), n_cell.begin()); }
            // One value for all directions, or one per direction.
            Vector<int> m;
            if (pp.queryarr("max_grid_size", m)) {
                AMREX_ALWAYS_ASSERT(m.size() == 1 || m.size() == 3);
                max_grid_size = (m.size() == 1) ? IntVect(m[0]) : IntVect(m[0], m[1], m[2]);
            }
        }
        {
            ParmParse pp("geometry");
            Vector<Real> lo(prob_lo.begin(), prob_lo.end());
            Vector<Real> hi(prob_hi.begin(), prob_hi.end());
            if (pp.queryarr("prob_lo", lo, 0, 3)) { std::copy(lo.begin(), lo.end(), prob_lo.begin()); }
            if (pp.queryarr("prob_hi", hi, 0, 3)) { std::copy(hi.begin(), hi.end(), prob_hi.begin()); }
        }
        {
            ParmParse pp("warpx");
            pp.query("eb_potential(x,y,z,t)", eb_potential);
        }
        {
            ParmParse pp("problem");
            pp.query("time", time);
        }
        {
            ParmParse pp("mlmg");
            pp.query("verbose", verbose);
            pp.query("bottom_verbose", bottom_verbose);
            pp.query("max_iter", max_iter);
            pp.query("reltol", reltol);
            pp.query("abstol", abstol);
            pp.query("bottom_reltol", bottom_reltol);
            pp.query("final_smooth", final_smooth);
        }
        {
            ParmParse pp;
            pp.query("nsolves", nsolves);
            pp.query("reset_phi", reset_phi);
        }
    }
};

void
main_main ()
{
    BL_PROFILE("main");

    static_assert(AMREX_SPACEDIM == 3, "This test is 3D only");

    Params p;
    p.read();

    Box const domain(IntVect(0), IntVect(p.n_cell[0]-1, p.n_cell[1]-1, p.n_cell[2]-1));
    RealBox const real_box(p.prob_lo.data(), p.prob_hi.data());
    Array<int,3> const is_periodic{0, 0, 0};
    Geometry const geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    BoxArray grids(domain);
    grids.maxSize(p.max_grid_size);
    DistributionMapping const dmap(grids);

    // WarpX::InitEB: eb2.geom_type = stl etc. are read by AMReX.
    Real t0 = ParallelDescriptor::second();
    EB2::Build(geom, 0, 20);
    Real const t_eb = ParallelDescriptor::second() - t0;

    // WarpX::AllocLevelMFs: ghost cells = ng_FieldSolver = 1 (staggered/Yee).
    auto factory = makeEBFabFactory(&EB2::IndexSpace::top(), geom, grids, dmap,
                                    {1, 1, 1}, EBSupport::full);

    BoxArray const nodal_grids = amrex::convert(grids, IntVect::TheNodeVector());
    MultiFab phi(nodal_grids, dmap, 1, 1, MFInfo{}, *factory);
    MultiFab rhs(nodal_grids, dmap, 1, 0, MFInfo{}, *factory);
    phi.setVal(0.0_rt);
    rhs.setVal(0.0_rt);  // -rho/eps0 with rho = 0
    rhs.OverrideSync(geom.periodicity());

    // PoissonBoundaryHandler::DefinePhiBCs for
    // boundary.field_lo = neumann neumann pec, field_hi = neumann neumann neumann
    Array<LinOpBCType,3> const lobc{LinOpBCType::Neumann, LinOpBCType::Neumann,
                                    LinOpBCType::Dirichlet};
    Array<LinOpBCType,3> const hibc{LinOpBCType::Neumann, LinOpBCType::Neumann,
                                    LinOpBCType::Neumann};

    Parser potential_parser(p.eb_potential);
    potential_parser.registerVariables({"x", "y", "z", "t"});
    auto const potential = potential_parser.compile<4>();
    Real const time = p.time;

    IntVect min_size(std::numeric_limits<int>::max());
    IntVect max_size(std::numeric_limits<int>::lowest());
    for (int i = 0, n = static_cast<int>(grids.size()); i < n; ++i) {
        min_size.min(grids[i].length());
        max_size.max(grids[i].length());
    }
    Print() << "\nrlehe_cylinder MLEBNodeFDLaplacian test\n"
            << "  domain       : " << domain.length() << " cells\n"
            << "  prob_lo      : " << p.prob_lo << "\n"
            << "  prob_hi      : " << p.prob_hi << "\n"
            << "  dx           : " << RealVect(geom.CellSize()) << "\n"
            << "  grids        : " << grids.size()
            << " (min " << min_size << ", max " << max_size << ")\n"
            << "  MPI ranks    : " << ParallelDescriptor::NProcs() << "\n"
            << "  EB potential : " << p.eb_potential << " at t = " << time << "\n"
            << "  EB build     : " << t_eb << " s\n"
            << "  reltol/abstol: " << p.reltol << " / " << p.abstol << "\n"
            << "  bottom_reltol: " << p.bottom_reltol << "\n\n";

    // Even solves use geometric multigrid, odd solves AlgMG.
    for (int isolve = 0; isolve < 2*p.nsolves; ++isolve) {
        bool const algebraic = (isolve % 2 == 1);
        if (p.reset_phi) { phi.setVal(0.0_rt); }

        BL_PROFILE_REGION(algebraic ? "algebraic" : "geometric");

        // ablastr::fields::computePhi builds a new linop and MLMG every call.
        ParallelDescriptor::Barrier();
        t0 = ParallelDescriptor::second();

        LPInfo info;
        // AlgMG solves the whole level; skip building the geometric levels.
        if (algebraic) { info.setMaxCoarseningLevel(0); }
        MLEBNodeFDLaplacian linop;
        linop.define(Vector<Geometry>{geom}, Vector<BoxArray>{grids},
                     Vector<DistributionMapping>{dmap}, info,
                     Vector<EBFArrayBoxFactory const*>{factory.get()});
        linop.setSigma({1.0_rt, 1.0_rt, 1.0_rt});
        linop.setEBDirichlet(
            [=] AMREX_GPU_HOST_DEVICE (Real x, Real y, Real z) noexcept -> Real
            {
                return potential(x, y, z, time);
            });
        linop.setDomainBC(lobc, hibc);

        MLMG mlmg(linop);
        mlmg.setVerbose(p.verbose);
        mlmg.setBottomVerbose(p.bottom_verbose);
        mlmg.setMaxIter(p.max_iter);
        mlmg.setBottomTolerance(p.bottom_reltol);  // also AlgMG's tolerance
        mlmg.setFinalSmooth(p.final_smooth);
        mlmg.setConvergenceNormType(MLMGNormType::greater);
        mlmg.setNoGpuSync(true);
        mlmg.setThrowException(true);  // report a failed solve and keep going
        mlmg.setMultigridType(algebraic ? MultigridType::algebraic
                                        : MultigridType::geometric);

        std::string status = "converged";
        try {
            mlmg.solve({&phi}, {&rhs}, p.reltol, p.abstol);
        } catch (MLMG::error const& e) {
            status = std::string("FAILED: ") + e.what();
        }
        Gpu::streamSynchronize();
        auto const& hist = mlmg.getResidualHistory();
        Real const final_resid = (status == "converged" || hist.empty())
            ? mlmg.getFinalResidual() : hist.back();

        Real t_solve = ParallelDescriptor::second() - t0;
        ParallelDescriptor::ReduceRealMax(t_solve);

        Print() << "\nSolve " << isolve/2 << " (" << (algebraic ? "algebraic" : "geometric") << ")\n"
                << "  MG levels          : " << linop.NMGLevels(0) << "\n"
                << "  initial rhs norm   : " << mlmg.getInitRHS() << "\n"
                << "  initial residual   : " << mlmg.getInitResidual() << "\n"
                << "  status             : " << status << "\n"
                << "  final residual     : " << final_resid << "\n"
                << "  MLMG iterations    : " << mlmg.getNumIters() << "\n"
                << "  bottom iterations  :";
        for (int n : mlmg.getNumCGIters()) { Print() << " " << n; }
        Print() << "\n"
                << "  phi min/max        : " << phi.min(0) << " / " << phi.max(0) << "\n"
                << "  wall time          : " << t_solve << " s (linop setup + solve)\n\n";
    }
}

} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
