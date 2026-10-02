// Standalone 3D MLMG test for the rlehe_cylinder WarpX case.
//
// Builds the same linear system as WarpX's lab-frame electrostatic solve
// (ablastr::fields::computePhi with EB) for inputs_D, without WarpX.
// rho = 0 and phi = 0 initially; the EB Dirichlet potential drives the solve.

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
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
    Array<int,3> numprocs{4, 4, 4};  // warpx.numprocs; all zero -> max_grid_size
    int max_grid_size = 256;

    Array<Real,3> prob_lo{-0.51_rt, -0.51_rt, -2.e-3_rt};
    Array<Real,3> prob_hi{ 0.51_rt,  0.51_rt, 50.e-3_rt};

    std::string eb_potential{" -40e3*(z>16.5e-3) - 41e3*(z>5.50e-3)*(z<16.5e-3)"};
    Real time = 0.0_rt;

    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 200;
    Real reltol = 1.e-1_rt;
    Real abstol = 1.e-1_rt;
    int final_smooth = 8;

    int nsolves = 1;
    int reset_phi = 1;

    void read ()
    {
        {
            ParmParse pp("amr");
            Vector<int> v(n_cell.begin(), n_cell.end());
            if (pp.queryarr("n_cell", v, 0, 3)) { std::copy(v.begin(), v.end(), n_cell.begin()); }
            pp.query("max_grid_size", max_grid_size);
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
            Vector<int> v;
            if (pp.queryarr("numprocs", v, 0, 3)) {
                std::copy(v.begin(), v.end(), numprocs.begin());
            } else {
                numprocs = {0, 0, 0};
            }
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
            pp.query("final_smooth", final_smooth);
        }
        {
            ParmParse pp;
            pp.query("nsolves", nsolves);
            pp.query("reset_phi", reset_phi);
        }
    }
};

// Same decomposition as WarpX::PostProcessBaseGrids with warpx.numprocs.
BoxArray
makeGrids (Box const& domain, Params const& p)
{
    if (p.numprocs[0] <= 0) {
        BoxArray ba(domain);
        ba.maxSize(p.max_grid_size);
        return ba;
    }

    IntVect const np(p.numprocs[0], p.numprocs[1], p.numprocs[2]);
    IntVect const domlo = domain.smallEnd();
    IntVect const domlen = domain.size();
    IntVect const sz = domlen / np;
    IntVect const extra = domlen - sz*np;
    auto range = [&] (int idim, int i) {
        int lo = (i < extra[idim]) ? i*(sz[idim]+1) : (i*sz[idim]+extra[idim]);
        int hi = (i < extra[idim]) ? lo+(sz[idim]+1)-1 : lo+sz[idim]-1;
        return std::make_pair(lo+domlo[idim], hi+domlo[idim]);
    };
    BoxList bl;
    for (int k = 0; k < np[2]; ++k) {
        auto [klo, khi] = range(2, k);
        for (int j = 0; j < np[1]; ++j) {
            auto [jlo, jhi] = range(1, j);
            for (int i = 0; i < np[0]; ++i) {
                auto [ilo, ihi] = range(0, i);
                bl.push_back(Box(IntVect(ilo,jlo,klo), IntVect(ihi,jhi,khi)));
            }
        }
    }
    return BoxArray(std::move(bl));
}

void
main_main ()
{
    static_assert(AMREX_SPACEDIM == 3, "This test is 3D only");

    Params p;
    p.read();

    Box const domain(IntVect(0), IntVect(p.n_cell[0]-1, p.n_cell[1]-1, p.n_cell[2]-1));
    RealBox const real_box(p.prob_lo.data(), p.prob_hi.data());
    Array<int,3> const is_periodic{0, 0, 0};
    Geometry const geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    BoxArray const grids = makeGrids(domain, p);
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
            << "  reltol/abstol: " << p.reltol << " / " << p.abstol << "\n\n";

    for (int isolve = 0; isolve < p.nsolves; ++isolve) {
        if (p.reset_phi) { phi.setVal(0.0_rt); }

        // ablastr::fields::computePhi builds a new linop and MLMG every call.
        ParallelDescriptor::Barrier();
        t0 = ParallelDescriptor::second();

        LPInfo const info;
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
        mlmg.setFinalSmooth(p.final_smooth);
        mlmg.setConvergenceNormType(MLMGNormType::greater);
        mlmg.setNoGpuSync(true);

        mlmg.solve({&phi}, {&rhs}, p.reltol, p.abstol);
        Gpu::streamSynchronize();

        Real t_solve = ParallelDescriptor::second() - t0;
        ParallelDescriptor::ReduceRealMax(t_solve);

        Print() << "\nSolve " << isolve << "\n"
                << "  MG levels          : " << linop.NMGLevels(0) << "\n"
                << "  initial rhs norm   : " << mlmg.getInitRHS() << "\n"
                << "  initial residual   : " << mlmg.getInitResidual() << "\n"
                << "  final residual     : " << mlmg.getFinalResidual() << "\n"
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
