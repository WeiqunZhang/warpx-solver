// Standalone 3D reproducer for MLMG with MLEBNodeFDLaplacian.
//
// This mirrors the geometry, boundary conditions, EB, and solver settings of
// run-marco/input_repro.txt without pulling in WarpX.  Both phi and rhs are
// zero when passed to MLMG; the inhomogeneous EB Dirichlet value supplies the
// nonzero initial residual.

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_GeometryShop.H>
#include <AMReX_EB2_IF_Parser.H>
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
#include <AMReX_RealBox.H>

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <string>
#include <utility>

using namespace amrex;

namespace {

struct Params
{
    Array<int,AMREX_SPACEDIM> n_cell{528, 80, 8};
    int max_grid_size = 528;
    int blocking_factor = 8;
    int refine_grid_layout = 1;

    Array<Real,AMREX_SPACEDIM> prob_lo{-0.0528_rt, -0.008_rt, -0.0008_rt};
    Array<Real,AMREX_SPACEDIM> prob_hi{ 0.0528_rt,  0.008_rt,  0.0008_rt};
    Array<int,AMREX_SPACEDIM> is_periodic{0, 1, 1};

    std::string eb_implicit_function{
        "0 + 2*(x<=-0.052) + 2*(x>=0.052) - (x>-0.052)*(x<0.052)"};
    std::string eb_potential{"-100000.0*(x<(-0.052+5.0e-3))"};

    int verbose = 2;
    int linop_verbose = 0;
    int bottom_verbose = 1;
    int max_iter = 200;
    Real reltol = 1.e-11_rt;
    Real abstol = 1.e-6_rt;
    int max_coarsening_level = 30;
    int final_smooth = 8;
    std::string bottom_solver{"default"};
    int bottom_max_iter = 200;
    Real bottom_reltol = 1.e-4_rt;
    int no_gpu_sync = 1;

    void read ()
    {
        {
            ParmParse pp("amr");
            Vector<int> values(n_cell.begin(), n_cell.end());
            if (pp.queryarr("n_cell", values, 0, AMREX_SPACEDIM)) {
                std::copy(values.begin(), values.end(), n_cell.begin());
            }
            pp.query("max_grid_size", max_grid_size);
            pp.query("blocking_factor", blocking_factor);
            pp.query("refine_grid_layout", refine_grid_layout);
        }
        {
            ParmParse pp("geometry");
            Vector<Real> lo(prob_lo.begin(), prob_lo.end());
            Vector<Real> hi(prob_hi.begin(), prob_hi.end());
            Vector<int> periodic(is_periodic.begin(), is_periodic.end());
            if (pp.queryarr("prob_lo", lo, 0, AMREX_SPACEDIM)) {
                std::copy(lo.begin(), lo.end(), prob_lo.begin());
            }
            if (pp.queryarr("prob_hi", hi, 0, AMREX_SPACEDIM)) {
                std::copy(hi.begin(), hi.end(), prob_hi.begin());
            }
            if (pp.queryarr("is_periodic", periodic, 0, AMREX_SPACEDIM)) {
                std::copy(periodic.begin(), periodic.end(), is_periodic.begin());
            }
        }
        {
            ParmParse pp("problem");
            pp.query("eb_implicit_function", eb_implicit_function);
            pp.query("eb_potential", eb_potential);
        }
        {
            ParmParse pp("mlmg");
            pp.query("verbose", verbose);
            pp.query("linop_verbose", linop_verbose);
            pp.query("bottom_verbose", bottom_verbose);
            pp.query("max_iter", max_iter);
            pp.query("reltol", reltol);
            pp.query("abstol", abstol);
            pp.query("max_coarsening_level", max_coarsening_level);
            pp.query("final_smooth", final_smooth);
            pp.query("bottom_solver", bottom_solver);
            pp.query("bottom_max_iter", bottom_max_iter);
            pp.query("bottom_reltol", bottom_reltol);
            pp.query("no_gpu_sync", no_gpu_sync);
        }

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell[idim] > 0,
                "amr.n_cell entries must be positive");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prob_hi[idim] > prob_lo[idim],
                "geometry.prob_hi must be greater than geometry.prob_lo");
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_grid_size > 0,
            "amr.max_grid_size must be positive");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(blocking_factor > 0,
            "amr.blocking_factor must be positive");
    }
};

MLMG::BottomSolver
parseBottomSolver (std::string const& name)
{
    if (name.empty() || name == "default") { return MLMG::BottomSolver::Default; }
    if (name == "bicgstab") { return MLMG::BottomSolver::bicgstab; }
    if (name == "cg")       { return MLMG::BottomSolver::cg; }
    if (name == "bicgcg")   { return MLMG::BottomSolver::bicgcg; }
    if (name == "cgbicg")   { return MLMG::BottomSolver::cgbicg; }
    if (name == "smoother") { return MLMG::BottomSolver::smoother; }
    amrex::Abort("Unknown mlmg.bottom_solver: " + name);
    return MLMG::BottomSolver::Default;
}

// This is the level-zero portion of AmrMesh::ChopGrids used by WarpX when
// amr.refine_grid_layout is enabled.  It preserves max_grid_size and the
// blocking factor while trying to create at least one grid per MPI rank.
BoxArray
makeGrids (Box const& domain, Params const& p)
{
    IntVect max_size(p.max_grid_size);
    BoxArray grids(domain);
    grids.maxSize(max_size);

    if (!p.refine_grid_layout) { return grids; }

    IntVect chunk = max_size;
    chunk.min(domain.length());
    int const target_size = ParallelDescriptor::NProcs();

    while (static_cast<int>(grids.size()) < target_size) {
        bool chopped = false;
        std::array<std::pair<int,int>,AMREX_SPACEDIM> chunk_dir{
            AMREX_D_DECL(std::make_pair(chunk[0], 0),
                         std::make_pair(chunk[1], 1),
                         std::make_pair(chunk[2], 2))};
        std::sort(chunk_dir.begin(), chunk_dir.end());

        for (int idx = AMREX_SPACEDIM-1; idx >= 0; --idx) {
            int const idim = chunk_dir[idx].second;
            int const new_chunk_size = chunk[idim] / 2;
            if (new_chunk_size > 0 && new_chunk_size % p.blocking_factor == 0) {
                chunk[idim] = new_chunk_size;
                grids.maxSize(chunk);
                chopped = true;
                break;
            }
        }
        if (!chopped) { break; }
    }

    return grids;
}

void
printBoxInfo (BoxArray const& grids)
{
    IntVect min_size(std::numeric_limits<int>::max());
    IntVect max_size(std::numeric_limits<int>::lowest());
    for (int i = 0, n = static_cast<int>(grids.size()); i < n; ++i) {
        IntVect const size = grids[i].length();
        min_size.min(size);
        max_size.max(size);
    }
    Print() << "  grids              : " << grids.size()
            << " (min " << min_size << ", max " << max_size << ")\n";
}

struct DebugFDLaplacian : public MLEBNodeFDLaplacian
{
    void printHierarchy () const
    {
        Print() << "  MG hierarchy:\n";
        for (int mglev = 0; mglev < m_num_mg_levels[0]; ++mglev) {
            BoxArray const& grids = m_grids[0][mglev];
            auto const& pmap = m_dmap[0][mglev].ProcessorMap();
            std::set<int> owners(pmap.begin(), pmap.end());
            Print() << "    level " << mglev
                    << ": domain " << m_geom[0][mglev].Domain().length()
                    << ", boxes " << grids.size()
                    << ", ranks " << owners.size() << "\n";
        }
    }
};

void
main_main ()
{
    static_assert(AMREX_SPACEDIM == 3, "This reproducer is 3D only");

    Params p;
    p.read();

    Box const domain(IntVect(0),
                     IntVect(AMREX_D_DECL(p.n_cell[0]-1,
                                         p.n_cell[1]-1,
                                         p.n_cell[2]-1)));
    RealBox const real_box(p.prob_lo.data(), p.prob_hi.data());
    Geometry const geom(domain, &real_box, CoordSys::cartesian,
                        p.is_periodic.data());

    BoxArray const grids = makeGrids(domain, p);
    DistributionMapping const dmap(grids);

    Parser eb_parser(p.eb_implicit_function);
    eb_parser.registerVariables({"x", "y", "z"});
    EB2::ParserIF const parser_if(eb_parser.compile<3>());
    EB2::GeometryShop<EB2::ParserIF,Parser> const shop(parser_if, eb_parser);
    EB2::Build(shop, geom, 0, 20);

    auto factory = makeEBFabFactory(&EB2::IndexSpace::top(), geom, grids, dmap,
                                    Vector<int>(AMREX_SPACEDIM, 2),
                                    EBSupport::full);

    BoxArray const nodal_grids = amrex::convert(grids, IntVect::TheNodeVector());
    MultiFab phi(nodal_grids, dmap, 1, 1, MFInfo{}, *factory);
    MultiFab rhs(nodal_grids, dmap, 1, 0, MFInfo{}, *factory);
    phi.setVal(0.0_rt);
    rhs.setVal(0.0_rt);
    rhs.OverrideSync(geom.periodicity());

    Array<LinOpBCType,AMREX_SPACEDIM> lobc;
    Array<LinOpBCType,AMREX_SPACEDIM> hibc;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        lobc[idim] = hibc[idim] = geom.isPeriodic(idim)
            ? LinOpBCType::Periodic : LinOpBCType::Neumann;
    }

    LPInfo info;
    info.setMaxCoarseningLevel(p.max_coarsening_level);

    DebugFDLaplacian linop;
    linop.setVerbose(p.linop_verbose);
    linop.define(Vector<Geometry>{geom}, Vector<BoxArray>{grids},
                 Vector<DistributionMapping>{dmap}, info,
                 Vector<EBFArrayBoxFactory const*>{factory.get()});
    linop.setSigma({1.0_rt, 1.0_rt, 1.0_rt});

    Parser potential_parser(p.eb_potential);
    potential_parser.registerVariables({"x", "y", "z"});
    auto const potential = potential_parser.compile<3>();
    linop.setEBDirichlet(
        [=] AMREX_GPU_HOST_DEVICE (Real x, Real y, Real z) noexcept -> Real
        {
            return static_cast<Real>(potential(x, y, z));
        });
    linop.setDomainBC(lobc, hibc);

    MLMG mlmg(linop);
    mlmg.setVerbose(p.verbose);
    mlmg.setBottomVerbose(p.bottom_verbose);
    mlmg.setMaxIter(p.max_iter);
    mlmg.setFinalSmooth(p.final_smooth);
    mlmg.setBottomSolver(parseBottomSolver(p.bottom_solver));
    mlmg.setBottomMaxIter(p.bottom_max_iter);
    mlmg.setBottomTolerance(p.bottom_reltol);
    mlmg.setConvergenceNormType(MLMGNormType::greater);
    mlmg.setNoGpuSync(p.no_gpu_sync);

    Print() << "\nWarpX EB MLEBNodeFDLaplacian reproducer\n"
            << "  domain             : " << domain.length() << " cells\n"
            << "  prob_lo            : " << p.prob_lo << "\n"
            << "  prob_hi            : " << p.prob_hi << "\n"
            << "  periodicity        : " << p.is_periodic << "\n"
            << "  domain BC          : Neumann / Periodic / Periodic\n"
            << "  MPI ranks          : " << ParallelDescriptor::NProcs() << "\n";
    printBoxInfo(grids);
    Print() << "  MG levels          : " << linop.NMGLevels(0) << "\n"
            << "  phi norm before    : " << phi.norm0() << "\n"
            << "  rhs norm before    : " << rhs.norm0() << "\n"
            << "  EB potential       : " << p.eb_potential << "\n\n";
    linop.printHierarchy();
    Print() << "\n";

    mlmg.solve({&phi}, {&rhs}, p.reltol, p.abstol);
    Gpu::streamSynchronize();

    Print() << "\nSolve diagnostics\n"
            << "  initial rhs norm   : " << mlmg.getInitRHS() << "\n"
            << "  initial residual   : " << mlmg.getInitResidual() << "\n"
            << "  final residual     : " << mlmg.getFinalResidual() << "\n"
            << "  MLMG iterations    : " << mlmg.getNumIters() << "\n"
            << "  phi norm after     : " << phi.norm0() << "\n"
            << "  bottom iterations  :";
    for (int n : mlmg.getNumCGIters()) { Print() << " " << n; }
    Print() << "\n\n";
}

} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
