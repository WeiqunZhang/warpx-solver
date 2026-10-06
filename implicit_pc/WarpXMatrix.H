// DOF numbering, curl-curl BC masks and PC matrix assembly, copied from WarpX
// (~/git/warpx, development ba7c87d7a) with the WarpX class plumbing removed.
#ifndef WARPX_MATRIX_H_
#define WARPX_MATRIX_H_

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Scan.H>

#include <limits>
#include <memory>

#if (AMREX_SPACEDIM == 2)
#  define WARPX_DIM_XZ
#elif (AMREX_SPACEDIM == 3)
#  define WARPX_DIM_3D
#else
#  error "Only 2D XZ and 3D are supported"
#endif

namespace wpc {

enum struct FieldBoundaryType { Periodic, PEC, PMC };

//! Yee staggering of E: Ex/Ey/Ez index types.
inline amrex::IndexType EIndexType (int dir)
{
#if defined(WARPX_DIM_3D)
    amrex::IntVect iv(1,1,1);
    iv[dir] = 0;
#else
    // XZ: Ex (cell x, node z), Ey (node, node), Ez (node x, cell z)
    amrex::IntVect iv(1,1);
    if (dir == 0) { iv[0] = 0; }
    if (dir == 2) { iv[1] = 0; }
#endif
    return amrex::IndexType(iv);
}

// WarpXSolverDOF.cpp: one iMultiFab per E component with {local, global} ids.
struct WarpXSolverDOF
{
    amrex::Array<std::unique_ptr<amrex::iMultiFab>,3> m_array;
    amrex::Long m_nDoFs_l = 0;
    amrex::Long m_nDoFs_g = 0;
    amrex::Long m_proc_begin = 0;

    void Define (amrex::Array<amrex::MultiFab,3> const& E, amrex::Periodicity const& period)
    {
        using namespace amrex;
        m_nDoFs_l = 0;
        for (int n = 0; n < 3; n++) {
            m_array[n] = std::make_unique<iMultiFab>(E[n].boxArray(), E[n].DistributionMap(),
                                                     2, E[n].nGrowVect());
            // WarpX.cpp SetDotMask
            const auto tmp = MultiFab{E[n].boxArray(), E[n].DistributionMap(), 1, 0,
                                      MFInfo().SetAlloc(false)};
            auto mask = tmp.OwnerMask(period);
            fill_local_dof(*m_array[n], *mask);
        }
        fill_global_dof();
        for (int n = 0; n < 3; n++) {
            m_array[n]->FillBoundaryAndSync(1, 1, m_array[n]->nGrowVect(), period);
        }
        amrex::Print() << "Defined DOF object for linear solves (total DOFs = " << m_nDoFs_g << ").\n";
    }

    void fill_local_dof (amrex::iMultiFab& dof, amrex::iMultiFab const& mask)
    {
        using namespace amrex;
        AMREX_ALWAYS_ASSERT(dof.boxArray().numPts() < static_cast<Long>(std::numeric_limits<int>::max()));
        dof.setVal(std::numeric_limits<int>::lowest());
#ifdef AMREX_USE_MPI
        const int nprocs = ParallelDescriptor::NProcs();
#endif
        for (MFIter mfi(dof); mfi.isValid(); ++mfi) {
            Box const& vbx = mfi.validbox();
            const auto npts = static_cast<int>(vbx.numPts());
            const BoxIndexer boxindex(vbx);
            auto const& m = mask.const_array(mfi);
            auto const& d = dof.array(mfi);
            const auto start_id = static_cast<int>(m_nDoFs_l);
            auto ndofs = Scan::PrefixSum<int>(
                npts,
                [=] AMREX_GPU_DEVICE (int offset) -> int
                {
                    auto [i,j,k] = boxindex(offset);
                    return m(i,j,k) ? 1 : 0;
                },
                [=] AMREX_GPU_DEVICE (int offset, int ps)
                {
                    auto [i,j,k] = boxindex(offset);
                    if (m(i,j,k)) {
                        d(i,j,k,0) = ps + start_id;
#ifdef AMREX_USE_MPI
                        if (nprocs == 1)
#endif
                        {
                            d(i,j,k,1) = ps + start_id;
                        }
                    }
                },
                Scan::Type::exclusive, Scan::retSum);
            m_nDoFs_l += Long(ndofs);
        }
    }

    void fill_global_dof ()
    {
        using namespace amrex;
        m_proc_begin = 0;
#ifndef AMREX_USE_MPI
        m_nDoFs_g = m_nDoFs_l;
#else
        const int nprocs = ParallelDescriptor::NProcs();
        if (nprocs == 1) {
            m_nDoFs_g = m_nDoFs_l;
        } else {
            Vector<Long> ndofs_allprocs(nprocs);
            MPI_Allgather(&m_nDoFs_l, 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                          ndofs_allprocs.data(), 1, ParallelDescriptor::Mpi_typemap<Long>::type(),
                          ParallelDescriptor::Communicator());
            Long proc_begin = 0;
            const int myproc = ParallelDescriptor::MyProc();
            m_nDoFs_g = 0;
            for (int iproc = 0; iproc < nprocs; ++iproc) {
                if (iproc < myproc) {
                    proc_begin += ndofs_allprocs[iproc];
                }
                m_nDoFs_g += ndofs_allprocs[iproc];
            }
            m_proc_begin = proc_begin;
            for (auto& y : m_array) {
                auto const& dof = y->arrays();
                ParallelFor(*y, IntVect(0), [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
                {
                    dof[b](i,j,k,1) = dof[b](i,j,k,0) + int(proc_begin);
                });
            }
            Gpu::streamSynchronize();
        }
#endif
    }
};

// ThetaImplicitEM::InitializeCurlCurlBCMasks (ThetaImplicitEM.cpp:201-503), XZ and
// 3D, PEC/PMC/periodic only.
inline void InitializeCurlCurlBCMasks (amrex::Array<amrex::MultiFab,3>& curl2_BC_mask,
                                       amrex::Array<amrex::MultiFab,3> const& E,
                                       amrex::Geometry const& geom,
                                       amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& bc_type_lo,
                                       amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& bc_type_hi)
{
    using namespace amrex;
#if defined(WARPX_DIM_XZ)
    const int ncomps[3] = {3, 4, 3};
#else
    const int ncomps[3] = {6, 6, 6};
#endif
    for (int n = 0; n < 3; ++n) {
        curl2_BC_mask[n].define(E[n].boxArray(), E[n].DistributionMap(), ncomps[n], 0);
        curl2_BC_mask[n].setVal(1.0_rt);
    }

    amrex::Box domain_box = geom.Domain();
    domain_box.convert(amrex::IntVect::TheNodeVector());
    const amrex::IntVect domain_lo = domain_box.smallEnd();
    const amrex::IntVect domain_hi = domain_box.bigEnd();

#if AMREX_SPACEDIM < 3
    // Set the BC masks for the out-of-plane components of the curl curl E operator
    for (amrex::MFIter mfi(curl2_BC_mask[1], false); mfi.isValid(); ++mfi) {
        const amrex::Box node_box = amrex::convert(mfi.validbox(),amrex::IntVect::TheNodeVector());
        for (int bdry_dir = 0; bdry_dir < AMREX_SPACEDIM; ++bdry_dir) {
            if (bc_type_lo[bdry_dir] == FieldBoundaryType::Periodic) { continue; }
            for (int bdry_side = 0; bdry_side < 2; ++bdry_side) {
                if (bdry_side == 0 && node_box.smallEnd()[bdry_dir] != domain_lo[bdry_dir]) {
                    continue;
                }
                if (bdry_side == 1 && node_box.bigEnd()[bdry_dir] != domain_hi[bdry_dir]) {
                    continue;
                }
                amrex::Box bdry_box = node_box;
                if (bdry_side == 0) { bdry_box.setBig(bdry_dir,domain_lo[bdry_dir]); }
                if (bdry_side == 1) { bdry_box.setSmall(bdry_dir,domain_hi[bdry_dir]); }
                amrex::Real val0 = 1.0_rt;
                amrex::Real val1 = 1.0_rt;
                const FieldBoundaryType bc_type = (bdry_side == 0) ? bc_type_lo[bdry_dir]:bc_type_hi[bdry_dir];
                if (bc_type == FieldBoundaryType::PEC){
                    val0 = 0.0_rt;
                    val1 = 0.0_rt;
                }
                if (bc_type == FieldBoundaryType::PMC){
                    val0 = 1.0_rt;
                    val1 = 2.0_rt;
                }
                amrex::Array4<amrex::Real> const& maskEy_arr = curl2_BC_mask[1].array(mfi);
                amrex::ParallelFor(bdry_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    maskEy_arr(i,j,k,2*bdry_dir  ) = val0;
                    maskEy_arr(i,j,k,2*bdry_dir+1) = val1;
                });
            }
        }
    }
#endif

    // Set the BC masks for the in-plane components of the curl curl E operator
    for (amrex::MFIter mfi(curl2_BC_mask[0], false); mfi.isValid(); ++mfi) {
        for (int bdry_dir = 0; bdry_dir < AMREX_SPACEDIM; ++bdry_dir) {
            if (bc_type_lo[bdry_dir] == FieldBoundaryType::Periodic) { continue; }
            for (int field_dir = 0; field_dir < 3; ++field_dir) {
#if AMREX_SPACEDIM == 3
                if (field_dir == bdry_dir) { continue; }
                const int tdir1 = (field_dir + 1) % AMREX_SPACEDIM; // next direction after field_dir
#else
                if (field_dir == 1) { continue; } // this is out-of-plane E in 2D
                if (bdry_dir == 0 && field_dir == 0) { continue; } // Ex is centered in bdry_dir = 0
                if (bdry_dir == 1 && field_dir == 2) { continue; } // Ez is centered in bdry_dir = 1
                const int tdir1 = bdry_dir;
#endif
                const amrex::IntVect Edir_nodal = curl2_BC_mask[field_dir].ixType().toIntVect();
                const amrex::Box edge_box = amrex::convert(mfi.validbox(),Edir_nodal);
                for (int bdry_side = 0; bdry_side < 2; ++bdry_side) {
                    if (bdry_side == 0 && edge_box.smallEnd()[bdry_dir] != domain_lo[bdry_dir]) {
                        continue;
                    }
                    if (bdry_side == 1 && edge_box.bigEnd()[bdry_dir] != domain_hi[bdry_dir]) {
                        continue;
                    }
                    amrex::Box bdry_box = edge_box;
                    if (bdry_side == 0) { bdry_box.setBig(bdry_dir,domain_lo[bdry_dir]); }
                    if (bdry_side == 1) { bdry_box.setSmall(bdry_dir,domain_hi[bdry_dir]); }
                    amrex::Real val0 = 1.0_rt;
                    amrex::Real val1 = 1.0_rt;
                    amrex::Real val2 = 1.0_rt;
                    const FieldBoundaryType bc_type = (bdry_side == 0) ? bc_type_lo[bdry_dir]:bc_type_hi[bdry_dir];
                    if (bc_type == FieldBoundaryType::PEC){
                        val0 = 0.0_rt;
                        val1 = 0.0_rt;
                        val2 = 0.0_rt;
                    }
                    if (bc_type == FieldBoundaryType::PMC){
                        val0 = 1.0_rt;
                        val1 = 2.0_rt;
                        val2 = 2.0_rt;
                    }
                    const int comp_shift = (tdir1 == bdry_dir) ? 0 : 3;
                    amrex::Array4<amrex::Real> const& mask_arr = curl2_BC_mask[field_dir].array(mfi);
                    amrex::ParallelFor(bdry_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        mask_arr(i,j,k,comp_shift+0) = val0;
                        mask_arr(i,j,k,comp_shift+1) = val1;
                        mask_arr(i,j,k,comp_shift+2) = val2;
                    });
                }
            }
        }
    }
}

namespace MatrixPCUtils
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool insertOrAdd( const int a_cidx, /*!< Column index (global) */
                      const amrex::Real a_val, /*!< Value to insert/add */
                      int* const a_cidxs, /*!< Column index array */
                      amrex::Real* const a_aij, /*!< array of values */
                      const int a_nnz, /*!< max number of non-zero columns */
                      int& a_ncol /*!< number of non-zero columns */)
    {
        if (a_cidx < 0) { return true; /* outside domain */ }
        int loc = -1;
        for (int icol = 0; icol < std::min(a_ncol,a_nnz); icol++) {
            if (a_cidxs[icol] == a_cidx) {
                loc = icol;
                break;
            }
        }
        if (loc < 0) {
            a_ncol++;
            if (a_ncol > a_nnz) { return false; }
            else {
                a_cidxs[a_ncol-1] = a_cidx;
                a_aij[a_ncol-1] = a_val;
            }
        } else {
            a_aij[loc] += a_val;
        }
        return true;
    }
}

//! MatrixPC (MatrixPC.H) without the Ops/T plumbing. Holds the padded
//! row-major arrays handed to PETSc.
struct MatrixPC
{
    int m_ndofs_l = 0;
    int m_pc_mat_nnz = 1;
    int m_num_realloc = 0;

    amrex::Gpu::DeviceVector<int> m_r_indices_g;
    amrex::Gpu::DeviceVector<int> m_num_nz;
    amrex::Gpu::DeviceVector<int> m_c_indices_g;
    amrex::Gpu::DeviceVector<amrex::Real> m_a_ij;

    // Inputs (set before Update)
    WarpXSolverDOF const* m_dofs = nullptr;
    amrex::Geometry m_geom;
    amrex::Array<amrex::MultiFab,3> const* m_curl2_BC_mask = nullptr;
    amrex::Array<amrex::MultiFab,3> const* m_bcoefs = nullptr; // mass matrices, may be null
    amrex::GpuArray<amrex::GpuArray<int,3>,3> m_MM_ncomp{}; // [dir][space_dir]
    amrex::Real m_thetaDt = 0;

    void Update ()
    {
        BL_PROFILE("MatrixPC::Update()");
        while(true) {
            auto nnz_diff = Assemble();
            AMREX_ALWAYS_ASSERT(nnz_diff >= 0);
            if (nnz_diff) {
                m_pc_mat_nnz += nnz_diff;
                m_num_realloc++;
            } else {
                break;
            }
        }
    }

    // MatrixPC::Assemble (MatrixPC.H:316-814), XZ and 3D branches.
    int Assemble ()
    {
        BL_PROFILE("MatrixPC::Assemble()");
        using namespace amrex;

        const Real thetaDt = m_thetaDt;
        constexpr Real c = 299792458.0_rt;
        const Real alpha = (thetaDt*c) * (thetaDt*c);

        const auto& dofs_mfarrvec = m_dofs->m_array;
        m_ndofs_l = int(m_dofs->m_nDoFs_l);

        m_r_indices_g.clear();
        m_num_nz.clear();
        m_c_indices_g.clear();
        m_a_ij.clear();

        auto n_rows = size_t(m_ndofs_l);
        auto n_cols = size_t(m_pc_mat_nnz) * size_t(m_ndofs_l);

        m_r_indices_g.resize(n_rows);
        m_num_nz.resize(n_rows);
        m_c_indices_g.resize(n_cols);
        m_a_ij.resize(n_cols);

        auto* r_indices_g_ptr = m_r_indices_g.data();
        auto* num_nz_ptr = m_num_nz.data();
        auto* c_indices_g_ptr = m_c_indices_g.data();
        auto* a_ij_ptr = m_a_ij.data();

        const auto nnz_max = m_pc_mat_nnz;
        auto nnz_actual = nnz_max;

        const auto& geom = m_geom;
        const auto dxi = geom.InvCellSizeArray();

        Gpu::Buffer<int> nnz_actual_d({nnz_max});
        auto* nnz_actual_ptr = nnz_actual_d.data();

        for (int dir = 0; dir < 3; dir++) {

            for (amrex::MFIter mfi(*dofs_mfarrvec[dir]); mfi.isValid(); ++mfi) {

                const amrex::Box bx = mfi.tilebox();
                const amrex::Box full_bx = mfi.fabbox();

                auto dof_arr = dofs_mfarrvec[dir]->const_array(mfi);

                // Set row indices and identity diagonal (unconditional).
                For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const int ridx_l = dof_arr(i,j,k,0);
                    if (ridx_l < 0) { return; }

                    int icol = 0;
                    const int ridx_g = dof_arr(i,j,k,1);

                    r_indices_g_ptr[ridx_l] = ridx_g;

                    {
                        const int cidx_g_lhs = dof_arr(i,j,k,1);
                        const amrex::Real val = 1.0_rt;
                        auto flag = MatrixPCUtils::insertOrAdd( cidx_g_lhs, val,
                                                                &c_indices_g_ptr[ridx_l*nnz_max],
                                                                &a_ij_ptr[ridx_l*nnz_max],
                                                                nnz_max, icol );
                        if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                    }

                    num_nz_ptr[ridx_l] = icol;
                });

                // Add the curl-curl stencil entries (only when alpha > 0)
                if (thetaDt > 0.0) {

                    const auto BC_mask_Edir_arr = (*m_curl2_BC_mask)[dir].const_array(mfi);

#if defined(WARPX_DIM_XZ)
                    int tdir = -1;
                    if  (dir == 0) { tdir = 2; }
                    else if (dir == 2) { tdir = 0; }
                    else { tdir = 1; }
                    auto dof_tdir_arr = dofs_mfarrvec[tdir]->const_array(mfi);
#elif defined(WARPX_DIM_3D)
                    const int tdir1 = (dir + 1) % 3;
                    const int tdir2 = (dir + 2) % 3;
                    GpuArray<Array4<const int>, AMREX_SPACEDIM>
                        const dof_arrays {{ AMREX_D_DECL( dofs_mfarrvec[dir]->const_array(mfi),
                                                          dofs_mfarrvec[tdir1]->const_array(mfi),
                                                          dofs_mfarrvec[tdir2]->const_array(mfi) ) }};
#endif

                    For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        int icol = num_nz_ptr[ridx_l]; //NOLINT (misc-const-correctness)

#if defined(WARPX_DIM_XZ)
                        // dir = 0: xhat\cdot[\nabla\times\nabla E] = d/dx[dEz/dz] - [d2/dz2]Ex
                        // dir = 1: yhat\cdot[\nabla\times\nabla E] = -[d2/dx2 + d2/dz2]Ey
                        // dir = 2: zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx] - [d2/dx2]Ez
                        {
                            // diagonal component (i,j) of second derivative operator
                            const int cidx_g_rhs = dof_arr(i,j,k,1);
                            amrex::Real val = 0.0_rt;
                            if (dir == 0) {
                                val = 2.0_rt * alpha * dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 2) {
                                val = 2.0_rt * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0);
                            } else if (dir == 1) {
                                const amrex::Real geom_factor = 2.0_rt;
                                val = geom_factor * alpha * dxi[0]*dxi[0] * BC_mask_Edir_arr(i,j,k,0)
                                    +      2.0_rt * alpha * dxi[1]*dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                            }
                            auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        if ((dir == 0) || (dir == 2)) {
                            {
                                // left component (i-1,j) or (i,j-1) of second derivative operator
                                const amrex::Real geom_factor = 1.0_rt;
                                const int cidx_g_rhs = (dir == 0 ? dof_arr(i,j-1,k,1) : dof_arr(i-1,j,k,1));
                                const amrex::Real val = -geom_factor * alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                // right component (i+1,j) or (i,j+1) of second derivative operator
                                const amrex::Real geom_factor = 1.0_rt;
                                const int cidx_g_rhs = (dir == 0 ? dof_arr(i,j+1,k,1) : dof_arr(i+1,j,k,1));
                                const amrex::Real val = -geom_factor * alpha * dxi[dir==0?1:0] * dxi[dir==0?1:0] * BC_mask_Edir_arr(i,j,k,1);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            // mixed derivative terms of the curl curl operator
                            const amrex::Real geom_m = 1.0_rt;
                            const amrex::Real geom_p = 1.0_rt;
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i,j-1,k,1) : dof_tdir_arr(i-1,j,k,1));
                                const amrex::Real val = geom_m * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = dof_tdir_arr(i,j,k,1);
                                const amrex::Real val = -geom_p * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j-1,k,1) : dof_tdir_arr(i-1,j+1,k,1));
                                const amrex::Real val = -geom_m * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                            {
                                const int cidx_g_rhs = (dir == 0 ? dof_tdir_arr(i+1,j,k,1) : dof_tdir_arr(i,j+1,k,1));
                                const amrex::Real val = geom_p * alpha * dxi[0] * dxi[1] * BC_mask_Edir_arr(i,j,k,2);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                              &c_indices_g_ptr[ridx_l*nnz_max],
                                                                              &a_ij_ptr[ridx_l*nnz_max],
                                                                              nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        } else if (dir==1) {
                            for (int jdir = 0; jdir <= 2; jdir+=2) {
                                {
                                    const amrex::Real geom_factor = 1.0_rt;
                                    const int cidx_g_rhs = (jdir == 0 ? dof_arr(i-1,j,k,1) : dof_arr(i,j-1,k,1));
                                    const amrex::Real val = -geom_factor * alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                                  &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                  &a_ij_ptr[ridx_l*nnz_max],
                                                                                  nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                                {
                                    const amrex::Real geom_factor = 1.0_rt;
                                    const int cidx_g_rhs = (jdir == 0 ? dof_arr(i+1,j,k,1) : dof_arr(i,j+1,k,1));
                                    const amrex::Real val = -geom_factor * alpha * dxi[jdir==0?0:1] * dxi[jdir==0?0:1] * BC_mask_Edir_arr(i,j,k,jdir+1);
                                    const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                                  &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                  &a_ij_ptr[ridx_l*nnz_max],
                                                                                  nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#elif defined(WARPX_DIM_3D)
                        // xhat\cdot[\nabla\times\nabla E] = d/dx[dEy/dy + dEz/dz] - [d2/dy2 + d2/dz2]Ex
                        // yhat\cdot[\nabla\times\nabla E] = d/dy[dEx/dx + dEz/dz] - [d2/dx2 + d2/dz2]Ey
                        // zhat\cdot[\nabla\times\nabla E] = d/dz[dEx/dx + dEy/dy] - [d2/dx2 + d2/dy2]Ez
                        amrex::IntVect dvec(AMREX_D_DECL(dir,tdir1,tdir2));
                        const amrex::IntVect ic(AMREX_D_DECL(i,j,k));
                        {
                            // diagonal component (i,j,k) of second derivative operator
                            const int cidx_g_rhs = dof_arrays[0](ic,1);
                            const amrex::Real val = 2.0_rt * alpha * ( dxi[dvec[1]]*dxi[dvec[1]] * BC_mask_Edir_arr(i,j,k,0)
                                                               + dxi[dvec[2]]*dxi[dvec[2]] * BC_mask_Edir_arr(i,j,k,3) );
                            const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                    &c_indices_g_ptr[ridx_l*nnz_max],
                                                                    &a_ij_ptr[ridx_l*nnz_max],
                                                                    nnz_max, icol );
                            if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                        }
                        // left and right components (i+/-1,j,k), (i,j+/-1,k), or (i,j,k+/-1) of second derivative operator
                        for (int ctr = -1; ctr <= 1; ctr += 2) {
                            for (int tdir = 1; tdir <= 2; tdir++) {
                                auto iv = ic; iv[dvec[tdir]] += ctr;
                                const int cidx_g_rhs = dof_arrays[0](iv,1);
                                const int comp_shift = (dvec[tdir] == tdir1) ? 0:3;
                                const amrex::Real val = -alpha * dxi[dvec[tdir]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+1);
                                const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                        &c_indices_g_ptr[ridx_l*nnz_max],
                                                                        &a_ij_ptr[ridx_l*nnz_max],
                                                                        nnz_max, icol );
                                if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                            }
                        }
                        // mixed derivative terms of the curl curl operator
                        for (int ctr_dir = -1; ctr_dir <= 0; ctr_dir++) {
                            for (int ctr_tdir = -1; ctr_tdir <= 0; ctr_tdir++) {
                                for (int tdir = 1; tdir <= 2; tdir++) {
                                    auto iv = ic; iv[dvec[0]] += (ctr_dir+1); iv[dvec[tdir]] += ctr_tdir;
                                    const auto sign = std::copysign(1,ctr_dir) * std::copysign(1,ctr_tdir);
                                    const int cidx_g_rhs = dof_arrays[tdir](iv,1);
                                    const int comp_shift = (dvec[tdir] == tdir1) ? 0:3;
                                    const amrex::Real val = amrex::Real(sign) * alpha * dxi[dvec[0]]*dxi[dvec[tdir]] * BC_mask_Edir_arr(i,j,k,comp_shift+2);
                                    const auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                            &c_indices_g_ptr[ridx_l*nnz_max],
                                                                            &a_ij_ptr[ridx_l*nnz_max],
                                                                            nnz_max, icol );
                                    if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                }
                            }
                        }
#endif
                        num_nz_ptr[ridx_l] = icol;
                    });
                }

                // Add the mass matrix piece
                // mm_comp = i0 + MM_width[0] + MM_comp[0]*(j0 + MM_width[1]) + MM_comp[0]*MM_comp[1]*(k0 + MM_width[2])
                if (m_bcoefs != nullptr) {

                    auto sigma_ii_arr = (*m_bcoefs)[dir].const_array(mfi);
                    amrex::GpuArray<int,3> MM_ncomp = {1,1,1};
                    amrex::GpuArray<int,3> MM_width = {0,0,0};
                    for (int space_dir=0; space_dir<AMREX_SPACEDIM; space_dir++) {
                        MM_ncomp[space_dir] = m_MM_ncomp[dir][space_dir];
                        MM_width[space_dir] = (MM_ncomp[space_dir] - 1)/2;
                    }

                    For(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const int ridx_l = dof_arr(i,j,k,0);
                        if (ridx_l < 0) { return; }

                        const amrex::IntVect iv_base = IntVect(AMREX_D_DECL(i,j,k));
                        int icol = num_nz_ptr[ridx_l];

                        int mm_comp = 0;
                        for (int comp2 = 0; comp2 < MM_ncomp[2]; comp2++) {
                            [[maybe_unused]] const int kk0 = comp2 - MM_width[2];
                            for (int comp1 = 0; comp1 < MM_ncomp[1]; comp1++) {
                                [[maybe_unused]] const int jj0 = comp1 - MM_width[1];
                                for (int comp0 = 0; comp0 < MM_ncomp[0]; comp0++) {
                                    const int ii0 = comp0 - MM_width[0]; // ..., -2, -1, 0, 1, 2, ...
                                    const amrex::IntVect iv_shift = IntVect(AMREX_D_DECL(ii0, jj0, kk0));
                                    if (full_bx.contains(iv_base + iv_shift)) {
                                        const int cidx_g_rhs = dof_arr(iv_base + iv_shift,1);
                                        const amrex::Real val = sigma_ii_arr(iv_base,mm_comp);
                                        auto flag = MatrixPCUtils::insertOrAdd( cidx_g_rhs, val,
                                                                                &c_indices_g_ptr[ridx_l*nnz_max],
                                                                                &a_ij_ptr[ridx_l*nnz_max],
                                                                                nnz_max, icol );
                                        if (!flag) { Gpu::Atomic::Max(nnz_actual_ptr, icol); }
                                    }
                                    ++mm_comp;
                                }
                            }
                        }

                        num_nz_ptr[ridx_l] = icol;
                    });
                }
                Gpu::synchronize();
            }
        }

        nnz_actual = std::max(nnz_actual, *(nnz_actual_d.copyToHost()));

        amrex::ParallelDescriptor::ReduceIntMax(&nnz_actual, 1);
        return (nnz_actual - nnz_max);
    }
};

}

#endif
