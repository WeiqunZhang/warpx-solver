// Analytic stand-in for WarpX's PC mass matrices (no particles).
//
// sigma(iv,d) = c^2 mu0 theta dt * sum_s n q_s^2 dt/(2 m_s) (b_d^2+1)/(1+b^2) * K(d) * (1 + noise*xi),
// i.e. WarpX's fpxx (MassMatricesDeposition.H, gamma = 1, b = q dt B0/(2m)) with the
// PC scaling of ImplicitSolver.cpp:1138. K is the autocorrelation of the shape-2
// deposit: quadratic in nodal directions, linear in centered ones. n is evaluated at
// the pair midpoint and xi is hashed from the unordered pair, so the block is
// symmetric like WarpX's folded matrices.
#ifndef WPC_MASS_MATRIX_H_
#define WPC_MASS_MATRIX_H_

#include "WarpXMatrix.H"

#include <AMReX_Parser.H>

#include <cstdint>

namespace wpc {

namespace PhysConst {
    constexpr amrex::Real c    = 299792458.0;
    constexpr amrex::Real ep0  = 8.8541878128e-12;
    constexpr amrex::Real q_e  = 1.602176634e-19;
    constexpr amrex::Real m_e  = 9.1093837015e-31;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::uint64_t splitmix64 (std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

//! Uniform in [-1,1) from two keys.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hash_uniform (std::uint64_t a, std::uint64_t b)
{
    auto h = splitmix64(a ^ splitmix64(b));
    return amrex::Real(double(h >> 11) * (2.0/9007199254740992.0) - 1.0);
}

struct Species
{
    std::string name;
    amrex::Real charge = -1;        // units of e
    amrex::Real mass = 1;           // units of m_e
    amrex::Real density_scale = 1;  // times mm.density
};

//! Number of mass-matrix components per direction for E component dir and
//! stencil half-width w (MM_ncomp in MatrixPC::Assemble).
inline amrex::GpuArray<amrex::GpuArray<int,3>,3> MassMatricesNComp (int w, int shape)
{
    amrex::GpuArray<amrex::GpuArray<int,3>,3> r{};
    for (int dir = 0; dir < 3; ++dir) {
        auto const ixt = EIndexType(dir);
        for (int sd = 0; sd < 3; ++sd) {
            if (sd < AMREX_SPACEDIM) {
                int const full = ixt.nodeCentered(sd) ? 1+2*shape : 1+2*(shape-1);
                r[dir][sd] = std::min(1+2*w, full);
            } else {
                r[dir][sd] = 1;
            }
        }
    }
    return r;
}

struct MassMatrixParams
{
    std::string density = "1.e23";
    amrex::Vector<Species> species;
    amrex::Array<amrex::Real,3> B0{0,0,0};
    amrex::Real noise = 0;
    int seed = 1;
};

inline void FillMassMatrices (amrex::Array<amrex::MultiFab,3>& sigma,
                              amrex::GpuArray<amrex::GpuArray<int,3>,3> const& MM_ncomp,
                              amrex::Array<amrex::MultiFab,3> const& E,
                              amrex::Geometry const& geom,
                              MassMatrixParams const& p,
                              amrex::Real dt, amrex::Real theta,
                              amrex::Real density_factor, int seed)
{
    using namespace amrex;
    BL_PROFILE("FillMassMatrices");

    Parser parser(p.density);
    parser.registerVariables({"x","y","z"});
    auto const nfunc = parser.compile<3>();

    // c^2 mu0 theta dt = theta dt / eps0
    Real const pc_factor = theta*dt/PhysConst::ep0;

    GpuArray<Real,3> coef{0,0,0};
    for (auto const& s : p.species) {
        Real const q = s.charge*PhysConst::q_e;
        Real const m = s.mass*PhysConst::m_e;
        Real const a = q*0.5_rt*dt/m;
        Real const bx = a*p.B0[0], by = a*p.B0[1], bz = a*p.B0[2];
        Real const b2 = bx*bx + by*by + bz*bz;
        Real const arog = s.density_scale*q*q*0.5_rt*dt/m/(1.0_rt+b2);
        coef[0] += arog*(bx*bx+1.0_rt);
        coef[1] += arog*(by*by+1.0_rt);
        coef[2] += arog*(bz*bz+1.0_rt);
    }

    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx = geom.CellSizeArray();
    auto const ncell = geom.Domain().length();
    GpuArray<int,AMREX_SPACEDIM> is_per{};
    for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) { is_per[sd] = geom.isPeriodic(sd); }
    Real const noise = p.noise;
    auto const useed = std::uint64_t(seed);

    for (int dir = 0; dir < 3; ++dir) {
        int ncomp = MM_ncomp[dir][0]*MM_ncomp[dir][1]*MM_ncomp[dir][2];
        sigma[dir].define(E[dir].boxArray(), E[dir].DistributionMap(), ncomp, 0);
        auto const nc = MM_ncomp[dir];
        GpuArray<int,AMREX_SPACEDIM> nodal{};
        for (int sd = 0; sd < AMREX_SPACEDIM; ++sd) { nodal[sd] = EIndexType(dir).nodeCentered(sd); }
        Real const cf = pc_factor*density_factor*coef[dir];
        auto const& ma = sigma[dir].arrays();
        ParallelFor(sigma[dir], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            constexpr Real K5[3] = {66.0_rt/120.0_rt, 26.0_rt/120.0_rt, 1.0_rt/120.0_rt};
            constexpr Real K3[2] = {4.0_rt/6.0_rt, 1.0_rt/6.0_rt};
            IntVect const iv(AMREX_D_DECL(i,j,k));
            int mm_comp = 0;
            for (int comp2 = 0; comp2 < nc[2]; comp2++) {
            for (int comp1 = 0; comp1 < nc[1]; comp1++) {
            for (int comp0 = 0; comp0 < nc[0]; comp0++) {
                int const dd[3] = {comp0-(nc[0]-1)/2, comp1-(nc[1]-1)/2, comp2-(nc[2]-1)/2};
                Real K = 1.0_rt;
                Real xm[3] = {0,0,0};
                std::uint64_t Lp = 0, Lq = 0;
                for (int sd = AMREX_SPACEDIM-1; sd >= 0; --sd) {
                    int const ad = dd[sd] < 0 ? -dd[sd] : dd[sd];
                    K *= nodal[sd] ? K5[ad] : K3[ad];
                    Real x = plo[sd] + (Real(iv[sd]) + (nodal[sd] ? 0.0_rt : 0.5_rt)
                                        + 0.5_rt*Real(dd[sd]))*dx[sd];
                    int ip = iv[sd], iq = iv[sd]+dd[sd];
                    if (is_per[sd]) {
                        Real const L = phi[sd]-plo[sd];
                        x -= L*std::floor((x-plo[sd])/L);
                        ip = ((ip % ncell[sd]) + ncell[sd]) % ncell[sd];
                        iq = ((iq % ncell[sd]) + ncell[sd]) % ncell[sd];
                    }
                    xm[sd] = x;
                    auto const stride = std::uint64_t(ncell[sd]+8);
                    Lp = Lp*stride + std::uint64_t(ip+4);
                    Lq = Lq*stride + std::uint64_t(iq+4);
                }
#if (AMREX_SPACEDIM == 2)
                Real const n = nfunc(xm[0], 0.0_rt, xm[1]);
#else
                Real const n = nfunc(xm[0], xm[1], xm[2]);
#endif
                Real xi = 0.0_rt;
                if (noise > 0.0_rt) {
                    auto const lo = Lp < Lq ? Lp : Lq;
                    auto const hi = Lp < Lq ? Lq : Lp;
                    xi = hash_uniform(lo*4+std::uint64_t(dir), hi ^ (useed*0x632be59bd9b4e019ULL));
                }
                ma[b](i,j,k,mm_comp) = cf*n*K*(1.0_rt + noise*xi);
                ++mm_comp;
            }}}
        });
    }
    Gpu::streamSynchronize();
}

}

#endif
