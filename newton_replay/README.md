# newton_replay

Replays the linear systems of Justin's 2D planar pinch (run-justin in the
WarpX tree: 256 x 32 cells, 400 particles per cell, theta-implicit EM with
mass matrices, 4 MPI ranks) outside WarpX: GMRES (amrex::GMRES_MV)
preconditioned by AlgMG, or PETSc GMRES with ASM + LU as WarpX's pc_petsc, on
the exact Jacobian and preconditioner matrices of the real run. Each Newton
iteration of each step is one system, so a dump of N steps replays the
N x (Newton iterations) linear solves, with WarpX's own iteration counts and
solutions for comparison. The dump records WarpX's per-rank DOF partition;
replayed on the same number of ranks, the rows (and so the ASM subdomains)
are distributed exactly as in WarpX.

The Newton residual itself needs the particles, so new Newton iterates cannot
be generated here; what is replayed is the sequence of linear solves WarpX did.

## 1. Dump the systems from WarpX

Apply `dump_linear_systems.patch` (also kept in `run-justin/` of the WarpX
tree) to WarpX, build, and run the case on a single MPI rank with

    newton.dump_linear_systems = 1
    newton.dump_dir = linsys          # default
    newton.dump_probe_width = 4       # half-width covering the Jacobian stencil

For every Newton iteration this writes `linsys/stepNNNN_iterNN/` with
`A.bin` (Jacobian, CSR), `P.bin` (pc_algmg / pc_petsc matrix, CSR), `b.bin`,
`x.bin` (WarpX's solution) and `info.txt` (sizes, theta*dt, Newton norms,
GMRES iterations and tolerance, probe self-check). The Jacobian is obtained by
probing WarpX's linear Jacobian operator with colored unit vectors; `info.txt`
reports `probe_check` (|Jv - Av|/|Jv| for a smooth v, should be ~1e-10) and
`linearity_check`. The reduced planar pinch (256x32 cells, 24640 DOFs) takes
about 20 MB per system.

`inputs_warpx_planar_pinch_2d` is the WarpX input used for the dump in
`linsys/`: run-justin's `planar_pinch_2d.in` (400 particles per cell) run on
4 MPI ranks for 10 steps with pc_algmg (degree 8, ratio 50), 20 systems,
547 MB. `linsys/` is not tracked by git; regenerate it with WarpX if it is
missing. Replay on 4 ranks to get WarpX's row partition.

## 2. Build and run

    make AMREX_HOME=/path/to/amrex -j16
    ./newton_replay2d.gnu.TPROF.MPI.ex inputs dump_dir=/path/to/linsys

Options are in `inputs`: `pc` (algmg on P, algmg_on_jacobian, none, petsc),
`pc_max_iter`, `gmres.*`, and `algmg.*` (smoother, chebyshev_degree,
chebyshev_ratio, strong_threshold, max_levels, ...). Rows are split evenly
across MPI ranks, so a dump made on one rank can be replayed on several.

`pc = petsc` mirrors WarpX's `newton.linear_solver = petsc_ksp` with
`jacobian.pc_type = pc_petsc`, `pc_petsc.type = asm`, `pc_petsc.sub_type = lu`:
PETSc GMRES (unpreconditioned norm, right preconditioning) on a shell operator
applying A, with ASM + LU built on P. `pc_petsc.asm_overlap` is the only PC
input; raw PETSc options go in `petsc.options`.
Set `USE_PETSC_TEST = FALSE` or `PETSC_DIR_TEST=...` in the GNUmakefile as
needed; the default PETSc is `$HOME/opt/petsc`.

The output table lists, per system, WarpX's GMRES iterations and the replayed
ones, AlgMG setup and GMRES solve time, the true relative residual of both
solutions, the relative difference of the solutions, the AlgMG level count
and operator complexity, the rows of the coarsest level and whether that is
within the 1024-row limit of AlgMG's LU bottom solver (`LU` column).

## 3. Coarsening-depth sweep

By default (`sweep_levels = 1`, `algmg.max_levels` unset) every system is
solved at AlgMG's full depth L and then with `max_levels = 1, ..., L-1`, so
one run compares stopping the coarsening early, with the LU bottom solve on
the CPU taking over at the coarsest level, against the default hierarchy.
A table is printed per depth, followed by a summary of the totals
(iterations, setup, solve, level and coarsest-size range, LU use). A system
whose full depth is below a table's `max_levels` contributes its full-depth
result there, which is what that `max_levels` gives for it. With
`max_levels = 1` the preconditioner is smoother sweeps on the fine level.
`algmg.max_levels = 2 3 4` runs exactly those depths; `sweep_levels = 0`
runs only the default. Timers synchronize the GPU, so the setup and solve
columns are wall-clock on GPU builds too.
