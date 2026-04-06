<img width="639" height="479" alt="pipe" src="https://github.com/user-attachments/assets/933945b6-88b7-4304-8c92-d8e5f875ad25" />


## Documentation: prototypes/full_version.cu
## For more information about the psysics involved,see the [report](./high_level_report.pdf)
## Overview

`full_version.cu` is a self-contained CUDA prototype that solves the **3D steady-state incompressible Navier-Stokes equations** for laminar pipe flow using the **Levenberg-Marquardt (LM) optimization method**. Instead of time-stepping, the code treats the discretized governing equations as a nonlinear system of residuals and drives them to zero via a damped Gauss-Newton iteration.

The solver operates entirely on the GPU using NVIDIA's CUDA toolkit and leverages several CUDA libraries:

| Library | Purpose |
|---------|---------|
| **cuSPARSE** | Sparse matrix format conversions (COO/CSR/CSC), sorting, transpose, SpGEMM, SpMV, matrix addition |
| **cuBLAS** | Dense vector norms (`cublasDdot`, `cublasDnrm2`) and AXPY updates |
| **cuDSS** | Direct sparse linear solver (LU-based analysis, factorization, solve) |
| **CUB** | Device-wide radix sort, segmented reduce, prefix scan, and flagged select |
| **cnpy** | NumPy `.npz` export for post-processing and visualization in Python |
| **OpenMP** | Multi-stream Jacobian construction  |

---

## Physical Problem

The code solves flow through a rectangular duct (pipe approximation) with:

- **Inlet** (x = 0): Prescribed parabolic velocity profile `u_inlet(y, z) = 16 * u0 * (y/M)(1 - y/M)(z/N)(1 - z/N)`
- **Outlet** (x = L): Zero-gradient (Neumann) for velocity; pinned zero pressure
- **Walls** (y and z boundaries): No-slip (`u = v = w = 0`); zero-gradient pressure

The state vector `y` is a 1D array of length `4 * xN * yN * zN` storing all unknowns interleaved as four blocks:

| Block | Indices | Variable |
|-------|---------|----------|
| 0 | `[0, nc)` | u-velocity |
| 1 | `[nc, 2*nc)` | v-velocity |
| 2 | `[2*nc, 3*nc)` | w-velocity |
| 3 | `[3*nc, 4*nc)` | pressure |

where `nc = xN * yN * zN` is the number of interior cells.

---

## Grid and Indexing

The computational domain is a uniform Cartesian grid of size `(xN+2) x (yN+2) x (zN+2)`, where the +2 accounts for one layer of ghost/boundary cells on each side.

### Index Functions

```cuda
// Standard 3D -> linear index (row-major: x varies slowest)
idx_3d(i, j, k, sizeY, sizeZ) = (i * sizeY + j) * sizeZ + k

// Batched 3D index with an extra "grain" dimension (for Jacobian batching)
idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain) = ((i * sizeY + j) * sizeZ + k) * grain + l
```

The `grain` parameter enables processing multiple finite-difference perturbations in a single kernel launch by interleaving data along the innermost dimension.

---

## File Structure

The file is organized into the following sections, listed in order of appearance:

### 1. Error Checking and Utilities (lines 31-84)

| Symbol | Description |
|--------|-------------|
| `CUDA_CHECK(call)` | Macro that wraps every CUDA API call and exits on error |
| `align_size(size, alignment)` | Rounds up `size` to the next multiple of `alignment` (default 256 bytes) for optimal GPU memory alignment |
| `print_gpu_array(d_array, n)` | Debug helper: copies a device array to host and prints all elements |

### 2. Index Functions (lines 50-63)

| Symbol | Description |
|--------|-------------|
| `idx_3d(i, j, k, sizeY, sizeZ)` | Maps 3D grid coordinates to a linear index. Available on both host and device (`__device__ __host__`) |
| `idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)` | Extended version for batched operations: appends a grain index `l` as the fastest-varying dimension |

### 3. Debug Print Functions (lines 66-179)

| Function | Description |
|----------|-------------|
| `print_gpu_array(d_array, n)` | Copies `n` doubles from GPU to host and prints them |
| `print_csr_matrix(rows, nnz, d_row_ptr, d_cols, d_vals)` | Downloads a CSR matrix from the GPU and prints row-by-row with `(col, value)` pairs |
| `print_coo_matrix_gpu(rows, cols, nnz, d_rows, d_cols, d_vals, max_print)` | Downloads a COO matrix and prints up to `max_print` triplets (default 20) |

### 4. CUDA Kernels -- Solution Batch Construction (lines 185-208)

| Kernel | Description |
|--------|-------------|
| `build_ysol_batch_kernel` | Creates `grain` copies of the state vector `y`, each with a single element perturbed by `h[t]` at position `t_batch[g]`. Used for batched finite-difference Jacobian computation |

### 5. CUDA Kernels -- Boundary Conditions (lines 210-570)

These kernels decompose the state vector into separate velocity/pressure fields and apply physical boundary conditions.

| Kernel | Description |
|--------|-------------|
| `boundary_conditions_initialization` | **Batched**: Unpacks the linearized state vector `ysol_batch` into separate 3D arrays `u, v, w, p` with ghost cells, for `grain` simultaneous perturbations |
| `boundary_conditions_apply` | **Batched**: Applies inlet, outlet, and wall boundary conditions to `u, v, w, p` for all `grain` copies simultaneously |
| `boundary_conditions_initialization_single` | **Single**: Same as above but for a single (non-batched) state vector |
| `boundary_conditions_apply_single` | **Single**: Same as above but for a single state vector |

#### Boundary Condition Summary

| Boundary | Velocity | Pressure |
|----------|----------|----------|
| **Walls** (y=0, y=M, z=0, z=N) | No-slip: `u = v = w = 0` | Zero-gradient (Neumann): copied from adjacent interior cell |
| **Inlet** (x=0) | `u = u_inlet(y,z)`, `v = w = 0` | Zero-gradient: `p[0] = p[1]` |
| **Outlet** (x=L) | Zero-gradient: copied from last interior cell | Pinned: `p = 0` (reference pressure) |

### 6. CUDA Kernels -- Navier-Stokes Residuals (lines 574-908)

These kernels evaluate the discretized residual `R(y)` of the incompressible Navier-Stokes equations using second-order central finite differences.

| Kernel | Equation | Output Range |
|--------|----------|--------------|
| `kernel_uv_velocity_single` | All four equations (u, v, w momentum + continuity) in a single kernel. **Used for single-state evaluation** | `out[0..4*nc-1]` |
| `kernel_u_momentum` | x-momentum: `conv_x + conv_y + conv_z + dp/dx - (1/Re)*laplacian(u)` | `out[l*NC + pos]` |
| `kernel_v_momentum` | y-momentum: `conv_x + conv_y + conv_z + dp/dy - (1/Re)*laplacian(v)` | `out[l*NC + nc + pos]` |
| `kernel_w_momentum` | z-momentum: `conv_x + conv_y + conv_z + dp/dz - (1/Re)*laplacian(w)` | `out[l*NC + 2*nc + pos]` |
| `kernel_continuity` | Continuity: `du/dx + dv/dy + dw/dz = 0` | `out[l*NC + 3*nc + pos]` |

The batched momentum/continuity kernels loop over `grain` perturbations, using stride-based indexing for coalesced memory access.

#### Discretization

All spatial derivatives use **second-order central differences**:

- **Convective terms**: `0.5 * dy*dz * (u_{i+1}^2 - u_{i-1}^2)` (conservative form)
- **Pressure gradient**: `(dy*dz / 2) * (p_{i+1} - p_{i-1})` (central difference)
- **Diffusion**: `(1/Re) * (dy*dz/dx) * (u_{i+1} - 2*u_i + u_{i-1})` (standard Laplacian stencil)

### 7. Jacobian Construction (lines 686-716)

| Kernel | Description |
|--------|-------------|
| `build_jacobian_entries` | Computes finite-difference Jacobian entries: `J[i, t] = (R_perturbed[i] - R_base[i]) / h[t]`. Non-zero entries (above threshold `1e-14`) are stored in COO format using `atomicAdd` on a global counter |

### 8. Host Functions -- Single-State Residual Evaluation (lines 911-1176)

| Function | Description |
|----------|-------------|
| `boundary_conditions_final(ysol, xN, yN, zN, u_inlet)` | Unpacks the state vector into separate `u, v, w, p` device arrays with boundary conditions applied. Returns a tuple of four device pointers. Caller must free |
| `uv_velocity_single(out, Re, y, ...)` | **Host-to-host**: Allocates GPU memory, evaluates the full residual `R(y)`, copies result back to host array `out`, frees all GPU memory |
| `uv_velocity_single(Re, y, ...)` | **Host-to-device**: Same but returns a device pointer to the residual. Caller must free |
| `uv_velocity_single_direct(Re, y, ...)` | **Device-to-device**: Same but expects `y` already on the GPU. Caller must free result |

### 9. Sparse Jacobian Computation (lines 1438-1718)

| Function | Description |
|----------|-------------|
| `Residuals_Sparse_Jacobian_finite_diff(Re, y, xN, yN, zN, u_inlet, dx, dy, dz)` | Main Jacobian computation. Uses batched finite differences with OpenMP-managed CUDA streams. Returns `pair<d_fold, tuple<d_rows, d_cols, d_vals, nnz>>` in COO format |

#### Algorithm:

1. Compute perturbation step sizes: `h[j] = eps * max(1, |y[j]|)` with Higham correction
2. Evaluate baseline residual `R(y)` using the batched kernel pipeline
3. For each chunk of `grain` columns:
   a. Build perturbed state vectors via `build_ysol_batch_kernel`
   b. Apply boundary conditions (batched)
   c. Evaluate residuals (batched momentum + continuity kernels)
   d. Compute `J[i,t] = (R_perturbed - R_base) / h[t]` and store non-zeros
4. Perform diagonal sanity check (count missing/zero diagonal entries)

### 10. Coordinate Generation (lines 1721-1748)

| Function | Description |
|----------|-------------|
| `coordinates(xcoor, ycoor, zcoor, xN, yN, zN, L, M, N)` | Generates a uniform Cartesian grid. Returns `(dx, dy, dz)` spacing. Fills coordinate vectors for all `(xN+2)*(yN+2)*(zN+2)` nodes |

### 11. Sparse Matrix Filtering (lines 1770-1916)

| Function | Description |
|----------|-------------|
| `generate_flags_kernel` | CUDA kernel: marks entries with `|value| > threshold` as 1 (keep), else 0 (discard) |
| `filter_csr_cub(threshold, rows, nnz, ...)` | Removes near-zero entries from a CSR matrix using CUB primitives. Pipeline: generate flags -> segmented reduce per row -> exclusive prefix sum for new offsets -> flagged select to compact columns and values |

### 12. cuSPARSE Error Checking (lines 1918-1926)

| Symbol | Description |
|--------|-------------|
| `CUSPARSE_CHECK(call)` | Macro for cuSPARSE error checking with file/line reporting |

### 13. Sparse Matrix Operations (lines 1928-2544)

| Function | Description |
|----------|-------------|
| `print_csc_matrix(...)` | Debug: prints a CSC matrix (not fully shown in prototype) |
| `compute_AtA_debug(...)` | Computes `J^T * J` (approximate Hessian) using cuSPARSE SpGEMM. Also computes `J^T` via CSR-to-CSC conversion. Filters the result to remove near-zero entries |
| `identity_csr_and_scale_kernel` | CUDA kernel: generates a scaled identity matrix `alpha * I` in CSR format |
| `create_identity_csr_and_scale(N, alpha, ...)` | Host wrapper for the identity kernel |
| `add_csr_cusparse(delta, m, n, ...)` | Computes `C = delta*A + delta*B` for two CSR matrices using `cusparseDcsrgeam2` |
| `scale_and_multiply_on_gpu(rows, cols, nnz, ...)` | Computes `y = alpha * A * x` for a CSC matrix `A` and dense vector `x` using cuSPARSE SpMV |

### 14. Direct Linear Solver (lines 2550-2681)

| Function | Description |
|----------|-------------|
| `CHECK_CUDSS(func, msg)` | Error-checking macro for the cuDSS library |
| `check_indices_sanity(nnz, num_rows, d_indices)` | Debug: validates that all index values are within `[0, num_rows)` |
| `solve_system_gpu(n, nnz, d_row_offsets, d_col_indices, d_values, d_b, d_x_out)` | Solves `Ax = b` using cuDSS (direct sparse solver). Performs analysis -> factorization -> solve with iterative refinement (3 steps). Matrix must be in CSR format |

### 15. COO Matrix Sorting (lines 2692-2855)

| Function | Description |
|----------|-------------|
| `sort_coo_matrix_cusparse(...)` | Sorts a COO matrix by row (primary) then column (secondary) using cuSPARSE `cusparseXcoosortByRow` with a gather-based value permutation |
| `sort_coo_cub(...)` | Alternative COO sorting using CUB `DeviceRadixSort::SortPairs` with a custom permutation kernel for columns and values |

### 16. Vector Operations (lines 2857-3038)

| Function | Description |
|----------|-------------|
| `CUBLAS_CHECK(call)` | Error-checking macro for cuBLAS |
| `square_norm(handle, residual, n)` | Returns `0.5 * ||x||^2` using `cublasDdot` (dot product with self) |
| `L2_norm_squared(handle, residual, n)` | Returns `||x||_2^2` using `cublasDnrm2` (numerically stable) |
| `vel_mag_kernel` | CUDA kernel: computes `sqrt(u^2 + v^2 + w^2)` element-wise |
| `compute_vel_mag(n, d_u, d_v, d_w)` | Host wrapper: allocates and returns velocity magnitude array |
| `copy_gpu_array_host(d_ptr, n)` | Copies a device array to a `std::vector<double>` (uses move semantics) |
| `addConstantKernel` / `addConstantToVector` | Adds a scalar constant to every element of a device vector |

### 17. Numerical Health Checks (lines 3042-3147)

| Function | Description |
|----------|-------------|
| `check_vector_finite_kernel` / `is_vector_finite(d_data, n)` | Checks for NaN, Inf, or extreme values (`> 1e12` or `< 1e-12`) in a device vector |
| `is_system_ill_defined(handle, d_residual, d_b, d_x, n, threshold)` | Diagnoses solver quality via relative residual `||r|| / ||b||` and expansion factor `||x|| / ||b||`. Warns if the system appears singular or ill-conditioned |

### 18. Levenberg-Marquardt Solver (lines 3149-3386)

| Function | Description |
|----------|-------------|
| `levenberg_marquardt_solver(Re, y, xN, yN, zN, u_inlet, dx, dy, dz, max_iterations, initial_lambda, lambda_factor, tolerance)` | Main nonlinear solver. Iteratively minimizes `||R(y)||^2` using the LM algorithm |

#### LM Algorithm Steps (per iteration):

1. **Jacobian & residual**: Call `Residuals_Sparse_Jacobian_finite_diff` to get `J` (COO) and `R(y)`
2. **Cost evaluation**: `cost = 0.5 * ||R||^2` via `square_norm`
3. **Convergence check**: Stop if `cost < tolerance`
4. **Sort Jacobian**: COO -> sorted COO (required for CSR conversion)
5. **Inner LM loop** (up to 10 attempts per outer iteration):
   a. Compute approximate Hessian: `H = J^T * J` (SpGEMM)
   b. Form damped system: `A = H + lambda * I` (CSR addition)
   c. Form gradient: `g = -J^T * R` (SpMV)
   d. Solve `A * delta = g` (cuDSS direct solver)
   e. Propose `y_new = y + delta`
   f. Evaluate `new_cost = 0.5 * ||R(y_new)||^2`
   g. **Accept** if `new_cost < cost`: update `y`, decrease `lambda /= factor`
   h. **Reject** otherwise: increase `lambda *= factor`, retry
6. **Debug output**: Exports Jacobian (COO) and LHS matrix (CSR) to `.npz` files via cnpy

### 19. Main Function (lines 3390-3517)

Sets up and runs the complete simulation pipeline:

1. **Problem definition**: Grid size (`5x5x5` default), physical parameters (`mu=0.001`, `rho=1.0`, `u0=1.0`), Reynolds number calculation
2. **Grid generation**: Uniform Cartesian coordinates via `coordinates()`
3. **Inlet profile**: Parabolic velocity `u_inlet = 16 * u0 * (y/M)(1-y/M)(z/N)(1-z/N)`
4. **Initial guess**: Reads from `solution.txt` (pre-computed starting point for debugging)
5. **Solver execution**: `levenberg_marquardt_solver(Re, y, ..., max_iter=25, lambda_0=0.001, factor=10, tol=1e-16)`
6. **Post-processing**: Reconstructs full 3D fields via `boundary_conditions_final`, computes velocity magnitude, exports everything to `simulation_results.npz` for Python visualization

---

## Data Flow Diagram

```
  y (state vector, host)
      |
      v
  [Residuals_Sparse_Jacobian_finite_diff]
      |
      +---> d_fold (base residual R(y), device)
      +---> J in COO format (d_rows, d_cols, d_vals, nnz)
                |
                v
          [sort_coo_matrix_cusparse]
                |
                v
          [compute_AtA_debug]  -->  J^T*J (CSR) + J^T (CSC)
                |                       |
                v                       v
    [add_csr_cusparse]         [scale_and_multiply_on_gpu]
    A = J^T*J + lambda*I         g = -J^T * R
                |                       |
                v                       v
              [solve_system_gpu]
              delta = A \ g
                |
                v
          y_new = y + delta  -->  [uv_velocity_single]  -->  new_cost
                |
          accept / reject (lambda adjustment)
```

---

## Key Constants and Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `xN, yN, zN` | `5, 5, 5` | Interior grid points per axis |
| `L, M, N` | `1.0, 0.2, 0.2` | Domain dimensions (length, height, width) |
| `mu` | `0.001` | Dynamic viscosity |
| `rho` | `1.0` | Fluid density |
| `u0` | `1.0` | Reference inlet velocity |
| `Re` | `100` | Reynolds number: `rho * (M/2) * u0 / mu` |
| `EPS` | `1e-6` | Finite-difference perturbation scale (sqrt of machine epsilon) |
| `initial_lambda` | `0.001` | Starting LM damping parameter |
| `lambda_factor` | `10` | Multiplicative factor for lambda adjustment |
| `tolerance` | `1e-16` | Convergence tolerance on `0.5 * ||R||^2` |
| `grain` | `1` | Number of simultaneous perturbations per kernel launch |
| `max_nnz` | `nCell * 1000` | Pre-allocated Jacobian buffer size |

---

## Dependencies

- **CUDA Toolkit** (runtime, cuSPARSE, cuBLAS, cuDSS, CUB)
- **OpenMP** (for potential multi-stream parallelism)
- **cnpy** (C++ library for reading/writing NumPy `.npz` files)
- **C++17** (structured bindings, `std::tuple`, `std::pair`)

---

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `jacobian_coo.npz` | NumPy archive | Sparse Jacobian in COO format: `values`, `row_indices`, `col_indices`, `shape` |
| `lhs.npz` | NumPy archive | LHS matrix `(J^T*J + lambda*I)` in CSR format: `values`, `col_indices`, `row_pointers`, `shape` |
| `simulation_results.npz` | NumPy archive | Full 3D fields: `u, v, w, p, velmag`, coordinates `xcoor, ycoor, zcoor`, grid sizes, domain dimensions |

---

## Notes

- This is a **prototype** file: all functionality resides in a single `.cu` file for rapid iteration. The production version is split across `src/` and `include/` directories.
- Several commented-out code blocks represent earlier versions of boundary condition kernels and Jacobian routines, preserved for reference.
- The `grain` parameter is currently set to `1`, meaning perturbations are processed one at a time. Increasing `grain` enables batched Jacobian columns but requires proportionally more GPU memory.
