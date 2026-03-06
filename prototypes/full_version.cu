/**
 * @file full_version.cu
 * @author Panagiotis Athanasiadis
 * @brief Does all the calculations
 * @version 0.1
 * @date 2026-02-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cudss.h> 
#include <cublas_v2.h>
#include <cnpy.h>



// ============================================================================ 
// ERROR CHECKING AND UTILITIES
// ============================================================================ 

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error " << err << " at "           \
                  << __FILE__ << ":" << __LINE__ << " -> "    \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while(0)

// Memory alignment utility
inline size_t align_size(size_t size, size_t alignment = 256) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// ============================================================================ 
// CONSISTENT INDEX FUNCTIONS
// ============================================================================ 

// Standard 3D indexing: (i, j, k) -> linear index
__device__ __host__ inline int idx_3d(int i, int j, int k, int sizeY, int sizeZ) {
    return (i * sizeY + j) * sizeZ + k;
}

// Batched 3D indexing with grain dimension
__device__ __host__ inline int idx_3d_batch(int i, int j, int k, int l,
                                             int sizeY, int sizeZ, int grain) {
    return ((i * sizeY + j) * sizeZ + k) * grain + l;
}


void print_gpu_array(const double* d_array, int n) {
    // Allocate host memory to hold the copy
    std::vector<double> h_array(n);

    // Copy from Device to Host
    cudaError_t status = cudaMemcpy(h_array.data(), d_array, n * sizeof(double), cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        printf("Error copying memory: %s\n", cudaGetErrorString(status));
        return;
    }

    // Print
    printf("GPU Array Content: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.8f ", h_array[i]);
    }
    printf("]\n");
}

void print_csr_matrix(int rows, int nnz, int* d_row_ptr, int* d_cols, double* d_vals) {
    // Allocate Host Memory
    int* h_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    int* h_cols    = (int*)malloc(nnz * sizeof(int));
    double* h_vals  = (double*)malloc(nnz * sizeof(double));

    // Copy from Device to Host
    CUDA_CHECK(cudaMemcpy(h_row_ptr, d_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols, d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals, d_vals, nnz * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\n--- Matrix Print (First 10 Rows) ---\n");
    printf("Format: Row [Start, End) -> (Col, Val)\n");

    // Loop through rows
    // (We limit to 10 to avoid flooding screen if matrix is huge)
    int print_limit = rows;

    for (int i = 0; i < print_limit; i++) {
        int start = h_row_ptr[i];
        int end   = h_row_ptr[i+1];
        
        printf("Row %d [%d, %d): ", i, start, end);
        
        // Loop through elements in this row
        if (start == end) {
            printf("(Empty)");
        } else {
            for (int j = start; j < end; j++) {
                printf("(%d, %.6f) ", h_cols[j], h_vals[j]);
            }
        }
        printf("\n");
    }

    printf("------------------------------------\n");

    // Cleanup Host Memory
    free(h_row_ptr); free(h_cols); free(h_vals);
}

/**
 * @brief Downloads a COO matrix from the GPU and prints it to stdout.
 *
 * This function is useful for debugging sparse matrix operations. It copies the 
 * data from Device to Host and prints the triplets (Row, Col, Value).
 *
 * @param[in] rows      Number of rows in the matrix.
 * @param[in] cols      Number of columns in the matrix.
 * @param[in] nnz       Number of non-zero elements.
 * @param[in] d_rows    Device pointer to row indices.
 * @param[in] d_cols    Device pointer to column indices.
 * @param[in] d_vals    Device pointer to values.
 * @param[in] max_print (Optional) Maximum number of elements to print. 
 * Defaults to 20. Pass -1 to print everything.
 */
void print_coo_matrix_gpu(int rows, int cols, int nnz, 
                          const int* d_rows, 
                          const int* d_cols, 
                          const double* d_vals,
                          int max_print = 20) 
{
    // 1. Allocate Host Memory
    std::vector<int> h_rows(nnz);
    std::vector<int> h_cols(nnz);
    std::vector<double> h_vals(nnz);

    // 2. Copy Data from Device to Host
    cudaMemcpy(h_rows.data(), d_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cols.data(), d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), d_vals, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // 3. Print Header
    std::cout << "--- COO Matrix (GPU) ---" << std::endl;
    std::cout << "Dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Non-zeros:  " << nnz << std::endl;
    std::cout << "------------------------" << std::endl;

    // 4. Determine print limit
    int limit = (max_print == -1) ? nnz : std::min(nnz, max_print);

    // 5. Print Elements
    for (int i = 0; i < limit; ++i) {
        std::cout << "Idx " << i << ": (" 
                  << h_rows[i] << ", " 
                  << h_cols[i] << ") = " 
                  << h_vals[i] << std::endl;
    }

    if (limit < nnz) {
        std::cout << "... (" << (nnz - limit) << " more elements omitted) ..." << std::endl;
    }
    std::cout << "------------------------" << std::endl;
}

// ============================================================================
// KERNEL: BUILD SOLUTION BATCH
// ============================================================================

__global__ void build_ysol_batch_kernel(
    const double* __restrict__ y,
    const double* __restrict__ h,
    double* __restrict__ ysol_batch,
    const int nCell,
    const int grain,
    const int* __restrict__ t_batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCell) return;

    // Copy base values for all grains
    for (int g = 0; g < grain; g++) {
        ysol_batch[g * nCell + i] = y[i];
    }

    // Update perturbed positions
    for (int g = 0; g < grain; g++) {
        int t = t_batch[g];
        if (i == t) {
            ysol_batch[g * nCell + t] += h[t];
        }
    }
}

// ============================================================================
// KERNEL: BOUNDARY CONDITIONS INITIALIZATION (BATCHED)
// ============================================================================

__global__ void 
boundary_conditions_initialization(
    const double* __restrict__ ysol_local_batch,
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ w,
    double* __restrict__ p,
    const int xN, const int yN, const int zN,
    const int grain)
{
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    const int nc = xN * yN * zN;
    
    // Linear indexing for better coalescing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = xN * yN * zN;
    
    if (tid >= total_threads) return;
    
    // Decompose linear index
    int k = tid % zN;
    int j = (tid / zN) % yN;
    int i = tid / (yN * zN);
    
    // Compute 3D indices
    int i3d = i + 1;
    int j3d = j + 1;
    int k3d = k + 1;
    
    for (int l = 0; l < grain; ++l) {
        int offset = l * 4 * nc;
        int out_idx = ((i3d * sizeY + j3d) * sizeZ + k3d) * grain + l;
        
        u[out_idx] = ysol_local_batch[offset + tid];
        v[out_idx] = ysol_local_batch[offset + nc + tid];
        w[out_idx] = ysol_local_batch[offset + 2*nc + tid];
        p[out_idx] = ysol_local_batch[offset + 3*nc + tid];
    }
}


// ============================================================================
// KERNEL: BOUNDARY CONDITIONS APPLY (BATCHED)
// ============================================================================

__global__ void boundary_conditions_apply(
    const double* __restrict__ u_inlet,
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ w,
    double* __restrict__ p,
    const int xN, const int yN, const int zN,
    const int grain)
{
    const int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= sizeX || j >= sizeY || k >= sizeZ) return;

    for (int l = 0; l < grain; ++l) {
        int idx_curr = idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain);

        // Y-direction boundaries
        if (j == 0) {
            p[idx_3d_batch(i, 0, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, 1, k, l, sizeY, sizeZ, grain)];
        } else if (j == yN + 1) {
            p[idx_3d_batch(i, yN + 1, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, yN, k, l, sizeY, sizeZ, grain)];
        }
        if (j == 0 || j == yN + 1) {
          u[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
          v[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
          w[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
        }

        // Z-direction boundaries
        if (k == 0) {
            p[idx_3d_batch(i, j, 0, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, 1, l, sizeY, sizeZ, grain)];
        } else if (k == zN + 1) {
            p[idx_3d_batch(i, j, zN + 1, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, zN, l, sizeY, sizeZ, grain)];
        }

        if (k == 0 || k == zN + 1  && i > 0 && i < xN + 1) {
          u[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
          v[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
          w[idx_3d_batch(i, j, k, l, sizeY, sizeZ, grain)] = 0.0; 
        }

        // X-direction boundaries
        if (i == 0) {
             u[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 
                 u_inlet[idx_3d(j, k, 0, sizeZ, 1)];
        //    u_inlet[idx_3d_batch(j, k, 0, l, sizeZ, 1, grain)];
            p[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(1, j, k, l, sizeY, sizeZ, grain)];
            v[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 0.0;         
            w[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 0.0;         

        } else if (i == xN + 1) {
            u[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                u[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
            v[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                v[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
            w[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                w[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
            p[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 0.0;
        }
    }
}



// ============================================================================
// KERNEL: BOUNDARY CONDITIONS INITIALIZATION (SINGLE)
// ============================================================================

__global__ void boundary_conditions_initialization_single(
    const double* __restrict__ ysol,
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ w,
    double* __restrict__ p,
    const int xN, const int yN, const int zN)
{
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    int idx = idx_3d(i, j, k, sizeY, sizeZ);

    u[idx] = ysol[pos];
    v[idx] = ysol[xN * yN * zN + pos];
    w[idx] = ysol[2 * xN * yN * zN + pos];
    p[idx] = ysol[3 * xN * yN * zN + pos];
}

// ============================================================================
// KERNEL: BOUNDARY CONDITIONS APPLY (SINGLE)
// ============================================================================

__global__ void boundary_conditions_apply_single(
    const double* __restrict__ u_inlet,
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ w,
    double* __restrict__ p,
    const int xN, const int yN, const int zN)
{
    const int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= sizeX || j >= sizeY || k >= sizeZ) return;

    // Y-direction boundaries
    if (j == 0) {
        p[idx_3d(i, 0, k, sizeY, sizeZ)] = p[idx_3d(i, 1, k, sizeY, sizeZ)];
    } else if (j == yN + 1) {
        p[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = p[idx_3d(i, yN, k, sizeY, sizeZ)];
    }

    if (j == 0 || j == yN + 1 && i > 0 && i < xN + 1) {
        u[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
        v[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
        w[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
    }

    // Z-direction boundaries
    if (k == 0) {
        p[idx_3d(i, j, 0, sizeY, sizeZ)] = p[idx_3d(i, j, 1, sizeY, sizeZ)];
    } else if (k == zN + 1) {
        p[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = p[idx_3d(i, j, zN, sizeY, sizeZ)];
    }

    if (k == 0 || k == zN + 1  && i > 0 && i < xN + 1) {
        u[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
        v[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
        w[idx_3d(i, j, k, sizeY, sizeZ)] = 0.0;    
    }

    // X-direction boundaries
    if (i == 0) {
        u[idx_3d(0, j, k, sizeY, sizeZ)] = u_inlet[idx_3d(j, k, 0, sizeZ, 1)];
        p[idx_3d(0, j, k, sizeY, sizeZ)] = p[idx_3d(1, j, k, sizeY, sizeZ)];
        v[idx_3d(0, j, k, sizeY, sizeZ)] = 0.0;         
        w[idx_3d(0, j, k, sizeY, sizeZ)] = 0.0;  
    } else if (i == xN + 1) {
        u[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = u[idx_3d(xN, j, k, sizeY, sizeZ)];
        v[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = v[idx_3d(xN, j, k, sizeY, sizeZ)];
        w[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = w[idx_3d(xN, j, k, sizeY, sizeZ)];
        p[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = 0.0;
    }

}

// ============================================================================
// KERNEL: UV VELOCITY (SINGLE)
// ============================================================================

__global__ void kernel_uv_velocity_single(
    double* __restrict__ out, double Re,
    double* __restrict__ u, double* __restrict__ v,
    double* __restrict__ p, double* __restrict__ w,
    const int xN, const int yN, const int zN,
    const double dx, const double dy, const double dz)
{
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int nCell = xN * yN * zN;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);

    // Helper lambda
    auto idx = [=] __device__ (int ii, int jj, int kk) -> int {
        return idx_3d(ii, jj, kk, sizeY, sizeZ);
    };

    // U-momentum
    {
        double conv_x = 0.5 * dy * dz * (u[idx(i+1,j,k)]*u[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*u[idx(i-1,j,k)]);
        double conv_y = 0.5 * dx * dz * (u[idx(i,j+1,k)]*v[idx(i,j+1,k)] - 
                                         u[idx(i,j-1,k)]*v[idx(i,j-1,k)]);
        double conv_z = 0.5 * dx * dy * (u[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         u[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        // double pres = (dy * dz) * (p[idx(i+1,j,k)] - p[idx(i,j,k)]);
        double pres = (dy * dz / 2.0) * (p[idx(i+1,j,k)] - p[idx(i-1,j,k)]);

        double diff = (1.0/Re) * (
            (dy*dz/dx) * (u[idx(i+1,j,k)] - 2.0*u[idx(i,j,k)] + u[idx(i-1,j,k)]) +
            (dx*dz/dy) * (u[idx(i,j+1,k)] - 2.0*u[idx(i,j,k)] + u[idx(i,j-1,k)]) +
            (dx*dy/dz) * (u[idx(i,j,k+1)] - 2.0*u[idx(i,j,k)] + u[idx(i,j,k-1)])
        );

        out[pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // V-momentum
    {
        double conv_x = 0.5 * dy * dz * (u[idx(i+1,j,k)]*v[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*v[idx(i-1,j,k)]);
        double conv_y = 0.5 * dx * dz * (v[idx(i,j+1,k)]*v[idx(i,j+1,k)] - 
                                         v[idx(i,j-1,k)]*v[idx(i,j-1,k)]);
        double conv_z = 0.5 * dx * dy * (v[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         v[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        // double pres = (dx * dz) * (p[idx(i,j+1,k)] - p[idx(i,j,k)]);
        double pres = (dx * dz / 2.0) * (p[idx(i,j+1,k)] - p[idx(i,j-1,k)]);

        double diff = (1.0/Re) * (
            (dy*dz/dx) * (v[idx(i+1,j,k)] - 2.0*v[idx(i,j,k)] + v[idx(i-1,j,k)]) +
            (dx*dz/dy) * (v[idx(i,j+1,k)] - 2.0*v[idx(i,j,k)] + v[idx(i,j-1,k)]) +
            (dx*dy/dz) * (v[idx(i,j,k+1)] - 2.0*v[idx(i,j,k)] + v[idx(i,j,k-1)])
        );

        out[nCell + pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // W-momentum
    {
        double conv_x = 0.5 * dy * dz * (u[idx(i+1,j,k)]*w[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*w[idx(i-1,j,k)]);
        double conv_y = 0.5 * dx * dz * (v[idx(i,j+1,k)]*w[idx(i,j+1,k)] - 
                                         v[idx(i,j-1,k)]*w[idx(i,j-1,k)]);
        double conv_z = 0.5 * dx * dy * (w[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         w[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        // double pres = (dx * dy) * (p[idx(i,j,k+1)] - p[idx(i,j,k)]);

        double pres = (dx * dy / 2.0) * (p[idx(i,j,k+1)] - p[idx(i,j,k-1)]);

        double diff = (1.0/Re) * (
            (dy*dz/dx) * (w[idx(i+1,j,k)] - 2.0*w[idx(i,j,k)] + w[idx(i-1,j,k)]) +
            (dx*dz/dy) * (w[idx(i,j+1,k)] - 2.0*w[idx(i,j,k)] + w[idx(i,j-1,k)]) +
            (dx*dy/dz) * (w[idx(i,j,k+1)] - 2.0*w[idx(i,j,k)] + w[idx(i,j,k-1)])
        );

        out[2*nCell + pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // Continuity
    {
        double cont = (dy*dz/2.0) * (u[idx(i+1,j,k)] - u[idx(i-1,j,k)]) +
                    (dx*dz/2.0) * (v[idx(i,j+1,k)] - v[idx(i,j-1,k)]) +
                    (dx*dy/2.0) * (w[idx(i,j,k+1)] - w[idx(i,j,k-1)]);

        out[3*nCell + pos] = cont;
    }
}

__device__ bool check_equal(double a, double b, double epsilon) {
    double diff = fabs(a - b);
    
    // 2. Handle the "Zero" edge case
    // If a and b are effectively zero, the relative error check below 
    // might fail (0/0)
    if (a == b || diff < epsilon) {
        return true;
    }

    return diff < (epsilon * fmax(fabs(a), fabs(b)));
}

// ============================================================================
// HOST FUNCTION: RESIDUALS AND SPARSE JACOBIAN
// ============================================================================
__global__ void build_jacobian_entries(
    const double* __restrict__ out,
    const double* __restrict__ fold,
    const double* __restrict__ h,
    int* __restrict__ row_idx,
    int* __restrict__ col_idx,
    double* __restrict__ values,
    int* __restrict__ counter,
    const int nCell, const int grain, const int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nCell || g >= grain) return;
    
    int idx = g * nCell + i;
    int batch_step = g + start;
    double df = (out[idx] - fold[i]) / h[batch_step];
    
    if (fabs(df) > 1e-14) {
    // if (!check_equal(df, 0, 1e-10)) {
        int pos = atomicAdd(counter, 1);    
        // printf("%d %d %f %d\n ",i,batch_step,df);
        row_idx[pos] = i;              // Row index
        col_idx[pos] = batch_step;      // Column index
        values[pos] = df;               // Value   
    }
}

// ============================================================================
// Modified Residuals_Sparse_Jacobian with GPU sparse matrix building
// ============================================================================

__global__ void 
kernel_u_momentum(
    int grain, double* __restrict__ out,
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ p, const double* __restrict__ w,
    const int xN, const int yN, const int zN,
    const double dx, const double dy, const double dz,
    const double Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const double dy_dz = dy * dz;
    const double dx_dz = dx * dz;
    const double dx_dy = dx * dy;
    const double inv_Re = 1.0 / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        double u_c  = u[base_idx];
        double u_ip = u[base_idx + stride_i];
        double u_im = u[base_idx - stride_i];
        double u_jp = u[base_idx + stride_j];
        double u_jm = u[base_idx - stride_j];
        double u_kp = u[base_idx + stride_k];
        double u_km = u[base_idx - stride_k];

        double result = 0.5 * dy_dz * (u_ip * u_ip - u_im * u_im);
        result += 0.5 * dx_dz * (u_jp * v[base_idx + stride_j] - u_jm * v[base_idx - stride_j]);
        result += 0.5 * dx_dy * (u_kp * w[base_idx + stride_k] - u_km * w[base_idx - stride_k]);
        // result += dy_dz * (p[base_idx + stride_i] - p[base_idx]);
        result += (dy_dz / 2.0) * (p[base_idx + stride_i] - p[base_idx - stride_i]);
        result -= inv_Re * ((dy_dz/dx) * (u_ip - 2.0*u_c + u_im) +
                           (dx_dz/dy) * (u_jp - 2.0*u_c + u_jm) +
                           (dx_dy/dz) * (u_kp - 2.0*u_c + u_km));

        out[l * NC + pos] = result;
    }
}

// V-Momentum Kernel
__global__ void 
kernel_v_momentum(
    int grain, double* __restrict__ out,
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ p, const double* __restrict__ w,
    const int xN, const int yN, const int zN,
    const double dx, const double dy, const double dz,
    const double Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const double dy_dz = dy * dz;
    const double dx_dz = dx * dz;
    const double dx_dy = dx * dy;
    const double inv_Re = 1.0 / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        double v_c  = v[base_idx];
        double v_ip = v[base_idx + stride_i];
        double v_im = v[base_idx - stride_i];
        double v_jp = v[base_idx + stride_j];
        double v_jm = v[base_idx - stride_j];
        double v_kp = v[base_idx + stride_k];
        double v_km = v[base_idx - stride_k];

        double result = 0.5 * dy_dz * (u[base_idx + stride_i] * v_ip - u[base_idx - stride_i] * v_im);
        result += 0.5 * dx_dz * (v_jp * v_jp - v_jm * v_jm);
        result += 0.5 * dx_dy * (v_kp * w[base_idx + stride_k] - v_km * w[base_idx - stride_k]);
        // result += dx_dz * (p[base_idx + stride_j] - p[base_idx]);
        result += (dx_dz / 2.0) * (p[base_idx + stride_j] - p[base_idx - stride_j]);
        result -= inv_Re * ((dy_dz/dx) * (v_ip - 2.0*v_c + v_im) +
                           (dx_dz/dy) * (v_jp - 2.0*v_c + v_jm) +
                           (dx_dy/dz) * (v_kp - 2.0*v_c + v_km));

        out[l * NC + nc + pos] = result;
    }
}

// W-Momentum Kernel
__global__ void
kernel_w_momentum(
    int grain, double* __restrict__ out,
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ p, const double* __restrict__ w,
    const int xN, const int yN, const int zN,
    const double dx, const double dy, const double dz,
    const double Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const double dy_dz = dy * dz;
    const double dx_dz = dx * dz;
    const double dx_dy = dx * dy;
    const double inv_Re = 1.0 / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        double w_c  = w[base_idx];
        double w_ip = w[base_idx + stride_i];
        double w_im = w[base_idx - stride_i];
        double w_jp = w[base_idx + stride_j];
        double w_jm = w[base_idx - stride_j];
        double w_kp = w[base_idx + stride_k];
        double w_km = w[base_idx - stride_k];

        double result = 0.5 * dy_dz * (u[base_idx + stride_i] * w_ip - u[base_idx - stride_i] * w_im);
        result += 0.5 * dx_dz * (v[base_idx + stride_j] * w_jp - v[base_idx - stride_j] * w_jm);
        result += 0.5 * dx_dy * (w_kp * w_kp - w_km * w_km);
        // result += dx_dy * (p[base_idx + stride_k] - p[base_idx]);
        result += (dx_dy / 2.0) * (p[base_idx + stride_k] - p[base_idx - stride_k]);
        result -= inv_Re * ((dy_dz/dx) * (w_ip - 2.0*w_c + w_im) +
                           (dx_dz/dy) * (w_jp - 2.0*w_c + w_jm) +
                           (dx_dy/dz) * (w_kp - 2.0*w_c + w_km));

        out[l * NC + 2*nc + pos] = result;
    }
}

// Continuity Kernel
__global__ void 
kernel_continuity(
    int grain, double* __restrict__ out,
    const double* __restrict__ u, const double* __restrict__ v,
    const double* __restrict__ w,
    const int xN, const int yN, const int zN,
    const double dx, const double dy, const double dz,
    const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        double result = 0.5 * dy * dz * (u[base_idx + stride_i] - u[base_idx - stride_i]);
        result += 0.5 * dx * dz * (v[base_idx + stride_j] - v[base_idx - stride_j]);
        result += 0.5 * dx * dy * (w[base_idx + stride_k] - w[base_idx - stride_k]);

        out[l * NC + 3*nc + pos] = result;
    }
}

//We might use nvida file i/o cuFile
std::tuple<double*,double*,double*,double*> boundary_conditions_final(double *ysol,
                       const int xN, const int yN, const int zN,
                       const double *u_inlet)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    
    double *d_u_inlet,*ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ysol, align_size(nCell * sizeof(double))));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(double))));
    
    //Copy data to Device
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize boundary conditions
    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(ysol, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Apply boundary conditions
    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_u_inlet));
    CUDA_CHECK(cudaFree(ysol));

    // print_gpu_array(ud, totalSize);

    return {ud,vd,wd,pd};
}







// ============================================================================
// HOST FUNCTION: UV VELOCITY SINGLE
// ============================================================================

void uv_velocity_single(double *out, const double Re, double *y,
                       const int xN, const int yN, const int zN,
                       const double *u_inlet,
                       const double dx, const double dy, const double dz)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    // Allocate device memory with alignment
    double *dev_out, *ysol, *d_u_inlet;
    double *ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&dev_out, align_size(nCell * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ysol, align_size(nCell * sizeof(double))));

    CUDA_CHECK(cudaMemset(ud, 0, totalSize * sizeof(double)));
CUDA_CHECK(cudaMemset(vd, 0, totalSize * sizeof(double)));
CUDA_CHECK(cudaMemset(wd, 0, totalSize * sizeof(double)));
CUDA_CHECK(cudaMemset(pd, 0, totalSize * sizeof(double)));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(double))));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ysol, y, nCell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize boundary conditions
    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(ysol, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Apply boundary conditions
    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Compute velocity
    dim3 threads(8, 8, 4);
    dim3 blocks((xN + threads.x - 1) / threads.x,
                (yN + threads.y - 1) / threads.y,
                (zN + threads.z - 1) / threads.z);
    kernel_uv_velocity_single<<<blocks, threads>>>(dev_out, Re, ud, vd, pd, wd, xN, yN, zN, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(out, dev_out, nCell * sizeof(double), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(dev_out));
    CUDA_CHECK(cudaFree(ud));
    CUDA_CHECK(cudaFree(vd));
    CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(pd));
    CUDA_CHECK(cudaFree(ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
}

//direct return a pointer to gpu memory, it allocates space , caller is resposnible for freeing 
double* uv_velocity_single(const double Re, double *y,
                       const int xN, const int yN, const int zN,
                       const double *u_inlet,
                       const double dx, const double dy, const double dz)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    // Allocate device memory with alignment
    double *dev_out, *ysol, *d_u_inlet;
    double *ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&dev_out, align_size(nCell * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ysol, align_size(nCell * sizeof(double))));

    CUDA_CHECK(cudaMemset(ud, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(vd, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(wd, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(pd, 0, totalSize * sizeof(double)));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(double))));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ysol, y, nCell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize boundary conditions
    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(ysol, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Apply boundary conditions
    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Compute velocity
    dim3 threads(8, 8, 4);
    dim3 blocks((xN + threads.x - 1) / threads.x,
                (yN + threads.y - 1) / threads.y,
                (zN + threads.z - 1) / threads.z);
    kernel_uv_velocity_single<<<blocks, threads>>>(dev_out, Re, ud, vd, pd, wd, xN, yN, zN, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());


    // Free memory
    CUDA_CHECK(cudaFree(ud));
    CUDA_CHECK(cudaFree(vd));
    CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(pd));
    CUDA_CHECK(cudaFree(ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
    return dev_out;
}

//direct read residual from gpu and returns a pointer to gpu memory, it allocates space , caller is resposnible for freeing 
double* uv_velocity_single_direct(const double Re, double *y,
                       const int xN, const int yN, const int zN,
                       const double *u_inlet,
                       const double dx, const double dy, const double dz)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    // Allocate device memory with alignment
    double *dev_out,*d_u_inlet;
    double *ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&dev_out, align_size(nCell * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(double))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(double))));

    CUDA_CHECK(cudaMemset(ud, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(vd, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(wd, 0, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(pd, 0, totalSize * sizeof(double)));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(double))));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize boundary conditions
    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(y, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Apply boundary conditions
    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    // Compute velocity
    dim3 threads(8, 8, 4);
    dim3 blocks((xN + threads.x - 1) / threads.x,
                (yN + threads.y - 1) / threads.y,
                (zN + threads.z - 1) / threads.z);
    kernel_uv_velocity_single<<<blocks, threads>>>(dev_out, Re, ud, vd, pd, wd, xN, yN, zN, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());


    // Free memory
    CUDA_CHECK(cudaFree(ud));
    CUDA_CHECK(cudaFree(vd));
    CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(pd));
    CUDA_CHECK(cudaFree(d_u_inlet));
    return dev_out;
}


/**
 * @brief Computes the sparse Jacobian matrix of the residual function using finite differences on the GPU.
 *
 * This function calculates the Jacobian matrix \f$ J = \frac{\partial R}{\partial y} \f$ by perturbing
 * the state vector @p y and evaluating the residuals using a finite difference approximation.
 * It utilizes OpenMP to manage concurrent CUDA streams, allowing for batched processing of 
 * perturbations to maximize GPU utilization.
 *
 * @param Re       The Reynolds number for the flow simulation.
 * @param y        Pointer to the current state vector (linearized 1D array of size 4*xN*yN*zN).
 * @param xN       Number of grid points in the X direction.
 * @param yN       Number of grid points in the Y direction.
 * @param zN       Number of grid points in the Z direction.
 * @param u_inlet  Pointer to the inlet velocity profile data (device or host pointer depending on implementation).
 * @param dx       Grid spacing in the X direction.
 * @param dy       Grid spacing in the Y direction.
 * @param dz       Grid spacing in the Z direction.
 *
 * @return A std::pair containing:
 * - **first**: `double*` - Device pointer to the base residual vector \f$ F(y) \f$.
 * - **second**: `std::tuple<int*, int*, double*, int>` - The Jacobian matrix in Coordinate (COO) format:
 * - `int*`: Device pointer to row indices.
 * - `int*`: Device pointer to column indices.
 * - `double*`: Device pointer to non-zero values.
 * - `int`: The total count of non-zero elements (nnz).
 *
 * @note The returned pointers point to **device memory** (GPU). The caller is responsible 
 * for freeing these resources using `cudaFree`.
 * @warning This function performs significant device memory allocation. Ensure sufficient GPU memory is available.
 */
std::pair<double*, std::tuple<int*, int*, double*, int>>
Residuals_Sparse_Jacobian_finite_diff(
    const double Re, double *y,
    const int xN, const int yN, const int zN,
    const double *u_inlet,
    const double dx, const double dy, const double dz)
{
    const int grain = 1;
    const int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    const int totalSize = sizeX * sizeY * sizeZ;
    const int nCell = 4 * xN * yN * zN;
    const double EPS = 1e-6;  // sqrt of machine epsilon for double
    std::vector<double> h(nCell);

    for (int j = 0; j < nCell; ++j) {
        double temp = y[j];
        // Use max to handle near-zero values
        h[j] = EPS * std::max(1.0, std::abs(temp));
        
        // Ensure we actually perturb the value (avoid roundoff)
        double y_perturbed = temp + h[j];
        h[j] = y_perturbed - temp;  // Actual perturbation achieved
        
        // Store back
        y[j] = temp;
    }

    double* d_fold=nullptr;
    d_fold=uv_velocity_single(Re, y, xN, yN, zN, u_inlet, dx, dy, dz);

    double *hd, *d_ysol, *d_u_inlet;
    CUDA_CHECK(cudaMalloc(&hd, align_size(nCell * sizeof(double))));
    CUDA_CHECK(cudaMemcpy(hd, h.data(), nCell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ysol, align_size(nCell * sizeof(double))));
    CUDA_CHECK(cudaMemcpy(d_ysol, y, nCell * sizeof(double), cudaMemcpyHostToDevice));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(double))));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(double), cudaMemcpyHostToDevice));

    size_t max_nnz = (size_t)nCell * 1000; //Guess based on the sparsity
    int *d_all_rows, *d_all_cols;
    double *d_all_vals;
    int *d_global_counter;
    
    CUDA_CHECK(cudaMalloc(&d_all_rows, max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_cols, max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_vals, max_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));

    int chunk_size = std::min(grain, nCell);
    int num_chunks = (nCell + chunk_size - 1) / chunk_size;
    int n_threads = std::min(1 , omp_get_max_threads());
    int chunks_per_thread = (num_chunks + n_threads - 1) / n_threads;

    std::cout << "Building Jacobian (split kernels)..." << std::endl;

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        double *dev_out, *ud, *vd, *wd, *pd, *d_ysol_batch;
        int *d_t_batch;

        size_t totalSize_grain = (size_t)totalSize * grain;
        CUDA_CHECK(cudaMalloc(&dev_out, align_size((size_t)grain * nCell * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize_grain * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize_grain * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize_grain * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize_grain * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&d_ysol_batch, align_size((size_t)grain * nCell * sizeof(double))));
        CUDA_CHECK(cudaMalloc(&d_t_batch, align_size(grain * sizeof(int))));

        CUDA_CHECK(cudaMemsetAsync(ud, 0, totalSize_grain * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(vd, 0, totalSize_grain * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(wd, 0, totalSize_grain * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(pd, 0, totalSize_grain * sizeof(double), stream));

        std::vector<int> t_batch_host(grain);
        int chunk_start = tid * chunks_per_thread;
        int chunk_end = std::min(num_chunks, (tid + 1) * chunks_per_thread);

        for (int chunk = chunk_start; chunk < chunk_end; ++chunk) {
            int start = chunk * chunk_size;
            int end = std::min(nCell, start + chunk_size);
            int current_grain = end - start;

            for (int t = start; t < end; ++t)
                t_batch_host[t - start] = t;

            CUDA_CHECK(cudaMemcpyAsync(d_t_batch, t_batch_host.data(),
                                      current_grain * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));

            int ysol_threads = 256;
            int ysol_blocks = (nCell + ysol_threads - 1) / ysol_threads;
            build_ysol_batch_kernel<<<ysol_blocks, ysol_threads, 0, stream>>>(
                d_ysol, hd, d_ysol_batch, nCell, current_grain, d_t_batch);
            // CUDA_CHECK(cudaGetLastError());

            int nc = xN * yN * zN;
            int threads = 256; //Prefer 1d for better coalesced memory access
            int blocks = (nc + threads - 1) / threads;
            boundary_conditions_initialization<<<blocks, threads, 0, stream>>>(
            d_ysol_batch, ud, vd, wd, pd, xN, yN, zN, current_grain);
            // CUDA_CHECK(cudaGetLastError());

            dim3 blk(8, 8, 4);
            dim3 grd((sizeX + blk.x - 1) / blk.x,
                    (sizeY + blk.y - 1) / blk.y,
                    (sizeZ + blk.z - 1) / blk.z);
            boundary_conditions_apply<<<grd, blk, 0, stream>>>(
                d_u_inlet, ud, vd, wd, pd, xN, yN, zN, current_grain);
            // CUDA_CHECK(cudaGetLastError());

            dim3 t(8, 8, 4);
            dim3 bks((xN + t.x - 1) / t.x,
                    (yN + t.y - 1) / t.y,
                    (zN + t.z - 1) / t.z);

            kernel_u_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);
            // CUDA_CHECK(cudaGetLastError());

            kernel_v_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);
            // CUDA_CHECK(cudaGetLastError());

            kernel_w_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);
            // CUDA_CHECK(cudaGetLastError());

            kernel_continuity<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, wd, xN, yN, zN,
                dx, dy, dz, sizeY, sizeZ);
            // CUDA_CHECK(cudaGetLastError());
            // ================================================================

            dim3 tt(32, 4);
            dim3 tg((nCell + tt.x - 1) / tt.x,
                   (current_grain + tt.y - 1) / tt.y);
            build_jacobian_entries<<<tg, tt, 0, stream>>>(
                dev_out, d_fold, hd,
                d_all_rows, d_all_cols, d_all_vals,
                d_global_counter,
                nCell, current_grain, start);
            CUDA_CHECK(cudaGetLastError());
            
            int total_attempted_nnz;
            cudaMemcpy(&total_attempted_nnz, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);
            if (total_attempted_nnz > max_nnz) {
                printf("Error: Jacobian buffer overflow! Needed %d, had %zu\n", total_attempted_nnz, max_nnz);
            }

            
            if (chunk % 1000 == 0) {
                std::cout << "Thread " << tid << ": " << chunk << "/" << chunk_end << std::endl;
            }
        }
    
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError()); //Check once after synchronize
        CUDA_CHECK(cudaFree(dev_out));
        CUDA_CHECK(cudaFree(ud));
        CUDA_CHECK(cudaFree(vd));
        CUDA_CHECK(cudaFree(wd));
        CUDA_CHECK(cudaFree(pd));
        CUDA_CHECK(cudaFree(d_ysol_batch));
        CUDA_CHECK(cudaFree(d_t_batch));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    int nnz;
    CUDA_CHECK(cudaMemcpy(&nnz, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Jacobian: " << nnz << " non-zeros" << std::endl;

    
    CUDA_CHECK(cudaFree(hd));
    CUDA_CHECK(cudaFree(d_ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
    CUDA_CHECK(cudaFree(d_global_counter));

    return {d_fold, std::make_tuple(d_all_rows, d_all_cols, d_all_vals, nnz)};
}


std::tuple<double, double, double>
coordinates(std::vector<double> &xcoor, std::vector<double> &ycoor,
            std::vector<double> &zcoor, const int xN, const int yN,
            const int zN, const double L, const double M, const double N) {
    int xSize = xN + 2;
    int ySize = yN + 2;
    int zSize = zN + 2;

    for (int iz = 0; iz < zSize; ++iz) {
        for (int iy = 0; iy < ySize; ++iy) {
            for (int ix = 0; ix < xSize; ++ix) {
                int idx = idx_3d(ix, iy, iz, ySize, zSize);
                xcoor[idx] = L * ix / (xSize - 1);
                ycoor[idx] = M * iy / (ySize - 1);
                zcoor[idx] = N * iz / (zSize - 1);
            }
        }
    }

    double dx = L / (xSize - 1);
    double dy = M / (ySize - 1);
    double dz = N / (zSize - 1);

    return std::make_tuple(dx, dy, dz);
}

// // Helper kernel to initialize indices: [0, 1, 2, ...]
// __global__ void init_indices(int* ptr, int nnz) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx < nnz) ptr[idx] = idx;
// }

// Helper kernel to permute data based on sorted indices
// out[i] = in[ sorted_indices[i] ]
// __global__ void permute_data(const int* in_cols, int* out_cols, 
//                              const double* in_vals, double* out_vals, 
//                              const int* sorted_indices, int nnz) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < nnz) {
//         int old_pos = sorted_indices[i];
//         out_cols[i] = in_cols[old_pos];
//         out_vals[i] = in_vals[old_pos];
//     }
// }



// -------------------------------------------------------------------------
// Kernel: Generate Selection Flags
// -------------------------------------------------------------------------
__global__ void generate_flags_kernel(int nnz, const double* values, char* flags, double threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        // 1 = Keep, 0 = Discard
        flags[idx] = (fabsf(values[idx]) > threshold) ? 1 : 0;
    }
}

// -------------------------------------------------------------------------
// Main Filter Function (CSR Format using CUB)
// -------------------------------------------------------------------------
void filter_csr_cub(double threshold,int rows, int old_nnz,
                    int* d_row_offsets, int* d_cols, double* d_vals,
                    int** d_new_row_offsets, int** d_new_cols, double** d_new_vals,
                    int* new_nnz_out) {
  
    // ---------------------------------------------------------------------
    // 1. Setup & Generate Flags
    // ---------------------------------------------------------------------
    char* d_flags;
    CUDA_CHECK(cudaMalloc((void**)&d_flags, old_nnz * sizeof(char)));

    int blockSize = 256;
    int gridSize = (old_nnz + blockSize - 1) / blockSize;
    
    generate_flags_kernel<<<gridSize, blockSize>>>(old_nnz, d_vals, d_flags, threshold);
    CUDA_CHECK(cudaGetLastError());

    // ---------------------------------------------------------------------
    // 2. Calculate New Row Counts (Segmented Reduce)
    // ---------------------------------------------------------------------
    // We sum the 'flags' within each row segment to see how many kept items per row.
    
    int* d_row_counts; 
    CUDA_CHECK(cudaMalloc((void**)&d_row_counts, rows * sizeof(int)));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // A. Query memory requirement
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // C. Run Reduction
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));

    // ---------------------------------------------------------------------
    // 3. Generate New Row Offsets (Prefix Scan)
    // ---------------------------------------------------------------------
    
    // Allocate Output Row Pointer Array (Size: Rows + 1)
    CUDA_CHECK(cudaMalloc((void**)d_new_row_offsets, (rows + 1) * sizeof(int)));
    
    // Cleanup previous temp storage to be safe (or reuse if you manage size manually)
    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    // A. Query memory requirement
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // C. Run Scan
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));

    // Calculate total NNZ (Read last count and last offset from GPU)
    int last_count = 0, last_offset = 0;
    
    // Note: d_row_counts has size 'rows'. Last index is rows-1.
    //       d_new_row_offsets has size 'rows+1'. Index 'rows' is currently empty/garbage.
    
    CUDA_CHECK(cudaMemcpy(&last_count, &d_row_counts[rows-1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, *d_new_row_offsets + (rows-1), sizeof(int), cudaMemcpyDeviceToHost));
    
    int final_nnz = last_offset + last_count;
    *new_nnz_out = final_nnz;
    
    // Write the final NNZ to the end of the row pointer array (Index 'rows')
    CUDA_CHECK(cudaMemcpy(*d_new_row_offsets + rows, &final_nnz, sizeof(int), cudaMemcpyHostToDevice));

    // printf("Filtering Result: %d -> %d elements threshold:%f \n", old_nnz, final_nnz,threshold);

    // ---------------------------------------------------------------------
    // 4. Compact Data Arrays (DeviceSelect)
    // ---------------------------------------------------------------------
    
    // Allocate Result Arrays
    CUDA_CHECK(cudaMalloc((void**)d_new_cols, final_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_new_vals, final_nnz * sizeof(double)));
    
    // CUB requires a pointer to store the number of selected items (on device)
    int* d_num_selected;
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected, sizeof(int)));

    // Cleanup temp storage again
    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    // A. Query memory requirement (Using d_vals as reference)
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // C. Run Select (Compact Values)
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));

    // D. Run Select (Compact Columns)
    // Note: We can reuse d_temp_storage because the operation size is identical
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_cols, d_flags, *d_new_cols, d_num_selected, old_nnz));

    // Verify correct number of items selected (Optional Check)
    int debug_nnz = 0;
    CUDA_CHECK(cudaMemcpy(&debug_nnz, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
    if (debug_nnz != final_nnz) {
        fprintf(stderr, "Mismatch! Calculated NNZ: %d, DeviceSelect NNZ: %d\n", final_nnz, debug_nnz);
    }

    // ---------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_row_counts));
    CUDA_CHECK(cudaFree(d_num_selected));
    
    // Wait for everything to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " - code " << cusparseGetErrorName(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

void print_csc_matrix(
    int num_rows,           // Number of rows
    int num_cols,           // Number of columns
    const int* d_col_ptr,   // DEVICE Pointer: Column Pointers
    const int* d_row_ind,   // DEVICE Pointer: Row Indices
    const double* d_val      // DEVICE Pointer: Values
) {
    printf("\n=== GPU CSC Matrix Dump (%d rows x %d cols) ===\n", num_rows, num_cols);

    // 1. Copy Column Pointers (Size = num_cols + 1)
    std::vector<int> h_col_ptr(num_cols + 1);
    CUDA_CHECK(cudaMemcpy(h_col_ptr.data(), d_col_ptr, (num_cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // 2. Determine Total NNZ from the last element of col_ptr
    int total_nnz = h_col_ptr[num_cols];
    printf("Total Non-Zeros (read from GPU): %d\n", total_nnz);

    if (total_nnz == 0) {
        printf("Matrix is empty.\n");
        return;
    }

    // 3. Allocate and Copy Row Indices and Values
    std::vector<int> h_row_ind(total_nnz);
    std::vector<double> h_val(total_nnz);

    CUDA_CHECK(cudaMemcpy(h_row_ind.data(), d_row_ind, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(), d_val, total_nnz * sizeof(double), cudaMemcpyDeviceToHost));

    printf("---------------------------------------------------\n");

    // 4. Iterate and Print (Same logic as before)
    // CSC iterates Column by Column
    for (int col = 0; col < num_cols; col++) {
        
        int start_idx = h_col_ptr[col];
        int end_idx   = h_col_ptr[col + 1];

        // Optional: Skip empty columns to reduce spam if matrix is huge
        if (start_idx == end_idx) continue; 

        printf("Column %d (Ptrs: %d -> %d):\n", col, start_idx, end_idx);

        for (int i = start_idx; i < end_idx; i++) {
            int row = h_row_ind[i];
            double val = h_val[i];

            printf("    Row %d : %.6f", row, val);

            // Validation: Check if row index is valid
            if (row < 0 || row >= num_rows) {
                printf("  <-- ERROR: Row Index %d out of bounds [0, %d)", row, num_rows);
            }
            // Validation: Check if column indices are sorted (Crucial for cuDSS/cuSPARSE)
            if (i > start_idx && row <= h_row_ind[i-1]) {
                printf("  <-- WARNING: Unsorted or Duplicate Row Index!");
            }
            printf("\n");
        }
    }
    printf("===================================================\n\n");
}


/**
 * @brief Computes the matrix multiplication \f$ C = A^T \cdot A \f$ using cuSPARSE.
 *
 * This function performs the following major steps:
 * 1. Converts the input matrix \f$ A \f$ from Coordinate (COO) to Compressed Sparse Row (CSR) format.
 * 2. Computes the explicit transpose \f$ A^T \f$ using `cusparseCsr2cscEx2`.
 * 3. Sets up the SpGEMM (Sparse General Matrix-Matrix Multiplication) descriptor for \f$ A^T \cdot A \f$.
 * 4. Computes the multiplication and filters the result based on a numeric threshold to remove near-zero values.
 * 5. Converts the final result \f$ C \f$ back to CSR format for output.
 *
 * @note This function handles internal memory allocation for the result arrays (`d_result_rows`, `d_result_cols`, `d_result_vals`, etc.).
 * **The caller is responsible for freeing these pointers using `cudaFree`.**
 *
 * @param[in]  d_rows_coo      Device pointer to the array of row indices for matrix \f$ A \f$ (COO format).
 * @param[in]  d_cols_coo      Device pointer to the array of column indices for matrix \f$ A \f$ (COO format).
 * @param[in]  d_vals_coo      Device pointer to the array of values for matrix \f$ A \f$ (COO format).
 * @param[in]  nnz             Number of non-zero elements in matrix \f$ A \f$.
 * @param[in]  num_rows        Number of rows in matrix \f$ A \f$.
 * @param[in]  num_cols        Number of columns in matrix \f$ A \f$.
 *
 * @param[out] d_result_rows   Pointer to a device pointer that will receive the row indices of the result matrix \f$ C \f$ (CSR).
 * @param[out] d_result_cols   Pointer to a device pointer that will receive the column indices of the result matrix \f$ C \f$ (CSR).
 * @param[out] d_result_vals   Pointer to a device pointer that will receive the values of the result matrix \f$ C \f$ (CSR).
 * @param[out] result_nnz      Pointer to an integer that will receive the number of non-zeros in the result matrix \f$ C \f$.
 *
 * @param[out] d_AT_cscOffsets Pointer to a device pointer that will receive the CSC col offsets (or CSR row offsets of \f$ A^T \f$).
 * @param[out] d_AT_columns    Pointer to a device pointer that will receive the CSC row indices (or CSR col indices of \f$ A^T \f$).
 * @param[out] d_AT_values     Pointer to a device pointer that will receive the CSC values (or CSR values of \f$ A^T \f$).
 *
 * @warning This function performs significant GPU memory allocation for intermediate buffers (SpGEMM work estimation and computation).
 * Ensure sufficient device memory is available.
 * @see cusparseSpGEMM_compute
 */
void compute_AtA_debug(
    int* d_rows_coo,
    int* d_cols_coo,
    double* d_vals_coo,
    int nnz,
    int num_rows,
    int num_cols,
    int** d_result_rows,
    int** d_result_cols,
    double** d_result_vals,
    int* result_nnz,
    int** d_AT_cscOffsets, 
    int** d_AT_columns,
    double** d_AT_values)
{
    // -------------------------------------------------------------------------
    // 0. Setup Input Data (Convert COO matrix to CSR)
    // -------------------------------------------------------------------------
    cusparseHandle_t handle; //Necessary for cusparse
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    // Convert COO to CSR for A
    int *d_csrRowPtrA, *d_csrColIndA;
    double *d_csrValA;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA, nnz * sizeof(double)));

    CUSPARSE_CHECK(cusparseXcoo2csr(handle, d_rows_coo, nnz, num_rows, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    
    //Copy the values 
    CUDA_CHECK(cudaMemcpy(d_csrColIndA, d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrValA, d_vals_coo, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    // std::cout << "Converted A to CSR" << std::endl;

    //-------------------------------------------------------------------------
    // 1. Explicit Transpose: Generate A^T
    //    We use cusparseCsr2cscEx2. The output CSC arrays of A 
    //    are exactly the CSR arrays of A^T. (see note at page 20 cusparse 13.1 manual)
    // -------------------------------------------------------------------------

    // Allocate memory for A^T (Same NNZ, but dimensions swapped if non-square)

    // Note: A^T has n rows and m cols
    CUDA_CHECK(cudaMalloc((void**)d_AT_cscOffsets, (num_cols + 1) * sizeof(int))); 

    CUDA_CHECK(cudaMalloc((void**)d_AT_columns, nnz * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)d_AT_values, nnz * sizeof(double)));

    size_t transposeBufferSize = 0;
    void* d_transposeBuffer = NULL;

    // Query buffer size for CSR -> CSC (which is effectively A -> A^T)
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns, // Output arrays
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        &transposeBufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_transposeBuffer, transposeBufferSize));

    // Perform the transpose
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        d_transposeBuffer
    ));
    
    // -------------------------------------------------------------------------
    // 2. Initialize cuSPARSE and Matrix Descriptors
    // -------------------------------------------------------------------------

    //CUDA_R_64F is double 
    //32-bit indices CUSPARSE_INDEX_32I is supported (double) or 64-bit indices (CUSPARSE_INDEX_64I)
    cusparseSpMatDescr_t matA, matAt, matC;
   
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                      d_csrRowPtrA, d_cols_coo, d_vals_coo,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    CUSPARSE_CHECK(cusparseCreateCsr(&matAt, num_cols, num_rows, nnz,
                                     *d_AT_cscOffsets, *d_AT_columns, *d_AT_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateCsr(&matC, num_cols, num_cols, 0,
                      nullptr, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    // std::cout << "Created matrix descriptors (A^T: " << num_cols << "x" << num_rows 
    //           << ", A: " << num_rows << "x" << num_cols << ")" << std::endl;
    
    // -------------------------------------------------------------------------
    // 3. SpGEMM Setup
    // -------------------------------------------------------------------------
    double alpha = 1.0, beta = 0.0;
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    // Set operations: opA = Transpose, opB = Non-Transpose
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    
    // -------------------------------------------------------------------------
    // 4. Work Estimation (Buffer Sizing)
    // -------------------------------------------------------------------------
    size_t bufferSize1 = 0;
    void* dBuffer1 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA,opB,
                                   &alpha, matAt, matA, &beta, matC,
                                   CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, nullptr));
    
    // std::cout << "Buffer size 1: " << bufferSize1 << " bytes" << std::endl;
    
    if (bufferSize1 > 0) {
        CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA,opB,
                                   &alpha, matAt, matA, &beta, matC,
                                   CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, dBuffer1));
    
    // std::cout << "Work estimation complete" << std::endl;
    
    // -------------------------------------------------------------------------
    // 5. Compute Structure 
    // -------------------------------------------------------------------------
    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, nullptr));
    
    // std::cout << "Buffer size 2: " << bufferSize2 << " bytes" << std::endl;
    
    if (bufferSize2 > 0) {
        CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
    }
    
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, dBuffer2));
    
    // std::cout << "Compute complete" << std::endl;
    
    // -------------------------------------------------------------------------
    // 6. Allocate C and Copy Results
    // -------------------------------------------------------------------------
    // Get result size
    int64_t C_num_rows, C_num_cols, C_nnz; //They need int64_t
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
    
    
    // Allocate result CSR
    int *d_csrRowPtrC, *d_csrColIndC;
    double *d_csrValC;
    
    CUDA_CHECK(cudaMalloc(&d_csrRowPtrC, (C_num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndC, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValC, C_nnz * sizeof(double)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));
    
    // Copy result
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, opA,opB,
                       &alpha, matAt, matA, &beta, matC,
                       CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    
    // std::cout << "Result copied" << std::endl;

    //Define pointers for the trimed version
    int *d_trim_row_ptr, *d_trim_cols_ptr;
    double *d_trim_vals_ptr;
    int trim_nnz = 0;
    
    //Filter threshold
    double threshold = 1e-8;

    
    //Trim for zeroes
    filter_csr_cub(threshold,C_num_rows, C_nnz, d_csrRowPtrC, d_csrColIndC, d_csrValC, &d_trim_row_ptr, &d_trim_cols_ptr, &d_trim_vals_ptr,&trim_nnz);


    // print_csr_matrix(num_cols, trim_nnz, d_trim_row_ptr, d_trim_cols_ptr, d_trim_vals_ptr);

    *d_result_rows= d_trim_row_ptr;
    *d_result_cols = d_trim_cols_ptr; 
    *d_result_vals = d_trim_vals_ptr;
    *result_nnz = trim_nnz;
    
    // 1. Free SpGEMM Buffers
    if (dBuffer1) cudaFree(dBuffer1);
    if (dBuffer2) cudaFree(dBuffer2);

    // 2. Free Input/Intermediate CSR Arrays
    cudaFree(d_csrRowPtrA);
    
    // 3. Free Unfiltered Result C Arrays
    cudaFree(d_csrRowPtrC);
    cudaFree(d_csrColIndC);
    cudaFree(d_csrValC);

    // 4. Free Transpose Buffer
    if (d_transposeBuffer) cudaFree(d_transposeBuffer);


    // 6. Destroy Descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matC);
    cusparseDestroySpMat(matAt);    

    // 7. Destroy Handle
    cusparseDestroy(handle);
}

__global__ void identity_csr_and_scale_kernel(int N, double alpha, 
                                    int* row_offsets, int* cols, double* vals) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Set Column: Diagonal means Col == Row
        cols[idx] = idx;
        
        // Set Value: Scaled Identity
        vals[idx] = alpha;
        
        // Set Row Pointer: For identity, row_ptr[i] is just i
        row_offsets[idx] = idx;
    }
    
    // Handle the very last element of the row pointer array (N+1 size)
    // It must equal NNZ (which is N)
    if (idx == N) {
        row_offsets[idx] = N;
    }
}


/**
 * @brief Initializes a square diagonal matrix in CSR format on the GPU.
 *
 * This function launches a CUDA kernel to generate a scaled identity matrix 
 * \f$ I \cdot \alpha \f$ of size \f$ N \times N \f$.
 *
 * The resulting CSR structure will have:
 * - **Row Offsets:** \f$ [0, 1, 2, ..., N] \f$
 * - **Column Indices:** \f$ [0, 1, 2, ..., N-1] \f$
 * - **Values:** \f$ [\alpha, \alpha, ..., \alpha] \f$
 *
 * @note **Memory Allocation Requirement:**
 * The caller is responsible for allocating device memory before calling this function:
 * - `d_row_offsets`: Must be size \f$ (N + 1) \times \text{sizeof(int)} \f$.
 * - `d_cols`: Must be size \f$ N \times \text{sizeof(int)} \f$.
 * - `d_vals`: Must be size \f$ N \times \text{sizeof(double)} \f$.
 *
 * @param[in]  N             The dimension of the square matrix (number of rows/columns).
 * @param[in]  alpha         The scalar value to place on the diagonal (scaling factor).
 * @param[out] d_row_offsets Device pointer to the array that will hold the CSR row offsets.
 * @param[out] d_cols        Device pointer to the array that will hold the column indices.
 * @param[out] d_vals        Device pointer to the array that will hold the non-zero values.
 */
void create_identity_csr_and_scale(int N, double alpha, 
                                        int* d_row_offsets, int* d_cols, double* d_vals) 
{
    int blockSize = 256;
    int gridSize = ( (N + 1) + blockSize - 1) / blockSize;
    // We launch N+1 threads to handle the extra row pointer element safely
    identity_csr_and_scale_kernel<<<gridSize, blockSize>>>(N, alpha, d_row_offsets, d_cols, d_vals);
    CUDA_CHECK(cudaGetLastError());
}




/**
 * @brief Computes the scaled sum of two CSR matrices: \f$ C = \delta \cdot A + \delta \cdot B \f$.
 *
 * This function uses the cuSPARSE `csrgeam` (General Matrix Addition) routine to perform
 * sparse matrix addition. It handles the complete workflow:
 * 1. **Buffer Estimation:** Determines workspace size.
 * 2. **Symbolic Phase:** Computes the non-zero structure (sparsity pattern) of \f$ C \f$ and its row offsets.
 * 3. **Memory Allocation:** Allocates the column indices and values arrays for \f$ C \f$.
 * 4. **Numeric Phase:** Computes the actual values of \f$ C \f$.
 *
 * @section Format Matrix Format
 * - **Input A:** CSR (Compressed Sparse Row)
 * - **Input B:** CSR (Compressed Sparse Row)
 * - **Output C:** CSR (Compressed Sparse Row)
 * * All matrices are assumed to be **0-indexed** and **row-major**.
 *
 * @note **Memory Responsibility:**
 * This function allocates device memory for the output matrix arrays:
 * - `*d_C_row_offsets`
 * - `*d_C_columns`
 * - `*d_C_values`
 *
 * **The caller is responsible for freeing these pointers using `cudaFree`.**
 *
 * @param[in]  delta           The scaling factor applied to both matrices (\f$ \alpha = \delta, \beta = \delta \f$).
 * @param[in]  m               Number of rows in matrices A, B, and C.
 * @param[in]  n               Number of columns in matrices A, B, and C.
 *
 * @param[in]  nnzA            Number of non-zero elements in matrix A.
 * @param[in]  d_A_row_offsets Device pointer to row offsets of matrix A (CSR).
 * @param[in]  d_A_columns     Device pointer to column indices of matrix A (CSR).
 * @param[in]  d_A_values      Device pointer to values of matrix A (CSR).
 *
 * @param[in]  nnzB            Number of non-zero elements in matrix B.
 * @param[in]  d_B_row_offsets Device pointer to row offsets of matrix B (CSR).
 * @param[in]  d_B_columns     Device pointer to column indices of matrix B (CSR).
 * @param[in]  d_B_values      Device pointer to values of matrix B (CSR).
 *
 * @param[out] nnzC_out        Pointer to an integer that will receive the total non-zeros in result C.
 * @param[out] d_C_row_offsets Pointer to a device pointer that will receive C's row offsets (CSR).
 * @param[out] d_C_columns     Pointer to a device pointer that will receive C's column indices (CSR).
 * @param[out] d_C_values      Pointer to a device pointer that will receive C's values (CSR).
 */
void add_csr_cusparse(
    double delta,int m, int n,                          // Matrix dimensions (rows, cols)
    int nnzA,                              // Non-zeros in A
    const int* d_A_row_offsets, const int* d_A_columns, const double* d_A_values,
    int nnzB,                              // Non-zeros in B
    const int* d_B_row_offsets, const int* d_B_columns, const double* d_B_values,
    int* nnzC_out,                         // Output: Non-zeros in C
    int** d_C_row_offsets, int** d_C_columns, double** d_C_values // Output: Pointers to C data
) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Scalar multipliers (alpha = 1.0, beta = 1.0 for simple A+B)
    const double alpha = delta;
    const double beta = delta;

    // 1. Create Matrix Descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrB));
    CUSPARSE_CHECK(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrC));
    CUSPARSE_CHECK(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));

    // 2. Allocate C Row Offsets (Always m + 1 size)
    CUDA_CHECK(cudaMalloc((void**)d_C_row_offsets, (m + 1) * sizeof(int)));

    // 3. Query Buffer Size
    // We need a buffer for the calculation. cuSPARSE calculates this for us.
    void* d_buffer = NULL;
    size_t bufferSize = 0;
    
    CUSPARSE_CHECK(cusparseDcsrgeam2_bufferSizeExt(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, d_A_values, *d_C_row_offsets, NULL, // C vals/cols are NULL for buffer query
        &bufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));


//     int test_offset_A, test_offset_B;
// cudaMemcpy(&test_offset_A, &d_A_row_offsets[m], sizeof(int), cudaMemcpyDeviceToHost);
// cudaMemcpy(&test_offset_B, &d_B_row_offsets[m], sizeof(int), cudaMemcpyDeviceToHost);
// printf("A Last Offset: %d (Expected %d), B Last Offset: %d (Expected %d)\n", 
//         test_offset_A, nnzA, test_offset_B, nnzB);

    // 4. Symbolic Phase: Calculate nnzC and C_row_offsets
    // This fills d_C_row_offsets and determines the total number of non-zeros.
    int nnzC = 0;
    CUSPARSE_CHECK(cusparseXcsrgeam2Nnz(
        handle, m, n,
        descrA, nnzA, d_A_row_offsets, d_A_columns,
        descrB, nnzB, d_B_row_offsets, d_B_columns,
        descrC, *d_C_row_offsets, &nnzC,d_buffer
    ));

    
    *nnzC_out = nnzC;

    // 5. Allocate C Columns and Values
    // Now that we know nnzC, we can allocate the rest of the matrix.
    CUDA_CHECK(cudaMalloc((void**)d_C_columns, nnzC * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_C_values, nnzC * sizeof(double)));

    // 6. Numeric Phase: Fill C_columns and C_values
    // Performs the actual addition.
    CUSPARSE_CHECK(cusparseDcsrgeam2(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, *d_C_values, *d_C_row_offsets, *d_C_columns,
        d_buffer
    ));

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_buffer));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrB));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrC));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

/**
 * @brief Computes a scaled sparse matrix-vector multiplication: \f$ y = \alpha \cdot A \cdot x \f$.
 *
 * This function performs the operation on the GPU using cuSPARSE. It handles the allocation
 * of the result vector \f$ y \f$ internally.
 *
 * @section Format Matrix Format
 * - **Input Matrix A:** **CSC (Compressed Sparse Column)**
 * - **Memory Layout:** **Column-Major** (Values are grouped by column).
 * * @note **Memory Responsibility:**
 * - This function **allocates** device memory for the output vector `*d_y_out`.
 * - **The caller is responsible for freeing `*d_y_out` using `cudaFree`.**
 * - The input pointers (`d_col_offsets`, `d_row_indices`, `d_values`, `d_x`) must point to valid device memory allocated by the caller.
 *
 * @param[in]  rows          Number of rows in matrix A (dimension of vector y).
 * @param[in]  cols          Number of columns in matrix A (dimension of vector x).
 * @param[in]  nnz           Number of non-zero elements in matrix A.
 * * @param[in]  d_col_offsets Device pointer to the CSC column offsets array.
 * Size: \f$ \text{cols} + 1 \f$.
 * @param[in]  d_row_indices Device pointer to the CSC row indices array.
 * Size: \f$ \text{nnz} \f$.
 * @param[in]  d_values      Device pointer to the non-zero values of matrix A.
 * Size: \f$ \text{nnz} \f$.
 * * @param[in]  d_x           Device pointer to the dense input vector x.
 * Size: \f$ \text{cols} \times \text{sizeof(double)} \f$.
 * @param[in]  alpha         Scalar scaling factor \f$ \alpha \f$.
 * * @param[out] d_y_out       Address of a pointer. The function will allocate memory at this location
 * and store the result vector y.
 * Size: \f$ \text{rows} \times \text{sizeof(double)} \f$.
 */
void scale_and_multiply_on_gpu(
    int rows, int cols, int nnz,
    const int* d_col_offsets,  // Input: Device Pointer
    const int* d_row_indices,  // Input: Device Pointer
    const double* d_values,     // Input: Device Pointer
    const double* d_x,          // Input: Device Pointer
    double alpha,               // Scalar
    double** d_y_out            // Output: Address of pointer to allocate
) {
    // 1. Allocate Result Memory on GPU
    // We dereference d_y_out to set the caller's pointer
    CUDA_CHECK( cudaMalloc((void**)d_y_out, rows * sizeof(double)) );
    
    // Initialize result to 0 (optional but good practice for safety)
    CUDA_CHECK( cudaMemset(*d_y_out, 0, rows * sizeof(double)) );

    // 2. Create cuSPARSE Context
    cusparseHandle_t handle;
    CUSPARSE_CHECK( cusparseCreate(&handle) );

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // 3. Create Descriptors using the DEVICE pointers passed in
    CUSPARSE_CHECK( cusparseCreateCsc(&matA, rows, cols, nnz,
                                      (void*)d_col_offsets, (void*)d_row_indices, (void*)d_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

    CUSPARSE_CHECK( cusparseCreateDnVec(&vecX, cols, (void*)d_x, CUDA_R_64F) );
    
    // Use the newly allocated pointer (*d_y_out) for the Y descriptor
    CUSPARSE_CHECK( cusparseCreateDnVec(&vecY, rows, *d_y_out, CUDA_R_64F) );

    // 4. Allocate Workspace
    void* d_buffer = nullptr;
    size_t bufferSize = 0;
    double beta = 0.0; //overwriting Y, not accumulating

    // int last_offset;
    // cudaMemcpy(&last_offset, &d_col_offsets[cols], sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Last offset: %d, NNZ: %d\n", last_offset, nnz);

    CUSPARSE_CHECK( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) );

    CUDA_CHECK( cudaMalloc(&d_buffer, bufferSize) );

    // 5. Execute Operation (GPU only)
    CUSPARSE_CHECK( cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer) );

    // 6. Cleanup Local Resources
    // Note: We DO NOT free d_col_offsets, d_x, or *d_y_out because the caller owns them.
    CUSPARSE_CHECK( cusparseDestroyDnVec(vecY) );
    CUSPARSE_CHECK( cusparseDestroyDnVec(vecX) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) );
    CUDA_CHECK( cudaFree(d_buffer) );
    CUSPARSE_CHECK( cusparseDestroy(handle) );
}





#define CHECK_CUDSS(func, msg) { \
    cudssStatus_t status = (func); \
    if (status != CUDSS_STATUS_SUCCESS) { \
        printf("cuDSS Error in %s at line %d. Status: %d\n", msg, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// ---------------------------------------------------------
// DEBUG: Sanity Check Indices on GPU
// ---------------------------------------------------------
void check_indices_sanity(int nnz, int num_rows, const int* d_indices) {
    // Simple way: Copy to host (slow but safe for debugging)
    std::vector<int> h_indices(nnz);
    cudaMemcpy(h_indices.data(), d_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<nnz; i++) {
        if(h_indices[i] < 0 || h_indices[i] >= num_rows) {
            printf("FATAL ERROR at index %d: Value %d is out of bounds [0, %d)\n", 
                   i, h_indices[i], num_rows);
            return; // Found the bug
        }
    }
    printf("Indices look valid (Range checks passed).\n");
}

/**
 * @brief Solves the linear system \f$ A \cdot x = b \f$ using the cuDSS direct solver.
 *
 * This function performs a complete direct solve workflow (Analysis -> Factorization -> Solve)
 * using the NVIDIA cuDSS library. It is suitable for general square sparse matrices.
 *
 * @section Format Matrix Format
 * - **Input Matrix A:** CSR (Compressed Sparse Row).
 * - **Input Vectors (b, x):** Dense vectors.
 * - **Index Type:** 32-bit Integers (`int32_t`).
 * - **Value Type:** 32-bit doubles (`double`).
 *
 * @note **Memory Responsibility:**
 * - This function **allocates** device memory for the solution vector `*d_x_out`.
 * - **The caller is responsible for freeing `*d_x_out` using `cudaFree`.**
 * - The input arrays (`d_row_offsets`, `d_col_indices`, `d_values`, `d_b`) must be pre-allocated on the device.
 *
 * @param[in]  n             The dimension of the square system (number of rows/columns).
 * @param[in]  nnz           The number of non-zero elements in matrix A.
 * @param[in]  d_row_offsets Device pointer to the CSR row offsets array. 
 * Size: \f$ n + 1 \f$.
 * @param[in]  d_col_indices Device pointer to the CSR column indices array.
 * Size: \f$ nnz \f$.
 * @param[in]  d_values      Device pointer to the non-zero values of matrix A.
 * Size: \f$ nnz \f$.
 * @param[in]  d_b           Device pointer to the dense right-hand side vector \f$ b \f$.
 * Size: \f$ n \times \text{sizeof(double)} \f$.
 * @param[out] d_x_out       Address of a pointer. The function allocates memory at this location
 * and stores the solution vector \f$ x \f$.
 * Size: \f$ n \times \text{sizeof(double)} \f$.
 *
 * @warning This function assumes the matrix indices are **32-bit integers** (passed as `CUDA_R_32I`),
 * even though the function parameters `n` and `nnz` are `int64_t`. Ensure your device arrays strictly contain 32-bit integers to avoid type mismatches.
 */
void solve_system_gpu(
    int64_t n,                  
    int64_t nnz,                
    const int* d_row_offsets,   
    const int* d_col_indices,   
    const double* d_values,      
    const double* d_b,           
    double** d_x_out             
) {
    // check_indices_sanity(nnz, n, d_row_offsets);
    // check_indices_sanity(nnz, n, d_col_indices); 

        // std::cout << "\n=== Inside solve_system_gpu ===" << std::endl;
        // std::cout << "n=" << n << ", nnz=" << nnz << std::endl;
        
        // check_indices_sanity(nnz, n, d_col_indices);  // Check columns
        
        // // Also check row offsets are valid
        // int* h_row_check = new int[2];
        // CUDA_CHECK(cudaMemcpy(h_row_check, d_row_offsets, sizeof(int), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(h_row_check + 1, d_row_offsets + n, sizeof(int), cudaMemcpyDeviceToHost));
        // std::cout << "Row offsets: first=" << h_row_check[0] << ", last=" << h_row_check[1] << std::endl;
        // delete[] h_row_check;

    // 1. Allocate Result Vector X on GPU
    CUDA_CHECK( cudaMalloc((void**)d_x_out, n * sizeof(double)) );
    CUDA_CHECK( cudaMemset(*d_x_out, 0, n * sizeof(double)) );

    // 2. Initialize cuDSS Handles
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t solverData;
    cudssMatrix_t matA, vecB, vecX;

    CHECK_CUDSS( cudssCreate(&handle), "cudssCreate" )
    CHECK_CUDSS( cudssConfigCreate(&config), "cudssConfigCreate" )
    CHECK_CUDSS( cudssDataCreate(handle, &solverData), "cudssDataCreate" )

    // Change the config 
    int alg = CUDSS_ALG_1;
    CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,&alg ,sizeof(alg) ), "Changing config");

    int ir_steps = 3; 
    CHECK_CUDSS(cudssConfigSet(config,  CUDSS_CONFIG_IR_N_STEPS,  &ir_steps, sizeof(ir_steps)),"Changing config");

    // 3. Create Matrix Wrappers 
    CHECK_CUDSS( cudssMatrixCreateCsr(&matA, 
                                      n, n, nnz, 
                                      (void*)d_row_offsets, 
                                      NULL,                 
                                      (void*)d_col_indices, 
                                      (void*)d_values, 
                                      CUDA_R_32I,           // Index Type
                                      CUDA_R_64F,           // Value Type (double)
                                      CUDSS_MTYPE_GENERAL, 
                                      CUDSS_MVIEW_FULL, 
                                      CUDSS_BASE_ZERO), "cudssMatrixCreateCsr" )

    // Vector X (Dense double) //At the moment supports only dense vectors
    CHECK_CUDSS( cudssMatrixCreateDn(&vecX, n, 1, n, 
                                     (void*)*d_x_out, 
                                     CUDA_R_64F,            // Value Type (double)
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(X)" )

    // Vector B (Dense double) //At the moment supports only dense vectors
    CHECK_CUDSS( cudssMatrixCreateDn(&vecB, n, 1, n, 
                                     (void*)d_b, 
                                     CUDA_R_64F,            // Value Type (double)
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(B)" )

    //Logging               
    
    
    // 4. Run Solver Workflow
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solverData, matA, vecX, vecB), "Analysis" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solverData, matA, vecX, vecB), "Factorization" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solverData, matA, vecX, vecB), "Solve" )

    // 5. Cleanup
    CHECK_CUDSS( cudssMatrixDestroy(matA), "Destroy A" )
    CHECK_CUDSS( cudssMatrixDestroy(vecB), "Destroy B" )
    CHECK_CUDSS( cudssMatrixDestroy(vecX), "Destroy X" )
    CHECK_CUDSS( cudssDataDestroy(handle, solverData), "Destroy Data" )
    CHECK_CUDSS( cudssConfigDestroy(config), "Destroy Config" )
    CHECK_CUDSS( cudssDestroy(handle), "Destroy Handle" )
}

#define CHECK_CUSPARSE(func) \
{ \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
               __LINE__, cusparseGetErrorString(status), status); \
    } \
}

/**
 * @brief Sorts a COO format matrix by row indices (primary) and column indices (secondary).
 * * This function ensures that the input COO arrays are sorted, which is often required 
 * for conversion to CSR or other sparse operations.
 * * @param[in]     num_rows Number of rows.
 * @param[in]     num_cols Number of columns.
 * @param[in]     nnz      Number of non-zeros.
 * @param[in,out] d_rows   Device ptr to row indices (Sorted in-place).
 * @param[in,out] d_cols   Device ptr to col indices (Sorted in-place).
 * @param[in,out] d_vals   Device ptr to values (Reordered in-place to match indices).
 */
void sort_coo_matrix_cusparse(
    int num_rows, 
    int num_cols, 
    int nnz,
    int* d_rows,    // Device Pointer: Row Indices (Modified in-place)
    int* d_cols,    // Device Pointer: Col Indices (Modified in-place)
    double* d_vals   // Device Pointer: Values (Modified in-place)
) {
    // 1. Create Local Handle
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    
    int* d_permutation = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_permutation, nnz * sizeof(int)));
    
    double* d_values_sorted=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_values_sorted, nnz * sizeof(double)));

    // 3. Create Permutation Vector 
    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, nnz, nnz,
                                    d_permutation, d_values_sorted,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

     // Create dense vector for wrapping the original coo values
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, nnz, d_vals,
                                        CUDA_R_64F) )

    // 2. Allocate Buffer for Sorting
    void* d_buffer = nullptr;
    size_t bufferSize = 0;
     
    // Query working space of COO sort
    CHECK_CUSPARSE( cusparseXcoosort_bufferSizeExt(handle, num_rows,
                                                   num_cols, nnz, d_rows,
                                                   d_cols, &bufferSize) )                                        

    // Query required buffer size
    CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
        handle, num_rows, num_cols, nnz,
        d_rows, d_cols, 
        &bufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

    CHECK_CUSPARSE( cusparseCreateIdentityPermutation(handle, nnz,
                                                      d_permutation) )
   
    CHECK_CUSPARSE( cusparseXcoosortByRow(handle, num_rows, num_cols, nnz,
                                          d_rows, d_cols, d_permutation,
                                          d_buffer) )

    

    CHECK_CUSPARSE( cusparseGather(handle, vec_values, vec_permutation) )
    
    CUDA_CHECK(cudaMemcpy(d_vals, d_values_sorted, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    // 6. Cleanup
    CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CUDA_CHECK(cudaFree(d_permutation));
    CUDA_CHECK(cudaFree(d_buffer));
}

// Helper kernel to initialize indices: [0, 1, 2, ...]
__global__ void init_indices(int* ptr, int nnz) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) ptr[idx] = idx;
}

// Helper kernel to permute data based on sorted indices
// out[i] = in[ sorted_indices[i] ]
__global__ void permute_data(const int* in_cols, int* out_cols, 
                             const double* in_vals, double* out_vals, 
                             const int* sorted_indices, int nnz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < nnz) {
        int old_pos = sorted_indices[i];
        out_cols[i] = in_cols[old_pos];
        out_vals[i] = in_vals[old_pos];
    }
}


void sort_coo_cub(int* d_rows, int* d_cols, double* d_vals, int nnz) {
    
    // Pointers for sorted/temp arrays
    int *d_indices, *d_indices_sorted;
    int *d_rows_sorted, *d_cols_sorted; 
    double *d_vals_sorted;
    
    // 1. Allocate Temporary Buffers
    //    We need an index array to track where the rows move, so we can move cols/vals later.
    CUDA_CHECK(cudaMalloc((void**)&d_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_rows_sorted, nnz * sizeof(int)));
    
    //    Allocation of output buffers for cols/vals
    CUDA_CHECK(cudaMalloc((void**)&d_cols_sorted, nnz * sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_vals_sorted, nnz * sizeof(double)));

    // 2. Initialize Permutation Index [0, 1, 2, ..., NNZ-1]
    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;
    
    init_indices<<<numBlocks, blockSize>>>(d_indices, nnz);
    CUDA_CHECK(cudaGetLastError());

    // 3. Determine Temporary Storage Size for CUB Radix Sort
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    //    Query workspace requirement
    //    Key: d_rows, Value: d_indices
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));
    
    //    Allocate workspace
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // 4. Run the Sort
    //    Sorts 'rows' into 'rows_sorted', and moves 'indices' into 'indices_sorted'
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));

    // 5. Permute Cols and Vals using the new index order
    //    We effectively "apply" the sort permutation to the other two arrays.
    permute_data<<<numBlocks, blockSize>>>(d_cols, d_cols_sorted, 
                                           d_vals, d_vals_sorted, 
                                           d_indices_sorted, nnz);
    CUDA_CHECK(cudaGetLastError());

    // 6. Copy sorted data back to original pointers
    //    (Since the function arguments are pointers-by-value, we must copy the data back 
    //     to the original locations unless you change the function signature to accept int**).
    CUDA_CHECK(cudaMemcpy(d_rows, d_rows_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, d_cols_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, d_vals_sorted, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

    // Cleanup
    CUDA_CHECK(cudaFree(d_indices)); 
    CUDA_CHECK(cudaFree(d_indices_sorted)); 
    CUDA_CHECK(cudaFree(d_rows_sorted)); 
    CUDA_CHECK(cudaFree(d_cols_sorted)); 
    CUDA_CHECK(cudaFree(d_vals_sorted)); 
    CUDA_CHECK(cudaFree(d_temp_storage));
}

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error:\n" \
                      << "  File:     " << __FILE__ << "\n" \
                      << "  Line:     " << __LINE__ << "\n" \
                      << "  Function: " << #call << "\n" \
                      << "  Status:   " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



/**
 * @brief Computes half the squared norm (0.5 * sum of squares) of a GPU array.
 *
 * This function calculates the dot product of the input vector with itself
 * using the highly optimized cuBLAS library, and then scales the final scalar 
 * result by 0.5 on the host (CPU).
 *
 * @note Mathematically, this computes half the squared norm ($0.5 \times ||x||^2$). 
 *
 * @param handle   A pre-initialized cuBLAS handle to manage GPU context and streams.
 * @param residual Pointer to the single-precision double array residing in device (GPU) memory.
 * @param n        The number of elements in the `residual` array.
 * @return double   The computed sum of the squared elements, multiplied by 0.5 written bask to (CPU) memory.
 */
double square_norm(cublasHandle_t handle, double * residual, int n)
{
    double result = 0.0;
    
    // Ensure the scalar result is written directly to the host variable 'result'
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    
    CUBLAS_CHECK(cublasDdot(handle, n, residual, 1, residual, 1, &result));
    
    return result * 0.5;
}

/**
 * @brief Computes  the exact L2 norm squared ( ||x||_2 ^2)  of a GPU array.
 *
 * This function calculates the true Euclidean norm (the square root of the sum 
 * of squared elements) of the input vector using the highly optimized 
 * cuBLAS library. 
 * * @note Unlike a standard dot product (`cublasSdot`), `cublasSnrm2` is specifically 
 * designed to be numerically stable. It guards against intermediate underflow 
 * or overflow that can happen when squaring very large or very small numbers.
 *
 * @param handle   A pre-initialized cuBLAS handle to manage GPU context and streams.
 * @param residual Pointer to the single-precision double array residing in device (GPU) memory.
 * @param n        The number of elements in the `residual` array.
 * @return double   The computed exact L2 norm of the array, multiplied by 0.5.
 */
double L2_norm_squared(cublasHandle_t handle, double * residual, int n)
{
    double result = 0.0;
    
    // Ensure the scalar result is written directly to the host variable 'result'
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    
    // Compute the exact L2 norm (Euclidean norm) of 'residual'
    // Parameters: handle, n, X, strideX, output_result
    CUBLAS_CHECK(cublasDnrm2(handle, n, residual, 1, &result));
    
    return result * result ;
}


/**
 * @brief CUDA kernel to calculate the element-wise magnitude of a 3D velocity field.
 * * Each thread computes the Euclidean norm for a single point in space using the formula:
 * velmag = sqrt(u^2 + v^2 + w^2).
 *
 * @param[in]  n      The total number of elements in the input vectors.
 * @param[in]  u      Device pointer to the x-velocity component array.
 * @param[in]  v      Device pointer to the y-velocity component array.
 * @param[in]  w      Device pointer to the z-velocity component array.
 * @param[out] velmag Device pointer to the array where the calculated magnitudes will be stored.
 */
__global__ void vel_mag_kernel(int n, const double* u, const double* v, const double* w, double* velmag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        velmag[i] = sqrtf(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}


/**
 * @brief Allocates GPU memory and computes the velocity magnitude across three components.
 * * This wrapper handles the memory allocation for the result vector and launches the 
 * vel_mag_kernel. It uses the CUDA_CHECK macro to ensure all GPU operations succeed.
 *
 * @warning This function allocates memory on the GPU using cudaMalloc. The caller is 
 * responsible for releasing this memory using @ref cudaFree() to prevent leaks.
 *
 * @param[in] n    The number of elements in the vectors.
 * @param[in] d_u  Device pointer to the input u-component vector.
 * @param[in] d_v  Device pointer to the input v-component vector.
 * @param[in] d_w  Device pointer to the input w-component vector.
 * * @return double* A pointer to the newly allocated device memory containing the 
 * velocity magnitude vector. Returns nullptr if allocation fails.
 */
double* compute_vel_mag(int n, double* d_u, double* d_v, double* d_w) {
    double* d_velmag = nullptr;

    // 1. Allocate GPU memory for the result vector
    size_t size = n * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_velmag, size));

    // 2. Configure execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // print_gpu_array(d_u, n);
    // 3. Launch the kernel
    vel_mag_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_u, d_v, d_w, d_velmag);
    
    // 4. Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Return the pointer to the newly allocated memory
    return d_velmag;
}

/**
 * @brief Transfers data from a GPU device array to a host std::vector.
 * * This function minimizes overhead by pre-allocating the vector and
 * using Move Semantics on the return. 
 *
 * @param d_ptr Pointer to the source data in GPU memory.
 * @param n     Number of elements (not bytes) to copy.
 * @return std::vector<double> The populated host vector.
 */
std::vector<double> copy_gpu_array_host(const double* d_ptr, int n) {
    // 1. Create the vector and reserve/resize immediately.
    std::vector<double> h_vec(n);

    // 2. Perform a single direct copy from Device to Host.
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost));

    // 3. Return by value. 
    // Modern C++ (C++11 and later) uses "Move Semantics." 
    // The vector's internal pointer is handed over to the caller 
    // rather than copying the whole array again.
    return h_vec;
}

/**
 * @brief CUDA kernel to add  constant to a vector element-wise.
 * * @param[in,out] d_vec    Pointer to the device array (vector) to be modified.
 * @param[in]     constant The scalar value to add to each element.
 * @param[in]     N        The total number of elements in the vector.
 */
__global__ void addConstantKernel(double *d_vec, double constant, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        d_vec[idx] += constant;
    }
}

/**
 * @brief Wrapper function to launch the vector addition CUDA kernel.
 * * This function calculates the appropriate execution configuration (grid and 
 * block dimensions) based on the vector size `N`, and launches the kernel. 
 * * @note The pointer `d_vec` must point to memory already allocated on the GPU 
 * (e.g., via `cudaMalloc`).
 * * @param[in,out] d_vec    Pointer to the device memory holding the vector.
 * @param[in]     constant The scalar value to add to each element.
 * @param[in]     N        The total number of elements in the vector.
 */
void addConstantToVector(double *d_vec, double constant, int N) {

    // Define execution configuration
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addConstantKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, constant, N);
    CUDA_CHECK(cudaGetLastError());
}


__global__ void check_vector_finite_kernel(const double* data, int n, int* flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (isinf(data[i]) || isnan(data[i] || data[i] > 1e12 || data[i] < 1e-12)) {
            atomicExch(flag, 1); // Set flag to 1 if a non-finite value is found
        }
    }
}

// Host wrapper to check for NaN/Inf in a device vector. Returns true if all values are finite.
bool is_vector_finite(const double* d_data, int n) {
    if (d_data == nullptr) {
        //returning true assumes it's not an error state.
        return true; 
    }

    int* d_flag;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    check_vector_finite_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, d_flag);
    CUDA_CHECK(cudaGetLastError());

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_flag));

    return h_flag == 0;
}

/**
 * Checks if the system Ax=b is ill-conditioned or singular.
 * * @param handle      cublasHandle
 * @param d_residual  Device pointer to vector r = (b - Ax)
 * @param d_b         Device pointer to vector b (Right Hand Side)
 * @param d_x         Device pointer to vector x (Solution)
 * @param n           Dimension of the system
 * @param threshold   Tolerance (default 1e-6 for double precision)
 * @return            true if the system is likely ill-defined/singular
 */
bool is_system_ill_defined(cublasHandle_t handle, 
                           double* d_residual, 
                           double* d_b, 
                           double* d_x, 
                           int n, 
                           double threshold = 1e-7) 
{
    // 1. Compute Norm of RHS (b)
    // We use sqrt() because your function returns the squared norm
    double norm_b_sq = L2_norm_squared(handle, d_b, n);
    double norm_b = sqrt(norm_b_sq);

    // Handle edge case: if b is all zeros, the solution should be zero.
    if (norm_b < 1e-15) {
        printf("System Check: RHS is effectively zero.\n");
        return false; 
    }

    // 2. Compute Norm of Residual (r = b - Ax)
    double norm_r_sq = L2_norm_squared(handle, d_residual, n);
    double norm_r = sqrt(norm_r_sq);

    // 3. Compute Norm of Solution (x) for growth check
    double norm_x_sq = L2_norm_squared(handle, d_x, n);
    double norm_x = sqrt(norm_x_sq);

    // --- Metrics ---

    // Metric A: Relative Residual
    // Ideally, this should be close to Machine Epsilon (1e-16 for double)
    double rel_residual = norm_r / norm_b;

    // Metric B: Expansion Factor (Heuristic)
    // If ||x|| is 1e15 times larger than ||b||, the matrix is likely near-singular
    double expansion = norm_x / norm_b;

    // --- Diagnosis ---
    
    bool is_bad = false;

    // Check 1: Did the solver fail to get close?
    if (rel_residual > threshold) {
        printf("WARNING: System is Ill-Conditioned or Singular.\n");
        printf("  > High Relative Residual: %e (Threshold: %e)\n", rel_residual, threshold);
        is_bad = true;
    }

    // Check 2: Did the solution explode?
    // A huge expansion factor often implies condition number is very high.
    if (expansion > 1e12) { // 1e12 is a safe heuristic for "too big"
        printf("WARNING: Matrix is near-singular (Massive solution growth).\n");
        printf("  > Expansion Factor ||x||/||b||: %e\n", expansion);
        is_bad = true;
    }

    if (!is_bad) {
        // Optional: Print success for debugging
         printf("System Stable. Rel Residual: %e\n", rel_residual);
    }

    return is_bad;
}

double* levenberg_marquardt_solver(
    double Re,
    std::vector<double>& y, // In-out: initial guess and final solution
    const int xN, const int yN, const int zN,
    const double* u_inlet, // Host pointer
    const double dx, const double dy, const double dz,
    int max_iterations,
    double initial_lambda,
    double lambda_factor,
    double tolerance
) {
    const int nCell = 4 * xN * yN * zN;
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    double* d_y;
    CUDA_CHECK(cudaMalloc(&d_y, nCell * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), nCell * sizeof(double), cudaMemcpyHostToDevice));

    double lambda = initial_lambda;

    if (lambda_factor <= 1.0) {
        std::cout << "ERROR: lambda_factor must be > 1.0, got " << lambda_factor << std::endl;
        lambda_factor = 10.0; // Use default
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "\n--- Iteration " << iter << ", Lambda: " << lambda << " ---" << std::endl;

        // Sync device to host at start of iteration
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, nCell * sizeof(double), cudaMemcpyDeviceToHost));

        // 1. Calculate Jacobian and Residual
        auto [d_r, jacobian_coo] = Residuals_Sparse_Jacobian_finite_diff(Re, y.data(), xN, yN, zN, u_inlet, dx, dy, dz);
        auto [d_rows_coo, d_cols_coo, d_vals_coo, nnz] = jacobian_coo;

        // 2. Calculate current cost
        double cost = square_norm(cublas_handle, d_r, nCell);
        std::cout << "Current cost: " << std::scientific << cost << std::endl;

        // 3. Check for convergence
        if (cost < tolerance) {
            std::cout << "Convergence reached." << std::endl;
            CUDA_CHECK(cudaFree(d_r));
            CUDA_CHECK(cudaFree(d_rows_coo));
            CUDA_CHECK(cudaFree(d_cols_coo));
            CUDA_CHECK(cudaFree(d_vals_coo));
            break;
        }

        // 4. Sort Jacobian from COO to CSR for SpMM
        sort_coo_matrix_cusparse(nCell, nCell, nnz, d_rows_coo, d_cols_coo, d_vals_coo);

        // --- Inner LM loop for adjusting lambda ---
        bool step_accepted = false;
        for (int lm_attempt = 0; lm_attempt < 10; ++lm_attempt) {
            // 5. Compute J^T*J and J^T
            int *d_hessian_rows, *d_hessian_cols, *d_AT_cscOffsets, *d_AT_columns;
            double *d_hessian_vals, *d_AT_values;
            int hessian_nnz;

            compute_AtA_debug(d_rows_coo, d_cols_coo, d_vals_coo, nnz, nCell, nCell,
                              &d_hessian_rows, &d_hessian_cols, &d_hessian_vals, &hessian_nnz,
                              &d_AT_cscOffsets, &d_AT_columns, &d_AT_values);

            // 6. Create scaled identity matrix lambda*I
            int *d_identity_row_ptr, *d_identity_cols_ptr;
            double *d_identity_vals_ptr;
            CUDA_CHECK(cudaMalloc(&d_identity_row_ptr, (nCell + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_identity_cols_ptr, nCell * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_identity_vals_ptr, nCell * sizeof(double)));
            create_identity_csr_and_scale(nCell, lambda, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr);

            // 7. Form LHS: A = J^T*J + lambda*I
            int *d_lhs_rows, *d_lhs_cols;
            double *d_lhs_vals;
            int lhs_nnz;
            add_csr_cusparse(1.0, nCell, nCell, hessian_nnz, d_hessian_rows, d_hessian_cols, d_hessian_vals,
                             nCell, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr,
                             &lhs_nnz, &d_lhs_rows, &d_lhs_cols, &d_lhs_vals);
            
            // 8. Form RHS: b = -J^T*r
            double* d_rhs = nullptr;
            scale_and_multiply_on_gpu(nCell, nCell, nnz, d_AT_cscOffsets, d_AT_columns, d_AT_values, d_r, -1.0, &d_rhs);


            // 9. Solve the linear system for the step delta
            double* d_delta = nullptr;
            solve_system_gpu(nCell, lhs_nnz, d_lhs_rows, d_lhs_cols, d_lhs_vals, d_rhs, &d_delta);
   
            // --- DIAGNOSTIC CHECK FOR SOLVER FAILURE ---
            if (!is_vector_finite(d_delta, nCell)) {
                std::cerr << "\nFATAL ERROR: The linear solver failed and produced non-finite values (NaN or Inf) in the delta vector.\n"
                          << "       This indicates the system matrix `(J^T*J + lambda*I)` is severely ill-conditioned.\n"
                          << "       Stopping iteration." << std::endl;
                
                // Free the corrupted delta vector before breaking
                CUDA_CHECK(cudaFree(d_delta));
                // Set flag to ensure outer loop knows we failed and stops
                step_accepted = false; 
                break; // Break from the inner lm_attempt loop
            }
            is_system_ill_defined(cublas_handle, d_r, d_rhs, d_delta, nCell);
            // print_gpu_array(d_delta, nCell);
            std::cout << "L2 norm of the step(delta): " << L2_norm_squared(cublas_handle, d_delta, nCell) << 
            " Square norm of step(delta): "<<square_norm(cublas_handle, d_delta, nCell)<<  
            " lambda: "<< lambda << std::endl <<std::endl;
            // --- END DIAGNOSTIC ---


            // 10. Propose new state: y_new = y + delta
            double* d_y_new;
            CUDA_CHECK(cudaMalloc(&d_y_new, nCell * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_y_new, d_y, nCell * sizeof(double), cudaMemcpyDeviceToDevice));
            const double alpha = 1.0;
            CUBLAS_CHECK(cublasDaxpy(cublas_handle, nCell, &alpha, d_delta, 1, d_y_new, 1));
            
            // 11. Evaluate cost of the new state
            std::vector<double> y_new_h = copy_gpu_array_host(d_y_new, nCell);
            std::vector<double> r_new_h(nCell);
            uv_velocity_single(r_new_h.data(), Re, y_new_h.data(), xN, yN, zN, u_inlet, dx, dy, dz);
            
            double* d_r_new;
            CUDA_CHECK(cudaMalloc(&d_r_new, nCell * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_r_new, r_new_h.data(), nCell * sizeof(double), cudaMemcpyHostToDevice));
            
            double new_cost = square_norm(cublas_handle, d_r_new, nCell);

            // 12. Accept or reject the step
            if (new_cost < cost) {
                std::cout << "  --> Step ACCEPTED. New cost: " << new_cost << " (improvement: " << cost - new_cost << ")" << std::endl;
                CUDA_CHECK(cudaMemcpy(d_y, d_y_new, nCell * sizeof(double), cudaMemcpyDeviceToDevice));
                y = y_new_h; // Update host vector as well
             
                lambda = std::max(lambda / lambda_factor, 1e-12);
                step_accepted = true;
            } else {
             
                std::cout << "  --> Step REJECTED. New cost: " << new_cost << " (no improvement)" << std::endl;
                lambda = std::min(lambda * lambda_factor, 1e12);
            }

            // Cleanup attempt-specific memory
            CUDA_CHECK(cudaFree(d_hessian_rows));
            CUDA_CHECK(cudaFree(d_hessian_cols));
            CUDA_CHECK(cudaFree(d_hessian_vals));
            CUDA_CHECK(cudaFree(d_AT_cscOffsets));
            CUDA_CHECK(cudaFree(d_AT_columns));
            CUDA_CHECK(cudaFree(d_AT_values));
            CUDA_CHECK(cudaFree(d_identity_row_ptr));
            CUDA_CHECK(cudaFree(d_identity_cols_ptr));
            CUDA_CHECK(cudaFree(d_identity_vals_ptr));
            CUDA_CHECK(cudaFree(d_lhs_rows));
            CUDA_CHECK(cudaFree(d_lhs_cols));
            CUDA_CHECK(cudaFree(d_lhs_vals));
            CUDA_CHECK(cudaFree(d_rhs));
            CUDA_CHECK(cudaFree(d_delta));
            CUDA_CHECK(cudaFree(d_y_new));
            CUDA_CHECK(cudaFree(d_r_new));

            if (step_accepted) {
                break; // Exit inner LM loop
            }
        } // End of LM attempts loop

        // Cleanup iteration-specific memory
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaFree(d_rows_coo));
        CUDA_CHECK(cudaFree(d_cols_coo));
        CUDA_CHECK(cudaFree(d_vals_coo));
        
        if (!step_accepted) {
            std::cout << "Warning: Failed to find a better step. Stopping." << std::endl;
            break;
        }

    } // End of main iteration loop

    // 13. Finalize
    // CUDA_CHECK(cudaMemcpy(y.data(), d_y, nCell * sizeof(double), cudaMemcpyDeviceToHost));
    
    // CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    std::cout << "Levenberg-Marquardt solver finished." << std::endl;
    return d_y;
}



int main()
{
    // Problem size
    int xN = 10, yN = 5, zN = 5;
    // int xN = 100, yN = 2, zN = 2;
    const int nCell = 4 * xN * yN * zN;

    int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    // Physicaleters
    double mu = 0.001;
    double L = 1.0;
    double M = 0.2;
    double N = 0.2;
    double rho = 1.0;
    double u0 = 1.0;

    double Re = (rho * (M / 2.0) * u0) / mu;

    std::cout << "Problem Setup:" << std::endl;
    std::cout << "  Grid: " << xN << " x " << yN << " x " << zN << std::endl;
    std::cout << "  Total cells: " << nCell << std::endl;
    std::cout << "  Reynolds number: " << Re << std::endl;

    // Generate coordinates
    std::vector<double> xcoor(sizeX * sizeY * sizeZ);
    std::vector<double> ycoor(sizeX * sizeY * sizeZ);
    std::vector<double> zcoor(sizeX * sizeY * sizeZ);

    //Will be use for plotting
    auto [dx, dy, dz] = coordinates(xcoor, ycoor, zcoor, xN, yN, zN, L, M, N);

    std::cout << "  dx = " << dx << ", dy = " << dy << ", dz = " << dz << std::endl;

    // Create inlet velocity profile
    int inletSize = sizeY * sizeZ;
    std::vector<double> u_inlet(inletSize);

    for (int j = 0; j < sizeY; ++j) {
        for (int k = 0; k < sizeZ; ++k) {
            double yv = M * j / (sizeY - 1);
            double zv = N * k / (sizeZ - 1);
            u_inlet[idx_3d(j, k, 0, sizeZ, 1)] = 
                16.0 * u0 * (yv / M) * (1.0 - yv / M) * (zv / N) * (1.0 - zv / N);
                //  u_inlet[idx_3d(j, k, 0, sizeZ, 1)] = 0;
        }
    }

    // Initial guess
    std::vector<double> y(nCell, 0.1);
    int blockSize = xN * yN * zN;
    for (int i = 0; i < blockSize; i++) {
        y[i] = u0;
    }
        
    

    
    // //------------------------------------------------------
    // //Run the method
    // //------------------------------------------------------
    // //
        double *d_solution=nullptr;

        d_solution=levenberg_marquardt_solver(Re, y, xN, yN, zN, u_inlet.data(), dx, dy, dz, 25,
         0.1, 10, 1e-10);
    //

    //------------------------------------------------------
    //Preprocessing for plots
    //------------------------------------------------------
    //
        auto [d_uvel, d_vvel, d_wvel, d_press]=boundary_conditions_final(d_solution, xN, yN, zN, u_inlet.data());
        int total_size = sizeX * sizeY * sizeZ;

        double *d_vel_mag=compute_vel_mag(total_size, d_uvel, d_vvel, d_wvel);
        // print_gpu_array(d_vel_mag,total_size);
        //Copy back to host
        // print_gpu_array(d_uvel, total_size);
        auto [h_uvel,h_vvel,h_wvel,h_pres,h_vel_mag] = std::make_tuple(copy_gpu_array_host(d_uvel,total_size),copy_gpu_array_host(d_vvel,total_size),
            copy_gpu_array_host(d_wvel,total_size),copy_gpu_array_host(d_press,total_size),copy_gpu_array_host(d_vel_mag,total_size));

        //Export them into nnz format for plotting in python with matplotlib
        cnpy::npz_save("simulation_results.npz", "u", h_uvel.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "w");
        cnpy::npz_save("simulation_results.npz", "v", h_vvel.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        cnpy::npz_save("simulation_results.npz", "w", h_wvel.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        cnpy::npz_save("simulation_results.npz", "p", h_pres.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        
        cnpy::npz_save("simulation_results.npz", "velmag", h_vel_mag.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        
        cnpy::npz_save("simulation_results.npz", "xcoor", xcoor.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        cnpy::npz_save("simulation_results.npz", "ycoor", ycoor.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        cnpy::npz_save("simulation_results.npz", "zcoor", zcoor.data(), {static_cast<size_t>(sizeX),static_cast<size_t>(sizeY),static_cast<size_t>(sizeZ)}, "a");
        
        cnpy::npz_save("simulation_results.npz", "xN", &xN, {1}, "a");
        cnpy::npz_save("simulation_results.npz", "yN", &yN, {1}, "a");
        cnpy::npz_save("simulation_results.npz", "zN", &zN, {1}, "a");

        cnpy::npz_save("simulation_results.npz", "L", &L, {1}, "a");
        cnpy::npz_save("simulation_results.npz", "M", &M, {1}, "a");
        cnpy::npz_save("simulation_results.npz", "N", &N, {1}, "a");

        //

    // Cleanup
                        
        // CUDA_CHECK(cudaFree(d_lhs_rows));
        // CUDA_CHECK(cudaFree(d_lhs_cols));
        // CUDA_CHECK(cudaFree(d_lhs_vals));
        // CUDA_CHECK(cudaFree(rhs));



        // CUDA_CHECK(cudaFree(d_rows_coo));
        // CUDA_CHECK(cudaFree(d_cols_coo));
        // CUDA_CHECK(cudaFree(d_vals_coo));
        
        CUDA_CHECK( cudaFree(d_solution));

        // CUBLAS_CHECK(cublasDestroy(cublas_handle));

        CUDA_CHECK( cudaFree(d_uvel));
        CUDA_CHECK( cudaFree(d_vvel));
        CUDA_CHECK( cudaFree(d_wvel));
        CUDA_CHECK( cudaFree(d_press));

        std::cout << "\nComputation complete!" << std::endl;

    return 0;
}
