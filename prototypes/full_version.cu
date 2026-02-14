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


void print_csr_matrix(int rows, int nnz, int* d_row_ptr, int* d_cols, float* d_vals) {
    // Allocate Host Memory
    int* h_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    int* h_cols    = (int*)malloc(nnz * sizeof(int));
    float* h_vals  = (float*)malloc(nnz * sizeof(float));

    // Copy from Device to Host
    CUDA_CHECK(cudaMemcpy(h_row_ptr, d_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols, d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals, d_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost));

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
                          const float* d_vals,
                          int max_print = 20) 
{
    // 1. Allocate Host Memory
    std::vector<int> h_rows(nnz);
    std::vector<int> h_cols(nnz);
    std::vector<float> h_vals(nnz);

    // 2. Copy Data from Device to Host
    cudaMemcpy(h_rows.data(), d_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cols.data(), d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), d_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost);

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
    const float* __restrict__ y,
    const float* __restrict__ h,
    float* __restrict__ ysol_batch,
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
    const float* __restrict__ ysol_local_batch,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
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
    const float* __restrict__ u_inlet,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
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

        // Z-direction boundaries
        if (k == 0) {
            p[idx_3d_batch(i, j, 0, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, 1, l, sizeY, sizeZ, grain)];
        } else if (k == zN + 1) {
            p[idx_3d_batch(i, j, zN + 1, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, zN, l, sizeY, sizeZ, grain)];
        }

        // X-direction boundaries
        if (i == 0) {
            u[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 
                u_inlet[idx_3d(j, k, 0, sizeZ, 1)];
            p[idx_3d_batch(0, j, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(1, j, k, l, sizeY, sizeZ, grain)];
        } else if (i == xN + 1) {
            u[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                u[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
            v[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                v[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
            w[idx_3d_batch(xN + 1, j, k, l, sizeY, sizeZ, grain)] = 
                w[idx_3d_batch(xN, j, k, l, sizeY, sizeZ, grain)];
        }
    }
}



// ============================================================================
// KERNEL: BOUNDARY CONDITIONS INITIALIZATION (SINGLE)
// ============================================================================

__global__ void boundary_conditions_initialization_single(
    const float* __restrict__ ysol,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
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
    const float* __restrict__ u_inlet,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
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

    // Z-direction boundaries
    if (k == 0) {
        p[idx_3d(i, j, 0, sizeY, sizeZ)] = p[idx_3d(i, j, 1, sizeY, sizeZ)];
    } else if (k == zN + 1) {
        p[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = p[idx_3d(i, j, zN, sizeY, sizeZ)];
    }

    // X-direction boundaries
    if (i == 0) {
        u[idx_3d(0, j, k, sizeY, sizeZ)] = u_inlet[idx_3d(j, k, 0, sizeZ, 1)];
        p[idx_3d(0, j, k, sizeY, sizeZ)] = p[idx_3d(1, j, k, sizeY, sizeZ)];
    } else if (i == xN + 1) {
        u[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = u[idx_3d(xN, j, k, sizeY, sizeZ)];
        v[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = v[idx_3d(xN, j, k, sizeY, sizeZ)];
        w[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = w[idx_3d(xN, j, k, sizeY, sizeZ)];
    }
}

// ============================================================================
// KERNEL: UV VELOCITY (SINGLE)
// ============================================================================

__global__ void kernel_uv_velocity_single(
    float* __restrict__ out, float Re,
    float* __restrict__ u, float* __restrict__ v,
    float* __restrict__ p, float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz)
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
        float conv_x = 0.5f * dy * dz * (u[idx(i+1,j,k)]*u[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*u[idx(i-1,j,k)]);
        float conv_y = 0.5f * dx * dz * (u[idx(i,j+1,k)]*v[idx(i,j+1,k)] - 
                                         u[idx(i,j-1,k)]*v[idx(i,j-1,k)]);
        float conv_z = 0.5f * dx * dy * (u[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         u[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        float pres = (dy * dz) * (p[idx(i+1,j,k)] - p[idx(i,j,k)]);

        float diff = (1.0f/Re) * (
            (dy*dz/dx) * (u[idx(i+1,j,k)] - 2.0f*u[idx(i,j,k)] + u[idx(i-1,j,k)]) +
            (dx*dz/dy) * (u[idx(i,j+1,k)] - 2.0f*u[idx(i,j,k)] + u[idx(i,j-1,k)]) +
            (dx*dy/dz) * (u[idx(i,j,k+1)] - 2.0f*u[idx(i,j,k)] + u[idx(i,j,k-1)])
        );

        out[pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // V-momentum
    {
        float conv_x = 0.5f * dy * dz * (u[idx(i+1,j,k)]*v[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*v[idx(i-1,j,k)]);
        float conv_y = 0.5f * dx * dz * (v[idx(i,j+1,k)]*v[idx(i,j+1,k)] - 
                                         v[idx(i,j-1,k)]*v[idx(i,j-1,k)]);
        float conv_z = 0.5f * dx * dy * (v[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         v[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        float pres = (dx * dz) * (p[idx(i,j+1,k)] - p[idx(i,j,k)]);

        float diff = (1.0f/Re) * (
            (dy*dz/dx) * (v[idx(i+1,j,k)] - 2.0f*v[idx(i,j,k)] + v[idx(i-1,j,k)]) +
            (dx*dz/dy) * (v[idx(i,j+1,k)] - 2.0f*v[idx(i,j,k)] + v[idx(i,j-1,k)]) +
            (dx*dy/dz) * (v[idx(i,j,k+1)] - 2.0f*v[idx(i,j,k)] + v[idx(i,j,k-1)])
        );

        out[nCell + pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // W-momentum
    {
        float conv_x = 0.5f * dy * dz * (u[idx(i+1,j,k)]*w[idx(i+1,j,k)] - 
                                         u[idx(i-1,j,k)]*w[idx(i-1,j,k)]);
        float conv_y = 0.5f * dx * dz * (v[idx(i,j+1,k)]*w[idx(i,j+1,k)] - 
                                         v[idx(i,j-1,k)]*w[idx(i,j-1,k)]);
        float conv_z = 0.5f * dx * dy * (w[idx(i,j,k+1)]*w[idx(i,j,k+1)] - 
                                         w[idx(i,j,k-1)]*w[idx(i,j,k-1)]);
        float pres = (dx * dy) * (p[idx(i,j,k+1)] - p[idx(i,j,k)]);

        float diff = (1.0f/Re) * (
            (dy*dz/dx) * (w[idx(i+1,j,k)] - 2.0f*w[idx(i,j,k)] + w[idx(i-1,j,k)]) +
            (dx*dz/dy) * (w[idx(i,j+1,k)] - 2.0f*w[idx(i,j,k)] + w[idx(i,j-1,k)]) +
            (dx*dy/dz) * (w[idx(i,j,k+1)] - 2.0f*w[idx(i,j,k)] + w[idx(i,j,k-1)])
        );

        out[2*nCell + pos] = conv_x + conv_y + conv_z + pres - diff;
    }

    // Continuity
    {
        float cont = (dy*dz/2.0f) * (u[idx(i+1,j,k)] - u[idx(i-1,j,k)]) +
                    (dx*dz/2.0f) * (v[idx(i,j+1,k)] - v[idx(i,j-1,k)]) +
                    (dx*dy/2.0f) * (w[idx(i,j,k+1)] - w[idx(i,j,k-1)]);

        out[3*nCell + pos] = cont;
    }
}


// ============================================================================
// HOST FUNCTION: RESIDUALS AND SPARSE JACOBIAN
// ============================================================================
__global__ void build_jacobian_entries(
    const float* __restrict__ out,
    const float* __restrict__ fold,
    const float* __restrict__ h,
    int* __restrict__ row_idx,
    int* __restrict__ col_idx,
    float* __restrict__ values,
    int* __restrict__ counter,
    const int nCell, const int grain, const int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nCell || g >= grain) return;
    
    int idx = g * nCell + i;
    int batch_step = g + start;
    float df = (out[idx] - fold[i]) / h[batch_step];
    
    if (fabsf(df) > 1e-8f) {
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
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const float dy_dz = dy * dz;
    const float dx_dz = dx * dz;
    const float dx_dy = dx * dy;
    const float inv_Re = 1.0f / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        float u_c  = u[base_idx];
        float u_ip = u[base_idx + stride_i];
        float u_im = u[base_idx - stride_i];
        float u_jp = u[base_idx + stride_j];
        float u_jm = u[base_idx - stride_j];
        float u_kp = u[base_idx + stride_k];
        float u_km = u[base_idx - stride_k];

        float result = 0.5f * dy_dz * (u_ip * u_ip - u_im * u_im);
        result += 0.5f * dx_dz * (u_jp * v[base_idx + stride_j] - u_jm * v[base_idx - stride_j]);
        result += 0.5f * dx_dy * (u_kp * w[base_idx + stride_k] - u_km * w[base_idx - stride_k]);
        result += dy_dz * (p[base_idx + stride_i] - p[base_idx]);
        result -= inv_Re * ((dy_dz/dx) * (u_ip - 2.0f*u_c + u_im) +
                           (dx_dz/dy) * (u_jp - 2.0f*u_c + u_jm) +
                           (dx_dy/dz) * (u_kp - 2.0f*u_c + u_km));

        out[l * NC + pos] = result;
    }
}

// V-Momentum Kernel
__global__ void 
kernel_v_momentum(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const float dy_dz = dy * dz;
    const float dx_dz = dx * dz;
    const float dx_dy = dx * dy;
    const float inv_Re = 1.0f / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        float v_c  = v[base_idx];
        float v_ip = v[base_idx + stride_i];
        float v_im = v[base_idx - stride_i];
        float v_jp = v[base_idx + stride_j];
        float v_jm = v[base_idx - stride_j];
        float v_kp = v[base_idx + stride_k];
        float v_km = v[base_idx - stride_k];

        float result = 0.5f * dy_dz * (u[base_idx + stride_i] * v_ip - u[base_idx - stride_i] * v_im);
        result += 0.5f * dx_dz * (v_jp * v_jp - v_jm * v_jm);
        result += 0.5f * dx_dy * (v_kp * w[base_idx + stride_k] - v_km * w[base_idx - stride_k]);
        result += dx_dz * (p[base_idx + stride_j] - p[base_idx]);
        result -= inv_Re * ((dy_dz/dx) * (v_ip - 2.0f*v_c + v_im) +
                           (dx_dz/dy) * (v_jp - 2.0f*v_c + v_jm) +
                           (dx_dy/dz) * (v_kp - 2.0f*v_c + v_km));

        out[l * NC + nc + pos] = result;
    }
}

// W-Momentum Kernel
__global__ void
kernel_w_momentum(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > xN || j > yN || k > zN) return;

    const float dy_dz = dy * dz;
    const float dx_dz = dx * dz;
    const float dx_dy = dx * dy;
    const float inv_Re = 1.0f / Re;
    const int pos = (i-1) * (yN * zN) + (j-1) * zN + (k-1);
    const int nc = xN * yN * zN;
    const int NC = 4 * nc;

    for (int l = 0; l < grain; ++l) {
        const int base_idx = ((i * sizeY + j) * sizeZ + k) * grain + l;
        const int stride_i = sizeY * sizeZ * grain;
        const int stride_j = sizeZ * grain;
        const int stride_k = grain;

        float w_c  = w[base_idx];
        float w_ip = w[base_idx + stride_i];
        float w_im = w[base_idx - stride_i];
        float w_jp = w[base_idx + stride_j];
        float w_jm = w[base_idx - stride_j];
        float w_kp = w[base_idx + stride_k];
        float w_km = w[base_idx - stride_k];

        float result = 0.5f * dy_dz * (u[base_idx + stride_i] * w_ip - u[base_idx - stride_i] * w_im);
        result += 0.5f * dx_dz * (v[base_idx + stride_j] * w_jp - v[base_idx - stride_j] * w_jm);
        result += 0.5f * dx_dy * (w_kp * w_kp - w_km * w_km);
        result += dx_dy * (p[base_idx + stride_k] - p[base_idx]);
        result -= inv_Re * ((dy_dz/dx) * (w_ip - 2.0f*w_c + w_im) +
                           (dx_dz/dy) * (w_jp - 2.0f*w_c + w_jm) +
                           (dx_dy/dz) * (w_kp - 2.0f*w_c + w_km));

        out[l * NC + 2*nc + pos] = result;
    }
}

// Continuity Kernel
__global__ void 
kernel_continuity(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
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

        float result = 0.5f * dy * dz * (u[base_idx + stride_i] - u[base_idx - stride_i]);
        result += 0.5f * dx * dz * (v[base_idx + stride_j] - v[base_idx - stride_j]);
        result += 0.5f * dx * dy * (w[base_idx + stride_k] - w[base_idx - stride_k]);

        out[l * NC + 3*nc + pos] = result;
    }
}

// ============================================================================
// HOST FUNCTION: UV VELOCITY SINGLE
// ============================================================================

void uv_velocity_single(float *out, const float Re, float *y,
                       const int xN, const int yN, const int zN,
                       const float *u_inlet,
                       const float dx, const float dy, const float dz)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    // Allocate device memory with alignment
    float *dev_out, *ysol, *d_u_inlet;
    float *ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&dev_out, align_size(nCell * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&ysol, align_size(nCell * sizeof(float))));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(float))));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ysol, y, nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(float), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMemcpy(out, dev_out, nCell * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(dev_out));
    CUDA_CHECK(cudaFree(ud));
    CUDA_CHECK(cudaFree(vd));
    CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(pd));
    CUDA_CHECK(cudaFree(ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
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
 * - **first**: `float*` - Device pointer to the base residual vector \f$ F(y) \f$.
 * - **second**: `std::tuple<int*, int*, float*, int>` - The Jacobian matrix in Coordinate (COO) format:
 * - `int*`: Device pointer to row indices.
 * - `int*`: Device pointer to column indices.
 * - `float*`: Device pointer to non-zero values.
 * - `int`: The total count of non-zero elements (nnz).
 *
 * @note The returned pointers point to **device memory** (GPU). The caller is responsible 
 * for freeing these resources using `cudaFree`.
 * @warning This function performs significant device memory allocation. Ensure sufficient GPU memory is available.
 */
std::pair<float*, std::tuple<int*, int*, float*, int>>
Residuals_Sparse_Jacobian_finite_diff(
    const float Re, float *y,
    const int xN, const int yN, const int zN,
    const float *u_inlet,
    const float dx, const float dy, const float dz)
{
    const int grain = 1;
    const int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    const int totalSize = sizeX * sizeY * sizeZ;
    const int nCell = 4 * xN * yN * zN;
    const float EPS = 1e-4f;

    std::vector<float> h(nCell);
    for (int j = 0; j < nCell; ++j) {
        float temp = y[j];
        h[j] = EPS * std::abs(temp);
        if (std::abs(h[j]) < 1e-8f) h[j] = EPS;
        y[j] = temp + h[j];
        h[j] = y[j] - temp;
        y[j] = temp;
    }

    std::vector<float> fold(nCell);
    uv_velocity_single(fold.data(), Re, y, xN, yN, zN, u_inlet, dx, dy, dz);

    float *dev_fold, *hd, *d_ysol, *d_u_inlet;
    CUDA_CHECK(cudaMalloc(&dev_fold, align_size(nCell * sizeof(float))));
    CUDA_CHECK(cudaMemcpy(dev_fold, fold.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&hd, align_size(nCell * sizeof(float))));
    CUDA_CHECK(cudaMemcpy(hd, h.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ysol, align_size(nCell * sizeof(float))));
    CUDA_CHECK(cudaMemcpy(d_ysol, y, nCell * sizeof(float), cudaMemcpyHostToDevice));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(float))));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(float), cudaMemcpyHostToDevice));

    size_t max_nnz = (size_t)nCell * 30;
    int *d_all_rows, *d_all_cols;
    float *d_all_vals;
    int *d_global_counter;
    
    CUDA_CHECK(cudaMalloc(&d_all_rows, max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_cols, max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_vals, max_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_global_counter, 0, sizeof(int)));

    int chunk_size = std::min(grain, nCell);
    int num_chunks = (nCell + chunk_size - 1) / chunk_size;
    int n_threads = std::min(2 , omp_get_max_threads());
    int chunks_per_thread = (num_chunks + n_threads - 1) / n_threads;

    std::cout << "Building Jacobian (split kernels)..." << std::endl;

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        float *dev_out, *ud, *vd, *wd, *pd, *d_ysol_batch;
        int *d_t_batch;

        size_t totalSize_grain = (size_t)totalSize * grain;
        CUDA_CHECK(cudaMalloc(&dev_out, align_size((size_t)grain * nCell * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize_grain * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize_grain * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize_grain * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize_grain * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&d_ysol_batch, align_size((size_t)grain * nCell * sizeof(float))));
        CUDA_CHECK(cudaMalloc(&d_t_batch, align_size(grain * sizeof(int))));

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

            int nc = xN * yN * zN;
            int threads = 256; //Prefer 1d for better coalesced memory access
            int blocks = (nc + threads - 1) / threads;
            boundary_conditions_initialization<<<blocks, threads, 0, stream>>>(
            d_ysol_batch, ud, vd, wd, pd, xN, yN, zN, current_grain);

            dim3 blk(8, 8, 4);
            dim3 grd((sizeX + blk.x - 1) / blk.x,
                    (sizeY + blk.y - 1) / blk.y,
                    (sizeZ + blk.z - 1) / blk.z);
            boundary_conditions_apply<<<grd, blk, 0, stream>>>(
                d_u_inlet, ud, vd, wd, pd, xN, yN, zN, current_grain);

            dim3 t(8, 8, 4);
            dim3 bks((xN + t.x - 1) / t.x,
                    (yN + t.y - 1) / t.y,
                    (zN + t.z - 1) / t.z);

            kernel_u_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);

            kernel_v_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);

            kernel_w_momentum<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, pd, wd, xN, yN, zN,
                dx, dy, dz, Re, sizeY, sizeZ);

            kernel_continuity<<<bks, t, 0, stream>>>(
                current_grain, dev_out, ud, vd, wd, xN, yN, zN,
                dx, dy, dz, sizeY, sizeZ);
            // ================================================================

            dim3 tt(32, 4);
            dim3 tg((nCell + tt.x - 1) / tt.x,
                   (current_grain + tt.y - 1) / tt.y);
            build_jacobian_entries<<<tg, tt, 0, stream>>>(
                dev_out, dev_fold, hd,
                d_all_rows, d_all_cols, d_all_vals,
                d_global_counter,
                nCell, current_grain, start);
            
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

    
    // CUDA_CHECK(cudaFree(dev_fold));
    CUDA_CHECK(cudaFree(hd));
    CUDA_CHECK(cudaFree(d_ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
    CUDA_CHECK(cudaFree(d_global_counter));

    return {dev_fold, std::make_tuple(d_all_rows, d_all_cols, d_all_vals, nnz)};
}


std::tuple<float, float, float>
coordinates(std::vector<float> &xcoor, std::vector<float> &ycoor,
            std::vector<float> &zcoor, const int xN, const int yN,
            const int zN, const float L, const float M, const float N) {
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

    float dx = L / (xSize - 1);
    float dy = M / (ySize - 1);
    float dz = N / (zSize - 1);

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
//                              const float* in_vals, float* out_vals, 
//                              const int* sorted_indices, int nnz) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < nnz) {
//         int old_pos = sorted_indices[i];
//         out_cols[i] = in_cols[old_pos];
//         out_vals[i] = in_vals[old_pos];
//     }
// }


// void sort_coo_cub(int* d_rows, int* d_cols, float* d_vals, int nnz) {
    
//     // Pointers for sorted/temp arrays
//     int *d_indices, *d_indices_sorted;
//     int *d_rows_sorted, *d_cols_sorted; 
//     float *d_vals_sorted;
    
//     // 1. Allocate Temporary Buffers
//     //    We need an index array to track where the rows move, so we can move cols/vals later.
//     CUDA_CHECK(cudaMalloc((void**)&d_indices, nnz * sizeof(int)));
//     CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, nnz * sizeof(int)));
//     CUDA_CHECK(cudaMalloc((void**)&d_rows_sorted, nnz * sizeof(int)));
    
//     //    Allocation of output buffers for cols/vals
//     CUDA_CHECK(cudaMalloc((void**)&d_cols_sorted, nnz * sizeof(int))); 
//     CUDA_CHECK(cudaMalloc((void**)&d_vals_sorted, nnz * sizeof(float)));

//     // 2. Initialize Permutation Index [0, 1, 2, ..., NNZ-1]
//     int blockSize = 256;
//     int numBlocks = (nnz + blockSize - 1) / blockSize;
    
//     init_indices<<<numBlocks, blockSize>>>(d_indices, nnz);

//     // 3. Determine Temporary Storage Size for CUB Radix Sort
//     void *d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;

//     //    Query workspace requirement
//     //    Key: d_rows, Value: d_indices
//     CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//         d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));
    
//     //    Allocate workspace
//     CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

//     // 4. Run the Sort
//     //    Sorts 'rows' into 'rows_sorted', and moves 'indices' into 'indices_sorted'
//     CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//         d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));

//     // 5. Permute Cols and Vals using the new index order
//     //    We effectively "apply" the sort permutation to the other two arrays.
//     permute_data<<<numBlocks, blockSize>>>(d_cols, d_cols_sorted, 
//                                            d_vals, d_vals_sorted, 
//                                            d_indices_sorted, nnz);

//     // 6. Copy sorted data back to original pointers
//     //    (Since the function arguments are pointers-by-value, we must copy the data back 
//     //     to the original locations unless you change the function signature to accept int**).
//     CUDA_CHECK(cudaMemcpy(d_rows, d_rows_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(d_cols, d_cols_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(d_vals, d_vals_sorted, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

//     // Cleanup
//     CUDA_CHECK(cudaFree(d_indices)); 
//     CUDA_CHECK(cudaFree(d_indices_sorted)); 
//     CUDA_CHECK(cudaFree(d_rows_sorted)); 
//     CUDA_CHECK(cudaFree(d_cols_sorted)); 
//     CUDA_CHECK(cudaFree(d_vals_sorted)); 
//     CUDA_CHECK(cudaFree(d_temp_storage));
// }



// -------------------------------------------------------------------------
// Kernel: Generate Selection Flags
// -------------------------------------------------------------------------
__global__ void generate_flags_kernel(int nnz, const float* values, char* flags, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        // 1 = Keep, 0 = Discard
        flags[idx] = (fabsf(values[idx]) > threshold) ? 1 : 0;
    }
}

// -------------------------------------------------------------------------
// Main Filter Function (CSR Format using CUB)
// -------------------------------------------------------------------------
void filter_csr_cub(float threshold,int rows, int old_nnz,
                    int* d_row_offsets, int* d_cols, float* d_vals,
                    int** d_new_row_offsets, int** d_new_cols, float** d_new_vals,
                    int* new_nnz_out) {
  
    // ---------------------------------------------------------------------
    // 1. Setup & Generate Flags
    // ---------------------------------------------------------------------
    char* d_flags;
    CUDA_CHECK(cudaMalloc((void**)&d_flags, old_nnz * sizeof(char)));

    int blockSize = 256;
    int gridSize = (old_nnz + blockSize - 1) / blockSize;
    
    generate_flags_kernel<<<gridSize, blockSize>>>(old_nnz, d_vals, d_flags, threshold);

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

    printf("Filtering Result: %d -> %d elements threshold:%f \n", old_nnz, final_nnz,threshold);

    // ---------------------------------------------------------------------
    // 4. Compact Data Arrays (DeviceSelect)
    // ---------------------------------------------------------------------
    
    // Allocate Result Arrays
    CUDA_CHECK(cudaMalloc((void**)d_new_cols, final_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_new_vals, final_nnz * sizeof(float)));
    
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
    const float* d_val      // DEVICE Pointer: Values
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
    std::vector<float> h_val(total_nnz);

    CUDA_CHECK(cudaMemcpy(h_row_ind.data(), d_row_ind, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(), d_val, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

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
            float val = h_val[i];

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
    float* d_vals_coo,
    int nnz,
    int num_rows,
    int num_cols,
    int** d_result_rows,
    int** d_result_cols,
    float** d_result_vals,
    int* result_nnz,
    int** d_AT_cscOffsets, 
    int** d_AT_columns,
    float** d_AT_values)
{
    // -------------------------------------------------------------------------
    // 0. Setup Input Data (Convert COO matrix to CSR)
    // -------------------------------------------------------------------------
    cusparseHandle_t handle; //Necessary for cusparse
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    // Convert COO to CSR for A
    int *d_csrRowPtrA, *d_csrColIndA;
    float *d_csrValA;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA, nnz * sizeof(float)));

    CUSPARSE_CHECK(cusparseXcoo2csr(handle, d_rows_coo, nnz, num_rows, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    
    //Copy the values 
    CUDA_CHECK(cudaMemcpy(d_csrColIndA, d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrValA, d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    std::cout << "Converted A to CSR" << std::endl;

    //-------------------------------------------------------------------------
    // 1. Explicit Transpose: Generate A^T
    //    We use cusparseCsr2cscEx2. The output CSC arrays of A 
    //    are exactly the CSR arrays of A^T. (see note at page 20 cusparse 13.1 manual)
    // -------------------------------------------------------------------------

    // Allocate memory for A^T (Same NNZ, but dimensions swapped if non-square)

    // Note: A^T has n rows and m cols
    CUDA_CHECK(cudaMalloc((void**)d_AT_cscOffsets, (num_cols + 1) * sizeof(int))); 

    CUDA_CHECK(cudaMalloc((void**)d_AT_columns, nnz * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)d_AT_values, nnz * sizeof(float)));

    size_t transposeBufferSize = 0;
    void* d_transposeBuffer = NULL;

    // Query buffer size for CSR -> CSC (which is effectively A -> A^T)
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns, // Output arrays
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        &transposeBufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_transposeBuffer, transposeBufferSize));

    // Perform the transpose
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        d_transposeBuffer
    ));
    
    // -------------------------------------------------------------------------
    // 2. Initialize cuSPARSE and Matrix Descriptors
    // -------------------------------------------------------------------------

    //CUDA_R_32F is float 
    //32-bit indices CUSPARSE_INDEX_32I is supported (float) or 64-bit indices (CUSPARSE_INDEX_64I)
    cusparseSpMatDescr_t matA, matAt, matC;
   
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                      d_csrRowPtrA, d_cols_coo, d_vals_coo,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CUSPARSE_CHECK(cusparseCreateCsr(&matAt, num_cols, num_rows, nnz,
                                     *d_AT_cscOffsets, *d_AT_columns, *d_AT_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateCsr(&matC, num_cols, num_cols, 0,
                      nullptr, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    std::cout << "Created matrix descriptors (A^T: " << num_cols << "x" << num_rows 
              << ", A: " << num_rows << "x" << num_cols << ")" << std::endl;
    
    // -------------------------------------------------------------------------
    // 3. SpGEMM Setup
    // -------------------------------------------------------------------------
    float alpha = 1.0f, beta = 0.0f;
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
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, nullptr));
    
    std::cout << "Buffer size 1: " << bufferSize1 << " bytes" << std::endl;
    
    if (bufferSize1 > 0) {
        CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA,opB,
                                   &alpha, matAt, matA, &beta, matC,
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, dBuffer1));
    
    std::cout << "Work estimation complete" << std::endl;
    
    // -------------------------------------------------------------------------
    // 5. Compute Structure 
    // -------------------------------------------------------------------------
    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, nullptr));
    
    std::cout << "Buffer size 2: " << bufferSize2 << " bytes" << std::endl;
    
    if (bufferSize2 > 0) {
        CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
    }
    
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, dBuffer2));
    
    std::cout << "Compute complete" << std::endl;
    
    // -------------------------------------------------------------------------
    // 6. Allocate C and Copy Results
    // -------------------------------------------------------------------------
    // Get result size
    int64_t C_num_rows, C_num_cols, C_nnz; //They need int64_t
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
    
    
    // Allocate result CSR
    int *d_csrRowPtrC, *d_csrColIndC;
    float *d_csrValC;
    
    CUDA_CHECK(cudaMalloc(&d_csrRowPtrC, (C_num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndC, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValC, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));
    
    // Copy result
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, opA,opB,
                       &alpha, matAt, matA, &beta, matC,
                       CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    
    std::cout << "Result copied" << std::endl;

    //Define pointers for the trimed version
    int *d_trim_row_ptr, *d_trim_cols_ptr;
    float *d_trim_vals_ptr;
    int trim_nnz = 0;
    
    //Filter threshold
    float threshold = 1e-7f;

    
    //Trim for zeroes
    filter_csr_cub(threshold,C_num_rows, C_nnz, d_csrRowPtrC, d_csrColIndC, d_csrValC, &d_trim_row_ptr, &d_trim_cols_ptr, &d_trim_vals_ptr,&trim_nnz);


    // print_csr_matrix(num_cols, trim_nnz, d_trim_row_ptr, d_trim_cols_ptr, d_trim_vals_ptr);

    *d_result_rows= d_trim_row_ptr;
    *d_result_cols = d_trim_cols_ptr; 
    *d_result_vals = d_trim_vals_ptr;
    *result_nnz = trim_nnz;
    
    // Convert result to COO
    // CUDA_CHECK(cudaMalloc(d_result_rows, trim_nnz * sizeof(int)));
    
    // CUSPARSE_CHECK(cusparseXcsr2coo(handle, d_trim_row_ptr, trim_nnz, C_num_rows,
                    //  *d_result_rows, CUSPARSE_INDEX_BASE_ZERO));
    
    
    // //Print first few results
    // int* h_result_rows = new int[(int)trim_nnz];
    // int* h_result_cols = new int[(int)trim_nnz];
    // float* h_result_vals = new float[(int)trim_nnz];;
    
    // CUDA_CHECK(cudaMemcpy(h_result_rows, *d_result_rows, trim_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(h_result_cols, *d_result_cols, trim_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(h_result_vals, *d_result_vals,trim_nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    // std::cout << "Results:" << std::endl;
    // for (int i = 0; i < (int)trim_nnz; i++) {
    //     std::cout << "  (" << h_result_rows[i] << ", " << h_result_cols[i] << ") = " << h_result_vals[i] << std::endl;
    // }

    // -------------------------------------------------------------------------
    // 7. CLEANUP 
    // -------------------------------------------------------------------------
    
    // delete[] h_result_rows;
    // delete[] h_result_cols;
    // delete[] h_result_vals;
    
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

    // 5. Free Filtered/Trimmed Intermediate Arrays
    // cudaFree(d_trim_row_ptr);


    
    // 6. Destroy Descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matC);
    cusparseDestroySpMat(matAt);    

    // 7. Destroy Handle
    cusparseDestroy(handle);
}

__global__ void identity_csr_and_scale_kernel(int N, float alpha, 
                                    int* row_offsets, int* cols, float* vals) 
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
 * - `d_vals`: Must be size \f$ N \times \text{sizeof(float)} \f$.
 *
 * @param[in]  N             The dimension of the square matrix (number of rows/columns).
 * @param[in]  alpha         The scalar value to place on the diagonal (scaling factor).
 * @param[out] d_row_offsets Device pointer to the array that will hold the CSR row offsets.
 * @param[out] d_cols        Device pointer to the array that will hold the column indices.
 * @param[out] d_vals        Device pointer to the array that will hold the non-zero values.
 */
void create_identity_csr_and_scale(int N, float alpha, 
                                        int* d_row_offsets, int* d_cols, float* d_vals) 
{
    int blockSize = 256;
    int gridSize = ( (N + 1) + blockSize - 1) / blockSize;
    // We launch N+1 threads to handle the extra row pointer element safely
    identity_csr_and_scale_kernel<<<gridSize, blockSize>>>(N, alpha, d_row_offsets, d_cols, d_vals);
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
    float delta,int m, int n,                          // Matrix dimensions (rows, cols)
    int nnzA,                              // Non-zeros in A
    const int* d_A_row_offsets, const int* d_A_columns, const float* d_A_values,
    int nnzB,                              // Non-zeros in B
    const int* d_B_row_offsets, const int* d_B_columns, const float* d_B_values,
    int* nnzC_out,                         // Output: Non-zeros in C
    int** d_C_row_offsets, int** d_C_columns, float** d_C_values // Output: Pointers to C data
) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Scalar multipliers (alpha = 1.0, beta = 1.0 for simple A+B)
    const float alpha = delta;
    const float beta = delta;

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
    
    CUSPARSE_CHECK(cusparseScsrgeam2_bufferSizeExt(
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
    CUDA_CHECK(cudaMalloc((void**)d_C_values, nnzC * sizeof(float)));

    // 6. Numeric Phase: Fill C_columns and C_values
    // Performs the actual addition.
    CUSPARSE_CHECK(cusparseScsrgeam2(
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
 * Size: \f$ \text{cols} \times \text{sizeof(float)} \f$.
 * @param[in]  alpha         Scalar scaling factor \f$ \alpha \f$.
 * * @param[out] d_y_out       Address of a pointer. The function will allocate memory at this location
 * and store the result vector y.
 * Size: \f$ \text{rows} \times \text{sizeof(float)} \f$.
 */
void scale_and_multiply_on_gpu(
    int rows, int cols, int nnz,
    const int* d_col_offsets,  // Input: Device Pointer
    const int* d_row_indices,  // Input: Device Pointer
    const float* d_values,     // Input: Device Pointer
    const float* d_x,          // Input: Device Pointer
    float alpha,               // Scalar
    float** d_y_out            // Output: Address of pointer to allocate
) {
    // 1. Allocate Result Memory on GPU
    // We dereference d_y_out to set the caller's pointer
    CUDA_CHECK( cudaMalloc((void**)d_y_out, rows * sizeof(float)) );
    
    // Initialize result to 0 (optional but good practice for safety)
    CUDA_CHECK( cudaMemset(*d_y_out, 0, rows * sizeof(float)) );

    // 2. Create cuSPARSE Context
    cusparseHandle_t handle;
    CUSPARSE_CHECK( cusparseCreate(&handle) );

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // 3. Create Descriptors using the DEVICE pointers passed in
    CUSPARSE_CHECK( cusparseCreateCsc(&matA, rows, cols, nnz,
                                      (void*)d_col_offsets, (void*)d_row_indices, (void*)d_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

    CUSPARSE_CHECK( cusparseCreateDnVec(&vecX, cols, (void*)d_x, CUDA_R_32F) );
    
    // Use the newly allocated pointer (*d_y_out) for the Y descriptor
    CUSPARSE_CHECK( cusparseCreateDnVec(&vecY, rows, *d_y_out, CUDA_R_32F) );

    // 4. Allocate Workspace
    void* d_buffer = nullptr;
    size_t bufferSize = 0;
    float beta = 0.0f; // We are overwriting Y, not accumulating

    // int last_offset;
    // cudaMemcpy(&last_offset, &d_col_offsets[cols], sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Last offset: %d, NNZ: %d\n", last_offset, nnz);

    CUSPARSE_CHECK( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) );

    CUDA_CHECK( cudaMalloc(&d_buffer, bufferSize) );

    // 5. Execute Operation (GPU only)
    CUSPARSE_CHECK( cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer) );

    // 6. Cleanup Local Resources
    // Note: We DO NOT free d_col_offsets, d_x, or *d_y_out because the caller owns them.
    CUSPARSE_CHECK( cusparseDestroyDnVec(vecY) );
    CUSPARSE_CHECK( cusparseDestroyDnVec(vecX) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) );
    CUDA_CHECK( cudaFree(d_buffer) );
    CUSPARSE_CHECK( cusparseDestroy(handle) );
}

void print_gpu_array(const float* d_array, int n) {
    // Allocate host memory to hold the copy
    std::vector<float> h_array(n);

    // Copy from Device to Host
    cudaError_t status = cudaMemcpy(h_array.data(), d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

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
 * - **Value Type:** 32-bit Floats (`float`).
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
 * Size: \f$ n \times \text{sizeof(float)} \f$.
 * @param[out] d_x_out       Address of a pointer. The function allocates memory at this location
 * and stores the solution vector \f$ x \f$.
 * Size: \f$ n \times \text{sizeof(float)} \f$.
 *
 * @warning This function assumes the matrix indices are **32-bit integers** (passed as `CUDA_R_32I`),
 * even though the function parameters `n` and `nnz` are `int64_t`. Ensure your device arrays strictly contain 32-bit integers to avoid type mismatches.
 */
void solve_system_gpu(
    int64_t n,                  
    int64_t nnz,                
    const int* d_row_offsets,   
    const int* d_col_indices,   
    const float* d_values,      
    const float* d_b,           
    float** d_x_out             
) {
    // check_indices_sanity(nnz, n, d_row_offsets);
    // check_indices_sanity(nnz, n, d_col_indices); //Error 

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
    CUDA_CHECK( cudaMalloc((void**)d_x_out, n * sizeof(float)) );
    CUDA_CHECK( cudaMemset(*d_x_out, 0, n * sizeof(float)) );

    // 2. Initialize cuDSS Handles
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t solverData;
    cudssMatrix_t matA, vecB, vecX;

    CHECK_CUDSS( cudssCreate(&handle), "cudssCreate" )
    CHECK_CUDSS( cudssConfigCreate(&config), "cudssConfigCreate" )
    CHECK_CUDSS( cudssDataCreate(handle, &solverData), "cudssDataCreate" )

    // 3. Create Matrix Wrappers 
    CHECK_CUDSS( cudssMatrixCreateCsr(&matA, 
                                      n, n, nnz, 
                                      (void*)d_row_offsets, 
                                      NULL,                 
                                      (void*)d_col_indices, 
                                      (void*)d_values, 
                                      CUDA_R_32I,           // Index Type
                                      CUDA_R_32F,           // Value Type (Float)
                                      CUDSS_MTYPE_GENERAL, 
                                      CUDSS_MVIEW_FULL, 
                                      CUDSS_BASE_ZERO), "cudssMatrixCreateCsr" )

    // Vector X (Dense Float)
    CHECK_CUDSS( cudssMatrixCreateDn(&vecX, n, 1, n, 
                                     (void*)*d_x_out, 
                                     CUDA_R_32F,            // Value Type (Float)
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(X)" )

    // Vector B (Dense Float)
    CHECK_CUDSS( cudssMatrixCreateDn(&vecB, n, 1, n, 
                                     (void*)d_b, 
                                     CUDA_R_32F,            // Value Type (Float)
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
    float* d_vals   // Device Pointer: Values (Modified in-place)
) {
    // 1. Create Local Handle
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    
    int* d_permutation = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_permutation, nnz * sizeof(int)));
    
    float* d_values_sorted=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_values_sorted, nnz * sizeof(float)));

    // 3. Create Permutation Vector 
    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, nnz, nnz,
                                    d_permutation, d_values_sorted,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

     // Create dense vector for wrapping the original coo values
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, nnz, d_vals,
                                        CUDA_R_32F) )

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
    
    CUDA_CHECK(cudaMemcpy(d_vals, d_values_sorted, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

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
                             const float* in_vals, float* out_vals, 
                             const int* sorted_indices, int nnz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < nnz) {
        int old_pos = sorted_indices[i];
        out_cols[i] = in_cols[old_pos];
        out_vals[i] = in_vals[old_pos];
    }
}


void sort_coo_cub(int* d_rows, int* d_cols, float* d_vals, int nnz) {
    
    // Pointers for sorted/temp arrays
    int *d_indices, *d_indices_sorted;
    int *d_rows_sorted, *d_cols_sorted; 
    float *d_vals_sorted;
    
    // 1. Allocate Temporary Buffers
    //    We need an index array to track where the rows move, so we can move cols/vals later.
    CUDA_CHECK(cudaMalloc((void**)&d_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_rows_sorted, nnz * sizeof(int)));
    
    //    Allocation of output buffers for cols/vals
    CUDA_CHECK(cudaMalloc((void**)&d_cols_sorted, nnz * sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_vals_sorted, nnz * sizeof(float)));

    // 2. Initialize Permutation Index [0, 1, 2, ..., NNZ-1]
    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;
    
    init_indices<<<numBlocks, blockSize>>>(d_indices, nnz);

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

    // 6. Copy sorted data back to original pointers
    //    (Since the function arguments are pointers-by-value, we must copy the data back 
    //     to the original locations unless you change the function signature to accept int**).
    CUDA_CHECK(cudaMemcpy(d_rows, d_rows_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, d_cols_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, d_vals_sorted, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    // Cleanup
    CUDA_CHECK(cudaFree(d_indices)); 
    CUDA_CHECK(cudaFree(d_indices_sorted)); 
    CUDA_CHECK(cudaFree(d_rows_sorted)); 
    CUDA_CHECK(cudaFree(d_cols_sorted)); 
    CUDA_CHECK(cudaFree(d_vals_sorted)); 
    CUDA_CHECK(cudaFree(d_temp_storage));
}


int main()
{
    // Problem size
    int xN = 100, yN = 50, zN = 50;
    // int xN = 100, yN = 2, zN = 2;
    const int nCell = 4 * xN * yN * zN;

    int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    // Physicaleters
    float mu = 0.001f;
    float L = 1.0f;
    float M = 0.2f;
    float N = 0.2f;
    float rho = 1.0f;
    float u0 = 1.0f;

    float Re = (rho * (M / 2.0f) * u0) / mu;

    std::cout << "Problem Setup:" << std::endl;
    std::cout << "  Grid: " << xN << " x " << yN << " x " << zN << std::endl;
    std::cout << "  Total cells: " << nCell << std::endl;
    std::cout << "  Reynolds number: " << Re << std::endl;

    // Generate coordinates
    std::vector<float> xcoor(sizeX * sizeY * sizeZ);
    std::vector<float> ycoor(sizeX * sizeY * sizeZ);
    std::vector<float> zcoor(sizeX * sizeY * sizeZ);

    //Will be use for plotting
    auto [dx, dy, dz] = coordinates(xcoor, ycoor, zcoor, xN, yN, zN, L, M, N);

    std::cout << "  dx = " << dx << ", dy = " << dy << ", dz = " << dz << std::endl;

    // Create inlet velocity profile
    int inletSize = sizeY * sizeZ;
    std::vector<float> u_inlet(inletSize);

    for (int j = 0; j < sizeY; ++j) {
        for (int k = 0; k < sizeZ; ++k) {
            float yv = M * j / (sizeY - 1);
            float zv = N * k / (sizeZ - 1);
            u_inlet[idx_3d(j, k, 0, sizeZ, 1)] = 
                16.0f * u0 * (yv / M) * (1.0f - yv / M) * (zv / N) * (1.0f - zv / N);
        }
    }

    // Initial guess
    std::vector<float> y(nCell, 0.1f);
    int blockSize = xN * yN * zN;
    for (int i = 0; i < blockSize; i++) {
        y[i] = u0;
    }

    auto [fold, sparse_matrix] = Residuals_Sparse_Jacobian_finite_diff(
        Re, y.data(), xN, yN, zN, u_inlet.data(), dx, dy, dz);
    
    auto [d_rows_coo, d_cols_coo, d_vals_coo, nnz] = sparse_matrix;

    
    std::cout << "Sparse matrix: " << nnz << " non-zeros" << std::endl;
    std::cout << "Sparsity: " << (100.0 * nnz / ((long long)nCell * nCell)) << "%" << std::endl;
    
    
    
    // //-------------------------------------------------------
    // // Option 1: Copy to host 
    // //-------------------------------------------------------
    //     std::vector<int> rows(nnz), cols(nnz);
    //     std::vector<float> vals(nnz);
    //     cudaMemcpy(rows.data(), d_rows_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(cols.data(), d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(vals.data(), d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    //     std::cout << "--- Printing Sparse Data ---" << std::endl;
    //     for (int i = 0; i < nnz; ++i) {
    //       std::cout << "Index " << i << ": "
    //                 << "(Row=" << rows[i] << ", "
    //                 << "Col=" << cols[i] << ") "
    //                 << "Val=" << vals[i] << std::endl;
    //             }
    
    //-------------------------------------------------------
    //Sorting for csr 
    //-------------------------------------------------------
    // sort_coo_cub(d_rows_coo, d_cols_coo, d_vals_coo, nnz);
     sort_coo_matrix_cusparse(nCell, nCell, nnz, d_rows_coo, d_cols_coo, d_vals_coo);
    // print_coo_matrix_gpu(nCell, nCell, nnz, d_rows_coo, d_cols_coo, d_vals_coo,-1); //Debug
    
    
    //-------------------------------------------------------
    //J^T * J
    //-------------------------------------------------------
        // Result pointers
        int *d_hessian_rows, *d_hessian_cols;
        float *d_hessian_vals;
        int hessian_nnz;
        //Transpose jacobian pointers into CSC form
        int *d_AT_cscOffsets, *d_AT_columns;
        float *d_AT_values;

        compute_AtA_debug(d_rows_coo, d_cols_coo, d_vals_coo, nnz, 
                        nCell, nCell,
                        &d_hessian_rows, &d_hessian_cols, &d_hessian_vals,
                        &hessian_nnz,&d_AT_cscOffsets, 
                        &d_AT_columns,&d_AT_values);
        //It is ok
        // std::cout << "Printing J^T * J"<<std::endl;        
        //print_csr_matrix(nCell, hessian_nnz, d_hessian_rows, d_hessian_cols, d_hessian_vals);
        // It is ok                      
        // std::cout << "Printing J^T"<<std::endl;                
        // print_csc_matrix(nCell, nCell, d_AT_cscOffsets, d_AT_columns, d_AT_values);

    //-------------------------------------------------------
    // λ * I
    //-------------------------------------------------------
        int *d_identity_row_ptr, *d_identity_cols_ptr;
        float *d_identity_vals_ptr;
        
        // Allocate Device Memory for identity matrix (CSR form)
        CUDA_CHECK(cudaMalloc(&d_identity_row_ptr, (nCell + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_identity_cols_ptr, nCell * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_identity_vals_ptr, nCell * sizeof(float)));

        float lambda_scalar=10e-3;


        create_identity_csr_and_scale(nCell, lambda_scalar, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr);
        
        // std::cout << "Scaled identity matrix"<<std::endl;
        // print_csr_matrix(nCell, nCell, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr);
        
    //-------------------------------------------------------
    //-J^T*r(y)(right hand side )
    //-------------------------------------------------------
        // std::cout << "\n residual"<<std::endl;
        //   print_gpu_array(fold, nCell); //Debug
        
        float *rhs=nullptr;
        scale_and_multiply_on_gpu(nCell, nCell, nnz, d_AT_cscOffsets, 
            d_AT_columns, d_AT_values, fold, -1, &rhs);
        
        CUDA_CHECK(cudaFree(fold));
        CUDA_CHECK(cudaFree(d_AT_cscOffsets));
        CUDA_CHECK(cudaFree(d_AT_columns));
        CUDA_CHECK(cudaFree(d_AT_values));
        // std::cout << "\n -J^T*r(y)"<<std::endl;
        //  print_gpu_array(rhs, nCell);   //Debug
        
        //-------------------------------------------------------
        //Add (J^T * J + λ * I)*delta
        //-------------------------------------------------------
            // Outputs 
            int *d_lhs_rows = nullptr;
            int *d_lhs_cols = nullptr;
            float *d_lhs_vals = nullptr;
            int nnzlhs = 0;
            
            float delta=1; //Iteration step
            
             int h_last_A, h_last_B;
            cudaMemcpy(&h_last_A, d_hessian_rows + nCell, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_last_B, d_identity_row_ptr + nCell, sizeof(int), cudaMemcpyDeviceToHost);

            // printf("--- CSR VALIDATION ---\n");
            // printf("A: m=%d, nnz=%d, LastOffset=%d\n", nCell, hessian_nnz, h_last_A);
            // printf("B: m=%d, nnz=%d, LastOffset=%d\n", nCell, nCell, h_last_B);

            // if (h_last_B != nCell) {
            //     printf("ERROR: B is not a valid Identity Row Offset array!\n");
            // }

            add_csr_cusparse(delta,nCell, nCell, hessian_nnz, d_hessian_rows, d_hessian_cols, 
                d_hessian_vals, nCell, d_identity_row_ptr,
                d_identity_cols_ptr, d_identity_vals_ptr, 
                &nnzlhs, &d_lhs_rows,&d_lhs_cols, &d_lhs_vals);
            
           
            
            //Free whatever we dont need anymore
            CUDA_CHECK(cudaFree(d_hessian_rows));
            CUDA_CHECK(cudaFree(d_hessian_cols));
            CUDA_CHECK(cudaFree(d_hessian_vals));
        

            
            CUDA_CHECK(cudaFree(d_identity_row_ptr));
            CUDA_CHECK(cudaFree(d_identity_cols_ptr));
            CUDA_CHECK(cudaFree(d_identity_vals_ptr));

            // std::cout << "(J^T * J + λ * I)*delta "<<std::endl;
            // print_csr_matrix(nCell, nnzlhs, d_lhs_rows, d_lhs_cols, d_lhs_vals);
            // check_indices_sanity(nnzlhs,nCell,d_lhs_rows);
            // check_indices_sanity(nnzlhs,nCell,d_lhs_cols);
            // std::cout << "(J^T * J + λ * I)*delta in CSC format"<<std::endl;
            // print_csc_matrix(nCell, nCell, d_lhs_cols, d_lhs_rows, d_lhs_vals);
            
            
            
            //------------------------------------------------------
            //Solve the system 
            //------------------------------------------------------
            //
                float *d_solution=nullptr;
                printf("DEBUG: Pointer passed to print: %p\n", (void*)d_lhs_cols);
                solve_system_gpu(nCell, nnzlhs, d_lhs_rows, d_lhs_cols, 
                d_lhs_vals, rhs,&d_solution );
            
                // print_gpu_array(d_solution, nCell);
            //

            // Cleanup
                        
    CUDA_CHECK(cudaFree(d_lhs_rows));
    CUDA_CHECK(cudaFree(d_lhs_cols));
    CUDA_CHECK(cudaFree(d_lhs_vals));
    CUDA_CHECK(cudaFree(rhs));



    CUDA_CHECK(cudaFree(d_rows_coo));
    CUDA_CHECK(cudaFree(d_cols_coo));
    CUDA_CHECK(cudaFree(d_vals_coo));
    CUDA_CHECK( cudaFree(d_solution));
    
    std::cout << "\nComputation complete!" << std::endl;

    return 0;
}