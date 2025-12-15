#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>



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

// ============================================================================
// COORDINATE GENERATION
// ============================================================================

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
// KERNEL: GET TRIPLETS (SPARSE JACOBIAN)
// ============================================================================

__global__ void get_triplets(
    float* __restrict__ out,
    float3* __restrict__ triplets,
    const float* __restrict__ h,
    const float* __restrict__ fold,
    const int nCell,
    const int grain,
    const int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nCell || g >= grain) return;

    int idx = g * nCell + i;
    int batch_step = g + start;
    
    float df = (out[idx] - fold[i]) / h[batch_step];

    if (fabsf(df) > 1e-8f) {
        triplets[idx].x = i + 1;  // MATLAB indexing
        triplets[idx].y = batch_step + 1;
        triplets[idx].z = df;
    } else {
        triplets[idx].x = -2.0f;  // Mark invalid
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


std::pair<std::vector<float>, std::tuple<int*, int*, float*, int>>
Residuals_Sparse_Jacobian_Split(
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

    size_t max_nnz = (size_t)nCell * 20;
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

    CUDA_CHECK(cudaFree(dev_fold));
    CUDA_CHECK(cudaFree(hd));
    CUDA_CHECK(cudaFree(d_ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
    CUDA_CHECK(cudaFree(d_global_counter));

    return {fold, std::make_tuple(d_all_rows, d_all_cols, d_all_vals, nnz)};
}

// ============================================================================
// Convert COO to CSR on GPU 
// ============================================================================

void convert_coo_to_csr_gpu(
    int* d_row_coo, int* d_col_coo, float* d_vals_coo, int nnz, int nCell,
    int** d_row_ptr, int** d_col_idx, float** d_values)
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Allocate CSR arrays
    CUDA_CHECK(cudaMalloc(d_row_ptr, (nCell + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_values, nnz * sizeof(float)));
    
    // Convert COO to CSR
    cusparseXcoo2csr(handle, d_row_coo, nnz, nCell,
                     *d_row_ptr, CUSPARSE_INDEX_BASE_ZERO);
    
    CUDA_CHECK(cudaMemcpy(*d_col_idx, d_col_coo, nnz * sizeof(int), 
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(*d_values, d_vals_coo, nnz * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    cusparseDestroy(handle);
}

void solve_with_cusolver_sparse(
    int* d_row_ptr, int* d_col_idx, float* d_vals, int nnz,
    float* d_residuals, float* d_solution, int nCell)
{
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);
    
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    // Use cusolverSpScsrlsvqr for sparse solve
    float tol = 1e-6f;
    int singularity;
    
    cusolverSpScsrlsvqr(handle, nCell, nnz, descrA,
                       d_vals, d_row_ptr, d_col_idx,
                       d_residuals, tol, 0, d_solution, &singularity);
    
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}


// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main()
{
    // Problem size
    // int xN = 100, yN = 50, zN = 50;
    int xN = 50, yN = 5, zN = 5;
    const int nCell = 4 * xN * yN * zN;

    int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;

    // Physical parameters
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

    auto [fold, sparse_matrix] = Residuals_Sparse_Jacobian_Split(
        Re, y.data(), xN, yN, zN, u_inlet.data(), dx, dy, dz);
    
    auto [d_rows_coo, d_cols_coo, d_vals_coo, nnz] = sparse_matrix;
    
    std::cout << "Sparse matrix: " << nnz << " non-zeros" << std::endl;
    std::cout << "Sparsity: " << (100.0 * nnz / ((long long)nCell * nCell)) << "%" << std::endl;
    
    // Option 1: Copy to host 
    std::vector<int> rows(nnz), cols(nnz);
    std::vector<float> vals(nnz);
    cudaMemcpy(rows.data(), d_rows_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cols.data(), d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vals.data(), d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "--- Printing Sparse Data ---" << std::endl;
    for (int i = 0; i < nnz; ++i) {
      std::cout << "Index " << i << ": "
                << "(Row=" << rows[i] << ", "
                << "Col=" << cols[i] << ") "
                << "Val=" << vals[i] << std::endl;
    }

    //Unecessery there was always on the device
    float *devu_fold,*d_solution;
    std::vector<float> solution(nCell);
    CUDA_CHECK(cudaMalloc(&devu_fold, nCell * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devu_fold, fold.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_solution, nCell * sizeof(float)));
    
    
    // // Option 2: Convert to CSR 
    int *d_row_ptr, *d_col_idx;
    float *d_values;
    convert_coo_to_csr_gpu(d_rows_coo, d_cols_coo, d_vals_coo, nnz, nCell,
                          &d_row_ptr, &d_col_idx, &d_values);
    
    //Solver is linear ou need a wrapper
    solve_with_cusolver_sparse(d_row_ptr, d_col_idx, d_values, nnz,
                              devu_fold, d_solution, nCell);
    
    CUDA_CHECK(cudaMemcpy( solution.data(),d_solution, nCell * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto i : solution) {
        std::cout << i <<std::endl ;
    }


    // Cleanup
    cudaFree(d_rows_coo);
    cudaFree(d_cols_coo);
    cudaFree(d_vals_coo);
    cudaFree(devu_fold);
    cudaFree(d_solution);



    std::cout << "\nComputation complete!" << std::endl;

    return 0;
}