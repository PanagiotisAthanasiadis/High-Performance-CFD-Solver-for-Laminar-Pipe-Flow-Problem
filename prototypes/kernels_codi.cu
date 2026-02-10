
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


// Standard 3D indexing: (i, j, k) -> linear index
__device__ __host__ inline int idx3(int i, int j, int k, int sizeY, int sizeZ) {
    return (i * sizeY + j) * sizeZ + k;
}

// Memory alignment utility
inline size_t align_size(size_t size, size_t alignment = 256) {
    return ((size + alignment - 1) / alignment) * alignment;
}


#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error " << err << " at "           \
                  << __FILE__ << ":" << __LINE__ << " -> "    \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while(0)


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
    int idx = idx3(i, j, k, sizeY, sizeZ);

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
        p[idx3(i, 0, k, sizeY, sizeZ)] = p[idx3(i, 1, k, sizeY, sizeZ)];
    } else if (j == yN + 1) {
        p[idx3(i, yN + 1, k, sizeY, sizeZ)] = p[idx3(i, yN, k, sizeY, sizeZ)];
    }

    // Z-direction boundaries
    if (k == 0) {
        p[idx3(i, j, 0, sizeY, sizeZ)] = p[idx3(i, j, 1, sizeY, sizeZ)];
    } else if (k == zN + 1) {
        p[idx3(i, j, zN + 1, sizeY, sizeZ)] = p[idx3(i, j, zN, sizeY, sizeZ)];
    }

    // X-direction boundaries
    if (i == 0) {
        u[idx3(0, j, k, sizeY, sizeZ)] = u_inlet[idx3(j, k, 0, sizeZ, 1)];
        p[idx3(0, j, k, sizeY, sizeZ)] = p[idx3(1, j, k, sizeY, sizeZ)];
    } else if (i == xN + 1) {
        u[idx3(xN + 1, j, k, sizeY, sizeZ)] = u[idx3(xN, j, k, sizeY, sizeZ)];
        v[idx3(xN + 1, j, k, sizeY, sizeZ)] = v[idx3(xN, j, k, sizeY, sizeZ)];
        w[idx3(xN + 1, j, k, sizeY, sizeZ)] = w[idx3(xN, j, k, sizeY, sizeZ)];
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
        return idx3(ii, jj, kk, sizeY, sizeZ);
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
