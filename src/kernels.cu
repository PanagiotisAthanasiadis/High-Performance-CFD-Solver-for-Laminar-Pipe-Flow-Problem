#include "common.cuh"
#include "kernels.cuh"

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
