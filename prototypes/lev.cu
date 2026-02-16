#include "lev.h"

// ============================================================================ 
// ERROR CHECKING AND UTILITIES
// ============================================================================ 

#define CUDA_CHECK(call) do {                                 
    cudaError_t err = (call);                                 
    if (err != cudaSuccess) {                                 
        std::cerr << "CUDA error " << err << " at "           
                  << __FILE__ << ":" << __LINE__ << " -> "    
                  << cudaGetErrorString(err) << std::endl;    
        std::exit(EXIT_FAILURE);                              
    }                                                         
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





void print_gpu_array(const float* d_array, int n) {
    // Allocate host memory to hold the copy
    std::vector<float> h_array(n);

    // Copy from Device to Host
    cudaError_t status = cudaMemcpy(h_array.data(), d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        printf("Error copying memory: %s
", cudaGetErrorString(status));
        return;
    }

    // Print
    printf("GPU Array Content: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.8f ", h_array[i]);
    }
    printf("]
");__global__ void check_vector_finite_kernel(const float* data, int n, int* flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (isinf(data[i]) || isnan(data[i])) {
            atomicExch(flag, 1); // Set flag to 1 if a non-finite value is found
        }
    }
}

// Host wrapper to check for NaN/Inf in a device vector. Returns true if all values are finite.
bool is_vector_finite(const float* d_data, int n) {
    if (d_data == nullptr) {
        // Decide how to handle null pointers; returning true assumes it's not an error state.
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

    printf("
--- Matrix Print (First 10 Rows) ---
");
    printf("Format: Row [Start, End) -> (Col, Val)
");

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
        printf("
");
    }

    printf("------------------------------------
");

    // Cleanup Host Memory
    free(h_row_ptr); free(h_cols); free(h_vals);
}

void print_coo_matrix_gpu(int rows, int cols, int nnz, 
                          const int* d_rows, 
                          const int* d_cols, 
                          const float* d_vals,
                          int max_print) 
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

    for (int g = 0; g < grain; g++) {
        ysol_batch[g * nCell + i] = y[i];
    }

    for (int g = 0; g < grain; g++) {
        int t = t_batch[g];
        if (i == t) {
            ysol_batch[g * nCell + t] += h[t];
        }
    }
}

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
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = xN * yN * zN;
    
    if (tid >= total_threads) return;
    
    int k = tid % zN;
    int j = (tid / zN) % yN;
    int i = tid / (yN * zN);
    
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

        if (j == 0) {
            p[idx_3d_batch(i, 0, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, 1, k, l, sizeY, sizeZ, grain)];
        } else if (j == yN + 1) {
            p[idx_3d_batch(i, yN + 1, k, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, yN, k, l, sizeY, sizeZ, grain)];
        }

        if (k == 0) {
            p[idx_3d_batch(i, j, 0, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, 1, l, sizeY, sizeZ, grain)];
        } else if (k == zN + 1) {
            p[idx_3d_batch(i, j, zN + 1, l, sizeY, sizeZ, grain)] = 
                p[idx_3d_batch(i, j, zN, l, sizeY, sizeZ, grain)];
        }

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

    if (j == 0) {
        p[idx_3d(i, 0, k, sizeY, sizeZ)] = p[idx_3d(i, 1, k, sizeY, sizeZ)];
    } else if (j == yN + 1) {
        p[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = p[idx_3d(i, yN, k, sizeY, sizeZ)];
    }

    if (k == 0) {
        p[idx_3d(i, j, 0, sizeY, sizeZ)] = p[idx_3d(i, j, 1, sizeY, sizeZ)];
    } else if (k == zN + 1) {
        p[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = p[idx_3d(i, j, zN, sizeY, sizeZ)];
    }

    if (i == 0) {
        u[idx_3d(0, j, k, sizeY, sizeZ)] = u_inlet[idx_3d(j, k, 0, sizeZ, 1)];
        p[idx_3d(0, j, k, sizeY, sizeZ)] = p[idx_3d(1, j, k, sizeY, sizeZ)];
    } else if (i == xN + 1) {
        u[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = u[idx_3d(xN, j, k, sizeY, sizeZ)];
        v[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = v[idx_3d(xN, j, k, sizeY, sizeZ)];
        w[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = w[idx_3d(xN, j, k, sizeY, sizeZ)];
    }
}

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

    auto idx = [=] __device__ (int ii, int jj, int kk) -> int {
        return idx_3d(ii, jj, kk, sizeY, sizeZ);
    };

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

    {
        float cont = (dy*dz/2.0f) * (u[idx(i+1,j,k)] - u[idx(i-1,j,k)]) +
                    (dx*dz/2.0f) * (v[idx(i,j+1,k)] - v[idx(i,j-1,k)]) +
                    (dx*dy/2.0f) * (w[idx(i,j,k+1)] - w[idx(i,j,k-1)]);

        out[3*nCell + pos] = cont;
    }
}

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
        row_idx[pos] = i;
        col_idx[pos] = batch_step;
        values[pos] = df;               
    }
}

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

std::tuple<float*,float*,float*,float*> boundary_conditions_final(float *ysol,
                       const int xN, const int yN, const int zN,
                       const float *u_inlet)
{
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    int nCell = 4 * xN * yN * zN;

    
    float *d_u_inlet,*ud, *vd, *wd, *pd;

    CUDA_CHECK(cudaMalloc(&ud, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&vd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&wd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&pd, align_size(totalSize * sizeof(float))));
    CUDA_CHECK(cudaMalloc(&ysol, align_size(nCell * sizeof(float))));

    int inletSize = sizeY * sizeZ;
    CUDA_CHECK(cudaMalloc(&d_u_inlet, align_size(inletSize * sizeof(float))));
    
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(ysol, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_u_inlet));
    CUDA_CHECK(cudaFree(ysol));

    return {ud,vd,wd,pd};
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

    CUDA_CHECK(cudaMemcpy(ysol, y, nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_inlet, u_inlet, inletSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 bi_th(8, 8, 4);
    dim3 bi_bl((xN + bi_th.x - 1) / bi_th.x,
               (yN + bi_th.y - 1) / bi_th.y,
               (zN + bi_th.z - 1) / bi_th.z);
    boundary_conditions_initialization_single<<<bi_bl, bi_th>>>(ysol, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    dim3 blk(8, 8, 4);
    dim3 grd((sizeX + blk.x - 1) / blk.x,
             (sizeY + blk.y - 1) / blk.y,
             (sizeZ + blk.z - 1) / blk.z);
    boundary_conditions_apply_single<<<grd, blk>>>(d_u_inlet, ud, vd, wd, pd, xN, yN, zN);
    CUDA_CHECK(cudaGetLastError());

    dim3 threads(8, 8, 4);
    dim3 blocks((xN + threads.x - 1) / threads.x,
                (yN + threads.y - 1) / threads.y,
                (zN + threads.z - 1) / threads.z);
    kernel_uv_velocity_single<<<blocks, threads>>>(dev_out, Re, ud, vd, pd, wd, xN, yN, zN, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out, dev_out, nCell * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev_out));
    CUDA_CHECK(cudaFree(ud));
    CUDA_CHECK(cudaFree(vd));
    CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(pd));
    CUDA_CHECK(cudaFree(ysol));
    CUDA_CHECK(cudaFree(d_u_inlet));
}

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
            int threads = 256;
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
                printf("Error: Jacobian buffer overflow! Needed %d, had %zu
", total_attempted_nnz, max_nnz);
            }

            
            if (chunk % 1000 == 0) {
                std::cout << "Thread " << tid << ": " << chunk << "/" << chunk_end << std::endl;
            }
        }
    
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
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

__global__ void generate_flags_kernel(int nnz, const float* values, char* flags, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        flags[idx] = (fabsf(values[idx]) > threshold) ? 1 : 0;
    }
}

void filter_csr_cub(float threshold,int rows, int old_nnz,
                    int* d_row_offsets, int* d_cols, float* d_vals,
                    int** d_new_row_offsets, int** d_new_cols, float** d_new_vals,
                    int* new_nnz_out) {
  
    char* d_flags;
    CUDA_CHECK(cudaMalloc((void**)&d_flags, old_nnz * sizeof(char)));

    int blockSize = 256;
    int gridSize = (old_nnz + blockSize - 1) / blockSize;
    
    generate_flags_kernel<<<gridSize, blockSize>>>(old_nnz, d_vals, d_flags, threshold);
    CUDA_CHECK(cudaGetLastError());

    int* d_row_counts; 
    CUDA_CHECK(cudaMalloc((void**)&d_row_counts, rows * sizeof(int)));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));

    CUDA_CHECK(cudaMalloc((void**)d_new_row_offsets, (rows + 1) * sizeof(int)));
    
    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));

    int last_count = 0, last_offset = 0;
    
    CUDA_CHECK(cudaMemcpy(&last_count, &d_row_counts[rows-1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, *d_new_row_offsets + (rows-1), sizeof(int), cudaMemcpyDeviceToHost));
    
    int final_nnz = last_offset + last_count;
    *new_nnz_out = final_nnz;
    
    CUDA_CHECK(cudaMemcpy(*d_new_row_offsets + rows, &final_nnz, sizeof(int), cudaMemcpyHostToDevice));

    printf("Filtering Result: %d -> %d elements threshold:%f 
", old_nnz, final_nnz,threshold);

    CUDA_CHECK(cudaMalloc((void**)d_new_cols, final_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_new_vals, final_nnz * sizeof(float)));
    
    int* d_num_selected;
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected, sizeof(int)));

    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));

    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_cols, d_flags, *d_new_cols, d_num_selected, old_nnz));

    int debug_nnz = 0;
    CUDA_CHECK(cudaMemcpy(&debug_nnz, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
    if (debug_nnz != final_nnz) {
        fprintf(stderr, "Mismatch! Calculated NNZ: %d, DeviceSelect NNZ: %d
", final_nnz, debug_nnz);
    }

    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_row_counts));
    CUDA_CHECK(cudaFree(d_num_selected));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define CUSPARSE_CHECK(call) 
    do { 
        cusparseStatus_t err = call; 
        if (err != CUSPARSE_STATUS_SUCCESS) { 
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ 
                      << " - code " << cusparseGetErrorName(err) << std::endl; 
            exit(1); 
        } 
    } while(0)

void print_csc_matrix(
    int num_rows,
    int num_cols,
    const int* d_col_ptr,
    const int* d_row_ind,
    const float* d_val
) {
    printf("
=== GPU CSC Matrix Dump (%d rows x %d cols) ===
", num_rows, num_cols);

    std::vector<int> h_col_ptr(num_cols + 1);
    CUDA_CHECK(cudaMemcpy(h_col_ptr.data(), d_col_ptr, (num_cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int total_nnz = h_col_ptr[num_cols];
    printf("Total Non-Zeros (read from GPU): %d
", total_nnz);

    if (total_nnz == 0) {
        printf("Matrix is empty.
");
        return;
    }

    std::vector<int> h_row_ind(total_nnz);
    std::vector<float> h_val(total_nnz);

    CUDA_CHECK(cudaMemcpy(h_row_ind.data(), d_row_ind, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(), d_val, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    printf("---------------------------------------------------
");

    for (int col = 0; col < num_cols; col++) {
        
        int start_idx = h_col_ptr[col];
        int end_idx   = h_col_ptr[col + 1];

        if (start_idx == end_idx) continue; 

        printf("Column %d (Ptrs: %d -> %d):
", col, start_idx, end_idx);

        for (int i = start_idx; i < end_idx; i++) {
            int row = h_row_ind[i];
            float val = h_val[i];

            printf("    Row %d : %.6f", row, val);

            if (row < 0 || row >= num_rows) {
                printf("  <-- ERROR: Row Index %d out of bounds [0, %d)", row, num_rows);
            }
            if (i > start_idx && row <= h_row_ind[i-1]) {
                printf("  <-- WARNING: Unsorted or Duplicate Row Index!");
            }
            printf("
");
        }
    }
    printf("===================================================

");
}


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
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    int *d_csrRowPtrA, *d_csrColIndA;
    float *d_csrValA;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA, nnz * sizeof(float)));

    CUSPARSE_CHECK(cusparseXcoo2csr(handle, d_rows_coo, nnz, num_rows, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    
    CUDA_CHECK(cudaMemcpy(d_csrColIndA, d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrValA, d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    std::cout << "Converted A to CSR" << std::endl;

    CUDA_CHECK(cudaMalloc((void**)d_AT_cscOffsets, (num_cols + 1) * sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)d_AT_columns, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_AT_values, nnz * sizeof(float)));

    size_t transposeBufferSize = 0;
    void* d_transposeBuffer = NULL;

    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        &transposeBufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_transposeBuffer, transposeBufferSize));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        *d_AT_values, *d_AT_cscOffsets, *d_AT_columns,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        d_transposeBuffer
    ));
    
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
    
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    
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
    
    int64_t C_num_rows, C_num_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
    
    
    int *d_csrRowPtrC, *d_csrColIndC;
    float *d_csrValC;
    
    CUDA_CHECK(cudaMalloc(&d_csrRowPtrC, (C_num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndC, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValC, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));
    
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, opA,opB,
                       &alpha, matAt, matA, &beta, matC,
                       CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    
    std::cout << "Result copied" << std::endl;

    int *d_trim_row_ptr, *d_trim_cols_ptr;
    float *d_trim_vals_ptr;
    int trim_nnz = 0;
    
    float threshold = 1e-7f;

    
    filter_csr_cub(threshold,C_num_rows, C_nnz, d_csrRowPtrC, d_csrColIndC, d_csrValC, &d_trim_row_ptr, &d_trim_cols_ptr, &d_trim_vals_ptr,&trim_nnz);


    *d_result_rows= d_trim_row_ptr;
    *d_result_cols = d_trim_cols_ptr; 
    *d_result_vals = d_trim_vals_ptr;
    *result_nnz = trim_nnz;
    
    if (dBuffer1) cudaFree(dBuffer1);
    if (dBuffer2) cudaFree(dBuffer2);

    cudaFree(d_csrRowPtrA);
    
    cudaFree(d_csrRowPtrC);
    cudaFree(d_csrColIndC);
    cudaFree(d_csrValC);

    if (d_transposeBuffer) cudaFree(d_transposeBuffer);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matC);
    cusparseDestroySpMat(matAt);    

    cusparseDestroy(handle);
}

__global__ void identity_csr_and_scale_kernel(int N, float alpha, 
                                    int* row_offsets, int* cols, float* vals) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        cols[idx] = idx;
        vals[idx] = alpha;
        row_offsets[idx] = idx;
    }
    
    if (idx == N) {
        row_offsets[idx] = N;
    }
}

void create_identity_csr_and_scale(int N, float alpha, 
                                        int* d_row_offsets, int* d_cols, float* d_vals) 
{
    int blockSize = 256;
    int gridSize = ( (N + 1) + blockSize - 1) / blockSize;
    identity_csr_and_scale_kernel<<<gridSize, blockSize>>>(N, alpha, d_row_offsets, d_cols, d_vals);
    CUDA_CHECK(cudaGetLastError());
}

void add_csr_cusparse(
    float delta,int m, int n,
    int nnzA,
    const int* d_A_row_offsets, const int* d_A_columns, const float* d_A_values,
    int nnzB,
    const int* d_B_row_offsets, const int* d_B_columns, const float* d_B_values,
    int* nnzC_out,
    int** d_C_row_offsets, int** d_C_columns, float** d_C_values
) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    const float alpha = delta;
    const float beta = delta;

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

    CUDA_CHECK(cudaMalloc((void**)d_C_row_offsets, (m + 1) * sizeof(int)));

    void* d_buffer = NULL;
    size_t bufferSize = 0;
    
    CUSPARSE_CHECK(cusparseScsrgeam2_bufferSizeExt(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, d_A_values, *d_C_row_offsets, NULL,
        &bufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

    int nnzC = 0;
    CUSPARSE_CHECK(cusparseXcsrgeam2Nnz(
        handle, m, n,
        descrA, nnzA, d_A_row_offsets, d_A_columns,
        descrB, nnzB, d_B_row_offsets, d_B_columns,
        descrC, *d_C_row_offsets, &nnzC,d_buffer
    ));

    
    *nnzC_out = nnzC;

    CUDA_CHECK(cudaMalloc((void**)d_C_columns, nnzC * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_C_values, nnzC * sizeof(float)));

    CUSPARSE_CHECK(cusparseScsrgeam2(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, *d_C_values, *d_C_row_offsets, *d_C_columns,
        d_buffer
    ));

    CUDA_CHECK(cudaFree(d_buffer));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrB));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrC));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

void scale_and_multiply_on_gpu(
    int rows, int cols, int nnz,
    const int* d_col_offsets,
    const int* d_row_indices,
    const float* d_values,
    const float* d_x,
    float alpha,
    float** d_y_out
) {
    CUDA_CHECK( cudaMalloc((void**)d_y_out, rows * sizeof(float)) );
    
    CUDA_CHECK( cudaMemset(*d_y_out, 0, rows * sizeof(float)) );

    cusparseHandle_t handle;
    CUSPARSE_CHECK( cusparseCreate(&handle) );

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CUSPARSE_CHECK( cusparseCreateCsc(&matA, rows, cols, nnz,
                                      (void*)d_col_offsets, (void*)d_row_indices, (void*)d_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

    CUSPARSE_CHECK( cusparseCreateDnVec(&vecX, cols, (void*)d_x, CUDA_R_32F) );
    
    CUSPARSE_CHECK( cusparseCreateDnVec(&vecY, rows, *d_y_out, CUDA_R_32F) );

    void* d_buffer = nullptr;
    size_t bufferSize = 0;
    float beta = 0.0f;

    CUSPARSE_CHECK( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) );

    CUDA_CHECK( cudaMalloc(&d_buffer, bufferSize) );

    CUSPARSE_CHECK( cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer) );

    CUSPARSE_CHECK( cusparseDestroyDnVec(vecY) );
    CUSPARSE_CHECK( cusparseDestroyDnVec(vecX) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) );
    CUDA_CHECK( cudaFree(d_buffer) );
    CUSPARSE_CHECK( cusparseDestroy(handle) );
}

#define CHECK_CUDSS(func, msg) { 
    cudssStatus_t status = (func); 
    if (status != CUDSS_STATUS_SUCCESS) { 
        printf("cuDSS Error in %s at line %d. Status: %d
", msg, __LINE__, status); 
        exit(EXIT_FAILURE); 
    } 
}

void check_indices_sanity(int nnz, int num_rows, const int* d_indices) {
    std::vector<int> h_indices(nnz);
    cudaMemcpy(h_indices.data(), d_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<nnz; i++) {
        if(h_indices[i] < 0 || h_indices[i] >= num_rows) {
            printf("FATAL ERROR at index %d: Value %d is out of bounds [0, %d)
", 
                   i, h_indices[i], num_rows);
            return;
        }
    }
    printf("Indices look valid (Range checks passed).
");
}

void solve_system_gpu(
    int64_t n,                  
    int64_t nnz,                
    const int* d_row_offsets,   
    const int* d_col_indices,   
    const float* d_values,      
    const float* d_b,           
    float** d_x_out             
) {
    check_indices_sanity(nnz, n, d_col_indices); 

    CUDA_CHECK( cudaMalloc((void**)d_x_out, n * sizeof(float)) );
    CUDA_CHECK( cudaMemset(*d_x_out, 0, n * sizeof(float)) );

    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t solverData;
    cudssMatrix_t matA, vecB, vecX;

    CHECK_CUDSS( cudssCreate(&handle), "cudssCreate" )
    CHECK_CUDSS( cudssConfigCreate(&config), "cudssConfigCreate" )
    CHECK_CUDSS( cudssDataCreate(handle, &solverData), "cudssDataCreate" )

    CHECK_CUDSS( cudssMatrixCreateCsr(&matA, 
                                      n, n, nnz, 
                                      (void*)d_row_offsets, 
                                      NULL,                 
                                      (void*)d_col_indices, 
                                      (void*)d_values, 
                                      CUDA_R_32I,
                                      CUDA_R_32F,
                                      CUDSS_MTYPE_GENERAL, 
                                      CUDSS_MVIEW_FULL, 
                                      CUDSS_BASE_ZERO), "cudssMatrixCreateCsr" )

    CHECK_CUDSS( cudssMatrixCreateDn(&vecX, n, 1, n, 
                                     (void*)*d_x_out, 
                                     CUDA_R_32F,
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(X)" )

    CHECK_CUDSS( cudssMatrixCreateDn(&vecB, n, 1, n, 
                                     (void*)d_b, 
                                     CUDA_R_32F,
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(B)" )

    
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solverData, matA, vecX, vecB), "Analysis" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solverData, matA, vecX, vecB), "Factorization" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solverData, matA, vecX, vecB), "Solve" )

    CHECK_CUDSS( cudssMatrixDestroy(matA), "Destroy A" )
    CHECK_CUDSS( cudssMatrixDestroy(vecB), "Destroy B" )
    CHECK_CUDSS( cudssMatrixDestroy(vecX), "Destroy X" )
    CHECK_CUDSS( cudssDataDestroy(handle, solverData), "Destroy Data" )
    CHECK_CUDSS( cudssConfigDestroy(config), "Destroy Config" )
    CHECK_CUDSS( cudssDestroy(handle), "Destroy Handle" )
}

#define CHECK_CUSPARSE(func) 
{ 
    cusparseStatus_t status = (func); 
    if (status != CUSPARSE_STATUS_SUCCESS) { 
        printf("CUSPARSE API failed at line %d with error: %s (%d)
", 
               __LINE__, cusparseGetErrorString(status), status); 
    } 
}

void sort_coo_matrix_cusparse(
    int num_rows, 
    int num_cols, 
    int nnz,
    int* d_rows,
    int* d_cols,
    float* d_vals
) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    
    int* d_permutation = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_permutation, nnz * sizeof(int)));
    
    float* d_values_sorted=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_values_sorted, nnz * sizeof(float)));

    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, nnz, nnz,
                                    d_permutation, d_values_sorted,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, nnz, d_vals,
                                        CUDA_R_32F) )

    void* d_buffer = nullptr;
    size_t bufferSize = 0;
     
    CHECK_CUSPARSE( cusparseXcoosort_bufferSizeExt(handle, num_rows,
                                                   num_cols, nnz, d_rows,
                                                   d_cols, &bufferSize) )                                        

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

    CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CUDA_CHECK(cudaFree(d_permutation));
    CUDA_CHECK(cudaFree(d_buffer));
}

__global__ void init_indices(int* ptr, int nnz) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nnz) ptr[idx] = idx;
}

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
    
    int *d_indices, *d_indices_sorted;
    int *d_rows_sorted, *d_cols_sorted; 
    float *d_vals_sorted;
    
    CUDA_CHECK(cudaMalloc((void**)&d_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices_sorted, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_rows_sorted, nnz * sizeof(int)));
    
    CUDA_CHECK(cudaMalloc((void**)&d_cols_sorted, nnz * sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_vals_sorted, nnz * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (nnz + blockSize - 1) / blockSize;
    
    init_indices<<<numBlocks, blockSize>>>(d_indices, nnz);
    CUDA_CHECK(cudaGetLastError());

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_rows, d_rows_sorted, d_indices, d_indices_sorted, nnz));

    permute_data<<<numBlocks, blockSize>>>(d_cols, d_cols_sorted, 
                                           d_vals, d_vals_sorted, 
                                           d_indices_sorted, nnz);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_rows, d_rows_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, d_cols_sorted, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, d_vals_sorted, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_indices)); 
    CUDA_CHECK(cudaFree(d_indices_sorted)); 
    CUDA_CHECK(cudaFree(d_rows_sorted)); 
    CUDA_CHECK(cudaFree(d_cols_sorted)); 
    CUDA_CHECK(cudaFree(d_vals_sorted)); 
    CUDA_CHECK(cudaFree(d_temp_storage));
}

#define CUBLAS_CHECK(call) 
    do { 
        cublasStatus_t status = (call); 
        if (status != CUBLAS_STATUS_SUCCESS) { 
            std::cerr << "cuBLAS Error:
" 
                      << "  File:     " << __FILE__ << "
" 
                      << "  Line:     " << __LINE__ << "
" 
                      << "  Function: " << #call << "
" 
                      << "  Status:   " << status << std::endl; 
            exit(EXIT_FAILURE); 
        } 
    } while (0)

float square_norm(cublasHandle_t handle, float * residual, int n)
{
    float result = 0.0f;
    
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    
    CUBLAS_CHECK(cublasSdot(handle, n, residual, 1, residual, 1, &result));
    
    return result * 0.5f;
}

float L2_norm_squared(cublasHandle_t handle, float * residual, int n)
{
    float result = 0.0f;
    
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    
    CUBLAS_CHECK(cublasSnrm2(handle, n, residual, 1, &result));
    
    return result * result ;
}

__global__ void vel_mag_kernel(int n, const float* u, const float* v, const float* w, float* velmag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        velmag[i] = sqrtf(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}

float* compute_vel_mag(int n, float* d_u, float* d_v, float* d_w) {
    float* d_velmag = nullptr;

    size_t size = n * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_velmag, size));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vel_mag_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_u, d_v, d_w, d_velmag);
    
    CUDA_CHECK(cudaGetLastError());
    
    return d_velmag;
}

std::vector<float> gpu_to_vector(const float* d_ptr, int n) {
    std::vector<float> h_vec(n);
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h_vec;
}

void levenberg_marquardt_solver(
    float Re,
    std::vector<float>& y, // In-out: initial guess and final solution
    const int xN, const int yN, const int zN,
    const float* u_inlet, // Host pointer
    const float dx, const float dy, const float dz,
    int max_iterations,
    float initial_lambda,
    float lambda_factor,
    float tolerance
) {
    const int nCell = 4 * xN * yN * zN;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    const int inletSize = sizeY * sizeZ;

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_y, nCell * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));

    float lambda = initial_lambda;

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "\n--- Iteration " << iter << ", Lambda: " << lambda << " ---" << std::endl;

        // 1. Calculate Jacobian and Residual
        auto [d_r, jacobian_coo] = Residuals_Sparse_Jacobian_finite_diff(Re, y.data(), xN, yN, zN, u_inlet, dx, dy, dz);
        auto [d_rows_coo, d_cols_coo, d_vals_coo, nnz] = jacobian_coo;

        // 2. Calculate current cost
        float cost = square_norm(cublas_handle, d_r, nCell);
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
            float *d_hessian_vals, *d_AT_values;
            int hessian_nnz;

            compute_AtA_debug(d_rows_coo, d_cols_coo, d_vals_coo, nnz, nCell, nCell,
                              &d_hessian_rows, &d_hessian_cols, &d_hessian_vals, &hessian_nnz,
                              &d_AT_cscOffsets, &d_AT_columns, &d_AT_values);

            // 6. Create scaled identity matrix lambda*I
            int *d_identity_row_ptr, *d_identity_cols_ptr;
            float *d_identity_vals_ptr;
            CUDA_CHECK(cudaMalloc(&d_identity_row_ptr, (nCell + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_identity_cols_ptr, nCell * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_identity_vals_ptr, nCell * sizeof(float)));
            create_identity_csr_and_scale(nCell, lambda, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr);

            // 7. Form LHS: A = J^T*J + lambda*I
            int *d_lhs_rows, *d_lhs_cols;
            float *d_lhs_vals;
            int lhs_nnz;
            add_csr_cusparse(1.0f, nCell, nCell, hessian_nnz, d_hessian_rows, d_hessian_cols, d_hessian_vals,
                             nCell, d_identity_row_ptr, d_identity_cols_ptr, d_identity_vals_ptr,
                             &lhs_nnz, &d_lhs_rows, &d_lhs_cols, &d_lhs_vals);
            
            // 8. Form RHS: b = -J^T*r
            float* d_rhs = nullptr;
            scale_and_multiply_on_gpu(nCell, nCell, nnz, d_AT_cscOffsets, d_AT_columns, d_AT_values, d_r, -1.0f, &d_rhs);

            // 9. Solve the linear system for the step delta
            float* d_delta = nullptr;
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
            // --- END DIAGNOSTIC ---

            // 10. Propose new state: y_new = y + delta
            float* d_y_new;
            CUDA_CHECK(cudaMalloc(&d_y_new, nCell * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_y_new, d_y, nCell * sizeof(float), cudaMemcpyDeviceToDevice));
            const float alpha = 1.0f;
            CUBLAS_CHECK(cublasSaxpy(cublas_handle, nCell, &alpha, d_delta, 1, d_y_new, 1));
            
            // 11. Evaluate cost of the new state
            std::vector<float> y_new_h = gpu_to_vector(d_y_new, nCell);
            std::vector<float> r_new_h(nCell);
            uv_velocity_single(r_new_h.data(), Re, y_new_h.data(), xN, yN, zN, u_inlet, dx, dy, dz);
            
            float* d_r_new;
            CUDA_CHECK(cudaMalloc(&d_r_new, nCell * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_r_new, r_new_h.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));
            
            float new_cost = square_norm(cublas_handle, d_r_new, nCell);

            // 12. Accept or reject the step
            if (new_cost < cost) {
                std::cout << "  --> Step ACCEPTED. New cost: " << new_cost << " (improvement: " << cost - new_cost << ")" << std::endl;
                CUDA_CHECK(cudaMemcpy(d_y, d_y_new, nCell * sizeof(float), cudaMemcpyDeviceToDevice));
                y = y_new_h; // Update host vector as well
                lambda /= lambda_factor;
                step_accepted = true;
            } else {
                std::cout << "  --> Step REJECTED. New cost: " << new_cost << " (no improvement)" << std::endl;
                lambda *= lambda_factor;
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
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, nCell * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    std::cout << "Levenberg-Marquardt solver finished." << std::endl;
}

