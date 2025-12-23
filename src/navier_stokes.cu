#include "common.cuh"
#include "kernels.cuh"
#include "navier_stokes.cuh"

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

std::pair<std::vector<float>, std::tuple<int*, int*, float*, int>>
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
