#include <catch2/catch_test_macros.hpp>
#include "common.cuh"
#include "coordinates.h"
#include "navier_stokes.cuh"
#include "solver.cuh"
#include <vector>
#include <iostream>
#include <cmath>

TEST_CASE("Main function logic") {
    cudaFree(0);
    // Problem size
    int xN = 10, yN = 2, zN = 2; // smaller size for testing
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

    REQUIRE(Re > 0);

    // Generate coordinates
    std::vector<float> xcoor(sizeX * sizeY * sizeZ);
    std::vector<float> ycoor(sizeX * sizeY * sizeZ);
    std::vector<float> zcoor(sizeX * sizeY * sizeZ);

    auto [dx, dy, dz] = coordinates(xcoor, ycoor, zcoor, xN, yN, zN, L, M, N);

    REQUIRE(dx > 0);
    REQUIRE(dy > 0);
    REQUIRE(dz > 0);

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

    REQUIRE(nnz > 0);

    //Unecessery there was always on the device
    float *devu_fold,*d_solution;
    std::vector<float> solution(nCell);
    CUDA_CHECK(cudaMalloc(&devu_fold, nCell * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devu_fold, fold.data(), nCell * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_solution, nCell * sizeof(float)));

    // Convert to CSR
    int *d_row_ptr, *d_col_idx;
    float *d_values;
    convert_coo_to_csr_gpu(d_rows_coo, d_cols_coo, d_vals_coo, nnz, nCell,
                          &d_row_ptr, &d_col_idx, &d_values);

    // //Solver is linear ou need a wrapper
    // solve_with_cusolver_sparse(d_row_ptr, d_col_idx, d_values, nnz,
    //                           devu_fold, d_solution, nCell);

    CUDA_CHECK(cudaMemcpy( solution.data(),d_solution, nCell * sizeof(float), cudaMemcpyDeviceToHost));

    // Some basic check on the solution
    REQUIRE(solution.size() == nCell);
    for(const auto& val : solution) {
        REQUIRE(!std::isnan(val));
        REQUIRE(!std::isinf(val));
    }


    // Cleanup
    cudaFree(d_rows_coo);
    cudaFree(d_cols_coo);
    cudaFree(d_vals_coo);
    cudaFree(devu_fold);
    cudaFree(d_solution);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
}
