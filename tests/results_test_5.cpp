#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "common.cuh"
#include "coordinates.h"
#include "navier_stokes.cuh"
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

// Helper function to read fold results
std::vector<float> read_fold_results(const std::string& filename) {
    std::vector<float> fold_values;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return fold_values;
    }
    
    float value;
    while (file >> value) {
        fold_values.push_back(value);
    }
    
    // Check if we stopped due to error (not just EOF)
    if (!file.eof() && file.fail()) {
        std::cerr << "Error reading from file: " << filename << std::endl;
        fold_values.clear(); // Optional: return empty vector on error
    }
        if (fold_values.empty()) {
        std::cout << "File is empty" << std::endl;
    } else {
        // Process fold_values
        std::cout << "Read " << fold_values.size() << " values" << std::endl;
    }

    return fold_values;
}

// Helper function to read jacobian results
void read_jacobian_results(const std::string& filename, std::vector<int>& rows, std::vector<int>& cols, std::vector<float>& vals) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    int r, c;
    float v;
    while (file >> r >> c >> v) {
        rows.push_back(r - 1); // Adjust to 0-based indexing
        cols.push_back(c - 1); // Adjust to 0-based indexing
        vals.push_back(v);
    }
}


TEST_CASE("Check values of fold and jacobian for (2,2,2)") {
    // Problem size
    int xN = 5, yN = 5, zN = 5;
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

    // Read expected results
    auto expected_fold = read_fold_results("../../tests/test_results_original/results_for_fold(5-5-5).txt");
    std::vector<int> expected_rows, expected_cols;
    std::vector<float> expected_vals;
    read_jacobian_results("../../tests/test_results_original/results_for_jacobian(5-5-5).txt", expected_rows, expected_cols, expected_vals);

    REQUIRE(fold.size() == nCell);
    REQUIRE(expected_fold.size() == nCell);
    for(size_t i = 0; i < fold.size(); ++i) {
        REQUIRE(std::abs(fold[i] - expected_fold[i]) < 1e-5);
    }
    
    REQUIRE(nnz == expected_vals.size());

    std::vector<int> h_rows_coo(nnz);
    std::vector<int> h_cols_coo(nnz);
    std::vector<float> h_vals_coo(nnz);

    CUDA_CHECK(cudaMemcpy(h_rows_coo.data(), d_rows_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols_coo.data(), d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals_coo.data(), d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    struct CooEntry {
        int r, c;
        float v;
        bool operator<(const CooEntry& other) const {
            if (r != other.r) return r < other.r;
            return c < other.c;
        }
    };

    std::vector<CooEntry> expected_coo;
    expected_coo.reserve(expected_vals.size());
    for(size_t i=0; i<expected_vals.size(); ++i) {
        expected_coo.push_back({expected_rows[i], expected_cols[i], expected_vals[i]});
    }

    std::vector<CooEntry> actual_coo;
    actual_coo.reserve(nnz);
    for(size_t i=0; i<nnz; ++i) {
        actual_coo.push_back({h_rows_coo[i], h_cols_coo[i], h_vals_coo[i]});
    }

    std::sort(expected_coo.begin(), expected_coo.end());
    std::sort(actual_coo.begin(), actual_coo.end());
    
    for(size_t i = 0; i < nnz; ++i) {
        REQUIRE(actual_coo[i].r == expected_coo[i].r);
        REQUIRE(actual_coo[i].c == expected_coo[i].c);
        // REQUIRE(std::abs(actual_coo[i].v - expected_coo[i].v) < 1e-5);
        REQUIRE_THAT(actual_coo[i].v, Catch::Matchers::WithinAbs(expected_coo[i].v, 1e-4));

    }

    // Cleanup
    cudaFree(d_rows_coo);
    cudaFree(d_cols_coo);
    cudaFree(d_vals_coo);
}
