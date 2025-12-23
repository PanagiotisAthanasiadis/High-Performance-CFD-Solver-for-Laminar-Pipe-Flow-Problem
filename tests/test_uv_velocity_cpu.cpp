#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "coordinates.h"
#include "uv_velocity.h"
#include <fstream>



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


TEST_CASE("Main function logic") {
    // Problem size
    int xN = 2, yN = 2, zN = 2; // smaller size for testing
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

    int totalSize = sizeX * sizeY * sizeZ;

    std::vector<float> u(totalSize, 0.0);
    std::vector<float> v(totalSize, 0.0);
    std::vector<float> w(totalSize, 0.0);
    std::vector<float> p(totalSize, 0.0);

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

 

    std::vector<float> fold = uv_velocity(Re, y, xN, yN, zN, u_inlet, dx, dy, dz, u, v, w, p);
    

    auto expected_fold = read_fold_results("../../tests/test_results_original/results_for_fold(2-2-2).txt");
  
    REQUIRE(fold.size() == nCell);
    REQUIRE(expected_fold.size() == nCell);
    for(size_t i = 0; i < fold.size(); ++i) {
        REQUIRE(std::abs(fold[i] - expected_fold[i]) < 1e-5);
    }

    std::cout << "Output vector (size = " << fold.size() << "):\n";
    for (size_t i = 0; i < fold.size(); ++i) {
        std::cout << "out[" << i << "] = " << std::fixed  << fold[i] << "\n";
    }
}