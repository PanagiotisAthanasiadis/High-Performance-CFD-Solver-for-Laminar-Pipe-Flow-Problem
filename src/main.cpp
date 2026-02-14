#include "common.cuh"
#include "coordinates.h"
#include "navier_stokes.cuh"
#include "solver.cuh"

// ============================================================================ 
// MAIN FUNCTION
// ============================================================================ 

int main()
{
    // Problem size
    // int xN = 100, yN = 50, zN = 50;
    int xN = 20, yN = 20, zN = 20;
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

    auto [fold, sparse_matrix] = Residuals_Sparse_Jacobian_finite_diff(
        Re, y.data(), xN, yN, zN, u_inlet.data(), dx, dy, dz);
    
    for (int i=0; i<fold.size(); i++)
    {
        std::cout << "i: " << fold[i]<<std::endl;
    }
    auto [d_rows_coo, d_cols_coo, d_vals_coo, nnz] = sparse_matrix;
    
    std::cout << "Sparse matrix: " << nnz << " non-zeros" << std::endl;
    std::cout << "Sparsity: " << (100.0 * nnz / ((long long)nCell * nCell)) << "%" << std::endl;
    
    // Option 1: Copy to host 
    // std::vector<int> rows(nnz), cols(nnz);
    // std::vector<float> vals(nnz);
    // cudaMemcpy(rows.data(), d_rows_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cols.data(), d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(vals.data(), d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "--- Printing Sparse Data ---" << std::endl;
    // for (int i = 0; i < nnz; ++i) {
    //   std::cout << "Index " << i << ": "
    //             << "(Row=" << rows[i] << ", "
    //             << "Col=" << cols[i] << ") "
    //             << "Val=" << vals[i] << std::endl;
    // }





    std::cout << "\nComputation complete!" << std::endl;

    return 0;
}
