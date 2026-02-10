#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

// --- Clad Include ---
#include "clad/Differentiator/Differentiator.h"

// Standard 3D indexing: (i, j, k) -> linear index
inline int idx_3d(int i, int j, int k, int sizeY, int sizeZ) {
    return (i * sizeY + j) * sizeZ + k;
}

// Coordinate generation function
template<typename T>
std::tuple<T, T, T> coordinates(std::vector<T>& xcoor, std::vector<T>& ycoor, 
                                std::vector<T>& zcoor,
                                int xN, int yN, int zN, 
                                T L, T M, T N) {
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    
    T dx = L / xN;
    T dy = M / yN;
    T dz = N / zN;
    
    for (int i = 0; i < sizeX; ++i) {
        for (int j = 0; j < sizeY; ++j) {
            for (int k = 0; k < sizeZ; ++k) {
                int idx = idx_3d(i, j, k, sizeY, sizeZ);
                xcoor[idx] = (i - 0.5) * dx;
                ycoor[idx] = (j - 0.5) * dy;
                zcoor[idx] = (k - 0.5) * dz;
            }
        }
    }
    
    return std::make_tuple(dx, dy, dz);
}

// Apply boundary conditions - NOT differentiated
template<typename T>
void apply_boundary_conditions(const T* y, int xN, int yN, int zN, 
                               const T* u_inlet,
                               T* u, T* v, T* w, T* p) {
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;

    // Initialize to zero
    for (int i = 0; i < totalSize; ++i) {
        u[i] = T(0.0);
        v[i] = T(0.0);
        w[i] = T(0.0);
        p[i] = T(0.0);
    }

    // Fill interior from y[]
    for (int i = 1; i <= xN; i++) {
        for (int j = 1; j <= yN; j++) {
            for (int k = 1; k <= zN; k++) {
                int pos = (i - 1) * yN * zN + (j - 1) * zN + (k - 1);
                u[idx_3d(i, j, k, sizeY, sizeZ)] = y[pos];
                v[idx_3d(i, j, k, sizeY, sizeZ)] = y[xN * yN * zN + pos];
                w[idx_3d(i, j, k, sizeY, sizeZ)] = y[2 * xN * yN * zN + pos];
                p[idx_3d(i, j, k, sizeY, sizeZ)] = y[3 * xN * yN * zN + pos];
            }
        }
    }

    // Boundary conditions - X-direction
    for (int j = 0; j < sizeY; j++) {
        for (int k = 0; k < sizeZ; k++) {
            u[idx_3d(0, j, k, sizeY, sizeZ)]      = u_inlet[j * sizeZ + k];
            u[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = u[idx_3d(xN, j, k, sizeY, sizeZ)];
            v[idx_3d(0, j, k, sizeY, sizeZ)]      = T(0.0);
            v[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = v[idx_3d(xN, j, k, sizeY, sizeZ)];
            w[idx_3d(0, j, k, sizeY, sizeZ)]      = T(0.0);
            w[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = w[idx_3d(xN, j, k, sizeY, sizeZ)];
            p[idx_3d(0, j, k, sizeY, sizeZ)]      = p[idx_3d(1, j, k, sizeY, sizeZ)];
            p[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = T(0.0);
        }
    }

    // Boundary conditions - Y-direction
    for (int i = 0; i < sizeX; i++) {
        for (int k = 0; k < sizeZ; k++) {
            u[idx_3d(i, 0, k, sizeY, sizeZ)]      = T(0.0);
            u[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = T(0.0);
            v[idx_3d(i, 0, k, sizeY, sizeZ)]      = T(0.0);
            v[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = T(0.0);
            w[idx_3d(i, 0, k, sizeY, sizeZ)]      = T(0.0);
            w[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = T(0.0);
            p[idx_3d(i, 0, k, sizeY, sizeZ)]      = p[idx_3d(i, 1, k, sizeY, sizeZ)];
            p[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = p[idx_3d(i, yN, k, sizeY, sizeZ)];
        }
    }

    // Boundary conditions - Z-direction
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            u[idx_3d(i, j, 0, sizeY, sizeZ)]      = T(0.0);
            u[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = T(0.0);
            v[idx_3d(i, j, 0, sizeY, sizeZ)]      = T(0.0);
            v[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = T(0.0);
            w[idx_3d(i, j, 0, sizeY, sizeZ)]      = T(0.0);
            w[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = T(0.0);
            p[idx_3d(i, j, 0, sizeY, sizeZ)]      = p[idx_3d(i, j, 1, sizeY, sizeZ)];
            p[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = p[idx_3d(i, j, zN, sizeY, sizeZ)];
        }
    }
}

// ONLY THIS FUNCTION IS DIFFERENTIATED
// Compute residual for a single output component
// u, v, w, p are CONST because we only read from them
template<typename T> 
T compute_residual_component(T Re, const T* y,
                             int xN, int yN, int zN,
                             T dx, T dy, T dz,
                             const T* u, const T* v, const T* w, const T* p,
                             int idx_out) {
    
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int nCell = xN * yN * zN;
    
    // Compute which cell and which equation
    int eq_type = idx_out / nCell;  // 0=u, 1=v, 2=w, 3=continuity
    int pos = idx_out % nCell;
    
    // Convert linear pos back to i,j,k
    int k = pos % zN;
    int j = (pos / zN) % yN;
    int i = pos / (yN * zN);
    
    // Adjust to actual grid indices (1-based for interior)
    i += 1;
    j += 1;
    k += 1;

    T result = T(0.0);

    if (eq_type == 0) {
        // U-momentum
        T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*u[idx_3d(i-1,j,k,sizeY,sizeZ)] );
        T conv_y = 0.5 * dx * dz * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - u[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
        T conv_z = 0.5 * dx * dy * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - u[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
        T pres    = (dy * dz) * ( p[idx_3d(i+1,j,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

        T diff = (1.0/Re) * (
            (dy*dz/dx) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
            (dx*dz/dy) * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
            (dx*dy/dz) * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j,k-1,sizeY,sizeZ)] )
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 1) {
        // V-momentum
        T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*v[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*v[idx_3d(i-1,j,k,sizeY,sizeZ)] );
        T conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
        T conv_z = 0.5 * dx * dy * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - v[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
        T pres    = (dx * dz) * ( p[idx_3d(i,j+1,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

        T diff = (1.0/Re) * (
            (dy*dz/dx) * ( v[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
            (dx*dz/dy) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
            (dx*dy/dz) * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j,k-1,sizeY,sizeZ)] )
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 2) {
        // W-momentum
        T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*w[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*w[idx_3d(i-1,j,k,sizeY,sizeZ)] );
        T conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*w[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*w[idx_3d(i,j-1,k,sizeY,sizeZ)] );
        T conv_z = 0.5 * dx * dy * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
        T pres    = (dx * dy) * ( p[idx_3d(i,j,k+1,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

        T diff = (1.0/Re) * (
            (dy*dz/dx) * ( w[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
            (dx*dz/dy) * ( w[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
            (dx*dy/dz) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j,k-1,sizeY,sizeZ)] )
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 3) {
        // Continuity
        T cont = (dy*dz/2.0) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)] )
                    + (dx*dz/2.0) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)] )
                    + (dx*dy/2.0) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)] );

        result = cont;
    }

    return result;
}

// ==========================================
// Clad-based Jacobian Computation
// ==========================================
void compute_jacobian(const std::vector<float>& y, const std::vector<float>& u_inlet,
                      std::vector<float>& residual, std::vector<float>& jac_values,
                      std::vector<int>& row_indices, std::vector<int>& col_indices,
                      int xN, int yN, int zN, float dx, float dy, float dz, float Re) {

    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;
    
    // Allocate field arrays
    std::vector<float> u(totalSize), v(totalSize), w(totalSize), p(totalSize);
    
    // Apply boundary conditions ONCE (not differentiated)
    apply_boundary_conditions(y.data(), xN, yN, zN, u_inlet.data(),
                             u.data(), v.data(), w.data(), p.data());

    int nCell = xN * yN * zN;
    int nVar = 4 * nCell;
    residual.resize(nVar);
    
    // Prepare dense Jacobian container
    std::vector<float> dense_jac(nVar * nVar, 0.0f);

    // Generate gradient function - differentiate w.r.t. "y"
    auto grad_func = clad::gradient(compute_residual_component<float>, "y");

    std::cout << "Computing Jacobian row by row..." << std::endl;
    
    // Compute each row of the Jacobian
    for (int row = 0; row < nVar; ++row) {
        if (row % 8 == 0) {
            std::cout << "  Processing row " << row << "/" << nVar << std::endl;
        }
        
        // Compute residual[row] 
        residual[row] = compute_residual_component(Re, y.data(), xN, yN, zN, dx, dy, dz,
                                                   u.data(), v.data(), w.data(), p.data(), row);
        
        // Compute gradient of residual[row] w.r.t. y
        std::vector<float> grad_y(nVar, 0.0f);
        float dummy_Re_grad = 0.0f;
        
        grad_func.execute(Re, y.data(), xN, yN, zN, dx, dy, dz,
                         u.data(), v.data(), w.data(), p.data(), row,
                         &dummy_Re_grad, grad_y.data());
        
        // Store in dense Jacobian
        for (int col = 0; col < nVar; ++col) {
            dense_jac[row * nVar + col] = grad_y[col];
        }
    }

    std::cout << "Converting to sparse format..." << std::endl;
    
    // Convert Dense to Sparse
    jac_values.clear();
    row_indices.clear();
    col_indices.clear();

    for (int i = 0; i < nVar; ++i) {
        for (int j = 0; j < nVar; ++j) {
            float val = dense_jac[i * nVar + j];
            if (std::abs(val) > 1e-12) { 
                jac_values.push_back(val);
                row_indices.push_back(i);
                col_indices.push_back(j);
            }
        }
    }
    
    std::cout << "Jacobian computation complete. Non-zeros: " << jac_values.size() << std::endl;
}

int main() {
    const int xN = 2, yN = 2, zN = 2;
    const int actualNCells = xN * yN * zN;
    const int nVar = 4 * actualNCells;
    
    int sizeX = xN + 2;
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    
    // Physical parameters
    double mu = 0.001;
    double L = 1.0;
    double M = 0.2; 
    double N = 0.2; 
    double rho = 1.0;
    double u0 = 1.0; 
    double Re = (rho * (M/2) * u0) / (mu);
    
    std::cout << "=== CFD Solver with Clad AD ===" << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Grid size: " << xN << " x " << yN << " x " << zN << std::endl;
    std::cout << "Total variables: " << nVar << std::endl;
    
    // Coordinate arrays
    std::vector<float> xcoor(sizeX * sizeY * sizeZ);
    std::vector<float> ycoor(sizeX * sizeY * sizeZ);
    std::vector<float> zcoor(sizeX * sizeY * sizeZ);
    
    auto [dx, dy, dz] = coordinates(xcoor, ycoor, zcoor, xN, yN, zN, 
                                    (float)L, (float)M, (float)N);
    
    std::cout << "Grid spacing: dx=" << dx << ", dy=" << dy << ", dz=" << dz << std::endl;
    
    // Inlet velocity profile (parabolic)
    std::vector<float> u_inlet(sizeY * sizeZ);
    for (int j = 0; j < sizeY; ++j) {
        for (int k = 0; k < sizeZ; ++k) {
            double yv = M * j / (sizeY - 1);
            double zv = N * k / (sizeZ - 1);
            u_inlet[j * sizeZ + k] = 16.0 * u0 * (yv / M) * (1.0 - yv / M) * (zv / N) * (1.0 - zv / N);
        }
    }
    
    // State vector initialization
    std::vector<float> y(nVar, 0.1);
    double blockSize = actualNCells;
    for(int i = 0; i < blockSize; i++) y[i] = u0;
    
    // Compute Jacobian
    std::vector<float> residual;
    std::vector<float> jac_values;
    std::vector<int> row_indices, col_indices;
    
    std::cout << "\n=== Computing Jacobian ===" << std::endl;
    compute_jacobian(y, u_inlet, residual, jac_values, row_indices, col_indices,
                    xN, yN, zN, (float)dx, (float)dy, (float)dz, (float)Re);
    
    // Print statistics
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Non-zero Jacobian entries: " << jac_values.size() << std::endl;
    std::cout << "Sparsity: " << (100.0 * jac_values.size() / (nVar * nVar)) << "%" << std::endl;
    
    // Print first few Jacobian entries
    std::cout << "\nFirst 20 Jacobian entries:" << std::endl;
    for (int i = 0; i < std::min(20, (int)jac_values.size()); ++i) {
        std::cout << "J[" << row_indices[i] << "," << col_indices[i] << "] = " 
                  << jac_values[i] << std::endl;
    }
    
    // Print residual statistics
    float sum_residual = 0.0;
    float max_residual = 0.0;
    for (float r : residual) {
        sum_residual += std::abs(r);
        max_residual = std::max(max_residual, std::abs(r));
    }
    std::cout << "\nResidual statistics:" << std::endl;
    std::cout << "  Mean |F|: " << sum_residual / residual.size() << std::endl;
    std::cout << "  Max |F|:  " << max_residual << std::endl;
    
    return 0;
}