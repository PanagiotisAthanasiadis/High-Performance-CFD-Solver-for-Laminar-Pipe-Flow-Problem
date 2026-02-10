#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

// --- Clad Include ---
#include "clad/Differentiator/Differentiator.h"

// Standard 3D indexing: (i, j, k) -> linear index
// Use a macro to avoid Clad trying to differentiate it
#define IDX_3D(i, j, k, sizeY, sizeZ) ((i) * (sizeY) + (j)) * (sizeZ) + (k)

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
                int idx = IDX_3D(i, j, k, sizeY, sizeZ);
                xcoor[idx] = (i - 0.5) * dx;
                ycoor[idx] = (j - 0.5) * dy;
                zcoor[idx] = (k - 0.5) * dz;
            }
        }
    }
    
    return std::make_tuple(dx, dy, dz);
}

// Helper to get field value from state vector y with boundary conditions applied
// Non-recursive version to avoid segfault
template<typename T>
inline T get_field_value(const T* y, int field_type, int i, int j, int k,
                         int xN, int yN, int zN, const T* u_inlet) {
    const int sizeY = yN + 2;
    const int sizeZ = zN + 2;
    const int nCell = xN * yN * zN;
    
    // Clamp to valid range and apply boundary conditions
    int i_clamped = i;
    int j_clamped = j;
    int k_clamped = k;
    
    // Handle x-direction boundaries
    if (i < 1) {
        i_clamped = 1;
        if (field_type == 0) { // u at inlet
            return u_inlet[j * sizeY + k];
        }
        // v, w, p at inlet
        return T(0.0);
    }
    if (i > xN) {
        i_clamped = xN;
        if (field_type == 3) { // pressure at outlet
            return T(0.0);
        }
        // u, v, w use outlet BC (extrapolation handled by clamping to xN)
    }
    
    // Handle y-direction boundaries
    if (j < 1) {
        j_clamped = 1;
        if (field_type != 3) return T(0.0); // u, v, w are zero at walls
        // pressure uses Neumann BC (extrapolation handled by clamping)
    }
    if (j > yN) {
        j_clamped = yN;
        if (field_type != 3) return T(0.0); // u, v, w are zero at walls
        // pressure uses Neumann BC (extrapolation handled by clamping)
    }
    
    // Handle z-direction boundaries
    if (k < 1) {
        k_clamped = 1;
        if (field_type != 3) return T(0.0); // u, v, w are zero at walls
        // pressure uses Neumann BC (extrapolation handled by clamping)
    }
    if (k > zN) {
        k_clamped = zN;
        if (field_type != 3) return T(0.0); // u, v, w are zero at walls
        // pressure uses Neumann BC (extrapolation handled by clamping)
    }
    
    // Get from interior
    const int pos = (i_clamped - 1) * yN * zN + (j_clamped - 1) * zN + (k_clamped - 1);
    return y[field_type * nCell + pos];
}

// ONLY THIS FUNCTION IS DIFFERENTIATED
// Now it directly accesses y through get_field_value, so Clad can trace dependencies
template<typename T> 
T compute_residual_component(T Re, const T* y,
                             int i, int j, int k, int eq_type,
                             int xN, int yN, int zN,
                             T dx, T dy, T dz,
                             const T* u_inlet) {
    
    T result = T(0.0);

    if (eq_type == 0) {
        // U-momentum
        T u_ip1 = get_field_value(y, 0, i+1, j, k, xN, yN, zN, u_inlet);
        T u_im1 = get_field_value(y, 0, i-1, j, k, xN, yN, zN, u_inlet);
        T u_jp1 = get_field_value(y, 0, i, j+1, k, xN, yN, zN, u_inlet);
        T u_jm1 = get_field_value(y, 0, i, j-1, k, xN, yN, zN, u_inlet);
        T u_kp1 = get_field_value(y, 0, i, j, k+1, xN, yN, zN, u_inlet);
        T u_km1 = get_field_value(y, 0, i, j, k-1, xN, yN, zN, u_inlet);
        T u_ijk = get_field_value(y, 0, i, j, k, xN, yN, zN, u_inlet);
        
        T v_jp1 = get_field_value(y, 1, i, j+1, k, xN, yN, zN, u_inlet);
        T v_jm1 = get_field_value(y, 1, i, j-1, k, xN, yN, zN, u_inlet);
        
        T w_kp1 = get_field_value(y, 2, i, j, k+1, xN, yN, zN, u_inlet);
        T w_km1 = get_field_value(y, 2, i, j, k-1, xN, yN, zN, u_inlet);
        
        T p_ip1 = get_field_value(y, 3, i+1, j, k, xN, yN, zN, u_inlet);
        T p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
        
        T conv_x = 0.5 * dy * dz * (u_ip1*u_ip1 - u_im1*u_im1);
        T conv_y = 0.5 * dx * dz * (u_jp1*v_jp1 - u_jm1*v_jm1);
        T conv_z = 0.5 * dx * dy * (u_kp1*w_kp1 - u_km1*w_km1);
        T pres = (dy * dz) * (p_ip1 - p_ijk);

        T diff = (1.0/Re) * (
            (dy*dz/dx) * (u_ip1 - 2.0*u_ijk + u_im1) +
            (dx*dz/dy) * (u_jp1 - 2.0*u_ijk + u_jm1) +
            (dx*dy/dz) * (u_kp1 - 2.0*u_ijk + u_km1)
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 1) {
        // V-momentum
        T v_ip1 = get_field_value(y, 1, i+1, j, k, xN, yN, zN, u_inlet);
        T v_im1 = get_field_value(y, 1, i-1, j, k, xN, yN, zN, u_inlet);
        T v_jp1 = get_field_value(y, 1, i, j+1, k, xN, yN, zN, u_inlet);
        T v_jm1 = get_field_value(y, 1, i, j-1, k, xN, yN, zN, u_inlet);
        T v_kp1 = get_field_value(y, 1, i, j, k+1, xN, yN, zN, u_inlet);
        T v_km1 = get_field_value(y, 1, i, j, k-1, xN, yN, zN, u_inlet);
        T v_ijk = get_field_value(y, 1, i, j, k, xN, yN, zN, u_inlet);
        
        T u_ip1 = get_field_value(y, 0, i+1, j, k, xN, yN, zN, u_inlet);
        T u_im1 = get_field_value(y, 0, i-1, j, k, xN, yN, zN, u_inlet);
        
        T w_kp1 = get_field_value(y, 2, i, j, k+1, xN, yN, zN, u_inlet);
        T w_km1 = get_field_value(y, 2, i, j, k-1, xN, yN, zN, u_inlet);
        
        T p_jp1 = get_field_value(y, 3, i, j+1, k, xN, yN, zN, u_inlet);
        T p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
        
        T conv_x = 0.5 * dy * dz * (u_ip1*v_ip1 - u_im1*v_im1);
        T conv_y = 0.5 * dx * dz * (v_jp1*v_jp1 - v_jm1*v_jm1);
        T conv_z = 0.5 * dx * dy * (v_kp1*w_kp1 - v_km1*w_km1);
        T pres = (dx * dz) * (p_jp1 - p_ijk);

        T diff = (1.0/Re) * (
            (dy*dz/dx) * (v_ip1 - 2.0*v_ijk + v_im1) +
            (dx*dz/dy) * (v_jp1 - 2.0*v_ijk + v_jm1) +
            (dx*dy/dz) * (v_kp1 - 2.0*v_ijk + v_km1)
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 2) {
        // W-momentum
        T w_ip1 = get_field_value(y, 2, i+1, j, k, xN, yN, zN, u_inlet);
        T w_im1 = get_field_value(y, 2, i-1, j, k, xN, yN, zN, u_inlet);
        T w_jp1 = get_field_value(y, 2, i, j+1, k, xN, yN, zN, u_inlet);
        T w_jm1 = get_field_value(y, 2, i, j-1, k, xN, yN, zN, u_inlet);
        T w_kp1 = get_field_value(y, 2, i, j, k+1, xN, yN, zN, u_inlet);
        T w_km1 = get_field_value(y, 2, i, j, k-1, xN, yN, zN, u_inlet);
        T w_ijk = get_field_value(y, 2, i, j, k, xN, yN, zN, u_inlet);
        
        T u_ip1 = get_field_value(y, 0, i+1, j, k, xN, yN, zN, u_inlet);
        T u_im1 = get_field_value(y, 0, i-1, j, k, xN, yN, zN, u_inlet);
        
        T v_jp1 = get_field_value(y, 1, i, j+1, k, xN, yN, zN, u_inlet);
        T v_jm1 = get_field_value(y, 1, i, j-1, k, xN, yN, zN, u_inlet);
        
        T p_kp1 = get_field_value(y, 3, i, j, k+1, xN, yN, zN, u_inlet);
        T p_ijk = get_field_value(y, 3, i, j, k, xN, yN, zN, u_inlet);
        
        T conv_x = 0.5 * dy * dz * (u_ip1*w_ip1 - u_im1*w_im1);
        T conv_y = 0.5 * dx * dz * (v_jp1*w_jp1 - v_jm1*w_jm1);
        T conv_z = 0.5 * dx * dy * (w_kp1*w_kp1 - w_km1*w_km1);
        T pres = (dx * dy) * (p_kp1 - p_ijk);

        T diff = (1.0/Re) * (
            (dy*dz/dx) * (w_ip1 - 2.0*w_ijk + w_im1) +
            (dx*dz/dy) * (w_jp1 - 2.0*w_ijk + w_jm1) +
            (dx*dy/dz) * (w_kp1 - 2.0*w_ijk + w_km1)
        );

        result = conv_x + conv_y + conv_z + pres - diff;
    }
    else if (eq_type == 3) {
        // Continuity
        T u_ip1 = get_field_value(y, 0, i+1, j, k, xN, yN, zN, u_inlet);
        T u_im1 = get_field_value(y, 0, i-1, j, k, xN, yN, zN, u_inlet);
        T v_jp1 = get_field_value(y, 1, i, j+1, k, xN, yN, zN, u_inlet);
        T v_jm1 = get_field_value(y, 1, i, j-1, k, xN, yN, zN, u_inlet);
        T w_kp1 = get_field_value(y, 2, i, j, k+1, xN, yN, zN, u_inlet);
        T w_km1 = get_field_value(y, 2, i, j, k-1, xN, yN, zN, u_inlet);
        
        T cont = (dy*dz/2.0) * (u_ip1 - u_im1) +
                 (dx*dz/2.0) * (v_jp1 - v_jm1) +
                 (dx*dy/2.0) * (w_kp1 - w_km1);

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
        
        // Pre-compute indices
        int eq_type = row / nCell;
        int pos = row % nCell;
        int k = pos % zN;
        int j = (pos / zN) % yN;
        int i = pos / (yN * zN);
        
        // Adjust to actual grid indices (1-based for interior)
        i += 1;
        j += 1;
        k += 1;
        
        // Compute residual[row] 
        residual[row] = compute_residual_component(Re, y.data(), i, j, k, eq_type,
                                                   xN, yN, zN, dx, dy, dz, u_inlet.data());
        
        // Compute gradient of residual[row] w.r.t. y
        std::vector<float> grad_y(nVar, 0.0f);
       
        
        grad_func.execute(Re, y.data(), i, j, k, eq_type, xN, yN, zN, dx, dy, dz, u_inlet.data(),
                  grad_y.data());
        
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
    if (nVar > 0) {
        std::cout << "Sparsity: " << (100.0 * jac_values.size() / (nVar * nVar)) << "%" << std::endl;
    }
    
    // Print first few Jacobian entries
    std::cout << "\nFirst 20 Jacobian entries:" << std::endl;
    for (int i = 0; i < std::min(208, (int)jac_values.size()); ++i) {
        std::cout << "J[" << row_indices[i] +1 << "," << col_indices[i] +1 << "] = " 
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