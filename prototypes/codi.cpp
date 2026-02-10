#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <tuple>
#include "/home/pathanasiadis/CoDiPack/include/codi.hpp"
#include <omp.h>
 //#include "kernels_codi.cu"

// Define the AD type using CoDiPack
using Real = codi::RealReverse;  // Reverse mode AD

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

template<typename T>
void boundary_conditions(const std::vector<T>& y, int xN, int yN, int zN, 
                         const std::vector<T>& u_inlet,
                         std::vector<T>& u, std::vector<T>& v, 
                         std::vector<T>& w, std::vector<T>& p) {
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;

    u.assign(totalSize, T(0.0));
    v.assign(totalSize, T(0.0));
    w.assign(totalSize, T(0.0));
    p.assign(totalSize, T(0.0));

    // --- Fill interior from y[] ---
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

    // --- Boundary conditions ---

    // X-direction boundaries
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

    // Y-direction boundaries
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

    // Z-direction boundaries
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

template<typename T> 
void uv_velocity(std::vector<T>& out, T Re, const std::vector<T>& y, 
                 int xN, int yN, int zN, const std::vector<T>& u_inlet,
                 T dx, T dy, T dz, 
                 std::vector<T>& u, std::vector<T>& v, 
                 std::vector<T>& w, std::vector<T>& p) {
    
    boundary_conditions(y, xN, yN, zN, u_inlet, u, v, w, p);
    
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int nCell = xN * yN * zN;
    
    out.resize(4 * nCell);

    for (int i = 1; i <= xN; ++i) {
        #pragma omp parallel for 
        for (int j = 1; j <= yN; ++j) {
            for (int k = 1; k <= zN; ++k) {
                int pos = (i - 1) * (yN * zN) + (j - 1) * zN + (k - 1);

                // U-momentum
                {
                    T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*u[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    T conv_y = 0.5 * dx * dz * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - u[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    T conv_z = 0.5 * dx * dy * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - u[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    T pres    = (dy * dz) * ( p[idx_3d(i+1,j,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    T diff = (1.0/Re) * (
                        (dy*dz/dx) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // V-momentum
                {
                    T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*v[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*v[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    T conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    T conv_z = 0.5 * dx * dy * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - v[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    T pres    = (dx * dz) * ( p[idx_3d(i,j+1,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    T diff = (1.0/Re) * (
                        (dy*dz/dx) * ( v[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ nCell + pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // W-momentum
                {
                    T conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*w[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*w[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    T conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*w[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*w[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    T conv_z = 0.5 * dx * dy * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    T pres    = (dx * dy) * ( p[idx_3d(i,j,k+1,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    T diff = (1.0/Re) * (
                        (dy*dz/dx) * ( w[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( w[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ 2*nCell + pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // Continuity
                {
                    T cont = (dy*dz/2.0) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)] )
                                + (dx*dz/2.0) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)] )
                                + (dx*dy/2.0) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)] );

                    out[ 3*nCell + pos ] = cont;
                }
            }
        }
    }
}



// Compute Jacobian using CoDiPack - Forward Mode (more reliable for dense Jacobian)
void compute_jacobian(const std::vector<float>& y_values,
                     const std::vector<float>& u_inlet_values,
                     std::vector<float>& residual,
                     std::vector<float>& jac_values,
                     std::vector<int>& row_indices,
                     std::vector<int>& col_indices,
                     int xN, int yN, int zN,
                     float dx, float dy, float dz, float Re) {
    
    int nCell = xN * yN * zN;
    int n = 4 * nCell;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int inlet_size = sizeY * sizeZ;
    
    // Clear output vectors
    jac_values.clear();
    row_indices.clear();
    col_indices.clear();
    residual.resize(n);
    
    std::cout << "Computing residual..." << std::endl;
    
    // First compute residual without AD
    {
        std::vector<Real> y_temp(n);
        std::vector<Real> u_inlet_temp(inlet_size);
        for (int i = 0; i < n; ++i) y_temp[i] = y_values[i];
        for (int i = 0; i < inlet_size; ++i) u_inlet_temp[i] = u_inlet_values[i];
        
        std::vector<Real> u, v, w, p;
        std::vector<Real> out;
        
        uv_velocity(out, Real(Re), y_temp, xN, yN, zN, u_inlet_temp, 
                   Real(dx), Real(dy), Real(dz), u, v, w, p);
        
        for (int i = 0; i < n; ++i) {
            residual[i] = codi::RealTraits::getPassiveValue(out[i]);
        }
    }
    
    std::cout << "Computing Jacobian using forward mode..." << std::endl;
    
    // Use forward mode: compute one column at a time by seeding each input
    using FwdReal = codi::RealForward;
    
    
    for (int col = 0; col < n; ++col) {
        if (col % 8 == 0) {
            std::cout << "  Processing column " << col << "/" << n << std::endl;
        }
        
        // Set up forward mode AD
        std::vector<FwdReal> y_fwd(n);
        std::vector<FwdReal> u_inlet_fwd(inlet_size);
        
        // Copy values
        for (int i = 0; i < n; ++i) {
            y_fwd[i] = y_values[i];
            y_fwd[i].setGradient(0.0);
        }
        
        // Seed the column input with derivative = 1.0
        y_fwd[col].setGradient(1.0);
        
        for (int i = 0; i < inlet_size; ++i) {
            u_inlet_fwd[i] = u_inlet_values[i];
            u_inlet_fwd[i].setGradient(0.0);
        }
        
        // Compute
        std::vector<FwdReal> u_fwd, v_fwd, w_fwd, p_fwd;
        std::vector<FwdReal> out_fwd;
        
        uv_velocity(out_fwd, FwdReal(Re), y_fwd, xN, yN, zN, u_inlet_fwd, 
                   FwdReal(dx), FwdReal(dy), FwdReal(dz), u_fwd, v_fwd, w_fwd, p_fwd);
        
        // Extract derivatives (this is the Jacobian column)
        for (int row = 0; row < n; ++row) {
            double deriv = out_fwd[row].getGradient();
            
            // Store non-zero entries
            if (std::abs(deriv) > 1e-15) {
                jac_values.push_back(deriv);
                row_indices.push_back(row);
                col_indices.push_back(col);
            }
        }
    }
    
    std::cout << "Jacobian computed: " << jac_values.size() << " non-zero entries" << std::endl;
    std::cout << "Sparsity: " << (100.0 * jac_values.size() / (n * n)) << "%" << std::endl;
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
    
    std::cout << "=== CFD Solver with CoDiPack AD ===" << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Grid: " << xN << "x" << yN << "x" << zN << " interior cells" << std::endl;
    std::cout << "Total grid (with ghosts): " << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
    
    // Coordinate arrays
    std::vector<float> xcoor(sizeX * sizeY * sizeZ);
    std::vector<float> ycoor(sizeX * sizeY * sizeZ);
    std::vector<float> zcoor(sizeX * sizeY * sizeZ);
    
    auto [dx, dy, dz] = coordinates(xcoor, ycoor, zcoor, xN, yN, zN, 
                                    (float)L, (float)M, (float)N);
    std::cout << "dx = " << dx << ", dy = " << dy << ", dz = " << dz << std::endl;
    
    // Inlet velocity profile (parabolic)
    std::vector<float> u_inlet(sizeY * sizeZ);
    for (int j = 0; j < sizeY; ++j) {
        for (int k = 0; k < sizeZ; ++k) {
            double yv = M * j / (sizeY - 1);
            double zv = N * k / (sizeZ - 1);
            u_inlet[j * sizeZ + k] = 16.0 * u0 * (yv / M) * (1.0 - yv / M) * (zv / N) * (1.0 - zv / N);
        }
    }
    
    std::cout << "Number of interior cells: " << actualNCells << std::endl;
    std::cout << "State variables: " << nVar << std::endl;
    
    // State vector initialization
    std::vector<float> y(nVar, 0.1);
    double blockSize = actualNCells;
    
    // Initialize u-velocity to u0
    for(int i = 0; i < blockSize; i++) {
        y[i] = u0;
    }
    
    // Initialize v, w, p to 0.1
    for(int i = blockSize; i < nVar; i++) {
        y[i] = 0.1;
    }
    
    std::cout << "\nInitial state vector statistics:" << std::endl;
    std::cout << "  u-velocity block (0-" << blockSize-1 << "): " << y[0] << " ... " << y[blockSize-1] << std::endl;
    std::cout << "  v-velocity block (" << blockSize << "-" << 2*blockSize-1 << "): " << y[blockSize] << std::endl;
    std::cout << "  w-velocity block (" << 2*blockSize << "-" << 3*blockSize-1 << "): " << y[2*blockSize] << std::endl;
    std::cout << "  pressure block (" << 3*blockSize << "-" << nVar-1 << "): " << y[3*blockSize] << std::endl;
    
    // Compute Jacobian
    std::vector<float> residual, jac_values;
    std::vector<int> row_indices, col_indices;
    
    std::cout << "\n=== Computing Jacobian ===" << std::endl;
    compute_jacobian(y, u_inlet, residual, jac_values, row_indices, col_indices,
                    xN, yN, zN, dx, dy, dz, Re);
    
    // Print first few Jacobian entries
    std::cout << "\nFirst 20 Jacobian entries:" << std::endl;
    for (int i = 0; i < std::min(20, (int)jac_values.size()); ++i) {
        std::cout << "J[" << row_indices[i] << "," << col_indices[i] << "] = " 
                  << jac_values[i] << std::endl;
    }
    
    // Print residual statistics
    std::cout << "\nResidual statistics:" << std::endl;
    float max_residual = 0.0;
    float sum_residual = 0.0;
    for (int i = 0; i < residual.size(); ++i) {
        sum_residual += std::abs(residual[i]);
        max_residual = std::max(max_residual, std::abs(residual[i]));
    }
    std::cout << "  Max |F|: " << max_residual << std::endl;
    std::cout << "  Mean |F|: " << sum_residual / residual.size() << std::endl;
    
    std::cout << "\nFirst 10 residual values:" << std::endl;
    for (int i = 0; i < std::max(0, (int)residual.size()); ++i) {
        std::cout << "F[" << i << "] = " << residual[i] << std::endl;
    }
    
    return 0;
}