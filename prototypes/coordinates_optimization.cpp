#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cmath> // For std::abs

// Standard 3D indexing: (i, j, k) -> linear index
inline int idx_3d(int i, int j, int k, int sizeY, int sizeZ) {
    return (i * sizeY + j) * sizeZ + k;
}

// --- YOUR ORIGINAL FUNCTION ---
std::tuple<double, double, double>
coordinates_original(std::vector<double> &xcoor, std::vector<double> &ycoor,
            std::vector<double> &zcoor, const int xN, const int yN,
            const int zN, const double L, const double M, const double N) {
    int xSize = xN + 2;
    int ySize = yN + 2;
    int zSize = zN + 2;

    for (int ix = 0; ix < xSize; ++ix) {
        for (int iy = 0; iy < ySize; ++iy) {
            for (int iz = 0; iz < zSize; ++iz) {
                int idx = idx_3d(ix, iy, iz, ySize, zSize);
                xcoor[idx] = L * ix / (xSize - 1);
                ycoor[idx] = M * iy / (ySize - 1);
                zcoor[idx] = N * iz / (zSize - 1);
            }
        }
    }

    double dx = L / (xSize - 1);
    double dy = M / (ySize - 1);
    double dz = N / (zSize - 1);

    return std::make_tuple(dx, dy, dz);
}

// --- OPTIMIZED FUNCTION ---
std::tuple<double, double, double>
coordinates_optimized(std::vector<double> &xcoor, std::vector<double> &ycoor,
            std::vector<double> &zcoor, const int xN, const int yN,
            const int zN, const double L, const double M, const double N) {
    
    int xSize = xN + 2;
    int ySize = yN + 2;
    int zSize = zN + 2;

    double dx = L / (xSize - 1);
    double dy = M / (ySize - 1);
    double dz = N / (zSize - 1);

    for (int ix = 0; ix < xSize; ++ix) {
        double current_x = ix * dx; 
        for (int iy = 0; iy < ySize; ++iy) {
            double current_y = iy * dy; 
            for (int iz = 0; iz < zSize; ++iz) {
                int idx = idx_3d(ix, iy, iz, ySize, zSize);
                xcoor[idx] = current_x;
                ycoor[idx] = current_y;
                zcoor[idx] = iz * dz; 
            }
        }
    }

    return std::make_tuple(dx, dy, dz);
}

int main() {
    // 1. Define grid parameters (3x3x3 grid)
    const int xN = 1, yN = 1, zN = 1; 
    const double L = 2.0, M = 2.0, N = 2.0; 

    int xSize = xN + 2;
    int ySize = yN + 2;
    int zSize = zN + 2;
    int totalNodes = xSize * ySize * zSize;

    // 2. Pre-allocate vectors for ORIGINAL function
    std::vector<double> x_orig(totalNodes), y_orig(totalNodes), z_orig(totalNodes);
    
    // 3. Pre-allocate vectors for OPTIMIZED function
    std::vector<double> x_opt(totalNodes), y_opt(totalNodes), z_opt(totalNodes);

    // 4. Run both functions
    auto [dx_orig, dy_orig, dz_orig] = coordinates_original(x_orig, y_orig, z_orig, xN, yN, zN, L, M, N);
    auto [dx_opt, dy_opt, dz_opt] = coordinates_optimized(x_opt, y_opt, z_opt, xN, yN, zN, L, M, N);

    // 5. Compare the results
    bool all_match = true;
    double tolerance = 1e-12; // Allowance for tiny floating-point math differences

    std::cout << "--- Comparing Step Sizes ---\n";
    std::cout << "Original: dx=" << dx_orig << ", dy=" << dy_orig << ", dz=" << dz_orig << "\n";
    std::cout << "Optimized: dx=" << dx_opt << ", dy=" << dy_opt << ", dz=" << dz_opt << "\n\n";

    std::cout << "--- Comparing Node Coordinates ---\n";
    std::cout << std::fixed << std::setprecision(2);
    
    for (int ix = 0; ix < xSize; ++ix) {
        for (int iy = 0; iy < ySize; ++iy) {
            for (int iz = 0; iz < zSize; ++iz) {
                int idx = idx_3d(ix, iy, iz, ySize, zSize);
                
                // Check if they are virtually identical
                if (std::abs(x_orig[idx] - x_opt[idx]) > tolerance ||
                    std::abs(y_orig[idx] - y_opt[idx]) > tolerance ||
                    std::abs(z_orig[idx] - z_opt[idx]) > tolerance) {
                    all_match = false;
                    std::cout << "MISMATCH at index " << idx << "!\n";
                }

                // Print the first few nodes so you can visually inspect them
                if (idx < 3) {
                    std::cout << "Node index " << idx << ":\n";
                    std::cout << "  Original -> [" << x_orig[idx] << ", " << y_orig[idx] << ", " << z_orig[idx] << "]\n";
                    std::cout << "  Optimized-> [" << x_opt[idx] << ", " << y_opt[idx] << ", " << z_opt[idx] << "]\n\n";
                }
            }
        }
    }

    if (all_match) {
        std::cout << "SUCCESS! Both functions produce the exact same grid arrays.\n";
    } else {
        std::cout << "WARNING! The arrays do not match.\n";
    }

    return 0;
}