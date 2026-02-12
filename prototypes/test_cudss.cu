#include <cuda_runtime.h>
#include <cudss.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdint.h>

// ---------------------------------------------------------
// Error Handling Helper
// ---------------------------------------------------------
#define CHECK_CUDSS(func, msg) { \
    cudssStatus_t status = (func); \
    if (status != CUDSS_STATUS_SUCCESS) { \
        printf("cuDSS Error in %s at line %d. Status: %d\n", msg, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

// ---------------------------------------------------------
// GPU Solver
// ---------------------------------------------------------
void solve_system_float(
    int64_t n,                  
    int64_t nnz,                
    const int* d_row_offsets,   
    const int* d_col_indices,   
    const float* d_values,      
    const float* d_b,           
    float** d_x_out             
) {
    // 1. Allocate Result Vector X on GPU
    CHECK_CUDA( cudaMalloc((void**)d_x_out, n * sizeof(float)) )
    CHECK_CUDA( cudaMemset(*d_x_out, 0, n * sizeof(float)) )

    // 2. Initialize cuDSS Handles
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t solverData;
    cudssMatrix_t matA, vecB, vecX;

    CHECK_CUDSS( cudssCreate(&handle), "cudssCreate" )
    CHECK_CUDSS( cudssConfigCreate(&config), "cudssConfigCreate" )
    CHECK_CUDSS( cudssDataCreate(handle, &solverData), "cudssDataCreate" )

    // 3. Create Matrix Wrappers using YOUR specific signature
    // Notice the NULL argument for 'row_end_offsets' and direct enum passing
    CHECK_CUDSS( cudssMatrixCreateCsr(&matA, 
                                      n, n, nnz, 
                                      (void*)d_row_offsets, 
                                      NULL,                 // <--- As per your snippet
                                      (void*)d_col_indices, 
                                      (void*)d_values, 
                                      CUDA_R_32I,           // Index Type
                                      CUDA_R_32F,           // Value Type (Float)
                                      CUDSS_MTYPE_GENERAL, 
                                      CUDSS_MVIEW_FULL, 
                                      CUDSS_BASE_ZERO), "cudssMatrixCreateCsr" )

    // Vector X (Dense Float)
    CHECK_CUDSS( cudssMatrixCreateDn(&vecX, n, 1, n, 
                                     (void*)*d_x_out, 
                                     CUDA_R_32F,            // Value Type (Float)
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(X)" )

    // Vector B (Dense Float)
    CHECK_CUDSS( cudssMatrixCreateDn(&vecB, n, 1, n, 
                                     (void*)d_b, 
                                     CUDA_R_32F,            // Value Type (Float)
                                     CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(B)" )

    // 4. Run Solver Workflow
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solverData, matA, vecX, vecB), "Analysis" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solverData, matA, vecX, vecB), "Factorization" )
    CHECK_CUDSS( cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solverData, matA, vecX, vecB), "Solve" )

    // 5. Cleanup
    CHECK_CUDSS( cudssMatrixDestroy(matA), "Destroy A" )
    CHECK_CUDSS( cudssMatrixDestroy(vecB), "Destroy B" )
    CHECK_CUDSS( cudssMatrixDestroy(vecX), "Destroy X" )
    CHECK_CUDSS( cudssDataDestroy(handle, solverData), "Destroy Data" )
    CHECK_CUDSS( cudssConfigDestroy(config), "Destroy Config" )
    CHECK_CUDSS( cudssDestroy(handle), "Destroy Handle" )
}

// ---------------------------------------------------------
// Main Test
// ---------------------------------------------------------
int main() {
    printf("--- cuDSS Float Solver Test (Explained) ---\n");

    // -----------------------------------------------------
    // 1. The Mathematical Problem
    // -----------------------------------------------------
    // We want to solve Ax = b for x.
    //
    // Matrix A (3x3):
    //       Col 0   Col 1   Col 2
    // Row 0 [  4      1       0  ]
    // Row 1 [  1      4       1  ]
    // Row 2 [  0      1       4  ]
    //
    // Vector b: [ 5, 6, 5 ]
    //
    // Expected Solution x: [ 1, 1, 1 ]

    int64_t n = 3;   // Number of rows/cols
    int64_t nnz = 7; // Number of Non-Zero values in A

    // -----------------------------------------------------
    // 2. CSR Format Explanation
    // -----------------------------------------------------
    
    // VALUES (h_vals):
    // Simply list all non-zero numbers, reading left-to-right, top-to-bottom.
    // Row 0: 4, 1
    // Row 1: 1, 4, 1
    // Row 2: 1, 4
    std::vector<float> h_vals = { 
        4.0f, 1.0f,       // Row 0 values
        1.0f, 4.0f, 1.0f, // Row 1 values
        1.0f, 4.0f        // Row 2 values
    };

    // COLUMN INDICES (h_cols):
    // For every value above, which column is it in?
    // 4 is in Col 0, 1 is in Col 1
    // 1 is in Col 0, 4 is in Col 1, 1 is in Col 2
    // 1 is in Col 1, 4 is in Col 2
    std::vector<int> h_cols = { 
        0, 1,    // Row 0 indices
        0, 1, 2, // Row 1 indices
        1, 2     // Row 2 indices
    };

    // ROW OFFSETS (h_rows):
    // Where does each row START in the 'h_vals' array?
    // Row 0 starts at index 0.
    // Row 1 starts at index 2 (because Row 0 had 2 items).
    // Row 2 starts at index 5 (because Row 1 had 3 items: 2 + 3 = 5).
    // End   marker is index 7 (total items).
    std::vector<int> h_rows = { 
        0,  // Start of Row 0
        2,  // Start of Row 1
        5,  // Start of Row 2
        7   // End of Matrix (Total NNZ)
    };

    // RHS Vector b
    std::vector<float> h_b = { 5.0f, 6.0f, 5.0f };

    // -----------------------------------------------------
    // 3. GPU Memory Setup
    // -----------------------------------------------------
    int *d_rows, *d_cols;
    float *d_vals, *d_b;

    // Allocate memory on the GPU
    CHECK_CUDA( cudaMalloc(&d_rows, (n + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc(&d_cols, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc(&d_vals, nnz * sizeof(float)) )
    CHECK_CUDA( cudaMalloc(&d_b,    n * sizeof(float)) )

    // Copy data from CPU (Host) to GPU (Device)
    CHECK_CUDA( cudaMemcpy(d_rows, h_rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_b,    h_b.data(),    n * sizeof(float), cudaMemcpyHostToDevice) )

    // -----------------------------------------------------
    // 4. Solve System
    // -----------------------------------------------------
    float* d_x = nullptr; // Pointer to store result address
    printf("Solving system...\n");
    
    // Call the solver function (defined previously)
    solve_system_float(n, nnz, d_rows, d_cols, d_vals, d_b, &d_x);

    // -----------------------------------------------------
    // 5. Verify Results
    // -----------------------------------------------------
    std::vector<float> h_x(n);
    
    // Copy result back to CPU
    CHECK_CUDA( cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost) )

    printf("Solution X:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.2f\n", i, h_x[i]);
    }
    printf("(Expected: 1.00, 1.00, 1.00)\n");

    // -----------------------------------------------------
    // 6. Cleanup
    // -----------------------------------------------------
    cudaFree(d_rows); 
    cudaFree(d_cols); 
    cudaFree(d_vals); 
    cudaFree(d_b);
    cudaFree(d_x); // Free the result vector allocated by the solver

    return 0;
}