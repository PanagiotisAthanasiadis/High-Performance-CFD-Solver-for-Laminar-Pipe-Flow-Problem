#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <stdio.h>

// Error checking utility
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(func) { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("CUSPARSE API failed at line %d with error: %d\n", \
               __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * Adds two CSR matrices (A + B = C) using cuSPARSE.
 * * Performs C = alpha * A + beta * B
 * * Note: This function handles the allocation of C's arrays internally.
 * The caller is responsible for freeing d_C_row_offsets, d_C_columns, and d_C_values.
 */
void add_csr_cusparse(
    int m, int n,                          // Matrix dimensions (rows, cols)
    int nnzA,                              // Non-zeros in A
    const int* d_A_row_offsets, const int* d_A_columns, const float* d_A_values,
    int nnzB,                              // Non-zeros in B
    const int* d_B_row_offsets, const int* d_B_columns, const float* d_B_values,
    int* nnzC_out,                         // Output: Non-zeros in C
    int** d_C_row_offsets, int** d_C_columns, float** d_C_values // Output: Pointers to C data
) {
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Scalar multipliers (alpha = 1.0, beta = 1.0 for simple A+B)
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // 1. Create Matrix Descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));

    // 2. Allocate C Row Offsets (Always m + 1 size)
    CHECK_CUDA(cudaMalloc((void**)d_C_row_offsets, (m + 1) * sizeof(int)));

    // 3. Query Buffer Size
    // We need a buffer for the calculation. cuSPARSE calculates this for us.
    void* d_buffer = NULL;
    size_t bufferSize = 0;
    
    CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, d_A_values, *d_C_row_offsets, NULL, // C vals/cols are NULL for buffer query
        &bufferSize
    ));

    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // 4. Symbolic Phase: Calculate nnzC and C_row_offsets
    // This fills d_C_row_offsets and determines the total number of non-zeros.
    int nnzC = 0;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(
        handle, m, n,
        descrA, nnzA, d_A_row_offsets, d_A_columns,
        descrB, nnzB, d_B_row_offsets, d_B_columns,
        descrC, *d_C_row_offsets, &nnzC,d_buffer
    ));

    // Check for calculation errors (indicated by nnzC = -1 usually implies error in old API, 
    // but here we trust the Status return)
    *nnzC_out = nnzC;

    // 5. Allocate C Columns and Values
    // Now that we know nnzC, we can allocate the rest of the matrix.
    CHECK_CUDA(cudaMalloc((void**)d_C_columns, nnzC * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)d_C_values, nnzC * sizeof(float)));

    // 6. Numeric Phase: Fill C_columns and C_values
    // Performs the actual addition.
    CHECK_CUSPARSE(cusparseScsrgeam2(
        handle, m, n,
        &alpha, descrA, nnzA, d_A_values, d_A_row_offsets, d_A_columns,
        &beta,  descrB, nnzB, d_B_values, d_B_row_offsets, d_B_columns,
        descrC, *d_C_values, *d_C_row_offsets, *d_C_columns,
        d_buffer
    ));

    // 7. Cleanup
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}


void print_csr_matrix(int rows, int nnz, int* d_row_ptr, int* d_cols, float* d_vals) {
    // Allocate Host Memory
    int* h_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    int* h_cols    = (int*)malloc(nnz * sizeof(int));
    float* h_vals  = (float*)malloc(nnz * sizeof(float));

    // Copy from Device to Host
    CHECK_CUDA(cudaMemcpy(h_row_ptr, d_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cols, d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vals, d_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n--- Matrix Print (First 10 Rows) ---\n");
    printf("Format: Row [Start, End) -> (Col, Val)\n");

    // Loop through rows
    // (We limit to 10 to avoid flooding screen if matrix is huge)
    int print_limit = (rows > 10) ? 10 : rows;

    for (int i = 0; i < print_limit; i++) {
        int start = h_row_ptr[i];
        int end   = h_row_ptr[i+1];
        
        printf("Row %d [%d, %d): ", i, start, end);
        
        // Loop through elements in this row
        if (start == end) {
            printf("(Empty)");
        } else {
            for (int j = start; j < end; j++) {
                printf("(%d, %.6f) ", h_cols[j], h_vals[j]);
            }
        }
        printf("\n");
    }

    if (rows > 10) printf("... (truncated)\n");
    printf("------------------------------------\n");

    // Cleanup Host Memory
    free(h_row_ptr); free(h_cols); free(h_vals);
}


// Example Main to demonstrate usage
int main() {
    // Example 3x3 Matrices
    // A = [[1, 0, 2], [0, 0, 0], [3, 0, 4]] -> nnz = 4
    int h_A_rows[] = {0, 2, 2, 4};
    int h_A_cols[] = {0, 2, 0, 2};
    float h_A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};

    // B = [[5, 0, 0], [0, 0, 0], [0, 6, 0]] -> nnz = 2
    int h_B_rows[] = {0, 1, 1, 2};
    int h_B_cols[] = {0, 1};
    float h_B_vals[] = {5.0f, 6.0f};

    int m = 3, n = 3;
    int nnzA = 4, nnzB = 2;

    // Device allocations for Inputs
    int *d_A_rows, *d_A_cols, *d_B_rows, *d_B_cols;
    float *d_A_vals, *d_B_vals;

    cudaMalloc(&d_A_rows, (m+1)*sizeof(int));
    cudaMalloc(&d_A_cols, nnzA*sizeof(int));
    cudaMalloc(&d_A_vals, nnzA*sizeof(float));
    cudaMalloc(&d_B_rows, (m+1)*sizeof(int));
    cudaMalloc(&d_B_cols, nnzB*sizeof(int));
    cudaMalloc(&d_B_vals, nnzB*sizeof(float));

    cudaMemcpy(d_A_rows, h_A_rows, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_cols, h_A_cols, nnzA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_vals, h_A_vals, nnzA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_rows, h_B_rows, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_cols, h_B_cols, nnzB*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_vals, h_B_vals, nnzB*sizeof(float), cudaMemcpyHostToDevice);

    // Outputs
    int *d_C_rows = nullptr;
    int *d_C_cols = nullptr;
    float *d_C_vals = nullptr;
    int nnzC = 0;

    // --- CALL FUNCTION ---
    add_csr_cusparse(
        m, n, 
        nnzA, d_A_rows, d_A_cols, d_A_vals, 
        nnzB, d_B_rows, d_B_cols, d_B_vals, 
        &nnzC, &d_C_rows, &d_C_cols, &d_C_vals
    );

    // // Print Results
    // printf("Result NNZ: %d\n", nnzC);
    
    // int* h_C_rows = new int[m+1];
    // int* h_C_cols = new int[nnzC];
    // float* h_C_vals = new float[nnzC];

    // cudaMemcpy(h_C_rows, d_C_rows, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_C_cols, d_C_cols, nnzC*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_C_vals, d_C_vals, nnzC*sizeof(float), cudaMemcpyDeviceToHost);

    // printf("C Values: ");
    // for(int i=0; i<nnzC; i++) printf("%f ", h_C_vals[i]);
    // printf("\n");

    //A
    print_csr_matrix(m, nnzA, d_A_rows, d_A_cols, d_A_vals);
    //B
    print_csr_matrix(m, nnzB, d_B_rows, d_B_cols, d_B_vals);
    //Result matrix
    print_csr_matrix(m, nnzC, d_C_rows, d_C_cols, d_C_vals);

    // Clean up
    // delete[] h_C_rows; delete[] h_C_cols; delete[] h_C_vals;
    cudaFree(d_A_rows); cudaFree(d_A_cols); cudaFree(d_A_vals);
    cudaFree(d_B_rows); cudaFree(d_B_cols); cudaFree(d_B_vals);
    cudaFree(d_C_rows); cudaFree(d_C_cols); cudaFree(d_C_vals);

    return 0;
}