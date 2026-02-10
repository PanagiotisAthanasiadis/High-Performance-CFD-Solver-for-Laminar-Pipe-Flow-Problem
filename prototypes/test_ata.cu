#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>

// -------------------------------------------------------------------------
// Helper Macros
// -------------------------------------------------------------------------
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " - code " << err << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - code " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// -------------------------------------------------------------------------
// Your Function (Pasted exactly as provided, with CUDA_CHECK added)
// -------------------------------------------------------------------------
void compute_AtA_debug(
    int* d_rows_coo,
    int* d_cols_coo,
    float* d_vals_coo,
    int nnz,
    int num_rows,
    int num_cols,
    int** d_result_rows,
    int** d_result_cols,
    float** d_result_vals,
    int* result_nnz)
{
    // 0. Setup Input Data (Convert COO matrix to CSR)
    cusparseHandle_t handle; 
    CUSPARSE_CHECK(cusparseCreate(&handle));
    
    // Convert COO to CSR for A
    int *d_csrRowPtrA, *d_csrColIndA;
    float *d_csrValA;
    
    CUDA_CHECK(cudaMalloc(&d_csrRowPtrA, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndA, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValA, nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseXcoo2csr(handle, d_rows_coo, nnz, num_rows, d_csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CHECK(cudaMemcpy(d_csrColIndA, d_cols_coo, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrValA, d_vals_coo, nnz * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // 1. Explicit Transpose: Generate A^T
    int *d_AT_csrOffsets, *d_AT_columns;
    float *d_AT_values;
    
    CUDA_CHECK(cudaMalloc(&d_AT_csrOffsets, (num_rows + 1) * sizeof(int))); 
    CUDA_CHECK(cudaMalloc(&d_AT_columns, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_AT_values, nnz * sizeof(float)));

    size_t transposeBufferSize = 0;
    void* d_transposeBuffer = NULL;

    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_AT_values, d_AT_csrOffsets, d_AT_columns, 
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        &transposeBufferSize
    ));

    CUDA_CHECK(cudaMalloc(&d_transposeBuffer, transposeBufferSize));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, num_rows, num_cols, nnz,
        d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_AT_values, d_AT_csrOffsets, d_AT_columns,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        d_transposeBuffer
    ));

    // 2. Initialize cuSPARSE and Matrix Descriptors
    cusparseSpMatDescr_t matA, matAt, matC;
   
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                      d_csrRowPtrA, d_csrColIndA, d_csrValA,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CUSPARSE_CHECK(cusparseCreateCsr(&matAt, num_rows, num_cols, nnz,
                                     d_AT_csrOffsets, d_AT_columns, d_AT_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateCsr(&matC, num_cols, num_cols, 0,
                      nullptr, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    // 3. SpGEMM Setup
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // 4. Work Estimation
    size_t bufferSize1 = 0;
    void* dBuffer1 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA,opB,
                                   &alpha, matAt, matA, &beta, matC,
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, nullptr));
    
    if (bufferSize1 > 0) CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
    
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle, opA,opB,
                                   &alpha, matAt, matA, &beta, matC,
                                   CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                   spgemmDesc, &bufferSize1, dBuffer1));
    
    // 5. Compute Structure 
    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, nullptr));
    
    if (bufferSize2 > 0) CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
    
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle, opA,opB,
                          &alpha, matAt, matA, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                          spgemmDesc, &bufferSize2, dBuffer2));
    
    // 6. Allocate C and Copy Results
    int64_t C_num_rows, C_num_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));
    
    std::cout << "Computed Matrix C: " << C_num_rows << "x" << C_num_cols 
              << " with " << C_nnz << " non-zeros." << std::endl;
    
    if (C_nnz == 0) {
        *result_nnz = 0; return; // Simplified error handling for brevity
    }
    
    int *d_csrRowPtrC, *d_csrColIndC;
    float *d_csrValC;
    
    CUDA_CHECK(cudaMalloc(&d_csrRowPtrC, (C_num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIndC, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrValC, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));
    
    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, opA,opB,
                       &alpha, matAt, matA, &beta, matC,
                       CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    
    CUDA_CHECK(cudaMalloc(d_result_rows, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_result_cols, C_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_result_vals, C_nnz * sizeof(float)));
    
    CUSPARSE_CHECK(cusparseXcsr2coo(handle, d_csrRowPtrC, C_nnz, C_num_rows,
                     *d_result_rows, CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CHECK(cudaMemcpy(*d_result_cols, d_csrColIndC, C_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(*d_result_vals, d_csrValC, C_nnz * sizeof(float), cudaMemcpyDeviceToDevice));
    
    *result_nnz = C_nnz;
    
    // Printing from inside the function as requested
    int* h_result_rows = new int[(int)C_nnz];
    int* h_result_cols = new int[(int)C_nnz];
    float* h_result_vals = new float[(int)C_nnz];
    
    CUDA_CHECK(cudaMemcpy(h_result_rows, *d_result_rows, C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_cols, *d_result_cols, C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_vals, *d_result_vals,C_nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "--- GPU Results (Printed from function) ---" << std::endl;
    for (int i = 0; i < (int)C_nnz; i++) {
        std::cout << "(" << h_result_rows[i] << ", " << h_result_cols[i] << ") = " << h_result_vals[i] << std::endl;
    }
    std::cout << "-------------------------------------------" << std::endl;

    // Cleanup (omitted specific cleanups for brevity, OS will reclaim on exit)
}


// -------------------------------------------------------------------------
// MAIN TEST
// -------------------------------------------------------------------------
int main() {
    // Defines matrix A: 
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    
    int num_rows = 3;
    int num_cols = 3;
    int nnz = 5;

    // Host Data (COO Format)
    std::vector<int> h_rows = {0, 0, 1, 2, 2};
    std::vector<int> h_cols = {0, 2, 1, 0, 2};
    std::vector<float> h_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Device Allocations
    int *d_rows, *d_cols;
    float *d_vals;
    
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_rows, h_rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Pointers for Result
    int *d_res_rows = nullptr;
    int *d_res_cols = nullptr;
    float *d_res_vals = nullptr;
    int res_nnz = 0;

    std::cout << "Starting computation..." << std::endl;

    // Call Function
    compute_AtA_debug(
        d_rows, d_cols, d_vals, 
        nnz, num_rows, num_cols, 
        &d_res_rows, &d_res_cols, &d_res_vals, 
        &res_nnz
    );

    std::cout << "Computation finished. Verifying NNZ..." << std::endl;
    
    // Expected NNZ for this specific AtA is 5
    if (res_nnz == 5) {
        std::cout << "SUCCESS: NNZ count matches expectation (5)." << std::endl;
    } else {
        std::cout << "FAILURE: NNZ count mismatch! Expected 5, got " << res_nnz << std::endl;
    }

    return 0;
}