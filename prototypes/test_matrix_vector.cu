#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <vector>

// --- Macros for Error Handling ---
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(func) { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE Error at line %d: %s\n", __LINE__, cusparseGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}


void scale_and_multiply_on_gpu(
    int rows, int cols, int nnz,
    const int* d_col_offsets,  // Input: Device Pointer
    const int* d_row_indices,  // Input: Device Pointer
    const float* d_values,     // Input: Device Pointer
    const float* d_x,          // Input: Device Pointer
    float alpha,               // Scalar
    float** d_y_out            // Output: Address of pointer to allocate
) {
    // 1. Allocate Result Memory on GPU
    // We dereference d_y_out to set the caller's pointer
    CHECK_CUDA( cudaMalloc((void**)d_y_out, rows * sizeof(float)) )
    
    // Initialize result to 0 (optional but good practice for safety)
    CHECK_CUDA( cudaMemset(*d_y_out, 0, rows * sizeof(float)) )

    // 2. Create cuSPARSE Context
    cusparseHandle_t handle;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // 3. Create Descriptors using the DEVICE pointers passed in
    CHECK_CUSPARSE( cusparseCreateCsc(&matA, rows, cols, nnz,
                                      (void*)d_col_offsets, (void*)d_row_indices, (void*)d_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, cols, (void*)d_x, CUDA_R_32F) )
    
    // Use the newly allocated pointer (*d_y_out) for the Y descriptor
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, rows, *d_y_out, CUDA_R_32F) )

    // 4. Allocate Workspace
    void* d_buffer = nullptr;
    size_t bufferSize = 0;
    float beta = 0.0f; // We are overwriting Y, not accumulating

    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )

    CHECK_CUDA( cudaMalloc(&d_buffer, bufferSize) )

    // 5. Execute Operation (GPU only)
    CHECK_CUSPARSE( cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer) )

    // 6. Cleanup Local Resources
    // Note: We DO NOT free d_col_offsets, d_x, or *d_y_out because the caller owns them.
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(d_buffer) )
}

// --- Main Test ---
int main() {
    // ---------------------------------------------------------
    // Setup Phase: Prepare data on GPU to simulate your real workflow
    // ---------------------------------------------------------
    int rows = 3, cols = 3, nnz = 3;
    
    // Host Data
    std::vector<int>   h_cols = { 0, 1, 2, 3 };
    std::vector<int>   h_rows = { 0, 1, 2 };
    std::vector<float> h_vals = { 1.0f, 3.0f, 1.0f }; // Identity Matrix
    std::vector<float> h_x    = { 10.0f, 20.0f, 30.0f };

    // Device Pointers
    int *d_cols, *d_rows;
    float *d_vals, *d_x;

    // Allocate & Copy (Simulating that data is ALREADY on GPU)
    CHECK_CUDA( cudaMalloc(&d_cols, (cols + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc(&d_rows, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc(&d_vals, nnz * sizeof(float)) )
    CHECK_CUDA( cudaMalloc(&d_x,    cols * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(d_cols, h_cols.data(), (cols + 1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_rows, h_rows.data(), nnz * sizeof(int),        cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float),      cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_x,    h_x.data(),    cols * sizeof(float),     cudaMemcpyHostToDevice) )

    // ---------------------------------------------------------
    // The Actual Function Call
    // ---------------------------------------------------------
    float *d_result = nullptr; // This is NULL now
    float alpha = 2.0f;        // Scale factor

    printf("Calling GPU function...\n");
    
    // Pass the ADDRESS of d_result (&d_result) so the function can allocate it
    scale_and_multiply_on_gpu(
        rows, cols, nnz,
        d_cols, d_rows, d_vals, d_x,
        alpha,
        &d_result // Pass address
    );

    printf("Function returned. Result is stored at GPU address: %p\n", d_result);

    // ---------------------------------------------------------
    // Verification (Copy back just to prove it worked)
    // ---------------------------------------------------------
    std::vector<float> h_result(rows);
    CHECK_CUDA( cudaMemcpy(h_result.data(), d_result, rows * sizeof(float), cudaMemcpyDeviceToHost) )

    printf("Result: ");
    for (float v : h_result) printf("%.1f ", v); 
    printf("\n"); // Expected: 2.0 * [10, 20, 30] = [20, 40, 60]

    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------
    cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_result); // Don't forget to free the result!

    return 0;
}