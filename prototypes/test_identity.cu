#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// --- Helper Macro for Error Checking ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// -------------------------------------------------------------------------
// 1. Kernel to Generate Identity Matrix (CSR Format)
// -------------------------------------------------------------------------
__global__ void generate_identity_csr_kernel(int N, float alpha, 
                                             int* row_offsets, int* cols, float* vals) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // We process N elements (the diagonal)
    if (idx < N) {
        // 1. Column Index: Simple sequence 0, 1, 2...
        cols[idx] = idx;
        
        // 2. Value: The scalar alpha
        vals[idx] = alpha;
        
        // 3. Row Offset: Simple sequence 0, 1, 2...
        // Because every row has exactly 1 element.
        row_offsets[idx] = idx;
    }
    
    // Handle the very last element of the row pointer array (Size is N+1)
    // It must equal the total NNZ (which is N)
    if (idx == N) {
        row_offsets[idx] = N;
    }
}

// -------------------------------------------------------------------------
// 2. Helper to Print CSR Matrix
// -------------------------------------------------------------------------
void print_csr_matrix(int rows, int nnz, int* d_row_ptr, int* d_cols, float* d_vals) {
    // Allocate Host Memory
    int* h_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    int* h_cols    = (int*)malloc(nnz * sizeof(int));
    float* h_vals  = (float*)malloc(nnz * sizeof(float));

    // Copy from Device to Host
    CUDA_CHECK(cudaMemcpy(h_row_ptr, d_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols, d_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals, d_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost));

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
                printf("(%d, %.1f) ", h_cols[j], h_vals[j]);
            }
        }
        printf("\n");
    }

    if (rows > 10) printf("... (truncated)\n");
    printf("------------------------------------\n");

    // Cleanup Host Memory
    free(h_row_ptr); free(h_cols); free(h_vals);
}

// -------------------------------------------------------------------------
// 3. Main Test
// -------------------------------------------------------------------------
int main() {
    // Settings
    int N = 10;            // 10x10 Matrix
    float alpha = 5.0f;    // Values will be 5.0
    int nnz = N;           // Identity has N non-zeros

    printf("Generating %dx%d Identity Matrix scaled by %.1f...\n", N, N, alpha);

    // Allocate Device Memory
    int *d_row_ptr, *d_cols;
    float *d_vals;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(float)));

    // Launch Kernel
    // We need enough threads to cover N+1 elements (for row_ptr)
    int blockSize = 256;
    int gridSize = (N + 1 + blockSize - 1) / blockSize;
    
    generate_identity_csr_kernel<<<gridSize, blockSize>>>(N, alpha, d_row_ptr, d_cols, d_vals);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print Result
    print_csr_matrix(N, nnz, d_row_ptr, d_cols, d_vals);

    // Cleanup Device Memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_vals));

    printf("Test Complete.\n");
    return 0;
}