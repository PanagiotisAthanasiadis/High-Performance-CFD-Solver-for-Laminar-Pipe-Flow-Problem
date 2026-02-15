#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------
// 1. Error Checking Macros
// ---------------------------------------------------------

// Checks standard CUDA runtime calls (cudaMalloc, cudaMemcpy, etc.)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error:\n" \
                      << "  File:     " << __FILE__ << "\n" \
                      << "  Line:     " << __LINE__ << "\n" \
                      << "  Function: " << #call << "\n" \
                      << "  Message:  " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Checks cuBLAS library calls (cublasCreate, cublasSdot, etc.)
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error:\n" \
                      << "  File:     " << __FILE__ << "\n" \
                      << "  Line:     " << __LINE__ << "\n" \
                      << "  Function: " << #call << "\n" \
                      << "  Status:   " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------------------------------------------------
// 2. The Math Function
// ---------------------------------------------------------

// Computes: 0.5 * sum(x_i ^ 2) using cuBLAS single-precision dot product
float compute_scaled_sq_sum(cublasHandle_t handle, const float* d_array, int N) {
    float result = 0.0f;

    // Ensure cuBLAS writes the scalar result directly to our CPU variable ('result')
    // This is the default behavior, but it's best practice to be explicit.
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    // Perform the self-dot product: (d_array dot d_array)
    // parameters: handle, N, X, strideX, Y, strideY, output_result
    CUBLAS_CHECK(cublasSdot(handle, N, d_array, 1, d_array, 1, &result));

    // Multiply by 0.5 on the CPU (cheaper than doing it on the GPU)
    return result * 0.5f;
}

// ---------------------------------------------------------
// 3. Example Usage
// ---------------------------------------------------------

int main() {
    int N = 10000000; // 1 million elements
    size_t bytes = N * sizeof(float);

    // 1. Create mock data on CPU (e.g., all 2.0s)
    std::vector<float> h_array(N, 2.0f);

    // 2. Allocate and copy to GPU
    float* d_array;
    CUDA_CHECK(cudaMalloc(&d_array, bytes));
    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), bytes, cudaMemcpyHostToDevice));

    // 3. Initialize cuBLAS Handle (Do this ONCE per thread/application)
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 4. Run your function!
    float final_ans = compute_scaled_sq_sum(handle, d_array, N);
    
    std::cout << "Computed Result: " << final_ans << std::endl;
    // Math check: 2.0^2 = 4.0. 
    // Sum of 1,000,000 4.0s = 4,000,000. 
    // Multiplied by 0.5 = 2,000,000.

    // 5. Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}