cmake_minimum_required(VERSION 3.18)
project(NavierStokesGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include ${CUDAToolkit_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(navier_stokes_gpu ${SOURCES})

set_target_properties(navier_stokes_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(navier_stokes_gpu PRIVATE
    cusparse
    cusolver
    cublas
    OpenMP::OpenMP_CXX
)

# CXX flags
target_compile_options(navier_stokes_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -g
    >
)

# CUDA flags
target_compile_options(navier_stokes_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --ptxas-options=-v
        -use_fast_math
        -O3
        --generate-line-info
        -arch=sm_61
        --default-stream=per-thread
        --extended-lambda
        -Xcompiler
        "-O3"
        -Xcompiler
        "-g"
        -Xcompiler
        "-fopenmp"
    >
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)