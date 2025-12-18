cmake_minimum_required(VERSION 3.18)



project(NavierStokesGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include(CTest)
enable_testing()



include_directories(${CMAKE_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

file(GLOB_RECURSE NAVIER_STOKES_CU_SOURCES "src/*.cu")

add_library(navier_stokes_lib STATIC ${NAVIER_STOKES_CU_SOURCES})

set_target_properties(navier_stokes_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(navier_stokes_lib PRIVATE
    cusparse
    cusolver
    cublas
    OpenMP::OpenMP_CXX
)

target_compile_options(navier_stokes_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -g
    >
)

target_compile_options(navier_stokes_lib PRIVATE
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

add_executable(navier_stokes_gpu src/main.cpp)
target_link_libraries(navier_stokes_gpu PRIVATE navier_stokes_lib)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(tests)