cmake_minimum_required(VERSION 3.18)
project(NavierStokesTests LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

include(CTest)


# Find all test files
file(GLOB TEST_FILES "main_test.cpp" "main_test_small.cpp")

foreach(test_file ${TEST_FILES})
    # Get the test name from the file name
    get_filename_component(test_name ${test_file} NAME_WE)

    # Add an executable for each test file
    add_executable(${test_name} ${test_file})

    # Link the test executable with the main library and Catch2
    target_link_libraries(${test_name} PRIVATE navier_stokes_lib Catch2::Catch2WithMain)

    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Add the test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
