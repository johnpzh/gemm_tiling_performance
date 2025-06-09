cmake_minimum_required(VERSION 3.20)
project(tiling_gemm)

set(CMAKE_CXX_STANDARD 20)

# Rapid Library
set(CMAKE_PREFIX_PATH "/share/micron/rapid/install/gcc-debug")
find_package(rapid CONFIG REQUIRED)

include_directories("include")
include_directories("/share/micron/rapid/install/gcc-release/include")
link_directories("/share/micron/rapid/install/gcc-release/lib64")

add_library(simplegemm src/gemm.cpp)
target_link_libraries(simplegemm PUBLIC rapid::rapid)

# GEMM Test correctness
add_executable(gemm_test_correct apps/gemm_test_correct.cpp)
target_link_libraries(gemm_test_correct PRIVATE simplegemm)

# GEMM DRAM
add_executable(gemm_dram apps/gemm_dram.cpp)
target_link_libraries(gemm_dram PRIVATE simplegemm)

# GEMM RAPID
add_executable(gemm_rapid apps/gemm_rapid.cpp)
target_link_libraries(gemm_rapid PRIVATE simplegemm)

add_executable(gemm_rapid.v2.inorder apps/gemm_rapid.v2.inorder.cpp)
target_link_libraries(gemm_rapid.v2.inorder PRIVATE simplegemm)

add_executable(gemm_rapid.v3.inorder apps/gemm_rapid.v3.inorder.cpp)
target_link_libraries(gemm_rapid.v3.inorder PRIVATE simplegemm)
#