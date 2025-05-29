cmake_minimum_required(VERSION 3.20)
project(tiling_gemm)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_library(libgemm src/gemm.cpp)

add_executable(gemm_dram apps/gemm_dram.cpp)
target_link_libraries(gemm_dram PRIVATE libgemm)
