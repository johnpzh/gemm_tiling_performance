#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
//#include <format>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <omp.h>
#include "gemm.h"

int main(int argc, char *argv[])
{
  /// app <matrix-size> <num-threads>
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix-size> <num-threads>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  uint64_t matrix_size = std::strtoull(argv[1], nullptr, 0);
  uint64_t num_threads = std::strtoull(argv[2], nullptr, 0);

  auto master_tt_start = std::chrono::high_resolution_clock::now();
  rapid_handle fam = rapid_initialize();

  omp_set_num_threads(num_threads);
  uint64_t A1 = matrix_size;
  uint64_t A2 = matrix_size;
  uint64_t B1 = A2;
  uint64_t B2 = matrix_size;
  uint64_t C1 = A1;
  uint64_t C2 = B2;
  double *A = create_matrix_in_fam(fam, A1, A2, 1.1);
  double *B = create_matrix_in_fam(fam, B1, B2, 2.2);
  double *C = create_matrix_in_fam(fam, C1, C2);

  memset(C, 0, C1 * C2 * sizeof(double));

  auto tt_start = std::chrono::high_resolution_clock::now();
  /// Kernel
//      gemm_v0(A, A1, A2,
//              B, B1, B2,
//              C);
  gemm_omp_v0(A, A1, A2,
              B, B1, B2,
              C);
  auto tt_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tt_duration = tt_end - tt_start;
  std::cout << "FAM no-tiling matrix_size: " << matrix_size
            << ", num_threads: " << num_threads
            << ", time_exe(s): " << tt_duration.count() << std::endl;

//        print_matrix(C, C1, C2);
//        no_tiling_exe_times.push_back(tt_duration.count());
//        destroy_matrix_in_fam(fam, A);
//        destroy_matrix_in_fam(fam, B);
//        destroy_matrix_in_fam(fam, C);
  destroy_matrix_in_fam(fam, A);
  destroy_matrix_in_fam(fam, B);
  destroy_matrix_in_fam(fam, C);

  auto master_tt_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> master_tt_duration = master_tt_end - master_tt_start;
  std::cout << "TOTAL_EXE_TIME(S): " << master_tt_duration.count() << std::endl;

  return 0;
}