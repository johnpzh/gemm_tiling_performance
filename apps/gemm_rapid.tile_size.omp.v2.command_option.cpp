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
  /// app <matrix-size> <tile-size> <num-threads>
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <matrix-size> <tile-size> <num-threads>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  uint64_t matrix_size = std::strtoull(argv[1], nullptr, 0);
  uint64_t tile_size = std::strtoull(argv[2], nullptr, 0);
  uint64_t num_threads = std::strtoull(argv[3], nullptr, 0);

  auto master_tt_start = std::chrono::high_resolution_clock::now();
  rapid_handle fam = rapid_initialize();

  uint64_t A1 = matrix_size;
  uint64_t A2 = matrix_size;
  uint64_t B1 = A2;
  uint64_t B2 = matrix_size;
  uint64_t C1 = A1;
  uint64_t C2 = B2;
  double *A = create_matrix_in_fam(fam, A1, A2, 1.1);
  double *B = create_matrix_in_fam(fam, B1, B2, 2.2);
  double *C = create_matrix_in_fam(fam, C1, C2);

  uint64_t A1_tile = tile_size;
  uint64_t A2_tile = tile_size;
  uint64_t B1_tile = A2_tile;
  uint64_t B2_tile = tile_size;

  omp_set_num_threads(num_threads);

  memset(C, 0, C1 * C2 * sizeof(double));

  auto tt_start = std::chrono::high_resolution_clock::now();
  /// Kernel
  gemm_omp_v1_tiling(A, A1, A2, A1_tile, A2_tile,
                     B, B1, B2, B1_tile, B2_tile,
                     C);
  auto tt_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tt_duration = tt_end - tt_start;
  double random_element = get_random_element(C, C1, C2);
  std::cout << "FAM tiling matrix_size: " << matrix_size
            << ", tile_dim_size: " << tile_size
            << ", num_threads: " << num_threads
            << ", random_element: " << random_element
            << ", time_exe(s): " << tt_duration.count() << std::endl;

  destroy_matrix_in_fam(fam, A);
  destroy_matrix_in_fam(fam, B);
  destroy_matrix_in_fam(fam, C);

  auto master_tt_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> master_tt_duration = master_tt_end - master_tt_start;
  std::cout << "TOTAL_EXE_TIME(S): " << master_tt_duration.count() << std::endl;

  return 0;
}
