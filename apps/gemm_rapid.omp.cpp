#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
//#include <format>
#include <stdio.h>
#include <unordered_map>
#include <omp.h>
#include "gemm.h"

int main()
{
  auto master_tt_start = std::chrono::high_resolution_clock::now();
  rapid_handle fam = rapid_initialize();

  uint64_t num_repeats = 4;

  std::vector<uint64_t> dim_sizes;
  for (uint64_t dim = 512; dim <= 8192; dim *= 2) {
//  for (uint64_t dim = 512; dim <= 1024; dim *= 2) {
    dim_sizes.push_back(dim);
  }
//  std::vector<double> gemm_no_tiling_avg_times;
//  std::vector<double> gemm_tiling_avg_times;

  std::vector<uint64_t> num_threads;
  for (uint64_t th = 1; th <= 16; th *= 2) {
//  for (uint64_t th = 1; th <= 2; th *= 2) {
    num_threads.push_back(th);
  }
  num_threads.push_back(28);

  /// No Tiling
  std::unordered_map<uint64_t, std::vector<double>> gemm_no_tiling_avg_time_table;
  for (uint64_t num_thd: num_threads) {
//    for (uint64_t dim_size = 512; dim_size <= 8192; dim_size *= 2) {
////  for (uint64_t dim_size = 512; dim_size <= 4096; dim_size *= 2) {
//      dim_sizes.push_back(dim_size);
    omp_set_num_threads(num_thd);
    for (uint64_t dim_size: dim_sizes) {
      std::vector<double> no_tiling_exe_times;

      for (uint64_t r_i = 0; r_i < num_repeats; ++r_i) {
        uint64_t A1 = dim_size;
        uint64_t A2 = dim_size;
        uint64_t B1 = A2;
        uint64_t B2 = dim_size;
        uint64_t C1 = A1;
        uint64_t C2 = B2;
        double *A = create_matrix_in_fam(fam, A1, A2, 1.1);
        double *B = create_matrix_in_fam(fam, B1, B2, 2.2);
        double *C = create_matrix_in_fam(fam, C1, C2);

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
        std::cout << "FAM no-tiling dim_size: " << dim_size << ", num_threads: " << num_thd << ", r_i: " << r_i << ", time_exe(s): "
                  << tt_duration.count() << std::endl;

//        print_matrix(C, C1, C2);
        no_tiling_exe_times.push_back(tt_duration.count());
        destroy_matrix_in_fam(fam, A);
        destroy_matrix_in_fam(fam, B);
        destroy_matrix_in_fam(fam, C);
      }
      /// Save all exe times
//    std::string no_tiling_filename = std::format("output.gemm.dram.no-tiling.size{}.log", dim_size);
      std::string no_tiling_filename;
      {
        std::stringstream ss;
        ss << "output.gemm.fam.no-tiling.matrix-dim-size-" << dim_size << ".thread-" << num_thd << ".log";
        no_tiling_filename = ss.str();
      }
      save_exe_times_into_file(no_tiling_filename, no_tiling_exe_times);
      /// Calculate average
      double no_tiling_avg_time = calculate_average_in_vector(no_tiling_exe_times);
//      gemm_no_tiling_avg_times.push_back(no_tiling_avg_time);
      gemm_no_tiling_avg_time_table[num_thd].push_back(no_tiling_avg_time);
    }
  }

  /// Tiling
  uint64_t tile_dim_size = 512;
  std::unordered_map<uint64_t, std::vector<double>> gemm_tiling_avg_time_table;
  for (uint64_t num_thd: num_threads) {
//    for (uint64_t dim_size = 512; dim_size <= 8192; dim_size *= 2) {
////  for (uint64_t dim_size = 512; dim_size <= 4096; dim_size *= 2) {
//      dim_sizes.push_back(dim_size);
    omp_set_num_threads(num_thd);
    for (uint64_t dim_size: dim_sizes) {
      std::vector<double> tiling_exe_times;

      for (uint64_t r_i = 0; r_i < num_repeats; ++r_i) {
        uint64_t A1 = dim_size;
        uint64_t A2 = dim_size;
        uint64_t A1_tile = tile_dim_size;
        uint64_t A2_tile = tile_dim_size;
        uint64_t B1 = A2;
        uint64_t B2 = dim_size;
        uint64_t B1_tile = A2_tile;
        uint64_t B2_tile = tile_dim_size;
        uint64_t C1 = A1;
        uint64_t C2 = B2;
        double *A = create_matrix_in_fam(fam, A1, A2, 1.1);
        double *B = create_matrix_in_fam(fam, B1, B2, 2.2);
        double *C = create_matrix_in_fam(fam, C1, C2);

        auto tt_start = std::chrono::high_resolution_clock::now();
        /// Kernel
//      gemm_v1_tiling(A, A1, A2, A1_tile, A2_tile,
//                     B, B1, B2, B1_tile, B2_tile,
//                     C);
        gemm_omp_v1_tiling(A, A1, A2, A1_tile, A2_tile,
                           B, B1, B2, B1_tile, B2_tile,
                           C);
        auto tt_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> tt_duration = tt_end - tt_start;
        std::cout << "FAM tiling dim_size: " << dim_size << ", num_threads: " << num_thd << ", r_i: " << r_i << ", time_exe(s): "
                  << tt_duration.count() << std::endl;

//        print_matrix(C, C1, C2);
        tiling_exe_times.push_back(tt_duration.count());
        destroy_matrix_in_fam(fam, A);
        destroy_matrix_in_fam(fam, B);
        destroy_matrix_in_fam(fam, C);
      }
      /// Save all exe times
//    std::string tiling_filename = std::format("output.gemm.dram.tiling.size{}.log", dim_size);
      std::string tiling_filename;
      {
        std::stringstream ss;
        ss << "output.gemm.fam.tiling.matrix-dim-size-" << dim_size << ".thread-" << num_thd << ".log";
        tiling_filename = ss.str();
      }
      save_exe_times_into_file(tiling_filename, tiling_exe_times);

      /// Calculate average
      double tiling_avg_time = calculate_average_in_vector(tiling_exe_times);
//      gemm_tiling_avg_times.push_back(tiling_avg_time);
      gemm_tiling_avg_time_table[num_thd].push_back(tiling_avg_time);
    }
  }

  /// Save results to a collection file

  { /// No-tiling file
    std::string collect_filename("output.gemm.fam.no-tiling.collection.csv");
    std::ofstream fout;
    fout.open(collect_filename);
    if (fout.is_open()) {
//    std::string header("Matrix_size,DRAM.No-Tiling(s),DRAM.Tiling(s)");
      std::string header("Matrix_size");
      for (uint64_t num_thd: num_threads) {
        header += ",RAPID.No-Tiling.Thread-" + std::to_string(num_thd);
      }
      fout << header << std::endl;
      for (uint64_t row_i = 0; row_i < dim_sizes.size(); ++row_i) {
//      fout << dim_sizes[row_i] << "," << gemm_no_tiling_avg_times[row_i] << "," << gemm_tiling_avg_times[row_i] << std::endl;
        fout << dim_sizes[row_i];
        for (uint64_t num_thd: num_threads) {
          fout << "," << gemm_no_tiling_avg_time_table[num_thd][row_i];
        }
        fout << std::endl;
      }

      std::cout << "Saved to file " << collect_filename << std::endl;
    } else {
      std::cerr << "Error: cannot open file " << collect_filename << std::endl;
    }
  }
  { /// Tiling file
    std::string collect_filename("output.gemm.fam.tiling.collection.csv");
    std::ofstream fout;
    fout.open(collect_filename);
    if (fout.is_open()) {
//    std::string header("Matrix_size,DRAM.No-Tiling(s),DRAM.Tiling(s)");
      std::string header("Matrix_size");
      for (uint64_t num_thd: num_threads) {
        header += ",RAPID.Tiling.Thread-" + std::to_string(num_thd);
      }
      fout << header << std::endl;
      for (uint64_t row_i = 0; row_i < dim_sizes.size(); ++row_i) {
//      fout << dim_sizes[row_i] << "," << gemm_no_tiling_avg_times[row_i] << "," << gemm_tiling_avg_times[row_i] << std::endl;
        fout << dim_sizes[row_i];
        for (uint64_t num_thd: num_threads) {
          fout << "," << gemm_tiling_avg_time_table[num_thd][row_i];
        }
        fout << std::endl;
      }

      std::cout << "Saved to file " << collect_filename << std::endl;
    } else {
      std::cerr << "Error: cannot open file " << collect_filename << std::endl;
    }
  }

  auto master_tt_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> master_tt_duration = master_tt_end - master_tt_start;
  std::cout << "TOTAL_EXE_TIME(S): " << master_tt_duration.count() << std::endl;

  return 0;
}
