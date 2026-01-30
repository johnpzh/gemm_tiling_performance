//
// Created by zhen.peng@pnnl.gov on 5/28/25.
//

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <sys/mman.h>  // For mmap
#include <fcntl.h>     // For open
#include <unistd.h>    // For close, ftruncate
#include <omp.h>
#include "gemm.h"

//--------
// Matrix
//--------

double *create_matrix_in_dram(uint64_t num_rows, uint64_t num_cols, double val)
{
  uint64_t size = num_rows * num_cols;
  double *matrix = (double *) malloc(size * sizeof(double));
  if (val) {
    std::fill(matrix, matrix + size, val);
  } else {
    memset(matrix, 0, size * sizeof(double));
  }

  return matrix;
}

void destroy_matrix_in_dram(double *matrix)
{
  free(matrix);
}

double *create_matrix_in_fam(rapid_handle fam, uint64_t num_rows, uint64_t num_cols, double val)
{
  uint64_t size = num_rows * num_cols;
  double *matrix = (double *) rapid_malloc(fam, size * sizeof(double));
  if (val) {
    std::fill(matrix, matrix + size, val);
  } else {
    memset(matrix, 0, size * sizeof(double));
  }

  return matrix;
}

void destroy_matrix_in_fam(rapid_handle fam, double *matrix)
{
  rapid_free(fam, matrix);
}

double *create_matrix_in_disk(uint64_t num_rows, uint64_t num_cols, double val)
{
  uint64_t size = num_rows * num_cols;
  size_t file_size = size * sizeof(double);

  /// Create a random filename
  char template_name[] = "/tmp/matrix_XXXXXX.dat";  // XXXXXX will be replaced
  int fd = mkstemps(template_name, 4);  // Creates file and returns fd, 4 is the length of the extension
  // template_name now contains the actual filename
  if (fd == -1) { /* handle error */
    std::cerr << "Error: cannot create file " << template_name << "\n";
    exit(EXIT_FAILURE);
  }

  /// Truncate file to the desired size
  if (ftruncate(fd, file_size) == -1) {
    std::cerr << "Error: cannot truncate file " << template_name << " to " << file_size << "bytes\n";
    close(fd);
    exit(EXIT_FAILURE);
  }

  /// Map the file into memory
  double *matrix = (double *) mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (matrix == MAP_FAILED) {
    std::cerr << "Error: cannot map file " << template_name << " to memory\n";
    close(fd);
    exit(EXIT_FAILURE);
  }

  if (val) {
    std::fill(matrix, matrix + size, val);
  } else {
    memset(matrix, 0, file_size);
  }

  // close fd (mmap keeps the mapping valid)
  close(fd);
  unlink(template_name);  // file auto-deletes when unmapped

  return matrix;
}

void destroy_matrix_in_disk(double *matrix, uint64_t num_rows, uint64_t num_cols)
{
  // free(matrix);
  size_t file_size = num_rows * num_cols * sizeof(double);
  if (munmap(matrix, file_size) == -1) {
    std::cerr << "Error: cannot unmap file from memory\n";
    exit(EXIT_FAILURE);
  }
}

void print_matrix(double *matrix, uint64_t num_rows, uint64_t num_cols)
{
  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < num_cols; ++j) {
      std::cout << matrix[i * num_cols + j] << ", ";
    }
    std::cout << "\n";
  }
}

//--------------
// GEMM Kernels
//--------------

void gemm_v0(double *A, uint64_t A1, uint64_t A2,
             double *B, uint64_t B1, uint64_t B2,
             double *C)
{
  for (uint64_t i = 0; i < A1; ++i) {
    for (uint64_t k = 0; k < A2; ++k) {
      for (uint64_t j = 0; j < B2; ++j) {
        /// C[i,j] += A[i,k] * B[k,j];
        C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
      }
    }
  }
}

/// C[i,j] = A[i,k] * B[k,j]
/// ii-kk-jj, i-k-j
void gemm_v1_tiling(double *A, uint64_t A1, uint64_t A2, uint64_t A1_tile, uint64_t A2_tile,
                    double *B, uint64_t B1, uint64_t B2, uint64_t B1_tile, uint64_t B2_tile,
                    double *C)
{
  for (uint64_t ii = 0; ii < A1; ii += A1_tile) {
    uint64_t i_bound = std::min(ii + A1_tile, A1);
    for (uint64_t kk = 0; kk < A2; kk += A2_tile) {
      uint64_t k_bound = std::min(kk + A2_tile, A2);
      for (uint64_t jj = 0; jj < B2; jj += B2_tile) {
        uint64_t j_bound = std::min(jj + B2_tile, B2);
        /// Tile
        for (uint64_t i = ii; i < i_bound; ++i) {
          for (uint64_t k = kk; k < k_bound; ++k) {
            for (uint64_t j = jj; j < j_bound; ++j) {
              C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
            }
          }
        }
      }
    }
  }
}

/// C[i,j] = A[i,k] * B[k,j]
/// jj-kk-ii, i-j-k
void gemm_v2_tiling_disorder(double *A, uint64_t A1, uint64_t A2, uint64_t A1_tile, uint64_t A2_tile,
                    double *B, uint64_t B1, uint64_t B2, uint64_t B1_tile, uint64_t B2_tile,
                    double *C)
{
  for (uint64_t jj = 0; jj < B2; jj += B2_tile) {
    uint64_t j_bound = std::min(jj + B2_tile, B2);
    for (uint64_t kk = 0; kk < A2; kk += A2_tile) {
      uint64_t k_bound = std::min(kk + A2_tile, A2);
      for (uint64_t ii = 0; ii < A1; ii += A1_tile) {
        uint64_t i_bound = std::min(ii + A1_tile, A1);
        /// Tile
        for (uint64_t i = ii; i < i_bound; ++i) {
          for (uint64_t j = jj; j < j_bound; ++j) {
            for (uint64_t k = kk; k < k_bound; ++k) {
              C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
            }
          }
        }
      }
    }
  }
}

/// C[i,j] = A[i,k] * B[k,j]
/// ii-kk-jj, i-j-k
void gemm_v3_tiling_disorder(double *A, uint64_t A1, uint64_t A2, uint64_t A1_tile, uint64_t A2_tile,
                    double *B, uint64_t B1, uint64_t B2, uint64_t B1_tile, uint64_t B2_tile,
                    double *C)
{
  for (uint64_t ii = 0; ii < A1; ii += A1_tile) {
    uint64_t i_bound = std::min(ii + A1_tile, A1);
    for (uint64_t kk = 0; kk < A2; kk += A2_tile) {
      uint64_t k_bound = std::min(kk + A2_tile, A2);
      for (uint64_t jj = 0; jj < B2; jj += B2_tile) {
        uint64_t j_bound = std::min(jj + B2_tile, B2);
        /// Tile
        for (uint64_t i = ii; i < i_bound; ++i) {
          for (uint64_t j = jj; j < j_bound; ++j) {
            for (uint64_t k = kk; k < k_bound; ++k) {
              C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
            }
          }
        }
      }
    }
  }
}

/// Parallel

void gemm_omp_v0(double *A, uint64_t A1, uint64_t A2,
                 double *B, uint64_t B1, uint64_t B2,
                 double *C)
{
  #pragma omp parallel for
  for (uint64_t i = 0; i < A1; ++i) {
    for (uint64_t k = 0; k < A2; ++k) {
      for (uint64_t j = 0; j < B2; ++j) {
        /// C[i,j] += A[i,k] * B[k,j];
        C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
      }
    }
  }
}

/// C[i,j] = A[i,k] * B[k,j]
/// ii-kk-jj, i-k-j
void gemm_omp_v1_tiling(double *A, uint64_t A1, uint64_t A2, uint64_t A1_tile, uint64_t A2_tile,
                        double *B, uint64_t B1, uint64_t B2, uint64_t B1_tile, uint64_t B2_tile,
                        double *C)
{
  #pragma omp parallel for
  for (uint64_t ii = 0; ii < A1; ii += A1_tile) {
    uint64_t i_bound = std::min(ii + A1_tile, A1);
    for (uint64_t kk = 0; kk < A2; kk += A2_tile) {
      uint64_t k_bound = std::min(kk + A2_tile, A2);
      for (uint64_t jj = 0; jj < B2; jj += B2_tile) {
        uint64_t j_bound = std::min(jj + B2_tile, B2);
        /// Tile
        for (uint64_t i = ii; i < i_bound; ++i) {
          for (uint64_t k = kk; k < k_bound; ++k) {
            for (uint64_t j = jj; j < j_bound; ++j) {
              C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];
            }
          }
        }
      }
    }
  }
}

void gemm_omp_v2_disk(double *A, uint64_t A1, uint64_t A2,
                      double *B, uint64_t B1, uint64_t B2,
                      double *C)
{
  size_t page_size = sysconf(_SC_PAGESIZE);
  size_t row_bytes = B2 * sizeof(double);
  double *row_start = C;

  size_t matrix_bytes = A1 * B2 * sizeof(double);

  #pragma omp parallel for
  for (uint64_t i = 0; i < A1; ++i) {
    for (uint64_t k = 0; k < A2; ++k) {
      for (uint64_t j = 0; j < B2; ++j) {
        /// C[i,j] += A[i,k] * B[k,j];
        C[i * B2 + j] += A[i * A2 + k] * B[k * B2 + j];

        /// Too Heavy
//        uintptr_t aligned_addr = (uintptr_t) row_start & ~(page_size - 1);
//        if (msync((void *) aligned_addr, row_bytes, MS_SYNC) == -1) {
//        double *entry_start = C + i * B2 + j;
//        uintptr_t aligned_addr = (uintptr_t) entry_start & ~(page_size - 1);
//        if (msync((void *) aligned_addr, page_size, MS_SYNC) == -1) {
//          std::cerr << "Error: cannot sync memory\n";
//          exit(EXIT_FAILURE);
//        }
      }
      /// Is this too much heavy?
//      uintptr_t aligned_addr = (uintptr_t) C & ~(page_size - 1);
//      if (msync((void *) aligned_addr, matrix_bytes, MS_SYNC) == -1) {
//        std::cerr << "Error: cannot sync memory\n";
//        exit(EXIT_FAILURE);
//      }

      /// Lighter
//      uintptr_t aligned_addr = (uintptr_t) row_start & ~(page_size - 1);
//      if (msync((void *) aligned_addr, row_bytes, MS_SYNC) == -1) {
//        std::cerr << "Error: cannot sync memory\n";
//        exit(EXIT_FAILURE);
//      }
    }
    row_start += B2;

    for (uint64_t row_i = 0; row_i < A1; ++row_i) {
      uint64_t divides = 32;
      uint64_t width = (A1 + divides - 1) / divides;
      for (uint64_t bucket_i = 0; bucket_i < B2; bucket_i += width) {
        double *pointer = C + row_i * B2 + bucket_i;
        uintptr_t aligned_addr = (uintptr_t) pointer & ~(page_size - 1);
        if (msync((void *) aligned_addr, page_size, MS_SYNC) == -1) {
          std::cerr << "Error: cannot sync memory\n";
          exit(EXIT_FAILURE);
        }
      }
    }
  }
}

//-----------
// Utilities
//-----------