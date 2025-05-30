//
// Created by zhen.peng@pnnl.gov on 5/28/25.
//

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstring>
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

//-----------
// Utilities
//-----------


