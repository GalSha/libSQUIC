/* squic.hpp
 *
 * Copyright (C) Matthias Bollhoefer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef squic_hpp
#define squic_hpp

#define SQUIC_VER 1.1

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Standard General Utilities Library
//#include <assert.h>

#include <cmath>
#include <cstdio>
#include <iomanip>  // Precision
#include <iostream>

// Data structures, algos and numerics
#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Sleep, time, fprintf etc...
#include <stdio.h>  // printf
#include <time.h>
#include <unistd.h>

#include <cstddef>  // sleep
#include <cstdlib>

// IO
#include <fstream>
#include <iostream>

// Parallel
#include <omp.h>

#include "cholmod.h"
#include "long_integer.h"

////////////////////////////////////////
// START: basic
////////////////////////////////////////
#define MSG printf

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define ABS(A) (((A) >= 0) ? (A) : (-(A)))
#define NM1(N) (N)
// #define NM1(N) (((N)>1)?(N-1):(N))
////////////////////////////////////////
// END: basic
////////////////////////////////////////

////////////////////////////////////////
// START: constants
////////////////////////////////////////
#define LINE_SEARCH_ITER_MAX 10
#define MAX_LOOP 16
#define EPS (double(1E-16))
#define INF (double(1E+16))
// Coefficent for Armijo’s rule
#define SIGMA 0.001

// guess an initial tolerance for the approximate inverse W~X^{-1}
#define DROP_TOL0 1e-1
// multiplicative gap between |fX-fX_last|/|fX| and the threshold
// for approximately computing X^{-1}
#define DROP_TOL_GAP 1e-2

// brute force initial guess for the average number of neighbours in the graph
//#define C0       20
// maximum rank for testing
//#define RK_MAX  200
// minimum rank for testing
//#define RK_MIN   10
// expectecd failures
#define FAIL 5
#define SHRINK_DROP_TOL 0.5

// maximum block size for partially generating S
// also used for exporting block triangular factor
// in two passes (large blocks sequentially but multithreaded
//                versus small blocks in parallel)
#define MAX_BLOCK_SIZE 256
////////////////////////////////////////
// END: constants
////////////////////////////////////////

////////////////////////////////////////
// START: settings
////////////////////////////////////////
#define DETERMINISTIC 0
#define BLOCK_FSPAI 1
#define COLLECTIVE_CDU 1
#define CHOL_SUITESPARSE 0
////////////////////////////////////////
// END: settings
////////////////////////////////////////

using namespace std;

extern "C" {
// sort ind in increasing order
void qqsorti_(integer *ind, integer *stack, integer *n);
// sort ind in increasing order and permute val accordingly
void qsort1_(double *val, integer *ind, integer *stack, integer *n);
// sort val in decreasing order and permute ind accordingly
void qsort2_(double *val, integer *ind, integer *stack, integer *n);
#ifdef _BLAS_LAPACK_32_
// LAPACK QR decomposition
void dgeqrf_(int *m, int *n, double *A, int *lda, double *TAU, double *WORK,
             int *LWORK, int *info);
void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *TAU,
             double *WORK, int *LWORK, int *info);
// LAPACK SVD
void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
             double *S, double *U, int *LDU, double *VT, int *LDVT,
             double *WORK, int *LWORK, int *INFO, int strlenu, int strlenvt);
// BLAS dnrm2
double dnrm2_(int *n, double *dx, int *incx);

// BLAS dscal
void dscal_(int *n, double *dx, double *da, int *incx);
// BLAS ddot
double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);
// BLAS daxpy
void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
// BLAS dcopy x -> y
void dcopy_(int *n, double *dx, int *incx, double *dy, int *incy);
// BLAS idamax
int idamax_(int *n, double *dx, int *incx);
// BLAS dgemv y <- alpha A x + beta y
void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA,
            double *X, int *INCX, double *BETA, double *Y, int *INCY,
            int strln);
// BLAS dgemm C <- alpha AB + beta C
void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA,
            double *A, int *LDA, double *B, int *LDB, double *BETA, double *C,
            int *LDC, int strlna, int strlnb);
// BLAS dpotri, invert an SPD matrix given its Cholesky factor
void dpotri_(char *TRANS, int *N, double *A, int *LDA, int *info, int strln);
// BLAS dtrsm op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M, int *N,
            double *ALPHA, double *A, int *LDA, double *B, int *LDB, int strlns,
            int strlnu, int strlnt, int strlnd);
#else
// LAPACK QR decomposition
void dgeqrf_(integer *m, integer *n, double *A, integer *lda, double *TAU,
             double *WORK, integer *LWORK, integer *info);
void dorgqr_(integer *m, integer *n, integer *k, double *A, integer *lda,
             double *TAU, double *WORK, integer *LWORK, integer *info);
// LAPACK SVD
void dgesvd_(char *JOBU, char *JOBVT, integer *M, integer *N, double *A,
             integer *LDA, double *S, double *U, integer *LDU, double *VT,
             integer *LDVT, double *WORK, integer *LWORK, integer *INFO,
             int strlenu, int strlenvt);
// BLAS dnrm2
double dnrm2_(integer *n, double *dx, integer *incx);
// BLAS dscal
void dscal_(integer *n, double *dx, double *da, integer *incx);
// BLAS ddot
double ddot_(integer *n, double *dx, integer *incx, double *dy, integer *incy);
// BLAS daxpy
void daxpy_(integer *n, double *da, double *dx, integer *incx, double *dy,
            integer *incy);
// BLAS dcopy x -> y
void dcopy_(integer *n, double *dx, integer *incx, double *dy, integer *incy);
// BLAS idamax
integer idamax_(integer *n, double *dx, integer *incx);
// BLAS dgemv y <- alpha A x + beta y
void dgemv_(char *TRANS, integer *M, integer *N, double *ALPHA, double *A,
            integer *LDA, double *X, integer *INCX, double *BETA, double *Y,
            integer *INCY, int strln);
// BLAS dgemm C <- alpha AB + beta C
void dgemm_(char *TRANSA, char *TRANSB, integer *M, integer *N, integer *K,
            double *ALPHA, double *A, integer *LDA, double *B, integer *LDB,
            double *BETA, double *C, integer *LDC, int strlna, int strlnb);
// BLAS dpotri, invert an SPD matrix given its Cholesky factor
void dpotri_(char *TRANS, integer *N, double *A, integer *LDA, integer *info,
             int strln);
// BLAS dtrsm op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, integer *M,
            integer *N, double *ALPHA, double *A, integer *LDA, double *B,
            integer *LDB, int strlns, int strlnu, int strlnt, int strlnd);
#endif
}

class SQUIC {
 public:
  // CSC-like Matrix Sructure with separate pointers for each column
  typedef struct {
    integer nr = 0;
    integer nc = 0;
    integer nnz = 0;
    integer *ncol;
    integer **rowind;
    double **val;
  } SparseMatrix;

  // Statistics for ouput
  struct {
    double logdetX = -1;   // size of 1
    double dgap = -1;      // size of 1
    double trSX = -1;      // size of 1
    double time_cov = -1;  // size of 1
    double time_total = -1;
    std::vector<double> opt;        // size of max_iter
    std::vector<double> time_itr;   // size of max_iter
    std::vector<double> time_chol;  // size of max_iter
    std::vector<double> time_inv;   // size of max_iter
    std::vector<double> time_lns;   // size of max_iter
    std::vector<double> time_upd;   // size of max_iter
    std::vector<double> objlist;
  } Stat;

  // Runtime configuration
  struct {
    integer generate_scm = DETERMINISTIC;
    integer block_upd = COLLECTIVE_CDU;
    integer inversion = BLOCK_FSPAI;
    integer verbose = 1;
    integer factorization = CHOL_SUITESPARSE;
    // off_diagonal=1 makes such that only the off-diagonal values are penalized
    // (regardless of lambda)
    integer off_diagonal = 0;
  } RunTimeConfig;

  // Other
  double opt = 0.0;
  double dGap = 0.0;
  double cputime = 0.0;
  integer current_iter = 0;
  integer optSize = 1;
  integer iterSize = 1;

  // Matrices
  SparseMatrix W;  // Approximate Inverse of Inverse Covariance
  SparseMatrix X;  // Approximate Inverse Covariance
  SparseMatrix S;  // Sample Covariance
  vector<double> mu;

 public:
  SQUIC(integer Y_n, integer Y_p, double *Y_value);
  ~SQUIC();

  void print(SparseMatrix *S, integer p);

  void print_stats();

  void Run(double Lambda, double tol, int maxIter, double term_tol);
  void Run(double Lambda, double tol, int maxIter, double term_tol,
           SparseMatrix *X0, SparseMatrix *W0);
  void Run(double Lambda, SparseMatrix *LambdaMatrix, double tol, int maxIter,
           double term_tol);
  void Run(double Lambda, SparseMatrix *LambdaMatrix, double tol, int maxIter,
           double term_tol, SparseMatrix *X0, SparseMatrix *W0);

  void InitSparseMatrix(SparseMatrix *M, double diagVal);
  void PrintSparse(SparseMatrix *S);
  void PrintSparseCompact(SparseMatrix *S);
  void FreeSparse(SparseMatrix *S);
  integer CheckSymmetry(SparseMatrix *S);
  integer IsDiag(const SparseMatrix *A);
  void NullSparse(SparseMatrix *S);
  void CopySparse(SparseMatrix *S, SparseMatrix *W);
  void ClearSparse(SparseMatrix *S);
  void SortSparse(SparseMatrix *S, integer *stack);
  void CooToCustomeData(integer *X_i, integer *X_j, double *X_val,
                        integer X_nnz, SparseMatrix *X);
  void CscToCustomeData(integer *X_row_index, integer *X_col_ptr, double *X_val,
                        integer nnz, SparseMatrix *X);

  void DeleteVars();

 private:
  // Input Data - dense data
  integer p;
  integer n;
  double *Y;

  typedef struct {
    integer i;
    integer j;
  } index_pair;

  typedef struct {
    integer nr = 0;      /* total number of rows */
    integer nc = 0;      /* total number of columns */
    integer nnz = 0;     /* number of nonzeros */
    integer nblocks = 0; /* total number of blocks */
    integer *nblockcol;  /* number columns per diagonal block */
    integer *nblockrow;  /* number of rows per sub-diagonal block */
    integer **colind;    /* column indices for each diagonal block */
    integer **rowind;    /* indices for the sub-diagonal block entries */
    double **valD;       /* numerical values of the square diagonal blocks */
    double **valE;       /* numerical values of the sub-diagonal blocks */
  } SparseBlockMatrix;

  // generic flag to indicate whether a symbolic analysis has been performed
  integer analyze_done = 0;
  // generic flag to indicate whether a numerical factorization has been
  // performed
  integer factorization_done = 0;

  // Cholmod control parameters
  struct {
    integer n = -1;
    integer nnz;
    cholmod_common common_parameters;
    cholmod_sparse *pA = NULL;
    cholmod_factor *pL = NULL;
  } Cholmod;

  SparseMatrix L;         // Cholesky L
  SparseMatrix P;         // Permutation for L
  SparseMatrix D;         // Update Matrix Delta
  SparseBlockMatrix BL;   // block Cholesky factor from LDL'
  SparseBlockMatrix BiD;  // block inverse diagonal factor iD=D^{-1} from LDL'

 private:
  void RunCore(double Lambda, double tol, int maxIter, double term_tol);
  void RunCore(double Lambda, SparseMatrix *LambdaMatrix, double tol,
               int maxIter, double term_tol);
  void InitVars();

  void InitSparseBlockMatrix(SparseBlockMatrix *M);

  void GenerateSXL3B_OMP(double *mu, double Lambda, double *nu, integer *ind,
                         integer **idx_src, integer *idxpos);
  void GenerateSXL3B_OMP(double *mu, double Lambda, SparseMatrix *LambdaMatrix,
                         double *nu, integer *ind, integer **idx_src,
                         integer *idxpos);

  void AugmentS(double *mu, integer *idx, integer *idxpos, SparseMatrix *W);
  void AugmentS_OMP(double *mu, integer *idx, integer *idxpos, SparseMatrix *W);
  void AugmentD(SparseMatrix *D, integer *idx, integer *idxpos,
                index_pair *activeSet, integer numActive);
  void AugmentD_OMP(SparseMatrix *D, integer *idx, integer *idxpos,
                    index_pair *activeSet, integer numActive);

  void BlockCoordinateDescentUpdate(const double Lambda, SparseMatrix *D,
                                    integer *idx, integer *idxpos, double *val,
                                    index_pair *activeSet, integer first,
                                    integer last, double &l1norm_D,
                                    double &diffD);
  void BlockCoordinateDescentUpdate(const double Lambda,
                                    SparseMatrix *LambdaMatrix, SparseMatrix *D,
                                    integer *idx, integer *idxpos, double *val,
                                    index_pair *activeSet, integer first,
                                    integer last, double &l1norm_D,
                                    double &diffD);
  void FreeBlockSparse(SparseBlockMatrix *S);

  void SortSparse_OMP(SparseMatrix *S, integer *stack);

  double DiagNewton(const double Lambda, SparseMatrix *D);
  double DiagNewton(const double Lambda, SparseMatrix *LambdaMatrix,
                    SparseMatrix *D);

  inline double softThresh(double z, double r) {
    return z / fabs(z) * std::max(fabs(z) - r, 0.0);
  }

  void SelectedActiveIndexSet(const double lambda, const double gammma,
                              vector<index_pair> &active_index_set) {
    // size of matrix
    integer p = S.nc;

    // temporary vectors for column j
    std::vector<index_pair> active_index_set_candidate;
    std::vector<double> abs_grad;
    std::vector<double> abs_grad_candidate;

    // we dont know how many nonzers we will be adding ... we assume 2p new
    // candidate values with p of them accepted
    active_index_set.reserve(p + active_index_set.capacity());
    active_index_set_candidate.reserve(2 * p);

    // absolute value of gradient at indexes of active_index
    abs_grad.reserve(p + active_index_set.capacity());
    abs_grad_candidate.reserve(p);

    // clear existing index set
    active_index_set.clear();

    if (1) {
      cout << "out_abs_grad{end+1}=sparse(" << p << "," << p << ");" << endl;
      cout << "out_abs_grad_candidate{end+1}=sparse(" << p << "," << p << ");"
           << endl;
    }

    // compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>lambda
    // To do so, scan S, X and W
    for (integer j = 0; j < p; j++) {
      // counters for S_{:,j}, X_{:,j}, W_{:,j}
      integer k, l, m;
      k = l = m = 0;
      while (k < S.ncol[j] || l < X.ncol[j] || m < W.ncol[j]) {
        // set row indices to a value larger than possible
        integer r, s, t;
        r = s = t = p;
        // row index r of S_rj
        if (k < S.ncol[j]) {
          r = S.rowind[j][k];
        }
        // row index s of X_sj
        if (l < X.ncol[j]) {
          s = X.rowind[j][l];
        }
        // row index t of W_tj
        if (m < W.ncol[j]) {
          t = W.rowind[j][m];
        }

        // determine smallest index i=min{r,s,t}<p
        integer i = r;
        if (s < i) {
          i = s;
        }
        if (t < i) {
          i = t;
        }
        // only load the values of the smallest index
        double Sij, Xij, Wij;
        Sij = Xij = Wij = 0.0;
        if (i == r) {
          Sij = S.val[j][k++];
        }
        if (i == s) {
          Xij = X.val[j][l++];
        }
        if (i == t) {
          Wij = W.val[j][m++];
        }

        // make sure that only the upper [no diagonal] triangular part is
        // referenced! the diagonal values are automaticly accepted in the
        // active index set
        if (i <= j) {
          double abs_Gij = fabs(Sij - Wij);
          double abs_Xij = fabs(Xij);
          index_pair active_index;
          active_index.i = i;
          active_index.j = j;

          // Existing nonzeros
          if (abs_Xij > EPS) {
            active_index_set.push_back(active_index);
            abs_grad.push_back(abs_Gij);
            if (1) {
              cout << "out_abs_grad{end}(" << i + 1 << "," << j + 1
                   << ")=" << abs_Gij << ";" << endl;
              cout << "out_abs_grad{end}(" << j + 1 << "," << i + 1
                   << ")=" << abs_Gij << ";" << endl;
            }
          }
          // New cadidates
          if (abs_Xij <= EPS && abs_Gij > lambda) {
            active_index_set_candidate.push_back(active_index);
            abs_grad_candidate.push_back(abs_Gij);
            if (1) {
              cout << "out_abs_grad_candidate{end}(" << i + 1 << "," << j + 1
                   << ")=" << abs_Gij << ";" << endl;
              cout << "out_abs_grad_candidate{end}(" << j + 1 << "," << i + 1
                   << ")=" << abs_Gij << ";" << endl;
            }
          }
        }
      }
    }

    // include element of the new index set which have a larger values than
    // the maximim value of the current index set
    double abs_grad_max = *std::max_element(abs_grad.begin(), abs_grad.end());
    for (integer i = 0; i < abs_grad_candidate.size(); i++) {
      if (abs_grad_candidate[i] > abs_grad_max * gammma) {
        active_index_set.push_back(active_index_set_candidate[i]);
      }
    }
  }

  double DiagNewton_new(const double Lambda, SparseMatrix *D,
                        vector<index_pair> &active_index_set) {
    p = S.nr;
    // new number of nonzeros for \Delta

    // make D matrix
    D->nnz = 0;
    for (integer j = 0; j < p; j++) {
      D->ncol[j] = 0;
    }

    for (integer k = 0; k < active_index_set.size(); k++) {
      D->ncol[active_index_set[k].j]++;
      D->nnz++;

      // off diagional add the other half
      if (active_index_set[k].i != active_index_set[k].j) {
        D->ncol[active_index_set[k].i]++;
        D->nnz++;
      }
    }

    for (integer j = 0; j < p; j++) {
      D->rowind[j] = (integer *)malloc(D->ncol[j] * sizeof(integer));
      D->val[j] = (double *)malloc(D->ncol[j] * sizeof(double));
    }

    return 1;
  }

  double DiagNewton_test(const double Lambda, SparseMatrix *D) {
    /*
    Lambda     Lagrangian parameter
    S          sparse representation of the sample covariance matrix such that
    at least its diagonal entries and S_ij s.t. |S_ij|>\Lambda are stored
    X          sparse DIAGONAL constrained inverse covariance matrix
    W          W ~ X^{-1} sparse DIAGONAL covariance matrix
    D          define sparse update matrix \Delta with same pattern as S
    D is defined as sparse matrix 0
    */

    integer i, j, k, l, m, r, s, p;

    double Sij, Wii, Wjj, Xjj,
        logdet,                // log(det(X))
        l1normX,               // |\Lambda*X|_1
        trSX,                  // trace(SX)
        a, b, c, f, valw, mu;  // parameters for optimization function

    p = S.nr;
    // new number of nonzeros for \Delta
    D->nnz = p;

    // init log(det(X))
    logdet = 0.0;
    // init |\Lambda*X|_1
    l1normX = 0.0;
    // init trace(SX)
    trSX = 0.0;
    for (j = 0; j < p; j++) {
      // W is diagonal
      Wjj = W.val[j][0];

      D->ncol[j] = 1;
      D->rowind[j] = (integer *)malloc(m * sizeof(integer));
      D->val[j] = (double *)malloc(m * sizeof(double));

      // scan S_{:,j}
      for (k = 0; k < S.ncol[j]; k++) {
        // row index i of column j
        i = S.rowind[j][k];
        Sij = S.val[j][k];
        // off-diagonal case
        if (i != j) {
          // cout<<">>>>" << i << " "<< j <<endl;

          if (fabs(Sij) > Lambda && false) {
            // W is diagonal
            Wii = W.val[i][0];
            // off-diagonal case a = W_ii W_jj
            a = Wii * Wjj;
            // off-diagonal case b = S_ij, W is diagonal => W_ij=0
            b = Sij;
            // X is diagion Xjj =0
            c = 0.0;

            mu = -c + softThresh(c - b / a, Lambda / a);

            // since S is exactly symmetric, the same update will take place
            // for \Delta_ji
            D->val[j][k] = mu;
          }  // end if |Sij|>Lambda
        } else {
          D->rowind[j][k] = i;
          D->val[j][k] = 0.0;

          // i=j
          // handle diagonal part for \Delta_jj
          // X is diagonal
          Xjj = fabs(X.val[j][0]);
          // update log(det(X))
          logdet += log(Xjj);
          // update |X|_1
          if (RunTimeConfig.off_diagonal) l1normX += Xjj;

          // update trace(SX) by X_jj S_jj
          trSX += Xjj * Sij;

          // a = W_jj^2
          a = Wjj * Wjj;
          // b = S_jj - W_jj
          b = Sij - Wjj;
          // c = X_jj
          c = Xjj;

          // parameters for optimization parameter mu = -c + s(c-f,valw)
          // where s(z,r)=sign(z)*max{|z|-r,0}
          f = b / a;
          if (RunTimeConfig.off_diagonal)
            valw = 0.0;
          else
            valw = Lambda / a;
          // current start column l of \Delta
          if (c > f) {
            mu = -f - valw;
            if (c + mu < 0.0) {
              D->val[j][k] = -Xjj;
              continue;
            }
          } else {
            mu = -f + valw;
            if (c + mu > 0.0) {
              D->val[j][k] = -Xjj;
              continue;
            }
          }
          D->val[j][k] = mu;
        }  // end if-else i!=j
      }    // end while
    }      // end for j
    // adjust |\Lambda X|_1
    l1normX *= Lambda;

    // objective value f(X) = -log(det(X)) + trace(SX) + |\Lambda*X|_1
    double fX = -logdet + trSX + l1normX;

    return fX;
  }

  void AugmentSparse(SparseMatrix *S, integer *idx, integer *idxpos,
                     SparseMatrix *D);

  void AugmentSparse_OMP(SparseMatrix *S, integer *idx, integer *idxpos,
                         SparseMatrix *D);

  bool LogDet(double *ld, double drop_tol);

  integer FactorizeLdl(SparseMatrix *A, SparseMatrix *L_factor,
                       SparseMatrix *L_perm, double *D_LDL, double drop_tol);
  integer FactorizeLdl(integer *A_rowind, integer *A_p, double *A_val,
                       integer nc, integer nr, integer nnz,
                       SparseMatrix *L_factor, SparseMatrix *L_perm,
                       double *D_LDL, double drop_tol);
  void ExportLdl(double *D_LDL);
  void BlockExportLdl();

  integer FactorizeCholmod(integer *A_rowind, integer *A_p, double *A_val,
                           integer nc, integer nr, integer nnz,
                           SparseMatrix *L_factor, SparseMatrix *L_perm,
                           double *D_LDL);
  void ExportCholmod(double *D_LDL);
  void BlockExportCholmod();

  double ProjLogDet(double Lambda, double drop_tol);
  double ProjLogDet(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol);

  void PrintBlockSparse(SparseBlockMatrix *A);

  void BlockNeumannInv(double droptol, double *SL);

  void ResetSparseBlockMatrix(SparseBlockMatrix *M);
  void DeleteSparseBlockMatrix(SparseBlockMatrix *M);
};

#endif
