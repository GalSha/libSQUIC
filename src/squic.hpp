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

#define SQUIC_VER 1.0

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Standard General Utilities Library
#include <cstdio>
#include <iostream>
#include <iomanip> // Precision
#include <cmath>
#include <assert.h>

// Data structures, algos and numerics
#include <vector>
#include <string>
#include <algorithm>
#include <map>

// Sleep, time, fprintf etc...
#include <unistd.h>
#include <time.h>
#include <cstdlib>
#include <cstddef> // sleep
#include <stdio.h> // printf

// IO
#include <fstream>
#include <iostream>

#include "long_integer.h"
#include "cholmod.h"

#define MSG printf

// maximum block size for partially generating S
// also used for exporting block triangular factor
// in two passes (large blocks sequentially but multithreaded
//                versus small blocks in parallel)
#define MAX_BLOCK_SIZE 256

#define NM1(N) (N)
// #define NM1(N) (((N)>1)?(N-1):(N))

using namespace std;

#define DETERMINISTIC 0

#define BLOCK_FSPAI 1

#define COLLECTIVE_CDU 1

#define CHOL_SUITESPARSE 0

extern "C"
{
	// sort ind in increasing order
	void qqsorti_(integer *ind, integer *stack, integer *n);
	// sort ind in increasing order and permute val accordingly
	void qsort1_(double *val, integer *ind, integer *stack, integer *n);
	// sort val in decreasing order and permute ind accordingly
	void qsort2_(double *val, integer *ind, integer *stack, integer *n);
#ifdef _BLAS_LAPACK_32_
	// LAPACK QR decomposition
	void dgeqrf_(int *m, int *n, double *A, int *lda, double *TAU, double *WORK, int *LWORK, int *info);
	void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *TAU, double *WORK, int *LWORK, int *info);
	// LAPACK SVD
	void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA, double *S, double *U, int *LDU,
	             double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO, int strlenu, int strlenvt);
	// BLAS dnrm2
	double dnrm2_(int *n, double *dx, int *incx);

	// BLAS dscal
	void dscal_(int *n, double *dx, double *da, int *incx);
	// BLAS ddot
	double ddot_(int *n, double *dx, int *incx,
	             double *dy, int *incy);
	// BLAS daxpy
	void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
	// BLAS dcopy x -> y
	void dcopy_(int *n, double *dx, int *incx,
	            double *dy, int *incy);
	// BLAS idamax
	int idamax_(int *n, double *dx, int *incx);
	// BLAS dgemv y <- alpha A x + beta y
	void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A,
	            int *LDA, double *X, int *INCX, double *BETA, double *Y,
	            int *INCY, int strln);
	// BLAS dgemm C <- alpha AB + beta C
	void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
	            double *ALPHA, double *A, int *LDA,
	            double *B, int *LDB,
	            double *BETA, double *C, int *LDC, int strlna, int strlnb);
	// BLAS dpotri, invert an SPD matrix given its Cholesky factor
	void dpotri_(char *TRANS, int *N, double *A, int *LDA, int *info, int strln);
	// BLAS dtrsm op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M, int *N,
	            double *ALPHA, double *A, int *LDA, double *B, int *LDB,
	            int strlns, int strlnu, int strlnt, int strlnd);
#else
	// LAPACK QR decomposition
	void dgeqrf_(integer *m, integer *n, double *A, integer *lda, double *TAU, double *WORK, integer *LWORK, integer *info);
	void dorgqr_(integer *m, integer *n, integer *k, double *A, integer *lda, double *TAU, double *WORK, integer *LWORK, integer *info);
	// LAPACK SVD
	void dgesvd_(char *JOBU, char *JOBVT, integer *M, integer *N, double *A, integer *LDA, double *S, double *U, integer *LDU,
	             double *VT, integer *LDVT, double *WORK, integer *LWORK, integer *INFO, int strlenu, int strlenvt);
	// BLAS dnrm2
	double dnrm2_(integer *n, double *dx, integer *incx);
	// BLAS dscal
	void dscal_(integer *n, double *dx, double *da, integer *incx);
	// BLAS ddot
	double ddot_(integer *n, double *dx, integer *incx,
	             double *dy, integer *incy);
	// BLAS daxpy
	void daxpy_(integer *n, double *da, double *dx, integer *incx, double *dy, integer *incy);
	// BLAS dcopy x -> y
	void dcopy_(integer *n, double *dx, integer *incx,
	            double *dy, integer *incy);
	// BLAS idamax
	integer idamax_(integer *n, double *dx, integer *incx);
	// BLAS dgemv y <- alpha A x + beta y
	void dgemv_(char *TRANS, integer *M, integer *N, double *ALPHA, double *A,
	            integer *LDA, double *X, integer *INCX, double *BETA, double *Y,
	            integer *INCY, int strln);
	// BLAS dgemm C <- alpha AB + beta C
	void dgemm_(char *TRANSA, char *TRANSB, integer *M, integer *N, integer *K,
	            double *ALPHA, double *A, integer *LDA,
	            double *B, integer *LDB,
	            double *BETA, double *C, integer *LDC, int strlna, int strlnb);
	// BLAS dpotri, invert an SPD matrix given its Cholesky factor
	void dpotri_(char *TRANS, integer *N, double *A, integer *LDA, integer *info, int strln);
	// BLAS dtrsm op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, integer *M, integer *N,
	            double *ALPHA, double *A, integer *LDA, double *B, integer *LDB,
	            int strlns, int strlnu, int strlnt, int strlnd);
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
		double logdetX = -1; // size of 1
		double dgap = -1;	 // size of 1
		double trSX = -1;	 // size of 1
		double time_cov = -1;	 // size of 1
		double time_total = -1;
		std::vector<double> opt;	   // size of max_iter
		std::vector<double> time_itr;  // size of max_iter
		std::vector<double> time_chol; // size of max_iter
		std::vector<double> time_inv;  // size of max_iter
		std::vector<double> time_lns;  // size of max_iter
		std::vector<double> time_upd;  // size of max_iter
		std::vector<double> objlist;
	} Stat;

	// Runtime configuration
	struct {
		integer generate_scm = DETERMINISTIC; 
		integer block_upd = COLLECTIVE_CDU;
		integer inversion = BLOCK_FSPAI;
		integer verbose = 1;
		integer factorization = CHOL_SUITESPARSE; 
		integer off_diagonal = 0;				  // 1; //
	} RunTimeConfig;

	// Other
	double opt = 0.0;
	double dGap = 0.0;
	double cputime = 0.0;
	integer current_iter = 0;
	integer optSize = 1;
	integer iterSize = 1;

	// Matrices
	SparseMatrix W; // Approximate Inverse of Inverse Covariance
	SparseMatrix X; // Approximate Inverse Covariance
	SparseMatrix S; // Sample Covariance

 public:
	SQUIC(integer Y_n, integer Y_p, double *Y_value);
	~SQUIC();

	void print(SparseMatrix *S, integer p);

	void print_stats();

	void run(double Lambda, double tol, int maxIter, double term_tol);
	void run(double Lambda, double tol, int maxIter, double term_tol, SparseMatrix *X0, SparseMatrix *W0);
	void run(double Lambda, SparseMatrix *LambdaMatrix, double tol, int maxIter, double term_tol);
	void run(double Lambda, SparseMatrix *LambdaMatrix, double tol, int maxIter, double term_tol, SparseMatrix *X0, SparseMatrix *W0);

	void init_SparseMatrix(SparseMatrix *M, double diagVal);
	void PrintSparse(SparseMatrix *S);
	void PrintSparseCompact(SparseMatrix *S);
	void FreeSparse(SparseMatrix *S);
	integer CheckSymmetry(SparseMatrix *S);
	integer IsDiag(const SparseMatrix *A);
	void NullSparse(SparseMatrix *S);
	void CopySparse(SparseMatrix *S, SparseMatrix *W);
	void ClearSparse(SparseMatrix *S);
	void SortSparse(SparseMatrix *S, integer *stack);
	void COO_to_CustomeData(integer *X_i, integer *X_j, double *X_val, integer X_nnz, SparseMatrix *X);
	void CSC_to_CustomeData(integer *X_row_index, integer *X_col_ptr, double *X_val, integer nnz, SparseMatrix *X);

	void SQUIC_DeleteVars();

 private:
	// Input Data - dense data
	integer p;
	integer n;
	double *Y;

	typedef struct {
		integer i;
		integer j;
	} IndexPair;

	typedef struct {
		integer nr = 0;		 /* total number of rows */
		integer nc = 0;		 /* total number of columns */
		integer nnz = 0;	 /* number of nonzeros */
		integer nblocks = 0; /* total number of blocks */
		integer *nblockcol;	 /* number columns per diagonal block */
		integer *nblockrow;	 /* number of rows per sub-diagonal block */
		integer **colind;	 /* column indices for each diagonal block */
		integer **rowind;	 /* indices for the sub-diagonal block entries */
		double **valD;		 /* numerical values of the square diagonal blocks */
		double **valE;		 /* numerical values of the sub-diagonal blocks */
	} SparseBlockMatrix;

	// generic flag to indicate whether a symbolic analysis has been performed
	integer analyze_done = 0;
	// generic flag to indicate whether a numerical factorization has been performed
	integer factorization_done = 0;

	// Cholmod control parameters
	struct {
		integer n = -1;
		integer nnz;
		cholmod_common common_parameters;
		cholmod_sparse *pA = NULL;
		cholmod_factor *pL = NULL;
	} Cholmod;


	SparseMatrix L; // Cholesky L
	SparseMatrix P; // Permutation for L

	SparseBlockMatrix BL;  // block Cholesky factor from LDL'
	SparseBlockMatrix BiD; // block inverse diagonal factor iD=D^{-1} from LDL'

 private:
	void run_core(double Lambda, double tol, int maxIter, double term_tol);
	void run_core(double Lambda, SparseMatrix *LambdaMatrix, double tol, int maxIter, double term_tol);
	void SQUIC_InitVars();

	void init_SparseBlockMatrix(SparseBlockMatrix *M);

	void GenerateSXL3B_OMP(double *mu, double Lambda, double *nu, integer *ind, integer **idx_src, integer *idxpos);
	void GenerateSXL3B_OMP(double *mu, double Lambda, SparseMatrix *LambdaMatrix, double *nu, integer *ind, integer **idx_src, integer *idxpos);




	void AugmentS(double *mu, integer *idx, integer *idxpos, SparseMatrix *W);
	void AugmentS_OMP(double *mu, integer *idx, integer *idxpos, SparseMatrix *W);
	void AugmentD(SparseMatrix *D, integer *idx, integer *idxpos, IndexPair *activeSet, integer numActive);
	void AugmentD_OMP(SparseMatrix *D, integer *idx, integer *idxpos, IndexPair *activeSet, integer numActive);

	void BlockCoordinateDescentUpdate(const double Lambda, SparseMatrix *D, integer *idx, integer *idxpos, double *val, IndexPair *activeSet, integer first, integer last, double &normD, double &diffD);
	void BlockCoordinateDescentUpdate(const double Lambda, SparseMatrix *LambdaMatrix, SparseMatrix *D, integer *idx, integer *idxpos, double *val, IndexPair *activeSet, integer first, integer last, double &normD, double &diffD);
	void FreeBlockSparse(SparseBlockMatrix *S);

	void SortSparse_OMP(SparseMatrix *S, integer *stack);

	double DiagNewton(const double Lambda, SparseMatrix *D);
	double DiagNewton(const double Lambda, SparseMatrix *LambdaMatrix, SparseMatrix *D);

	void AugmentSparse(SparseMatrix *S, integer *idx, integer *idxpos, SparseMatrix *D);

	void AugmentSparse_OMP(SparseMatrix *S, integer *idx, integer *idxpos, SparseMatrix *D);

	integer logDet(double *D_LDL, double *ld, double drop_tol);

	integer Factorize_LDL(SparseMatrix *A, SparseMatrix *L_factor, SparseMatrix *L_perm, double *D_LDL, double drop_tol);
	integer Factorize_LDL(integer *A_rowind, integer *A_p, double *A_val, integer nc, integer nr, integer nnz, SparseMatrix *L_factor, SparseMatrix *L_perm, double *D_LDL, double drop_tol);
	void Export_LDL(double *D_LDL);
	void BlockExport_LDL();

	integer Factorize_CHOLMOD(integer *A_rowind, integer *A_p, double *A_val, integer nc, integer nr, integer nnz, SparseMatrix *L_factor, SparseMatrix *L_perm, double *D_LDL);
	void Export_CHOLMOD(double *D_LDL);
	void BlockExport_CHOLMOD();


	double projLogDet(double Lambda, double drop_tol);
	double projLogDet(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol);

	void PrintBlockSparse(SparseBlockMatrix *A);

	void BNINV(double droptol, double *SL);

	void reset_SparseBlockMatrix(SparseBlockMatrix *M);
	void delete_SparseBlockMatrix(SparseBlockMatrix *M);
};


#endif
