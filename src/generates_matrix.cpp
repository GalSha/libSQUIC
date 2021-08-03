/* generates_matrix.cpp
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

#ifndef generates_matrix_cpp
#define generates_matrix_cpp

#include <stdio.h>
#include <stdlib.h>

// Parallel
#include <omp.h>

// METIS
#include "metis.h"

#include "squic.hpp"

using namespace std;

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define ABS(A) (((A) >= 0) ? (A) : (-(A)))

// GenerateSR - Level 3 BLAS version

// brute force initial gues for the average number of neighbours in the graph
#define MY_C0 20
// maximum rank for testing
#define RK_MAX 200
// minimum rank for testing
#define RK_MIN 10
// expectecd failures
#define MY_FAIL 5

// accelaration using level 3 BLAS compared with the scalar implementation
#define L3B_ACCEL 0.01
#define C_FACTOR MY_FAIL


// OpenMP version of GenerateSXL3B
void SQUIC::GenerateSXL3B_OMP(double *mu, double Lambda, SparseMatrix *LambdaMatrix, double *nu, integer *ind, integer **idx_src, integer *idxpos)
{
	/*
	  Y          n samples Y=[y_1,...y_n] from which the low-rank empirical
	             covariance matrix S=1/NM1(n) sum_i (y_i-mu)(y_i-mu)^T can be build, where
	             mu=1/n sum_i y_i is the mean value
	       Y is stored by columns
	  p          size parameter space
	  n          number of samples
	  mu         mean value
	  Lambda     Lagrangian parameter
	  S          sparse representation of the sample covariance matrix
	  nu         1/NM1(n)||Y_{i,:}-\mu||^2 diagonal entries of S sorted in decreasing order
	  ind        indices 0,...,p-1 associated with. nu

	  idx     \  pre-allocated auxiliary buffers of size p
	  idxpos  /  initially we assume that idxpos is zero
	*/

	integer i, j, k, l, s,
		*rowind,
		*idx,
		max_block, // true maximum block size <=MAX_BLOCK_SIZE
		cnt_block,
		cnt_loop,
		*ibuff,
		cnt, cntl, loop;

	double *pZ, // pointer to Z_{i,:}
		*val,
		dbuff, a, b,
		*v, *z, *zl, // vectors for matrix-vector multiplication
		*Z;			 // 1/sqrt(NM1(n))*[Y-mu] computed explicitly for
					 // efficiency reasons only
#if defined PRINT_MSGL3B || defined PRINT_MSG
	double timeBegin = omp_get_wtime(), time_matvec = 0.0, time_scan = 0.0;
#endif

#if defined PRINT_MSGL3B || defined PRINT_MSG
	printf("start up GenerateSXL3B_OMP using %2d threads\n", omp_get_max_threads());
	fflush(stdout);
#endif
#ifdef PRINT_MSGL3B
	timeBegin = omp_get_wtime();
#endif

	Z = (double *)malloc(p * n * sizeof(double));
	// Z <- Y
	memcpy(Z, Y, p * n * sizeof(double));

	// compute (i)   Z <- [Z-mu]/sqrt(NM1(n))
	//         (ii) \nu_i=||y_i-\mu||^2 = NM1(n) Z_{i,:}Z_{i,:}^T
	// aux variables for scaling
	dbuff = 1.0 / sqrt((double)NM1(n));
	for (i = 0; i < p; i++, mu++)
	{
		ind[i] = i;
		// || Y_{i,:}-\mu ||^2== n S_ii = NM1(n) Z_{i,:}Z_{i,:}^T
		nu[i] = 0;
		for (pZ = Z + i, j = 0; j < n; j++, pZ += p)
		{
			// Z_ij <- Y_ij-mu_i
			*pZ -= *mu;
			// accumulate (Y_ij-mu_i)^2 over all j
			nu[i] += *pZ * *pZ;
			// Z_ij <- [Y_ij-mu_i]/sqrt(NM1(n))
			*pZ *= dbuff;
		} // end for j
	}	  // end for i
	// re-adjust mu
	mu -= p;

	// sort \nu in decreasing order and permute ind accordingly, idx used as stack
	idx = *idx_src;
	qsort2_(nu, ind, idx, &p);

	// set up data structures for S as far as it is possible right now
	S.nnz = 0;
	S.nr = S.nc = p;

	S.ncol = (integer *)malloc(p * sizeof(integer));
	S.rowind = (integer **)malloc(p * sizeof(integer *));
	S.val = (double **)malloc(p * sizeof(double *));

	// allocate vectors for matrix-vector multiplication
	v = (double *)malloc(n * MAX_BLOCK_SIZE * sizeof(double));
	z = (double *)malloc(p * MAX_BLOCK_SIZE * sizeof(double));
	// initialization, necessary to avoid 0*NaN or 0*inf inside dgemm_
	for (i = 0; i < p * MAX_BLOCK_SIZE; i++)
	{
		z[i] = 0.0;
	}
#ifdef PRINT_MSGL3B
	printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBegin));
	fflush(stdout);
#endif

	// count the number of computed columns of S
	cnt = 0;
	// beginning of an outer loop
	loop = 0;
	cnt_loop = 0;
	integer ii, kk, ll, *pi;
	double myLambda;
	while (cnt < p)
	{
		// collect up to MAX_BLOCK_SIZE columns simultaneously
#ifdef PRINT_MSGL3B
		printf("collect up %ld columns simultaneously\n", MAX_BLOCK_SIZE);
		fflush(stdout);
#endif
#if defined PRINT_MSGL3B || defined PRINT_MSG
		timeBegin = omp_get_wtime();
#endif

		// maximum block size is usually fixed except for the last block
		max_block = MIN(MAX_BLOCK_SIZE, p - loop);
		// compute v=Z_{loop:loop+max_block-1,:}^T
		i = 1;
		k = n;
		for (pZ = Z + loop, j = 0; j < n; j++, pZ += p)
		{
#ifdef _BLAS_LAPACK_32_
			int mymax_block = max_block, myi = i, myk = k;
			dcopy_(&mymax_block, pZ, &myi, v + j, &myk);
#else
			dcopy_(&max_block, pZ, &i, v + j, &k);
#endif
		}

		// z=Z_{loop:end,:}*v == 1.0 * Z_{loop:end:,:}*v + 0.0 * z(loop:end,:)
		// z has been initialized after allocation, thus there will be no 0*NaN or 0*inf
		a = 1.0;
		b = 0.0;
		k = p - loop;
#ifdef _BLAS_LAPACK_32_
		int myk = k, mymax_block = max_block, myn = n, myp = p;
		dgemm_((char *)"N", (char *)"N", &myk, &mymax_block, &myn, &a, Z + loop, &myp, v, &myn, &b, z + loop, &myp, 1, 1);
#else
		dgemm_((char *)"N", (char *)"N", &k, &max_block, &n, &a, Z + loop, &p, v, &n, &b, z + loop, &p, 1, 1);
#endif
#if defined PRINT_MSGL3B || defined PRINT_MSG
		time_matvec += (omp_get_wtime() - timeBegin);
#endif
#ifdef PRINT_MSGL3B
		printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBegin));
		fflush(stdout);
#endif
#ifdef PRINT_MSGL3B
		printf("scan block matrix for sufficiently large entries\n");
		fflush(stdout);
#endif
#if defined PRINT_MSGL3B || defined PRINT_MSG
		timeBegin = omp_get_wtime();
#endif

		// scan z for the sufficiently large entries
		// printf("maximum available cores   %ld\n",omp_get_num_procs());
		// printf("maximum available threads %ld\n",omp_get_max_threads());
		// printf("number of active threads  %ld\n",omp_get_num_threads());
#pragma omp parallel for shared(max_block, loop, z, p, S, Lambda) private(cntl, zl, s, i, dbuff, rowind, val, ii, kk, ll, myLambda, pi)
		for (cnt_block = 0; cnt_block < max_block; cnt_block++)
		{
			//#pragma omp critical
			//{
			//  printf("thread %2d active\n",omp_get_thread_num());
			//  fflush(stdout);
			//}
			// check column cntl of S(:,cntl)
			cntl = loop + cnt_block;
			// associated numerical values
			zl = z + cnt_block * p;

			// scan lower triangular part
			s = 0;
			// abbrevations for the current column of LambdaMatrix
			pi = LambdaMatrix->rowind[cntl];
			ll = LambdaMatrix->ncol[cntl];
			kk = 0;
			for (i = cntl; i < p; i++)
			{
				dbuff = fabs(zl[i]);
				// check for a different value of Lambda rather than the default value
				myLambda = Lambda;
				for (; kk < ll; kk++)
				{
					ii = pi[kk];
					if (ii == i)
					{
						myLambda = LambdaMatrix->val[cntl][kk];
						break;
					}
					else if (ii >= i)
						break;
				} // end for kk
				// |S(i,cntl)|>=Lambda or i=cntl
				if (dbuff >= myLambda || i == cntl)
				{
					s++;
				}
			} // end for i
			S.ncol[cntl] = s;
			S.rowind[cntl] = (integer *)malloc(s * sizeof(integer));
			S.val[cntl] = (double *)malloc(s * sizeof(double));
			rowind = S.rowind[cntl];
			val = S.val[cntl];
			// scan lower triangluar part again
			s = 0;
			kk = 0;
			for (i = cntl; i < p; i++)
			{
				dbuff = zl[i];
				// check for a different value of Lambda rather than the default value
				myLambda = Lambda;
				for (; kk < ll; kk++)
				{
					ii = pi[kk];
					if (ii == i)
					{
						myLambda = LambdaMatrix->val[cntl][kk];
						break;
					}
					else if (ii >= i)
						break;
				} // end for kk
				// |S(i,cntl)|>=Lambda or i=cntl
				if (fabs(dbuff) >= myLambda || i == cntl)
				{
					rowind[s] = i;
					val[s] = dbuff;
					s++;
				}
			} // end for i
		}	  // end for cnt_block
		// end OMP parallel for

		cnt = loop + max_block;
#if defined PRINT_MSGL3B || defined PRINT_MSG
		time_scan += (omp_get_wtime() - timeBegin);
#endif
#ifdef PRINT_MSGL3B
		printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBegin));
		fflush(stdout);
#endif

		// increase counter for the number of mat-vecs
		loop += max_block;
		cnt_loop++;

#if defined PRINT_INFOL3B || defined PRINT_MSGL3B
		printf("loop=%ld\n", loop);
		fflush(stdout);
#endif
	} // end while

#ifdef PRINT_MSG
	printf("time mat-vec: %8.1le [sec], time scanning: %8.1le [sec]\n", time_matvec, time_scan);
	fflush(stdout);
#endif

	// transpose matrix in order to make it exactly symmetric
	// init idx
	for (j = 0; j < p; j++)
	{
		idx[j] = 0;
	}
	// scan strict lower triangular part
	for (j = 0; j < p; j++)
	{
		rowind = S.rowind[j];
		// initially our matrix has started with the diagonal entry by construction
		// scan strict lower triangular part, column j
		l = S.ncol[j];
		for (k = 1; k < l; k++)
		{
			i = rowind[k];
			// additional entry in column i
			idx[i]++;
		} // end for k
	}	  // end for j
	ibuff = (integer *)malloc(p * sizeof(integer));
	// re-allocate additional memory
	for (j = 0; j < p; j++)
	{
		i = idx[j];
		// store additional nonzeros
		ibuff[j] = i;
		l = S.ncol[j] + i;
		S.rowind[j] = (integer *)realloc(S.rowind[j], l * sizeof(integer));
		S.val[j] = (double *)realloc(S.val[j], l * sizeof(double));
		// increase nz
		S.ncol[j] = l;
		// shift strict lower triangular entries to the back
		rowind = S.rowind[j];
		val = S.val[j];
		for (k = l - 1; k >= i; k--)
		{
			rowind[k] = rowind[k - i];
			val[k] = val[k - i];
		} // end for k
		// reset counter
		idx[j] = 0;

	} // end for j
	// insert strict upper triangular part in front
	for (j = 0; j < p; j++)
	{
		rowind = S.rowind[j];
		val = S.val[j];
		// scan strict lower triangular part
		l = S.ncol[j];
		for (k = ibuff[j] + 1; k < l; k++)
		{
			i = rowind[k];
			// additional entry in column i, insert entries in the gap
			s = idx[i];
			S.rowind[i][s] = j;
			S.val[i][s] = val[k];
			idx[i]++;
		} // end for k
	}	  // end for j

	// count nnz of S
	for (j = 0; j < p; j++)
	{
		S.nnz += S.ncol[j];
	}

#ifdef PRINT_INFOSR
	printf("loop=%ld, nnz(S)/p=%8.1le\n", loop, S.nnz / (double)p);
#endif

#ifdef PRINT_CHECK
	i = CheckSymmetry(S);
	if (i)
	{
		printf("!!! GenerateSXL3B_OMP: S is nonsymmetric!!!\n");
		fflush(stdout);
	}
#endif

	free(ibuff);
	free(Z);
	free(z);
	free(v);
} // end GenerateSXL3B_OMP



#endif
