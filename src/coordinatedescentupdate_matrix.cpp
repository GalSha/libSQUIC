/* coordinatedescentupdate_matrix.cpp
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

#ifndef cdu_matrix_cpp
#define cdu_matrix_cpp

#include <stdio.h>
#include <cstdlib>

// Parallel
#include <omp.h>

// METIS
#include "metis.h"

#include "squic.hpp"

// #define PRINT_INFO
// #define PRINT_INFO1

//#define PRINT_INFO_BCDU

using namespace std;

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define ABS(A) (((A) >= 0) ? (A) : (-(A)))


// update several \Delta_ij on the active set (I_free)
// We assume that all \Delta_ij, \Delta_ji have existing (possibly 0) entries to cover the
// changes
// slightly improved version using partiall OpenMP
void SQUIC::BlockCoordinateDescentUpdate(const double Lambda, SparseMatrix *LambdaMatrix,
										 SparseMatrix *D,
										 integer *idx, integer *idxpos, double *val,
										 IndexPair *activeSet, integer first, integer last,
										 double &normD, double &diffD)
{
	/*
    S          sparse representation of the sample covariance matrix
    Lambda     Lagrangian parameter
    X          sparse constrained inverse covariance matrix
    W          W ~ X^{-1} sparse covariance matrix
    D          sparse update matrix \Delta
    idx     \
    idxpos   } pre-allocated auxiliary buffers of size p to compute sparse-sparse
    val     /  product \Delta W_{:,j}; initially we assume that idxpos and val are zero
    activeSet  active set of positions (i,j) for descent update 
    first,last 
    normD      1/2 |\Delta|_1
    diffD      |(\mu e_ie_j^T)_{i,j}|_1   for all active (i,j)
  */

	// index counters etc
	integer k, l, m, p = S.nr, cnt, lw, ld, r, kk, ll, mm, i, j, q, jj;
	// values for optimization and sparse-sparse products
	double a, b, c, d, f, mu, valw;
	// we scan the sparse matrices for certain values at position (i,j), (i,i), (j,j)
	// if these are not present, then their values must be zero
	double Wij = 0.0, Wii = 0.0, Wjj = 0.0, Wir, Wjr;
	double Sij = 0.0, Xij = 0.0, Dij = 0.0;
	// pointer to the current column of Y
	double *pY, *diagW = (double *)malloc((size_t)p * sizeof(double *));
	// sparse matrices A,B,C with pattern taken from the active set, resp. its
	// transposed pattern M, needed for symmetry reasons when \Delta is updated
	// A <- W.^2 + diag(W) * diag(W)'
	// B <- S - W + W' * \Delta * W
	// C <- X + \Delta
	// M <- (-C + s(C-B./A,\Lambda./A))', where s(z,r)=sgn(z) * max{|z|-r,0}
	//
	// since the pattern from the active set refers to indices (i,j) such that
	// i<=j, the matrices A,B,C will be upper triangular, whereas M will be strictly
	// lower triangular, i.e., the diagonal part in the transposed pattern is excluded
	SparseMatrix A, B, C, M;
	integer Ancol, *pArowind, Bncol, *pBrowind, Cncol, *pCrowind, Mncol, *pMrowind,
		Dncol, *pDrowind, Sncol, *pSrowind, Wncol, *pWrowind, Xncol, *pXrowind;
	double *pAval, *pBval, *pCval, *pMval,
		*pDval, *pSval, *pWval, *pXval;

#ifdef PRINT_INFO_BCDU
	double timeBegin = omp_get_wtime();
	double par_loop1 = 0, timeBeginlocal;
	double par_loop2 = 0;
	printf("BCDU entered, first=%ld,last=%ld\n", first, last);
	fflush(stdout);
#endif

	// convert Active set list into a sparse matrix in order to augment D efficiently
	A.nr = A.nc = p;
	A.ncol = (integer *)calloc((size_t)p, sizeof(integer));
	A.rowind = (integer **)malloc((size_t)p * sizeof(integer *));
	A.val = (double **)malloc((size_t)p * sizeof(double *));

	// count nnz per column
	for (k = first; k <= last; k++)
	{
		j = activeSet[k].j;
		A.ncol[j]++;
	} // end for k
#ifdef PRINT_INFO_BCDU1
	printf("A, number of nonzeros per column counted\n");
	fflush(stdout);
#endif
	// allocate index memory
	A.nnz = 0;
	for (j = 0; j < p; j++)
	{
		l = A.ncol[j];
		A.rowind[j] = (integer *)malloc((size_t)l * sizeof(integer));
		A.val[j] = (double *)malloc((size_t)l * sizeof(double));
		for (m = 0; m < l; m++)
			A.val[j][m] = 0.0;
		A.nnz += l;
		// reset counter
		A.ncol[j] = 0;
	} // end for j
#ifdef PRINT_INFO_BCDU1
	printf("A, index memory allocated\n");
	fflush(stdout);
#endif
	// insert row indices per column
	for (k = first; k <= last; k++)
	{
		i = activeSet[k].i;
		j = activeSet[k].j;
		l = A.ncol[j];
		A.rowind[j][l] = i;
		A.ncol[j] = l + 1;
	} // end for k
#ifdef PRINT_INFO_BCDU1
	printf("A, row indices inserted\n");
	fflush(stdout);
#endif
	// make sure that the indices in each column are sorted in increasing order
	for (j = 0; j < p; j++)
	{
		l = A.ncol[j];
		qqsorti_(A.rowind[j], idx, &l);
	} // end for j
#ifdef PRINT_INFO_BCDU1
	printf("A generated\n");
	fflush(stdout);
#endif
	// B <- A
	CopySparse(&B, &A);
	// C <- A
	CopySparse(&C, &A);

	// compute M=A'
	// convert Active set list into a sparse matrix in order to augment D efficiently
	M.nr = M.nc = p;
	M.ncol = (integer *)calloc((size_t)p, sizeof(integer));
	M.rowind = (integer **)malloc((size_t)p * sizeof(integer *));
	M.val = (double **)malloc((size_t)p * sizeof(double *));

	// count nnz per column
	for (k = first; k <= last; k++)
	{
		// interchange i<->j
		j = activeSet[k].i;
		i = activeSet[k].j;
		if (i != j)
			M.ncol[j]++;
	} // end for k
	// allocate index memory
	M.nnz = 0;
	for (j = 0; j < p; j++)
	{
		l = M.ncol[j];
		M.rowind[j] = (integer *)malloc((size_t)l * sizeof(integer));
		M.val[j] = (double *)malloc((size_t)l * sizeof(double));
		M.nnz += l;
		// reset counter
		M.ncol[j] = 0;
	} // end for j
	// do not insert indices or values, these will be automtically
	// filled in when mu is computed

#ifdef PRINT_INFO_BCDU1
	printf("M generated\n");
	fflush(stdout);
#endif

#pragma omp parallel for shared(p, B, S, diagW, W, A) private(m, k, i, Bncol, pBrowind, pBval, Sncol, pSrowind, pSval, Wncol, pWrowind, pWval, pAval)
	for (j = 0; j < p; j++)
	{

		// B <- S - W + W' * \Delta * W
		// 1. step: extract entries of S to B, B <- S
		// scan through columns j of S and B, where S and B have row indices
		// that are taken in increasing order
		m = k = 0;
		Bncol = B.ncol[j];
		pBrowind = B.rowind[j];
		pBval = B.val[j];
		Sncol = S.ncol[j];
		pSrowind = S.rowind[j];
		pSval = S.val[j];
		while (m < Bncol && k < Sncol)
		{
			// current row index i of S_ij
			i = pSrowind[k];
			// check whether index pair (i,j) exists in B
			while (m < Bncol)
				// we have reached or passed i
				if (pBrowind[m] >= i)
					break;
				else
					m++;
			// if we stopped inside column j
			if (m < Bncol)
			{
				// we found row index i
				if (pBrowind[m] == i)
				{
					pBval[m] = pSval[k];
					m++;
				} // end if
			}	  // end if
			k++;
		} // end while

		// extract diagonal part of W
		// scan through column j of W
		k = 0;
		diagW[j] = 0.0;
		Wncol = W.ncol[j];
		pWrowind = W.rowind[j];
		pWval = W.val[j];
		while (k < Wncol)
		{
			// current row index i of W_ij
			i = pWrowind[k];
			// did we pass or meet W_jj?
			if (i >= j)
			{
				if (i == j)
					diagW[j] = pWval[k];
				break;
			} // end if
			else
				k++;
		} // end while

		// A <- W.^2 + diag(W) * diag(W)'
		// 1. step: extract entries of W to A, A <- W.^2
		// B <- S - W + W' * \Delta * W
		// 2. step: extract entries of W to B, B <- B - W (= S - W)
		// scan through columns j of W and B, where W and B have row indices
		// that are taken in increasing order
		m = k = 0;
		pAval = A.val[j];
		while (m < Bncol && k < Wncol)
		{
			// current row index i of W_ij
			i = pWrowind[k];
			// check whether index pair (i,j) exists in B
			while (m < Bncol)
				// we have reached or passed i
				if (pBrowind[m] >= i)
					break;
				else
					m++;
			// if we stopped inside column j
			if (m < Bncol)
			{
				// we found row index i
				if (pBrowind[m] == i)
				{
					// B_ij <- B_ij - W_ij
					pBval[m] -= pWval[k];
					// A_ij <- W_ij^2
					pAval[m] = pWval[k] * pWval[k];
					m++;
				} // end if
			}	  // end if
			k++;
		} // end while
	}	  // end for j
	// end omp parallel for
#ifdef PRINT_INFO_BCDU
	printf("B, 1. step completed, W, diagonal part extracted, A,1 . step, B, 2. step completed, %8.1le [sec]\n", omp_get_wtime() - timeBegin);
	fflush(stdout);
	timeBegin = omp_get_wtime();
#endif

#pragma omp parallel for shared(p, A, diagW, X, D, C) private(m, i, k, l, Xij, Dij, Ancol, pArowind, pAval, Cncol, pCrowind, pCval, \
															  Dncol, pDrowind, pDval, Xncol, pXrowind, pXval)
	for (j = 0; j < p; j++)
	{

		// A <- W.^2 + diag(W) * diag(W)' f.a. i!=j
		// 2. step: extract further diagonal entries of W to  A, A <- A+diagW*diagW'
		Ancol = A.ncol[j];
		pArowind = A.rowind[j];
		pAval = A.val[j];
		for (m = 0; m < Ancol; m++)
		{
			i = pArowind[m];
			if (i != j)
				// A_ij <- A_ij + W_ii*W_jj
				pAval[m] += diagW[i] * diagW[j];
		} // end for m

		// C <- X + \Delta
		// extract X+\Delta to C
		k = l = m = 0;
		Cncol = C.ncol[j];
		pCrowind = C.rowind[j];
		pCval = C.val[j];
		Dncol = D->ncol[j];
		pDrowind = D->rowind[j];
		pDval = D->val[j];
		Xncol = X.ncol[j];
		pXrowind = X.rowind[j];
		pXval = X.val[j];
		while ((k < Xncol || l < Dncol) && m < Cncol)
		{
			// we are seeking for X_ij and \Delta_ij
			i = pCrowind[m];
			// initially suppose that these values are 0
			Xij = Dij = 0.0;
			// scan through column j of X
			while (k < Xncol)
			{
				// did we pass or reach i?
				if (pXrowind[k] >= i)
				{
					// did we find X_ij?
					if (pXrowind[k] == i)
						Xij = pXval[k++];
					// done
					break;
				}
				else // advance to the next row in X(:,j)
					k++;
			} // end while
			// scan through column j of \Delta
			while (l < Dncol)
			{
				// did we pass or reach i?
				if (pDrowind[l] >= i)
				{
					// did we find \Delta_ij?
					if (pDrowind[l] == i)
						Dij = pDval[l++];
					// done
					break;
				}
				else // advance to the next row \Delta(:,j)
					l++;
			} // end while
			// C_ij<-X_ij+\Delta_ij
			pCval[m] = Xij + Dij;
			m++;
		} // end while
	}	  // end for j
	// end omp parallel for
#ifdef PRINT_INFO_BCDU
	printf("A, 2. step completed, C, completed, %8.1le [sec]\n", omp_get_wtime() - timeBegin);
	fflush(stdout);
	timeBegin = omp_get_wtime();
#endif

#define RANDOMIZE
#ifdef RANDOMIZE
	// randomize for better performance
	integer *colind = (integer *)malloc((size_t)p * sizeof(integer));
	for (j = 0; j < p; j++)
		colind[j] = j;
	for (j = 0; j < p; j++)
	{
		k = j + rand() % (p - j);
		kk = colind[j];
		colind[j] = colind[k];
		colind[k] = kk;
	} // end for j
	// for (j=0; j<p; j++)
	//    printf("%4ld",colind[j]);
	// printf("\n"); fflush(stdout);
#endif

	// -------------------------------------------------------
	// --------------------- MAIN LOOP -----------------------
	// B <- S - W + W' * \Delta * W
	// 3. step: finally we update B by B <- B + W' * \Delta * W on the pattern of B
	// interlace the computation with updating Delta using soft-thresholding, column-by-column
	integer ii, kkk, lll, *pi;
	double myLambda;
	for (jj = 0; jj < p; jj++)
	{
#ifdef RANDOMIZE
		j = colind[jj];
#else
		j = jj;
#endif
		// is there any need to compute W(:,i)'*\Delta*W(:,j) for some i?
		Bncol = B.ncol[j];
		if (Bncol > 0)
		{
			pBrowind = B.rowind[j];
			pBval = B.val[j];
			// compute sparse-sparse \Delta W_{:,j} and use buffers (val,idx,idxpos).
			// to do so we build sparse linear combinations U <- \sum_l \Delta_{:,l} W_lj
			// of the columns of \Delta
			// !!! also include M, since M covers the updates of \Delta for its strict
			// !!! lower triangular part that has not yet been updated
			// !!! thus also build sparse linear combinations U <- U+\sum_l M_{:,l} W_lj
			Wncol = W.ncol[j];
			pWrowind = W.rowind[j];
			pWval = W.val[j];
			cnt = 0;
			for (k = 0; k < Wncol; k++)
			{
				// row index lw of W_{lw,j}
				lw = pWrowind[k];
				// W_{lw,j}
				valw = pWval[k];
				// update buffer with W_{lw,j} * \Delta_{:,lw}
				Dncol = D->ncol[lw];
				pDrowind = D->rowind[lw];
				pDval = D->val[lw];
				for (m = 0; m < Dncol; m++)
				{
					// row index ld of \Delta_{ld,lw}
					ld = pDrowind[m];
					// check if index ld already exists in the auxiliary buffer
					r = idxpos[ld];
					// fill-in index?
					if (!r)
					{
						// store fill-in index (and value) at position cnt
						idx[cnt] = ld;
						// checkmark its position, for technical reasons we shift by 1
						idxpos[ld] = ++cnt;
					} // end if
					// for technical reasons, shift position back by 1
					r = idxpos[ld] - 1;
					// update by \Delta_{:,lw}W_{lw,j}
					val[r] += valw * pDval[m];
				} // end for m
				// !!! also update buffer with W_{lw,j} * M_{:,lw}
				Mncol = M.ncol[lw];
				pMrowind = M.rowind[lw];
				pMval = M.val[lw];
				for (m = 0; m < Mncol; m++)
				{
					// row index ld of M_{ld,lw}
					ld = pMrowind[m];
					// check if index ld already exists in the auxiliary buffer
					r = idxpos[ld];
					// fill-in index?
					if (!r)
					{
						// store fill-in index (and value) at position cnt
						idx[cnt] = ld;
						// checkmark its position, for technical reasons we shift by 1
						idxpos[ld] = ++cnt;
					} // end if
					// for technical reasons, shift position back by 1
					r = idxpos[ld] - 1;
					// update by M_{:,lw}W_{lw,j}
					val[r] += valw * pMval[m];
				} // end for m
			}	  // end for k
#ifdef PRINT_INFO1
			for (k = 0; k < cnt; k++)
				printf("%2d: %8.1le\n", idx[k], val[k]);
#endif
			// now that the sparse-sparse product is built we clear the checkmarks
			// and sort the arrays
			for (k = 0; k < cnt; k++)
			{
				ld = idx[k];
				idxpos[ld] = 0;
			} // end for k
			// now sort idx and val simultaneously such that idx is sorted in increasing order
			// use idxpos as stack
			qsort1_(val, idx, idxpos, &cnt);
			// clean up stack
			for (k = 0; k < cnt; k++)
				idxpos[k] = 0;
#ifdef PRINT_INFO1
			for (k = 0; k < cnt; k++)
				printf("%2d: %8.1le\n", idx[k], val[k]);
#endif
#ifdef PRINT_INFO_BCDU
#pragma omp single
			timeBeginlocal = omp_get_wtime();
#endif
			for (kk = 0; kk < Bncol; kk++)
			{
				i = pBrowind[kk];
				d = pBval[kk];
				Wncol = W.ncol[i];
				pWrowind = W.rowind[i];
				pWval = W.val[i];
				// now compute sparse-sparse scalar product W_{:,i}^T*val and update B_ij
				// recall that the sparse vectors have ascending indices
				k = l = 0;
				while (k < cnt && l < Wncol)
				{
					// row index ld of  \Delta W_{:,j}
					ld = idx[k];
					// row index lw of W_{:,i}
					lw = pWrowind[l];
					if (ld < lw)
						k++;
					else if (lw < ld)
						l++;
					else
					{ // indices match, update b by scalar product contribution
						d += pWval[l] * val[k];
						k++;
						l++;
					} // end if-elseif-else
				}	  // end while
				pBval[kk] = d;
				// now we have finally achieved B_ij = S_ij - W_ij + W_{:,i}^T \Delta W_{:,j}
#ifdef PRINT_INFO1
				printf("B(%ld,%ld)=%8.1le\n", i, j, B.val[j][kk]);
#endif
			} // end for kk
#ifdef PRINT_INFO_BCDU
#pragma omp single
			par_loop1 += omp_get_wtime() - timeBeginlocal;
#endif
			// finally also clean val array
			for (k = 0; k < cnt; k++)
				val[k] = 0.0;

			// now that we have commonly computed A, B, C we start updating Delta using soft-thresholding
			// note that at this stage, only column j of B is correctly computed, i.e., the second order
			// term W_{:,i}'\Delta W_{:,j} has only been added to columns j in {colind[0],...,colind[jj]}
			// of B. Changes in \Delta_ij by some mu require to update C_ij <- C_ij+mu as well as
			// B_{rj} <- B_{rj} + mu (W_{ir}W_{jj}+W_{rj}W_{ij}) for all r>i such that B_{rj} is
			// available and such that W_{ir} or (W_{jr} and W_{ij}) are available. This will make the
			// information in B_{:,j} become consistent again
			// B_{:,colind[s]} such that s<jj are not considered anymore, similarly B_{rj} s.t. r<=i.
			// B_{:,colind[s]} s.t. s>jj have not been computed anyway, thus these columns need not be
			// updated either
			m = k = 0;
			// row index to check for W_{ij}!=0
			q = 0;
			Dncol = D->ncol[j];
			pDrowind = D->rowind[j];
			pDval = D->val[j];
			// abbrevations for the current column of LambdaMatrix
			pi = LambdaMatrix->rowind[j];
			lll = LambdaMatrix->ncol[j];
			kkk = 0;
			while (m < A.ncol[j] && k < Dncol)
			{
				i = A.rowind[j][m];

				// find out whether W_ij exists
				Wij = 0.0;
				Wncol = W.ncol[j];
				pWrowind = W.rowind[j];
				pWval = W.val[j];
				while (q < Wncol)
				{
					if (pWrowind[q] >= i)
					{
						break;
					} // end if
					else
						q++;
				} // end while
				if (q < Wncol)
				{
					if (pWrowind[q] == i)
					{
						// extract current value \Delta_ij
						Wij = pWval[q++];
					} // end if
				}	  // end if

				// find out whether \Delta_ij exists
				Dij = 0.0;
				while (k < Dncol)
				{
					if (pDrowind[k] >= i)
					{
						break;
					} // end if
					else
						k++;
				} // end while
				if (k < Dncol)
				{
					if (pDrowind[k] == i)
					{
						// extract current value \Delta_ij
						Dij = pDval[k];
						// now compute values
						a = A.val[j][m];
						b = B.val[j][m];
						c = C.val[j][m];
						// parameters for optimization parameter mu = -c + s(c-f,valw)
						// where s(z,r)=sign(z)*max{|z|-r,0}

						// check for a different value of Lambda rather than the default value
						myLambda = Lambda;
						for (; kkk < lll; kkk++)
						{
							ii = pi[kkk];
							if (ii == i)
							{
								myLambda = LambdaMatrix->val[j][kkk];
								break;
							}
							else if (ii >= i)
								break;
						} // end for kkk

						if (RunTimeConfig.off_diagonal)
						{
							if (i == j)
								valw = 0.0;
							else
								valw = myLambda / a;
						} // end if
						else
							valw = myLambda / a;

#ifdef PRINT_INFO1
						printf("c=%8.1le\n", c);
#endif

						f = b / a;
						// downdate |D|_1
						normD -= fabs(Dij);
						// \Delta updated by \mu(e_ie_j^T+e_je_i^T)
						if (c > f)
						{
							mu = -f - valw;
							if (c + mu < 0.0)
								mu = -c;
						}
						else
						{
							mu = -f + valw;
							if (c + mu > 0.0)
								mu = -c;
						}
						// update D_ij
						Dij += mu;
						pDval[k] = Dij;
#ifdef PRINT_INFO1
						printf("Delta(%2d,%2d)=%8.1le\n", i + 1, j + 1, D->val[j][k]);
#endif
						k++;

						// increment |(\mu e_ie_j^T)_{i,j}|_1
						diffD += fabs(mu);
						// update |D|_1
						normD += fabs(Dij);

						// copy values to M with the memory structure of A' to hold mu
						// at position (j,i) instead of (i,j)
						if (i != j)
						{
							kk = M.ncol[i];
							M.rowind[i][kk] = j;
							M.val[i][kk] = mu;
							M.ncol[i] = kk + 1;
						} // end if

#ifdef PRINT_INFO1
						printf("diffD=%8.1le, normD=%8.1le\n", diffD, normD);
#endif
						// theoretically we need to update C
						// updating C <- C + mu (e_ie_j^T + e_je_i)^T
						// this only affects the upper triangular part, since the lower triangular
						// part is not stored anyway, but the upper triangular part is not referenced
						// anymore in future computations, therefore we can leave it as it is
						// C.val[j][m]+=mu;

						// now also update B
						// updating B <- B + mu (W_{i,:}^T (e_ie_j^T+e_je_i^T) W_{j,:})
						//             = B + mu (W_{i,:}^TW_{j,:}+W_{j,:}^TW_{i,:}) in column j, i.e.,
						//    B_{:,j} <- B_{:,j} + mu (W_{i,:}^TW_{jj}+W_{j,:}^TW_{ij})
						// for all r>i such that
						// 1) B_{rj} is available
						// AND
						// 2) W_{ir} OR   (W_{jr} AND W_{ij})   are available.
#ifdef PRINT_INFO_BCDU
#pragma omp single
						timeBeginlocal = omp_get_wtime();
#endif
						Bncol = B.ncol[j];
						pBrowind = B.rowind[j];
						pBval = B.val[j];
						for (mm = m + 1; mm < Bncol; mm++)
						{
							r = pBrowind[mm];
							Wncol = W.ncol[r];
							pWrowind = W.rowind[r];
							pWval = W.val[r];
							Wir = 0.0;
							Wjr = 0.0;
							// scan through column r of W to find W_{ir} and W_{jr} where i<=j
							for (ll = 0; ll < Wncol; ll++)
							{
								lw = pWrowind[ll];
								// did we reach or pass W_{ir}
								if (lw >= i)
								{
									if (lw == i)
									{
										// we did indeed meet Wir
										Wir = pWval[ll];
										// we can terminate the loop if Wij=0;
										if (Wij == 0.0)
											break;
									} // end if
								}	  // if
								// did we reach or pass W_{jr}
								if (lw >= j)
								{
									if (lw == j)
									{
										// we did indeed meet Wjr
										Wjr = pWval[ll];
									} // if
									break;
								} // if
							}	  // end for ll
							// B_{rj} <- B_{rj} + mu (W_{ir}W_{jj} + W_{jr}W_{ij})
							// in the diagonal index case (j,j) we only have a single
							// update
							if (i == j)
								pBval[mm] += mu * Wjr * Wij;
							else
								pBval[mm] += mu * (Wir * diagW[j] + Wjr * Wij);
						} // end for mm
#ifdef PRINT_INFO_BCDU
#pragma omp single
						par_loop2 += omp_get_wtime() - timeBeginlocal;
#endif
					} // end if
					else
					{
						printf("Delta_{%ld,%ld} is missing in the sparse data structure\n", i, j);
						exit(-1);
					} // end else
				}
				else
				{
					printf("Delta_{%ld,%ld} is missing in the sparse data structure\n", i, j);
					exit(-1);
				} // end else

				m++;
			} // end while
		}	  // end if
	}		  // end for jj
	// -------------------------------------------------------
	// ------------------- END MAIN LOOP ---------------------

#ifdef PRINT_INFO_BCDU
	printf("Delta, upper triangular part updated, %8.1le [sec], loop1: %8.1le [sec], loop2: %8.1le [sec]\n", omp_get_wtime() - timeBegin, par_loop1, par_loop2);
	fflush(stdout);
	timeBegin = omp_get_wtime();
#endif

#ifdef RANDOMIZE
	for (j = 0; j < p; j++)
	{
		// now sort M.rowind[j] and M.val[j] simultaneously such that
		// M.rowind[j] is sorted in increasing order and use idx as stack
		l = M.ncol[j];
		qsort1_(M.val[j], M.rowind[j], idx, &l);
	} // end for j
#endif
	// for symmetry reasons also update \Delta_{ji} using the transposed values stored in M
	// but skip diagonal part in order to avoid duplicate updates
#pragma omp parallel for shared(p, M, D) private(m, k, i, mu, Mncol, pMrowind, pMval, Dncol, pDrowind, pDval)
	for (j = 0; j < p; j++)
	{
		m = k = 0;
		Mncol = M.ncol[j];
		pMrowind = M.rowind[j];
		pMval = M.val[j];
		Dncol = D->ncol[j];
		pDrowind = D->rowind[j];
		pDval = D->val[j];
		while (m < Mncol && k < Dncol)
		{
			i = pMrowind[m];
			if (i != j)
			{ // should be satisfied by contruction
				mu = pMval[m];
				// find out whether \Delta_ij exists
				while (k < Dncol)
				{
					if (pDrowind[k] >= i)
					{
						break;
					} // end if
					else
						k++;
				} // end while
				if (k < Dncol)
				{
					if (pDrowind[k] == i)
					{
						pDval[k] += mu;
						k++;
					} // end if
					else
					{
						printf("Delta_{%ld,%ld} is missing in the sparse data structure\n", i, j);
						exit(-1);
					} // end else
				}
				else
				{
					printf("Delta_{%ld,%ld} is missing in the sparse data structure\n", i, j);
					exit(-1);
				} // end else
			}	  // end if

			m++;
		} // end while
	}	  // end for j
	// end omp parallel for

#ifdef PRINT_INFO_BCDU
	printf("Delta, strict lower triangular part updated, %8.1le [sec]\n", omp_get_wtime() - timeBegin);
	fflush(stdout);
#endif

	FreeSparse(&A);
	FreeSparse(&B);
	FreeSparse(&C);
	FreeSparse(&M);
	free(diagW);
#ifdef RANDOMIZE
	free(colind);
#endif

} // end BlockCoordinateDescentUpdate (slightly improved version)

#endif
