/* bninv.cpp
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

#ifndef _BLOCK_NEUMANN_INVERSION_
#define _BLOCK_NEUMANN_INVERSION_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip> // Precision

#include <math.h>

// Parallel
#include <omp.h>

#include "squic.hpp"

#ifdef _USE_MKL_
#include "mymkl.h"
#endif

using namespace std;

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define FABS(A) (((A) >= 0) ? (A) : (-(A)))

#define STDERR stderr
#define STDOUT stdout

#define ELBOW 10

// #define PRINT_CHECK
// #define PRINT_INFO
// #define PRINT_INFOnnz
// #define PRINT_INFO1
// #define PRINT_INFO2
// #define printf mexPrintf

//#define _PROFILING_

/* ====================================================================
   === bfspaisym Function =============================================
   ====================================================================

    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

    $Id: bfspaisym.c 4061 2018-03-04 21:38:56Z bolle $

    Notice:

        JANUS block ILU

	Copyright (c) 2018 by TU Braunschweig.  All Rights Reserved.

	THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
	EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

    Availability:

	This file is located at

	http://bilu.tu-bs.de/
	
*/

void SQUIC::BNINV(double droptol, double *SL)

{
	integer i, j, k, l, m, n, r, s, t, nnz, flag, *ia, *ja, *ja2, cnt, cnt2, cntb,
		ss, tt, ttt, i2, i3, jj, ii, kk, r2, nblocks, *block, ncols, mrows,
		*idxl, *idxl2, *idxposl, **idxl_src, **idxl2_src, **idxposl_src,
		*idxb, *idxbpos, **idxb_src, **idxbpos_src, *iLTbegin, **iLTbegin_src,
		mythreadnum, iL_inplace, iL_new_inplace, BiD_inplace;
	char *transa, *transb;
	size_t *iL_size_rowind, *iL_size_valE,
		*size_vall_src, *size_vall2_src,
		*iL_new_size_rowind, *iL_new_size_valE;
	double nrm, *nrml, mytol, buff, *valr;
	double *a, alpha, beta, *pr,
		*Ainv_valuesR, *vall, **vall_src, *vall2, **vall2_src;
#ifdef _PROFILING_
	double timeBegin,
		timeBeginLocal,
		time_bfspai,
		time_neumann = 0.0,
		time_matrix_matrix = 0.0,
		time_matrix_matrix1 = 0.0,
		time_matrix_matrix2 = 0.0,
		time_iLT;

	time_bfspai = omp_get_wtime();
#endif

	// use a finer drop tolerance locally
	// mytol=0.1*droptol;
	mytol = droptol;
	if (RunTimeConfig.verbose > 2)
	{
		printf("- use tolerance %8.1le for approximate inversion\n", mytol);
		fflush(stdout);
	} // end if

	// extract elements from PREC

	// 0. n
	n = BL.nc;
#ifdef PRINT_INFO
	printf("BNINV: n=%ld\n", n);
	fflush(stdout);
#endif
#ifdef PRINT_INFO2
	if (SL != NULL)
	{
		printf("SL:\n");
		fflush(stdout);
		for (i = 0; i < n; i++)
			printf("%12.4le", SL[i]);
		printf("\n");
		fflush(stdout);
	}
	printf("p:\n");
	fflush(stdout);
	for (i = 0; i < n; i++)
		// printf("%12d",perm[i]);
		printf("%12d", P.rowind[i][0]);
	printf("\n");
	fflush(stdout);

	printf("BL:\n");
	fflush(stdout);
	PrintBlockSparse(&BL);
	printf("\n");
	fflush(stdout);
	printf("BiD:\n");
	fflush(stdout);
	PrintBlockSparse(&BiD);
	printf("\n");
	fflush(stdout);
#endif

	// 1. BL
	nblocks = BL.nblocks;
#ifdef PRINT_INFO
	printf("BNINV: nblocks=%ld\n", nblocks);
	fflush(stdout);
#endif

	// 2. BiD
	// 3. p
	// 4. SL might be NULL

	// approximate inverse iL~L^{-1}
	SparseBlockMatrix iL, iL_new;

	iL.nr = iL.nc = n;
	iL.nblocks = nblocks;
	iL.nblockcol = BL.nblockcol;
	iL.nblockrow = (integer *)malloc((size_t)nblocks * sizeof(integer));
	iL.colind = BL.colind;
	iL.rowind = (integer **)malloc((size_t)nblocks * sizeof(integer *));
	iL.valD = NULL;
	iL.valE = (double **)malloc((size_t)nblocks * sizeof(double *));
	iL_size_rowind = (size_t *)malloc((size_t)nblocks * sizeof(size_t));
	iL_size_valE = (size_t *)malloc((size_t)nblocks * sizeof(size_t));

	iL_new.nr = iL_new.nc = n;
	iL_new.nblocks = nblocks;
	iL_new.nblockcol = iL.nblockcol;
	iL_new.nblockrow = (integer *)malloc((size_t)nblocks * sizeof(integer));
	iL_new.colind = iL.colind;
	iL_new.rowind = (integer **)malloc((size_t)nblocks * sizeof(integer *));
	iL_new.valD = NULL;
	iL_new.valE = (double **)malloc((size_t)nblocks * sizeof(double *));
	iL_new_size_rowind = (size_t *)malloc((size_t)nblocks * sizeof(size_t));
	iL_new_size_valE = (size_t *)malloc((size_t)nblocks * sizeof(size_t));

	// invert L using a truncated Neumann series until the error drops
	// below mytol
	// initial pattern is that of L, except that the entries below the
	// diagonal part have flipped sign
	// iL=I-tril(L,-1)
	// inverse mapping index -> block number
	block = (integer *)calloc((size_t)n + 1, sizeof(integer));
#pragma omp parallel for shared(nblocks, BL, iL, block, iL_size_rowind, iL_size_valE, iL_new, iL_new_size_rowind, iL_new_size_valE, n) private(ja, ncols, k, i, mrows, a)
	for (j = 0; j < nblocks; j++)
	{

		// set up block reference for block column j
		ja = BL.colind[j];
		ncols = BL.nblockcol[j];
		for (k = 0; k < ncols; k++)
		{
			i = ja[k];
			// block j covers index i
			block[i] = j;
#ifdef PRINT_CHECK
			if (i < 0 || i >= n)
				printf(" 1. scalar index out of range when referring to block[%ld]=%ld\n", i, j);
			fflush(stdout);
			if (j < 0 || j >= nblocks)
				printf(" 1. block index out of range when referring to block[%ld]=%ld\n", i, j);
			fflush(stdout);
#endif
		} // end for k

		// copy BL.rowind[j], -BL.valE[j]
		mrows = BL.nblockrow[j];
		iL.nblockrow[j] = mrows;
		iL_size_rowind[j] = 2 * mrows + ELBOW;
		iL.rowind[j] = (integer *)malloc(iL_size_rowind[j] * sizeof(integer));
		memcpy(iL.rowind[j], BL.rowind[j], mrows * sizeof(integer));
		iL_size_valE[j] = (2 * mrows + ELBOW) * ncols;
		iL.valE[j] = (double *)malloc(iL_size_valE[j] * sizeof(double));
		memcpy(iL.valE[j], BL.valE[j], mrows * ncols * sizeof(double));
		// switch sign
		k = mrows * ncols;
		a = iL.valE[j];
		for (i = 0; i < k; i++, a++)
			*a = -*a;
		iL_new.nblockrow[j] = 0;
		iL_new.rowind[j] = NULL;
		iL_new_size_rowind[j] = 0;
		;
		iL_new.valE[j] = NULL;
		iL_new_size_valE[j] = 0;
	} // end for j
	  // end omp parallel for

#ifdef PRINT_INFO
	printf("BNINV: iL=I-tril(L,-1) done\n");
	fflush(stdout);
	// PrintBlockSparse(&iL);fflush(stdout);
#endif

	// arrays idxl, idxposl and vall, vall2 are used to uncompress sparse arrays and
	// in particular idxposl is used to avoid duplicate entries when more than
	// one sparse vector is uncompressed
	// idxposl must be initialized with 0 (and always reset to 0)

	// allocate additional memory for multithreaded Neumann series
	k = omp_get_max_threads();
	// printf("maximum number of available threads=%ld\n",k);
	idxl_src = (integer **)malloc((size_t)k * sizeof(integer *));
	idxl2_src = (integer **)malloc((size_t)k * sizeof(integer *));
	idxposl_src = (integer **)malloc((size_t)k * sizeof(integer *));
	vall_src = (double **)malloc((size_t)k * sizeof(double *));
	vall2_src = (double **)malloc((size_t)k * sizeof(double *));
	nrml = (double *)malloc((size_t)k * sizeof(double));
	size_vall_src = (size_t *)malloc((size_t)k * sizeof(size_t));
	size_vall2_src = (size_t *)malloc((size_t)k * sizeof(size_t));
	for (i = 0; i < k; i++)
	{
		idxl_src[i] = (integer *)malloc((size_t)n * sizeof(integer));
		idxl2_src[i] = (integer *)malloc((size_t)n * sizeof(integer));
		// as always, idxposl must be initialized with 0
		idxposl_src[i] = (integer *)calloc((size_t)n, sizeof(integer));
		size_vall_src[i] = n * ELBOW;
		vall_src[i] = (double *)malloc(size_vall_src[i] * sizeof(double));
		size_vall2_src[i] = n * ELBOW;
		vall2_src[i] = (double *)malloc(size_vall2_src[i] * sizeof(double));
	} // end for i

#ifdef _PROFILING_
	timeBegin = omp_get_wtime();
#endif
	// flag for termination of Neumann series
	flag = -1;
	// counter for the number of powers
	m = 1;
	while (flag)
	{
		// count maximum off-diagonal error between two powers
		// here first build maximum locally for each thread and finally maximize
		k = omp_get_max_threads();
		for (i = 0; i < k; i++)
			nrml[i] = 0.0;

			// Horner scheme iL_new <- iL*(I-PREC.BL)+I
#pragma omp parallel for shared(nblocks, n, BL, idxl_src, idxl2_src, idxposl_src, vall_src, vall2_src, block, iL_new, iL_new_size_rowind, iL_new_size_valE, size_vall_src, iL, size_vall2_src, nrml) private(mythreadnum, idxl, idxl2, idxposl, vall, vall2, r, ncols, mrows, ia, cnt, l, s, r2, cnt2, ja, jj, t, k, ii, i2, tt, a, transa, transb, alpha, beta, buff, pr, iL_inplace, iL_new_inplace)
		for (j = 0; j < nblocks; j++)
		{
			// who am I
			mythreadnum = omp_get_thread_num();

			// local versions of idx, idxpos, val to avoid memory conflicts
			idxl = idxl_src[mythreadnum];
			idxl2 = idxl2_src[mythreadnum];
			idxposl = idxposl_src[mythreadnum];
			vall = vall_src[mythreadnum];
			vall2 = vall2_src[mythreadnum];
#ifdef PRINT_CHECK
			for (r = 0; r < n; r++)
			{
				if (idxposl[r])
					printf("idxposl[%ld] still dirty\n", r + 1);
				idxposl[r] = 0;
			} // end for r
#endif

			// BL{j}
			ncols = BL.nblockcol[j];
			mrows = BL.nblockrow[j];
			// row index set BL{j}
			ia = BL.rowind[j];

			// first pass, determine nonzero row pattern
			// counter for fill
			cnt = 0;
			// always initialize with pattern of row indices of BL{j}
			for (r = 0; r < mrows; r++)
			{
				// grab nonzero row indices of BL{j}
				l = ia[r];
				// add index
				idxl[cnt] = l;
				// check mark nonzero, for technical reasons shifted by 1
				idxposl[l] = ++cnt;
			} // end for r
			r = 0;
			while (r < mrows)
			{
				// grab nonzero row indices of BL{j}
				l = ia[r];
				// associated block column of iL
				s = block[l];
#ifdef PRINT_CHECK
				if (l < 0 || l >= n)
					printf(" 2. scalar index out of range when referring to %ld=block[%ld]\n", s, l);
				fflush(stdout);
				if (s < 0 || s >= nblocks)
					printf(" 2. block index out of range when referring to %ld=block[%ld]\n", s, l);
				fflush(stdout);
#endif
				// make sure that this block column has not been considered yet
				r2 = r + 1;
				while (r2 < mrows)
				{
					l = ia[r2];
#ifdef PRINT_CHECK
					if (l < 0 || l >= n)
						printf(" 3. scalar index out of range when referring to %ld!=block[%ld]\n", s, l);
					fflush(stdout);
					if (s < 0 || s >= nblocks)
						printf(" 3. block index out of range when referring to %ld!=block[%ld]\n", s, l);
					fflush(stdout);
#endif
					if (block[l] != s)
						break;
					r2++;
				} // end while r2
				// scan block column s of iL for additional nonzero row indices
				cnt2 = 0;
				// short cuts
				ja = iL.rowind[s];
				jj = iL.nblockrow[s];
				for (k = 0; k < jj; k++)
				{
					// row index l inside block s of iL{s}
					l = ja[k];
					// check whether l is already a fill position
					t = idxposl[l];
					// no
					if (!t)
					{
						// add index and store it separately in idxl2
						idxl2[cnt2++] = l;
					} // end if
				}	  // end for k

				// now idxl[0,...,cnt-1] and idxl2[0,...,cnt2-1] are
				// distinct index arrays each with indices in increasing order,
				// we merge them in idxl
				// last positions
				jj = cnt - 1;
				ii = cnt2 - 1;
				// total number of nonzeros=position behind the last entry
				i2 = cnt + cnt2;
				// last existing index though it exists
				if (jj >= 0)
					tt = idxl[jj];
				else
					tt = -1;
				// last new index
				if (ii >= 0)
					t = idxl2[ii];
				while (ii >= 0)
				{
					// new index greater than given index
					if (t > tt)
					{
						idxposl[t] = i2--;
						idxl[i2] = t;
						ii--;
						// get next new row index
						if (ii >= 0)
							t = idxl2[ii];
					}
					else
					{ // tt>t, old index behind new index
						while (tt > t)
						{
							idxposl[tt] = i2--;
							idxl[i2] = tt;
							jj--;
							// get "next" old row index though it exists
							if (jj >= 0)
								tt = idxl[jj];
							else
								tt = -1;
						} // end while tt>t
					}	  // end if-else t>tt
				}		  // end while i>=0
				cnt += cnt2;

				// advance to the next sub-diagonal block inside block column j
				r = r2;
			} // end while r

			// adjust memory for up to cnt new rows
			iL_new.nblockrow[j] = cnt;
			// iL_new{j} used for the first time?
			if (iL_new.rowind[j] == NULL)
			{
				iL_new_size_rowind[j] = 2 * cnt + 1 + ELBOW;
				iL_new.rowind[j] = (integer *)malloc(iL_new_size_rowind[j] * sizeof(integer));
			}
			else
			{
				iL_new_size_rowind[j] = MAX(cnt, iL_new_size_rowind[j]);
				iL_new.rowind[j] = (integer *)realloc(iL_new.rowind[j], iL_new_size_rowind[j] * sizeof(integer));
			}
			if (iL_new.valE[j] == NULL)
			{
				iL_new_size_valE[j] = (2 * cnt + 1 + ELBOW) * ncols;
				iL_new.valE[j] = (double *)malloc(iL_new_size_valE[j] * sizeof(double));
			}
			else
			{
				iL_new_size_valE[j] = MAX(cnt * ncols, iL_new_size_valE[j]);
				iL_new.valE[j] = (double *)realloc(iL_new.valE[j], iL_new_size_valE[j] * sizeof(double));
			}

			// init numerical values
			a = iL_new.valE[j];
			k = cnt * ncols;
			for (r = 0; r < k; r++)
				a[r] = 0.0;
			// copy numerical values of -BL{j}
			pr = BL.valE[j];
			// remember that idxposl is shifted by 1
			a--;
			for (r = 0; r < ncols; r++, pr += mrows, a += cnt)
			{
				// scatter column r of -BL{j}
				for (k = 0; k < mrows; k++)
				{
					l = ia[k];
					// idxposl also starts with 1, but a is decreased by 1
					t = idxposl[l];
#ifdef PRINT_CHECK
					if (t - 1 < 0 || t - 1 >= cnt)
					{
						printf("position %ld out of range for row %ld\n", t, l + 1);
						fflush(stdout);
					}
#endif
					a[t] = -pr[k];
				} // end for k
			}	  // end for r
			pr = BL.valE[j];

			// second pass, compute j-th block column of iL_new <- iL*(I-L)+I
			// scatter data from block column i of iL to buffer
			r = 0;
			while (r < mrows)
			{
				// grab nonzero row indices of BL{j}
				l = ia[r];
				// associated block column of iL
				s = block[l];
#ifdef PRINT_CHECK
				if (l < 0 || l >= n)
					printf(" 4. scalar index out of range when referring to %ld=block[%ld]\n", s, l);
				fflush(stdout);
				if (s < 0 || s >= nblocks)
					printf(" 4. block index out of range when referring to %ld=block[%ld]\n", s, l);
				fflush(stdout);
#endif
				// make sure that this block column has not been considered yet
				r2 = r + 1;
				while (r2 < mrows)
				{
					l = ia[r2];
#ifdef PRINT_CHECK
					if (l < 0 || l >= n)
						printf(" 5. scalar index out of range when referring to %ld!=block[%ld]\n", s, l);
					fflush(stdout);
					if (s < 0 || s >= nblocks)
						printf(" 5. block index out of range when referring to %ld!=block[%ld]\n", s, l);
					fflush(stdout);
#endif
					if (block[l] != s)
						break;
					r2++;
				} // end while r2
				l = ia[r2 - 1] - ia[r] + 1;
				if (l == r2 - r)
				{
					iL_inplace = -1;
#ifdef PRINT_INFO2
					printf("iL{%ld} need not be copied, since %ld...%ld are %ld contiguous indices\n",
						   s + 1, ia[r] + 1, ia[r2 - 1] + 1, r2 - r);
					fflush(stdout);
#endif
				}
				else
					iL_inplace = 0;

				// extract associated sub-diagonal rows inside block column s of iL
				// make sure that the buffer is big enough
				jj = iL.nblockrow[s];
				if (jj > 0)
				{
					ja = iL.colind[s];
					if (iL_inplace)
					{
						i2 = ia[r];
						vall = iL.valE[s] + jj * (i2 - ja[0]);
					}
					else
					{ // a copy of selected columns of iL{s} is required
						size_vall_src[mythreadnum] = MAX(size_vall_src[mythreadnum], jj * (r2 - r));
						vall = vall_src[mythreadnum];
						vall_src[mythreadnum] = (double *)realloc(vall, size_vall_src[mythreadnum] * sizeof(double));

						// extract sub-diagonal part of iL{s} and make sure that only
						// those columns of iL{s} are taken into account that are
						// associated with the subdiagonal block s inside BL{j}
						vall = vall_src[mythreadnum];
						for (l = r; l < r2; l++, vall += jj)
						{
							// grab nonzero row indices of BL{j} associated with block s
							i2 = ia[l];
							// i2 refers to a column index i2 of block column s inside
							// iL, this block column is physically stored at position
							// i2-ja[0]
							a = iL.valE[s] + jj * (i2 - ja[0]);
							memcpy(vall, a, jj * sizeof(double));
						} // end for l
						vall = vall_src[mythreadnum];
					} // end if-else iL_inplace

					// gather rows of iL_new{j} associated with the sub-diagonal
					// row indices of iL{s}
					ja = iL.rowind[s];
					if (idxposl[ja[jj - 1]] - idxposl[ja[0]] + 1 == jj)
					{
						iL_new_inplace = -1;
#ifdef PRINT_INFO2
						printf("block %ld of iL_new{%ld} need not be copied, since %ld...%ld are %ld contiguous indices\n",
							   s + 1, j + 1, ja[0] + 1, ja[jj - 1] + 1, jj);
						fflush(stdout);
						/*
			  printf("required indices\n");fflush(stdout);
			  for (l=0; l<jj; l++)
			      printf("%12ld",ja[l]+1);
			  printf("\n");fflush(stdout);
			  printf("new positions\n");fflush(stdout);
			  for (l=0; l<jj; l++)
			      printf("%12ld",idxposl[ja[l]]);
			  printf("\n");fflush(stdout);
			  */
#endif
					}
					else
						iL_new_inplace = 0;
					// iL_new_inplace=0;

					// level-1 BLAS case, AXPY or sparse AXPYI
					// iL_new{j} = -1*iL{s}*BL{j}+1*iL_new{j} reduces to
					// iL_new{j} = -BL{j}[r]*iL{s} +iL_new{j}
					// vector    =   scalar *vector+vector
					if (ncols == 1 && r2 - r == 1)
					{
						// AXPY-case iL_new{j} = -BL{j}*iL{s} + iL_new{j}
						if (iL_new_inplace)
						{
							i2 = ja[0];
							t = idxposl[i2];
							a = iL_new.valE[j] - 1 + t;
							// call AXPY
							// iL_new{j} = -BL{j}*iL{s} + iL_new{j}
							//         a = -pr[r]*vall  + a
							alpha = -pr[r];
							l = 1;
#ifdef _BLAS_LAPACK_32_
							int myjj = jj, myl = l;
							daxpy_(&myjj, &alpha, vall, &myl, a, &myl);
#else
							daxpy_(&jj, &alpha, vall, &l, a, &l);
#endif
						} // end if iL_new_inplace
						else
						{ // we need to perform a sparse AXPYI
#ifdef _USE_MKL_
							// use sparse AXPYI iL_new{j}[idxl2]= -pr[r] * vall + iL_new{j}[idxl2]
							for (l = 0; l < jj; l++)
							{
								// required row from block s of iL
								i2 = ja[l];
								// physical position t inside iL_new{j}, recall
								// that idxposl is shifted by 1, but a is adjusted
								// appropriately, this position must exist by
								// construction of the set of row indices for
								// iL_new{j}
								idxl2[l] = idxposl[i2];
							} // end for l
							a = iL_new.valE[j] - 1;
							// iL_new{j}[idxl2] = -pr[r] * vall + iL_new{j}[idxl2]
							alpha = -pr[r];
#ifdef _BLAS_LAPACK_32_
							int myjj = jj, mykk,
								*myidxl2 = (int *)malloc((size_t)jj * sizeof(int));
							for (mykk = 0; mykk < myjj; mykk++)
								myidxl2[mykk] = idxl2[mykk];
							cblas_daxpyi(myjj, alpha, vall, myidxl2, a);
							free(myidxl2);
#else
							cblas_daxpyi(jj, alpha, vall, idxl2, a);
#endif

#else // hand-coded loop \
	  // update iL_new{j}[idxposl[ja]] += -pr[r] * vall
							alpha = -pr[r];

							// position of the numerical values of iL_new{j} adjusted by
							// 1 beforehand
							a = iL_new.valE[j] - 1;
							for (l = 0; l < jj; l++)
							{
								// required row from block s of iL
								i2 = ja[l];
								// physical position t inside iL_new{j}, recall
								// that idxposl is shifted by 1, but a is adjusted
								// appropriately, this position must exist by
								// construction of the set of row indices for
								// iL_new{j}
								t = idxposl[i2];
								a[t] += alpha * vall[l];
							} // end for l
#endif
						} // end if-else iL_new_inplace
					}	  // end if scalar case
					// level-2 BLAS GEMV case
					// else if (ncols==1) {
					// }
					// level-3 BLAS GEMM case
					else
					{
						if (iL_new_inplace)
						{
							i2 = ja[0];
							t = idxposl[i2];
							a = iL_new.valE[j] - 1 + t;
#ifdef PRINT_INFO2
							printf("no gather required\n");
							fflush(stdout);
#endif
						}
						else
						{ // we need to copy the rows of iL_new{j} associated with block s
#ifdef PRINT_INFO2
							printf("perform gather\n");
							fflush(stdout);
#endif
							size_vall2_src[mythreadnum] = MAX(size_vall2_src[mythreadnum], jj * ncols);
							vall2 = vall2_src[mythreadnum];
							vall2_src[mythreadnum] = (double *)realloc(vall2, size_vall2_src[mythreadnum] * sizeof(double));
							vall2 = vall2_src[mythreadnum];
							for (l = 0; l < jj; l++)
							{
								// required row from block s of iL
								i2 = ja[l];
								// physical position t inside iL_new{j}, recall
								// that idxposl is shifted by 1, but a is adjusted
								// appropriately, this position must exist by
								// construction of the set of row indices for
								// iL_new{j}
								idxl2[l] = idxposl[i2];
							} // end for l
							// position of the numerical values of iL_new{j} adjusted by
							// 1 beforehand
							a = iL_new.valE[j] - 1;
							for (k = 0; k < ncols; k++, vall2 += jj, a += cnt)
							{
#ifdef _USE_MKL_
								// gather column k from a into vall2
#ifdef _BLAS_LAPACK_32_
								int myjj = jj, mykk,
									*myidxl2 = (int *)malloc((size_t)jj * sizeof(int));
								for (mykk = 0; mykk < myjj; mykk++)
									myidxl2[mykk] = idxl2[mykk];
								cblas_dgthr(myjj, a, vall2, myidxl2);
								free(myidxl2);
#else
								cblas_dgthr(jj, a, vall2, idxl2);
#endif
#else
								for (l = 0; l < jj; l++)
								{
									// required row from block s of iL
									// physical position t inside iL_new{j}, recall
									// that idxposl is shifted by 1, but a is adjusted
									// appropriately, this position must exist by
									// construction of the set of row indices for
									// iL_new{j}
									t = idxl2[l];
#ifdef PRINT_CHECK
									if (t - 1 < 0 || t - 1 >= cnt)
									{
										printf("position %ld out of range for row %ld\n", t, l + 1);
										fflush(stdout);
									}
#endif
									vall2[l] = a[t];
								} // end for l
#endif
							} // end for k
							vall2 = vall2_src[mythreadnum];
						} // end if-else iL_new_inplace

						// call level-3 BLAS
						// iL_new{j} = -1*iL{s}*BL{j}+1*iL_new{j}
						//     vall2 = -1*vall *BL{j}+1*vall2
						transa = "n";
						transb = "n";
						alpha = -1.0;
						beta = 1.0;
						l = r2 - r;
						if (iL_new_inplace)
						{
#ifdef PRINT_INFO2
							printf("%ld,%ld,%ld,%ld\n", jj, ncols, l, cnt);
							fflush(stdout);
							for (i2 = 0; i2 < jj; i2++)
							{
								for (k = 0; k < ncols; k++)
									printf("%12.4le", a[i2 + k * cnt]);
								printf("\n");
								fflush(stdout);
							}
#endif
#ifdef _BLAS_LAPACK_32_
							int myjj = jj, myncols = ncols, myl = l, mymrows = mrows, mycnt = cnt;
							dgemm_(transa, transb, &myjj, &myncols, &myl, &alpha,
								   vall, &myjj, pr + r, &mymrows, &beta,
								   a, &mycnt, 1, 1);
#else
							dgemm_(transa, transb, &jj, &ncols, &l, &alpha,
								   vall, &jj, pr + r, &mrows, &beta,
								   a, &cnt, 1, 1);
#endif
						}
						else
						{
#ifdef _BLAS_LAPACK_32_
							int myjj = jj, myncols = ncols, myl = l, mymrows = mrows;
							dgemm_(transa, transb, &myjj, &myncols, &myl, &alpha,
								   vall, &myjj, pr + r, &mymrows, &beta,
								   vall2, &myjj, 1, 1);
#else
							dgemm_(transa, transb, &jj, &ncols, &l, &alpha,
								   vall, &jj, pr + r, &mrows, &beta,
								   vall2, &jj, 1, 1);
#endif
						} // end else

						if (!iL_new_inplace)
						{
#ifdef PRINT_INFO2
							printf("perform scatter\n");
							fflush(stdout);
#endif
							// scatter numerical values back to iL_new{j}
							// position of the numerical values of iL_new{j} adjusted by
							// 1 beforehand
							a = iL_new.valE[j] - 1;
							for (k = 0; k < ncols; k++, vall2 += jj, a += cnt)
							{
#ifdef _USE_MKL_
								// scatter column k from vall2 into a
#ifdef _BLAS_LAPACK_32_
								int myjj = jj, mykk,
									*myidxl2 = (int *)malloc((size_t)jj * sizeof(int));
								for (mykk = 0; mykk < myjj; mykk++)
									myidxl2[mykk] = idxl2[mykk];
								cblas_dsctr(myjj, vall2, myidxl2, a);
								free(myidxl2);
#else
								cblas_dsctr(jj, vall2, idxl2, a);
#endif
#else
								for (l = 0; l < jj; l++)
								{
									// required row from block s of iL
									// physical position t inside iL_new{j}, recall
									// that idxposl is shifted by 1, but a is adjusted
									// appropriately, this position must exist by
									// construction of the set of row indices for
									// iL_new{j}
									t = idxl2[l];
#ifdef PRINT_CHECK
									if (t - 1 < 0 || t - 1 >= cnt)
									{
										printf("position %ld out of range for row %ld\n", t, l + 1);
										fflush(stdout);
									}
#endif
									a[t] = vall2[l];
								} // end for l
#endif
							} // end for k
						}	  // end if !iL_new_inplace
						else
						{
#ifdef PRINT_INFO2
							printf("no scatter required\n");
							fflush(stdout);
#endif
						} // end if-else !iL_new_inplace
					}	  // end if-elseif-else scalar case
				}		  // end if jj>0

				// advance to the next block column
				r = r2;
			} // end while r
			// computation of iL_new{j} <- iL*(I{:,j}-BL{j})+I{:,j} done

			// update maximum sub-diagonal error for the existing pattern
			ja = iL.rowind[j];
			jj = iL.nblockrow[j];
			for (r = 0; r < jj; r++)
			{
				// row index l of block column j inside iL
				l = ja[r];
				// does there exist a fill-in value
				t = idxposl[l];
				// iL{j} starting at row l, phyiscally starting at position r
				a = iL.valE[j] + r;
				if (t)
				{ // yes
					// iL_new{j} starting at row l, phyiscally starting at position t-1
					vall = iL_new.valE[j] + t - 1;
					for (s = 0; s < ncols; s++, a += jj, vall += cnt)
						nrml[mythreadnum] = MAX(nrml[mythreadnum], FABS(*a - *vall));
					// flag idxposl[l] as being used for nrm update
					idxposl[l] = -1;
				}
				else // whoops! a drop out case, this should rarely happen, maybe never
					for (s = 0; s < ncols; s++, a += jj)
						nrml[mythreadnum] = MAX(nrml[mythreadnum], FABS(*a));
			} // end for r

			// sparsify iL_new{j}
			vall = iL_new.valE[j];
			ja = iL_new.rowind[j];
			// scalar column
			if (ncols == 1)
			{
				s = 0;
				for (r = 0; r < cnt; r++)
				{
					// new row index l
					l = idxl[r];
					// associated numerical value
					alpha = vall[r];
					// maintain large entries in absolute value and their row indices
					buff = FABS(alpha);
					if (buff > 0.1 * mytol)
					{
						vall[s] = alpha;
						ja[s++] = l;
					} // end if
					// val[r] is a real fill-in, update nrm
					if (idxposl[l] > 0)
						nrml[mythreadnum] = MAX(nrml[mythreadnum], buff);
					// reset buffer
					idxposl[l] = 0;
				} // end for r
			}	  // end if scalar column case
			else
			{ // block column case
				// first pass: detect rows of small norm
				for (r = 0; r < cnt; r++)
				{
					// new row index l
					l = idxl[r];
					// maximum entry in row r of iL_new{j} physically starting at position r
#ifdef _BLAS_LAPACK_32_
					int myncols = ncols, mycnt = cnt;
					ii = idamax_(&myncols, vall + r, &mycnt) - 1;
#else
					ii = idamax_(&ncols, vall + r, &cnt) - 1;
#endif
					// flag index if the value is sufficiently small
					buff = FABS(vall[r + ii * cnt]);
					if (buff <= 0.1 * mytol)
						idxl[r] = -1;
					// vall[r] is a real fill-in, update nrm
					if (idxposl[l] > 0)
						nrml[mythreadnum] = MAX(nrml[mythreadnum], buff);
					// reset buffer
					idxposl[l] = 0;
				} // end for r
				// second pass: scratch out rows of small norm
				// pointer to the remaining numerical values
				pr = vall;
				// proceed column by column since the numerical values are stored
				// by columns
				for (k = 0; k < ncols; k++, vall += cnt)
				{
					// reduce column k
					for (r = 0; r < cnt; r++)
					{
						// new index l
						l = idxl[r];
						// this row should be kept
						if (l >= 0)
							*pr++ = vall[r];
					} // end for r
				}	  // end for k
				// third pass: scratch out indices of rows that were dropped
				s = 0;
				for (r = 0; r < cnt; r++)
				{
					// new index l
					l = idxl[r];
					if (l >= 0)
						ja[s++] = l;
				} // end for r
			}	  // end if-else ncols=1
			iL_new.nblockrow[j] = s;
			// now the computation of iL_new{j} is finished including dropping
			// rows with all entries of small size in absolute value
		} // end for j
		// end omp parallel for

#ifdef PRINT_INFO
		printf("SYMbfspai: iL_new=iL*(I-PREC.BL)+I done\n");
		fflush(stdout);
		// PrintBlockSparse(&iL_new);fflush(stdout);
#endif

		// swap pointers of iL and iL_new
#pragma omp parallel for shared(nblocks, iL, iL_new, iL_size_rowind, iL_new_size_rowind, iL_size_valE, iL_new_size_valE) private(i, a, ja, kk)
		for (j = 0; j < nblocks; j++)
		{
			i = iL.nblockrow[j];
			iL.nblockrow[j] = iL_new.nblockrow[j];
			iL_new.nblockrow[j] = i;

			ja = iL.rowind[j];
			iL.rowind[j] = iL_new.rowind[j];
			iL_new.rowind[j] = ja;

			a = iL.valE[j];
			iL.valE[j] = iL_new.valE[j];
			iL_new.valE[j] = a;

			kk = iL_size_rowind[j];
			iL_size_rowind[j] = iL_new_size_rowind[j];
			iL_new_size_rowind[j] = kk;

			kk = iL_size_valE[j];
			iL_size_valE[j] = iL_new_size_valE[j];
			iL_new_size_valE[j] = kk;
		} // end for j
		// end omp parallel for

		// accumulate local thread-dependent off-diagonal errors between two powers
		k = omp_get_max_threads();
		nrm = 0.0;
		for (i = 0; i < k; i++)
			nrm = MAX(nrml[i], nrm);

		// increase number of power in the Neumann series
		m++;
#ifdef PRINT_INFO
		printf("BNINV: power m=%2ld, nrm=%8.1le\n", m, nrm);
		fflush(stdout);
#endif
		if (RunTimeConfig.verbose > 2)
		{
			integer myiTemp = 0;
			for (integer j = 0; j < nblocks; ++j)
			{
				myiTemp += (iL_new.nblockrow[j] + .5 * iL_new.nblockcol[j] + .5) * iL_new.nblockcol[j];
			} // end for j

			printf("- BNINV: power m=%2ld, nrm=%8.1le, nnz(iL)/p=%8.1le, nnz(L)/p=%8.1le\n", m, nrm, myiTemp / (double)p, (BL.nnz + 0.5 * BiD.nnz + 0.5 * p) / (double)p);
			fflush(stdout);
		} // end if

		// check maximum off-diagonal error between two Neumann series to stop
		if (nrm <= mytol)
			flag = 0;
	} // end while flag
#ifdef _PROFILING_
	time_neumann = omp_get_wtime() - timeBegin;
#endif

	// no flexible memory for iL/iL_new required anymore
	free(iL_size_rowind);
	free(iL_size_valE);
	free(iL_new_size_rowind);
	free(iL_new_size_valE);

#ifdef PRINT_INFO
	printf("SYMbfspai: sparsify iL more aggressively\n");
	fflush(stdout);
	// PrintBlockSparse(&iL);fflush(stdout);
#endif

	// sparsify iL more aggressively
#pragma omp parallel for shared(nblocks, iL, mytol) private(vall, ja, mrows, ncols, r, l, ii, buff, pr, k, cnt, alpha)
	for (j = 0; j < nblocks; j++)
	{
#ifdef PRINT_INFO2
		printf("1. pass, thread=%ld, j=%ld, mrows=%ld\n", omp_get_thread_num(), j, iL.nblockrow[j]);
		fflush(stdout);
#endif

		// sparsify iL{j}
		vall = iL.valE[j];
		ja = iL.rowind[j];
		mrows = iL.nblockrow[j];
		ncols = iL.nblockcol[j];
		// scalar column
		if (ncols == 1)
		{
			cnt = 0;
			for (r = 0; r < mrows; r++)
			{
				// index l
				l = ja[r];
				// associated numerical value
				alpha = vall[r];
				// maintain large entries in absolute value and their row indices
				if (FABS(alpha) > mytol)
				{
					vall[cnt] = alpha;
					ja[cnt++] = l;
				} // end if
			}	  // end for r
			iL.nblockrow[j] = cnt;
		}
		else
		{ // block case
			// first pass: detect rows with small numerical values
			for (r = 0; r < mrows; r++)
			{
				// index l stored at position r
				l = ja[r];
				// maximum entry in row r of iL_new{j} physically starting at position r
#ifdef _BLAS_LAPACK_32_
				int myncols = ncols, mymrows = mrows;
				ii = idamax_(&myncols, vall + r, &mymrows) - 1;
#else
				ii = idamax_(&ncols, vall + r, &mrows) - 1;
#endif
				// flag index if the value is sufficiently small
				buff = FABS(vall[r + ii * mrows]);
#ifdef PRINT_INFO2
				printf("||L(%ld,:)||=|L(%ld,%ld)|=%12.4le\n", l + 1, l + 1, iL.colind[j][ii] + 1, buff);
				fflush(stdout);
#endif
				if (buff <= mytol)
				{
#ifdef PRINT_INFO2
					printf("small value\n");
					fflush(stdout);
#endif
					ja[r] = -1;
				} // end if
			}	  // end for r
#ifdef PRINT_INFO2
			printf("2. pass, thread=%ld, j=%ld\n", omp_get_thread_num(), j);
			fflush(stdout);
#endif
			// second pass: scratch out rows of small norm
			// pointer to the remaining numerical values
			pr = vall;
			// proceed column by column since the numerical values are stored
			// by columns
			for (k = 0; k < ncols; k++, vall += mrows)
			{
				// reduce column k
				for (r = 0; r < mrows; r++)
				{
					// index l
					l = ja[r];
					// this row should be kept
					if (l >= 0)
					{
#ifdef PRINT_INFO2
						printf("||L(%ld,:)||: large value\n", l + 1);
						fflush(stdout);
#endif
						*pr++ = vall[r];
					}
				} // end for r
			}	  // end for k
#ifdef PRINT_INFO2
			printf("3. pass, thread=%ld, j=%ld\n", omp_get_thread_num(), j);
			fflush(stdout);
#endif
			// third pass: scratch out indices of rows that were dropped
			cnt = 0;
			for (r = 0; r < mrows; r++)
			{
				// index l
				l = ja[r];
				// shift index
				if (l >= 0)
				{
#ifdef PRINT_INFO2
					printf("||L(%ld,:)||: index of large value\n", l + 1);
					fflush(stdout);
#endif
					ja[cnt++] = l;
				} // end if
			}	  // end for r
			iL.nblockrow[j] = cnt;

#ifdef PRINT_INFO2
			printf("large indices\n");
			fflush(stdout);
			for (r = 0; r < cnt; r++)
				printf("%12ld", ja[r] + 1);
			printf("\n");
			fflush(stdout);
#endif
#ifdef PRINT_INFO2
			printf("3. pass completed, thread=%ld, j=%ld, mrows=%ld\n", omp_get_thread_num(), j, cnt);
			fflush(stdout);
#endif
		} // end if-else ncols=1
#ifdef PRINT_INFO
		printf("new sizes %ld, %ld\n", cnt, cnt * ncols);
		fflush(stdout);
#endif
		iL.rowind[j] = (integer *)realloc(iL.rowind[j], (size_t)cnt * sizeof(integer));
		iL.valE[j] = (double *)realloc(iL.valE[j], (size_t)cnt * ncols * sizeof(double));
	} // end for j
	  // end OMP parallel for

#ifdef PRINT_INFOnnz
	jj = 0;
	for (j = 0; j < nblocks; j++)
		jj += iL.nblockcol[j] * (iL.nblockcol[j] + iL.nblockrow[j]);
	printf("BNINV: approximate inverse sparsified, nnz=%ld\n", jj);
	fflush(stdout);
	// PrintBlockSparse(&iL);fflush(stdout);
#endif

#ifdef _PROFILING_
	timeBegin = omp_get_wtime();
#endif
	k = omp_get_max_threads();
	idxb_src = (integer **)malloc((size_t)k * sizeof(integer *));
	idxbpos_src = (integer **)malloc((size_t)k * sizeof(integer *));
	iLTbegin_src = (integer **)malloc((size_t)k * sizeof(integer *));
	for (i = 0; i < k; i++)
	{
		idxb_src[i] = (integer *)malloc((size_t)nblocks * sizeof(integer));
		idxbpos_src[i] = (integer *)calloc((size_t)nblocks, sizeof(integer));
		iLTbegin_src[i] = (integer *)calloc((size_t)nblocks, sizeof(integer));
	} // end for i
	// compute W=S*P*iL^T*BiD*iL*P^T*S as follows
	// 1. compute block graph of iL^T and store the information of the starting
	//    physical position and size of the super-diagonal blocks t of iL^T
	//    inside iL
	// 2. compute iL_new <- iL^T*BiD*iL (block lower triangular matrix)
	// 3. rescale iL_new and sparsify it when transferring it to W
	// 4. complete W with its transpose for the MATLAB interface data
	SparseMatrix iLT;
	integer **iLTstart, **iLTsize, **Lstart;

	// step 1: compute block graph of iL^T and store the information of the
	//         starting physical position and size of the super-diagonal blocks
	//         t of iL^T inside iL
	// initially compute block pattern of iL^T and store it in iLT
	iLT.nr = iLT.nc = nblocks;
	iLT.ncol = (integer *)calloc((size_t)nblocks, sizeof(integer));
	iLT.rowind = (integer **)malloc((size_t)nblocks * sizeof(integer *));
	iLT.val = NULL;

	// also keep track of the start of sub-diagonal block s in a certain block
	// column iL{j}, i.e., if iLT.rowind[s][i]==j, then iLTstart[s][i] refers to
	// the physical beginning of block s inside iL{j}
	iLTstart = (integer **)malloc((size_t)nblocks * sizeof(integer *));
	// similarly, if iLT.rowind[s][i]==j, then iLTsize[s][i] refers to the number
	// of rows of block s inside iL{j}
	iLTsize = (integer **)malloc((size_t)nblocks * sizeof(integer *));

	// first pass: count number of blocks in iL^T for each block column s of iLT
	// short cut ia, initially iLT.ncol is used to count the number of blocks
	ia = iLT.ncol;
	for (j = 0; j < nblocks; j++)
	{
		// number of sub-diagonal rows in iL{j}
		mrows = iL.nblockrow[j];
		// short cut for associated row indices
		ja = iL.rowind[j];
		// scan sub-diagonal row indices to detect the associated block structure
		r = 0;
		while (r < mrows)
		{
			// grab nonzero row indices of iL{j}
			l = ja[r];
			// associated block column s of iL^T, iL{s,j} ~ iLT{j,s}
			s = block[l];
#ifdef PRINT_CHECK
			if (l < 0 || l >= n)
				printf(" 6. scalar index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
			if (s < 0 || s >= nblocks)
				printf(" 6. block index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
#endif
			// increment number of blocks in column s of iL^T
			ia[s]++;
			// advance to the subsequent row in iL{j} that does not belong to
			// block s anymore
			r2 = r + 1;
			while (r2 < mrows)
			{
				l = ja[r2];
#ifdef PRINT_CHECK
				if (l < 0 || l >= n)
					printf(" 7. scalar index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
				if (s < 0 || s >= nblocks)
					printf(" 7. block index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
#endif
				if (block[l] != s)
					break;
				r2++;
			} // end while r2
			// advance to the next block row inside iL{j}
			r = r2;
		} // end while r<mrows
	}	  // end for j
	// intermediate step: now we now how many blocks are contained in each block
	// row and can allocate the required memory
	for (j = 0; j < nblocks; j++)
	{
		iLT.rowind[j] = (integer *)malloc((size_t)ia[j] * sizeof(integer));
		iLTstart[j] = (integer *)malloc((size_t)ia[j] * sizeof(integer));
		iLTsize[j] = (integer *)malloc((size_t)ia[j] * sizeof(integer));
		// reset ia
		ia[j] = 0;
	} // end for j
	// second pass: set up block structure for iL^T
	for (j = 0; j < nblocks; j++)
	{
		// number of sub-diagonal rows in iL{j}
		mrows = iL.nblockrow[j];
		// short cut for associated row indices
		ja = iL.rowind[j];
		// scan sub-diagonal row indices to detect the associated block structure
		r = 0;
		while (r < mrows)
		{
			// grab nonzero row indices of iL{j}
			l = ja[r];
			// associated block column of iL^T
			s = block[l];
#ifdef PRINT_CHECK
			if (l < 0 || l >= n)
				printf(" 8. scalar index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
			if (s < 0 || s >= nblocks)
				printf(" 8. block index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
#endif
			// get current position in block row s
			i = ia[s];
			// insert reference to super-diagonal block j
			iLT.rowind[s][i] = j;
			// insert start position of block s inside iL{j}
			iLTstart[s][i] = r;
			// make sure that this block column has not been considered yet
			r2 = r + 1;
			while (r2 < mrows)
			{
				l = ja[r2];
#ifdef PRINT_CHECK
				if (l < 0 || l >= n)
					printf(" 9. scalar index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
				if (s < 0 || s >= nblocks)
					printf(" 9. block index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
#endif
				if (block[l] != s)
					break;
				r2++;
			} // end while r2
			// insert size of block s inside iL{j}
			iLTsize[s][i] = r2 - r;
			// increment number of block in colmn s
			ia[s] = i + 1;
			// advance to the next block
			r = r2;
		} // end while r
	}	  // end for j
#ifdef PRINT_INFO2
	printf("companion block matrix iL^T set up\n");
	fflush(stdout);
	PrintSparse(&iLT);
	printf("iL^T block size\n");
	fflush(stdout);
	for (j = 0; j < nblocks; j++)
	{
		mrows = iLT.ncol[j];
		printf("block column % ld: ", j + 1);
		fflush(stdout);
		for (k = 0; k < mrows; k++)
		{
			printf("%8ld", iLTsize[j][k]);
		} // end for k
		printf("\n");
		fflush(stdout);
	} // end for j
	printf("\n");
	fflush(stdout);
	printf("iL^T block start\n");
	fflush(stdout);
	for (j = 0; j < nblocks; j++)
	{
		mrows = iLT.ncol[j];
		printf("block column % ld: ", j + 1);
		fflush(stdout);
		for (k = 0; k < mrows; k++)
		{
			printf("%8ld", iLTstart[j][k]);
		} // end for k
		printf("\n");
		fflush(stdout);
	} // end for j
	printf("\n");
	fflush(stdout);
#endif
#ifdef _PROFILING_
	time_iLT = omp_get_wtime() - timeBegin;
#endif

	// now we know the block pattern of iL^T including the start and the size of
	// each block s inside iL{j}

	// step 2: compute iL_new <- iL^T*BiD*iL (block lower triangular matrix)
#pragma omp parallel for shared(nblocks, iLT, idxl_src, idxposl_src, idxl2_src, iL, block, iL_new, BiD, vall_src, size_vall_src, vall2_src, size_vall2_src, iLTsize, iLTstart, idxb_src, idxbpos_src, iLTbegin_src) private(mythreadnum, ia, idxl, idxposl, idxl2, ncols, mrows, ja, i, cnt, r, l, s, r2, ja2, cnt2, jj, ii, i2, tt, ttt, ss, i3, t, k, a, pr, vall, vall2, transa, transb, alpha, beta, idxb, idxbpos, iLTbegin, cntb, BiD_inplace)
	for (j = 0; j < nblocks; j++)
	{
		// who am I
		mythreadnum = omp_get_thread_num();
#ifdef PRINT_INFO2
		printf("block column %ld\n", j);
		fflush(stdout);
#endif
		// local copy
		ia = iLT.ncol;
		// idxl is used to store the row indices of block column j of iL^T*BiD*iL
		idxl = idxl_src[mythreadnum];
		// idxposl is used to refer to the associated position of the row
		// indices, shifted by 1
		idxposl = idxposl_src[mythreadnum];
		// idxl2[s] stores the physical beginning of block s inside block column
		// j of iL^T*BiD*iL
		idxl2 = idxl2_src[mythreadnum];
		// idxb is used to store the block row indices of block column j of iL^T*BiD*iL
		idxb = idxb_src[mythreadnum];
		// idxbpos is used to refer to the associated position of the block row
		// indices, shifted by 1
		idxbpos = idxbpos_src[mythreadnum];
		// beginning of super-diagonal block entries t>=j
		iLTbegin = iLTbegin_src[mythreadnum];
		// number of columns and sub-diagonal rows inside iL{j}
		ncols = iL.nblockcol[j];
		mrows = iL.nblockrow[j];

		// compute j-th block column of iL^T*BiD*iL
#ifdef PRINT_INFO2
		printf("block column %ld, init with indices of the diagonal block (%ld,%ld)\n", j, j, j);
		fflush(stdout);
#endif
#ifdef PRINT_CHECK
		for (i = 0; i < nblocks; i++)
			idxl2[i] = -1;
#endif

#ifdef _PROFILING_
		timeBeginLocal = omp_get_wtime();
#endif
		ja = iL.rowind[j];
		// first pass, determine nonzero block-diagonal row pattern of iL^T*BiD*iL{j}
		cntb = 0;
#ifdef PRINT_CHECK
		if (cntb >= nblocks || cntb < 0)
		{
			printf("illegal counter value\n");
			fflush(stdout);
		}
		if (j >= nblocks || j < 0)
		{
			printf("illegal block number\n");
			fflush(stdout);
		}
#endif
		idxb[cntb] = j;
		idxbpos[j] = ++cntb;
		// counter for the sub-diagonal rows of iL{j}
		r = 0;
		while (r < mrows)
		{
			// grab nonzero row indices of iL{j}
			l = ja[r];
			// associated block column s of iL^T
			s = block[l];
			// advance to the subsequent row r2 that follows block s inside iL{j}
			r2 = r + 1;
			while (r2 < mrows)
			{
				l = ja[r2];
				if (block[l] != s)
					break;
				r2++;
			} // end while r2<mrows
			if (!idxbpos[s])
			{
#ifdef PRINT_CHECK
				if (cntb >= nblocks || cntb < 0)
				{
					printf("illegal counter value\n");
					fflush(stdout);
				}
				if (s >= nblocks || s < 0)
				{
					printf("illegal block number\n");
					fflush(stdout);
				}
#endif
				idxb[cntb] = s;
				idxbpos[s] = ++cntb;
			}
			// scan super-diagonal blocks t of iLT{s}
			for (i3 = iLTbegin[s]; i3 < ia[s]; i3++)
			{
				// super-diagonal block t of iLT{s}, it suffices to consider
				// t>=j since the matrix-product is symmetric
				t = iLT.rowind[s][i3];

				// store the beginning of super-diagonal blocks t>= j
				// since the entries of iLT.rowind[s] are sorted in increasing
				// order, iLTbegin[s] is overwritten unless t>j, in this case
				// the last time it was overwritten refers to the largest t
				// such that t<=j
				if (t <= j)
					iLTbegin[s] = i3;
				// super-diagonal block t does not exist yet
				if (!idxbpos[t] && t >= j)
				{
#ifdef PRINT_CHECK
					if (cntb >= nblocks || cntb < 0)
					{
						printf("illegal counter value\n");
						fflush(stdout);
					}
					if (t >= nblocks || t < 0)
					{
						printf("illegal block number\n");
						fflush(stdout);
					}
#endif
					idxb[cntb] = t;
					idxbpos[t] = ++cntb;
				} // end if
			}	  // end for i3
			// advance to the next sub-diagonal block inside iL{s}
			r = r2;
		} // end while r<mrows
		// sort block index list in increasing order
		qqsorti_(idxb, idxl, &cntb);
		// re-init check mark array
		l = 0;
		for (i = 0; i < cntb; i++)
		{
			k = idxb[i];
			idxbpos[k] = i + 1;
			idxl2[k] = l;
			l = l + iL.nblockcol[k];
		} // end for i
		cnt = l;
#ifdef PRINT_INFO1
		printf("cnt=%ld\n", cnt);
		printf("idxb\n");
		for (i = 0; i < cntb; i++)
			printf("%4ld", idxb[i] + 1);
		printf("\n");
		fflush(stdout);
		printf("idxbpos\n");
		for (i = 0; i < cntb; i++)
			printf("%4ld", idxbpos[idxb[i]] - 1);
		printf("\n");
		fflush(stdout);
		printf("idxl2\n");
		for (i = 0; i < cntb; i++)
			printf("%4ld", idxl2[idxb[i]]);
		printf("\n");
		fflush(stdout);
		printf("iLTbegin\n");
		for (i = 0; i < nblocks; i++)
			printf("%4ld", iLTbegin[i]);
		printf("\n");
		fflush(stdout);
#endif
#ifdef PRINT_INFO2
		for (i = 0; i < cntb; i++)
			printf("%12ld", idxb[i] + 1);
		printf("\n");
		fflush(stdout);
#endif
		// end first pass
#ifdef _PROFILING_
		time_matrix_matrix1 += omp_get_wtime() - timeBeginLocal;
#endif

#ifdef PRINT_INFO
		printf("new sizes %ld, %ld!\n", cnt, cnt * ncols);
		fflush(stdout);
#endif
		// intermediate step, adjust memory for up to cnt new rows
		iL_new.nblockrow[j] = cnt;
		if (iL_new.rowind[j] == NULL)
		{
			iL_new.rowind[j] = (integer *)malloc((size_t)cnt * sizeof(integer));
		}
		else
		{
			iL_new.rowind[j] = (integer *)realloc(iL_new.rowind[j], (size_t)cnt * sizeof(integer));
		}
		if (iL_new.valE[j] == NULL)
		{
			iL_new.valE[j] = (double *)malloc((size_t)cnt * ncols * sizeof(double));
		}
		else
		{
			iL_new.valE[j] = (double *)realloc(iL_new.valE[j], (size_t)cnt * ncols * sizeof(double));
		}
		// copy index lists from the single blocks
		ja = iL_new.rowind[j];
		for (i = 0; i < cntb; i++)
		{
			k = idxb[i];
			l = iL.nblockcol[k];
			memcpy(ja, iL.colind[k], l * sizeof(integer));
			ja += l;
		} // end for i
#ifdef PRINT_INFO
		printf("row index list\n");
		for (i = 0; i < cnt; i++)
			printf("%4ld", iL_new.rowind[j][i] + 1);
		printf("\n");
		fflush(stdout);
#endif
		// clear numerical values
		k = cnt * ncols;
		a = iL_new.valE[j];
		for (i = 0; i < k; i++)
			a[i] = 0.0;
			// end intermediate step

#ifdef _PROFILING_
		timeBeginLocal = omp_get_wtime();
#endif
		// second pass, determine matrix-matrix product using level-3 BLAS
		// iL^T*BiD*iL{j} is decomposed as BiD*iL{j} + (iL-I)^T * BiD*iL{j}
		// according to the block (lower triangular) structure of iL_new that
		// covers the required blocks of this product

		// start with the diagonal block j of iL{j}
#ifdef PRINT_INFO2
		printf("block column %ld, copy diagonal block (%ld,%ld)\n", j, j, j);
		fflush(stdout);
#endif
		// copy diagonal block BiD{j}
		pr = BiD.valD[j];
		a = iL_new.valE[j];
		for (r = 0; r < ncols; r++, a += cnt, pr += ncols)
			memcpy(a, pr, ncols * sizeof(double));

		// next add the diagonal block associated with sub-diagonal block s of
		// iL{j} if it is not yet present, followed by the super-diagonal blocks t
		// of iLT{s}
		// short cut
		ja = iL.rowind[j];
		// counter for the sub-diagonal rows of iL{j}
		r = 0;
		while (r < mrows)
		{
			// grab nonzero row indices of iL{j}
			l = ja[r];
			// associated block column s of iL^T
			s = block[l];
#ifdef PRINT_CHECK
			if (l < 0 || l >= n)
				printf("16. scalar index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
			if (s < 0 || s >= nblocks)
				printf("16. block index out of range when referring to %ld=block[%ld]\n", s, l);
			fflush(stdout);
#endif

#ifdef PRINT_INFO2
			printf("block column %ld, use block column %ld of iL^T\n", j, s);
			fflush(stdout);
#endif

			// advance to the subsequent row r2 that follows block s inside iL{j}
			r2 = r + 1;
			while (r2 < mrows)
			{
				l = ja[r2];
#ifdef PRINT_CHECK
				if (l < 0 || l >= n)
					printf("17. scalar index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
				if (s < 0 || s >= nblocks)
					printf("17. block index out of range when referring to %ld!=block[%ld]\n", s, l);
				fflush(stdout);
#endif
				if (block[l] != s)
					break;
				r2++;
			} // end while r2<mrows

			// call level-3 BLAS to compute vall= 1*BiD{s}*iL{j} + 0*vall
			//                                  = 1*vall2 *iL{j} + 0*vall
#ifdef PRINT_INFO2
			printf("block column %ld, use block column %ld of iL^T, build block (%ld,%ld)\n", j, s, s, j);
			fflush(stdout);
#endif
			jj = iL.nblockcol[s] * ncols;
			size_vall_src[mythreadnum] = MAX(size_vall_src[mythreadnum], jj);
			vall = vall_src[mythreadnum];
			vall_src[mythreadnum] = (double *)realloc(vall, size_vall_src[mythreadnum] * sizeof(double));
			// for technical reasons, init vall with 0
			vall = vall_src[mythreadnum];
			for (i2 = 0; i2 < jj; i2++)
				vall[i2] = 0.0;

			l = ja[r2 - 1] - ja[r] + 1;
			if (l == r2 - r)
			{
				BiD_inplace = -1;
#ifdef PRINT_INFO1
				printf("BiD{%ld} need not be copied, since %ld...%ld are %ld contiguous indices\n",
					   s + 1, ja[r] + 1, ja[r2 - 1] + 1, r2 - r);
				fflush(stdout);
#endif
			}
			else
				BiD_inplace = 0;

			// scalar case vall   = 1*BiD{s}*iL{j}  + 0*vall
			//             vall   = iL{j} *vall2    + 0*vall
			//             vector = scalar*vector
			if (ncols == 1 && r2 - r == 1)
			{
				// compute vall = iL{j}*BiD{s}
				// associated columns and rows of BiD{s}
				pr = BiD.valD[s];
				// number of rows/columns in block s of BiD
				jj = iL.nblockcol[s];
				// first index of diagonal block s
				ii = iL.colind[s][0];
				vall2 = pr + jj * (ja[r] - ii);
				alpha = iL.valE[j][r];
				for (l = 0; l < jj; l++)
					*vall++ = alpha * *vall2++;
				vall -= jj;
			}
			// block case use level-3 BLAS
			else
			{
				if (BiD_inplace)
				{
					// associated columns and rows of BiD{s}
					pr = BiD.valD[s];
					// number of rows/columns in block s of BiD
					jj = iL.nblockcol[s];
					// first index of diagonal block s
					ii = iL.colind[s][0];
					vall2 = pr + jj * (ja[r] - ii);
				}
				else
				{
					jj = iL.nblockcol[s] * (r2 - r);
					size_vall2_src[mythreadnum] = MAX(size_vall2_src[mythreadnum], jj);
					vall2 = vall2_src[mythreadnum];
					vall2_src[mythreadnum] = (double *)realloc(vall2, size_vall2_src[mythreadnum] * sizeof(double));
					vall2 = vall2_src[mythreadnum];

					// extract associated columns and rows of BiD{s}
					pr = BiD.valD[s];
					// number of rows in block s
					jj = iL.nblockcol[s];
					// first index of diagonal block s
					ii = iL.colind[s][0];
					for (l = r; l < r2; l++, vall2 += jj)
					{
						// copy entries of column r of BiD{s}
						memcpy(vall2, pr + jj * (ja[l] - ii), jj * sizeof(double));
					} // end for l
					vall2 = vall2_src[mythreadnum];
				} // end if-else BiD_inplace

#ifdef PRINT_INFO2
				printf("            ");
				for (l = r; l < r2; l++)
				{
					printf("%12ld", ja[l] + 1);
				} // end for l
				printf("\n");
				fflush(stdout);
				for (i3 = 0; i3 < jj; i3++)
				{
					printf("%12ld", iL.colind[s][i3] + 1);
					for (l = r; l < r2; l++)
					{
						printf("%12.4le", vall2[jj * (l - r) + i3]);
					} // end for l
					printf("\n");
					fflush(stdout);
				} // end for i3
#endif

				// vall = 1*vall2*iL{j} + 0*vall
				transa = "n";
				transb = "n";
				alpha = 1.0;
				beta = 0.0;
				l = r2 - r;
#ifdef _BLAS_LAPACK_32_
				int myjj = jj, myncols = ncols, myl = l, mymrows = mrows;
				dgemm_(transa, transb, &myjj, &myncols, &myl, &alpha,
					   vall2, &myjj, iL.valE[j] + r, &mymrows, &beta,
					   vall, &myjj, 1, 1);
#else
				dgemm_(transa, transb, &jj, &ncols, &l, &alpha,
					   vall2, &jj, iL.valE[j] + r, &mrows, &beta,
					   vall, &jj, 1, 1);
#endif
			} // end if-else ncols=1 and r2-r=1

			// init diagonal block contribution of block row s
			// starting address of block s inside iL^T*BiD*iL{j}
			a = iL_new.valE[j] + idxl2[s];
			for (l = 0; l < ncols; l++, a += cnt, vall += jj)
			{
				// copy entries of column l
				memcpy(a, vall, jj * sizeof(double));
			} // end for l
			vall = vall_src[mythreadnum];

#ifdef PRINT_INFO2
			printf("            ");
			for (l = 0; l < ncols; l++)
			{
				printf("%12ld", iL.colind[j][l] + 1);
			} // end for l
			printf("\n");
			fflush(stdout);
			a = iL_new.valE[j] + idxl2[s];
			for (i3 = 0; i3 < jj; i3++)
			{
				printf("%12ld", iL.colind[s][i3] + 1);
				for (l = 0; l < ncols; l++)
				{
					printf("%12.4le", a[cnt * l + i3]);
				} // end for l
				printf("\n");
				fflush(stdout);
			} // end for i3
#endif

			// update iL_new{j} = 1*iLT{s}*BiD{s}*iL{j} + 1*iL_new{j}, this
			// is done by updating with all super-diagonal blocks t of iLT{s}
			for (i3 = iLTbegin[s]; i3 < ia[s]; i3++)
			{
				// super-diagonal block t of iLT{s}, s>t>=j
				t = iLT.rowind[s][i3];
#ifdef PRINT_INFO2
				printf("block column %ld, use block column %ld of iL^T, block %ld\n", j, s, t);
				fflush(stdout);
#endif

				if (t >= j)
				{
					// reduce BiD{s}*iL{j} (cached in vall) to those rows required by
					// super-diagonal block t of iLT{s}, this block refers to a sub-
					// diagonal block s of iL{t}, therefore we can access its indices
					// via iL.rowind[t] and furthermore, since iLT.rowind[s][i3]=t, we
					// have that iLTstart[s][i3] refers to the physical begining of
					// block s inside iL{t} and iLTsize[s][i3] refers to its size
					jj = iLTsize[s][i3];
					tt = iLTstart[s][i3];
					// initial index of diagonal block s
					ii = iL.colind[s][0];
					// shift vall by its initial index of diagonal block s
					vall -= ii;
					// sub-diagonal row indices associated with iL{t}, the sub-
					// diagonal block s starts at positition tt
					ja2 = iL.rowind[t] + tt;
					if (ja2[jj - 1] - ja2[0] + 1 == jj)
					{
						BiD_inplace = -1;
#ifdef PRINT_INFO1
						printf("block row %ld of BiD{%ld}*iL{%ld} need not be copied, since %ld...%ld are %ld contiguous row indices\n",
							   t + 1, s + 1, j + 1, ja2[0] + 1, ja2[jj - 1] + 1, jj);
						fflush(stdout);
#endif
					}
					else
					{
						BiD_inplace = 0;
					}

					// scalar case iL_new{j} = 1*iLT{s}*BiD{s}*iL{j} + 1*iL_new{j}
					//             iL_new{j} = 1*iLT{s}*vall         + 1*iL_new{j}
					//             iL_new{j} = vall[ja2[0]]*iLT{s}   + iL_new{j}
					//              vector   =    scalar   *vector   + vector
					if (ncols == 1 && jj == 1)
					{
						vall2 = vall + ja2[0];
						a = iL.valE[t] + tt;
						ttt = 1;
						vall = vall_src[mythreadnum];
#ifdef _BLAS_LAPACK_32_
						int mynblockcol = iL.nblockcol[t], mynblockrow = iL.nblockrow[t], myttt = ttt;
						daxpy_(&mynblockcol, vall2,
							   a, &mynblockrow, iL_new.valE[j] + idxl2[t], &myttt);
#else
						daxpy_(&(iL.nblockcol[t]), vall2,
							   a, &(iL.nblockrow[t]), iL_new.valE[j] + idxl2[t], &ttt);
#endif
					}
					// block case use level-3 BLAS
					else
					{
						if (BiD_inplace)
						{
							vall2 = vall + ja2[0];
						}
						else
						{
							size_vall2_src[mythreadnum] = MAX(size_vall2_src[mythreadnum], jj * ncols);
							vall2 = vall2_src[mythreadnum];
							vall2_src[mythreadnum] = (double *)realloc(vall2, size_vall2_src[mythreadnum] * sizeof(double));
							// extraction
							vall2 = vall2_src[mythreadnum];
							for (ttt = 0; ttt < ncols; ttt++, vall += iL.nblockcol[s])
							{
#ifdef _USE_MKL_
								// gather column ttt from vall into vall2
#ifdef _BLAS_LAPACK_32_
								int myjj = jj, mykk,
									*myja2 = (int *)malloc((size_t)jj * sizeof(int));
								for (mykk = 0; mykk < myjj; mykk++)
									myja2[mykk] = ja2[mykk];
								cblas_dgthr(myjj, vall, vall2, myja2);
								free(myja2);
#else
								cblas_dgthr(jj, vall, vall2, ja2);
#endif
								vall2 += jj;
#else
								// copy jj entries of column ttt
								for (i2 = 0; i2 < jj; i2++)
								{
									*vall2++ = vall[ja2[i2]];
								} // end for i3
#endif
							} // end for i2
							vall2 = vall2_src[mythreadnum];
						} // end if-else BiD_inplace
						vall = vall_src[mythreadnum];

						// update using level-3 BLAS
						// update super-diagonal block row t of
						// iL_new{j} = 1*iLT{s}*BiD{s}*iL{j} + 1*iL_new{j}
						//           = 1*iLT{s}*  "vall"     + 1*iL_new{j}
						//           = 1*iLT{s}*vall2        + 1*iL_new{j}
#ifdef PRINT_INFO2
						printf("block column %ld, use block column %ld of iL^T, block %ld, update block (%ld,%ld)\n", j, s, t, t, j);
						fflush(stdout);
#endif
						// super-diagonal block t of iLT{s} refers to sub-diagonal
						// block s of iL{t}, therefore we can access it via iL.valE[t]
						// and furthermore, since iLT.rowind[s][i3]=t, we have that
						// iLTstart[s][i3] refers to the physical begining of block s
						// inside iL{t}
						a = iL.valE[t] + tt;
						transa = "t";
						transb = "n";
						alpha = 1.0;
						beta = 1.0;
						if (BiD_inplace)
						{
#ifdef _BLAS_LAPACK_32_
							int mynblockcol = iL.nblockcol[t], myncols = ncols, myjj = jj, mynblockrow = iL.nblockrow[t], mynblockcols = iL.nblockcol[s], mycnt = cnt;
							dgemm_(transa, transb, &mynblockcol, &myncols, &myjj, &alpha,
								   a, &mynblockrow, vall2, &mynblockcols, &beta,
								   iL_new.valE[j] + idxl2[t], &mycnt, 1, 1);
#else
							dgemm_(transa, transb, &(iL.nblockcol[t]), &ncols, &jj, &alpha,
								   a, &(iL.nblockrow[t]), vall2, &(iL.nblockcol[s]), &beta,
								   iL_new.valE[j] + idxl2[t], &cnt, 1, 1);
#endif
						}
						else
						{
#ifdef _BLAS_LAPACK_32_
							int mynblockcol = iL.nblockcol[t], myncols = ncols, myjj = jj, mynblockrow = iL.nblockrow[t], mycnt = cnt;
							dgemm_(transa, transb, &mynblockcol, &myncols, &myjj, &alpha,
								   a, &mynblockrow, vall2, &myjj, &beta,
								   iL_new.valE[j] + idxl2[t], &mycnt, 1, 1);
#else
							dgemm_(transa, transb, &(iL.nblockcol[t]), &ncols, &jj, &alpha,
								   a, &(iL.nblockrow[t]), vall2, &jj, &beta,
								   iL_new.valE[j] + idxl2[t], &cnt, 1, 1);
#endif
						}
#ifdef PRINT_INFO2
						printf("            ");
						for (l = 0; l < ncols; l++)
						{
							printf("%12ld", iL.colind[j][l] + 1);
						} // end for l
						printf("\n");
						fflush(stdout);
						a = iL_new.valE[j] + idxl2[t];
						for (ii = 0; ii < iL.nblockcol[t]; ii++)
						{
							printf("%12ld", iL.colind[t][ii] + 1);
							for (l = 0; l < ncols; l++)
							{
								printf("%12.4le", a[cnt * l + ii]);
							} // end for l
							printf("\n");
							fflush(stdout);
						} // end for ii
#endif
					} // end if-else ncols=1 and jj=1
				}	  // end if t>=j
			}		  // end for i3

			// advance to the next sub-diagonal block inside iL{s}
			r = r2;
		} // end while r<mrows
#ifdef _PROFILING_
		time_matrix_matrix2 += omp_get_wtime() - timeBeginLocal;
#endif
		// end second pass

		// re-init check mark array
		for (i = 0; i < cntb; i++)
		{
			k = idxb[i];
			idxbpos[k] = 0;
		} // end for i

	} // end for j
	  // end omp parallel for
#ifdef PRINT_INFO
	printf("companion block matrix iL^T*BiD*iL computed\n");
	fflush(stdout);
	// PrintBlockSparse(&iL_new);fflush(stdout);
#endif

	// release block mapping and iLT
	free(block);
	for (j = 0; j < nblocks; j++)
	{
		free(iLT.rowind[j]);
		free(iLTstart[j]);
		free(iLTsize[j]);
	}
	free(iLTstart);
	free(iLTsize);
	free(iLT.rowind);
	free(iLT.ncol);
#ifdef _PROFILING_
	time_matrix_matrix = omp_get_wtime() - timeBegin;
#endif

	// step 3: rescale iL_new and sparsify it when transferring it to W

	// next compute (P^TSP) (iL^T BiD iL) (P^TSP) = (P^TSP) iL_new (P^TSP)
	// iL_new is block lower triangular

	if (SL != NULL)
	{
#pragma omp parallel for shared(nblocks, iL, iL_new, p, SL) private(ncols, ia, mrows, ja, i, a, k, jj, r, buff, alpha)
		for (j = 0; j < nblocks; j++)
		{
			ncols = iL.nblockcol[j];
			// short cut
			ia = iL.colind[j];
			mrows = iL_new.nblockrow[j];
			// short cut
			ja = iL_new.rowind[j];
			// rescale block column j appropriately
			// at first, column-scaling from the right
			i = 1;
			// short cut
			a = iL_new.valE[j];
			for (k = 0; k < ncols; k++, a += mrows)
			{
				// scalar column index jj
				jj = ia[k];
				// row index r of P(r,jj)
				// r=perm[jj];
				r = P.rowind[jj][0];
				// diagonal entry S(r,r)
#ifdef _BLAS_LAPACK_32_
				int mymrows = mrows, myi = i;
				dscal_(&mymrows, SL + r, a, &myi);
#else
				dscal_(&mrows, SL + r, a, &i);
#endif
			} // end for k
			// second, row-scaling from the left
			// short cut
			a = iL_new.valE[j];
			for (k = 0; k < ncols; k++, a += mrows)
			{
				for (i = 0; i < mrows; i++)
				{
					// scalar row index jj
					jj = ja[i];
					// row index r of P(r,jj)
					// r=perm[jj];
					r = P.rowind[jj][0];
					// diagonal entry S(r,r)
					a[i] *= SL[r];
				} // end for i
			}	  // end for k
		}		  // end for j
				  // end omp parallel for
	}			  // end if SL!=0

#ifdef PRINT_INFO
	printf("iL^T*BiD*iL sparsified\n");
	fflush(stdout);
	// PrintBlockSparse(&iL_new);fflush(stdout);
#endif

	// extract diagonal part
	valr = (double *)vall_src[0];
#pragma omp parallel for shared(nblocks, iL, iL_new, valr) private(ncols, ia, mrows, ja, a, k, ii, i)
	for (j = 0; j < nblocks; j++)
	{
		ncols = iL.nblockcol[j];
		// short cut
		ia = iL.colind[j];
		mrows = iL_new.nblockrow[j];
		// short cut
		ja = iL_new.rowind[j];
		// short cut
		a = iL_new.valE[j];
		for (k = 0; k < ncols; k++, a += mrows)
		{
			// column index ii
			ii = ia[k];
			valr[ii] = 0.0;
			for (i = 0; i < mrows; i++)
			{
				// diagonal entry found
				if (ja[i] == ii)
				{
					valr[ii] = sqrt(FABS(a[i]));
					break;
				}
			} // end for i
		}	  // end for k
	}		  // end for j
			  // end omp parallel for
#ifdef PRINT_INFO
	printf("diagonal part extracted\n");
	fflush(stdout);
#endif

	// copy   P (P^TSP) (iL^T BiD iL) (P^TSP) P^T = P iL_new P^T   to W and
	// sparsify during copying, we sparsify each off-diagonal entry using
	// |W(i,j)| > epsilon * sqrt(|W(i,i)|*|W(j,j)|)
	FreeSparse(&W);
	W.nr = W.nc = n;
	// ncol is set to 0 in order to count the nz
	W.ncol = (integer *)calloc((size_t)n, sizeof(integer));
	W.rowind = (integer **)malloc((size_t)n * sizeof(integer *));
	W.val = (double **)malloc((size_t)n * sizeof(double *));

	// sparsify and copy to W
#pragma omp parallel for shared(nblocks, iL, iL_new, mytol, valr, p, n, W) private(ncols, ia, mrows, ja, a, k, ii, cnt, i, jj, r, s)
	for (j = 0; j < nblocks; j++)
	{
		ncols = iL.nblockcol[j];
		// short cut
		ia = iL.colind[j];
		mrows = iL_new.nblockrow[j];
		// short cut
		ja = iL_new.rowind[j];
		// short cut
		a = iL_new.valE[j];
		for (k = 0; k < ncols; k++, a += mrows)
		{
			// current column index ii
			ii = ia[k];
			// first pass, count number of remaining entries
			cnt = 0;
			for (i = 0; i < mrows; i++)
			{
				// (sub-)diagonal row entry jj found
				jj = ja[i];
				if (jj >= ii)
				{
					if (FABS(a[i]) > mytol * valr[ii] * valr[jj] || ii == jj)
						cnt++;
				} // end if jj>=ii
			}	  // end for i

			// intermediate step: allocate memory
			// row index r of P(r,ii)
			// r=perm[ii];
			r = P.rowind[ii][0];
#ifdef PRINT_CHECK
			if (r < 0 || r >= n)
			{
				printf("permutation index out of range\n");
				fflush(stdout);
			}
#endif
			W.ncol[r] = cnt;
			W.rowind[r] = (integer *)malloc((size_t)cnt * sizeof(integer));
			W.val[r] = (double *)malloc((size_t)cnt * sizeof(double));

			// second pass, copy remaining entries
			cnt = 0;
			for (i = 0; i < mrows; i++)
			{
				// (sub-)diagonal row entry jj found
				jj = ja[i];
				if (jj >= ii)
				{
					if (FABS(a[i]) > mytol * valr[ii] * valr[jj] || ii == jj)
					{
						// transfer permutated index to W
						// start column jj of P, a permutation matrix
						// row index s of P(s,jj)
						// s=perm[jj];
						s = P.rowind[jj][0];
#ifdef PRINT_CHECK
						if (s < 0 || s >= n)
						{
							printf("permutation index out of range\n");
							fflush(stdout);
						}
#endif
						W.rowind[r][cnt] = s;
						W.val[r][cnt] = a[i];
						cnt++;
					} // end if |iL_new(jj,ii)|>mytol*sqrt(|iL_new(ii,ii)|*|iL_new(jj,jj)|) or ii=jj
				}	  // end if jj>=ii
			}		  // end for i
		}			  // end for k
	}				  // end for j
					  // end omp parallel for
#ifdef PRINT_INFO
	printf("P*S*(iL^T*BiD*iL)*S*P^T copied and sparsified\n");
	fflush(stdout);
	// PrintSparse(&W);fflush(stdout);
#endif

	// release iL
	for (j = 0; j < nblocks; j++)
	{
		if (iL.rowind[j] != NULL)
			free(iL.rowind[j]);
		if (iL.valE[j] != NULL)
			free(iL.valE[j]);
	} // end for j
	free(iL.nblockrow);
	free(iL.rowind);
	free(iL.valE);

	// release iL_new
	for (j = 0; j < nblocks; j++)
	{
		if (iL_new.rowind[j] != NULL)
			free(iL_new.rowind[j]);
		if (iL_new.valE[j] != NULL)
			free(iL_new.valE[j]);
	} // end for j
	free(iL_new.nblockrow);
	free(iL_new.rowind);
	free(iL_new.valE);
#ifdef PRINT_INFO
	printf("iL and iL_new discarded\n");
	fflush(stdout);
#endif

	// 4. complete W with its transpose

	// now supplement W with W^T and store it in W
	// counter for nnz of W^T excluding diag(W)
	idxl = idxl_src[0];
	for (j = 0; j < n; j++)
		idxl[j] = 0;
	// short cut
	ia = W.ncol;
	// first pass: compute additional amount of memory
	for (j = 0; j < n; j++)
	{
		mrows = ia[j];
		// short cut
		ja = W.rowind[j];
		for (k = 0; k < mrows; k++)
		{
			i = ja[k];
			// off-diagonal entry
			if (i != j)
				idxl[i]++;
		} // end for k
	}	  // end for j
#ifdef PRINT_INFO2
	printf("1. pass completed, W supplemented with its transpose\n");
	fflush(stdout);
#endif

	// intermediate step, re-allocate memory
	for (j = 0; j < n; j++)
	{
		k = ia[j] + idxl[j];
		W.rowind[j] = (integer *)realloc(W.rowind[j], (size_t)k * sizeof(integer));
		W.val[j] = (double *)realloc(W.val[j], (size_t)k * sizeof(double));
		// re-set counter
		idxl[j] = 0;
	} // end for j
#ifdef PRINT_INFO2
	printf("memory allocation completed, W supplemented with its transpose\n");
	fflush(stdout);
#endif

	// second pass: fill additional entries
	for (j = 0; j < n; j++)
	{
		// associated number of rows in column j
		mrows = ia[j];
		// short cuts
		ja = W.rowind[j];
		a = W.val[j];
		for (k = 0; k < mrows; k++)
		{
			i = ja[k];
			// fill off-diagonal entry
			if (i != j)
			{
				// associated number of rows in column i
				ii = ia[i];
				// current additional number of entry for W^T excluding diag(W)
				jj = idxl[i];
				// insert entry at the end of column i
				W.rowind[i][ii + jj] = j;
				W.val[i][ii + jj] = a[k];
				// increase counter for the number of additional entries
				idxl[i] = jj + 1;
			}
		} // end for k
	}	  // end for j
#ifdef PRINT_INFO
	printf("W supplemented with its transpose\n");
	fflush(stdout);
#endif

	// short cut
	ia = W.ncol;
	// update number of nonzeros
	for (j = 0; j < n; j++)
		ia[j] += idxl[j];
	nnz = 0;
	for (j = 0; j < n; j++)
		nnz += ia[j];
	W.nnz = nnz;
#ifdef PRINT_INFOnnz
	printf("BNINV: final (nonsymmetric) approximate inverse computed, nnz=%ld\n", nnz);
	fflush(stdout);
	// Sparse(W);fflush(stdout);
#endif

	// sort columns of W
#pragma omp parallel for shared(n, idxl_src, ia, W) private(mythreadnum, idxl)
	for (j = 0; j < n; j++)
	{
		// who am I
		mythreadnum = omp_get_thread_num();
		// local version of idx to avoid memory conflicts
		idxl = idxl_src[mythreadnum];

		// sort column j prior to exporting
		qsort1_(W.val[j], W.rowind[j], idxl, ia + j);
	} // end for j
	  // end omp parallel for
#ifdef PRINT_INFO
	printf("BNINV: W sorted\n");
	fflush(stdout);
#endif

	// release OpenMP-based memory
	k = omp_get_max_threads();
	for (i = 0; i < k; i++)
	{
		free(idxl_src[i]);
		free(idxl2_src[i]);
		free(idxposl_src[i]);
		free(vall_src[i]);
		free(vall2_src[i]);
		free(idxb_src[i]);
		free(idxbpos_src[i]);
		free(iLTbegin_src[i]);
	} // end for i
	free(idxl_src);
	free(idxl2_src);
	free(idxposl_src);
	free(vall_src);
	free(vall2_src);
	free(idxb_src);
	free(idxbpos_src);
	free(iLTbegin_src);
	free(size_vall_src);
	free(size_vall2_src);
	free(nrml);

#ifdef PRINT_INFO
	printf("BNINV: memory released\n");
	fflush(stdout);
#endif

#ifdef _PROFILING_
	time_bfspai = omp_get_wtime() - time_bfspai;
	printf("profiling summary\n");
	printf("1) Neumann series                           %12.4le\n", time_neumann);
	printf("2) iL^T*BiD*iL                              %12.4le\n", time_matrix_matrix);
	printf("2.0) iL^T setup                             %12.4le\n", time_iLT);
	printf("2.1) first pass  iL^T*BiD*iL                %12.4le\n", time_matrix_matrix1);
	printf("2.2) second pass iL^T*BiD*iL                %12.4le\n", time_matrix_matrix2);
	printf("3) remaining parts                          %12.4le\n", MAX(0.0, time_bfspai - time_neumann - time_matrix_matrix));
	printf("Total BNINV time %12.4le\n\n", time_bfspai);

	fflush(stdout);
#endif

}; // end BNINV

void SQUIC::PrintBlockSparse(SparseBlockMatrix *B)
{
	integer j, k, l, nblocks = B->nblocks, ncols, mrows, *ja, *ia;
	double *valD, *valE, **BvalD = B->valD, **BvalE = B->valE;

	printf("total number of blocks:%ld\n", nblocks);
	fflush(stdout);
	for (j = 0; j < nblocks; j++)
	{
		printf("block column %3ld:\n", j + 1);
		fflush(stdout);

		if (B->nblockcol != NULL)
			ncols = B->nblockcol[j];
		else
			ncols = 0;
		if (B->nblockrow != NULL)
			mrows = B->nblockrow[j];
		else
			mrows = 0;
		if (B->colind != NULL)
			ja = B->colind[j];
		else
			ja = NULL;
		if (B->rowind != NULL)
			ia = B->rowind[j];
		else
			ia = NULL;
		if (BvalD != NULL)
		{
			valD = B->valD[j];
		}
		else
			valD = NULL;
		if (BvalE != NULL)
		{
			valE = B->valE[j];
		}
		else
			valE = NULL;

		printf("            ");
		for (k = 0; k < ncols; k++)
			printf("%12ld", ja[k] + 1);
		printf("\n");
		fflush(stdout);
		if (valD != NULL)
		{
			for (l = 0; l < ncols; l++)
			{
				printf("%12ld", ja[l] + 1);
				for (k = 0; k < ncols; k++)
				{
					printf("%12.4le", valD[k * ncols + l]);
				} // end for l
				printf("\n");
				fflush(stdout);
			} // end for k
		}
		printf("\n");
		fflush(stdout);
		if (valE != NULL && mrows > 0)
		{
			for (l = 0; l < mrows; l++)
			{
				printf("%12ld", ia[l] + 1);
				for (k = 0; k < ncols; k++)
				{
					printf("%12.4le", valE[k * mrows + l]);
				} // end for l
				printf("\n");
				fflush(stdout);
			} // end for k
		}
		printf("\n");
		fflush(stdout);
	} // end for j
};	  // end PrintBlockSparse

#endif
