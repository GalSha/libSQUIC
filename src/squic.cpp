/* squic.cpp
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

#ifndef squic_cpp
#define squic_cpp

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Standard General Utilities Library
#include <cstdio>
#include <iostream>
#include <iomanip> // Precision

// Data structures, algos and numerics
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <assert.h>

// Sleep, time, fprintf etc...
#include <unistd.h>
#include <time.h>
#include <cstdlib>
#include <cstddef> // sleep
#include <stdio.h> // MSG(

// IO
#include <fstream>
#include <iostream>

// Parallel
#include <omp.h>

// METIS
#include "metis.h"

#include "squic.hpp"

 using namespace std;

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))
#define ABS(A) (((A) >= 0) ? (A) : (-(A)))
#define EPS (double(2.2204E-16))

#define MAX_LINEITR 50;

#define MAX_LOOP 16

//#define PRINT_INFO_BCDU
// #define PRINT_INFO1
// #define PRINT_INFOL3B
// #define PRINT_INFOL3B0
//#define PRINT_INFOSR
// #define PRINT_CHECK
//#define PRINT_MSG
// #define PRINT_MSGL3B

// guess an initial tolerance for the approximate inverse W~X^{-1}
#define DROP_TOL0 1e-1
// multiplicative gap between |fX-fXprev|/|fX| and the threshold
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

SQUIC::~SQUIC() {
	delete[] Y;
	FreeSparse(&W);
	FreeSparse(&X);
}; // end ~SQUIC

SQUIC::SQUIC(integer Y_n, integer Y_p, double *Y_value) {
	n = Y_n;
	p = Y_p;
	Y = new double[Y_p * Y_n];
	memcpy(Y, Y_value, Y_p * Y_n * sizeof(double));

	init_SparseMatrix(&X, 1.0);
	init_SparseMatrix(&W, 1.0);
}; // end SQUIC

void SQUIC::SQUIC_InitVars() {

	opt = 0.0;
	dGap = 0.0;
	cputime = 0.0;
	current_iter = 0;
	optSize = 1;
	iterSize = 1;

	Stat.opt.clear();
	Stat.time_itr.clear();
	Stat.time_chol.clear();
	Stat.time_inv.clear();
	Stat.time_lns.clear();
	Stat.time_upd.clear();

	init_SparseMatrix(&L, 0.0);
	init_SparseMatrix(&P, 1.0);

	init_SparseBlockMatrix(&BL);
	init_SparseBlockMatrix(&BiD);

	// Initilize Factorization Routine
		// (CHOL_SUITESPARSE==RunTimeConfig.factorization)
		//if (RunTimeConfig.verbose) {
		//	MSG("\n+ Init Cholesky\n");
		//	fflush(stdout);
		//}
		Cholmod.pA = NULL;
		Cholmod.pL = NULL;

		CHOLMOD_START(&Cholmod.common_parameters);

		// use Cholesky factorization LL' (optionally one could use LDL')
		Cholmod.common_parameters.final_ll = true;
		// terminate the factorization early if not SPD
		Cholmod.common_parameters.quick_return_if_not_posdef = true;
		// allow a little bit of extra space when calling Cholesky
		Cholmod.common_parameters.final_pack = false;
		// automatically decide whether to use simplicial or supernodal approach
		// Cholmod.common_parameters.supernodal = CHOLMOD_AUTO;
		Cholmod.common_parameters.supernodal = CHOLMOD_SUPERNODAL;
		// number of ordering methods to try
		Cholmod.common_parameters.nmethods = 1;
		//Cholmod.common_parameters.method[0].ordering = CHOLMOD_GIVEN ;
		//Cholmod.common_parameters.postorder = true ;
		//Cholmod.common_parameters.grow0 = 10.0;

		Cholmod.common_parameters.print = -1;

		Cholmod.pA = new cholmod_sparse();

		// Be carefull this means that the matrix is (symmetric) but only the lower triangular part is stored
		Cholmod.pA->stype = -1;
		// int or long int depending on the compiler options
		Cholmod.pA->itype = CHOLMOD_INTGER_TYPE;
		// indices are provided in increasing order
		Cholmod.pA->sorted = true;
		// only half of the matrix is stored
		Cholmod.pA->packed = true;
		// real-valued matrix
		Cholmod.pA->xtype = CHOLMOD_REAL;
		// double precision
		Cholmod.pA->dtype = CHOLMOD_DOUBLE;

};	  // end SQUIC_InitVars

void SQUIC::SQUIC_DeleteVars() {
		// (CHOL_SUITESPARSE==RunTimeConfig.factorization)
		//if (RunTimeConfig.verbose)
		//{
		//	MSG("\n+ finish Cholesky\n");
		//	fflush(stdout);
		//}
		if (Cholmod.pL != NULL)
			CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
		CHOLMOD_FINISH(&Cholmod.common_parameters);
		delete Cholmod.pA;

	FreeSparse(&S);
	FreeSparse(&L);
	FreeSparse(&P);

	FreeBlockSparse(&BL);
	FreeBlockSparse(&BiD);
}; // end SQUIC_DeleteVars

// randomized version, entry point without initial guesses (warm start)
void SQUIC::run(double Lambda, double drop_tol, int maxIter, double term_tol) {
	run_core(Lambda, drop_tol, maxIter, term_tol);
} // end SQUIC::run

// randomized version, entry point WITH initial guesses
void SQUIC::run(double Lambda, double drop_tol, int maxIter, double term_tol, SparseMatrix *X0, SparseMatrix *W0) {
	FreeSparse(&X);
	if (X0 == NULL)
		init_SparseMatrix(&X, 1.0);
	else
		CopySparse(&X, X0);
	FreeSparse(&W);
	if (W0 == NULL)
		init_SparseMatrix(&W, 1.0);
	else
		CopySparse(&W, W0);
	run_core(Lambda, drop_tol, maxIter, term_tol);
} // end SQUIC::run

// core randomized active set routine after optionally providing initial guesses for X and W
void SQUIC::run_core(double Lambda, double drop_tol, int maxIter, double term_tol) {

	// init runtime variables
	SQUIC_InitVars();
	// Statistics for output
	int nn;

	Stat.time_total = -omp_get_wtime();

	if (RunTimeConfig.verbose > 0) {
		MSG("----------------------------------------------------------------\n");
		MSG("                     SQUIC Version %.2f                         \n", SQUIC_VER);
		MSG("----------------------------------------------------------------\n");
		MSG("Input Matrices\n");
		MSG(" nnz(X0)/p:   %e\n", double(X.nnz) / double(p));
		MSG(" nnz(W0)/p:   %e\n", double(W.nnz) / double(p));
		MSG(" nnz(M)/p:    ignored\n");
		MSG(" Y:           %d x %d \n", p, n);
		MSG("Runtime Configs   \n");

		// (RunTimeConfig.generate_scm==DETERMINISTIC)
			MSG(" Sample Cov.:   Deterministic\n");

		// (RunTimeConfig.block_upd==COLLECTIVE_CDU)
		MSG(" CordDec Vers:  Collective coordinate descent update \n");

		// RunTimeConfig.inversion==BLOCK_FSPAI
			MSG(" Inversion:     Approx. block Neumann series\n");

		// (CHOL_SUITESPARSE==RunTimeConfig.factorization)
			MSG(" Fact. Routine: CHOLMOD\n");
		//if (RunTimeConfig.off_diagonal)
		//	MSG(" off-dgl 1-nrm: yes\n");
		//else
		//	MSG(" off-dgl 1-nrm: no\n");

		MSG("Parameters       \n");
		MSG(" verbose:     %d \n", RunTimeConfig.verbose);
		MSG(" lambda:      %e \n", Lambda);
		MSG(" max_iter:    %d \n", maxIter);
		MSG(" term_tol:    %e \n", term_tol);
		MSG(" inv_tol:     %e \n", drop_tol);
		MSG(" threads:     %d \n", omp_get_max_threads());
		MSG("\n");

		MSG("#SQUIC Started \n");
		fflush(stdout);
	}

	if (RunTimeConfig.verbose == 0) {
		MSG("#SQUIC Version %.2f : p=%g n=%g lambda=%g max_iter=%g term_tol=%g drop_tol=%g ",
		    SQUIC_VER,
		    double(p),
		    double(n),
		    double(Lambda),
		    double(maxIter),
		    double(term_tol),
		    double(drop_tol));
		fflush(stdout);
	}

	double timeBegin = omp_get_wtime();

	// PrintSparse(X0);
	// PrintSparse(W0);
	// Initialize Matrices

	// Diagonal of in LDL
	double D_LDL[p];
	// permutation vector for reordering
	integer perm[p];
	// diagonal scaling matrix
	double SL[p];

	integer maxNewtonIter = maxIter;
	double cdSweepTol = 0.05;
	integer max_lineiter = MAX_LINEITR;
	double fX = 1e+15;
	double fX1 = 1e+15;
	double fXprev = 1e+15, subgrad, full_subgrad;
	double sigma = 0.001;
	integer i, j, k, l, m, r, s, t, max_loop = MAX_LOOP, SX, flagld = 0;
	integer *idx, *idxpos;
	double *val, Wij, Sij, Xij, Dij, g;
	double temp_double;

	// tolerance used to compute log(det(X)) and X^{-1} approximately
	double current_drop_tol = MAX(DROP_TOL0, drop_tol);
	double *W_a;

	// parallel version requires omp_get_max_threads() local versions of idx, idxpos
	// to avoid memory conflicts between the threads
	k = omp_get_max_threads();
	// #pragma omp critical
	// MSG(("maximum number of available threads %ld\n",k);
	// used as stack and list of indices
	idx = (integer *)malloc((size_t)k * p * sizeof(integer));

	// used as check mark array for nonzero entries, must be initialized with 0
	idxpos = (integer *)calloc((size_t)k * p, sizeof(integer));

	// used as list of nonzero values, must be initialized with 0.0
	val = (double *)malloc((size_t)k * p * sizeof(double));
	for (j = 0; j < k * p; j++)
		val[j] = 0.0;

	integer ierr;
	double *pr;

	// empty container for sparse update matrix \Delta
	SparseMatrix D;
	init_SparseMatrix(&D, 0.0);

	// arithmetic mean value of Y
	double *pY;
	double *mu = (double *)malloc((size_t)p * sizeof(double));
	double max_var = 0.0;
	double temp;

	// generate initial sparse representation of the sample covariance
	// matrix S such that |S_ij|>\Lambda + its diagonal entries
	// nu, ind carry the S_ii in decreasing order
	double *nu = (double *)malloc(p * sizeof(double));
	integer *ind = (integer *)malloc(p * sizeof(integer));

	//////////////////////////////////////////////////
	// compute sample covariance matrix
	//////////////////////////////////////////////////s
	Stat.time_cov = -omp_get_wtime();

	for (i = 0; i < p; i++) {
		// compute mean
		mu[i] = 0.0;
		for (j = 0, pY = Y + i; j < n; j++, pY += p) {
			mu[i] += *pY;
		}
		mu[i] /= (double)n;
	}

	// generate at least the diagonal part of S and those S_ij s.t. |S_ij|>Lambda

#ifdef PRINT_MSG
	double timeBeginS = omp_get_wtime();
	MSG("use %d threads\n\n", omp_get_max_threads());
	MSG("Compute initial S...\n");
	fflush(stdout);
#endif

	// Compute Sparse Sample Covariance Matrix
	// generate at least the diagonal part of S and those S_ij s.t. |S_ij|>Lambda
		// (RunTimeConfig.generate_scm==DETERMINISTIC)
		GenerateSXL3B_OMP(mu, Lambda, nu, ind, &idx, idxpos);
		SX = -1;

#ifdef PRINT_MSG
	MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginS));
	fflush(stdout);
#endif

#ifdef PRINT_CHECK
	SortSparse(&S, idx);
	ierr = CheckSymmetry(&S);
	if (ierr) {
		MSG("!!!SQUIC(genSparseCov): S is nonsymmetric!!!\n");
		fflush(stdout);
	}
#endif
#ifdef PRINT_MSG
	double timeBeginAS = omp_get_wtime();
	MSG("Augment S with the pattern of X...\n");
	fflush(stdout);
#endif
	// AugmentS(mu, idx, idxpos, &X);
	AugmentS_OMP(mu, idx, idxpos, &X);
	// sort nonzero entries in each column of S in increasing order
	// Remark: it suffices to sort S once after GenerateS and AugmentS
	// SortSparse(&S, idx);
	SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
	MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
	fflush(stdout);
#endif

#ifdef PRINT_CHECK
	ierr = CheckSymmetry(&S);
	if (ierr) {
		MSG("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
		fflush(stdout);
	}
#endif

	Stat.time_cov += omp_get_wtime();
	if (RunTimeConfig.verbose > 0) {
		printf("* sample covariance matrix S: time=%.2e nnz(S)/p=%.2e\n",  Stat.time_cov, double(S.nnz) / double(p));
		fflush(stdout);
	}



	//////////////////////////////////////////////////
	// END computation of the sample covariance matrix
	//////////////////////////////////////////////////

	// initially use active Set of tentative size 2p
	integer nactiveSet = 2 * p;
	IndexPair *activeSet = (IndexPair *)malloc((size_t)nactiveSet * sizeof(IndexPair));
	double *activeSetweight2 = (double *)malloc((size_t)nactiveSet * sizeof(double));

	// |\Lambda*X|_1
	double l1normX = 0.0;

	// trace(SX)
	double trSX = 0.0;

	// log(det(X))
	double logdetX = 0.0;

	// scan matrices S and X
	for (j = 0; j < p; j++) {

		// counters for X_{:,j}, S_{:,j}
		k = l = 0;
		while (k < X.ncol[j] || l < S.ncol[j]) {

			// set row indices to a value larger than possible
			r = s = p;

			// row index r of X_rj
			if (k < X.ncol[j]) {
				r = X.rowind[j][k];
			}
			// row index s of S_sj
			if (l < S.ncol[j]) {
				s = S.rowind[j][l];
			}

			// determine smallest index i=min{r,s}<p
			i = r;
			if (s < i) {
				i = s;
			}

			// only load the values of the smallest index
			Xij = Sij = 0.0;
			if (r == i) {
				Xij = X.val[j][k++];
			}
			if (s == i) {
				Sij = S.val[j][l++];
			}

			// increment |X|_1
			if (RunTimeConfig.off_diagonal) {
				if (i != j)
					l1normX += fabs(Xij);
			} // end if
			else
				l1normX += fabs(Xij);

			// increment trace(SX)
			trSX += Sij * Xij;

		} // end while
	}	  // end for j

	// adjust |\Lambda X|_1 by Lambda
	l1normX *= Lambda;
	// Do this after determining Lambda

	// number of active set elements
	integer numActive = 0;

	// counter for the number of computed path steps < pathLen
	integer pathIdx = 0;

	//////////////////////////////////////////////////////////////////////////////////
	// Newton Iteration
	// outer Newton iteration loop, at most maxIter iteration steps unless convergence
	//////////////////////////////////////////////////////////////////////////////////

	// counter for the number of Newton iteration steps <= maxIter
	integer NewtonIter = 1;

	// outer Newton iteration loop, at most maxIter iteration steps unless convergence
	for (; NewtonIter <= maxNewtonIter; NewtonIter++) {
		// MSG("!!! current tolerance: %8.1le!!!\n", current_drop_tol); fflush(stdout);

		Stat.time_itr.push_back(-omp_get_wtime());

		if (RunTimeConfig.verbose > 1) {
			MSG("\n");
		}

		current_iter++;

		double iterTime = omp_get_wtime();

		// |\Delta|_1
		double normD = 0.0;
		// |(\mu e_ie_j^T)_{i,j}|_1
		double diffD = 0.0;
		// tentative norm ~ |grad g|
		subgrad = 1e+15;
		full_subgrad = 1e+15;

		// initial step and diagonal initial inverse covariance matrix X
		if (NewtonIter == 1 && IsDiag(&X)) {

			// Define \Delta on the pattern of S and return the objective value of f(X)
			// we assume that X AND its selected inverse W are diagonal
			// we point out that the initial S covers at least the diagonal part
			// and those S_ij s.t. |S_ij|>\Lambda
#ifdef PRINT_MSG
			double timeBeginDN = omp_get_wtime();
			MSG("Diagonal Newton step...\n");
			fflush(stdout);
#endif
			temp_double = -omp_get_wtime();

			fX = DiagNewton(Lambda, &D);

			temp_double += omp_get_wtime();
			if (RunTimeConfig.verbose > 1) {
				MSG("+ Diagonal Update: time=%e\n", temp_double);
				fflush(stdout);
			}
#ifdef PRINT_MSG
			MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginDN));
			fflush(stdout);
#endif
		} else {

			// Compute the active set and the minimum norm subgradient:
			numActive = 0;

			// update matrix \Delta=0
			ClearSparse(&D);

			// augment S by the pattern of W
#ifdef PRINT_MSG
			timeBeginAS = omp_get_wtime();
			MSG("Augment S with the pattern of W...\n");
			fflush(stdout);
#endif
			// AugmentS(mu, idx, idxpos, &W);
			AugmentS_OMP(mu, idx, idxpos, &W);
			// SortSparse(&S, idx);
			SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
			MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
			fflush(stdout);
#endif
#ifdef PRINT_CHECK
			ierr = CheckSymmetry(&S);

			if (ierr) {
				MSG("!!!SQUIC(Augment S with pattern of W): S is nonsymmetric!!!\n");
				fflush(stdout);
			}
#endif

			//if (RunTimeConfig.verbose > 1)
			//{
			//	MSG("+ Compute Active Set \n");
			//	fflush(stdout);
			//}
			temp_double = omp_get_wtime();

			///////////////////////////////////////////////////////////////////
			// Compute Active Set
			// compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>Lambda
			///////////////////////////////////////////////////////////////////

			// now the pattern of S covers that of W, |S_ij|>\Lambda and its
			// diagonal part
			integer nz = 0;
			// |grad g|_1
			subgrad = 0.0;
			full_subgrad = 0.0;

			// compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>Lambda
			// To do so, scan S, X and W
			for (j = 0; j < p; j++) {
				// counters for S_{:,j}, X_{:,j}, W_{:,j}
				k = l = m = 0;
				while (k < S.ncol[j] || l < X.ncol[j] || m < W.ncol[j]) {
					// set row indices to a value larger than possible
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
					i = r;
					if (s < i) {
						i = s;
					}
					if (t < i) {
						i = t;
					}
					// only load the values of the smallest index
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

					// make sure that only the upper triangular part is referenced!
					if (i <= j) {
						g = Sij - Wij;

						//g_lb = fabs(fabs(Sij) - fabs(Wij));

						// X_ij!=0 or |S_ij-W_ij|>Lambda
						//if (Xij != 0.0 || fabs(g) > Lambda) {
						if (fabs(Xij) > EPS || fabs(g) > Lambda) {
							// augment I_free
							if (numActive >= nactiveSet) {
								nactiveSet += p;
								activeSet = (IndexPair *)realloc(activeSet,
								                                 nactiveSet * sizeof(IndexPair));
								activeSetweight2 = (double *)realloc(activeSetweight2,
								                                     nactiveSet * sizeof(double));
							}

							activeSet[numActive].i = i;
							activeSet[numActive].j = j;

							// update |grad g|_1
							if (RunTimeConfig.off_diagonal) {
								if (i != j) {
									if (Xij > 0.0)
										g += Lambda;
									else if (Xij < 0.0)
										g -= Lambda;
									else
										g = fabs(g) - Lambda;
								}
							} // end if
							else {
								// NOT RunTimeConfig.off_diagonal
								if (Xij > 0.0)
									g += Lambda;
								else if (Xij < 0.0)
									g -= Lambda;
								else
									g = fabs(g) - Lambda;
							} // end if-else RunTimeConfig.off_diagonal

							subgrad += fabs(g);
							full_subgrad += fabs(g);
							full_subgrad += (i != j) * fabs(g);

							// use sub gradient contribution as alternative weight
							activeSetweight2[numActive] = fabs(g);
							numActive++;
						} // end if  Xij!=0 or |g|>Lambda
					}	  // end if i>=j
				}		  // end while
			}			  // end for j

			if (RunTimeConfig.verbose > 1) {
				MSG("+ Active Set: size=%ld time=%e\n", numActive, omp_get_wtime() - temp_double);
				fflush(stdout);
			}
			///////////////////////////////////////////////////////////////////
			// END Compute Active Set
			// compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>Lambda
			///////////////////////////////////////////////////////////////////

			// augment the pattern of \Delta with the pattern from the active set
#ifdef PRINT_MSG
			double timeBeginAD = omp_get_wtime();
			MSG("Augment D...\n");
			fflush(stdout);
#endif
			// AugmentD(&D, idx, idxpos, activeSet, numActive);
			AugmentD_OMP(&D, idx, idxpos, activeSet, numActive);
			// SortSparse(&D, idx);
			SortSparse_OMP(&D, idx);
#ifdef PRINT_MSG
			MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAD));
			fflush(stdout);
#endif
#ifdef PRINT_CHECK
			ierr = CheckSymmetry(&D);
			if (ierr) {
				MSG("!!!SQUIC(Augment D): D is nonsymmetric!!!\n");
				fflush(stdout);
			}
#endif

			// --------------------------------------------------------------------
			// compute refined W=inv(X) using an external function call from MATLAB
			// --------------------------------------------------------------------
			// did we already successfully compute a Cholesky decomposition?
			if (flagld) {
				// if we wish to refine W to improve the accuracy in the
				// update step, right here is the place for a refined
				// computation of W
				// [W,opts]=refineinv(L,D,P,S, W, indi,indj, subgrad/l1normX, opts)

				flagld = 0;

				// in "norefine" is used, this does nothing ...
				// We assume this funciton is used.

			} // end if flagld
			// ------------------------------------------------------------------------
			// END compute refined W=inv(X) using an external function call from MATLAB
			// ------------------------------------------------------------------------

			////////////////////////////////////////
			// Coordinate Decent Update
			////////////////////////////////////////
			Stat.time_upd.push_back(-omp_get_wtime());

				// Set seed !!!!!!!!!!!
				srand(1);

				for (integer cdSweep = 1; cdSweep <= 1 + NewtonIter / 3; cdSweep++) {
					diffD = 0.0;

					// random swap order of elements in the active set
					for (integer i = 0; i < numActive; i++) {
						integer j = i + rand() % (numActive - i);
						integer k1 = activeSet[i].i;
						integer k2 = activeSet[i].j;
						activeSet[i].i = activeSet[j].i;
						activeSet[i].j = activeSet[j].j;
						activeSet[j].i = k1;
						activeSet[j].j = k2;

						double v1 = activeSetweight2[i];
						activeSetweight2[i] = activeSetweight2[j];
						activeSetweight2[j] = v1;

#ifdef PRINT_INFO
						MSG("Add (%2ld,%2ld) %ld to active set\n", k1, k2);
#endif
					}

					// update \Delta_ij  where
					// \Delta' differs from \Delta only in positions (i,j), (j,i)
					// normD, diffD will be updated
#ifdef PRINT_MSG
					double timeBeginCDU = omp_get_wtime();
					MSG("Collective coordinate descent update...\n");
					fflush(stdout);
#endif

					// (RunTimeConfig.block_upd == COLLECTIVE_CDU)
						BlockCoordinateDescentUpdate(Lambda, &D,
						                             idx, idxpos, val,
						                             activeSet, 0, numActive - 1,
						                             normD, diffD);

#ifdef PRINT_MSG
					MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginCDU));
					fflush(stdout);
#endif

					if (diffD <= normD * cdSweepTol)
						break;
				} // end for cdSweep
		}		  // end if-else NewtonIter=1 and IsDiag(X)
#ifdef PRINT_INFOX
		MSG("D:\n");
		PrintSparse(&D);
#endif
		nn = Stat.time_upd.size();
		if (nn > 0) {
			Stat.time_upd[nn - 1] += omp_get_wtime();
			if (RunTimeConfig.verbose > 1) {
				MSG("+ Coordinate Descent Update: time=%e\n", Stat.time_upd[nn - 1]);
				fflush(stdout);
			}
		}
		////////////////////////////////////////
		// END coordinate decent update
		////////////////////////////////////////

		////////////////////////////////////////
		// Factorize log-det for fX == 1e+15
		////////////////////////////////////////
		if (fX == 1e+15) {

			MSG("-------------------------------\n");
			fflush(stdout);
			// (RunTimeConfig.inversion == BLOCK_FSPAI) 
				//MSG("nnz(L)/p=%8.1le\n", double(BL.nnz + BiD.nnz / 2.0) / double(p));
				//fflush(stdout);
#ifdef PRINT_MSG
			double timeBeginLD = omp_get_wtime();

			MSG("log(det(X))...\n");
			fflush(stdout);
#endif

			// ------------------
			// compute log(det X)
			// ------------------
			//if (RunTimeConfig.verbose)
			//	{
			//		MSG("\n+ Cholesky decomposition + logDet\n");
			//		fflush(stdout);
			//	}
			Stat.time_chol.push_back(-omp_get_wtime());
			temp_double = omp_get_wtime();

			ierr = logDet(D_LDL, &logdetX, current_drop_tol);

			nn = Stat.time_chol.size();
			if (nn > 0) {
				Stat.time_chol[nn - 1] += omp_get_wtime();
				if (ierr) {
					if (RunTimeConfig.verbose > 1) {
						MSG("- Factorization failed: time=%e\n", Stat.time_chol[nn - 1]);
						fflush(stdout);
					}
				} else {
					if (RunTimeConfig.verbose > 1) {
							// (RunTimeConfig.factorization==CHOL_SUITESPARSE)
							MSG("- Cholesky successful: time=%e nnz(L)/p=%8.1le  logdetX=%e", Stat.time_chol[nn - 1], double(Cholmod.common_parameters.lnz) / p, logdetX);
							fflush(stdout);
					}
				}
			}
			// ----------------------
			// END compute log(det X)
			// ----------------------

#ifdef PRINT_MSG
			MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginLD));
			fflush(stdout);
#endif
			//MSG("LOGDET\n");
			//fflush(stdout);
			//MSG("nnz(X)/p=%8.1le va=%lf\n", double(X.nnz) / double(p), logdetX);
			//fflush(stdout);
			//MSG("-------------------------------\n");
			//fflush(stdout);

#ifdef PRINT_INFO
			MSG("ierr=%8ld\n", ierr);
			fflush(stdout);
#endif
			if (ierr != 0) {

				MSG("\n Error! Lack of positive definiteness!");
				//	iter[0] = -1;
				free(activeSet);
				free(activeSetweight2);
				SQUIC_DeleteVars();
				// FreeSparse(&S);
				FreeSparse(&D);

				free(mu);
				free(nu);
				free(ind);
				free(idx);
				free(idxpos);
				free(val);
				return;
			}
#ifdef PRINT_INFO
			MSG("log(det(X))=%8.1le\n", logdetX);
			fflush(stdout);
#endif

			// update objective function f(x)=-log(det(X))+trace(SX)+|\Lambda*X|_1
			fX = (trSX + l1normX) - logdetX;
		} // end if (fX == 1e+15)
		////////////////////////////////////////
		// END Factorize log-det for fX == 1e+15
		////////////////////////////////////////

		// compute trace((S-X^{-1})\Delta)
		double trgradgD = 0.0;
		for (j = 0; j < p; j++) {
			// counters for S_{:,j}, W_{:,j}, D_{:,j}
			k = l = m = 0;
			while (k < S.ncol[j] || l < W.ncol[j] || m < D.ncol[j]) {

				// set row indices to a value larger than possible
				r = s = t = p;
				if (k < S.ncol[j]) {
					r = S.rowind[j][k];
				}
				if (l < W.ncol[j]) {
					s = W.rowind[j][l];
				}
				if (m < D.ncol[j]) {
					t = D.rowind[j][m];
				}

				// compute smallest index i=min{r,s,t}<p
				i = r;
				if (s < i) {
					i = s;
				}
				if (t < i) {
					i = t;
				}

				// only load the values of the smallest index
				Sij = Wij = Dij = 0.0;
				if (i == r) {
					Sij = S.val[j][k++];
				}
				if (i == s) {
					Wij = W.val[j][l++];
				}
				if (i == t) {
					Dij = D.val[j][m++];
				}

				// update trace((S-X^{-1})\Delta)
				trgradgD += (Sij - Wij) * Dij;

			} // end while
		}	  // end for j
#ifdef PRINT_INFO
		MSG("trace((S-X^{-1})Delta)=%8.1le\n", trgradgD);
		fflush(stdout);
#endif

		// line search; we overwrite X by X + \alpha \Delta
		// if necessary, we downdate X back
		double alpha = 1.0;
		double l1normXD = 0.0;
		double fX1prev = 1e+15;
		// augment X with the pattern of \Delta

#ifdef PRINT_MSG
		double timeBeginSP = omp_get_wtime();
		MSG("Augment X with pattern of Delta...\n");
		fflush(stdout);
#endif
		// AugmentSparse(&X, idx, idxpos, &D);
		AugmentSparse_OMP(&X, idx, idxpos, &D);
		// SortSparse(&X, idx);
		SortSparse_OMP(&X, idx);
#ifdef PRINT_MSG
		MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginSP));
		fflush(stdout);
#endif
#ifdef PRINT_CHECK
		ierr = CheckSymmetry(X);
		if (ierr) {
			MSG("!!!SQUIC(Augment X with pattern of Delta): X is nonsymmetric!!!\n");
			fflush(stdout);
		}
#endif
		// MSG("augmented X:\n"); PrintSparse(X);

#ifdef PRINT_MSG
		timeBeginAS = omp_get_wtime();
		MSG("Augment S with the pattern of X...\n");
		fflush(stdout);
#endif
		// AugmentS(mu, idx, idxpos, &X);
		AugmentS_OMP(mu, idx, idxpos, &X);
		// SortSparse(&S, idx);
		SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
		MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
		fflush(stdout);
#endif
#ifdef PRINT_CHECK
		ierr = CheckSymmetry(&S);
		if (ierr) {
			MSG("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
			fflush(stdout);
		}
#endif

		//MSG("-------------------------------\n");
		//fflush(stdout);
		//MSG("Log Det\n");
		//fflush(stdout);
		//MSG("nnz(X)/p=%8.1le\n", double(X.nnz) / double(p));
		//fflush(stdout);

		if (full_subgrad * Lambda < l1normX * term_tol)
		{
			Stat.time_total += omp_get_wtime();
			NewtonIter--;
			break;
		}
		/////////////////////////
		// Line Search
		// Factorization
		/////////////////////////
		Stat.time_lns.push_back(-omp_get_wtime());

		int accept_update = 0;

		// Update of X
		double loc_current_drop_tol = current_drop_tol;

		flagld = 0;
		integer lineiter = 0;
		// for (integer lineiter=0; lineiter<max_lineiter; lineiter++) {
		for (lineiter = 0; lineiter < max_lineiter && flagld == 0; lineiter++) {

			double l1normX1 = 0.0;
			double trSX1 = 0.0;
			double logdetX1 = 0.0;

			////////////////////////////////////////
			// Break Loop Condition
			////////////////////////////////////////
			if (accept_update) {

				if (RunTimeConfig.verbose > 1) {
					MSG("+ Line Search: alpha=%e\n", alpha);
					fflush(stdout);
				}
				break;
			}

			////////////////////////////////////////
			// Update alpha
			////////////////////////////////////////
			if (lineiter > 0) {
				alpha *= 0.5;
				// possibly the LDL decomposition is not accurate enough
				loc_current_drop_tol *= SHRINK_DROP_TOL;
			}

			////////////////////////////////////////
			// update X <- X + \alpha \Delta
			////////////////////////////////////////

			for (j = 0; j < p; j++) {
				// counters for S_{:,j}, X_{:,j}, D_{:,j}
				k = l = m = 0;
				while (k < S.ncol[j] || l < X.ncol[j] || m < D.ncol[j]) {
					// set row indices to a value larger than possible
					r = s = t = p;
					if (k < S.ncol[j]) {
						r = S.rowind[j][k];
					}
					if (l < X.ncol[j]) {
						s = X.rowind[j][l];
					}
					if (m < D.ncol[j]) {
						t = D.rowind[j][m];
					}
					// compute smallest index i=min{r,s,t}<p
					i = r;
					if (s < i) {
						i = s;
					}
					if (t < i) {
						i = t;
					}
					// only load the values of the smallest index
					Sij = Xij = Dij = 0.0;
					if (i == r) {
						Sij = S.val[j][k++];
					}
					if (i == t) {
						Dij = D.val[j][m++];
					}
					// load and update X_ij <- X_ij + \alpha \Delta_ij
					if (i == s) {
						Xij = X.val[j][l];
						Xij += Dij * alpha;
						X.val[j][l++] = Xij;
					}

					// |X|_1
					if (RunTimeConfig.off_diagonal) {
						if (i != j)
							l1normX1 += fabs(Xij);
					} // end if
					else
						l1normX1 += fabs(Xij);

					// trace(SX)
					trSX1 += Sij * Xij;
				} // end while
			}	  // end for j
			// adjust by \Lambda
			l1normX1 *= Lambda;

			////////////////////////////////////////
			// Compute Cholesky+log(det X)
			////////////////////////////////////////
			//if (RunTimeConfig.verbose)
			//{
			//	MSG("\n+ Cholesky decomposition + logDet\n");
			//	fflush(stdout);
			//}
			Stat.time_chol.push_back(-omp_get_wtime());
			temp_double = omp_get_wtime();
#ifdef PRINT_MSG
			double timeBeginLD = omp_get_wtime();
			MSG("log(det(X))...\n");
			fflush(stdout);
#endif

			ierr = logDet(D_LDL, &logdetX1, loc_current_drop_tol);

			nn = Stat.time_chol.size();
			if (nn > 0) {
				Stat.time_chol[nn - 1] += omp_get_wtime();
				if (ierr) {
					if (RunTimeConfig.verbose > 1) {
						MSG("- Factorization failed: iteration=%li time=%e alpha=%e\n", lineiter, Stat.time_chol[nn - 1], alpha);
						fflush(stdout);
					}
				} else {
					if (RunTimeConfig.verbose > 1) {
							// (RunTimeConfig.factorization==CHOL_SUITESPARSE)
							MSG("- Factorization successful: iteration=%li time=%e alpha=%e nnz(L)/p=%8.1le logdetX=%e\n", lineiter, Stat.time_chol[nn - 1], alpha, double(Cholmod.common_parameters.lnz) / p, logdetX);
							fflush(stdout);
					}
				}
			}

#ifdef PRINT_MSG
			MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginLD));
			fflush(stdout);
#endif

			// Cholesky failed, retry line search with smaller alpha
			if (ierr != 0) {
				// no Cholesky decomposition available
				flagld = 0;

				// downdate X <- X - \alpha \Delta
				for (j = 0; j < p; j++) {
					// counters for X_{:,j}, D_{:,j}
					l = m = 0;
					while (l < X.ncol[j] || m < D.ncol[j]) {
						// set row indices to a value larger than possible
						s = t = p;
						if (l < X.ncol[j]) {
							s = X.rowind[j][l];
						}
						if (m < D.ncol[j]) {
							t = D.rowind[j][m];
						}
						// compute smallest index i=min{s,t}<p
						i = s;
						if (t < i) {
							i = t;
						}
						// only load the values of the smallest index
						Xij = Dij = 0.0;
						if (i == t) {
							Dij = D.val[j][m++];
						}
						// load and downdate X_ij <- X_ij + \alpha \Delta_ij
						if (i == s) {
							Xij = X.val[j][l];
							Xij -= Dij * alpha;
							X.val[j][l++] = Xij;
						}
					} // end while
				}	  // end for j

				// try line search with alpha/2 instead
				//alpha *= 0.5;  <---- do this at the very top of the iteration
				continue;
			} else {
				flagld = -1;
			}

			// update objective function f(X_1)=-log(det(X_1))+trace(SX_1)+|\Lambda X_1|_1
			fX1 = (trSX1 + l1normX1) - logdetX1;
			if (alpha == 1.0) {
				l1normXD = l1normX1;
			}

			// line search successful, accept and break
			if (fX1 <= fX + alpha * sigma * (trgradgD + l1normXD - l1normX) || normD == 0) {
				fXprev = fX;
				fX = fX1;
				l1normX = l1normX1;
				logdetX = logdetX1;
				trSX = trSX1;

				accept_update = 1;
				continue;
			}

			// line search successful, accept and break
			// AE Edit - if (fX1prev < fX1) {
			if (fXprev < fX1) {
				fXprev = fX;
				l1normX = l1normX1;
				logdetX = logdetX1;
				trSX = trSX1;
				accept_update = 1;
				continue;
			}

			fX1prev = fX1;

			// downdate X <- X - \alpha \Delta
			for (j = 0; j < p; j++) {
				// counters for X_{:,j}, D_{:,j}
				l = m = 0;
				while (l < X.ncol[j] || m < D.ncol[j]) {
					// set row indices to a value larger than possible
					s = t = p;
					if (l < X.ncol[j]) {
						s = X.rowind[j][l];
					}
					if (m < D.ncol[j]) {
						t = D.rowind[j][m];
					}
					// compute smallest index i=min{s,t}<p
					i = s;
					if (t < i) {
						i = t;
					}
					// only load the values of the smallest index
					Xij = Dij = 0.0;
					if (i == t) {
						Dij = D.val[j][m++];
					}
					// load and downdate X_ij <- X_ij + \alpha \Delta_ij
					if (i == s) {
						Xij = X.val[j][l];
						Xij -= Dij * alpha;
						X.val[j][l++] = Xij;
					}
				} // end while
			}	  // end for j

			// next try line search with alpha/2
			//alpha *= 0.5;  <---- do this at the very top of the iteration
		} // end for lineiter

		// AE edit - if choleksy fails, dont export and invert L, just end newton
		nn = Stat.time_lns.size();
		if (nn > 0) {
			Stat.time_lns[nn - 1] += omp_get_wtime();
			if (RunTimeConfig.verbose > 1) {
				MSG("+ Line-Search: time=%e\n", Stat.time_lns[nn - 1]);
				fflush(stdout);
			}
		}

		double L_nnz_per_row = 0;
		if (flagld == 0) {
			if (Stat.time_itr.size() > 0) {

			} Stat.time_itr[Stat.time_itr.size() - 1] += omp_get_wtime();
			MSG("# Cholesky Failed: Exiting at current iterate.");
			break;
		}

		if (RunTimeConfig.verbose > 1) {
			MSG("+ Objective value decreased by = %e.\n", fXprev - fX);
			MSG("+ fXprev = %e.\n", fXprev);
			MSG("+ fX = %e.\n", fX);
			MSG("+ fabs((fX - fXprev) / fX = %e.\n", fabs((fX - fXprev) / fX));
		}

		///////////////////////////////////////////
		/// export triangular factorization
		///////////////////////////////////////////
#ifdef PRINT_MSG
		integer timeBeginEX = omp_get_wtime();
		MSG("export factorization...\n");
		fflush(stdout);
#endif

		// export L,D to sparse matrix format along with the permutation
		// at the same time discard the Cholesky factorization
		// (RunTimeConfig.inversion == BLOCK_FSPAI) 
			BlockExport_LDL();
			L_nnz_per_row = double(BL.nnz + BiD.nnz / 2.0) / double(p);
			//	MSG("nnz(L)/p=%8.1le\n", double(BL.nnz + BiD.nnz / 2.0) / double(p));
			//	fflush(stdout);

		//MSG("-------------------------------\n");
		//fflush(stdout);

		/*
		MSG("L:\n");
		PrintSparse(&L);
		MSG("1/D_LDL:\n");
		for (integer ii=0; ii<p; ii++)
		    MSG("%12.4le",1.0/D_LDL[ii]);
		MSG("\n\n");
		*/

		// next time we call Cholesky the pattern may change
		analyze_done = 0;

#ifdef PRINT_MSG
		MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginEX));
		fflush(stdout);
#endif
		///////////////////////////////////////////
		/// END export triangular factorization
		///////////////////////////////////////////

		///////////////////////////////////////
		// Inversion
		// Compute approximate inverse of X
		///////////////////////////////////////

		Stat.time_inv.push_back(-omp_get_wtime());
		// --------------------------------------------------------------
		// compute W=inv(X) using an external function call from MATLAB
		// --------------------------------------------------------------
		//MSG("-------------------------------\n");
		//fflush(stdout);
		//MSG("SELINV/NINV\n");
		//fflush(stdout);
		//MSG("nnz(X)/p=%8.1le\n", double(X.nnz) / double(p));
		//fflush(stdout);
#ifdef PRINT_MSG
		double timeBeginSI = omp_get_wtime();
#endif

		// compute W ~ X^{-1} using (block incomplete) Cholesky decomposition
		//(RunTimeConfig.inversion == BLOCK_FSPAI) 
			BNINV(current_drop_tol, NULL);

		// PrintSparse(&W);

#ifdef PRINT_MSG
		MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginSI));
		fflush(stdout);
#endif
		//MSG("nnz(W)/p=%8.1le\n", double(W.nnz) / double(p));
		//fflush(stdout);
		//MSG("-------------------------------\n");
		//fflush(stdout);

		// obsolete
		// // SortSparse(&W, idx);
		// SortSparse_OMP(&W, idx);

		nn = Stat.time_inv.size();
		if (nn > 0) {
			Stat.time_inv[nn - 1] += omp_get_wtime();
			if (RunTimeConfig.verbose > 1) {
				MSG("+ Approximate Matrix Inversion: time=%e nnz(W)/p=%8.1le \n", Stat.time_inv[nn - 1], double(W.nnz) / double(p));
				fflush(stdout);
			}
		}
		// ----------------------------------------------------------------
		// END compute W=inv(X) using an external function call from MATLAB
		// ----------------------------------------------------------------
		///////////////////////////////////////////////////
		// END computation of the  approximate inverse of X
		///////////////////////////////////////////////////

		/*
		if (RunTimeConfig.verbose > 2)
		{
			MSG("\n\n==============================================\n");
			MSG("=========Intermediate Matrix Values===========\n");
			MSG("==============================================\n\n");

			MSG("S=[\n");
			print(&S, p);
			MSG("]\n");
			fflush(stdout);

			MSG("X=[\n");
			print(&X, p);
			MSG("]\n");
			fflush(stdout);

			MSG("W=[\n");
			print(&W, p);
			MSG("]\n");
			fflush(stdout);


			MSG("D=[\n");
			integer precis = 4;
			integer flag = 0;
			integer space = 9;
			std::cout.setf(std::ios::fixed, std::ios::floatfield);
			std::cout.precision(precis);
			for (integer i = 0; i < p; i++)
				std::cout << std::right << std::setw(space) << D_LDL[i] << " ";
			std::cout << std::endl;
			MSG("]\n");
			fflush(stdout);

			MSG("P=[\n");
			print(&P, p);
			MSG("]\n");
			fflush(stdout);

			MSG("Delta=[\n");
			print(&D, p);
			MSG("]\n");
			fflush(stdout);

			MSG("==============================================\n");
			fflush(stdout);
		}

		*/


		//printf(">>>>>>>> fX=%e, fXprev=%e \n", fX, fXprev);

		nn = Stat.time_itr.size();
		if (nn > 0) {
			Stat.time_itr[nn - 1] += omp_get_wtime();

			if (RunTimeConfig.verbose > 0) {
				MSG("* iter=%d time=%.2e obj=%.2e |delta(obj)|/obj=%.2e nnz(X,L,W)/p=[%.2e %.2e %.2e] lns_iter=%d \n",
				    NewtonIter,
				    Stat.time_itr[nn - 1],
				    fX,
				    fabs((fX - fXprev) / fX),
				    double(X.nnz) / double(p),
				    L_nnz_per_row,
				    double(W.nnz) / double(p),
				    lineiter);
				fflush(stdout);
			}
		} else {
			if (RunTimeConfig.verbose > 0) {
				MSG("* iter=%d obj=%.2e |delta(obj)|/obj=%.2e nnz(X,L,W)/p=[%.2e %.2e %.2e] lns_iter=%d \n",
				    NewtonIter,
				    fX,
				    fabs((fX - fXprev) / fX),
				    double(X.nnz) / double(p),
				    L_nnz_per_row,
				    double(W.nnz) / double(p),
				    lineiter);
				fflush(stdout);
			}
		}

		Stat.opt.push_back(fX);

		//if (RunTimeConfig.verbose)
		//{
		//	MSG("=================\n");
		//	fflush(stdout);
		//}

		//MSG("* Iteration time: %e seconds\n", omp_get_wtime() - iterTime);

		// Check for convergence.

		// not yet converged
		// we use at least the relative residual between two consecutive calls
		// + some upper/lower bound
		current_drop_tol = DROP_TOL_GAP * fabs((fX - fXprev) / fX);
		current_drop_tol = MAX(current_drop_tol, drop_tol);
		current_drop_tol = MIN(MAX(DROP_TOL0, drop_tol), current_drop_tol);
		//if (subgrad * alpha >= l1normX * term_tol && (fabs((fX - fXprev) / fX) >= EPS))
		//if (fabs((fX - fXprev) / fX) >= term_tol) {
		//	continue;
		//}

		//break;
		continue;
	} // end for(; NewtonIter < maxNewtonIter; .....

	opt = fX;

	//cout << "dGap : " << dGap << endl;

	// The computed W does not satisfy |W - S|< Lambda.  Project it.
	// now the meaning of U changes to case (c) and U is used as a buffer
	// only, since we are interested in log(det(project(W))), which in turn
	// requires its Cholesky factorization
	// compute U<-projected(W)
#ifdef PRINT_MSG
	timeBeginAS = omp_get_wtime();
	MSG("augment S with the pattern of refined W...\n");
	fflush(stdout);
#endif

	// --------------------------------------------------------------------
	// compute refined W=inv(X) using an external function call from MATLAB
	// --------------------------------------------------------------------
	// did we already successfully compute a Cholesky decomposition?
	if (flagld) {
		// if we wish to refine W to improve the accuracy in the
		// update step, right here is the place for a refined
		// computation of W

		// in "norefine" is used, this does nothing ...
		// We assume this funciton is used.
	} // end if flagld
	// ------------------------------------------------------------------------
	// END compute refined W=inv(X) using an external function call from MATLAB
	// ------------------------------------------------------------------------

	// AugmentS(mu, idx, idxpos, &W);
	AugmentS_OMP(mu, idx, idxpos, &W);
	// SortSparse(&S, idx);
	SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
	MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
	fflush(stdout);
#endif
#ifdef PRINT_CHECK
	ierr = CheckSymmetry(&S);
	if (ierr) {
		MSG("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
		fflush(stdout);
	}
#endif
#ifdef PRINT_MSG
	double timeBeginPLD = omp_get_wtime();
	MSG("projected Log Det\n");
	fflush(stdout);
#endif

	// own flag to indicate that the symbolic analysis of the LDL solver has not
	// yet been performed, for computing the log of the determinant of
	// the projected W no a priori analysis is available,  the projected
	// matrix differs from W by some soft thresholding and is usually much
	// denser than X
	analyze_done = 0;

	//////////////////////////////////////////////////////////////////////////////////
	// Project Log Det
	//////////////////////////////////////////////////////////////////////////////////
	Stat.time_chol.push_back(-omp_get_wtime());
	temp_double = omp_get_wtime();
	// at this point QUIC assumes that the final projected X is SPD
	double logdetW = projLogDet(Lambda, current_drop_tol);

	nn = Stat.time_chol.size();
	if (nn > 0)
		Stat.time_chol[nn - 1] += omp_get_wtime();

	if (RunTimeConfig.verbose == 4) {
			//(RunTimeConfig.factorization==CHOL_SUITESPARSE)
			MSG("* project log det: time=%0.2e nnz(X,L,W)/p=[%0.2e %0.2e %0.2e]\n",
			    (omp_get_wtime() - temp_double),
			    double(X.nnz) / double(p),
			    double(Cholmod.common_parameters.lnz) / double(p),
			    double(W.nnz) / double(p)

			   );
		fflush(stdout);
	}

	//////////////////////////////////////////////////////////////////////////////////
	// END Project Log Det
	//////////////////////////////////////////////////////////////////////////////////



	if (Stat.time_total < 0) Stat.time_total += omp_get_wtime();

	if (RunTimeConfig.verbose > 0) {
		MSG("#SQUIC Finished: time=%0.2e nnz(X,W)/p=[%0.2e %0.2e] \n\n",  Stat.time_total, double(X.nnz) / double(p), double(W.nnz) / double(p));
		fflush(stdout);
	}
	if (RunTimeConfig.verbose == 0) {
		MSG("Iter %d: time=%0.2e nnz(X,W)/p=[%0.2e %0.2e] \n",  NewtonIter, Stat.time_total, double(X.nnz) / double(p), double(W.nnz) / double(p));
		MSG("GLASSO_ITER%dGLASSO_ITERGLASSO_TIME%0.6fGLASSO_TIME\n",  NewtonIter, Stat.time_total);
		fflush(stdout);
	}

#ifdef PRINT_MSG
	MSG("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginPLD));
	fflush(stdout);
#endif

	double gap = -logdetW - p - logdetX + trSX + l1normX;
	dGap = gap;

	Stat.dgap = gap;
	Stat.trSX = trSX;
	Stat.logdetX = logdetX;

	free(activeSet);
	free(activeSetweight2);

	//  SQUIC_DeleteVars(); // you have to call this manually!!!!!!
	//	FreeSparse(&S);
	FreeSparse(&D);

	free(mu);
	free(nu);
	free(ind);
	free(idx);
	free(idxpos);
	free(val);

	// if (RunTimeConfig.verbose)
	// {
	//	print_stats();
	//	fflush(stdout);
	// }

	return;
}; // end run


void SQUIC::FreeSparse(SparseMatrix *S) {
	integer j, p;

	if (S == NULL)
		return;
	p = S->nc;
	if (S->rowind != NULL) {
		for (j = 0; j < p; j++) {
			if (S->rowind[j] != NULL)
				free(S->rowind[j]);
		} // end for j
		free(S->rowind);
		S->rowind = NULL;
	} // end if
	if (S->val != NULL) {
		for (j = 0; j < p; j++) {
			if (S->val[j] != NULL)
				free(S->val[j]);
		} // end for j
		free(S->val);
		S->val = NULL;
	} // end if
	if (S->ncol != NULL) {
		free(S->ncol);
		S->ncol = NULL;
	} // end if
	S->nr = S->nc = S->nnz = 0;
}; // end FreeSparse

void SQUIC::FreeBlockSparse(SparseBlockMatrix *S) {
	integer j, p;

	if (S == NULL)
		return;
	p = S->nc;
	if (S->rowind != NULL) {
		for (j = 0; j < p; j++)
			if (S->rowind[j] != NULL)
				free(S->rowind[j]);
		free(S->rowind);
		S->rowind = NULL;
	} // end if

	if (S->colind != NULL) {
		for (j = 0; j < p; j++)
			if (S->colind[j] != NULL)
				free(S->colind[j]);
		free(S->colind);
		S->colind = NULL;
	} // end if

	if (S->valD != NULL) {
		for (j = 0; j < p; j++)
			if (S->valD[j] != NULL)
				free(S->valD[j]);
		free(S->valD);
		S->valD = NULL;
	} // end if

	if (S->valE != NULL) {
		for (j = 0; j < p; j++)
			if (S->valE[j] != NULL)
				free(S->valE[j]);
		free(S->valE);
		S->valE = NULL;
	} // end if

	if (S->nblockrow != NULL) {
		free(S->nblockrow);
		S->nblockrow = NULL;
	} // end if

	if (S->nblockcol != NULL) {
		free(S->nblockcol);
		S->nblockcol = NULL;
	} // end if

	S->nr = S->nc = S->nblocks = S->nnz = 0;
}; // end FreeBlockSparse

void SQUIC::NullSparse(SparseMatrix *S) {
	integer j, p = S->nc;
	for (j = 0; j < p; j++) {
		free(S->rowind[j]);
		free(S->val[j]);
		S->rowind[j] = NULL;
		S->val[j] = NULL;
		S->ncol[j] = 0;
	} // end for j

	S->nr = S->nc = S->nnz = 0;
};

// check for symmetry of S
integer SQUIC::CheckSymmetry(SparseMatrix *S) {

	integer i, j, k, l, p = S->nr, ierr = 0;
	SparseMatrix ST;
	integer *idx = (integer *)calloc(p, sizeof(integer));

	ST.nr = ST.nc = p;
	ST.ncol = (integer *)calloc(p, sizeof(integer));
	ST.rowind = (integer **)malloc(p * sizeof(integer *));
	ST.val = (double **)malloc(p * sizeof(double *));

	// 1. count number of nonzeros per row
	for (j = 0; j < p; j++) {
		for (k = 0; k < S->ncol[j]; k++) {
			i = S->rowind[j][k];
			idx[i]++;
		} // end for k
	}	  // end for j

	// 1. check nonzeros per column versus nonzeros per row and allocate memory
	for (i = 0; i < p; i++) {
		k = idx[i];
		ST.ncol[i] = k;
		if (k > 0) {
			ST.rowind[i] = (integer *)malloc(k * sizeof(integer));
			ST.val[i] = (double *)malloc(k * sizeof(double));
		} else {
			ST.rowind[i] = NULL;
			ST.val[i] = NULL;
		}

		l = S->ncol[i];
		if (k != l) {
			ierr = -1;
		} // end if
		// reset nz counter
		idx[i] = 0;
	} // end for i
	if (ierr) {
		free(idx);
		for (i = 0; i < p; i++) {
			if (ST.rowind[i] != NULL) {
				free(ST.rowind[i]);
			}
			if (ST.val[i] != NULL) {
				free(ST.val[i]);
			}
		}
		free(ST.rowind);
		free(ST.val);
		free(ST.ncol);
		return (ierr);
	}

	// store S^T
	for (j = 0; j < p; j++) {
		for (k = 0; k < S->ncol[j]; k++) {
			i = S->rowind[j][k];
			l = idx[i]++;
			ST.rowind[i][l] = j;
			ST.val[i][l] = S->val[j][k];
		} // end for k
	}	  // end for j
	for (j = 0; j < p; j++) {
		for (k = 0; k < S->ncol[j]; k++) {
			if (S->rowind[j][k] != ST.rowind[j][k] || ST.val[j][k] != S->val[j][k]) {
				ierr = -1;
			} // end if
		}	  // end for k,l
		// reset nz counter
		idx[j] = 0;
	} // end for j

	free(idx);
	for (i = 0; i < p; i++) {
		if (ST.rowind[i] != NULL) {
			free(ST.rowind[i]);
		}
		if (ST.val[i] != NULL) {
			free(ST.val[i]);
		}
	}
	free(ST.rowind);
	free(ST.val);
	free(ST.ncol);

	return (ierr);
};

void SQUIC::PrintSparseCompact(SparseMatrix *S) {
	integer i, j, k, p = S->nr;
	double val;

	MSG("%ldx%ld, nnz=%ld\n", S->nr, S->nc, S->nnz);
	fflush(stdout);

	for (j = 0; j < p; j++) {

		MSG("column %3ld\n", j + 1);
		fflush(stdout);

		for (k = 0; k < S->ncol[j]; k++) {
			i = S->rowind[j][k];
			MSG("%8ld", i + 1);
		} // end for k

		MSG("\n");
		fflush(stdout);
		for (k = 0; k < S->ncol[j]; k++) {
			val = S->val[j][k];
			MSG("%lf ", val);
		} // end for k

		MSG("\n");
		fflush(stdout);
	} // end for j
	MSG("\n");
	fflush(stdout);
};

void SQUIC::PrintSparse(SparseMatrix *S) {

	int precision = 4;
	int flag = 0;
	int space = 9;

	cout.setf(ios::fixed, ios::floatfield);
	cout.precision(precision);

	for (integer i = 0; i < S->nr; i++) {
		for (integer j = 0; j < S->nc; ++j) {

			flag = 0;
			for (integer k = 0; k < S->ncol[j]; k++) {
				if (S->rowind[j][k] == i) {
					cout << std::right << setw(space) << S->val[j][k] << " ";
					flag = 1;
					break;
				}
			}

			if (!flag) {

				cout << std::right << setw(space) << 0 << " ";
			}
		}
		cout << endl;
		fflush(stdout);
	}
};

// init sparse matrix with 0
void SQUIC::ClearSparse(SparseMatrix *S) {
	integer j, k, p = S->nr;

	// #pragma omp parallel for schedule(static,1) shared(p,S) private(k)
	for (j = 0; j < p; j++)
		for (k = 0; k < S->ncol[j]; k++) {
			S->val[j][k] = 0.0;
		}
}; // end ClearSparse

void SQUIC::init_SparseMatrix(SparseMatrix *M, double diagVal) {

	if (diagVal != 0.0) {

		M->nr = M->nc = p;
		M->nnz = p;

		M->ncol = (integer *)malloc(p * sizeof(integer));
		M->rowind = (integer **)malloc(p * sizeof(integer *));
		M->val = (double **)malloc(p * sizeof(double));

		for (integer j = 0; j < p; j++) {
			M->rowind[j] = (integer *)malloc(1 * sizeof(integer));
			M->val[j] = (double *)malloc(1 * sizeof(double));
			M->rowind[j][0] = j;
			M->val[j][0] = diagVal;
			M->ncol[j] = 1;
		} // end for j
	} else {
		M->nr = M->nc = p;
		M->nnz = 0;
		// columns are empty, init with 0
		M->ncol = (integer *)calloc(p, sizeof(integer));
		M->rowind = (integer **)malloc(p * sizeof(integer *));
		M->val = (double **)malloc(p * sizeof(double *));
		for (integer j = 0; j < p; j++) {
			M->rowind[j] = NULL;
			M->val[j] = NULL;
		} // end for j
	}
};

void SQUIC::init_SparseBlockMatrix(SparseBlockMatrix *M) {

	// total size
	M->nr = M->nc = p;
	// initially, the total number of blocks is unknown but bounded by p
	M->nblocks = M->nnz = 0;
	// number columns per diagonal block
	M->nblockcol = (integer *)calloc((size_t)p, sizeof(integer));
	// number of rows per sub-diagonal block
	M->nblockrow = (integer *)calloc((size_t)p, sizeof(integer));
	// column indices for each diagonal block
	M->colind = (integer **)malloc((size_t)p * sizeof(integer *));
	// row indices for the sub-diagonal block entries
	M->rowind = (integer **)malloc((size_t)p * sizeof(integer *));
	M->valD = (double **)malloc((size_t)p * sizeof(double *));
	M->valE = (double **)malloc((size_t)p * sizeof(double *));
	for (integer j = 0; j < p; j++) {
		M->rowind[j] = NULL;
		M->colind[j] = NULL;
		M->valD[j] = NULL;
		M->valE[j] = NULL;
	} // end for j
};	  // end init_SparseBlockMatrix

void SQUIC::reset_SparseBlockMatrix(SparseBlockMatrix *M) {

	for (integer j = 0; j < p; j++) {
		if (M->rowind[j] != NULL) {
			free(M->rowind[j]);
			M->rowind[j] = NULL;
		}
		if (M->colind[j] != NULL) {
			free(M->colind[j]);
			M->colind[j] = NULL;
		}
		if (M->valD[j] != NULL) {
			free(M->valD[j]);
			M->valD[j] = NULL;
		}
		if (M->valE[j] != NULL) {
			free(M->valE[j]);
			M->valE[j] = NULL;
		}
	} // end for j
};	  // end reset_SparseBlockMatrix

void SQUIC::delete_SparseBlockMatrix(SparseBlockMatrix *M) {

	for (integer j = 0; j < p; j++) {
		if (M->rowind[j] != NULL) {
			free(M->rowind[j]);
			M->rowind[j] = NULL;
		}
		if (M->colind[j] != NULL) {
			free(M->colind[j]);
			M->colind[j] = NULL;
		}
		if (M->valD[j] != NULL) {
			free(M->valD[j]);
			M->valD[j] = NULL;
		}
		if (M->valE[j] != NULL) {
			free(M->valE[j]);
			M->valE[j] = NULL;
		}
	} // end for j
	// total size
	M->nr = M->nc = 0;
	// initially, the total number of blocks is unknown but bounded by p
	M->nblocks = M->nnz = 0;
	// number columns per diagonal block
	if (M->nblockcol != NULL) {
		free(M->nblockcol);
		M->nblockcol = NULL;
	}
	// number of rows per sub-diagonal block
	if (M->nblockrow != NULL) {
		free(M->nblockrow);
		M->nblockrow = NULL;
	}
	// column indices for each diagonal block
	if (M->colind != NULL) {
		free(M->colind);
		M->colind = NULL;
	}
	if (M->rowind != NULL) {
		free(M->rowind);
		M->rowind = NULL;
	}
	if (M->valD != NULL) {
		free(M->valD);
		M->valD = NULL;
	}
	if (M->valE != NULL) {
		free(M->valE);
		M->valE = NULL;
	}
}; // end delete_SparseBlockMatrix

// S <- W
void SQUIC::CopySparse(SparseMatrix *S, SparseMatrix *W) {

	integer i, j, k, p = W->nc;

	S->nr = W->nr;
	S->nc = W->nc;
	S->nnz = W->nnz;
	S->ncol = (integer *)malloc(p * sizeof(integer));
	memcpy(S->ncol, W->ncol, p * sizeof(integer));
	S->rowind = (integer **)malloc(p * sizeof(integer *));
	S->val = (double **)malloc(p * sizeof(double *));

	// #pragma omp parallel for schedule(static,1) shared(p,S,W) private(k)
	for (j = 0; j < p; j++) {
		k = W->ncol[j];
		S->rowind[j] = (integer *)malloc(k * sizeof(integer));
		S->val[j] = (double *)malloc(k * sizeof(double));
		memcpy(S->rowind[j], W->rowind[j], k * sizeof(integer));
		memcpy(S->val[j], W->val[j], k * sizeof(double));
	} // end for j
};

// Augment sparse matrix S with the pattern of W init with 0
void SQUIC::AugmentSparse(SparseMatrix *S, integer *idx, integer *idxpos, SparseMatrix *W) {

	/*
	S          sparse matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	W          sparse matrix
	*/

	integer i, j, k, p = W->nc,
	                 cnt; // counter for the number of nonzeros per column

	S->nnz = 0;
	for (j = 0; j < p; j++) {

		// initially set check marks for the existing pattern of S
		// counter for the number of nonzeros per column
		cnt = 0;
		for (k = 0; k < S->ncol[j]; k++) {
			// index i of S_ij
			i = S->rowind[j][k];
			idx[cnt++] = i;
			idxpos[i] = cnt;
		} // end for k

		// now check if W has any additional nonzeros
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (!idxpos[i]) {
				idx[cnt++] = i;
				// flag them negative to indicate fill-in
				idxpos[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		S->rowind[j] = (integer *)realloc(S->rowind[j], (size_t)cnt * sizeof(integer));
		S->val[j] = (double *)realloc(S->val[j], (size_t)cnt * sizeof(double));

		// restore original number of nonzeros in S
		cnt = S->ncol[j];
		// check W again and only add the entries which were flagged negative
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (idxpos[i] < 0) {
				S->rowind[j][cnt] = i;
				S->val[j][cnt++] = 0.0;
			} // end if
		}	  // end for k
		// new number of nonzeros in S
		S->ncol[j] = cnt;
		S->nnz += cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idx[k];
			idxpos[i] = 0;
		} // end for k
	}	  // end for j
};		  // end AugmentSparse

// Augment sparse matrix S with the pattern of W init with 0
void SQUIC::AugmentSparse_OMP(SparseMatrix *S, integer *idx, integer *idxpos,
                              SparseMatrix *W) {
	/*
	S          sparse matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	W          sparse matrix
	*/

	integer i, j, k, p = W->nc, *idxl, *idxposl, Sncolj, *Srowindj, Wncolj, *Wrowindj,
	                 mythreadnum, // individual thread number
	                 cnt;					  // counter for the number of nonzeros per column
	double *Svalj;

	// -----------------------------------------------------------
	#pragma omp parallel for shared(p, idx, idxpos, S, W) private(mythreadnum, idxl, idxposl, cnt, Sncolj, Srowindj, k, i, Wncolj, Wrowindj, Svalj)
	for (j = 0; j < p; j++) {

		// who am I
		mythreadnum = omp_get_thread_num();
		// #pragma omp critical
		// MSG("active thread %d\n",omp_get_thread_num());

		// local versions of idx, idxpos to avoid memory conflicts
		idxl = idx + p * mythreadnum;
		idxposl = idxpos + p * mythreadnum;

#ifdef PRINT_CHECK
		for (k = 0; k < p; k++)
			if (idxposl[k] != 0)
				MSG("AugmentSparse_OMP, thread %ld: idxposl[%ld]!=0\n", mythreadnum, k + 1);
#endif
		// initially set check marks for the existing pattern of S
		// counter for the number of nonzeros per column
		cnt = 0;
		// short cuts
		Sncolj = S->ncol[j];
		Srowindj = S->rowind[j];
		for (k = 0; k < Sncolj; k++) {
			// row index i of S_ij
			i = Srowindj[k];
			idxl[cnt++] = i;
			idxposl[i] = cnt;
		} // end for k

		// now check if W has any additional nonzeros
		// short cuts
		Wncolj = W->ncol[j];
		Wrowindj = W->rowind[j];
		for (k = 0; k < Wncolj; k++) {
			// index i of W_ij
			i = Wrowindj[k];
			// this is an additional fill-in for S
			if (!idxposl[i]) {
				idxl[cnt++] = i;
				// flag them negative to indicate fill-in
				idxposl[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		S->rowind[j] = (integer *)realloc(S->rowind[j], cnt * sizeof(integer));
		S->val[j] = (double *)realloc(S->val[j], cnt * sizeof(double));
		// (re-)new short cuts
		Srowindj = S->rowind[j];
		Svalj = S->val[j];

		// restore original number of nonzeros in S
		cnt = Sncolj;
		// check W again and only add the entries which were flagged negative
		for (k = 0; k < Wncolj; k++) {
			// index i of W_ij
			i = Wrowindj[k];
			// this is an additional fill-in for S
			if (idxposl[i] < 0) {
				Srowindj[cnt] = i;
				Svalj[cnt++] = 0.0;
			} // end if
		}	  // end for k
		// new number of nonzeros in S
		S->ncol[j] = cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idxl[k];
			idxposl[i] = 0;
		} // end for k
	}	  // end for j
	// end omp parallel for
	// -----------------------------------------------------------

	S->nnz = 0;
	for (j = 0; j < p; j++)
		S->nnz += S->ncol[j];
}; // end AugmentSparse_OMP

// Define D on the pattern of S and return the objective value of f(X)
double SQUIC::DiagNewton(const double Lambda, SparseMatrix *D) {

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
	       logdet,				  // log(det(X))
	       l1normX,			  // |\Lambda*X|_1
	       trSX,				  // trace(SX)
	       a, b, c, f, valw, mu; // parameters for optimization function

	p = S.nr;
	// new number of nonzeros for \Delta
	D->nnz = S.nnz;

	// init log(det(X))
	logdet = 0.0;
	// init |\Lambda*X|_1
	l1normX = 0.0;
	// init trace(SX)
	trSX = 0.0;
	for (j = 0; j < p; j++) {
		// W is diagonal
		Wjj = W.val[j][0];

		m = S.ncol[j];
		D->ncol[j] = m;
		D->rowind[j] = (integer *)malloc(m * sizeof(integer));
		D->val[j] = (double *)malloc(m * sizeof(double));
		// scan S_{:,j}
		for (k = 0; k < m; k++) {
			// row index i of column j
			i = S.rowind[j][k];
			Sij = S.val[j][k];
			D->rowind[j][k] = i;
			D->val[j][k] = 0.0;
			// off-diagonal case
			if (i != j) {
				if (fabs(Sij) > Lambda) {
					// W is diagonal
					Wii = W.val[i][0];
					// off-diagonal case a = W_ii W_jj
					a = Wii * Wjj;
					// off-diagonal case b = S_ij, W is diagonal => W_ij=0
					b = Sij;

					// parameters for optimization parameter mu = -s(-f,valw) (c=0?)
					// where s(z,r)=sign(z)*max{|z|-r,0}
					f = b / a;
					valw = Lambda / a;
					// S_ij<0
					if (0.0 > f) {
						// \mu=(|S_ij|-\Lambda)/a
						mu = -f - valw;
						// mu>0 <=> |Sij|>\Lambda
						// |S_ij|<\Lambda_ij?
						if (mu < 0.0) {
							mu = 0.0;
						}
					} else {
						// S_ij>0
						// \mu=(-|S_ij|+\Lambda)/a
						mu = -f + valw;
						// mu<0 <=> |Sij|>\Lambda
						// |S_ij|<\Lambda_ij?
						if (mu > 0.0) {
							mu = 0.0;
						}
					}
					// since S is exactly symmetric, the same update will take place
					// for \Delta_ji
					D->val[j][k] = mu;
				} // end if |Sij|>Lambda
			} else {
				// i=j
				// handle diagonal part for \Delta_jj
				// X is diagonal
				Xjj = fabs(X.val[j][0]);
				// update log(det(X))
				logdet += log(Xjj);
				// update |X|_1
				if (RunTimeConfig.off_diagonal)
					l1normX += Xjj;

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
			} // end if-else i!=j
		}	  // end while
	}		  // end for j
	// adjust |\Lambda X|_1
	l1normX *= Lambda;

	// objective value f(X) = -log(det(X)) + trace(SX) + |\Lambda*X|_1
	double fX = -logdet + trSX + l1normX;

	return fX;
}; // end DiagNewton

// check symmetric positive definite matrix A for off-diagonal contributions
// We assume that the diagonal part is always included.
// A matrix is considered to be non-diagonal if every column consists of more
// than one entry
integer SQUIC::IsDiag(const SparseMatrix *A) {
	for (integer i = 0; i < A->nr; i++)
		if (A->ncol[i] > 1)
			return (integer)0;

	return (integer)1;
};

// to set up S properly, its row indices should be sorted in increasing order
void SQUIC::SortSparse(SparseMatrix *S, integer *stack) {

	integer j, k, p = S->nc;

	// #pragma omp parallel for schedule(static,1) shared(p,S,mystack) private(k)
	// stack has to be of size p*omp_get_max_threads() such that each tread
	// gets its local chunk of size p, i.e. integer* my_stack=stack+p*omp_get_thread_num();
	for (j = 0; j < p; j++) {
		// start column j
		k = S->ncol[j];
		// sort S_{:,j} with quicksort such that the indices are taken in increasing order
		qsort1_(S->val[j], S->rowind[j], stack, &k);
	} // end for j
};

// to set up sparse matrices properly, its row indices should be sorted in increasing order
// OpenMP version
void SQUIC::SortSparse_OMP(SparseMatrix *S, integer *stack) {

	integer j, k, p = S->nc, mythreadnum, *stackl;

	#pragma omp parallel for shared(p, stack, S) private(mythreadnum, stackl, k)
	for (j = 0; j < p; j++) {
		// who am I
		mythreadnum = omp_get_thread_num();
		// local versions of stack to avoid memory conflicts
		stackl = stack + p * mythreadnum;

		// start column j
		k = S->ncol[j];
		// sort S_{:,j} with quicksort such that the indices are taken in increasing order
		qsort1_(S->val[j], S->rowind[j], stackl, &k);
	} // end for j
	// end omp parallel
};	  // end SortSparse_OMP

// Augment S with the pattern of W and generate the associated entries
//void SQUIC::AugmentS(double *mu, integer *idx, integer *idxpos, SparseMatrix *T) {
void SQUIC::AugmentS(double *mu, integer *idx, integer *idxpos, SparseMatrix *W) {

	//	return;

	//	cout << "Calling - AugmentS" << endl;

	/*
	Y          n samples Y=[y_1,...y_n] from which the low-rank empirical
	covariance matrix S=1/NM1(n) sum_i (y_i-mu)(y_i-mu)^T can be built, where
	mu=1/n sum_i y_i is the mean value
	Y is stored by columns
	p          size parameter space
	n          number of samples
	mu         mean value
	S          sparse representation of the sample covariance matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	W          W ~ X^{-1} sparse matrix
	*/

	integer i, j, k, r,
	        cnt; // counter for the number of nonzeros per column

	double *pY, // pointer to Y_{i,:}
	       Sij;

	S.nnz = 0;
	for (j = 0; j < p; j++) {

		// initially set check marks for the existing pattern of S
		// counter for the number of nonzeros per column
		cnt = 0;
		for (k = 0; k < S.ncol[j]; k++) {
			// index i of S_ij
			i = S.rowind[j][k];
			idx[cnt++] = i;
			idxpos[i] = cnt;
		} // end for k

		// now check if W has any additional nonzeros
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (!idxpos[i]) {
				idx[cnt++] = i;
				// flag them negative to indicate fill-in
				idxpos[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		S.rowind[j] = (integer *)realloc(S.rowind[j], cnt * sizeof(integer));
		S.val[j] = (double *)realloc(S.val[j], cnt * sizeof(double));

		// restore original number of nonzeros in S
		cnt = S.ncol[j];
		// check W again and only add the entries which were flagged negative
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (idxpos[i] < 0) {

				Sij = 0.0;
				for (r = 0, pY = Y; r < n; r++, pY += p) {
					Sij += (pY[i] - mu[i]) * (pY[j] - mu[j]);
				}
				S.rowind[j][cnt] = i;
				S.val[j][cnt++] = Sij / NM1(n);

			} // end if
		}	  // end for k
		// new number of nonzeros in S
		S.ncol[j] = cnt;
		S.nnz += cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idx[k];
			idxpos[i] = 0;
		} // end for k
	}	  // end for j
};

// Augment S with the pattern of W and generate the associated entries
void SQUIC::AugmentS_OMP(double *mu, integer *idx, integer *idxpos, SparseMatrix *W) {
	/*
	Y          n samples Y=[y_1,...y_n] from which the low-rank empirical
	           covariance matrix S=1/NM1(n) sum_i (y_i-mu)(y_i-mu)^T can be built, where
	           mu=1/n sum_i y_i is the mean value
	       Y is stored by columns
	p          size parameter space
	n          number of samples
	mu         mean value
	S          sparse representation of the sample covariance matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	W          W ~ X^{-1} sparse matrix
	*/

	integer i, j, k, r, *idxl, *idxposl,
	        mythreadnum, // individual thread number
	        cnt;		 // counter for the number of nonzeros per column

	double *pY, // pointer to Y_{i,:}
	       Sij;

	#pragma omp parallel for shared(p, idx, idxpos, S, W, Y, n, mu) private(mythreadnum, idxl, idxposl, cnt, k, i, Sij, r, pY)
	for (j = 0; j < p; j++) {

		// who am I
		mythreadnum = omp_get_thread_num();
		// local versions of idx, idxpos to avoid memory conflicts
		idxl = idx + p * mythreadnum;
		idxposl = idxpos + p * mythreadnum;

#ifdef PRINT_CHECK
		for (k = 0; k < p; k++)
			if (idxposl[k] != 0)
				MSG("AugmentS_OMP, thread %ld: idxposl[%ld]!=0\n", mythreadnum, k + 1);
#endif
		// initially set check marks for the existing pattern of S
		// counter for the number of nonzeros per column
		cnt = 0;
		for (k = 0; k < S.ncol[j]; k++) {
			// index i of S_ij
			i = S.rowind[j][k];
			idxl[cnt++] = i;
			idxposl[i] = cnt;
		} // end for k

		// now check if W has any additional nonzeros
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (!idxposl[i]) {
				idxl[cnt++] = i;
				// flag them negative to indicate fill-in
				idxposl[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		S.rowind[j] = (integer *)realloc(S.rowind[j], cnt * sizeof(integer));
		S.val[j] = (double *)realloc(S.val[j], cnt * sizeof(double));

		// restore original number of nonzeros in S
		cnt = S.ncol[j];
		// check W again and only add the entries which were flagged negative
		for (k = 0; k < W->ncol[j]; k++) {
			// index i of W_ij
			i = W->rowind[j][k];
			// this is an additional fill-in for S
			if (idxposl[i] < 0) {
				Sij = 0.0;
				for (r = 0, pY = Y; r < n; r++, pY += p)
					Sij += (pY[i] - mu[i]) * (pY[j] - mu[j]);
				S.rowind[j][cnt] = i;
				S.val[j][cnt++] = Sij / NM1(n);
			} // end if
		}	  // end for k
		// new number of nonzeros in S
		S.ncol[j] = cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idxl[k];
			idxposl[i] = 0;
		} // end for k
	}	  // end for j
	// end omp parallel for

	// finally count number of nonzeros
	S.nnz = 0;
	for (j = 0; j < p; j++)
		S.nnz += S.ncol[j];
}; // end AugmentS_OMP

// augment the pattern of \Delta with the pattern from the active set
void SQUIC::AugmentD(SparseMatrix *D, integer *idx, integer *idxpos, IndexPair *activeSet, integer numActive) {

	/*
	D          sparse symmetric update matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	activeSet  list of acitve pairs
	numActive  number of pairs
	*/

	integer i, j, k, l, p = D->nr,
	                    cnt; // counter for the number of nonzeros per column
	SparseMatrix Active;

	// convert Active set list into a sparse matrix in order to augment D efficiently
	Active.nr = Active.nc = p;
	Active.ncol = (integer *)calloc(p, sizeof(integer));
	Active.rowind = (integer **)malloc(p * sizeof(integer *));
	Active.val = NULL;

	// count nnz per column
	for (k = 0; k < numActive; k++) {
		i = activeSet[k].i;
		j = activeSet[k].j;
		Active.ncol[j]++;
		// take symmetry into account
		if (i != j) {
			Active.ncol[i]++;
		}
	} // end for k
	// allocate index memory
	Active.nnz = 0;
	for (j = 0; j < p; j++) {
		l = Active.ncol[j];
		Active.rowind[j] = (integer *)malloc(l * sizeof(integer));
		Active.nnz += l;
		// reset counter
		Active.ncol[j] = 0;
	} // end for j
	// insert row indices per column
	for (k = 0; k < numActive; k++) {
		i = activeSet[k].i;
		j = activeSet[k].j;
		l = Active.ncol[j];
		Active.rowind[j][l] = i;
		Active.ncol[j] = l + 1;
		// take symmetry into account
		if (i != j) {
			l = Active.ncol[i];
			Active.rowind[i][l] = j;
			Active.ncol[i] = l + 1;
		} // end if
	}	  // end for k

	// now we may supplement \Delta with the nonzeros from the active set

	D->nnz = 0;
	for (j = 0; j < p; j++) {

		// initially set check marks for the existing pattern of \Delta
		// counter for the number of nonzeros per column
		cnt = 0;
		for (k = 0; k < D->ncol[j]; k++) {
			// index i of D_ij
			i = D->rowind[j][k];
			// add i to the list of indices
			idx[cnt++] = i;
			// check index i as used
			idxpos[i] = cnt;
		} // end for k

		// now check if ActiveSet has any additional nonzeros
		for (k = 0; k < Active.ncol[j]; k++) {
			// index i of Active_ij
			i = Active.rowind[j][k];
			// this is an additional fill-in for \Delta
			if (!idxpos[i]) {
				idx[cnt++] = i;
				// flag them negative to indicate fill-in
				idxpos[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		D->rowind[j] = (integer *)realloc(D->rowind[j], cnt * sizeof(integer));
		D->val[j] = (double *)realloc(D->val[j], cnt * sizeof(double));

		// restore original number of nonzeros in \Delta
		cnt = D->ncol[j];
		// check Active again and only add the entries which were flagged negative
		for (k = 0; k < Active.ncol[j]; k++) {
			// index i of Active_ij
			i = Active.rowind[j][k];
			// this is an additional fill-in for \Delta
			if (idxpos[i] < 0) {
				D->rowind[j][cnt] = i;
				D->val[j][cnt++] = 0.0;
			} // end if
		}	  // end for k
		// new number of nonzeros in \Delta
		D->ncol[j] = cnt;
		// update new fill in column j
		D->nnz += cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idx[k];
			idxpos[i] = 0;
		} // end for k
	}	  // end for j

	// release index memory
	for (j = 0; j < p; j++) {
		free(Active.rowind[j]);
	}
	free(Active.ncol);
	free(Active.rowind);
}; // end AugmentD

// augment the pattern of \Delta with the pattern from the active set
void SQUIC::AugmentD_OMP(SparseMatrix *D, integer *idx, integer *idxpos, IndexPair *activeSet, integer numActive) {
	/*
	D          sparse symmetric update matrix
	idx   \    pre-allocated arrays to hold the list of indices and the check marks
	idxpos/    idxpos must be initialized with 0
	activeSet  list of acitve pairs
	numActive  number of pairs
	*/

	integer i, j, k, l, p = D->nr, *idxl, *idxposl,
	                    mythreadnum, // individual thread number
	                    cnt;						 // counter for the number of nonzeros per column
	SparseMatrix Active;

	// convert Active set list into a sparse matrix in order to augment D efficiently
	Active.nr = Active.nc = p;
	Active.ncol = (integer *)calloc(p, sizeof(integer));
	Active.rowind = (integer **)malloc(p * sizeof(integer *));
	Active.val = NULL;

	// count nnz per column
	for (k = 0; k < numActive; k++) {
		i = activeSet[k].i;
		j = activeSet[k].j;
		Active.ncol[j]++;
		// take symmetry into account
		if (i != j)
			Active.ncol[i]++;
	} // end for k
	// allocate index memory
	Active.nnz = 0;
	for (j = 0; j < p; j++) {
		l = Active.ncol[j];
		Active.rowind[j] = (integer *)malloc(l * sizeof(integer));
		Active.nnz += l;
		// reset counter
		Active.ncol[j] = 0;
	} // end for j
	// insert row indices per column
	for (k = 0; k < numActive; k++) {
		i = activeSet[k].i;
		j = activeSet[k].j;
		l = Active.ncol[j];
		Active.rowind[j][l] = i;
		Active.ncol[j] = l + 1;
		// take symmetry into account
		if (i != j) {
			l = Active.ncol[i];
			Active.rowind[i][l] = j;
			Active.ncol[i] = l + 1;
		} // end if
	}	  // end for k

	// now we may supplement \Delta with the nonzeros from the active set
	#pragma omp parallel for shared(p, idx, idxpos, D, Active) private(mythreadnum, idxl, idxposl, cnt, k, i)
	for (j = 0; j < p; j++) {

		// who am I
		mythreadnum = omp_get_thread_num();
		// local versions of idx, idxpos to avoid memory conflicts
		idxl = idx + p * mythreadnum;
		idxposl = idxpos + p * mythreadnum;

		// initially set check marks for the existing pattern of \Delta
		// counter for the number of nonzeros per column
		cnt = 0;
		for (k = 0; k < D->ncol[j]; k++) {
			// index i of D_ij
			i = D->rowind[j][k];
			// add i to the list of indices
			idxl[cnt++] = i;
			// check index i as used
			idxposl[i] = cnt;
		} // end for k

		// now check if ActiveSet has any additional nonzeros
		for (k = 0; k < Active.ncol[j]; k++) {
			// index i of Active_ij
			i = Active.rowind[j][k];
			// this is an additional fill-in for \Delta
			if (!idxposl[i]) {
				idxl[cnt++] = i;
				// flag them negative to indicate fill-in
				idxposl[i] = -cnt;
			} // end if
		}	  // end for k

		// re-allocate memory for column j
		D->rowind[j] = (integer *)realloc(D->rowind[j], cnt * sizeof(integer));
		D->val[j] = (double *)realloc(D->val[j], cnt * sizeof(double));

		// restore original number of nonzeros in \Delta
		cnt = D->ncol[j];
		// check Active again and only add the entries which were flagged negative
		for (k = 0; k < Active.ncol[j]; k++) {
			// index i of Active_ij
			i = Active.rowind[j][k];
			// this is an additional fill-in for \Delta
			if (idxposl[i] < 0) {
				D->rowind[j][cnt] = i;
				D->val[j][cnt++] = 0.0;
			} // end if
		}	  // end for k
		// new number of nonzeros in \Delta
		D->ncol[j] = cnt;

		// clear check marks
		for (k = 0; k < cnt; k++) {
			i = idxl[k];
			idxposl[i] = 0;
		} // end for k
	}	  // end for j
	// end omp parallel for

	// count final number of nonzeros
	D->nnz = 0;
	for (j = 0; j < p; j++)
		D->nnz += D->ncol[j];

	// release index memory
	for (j = 0; j < p; j++)
		free(Active.rowind[j]);
	free(Active.ncol);
	free(Active.rowind);
} // end AugmentD_OMP

void SQUIC::print(SparseMatrix *S, integer p) {

	integer percis = 4;
	integer flag = 0;
	integer space = 9;

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.precision(percis);

	for (integer i = 0; i < p; i++) {
		for (integer j = 0; j < p; ++j) {

			flag = 0;
			for (integer k = 0; k < S->ncol[j]; k++) {
				if (S->rowind[j][k] == i) {
					std::cout << std::right << std::setw(space) << S->val[j][k] << " ";
					flag = 1;
					break;
				};
			}

			if (!flag) {
				std::cout << std::right << std::setw(space) << "  0  "
				          << " ";
			};
		}
		std::cout << std::endl;
	}
}

void SQUIC::print_stats() {

	MSG("\n=================\nStats\n=================\n");
	fflush(stdout);

	int i, nn;
	double sum;

	MSG("dgap=%e\n\n", Stat.dgap);
	fflush(stdout);
	MSG("time_total=%e\n\n", Stat.time_total);
	fflush(stdout);
	MSG("time_cov=%e\n\n", Stat.time_cov);
	fflush(stdout);

	nn = Stat.opt.size();
	if (nn > 0) {
		MSG("objective_value=[");
		for (i = 0; i < nn - 1; ++i) {
			MSG("%e,", Stat.opt[i]);
		}
		MSG("%e]\n\n", Stat.opt[nn - 1]);
	} else
		MSG("objective_value=[]\n\n");
	fflush(stdout);

	nn = Stat.time_itr.size();
	sum = 0.0;
	if (nn > 0) {
		MSG("time_iteration=[");
		for (i = 0; i < nn - 1; ++i) {
			sum += Stat.time_itr[i];
			MSG("%e,", Stat.time_itr[i]);
		}
		sum += Stat.time_itr[nn - 1];
		MSG("%e]\n", Stat.time_itr[nn - 1]);
	} else
		MSG("time_iteration=[]\n");
	MSG("total_time_iteration=%e\n\n", sum);
	fflush(stdout);

	nn = Stat.time_chol.size();
	sum = 0.0;
	if (nn > 0) {
		MSG("time_cholesky_logDet=[");
		for (i = 0; i < nn - 1; ++i) {
			sum += Stat.time_chol[i];
			MSG("%e,", Stat.time_chol[i]);
		}
		sum += Stat.time_chol[nn - 1];
		MSG("%e]\n", Stat.time_chol[nn - 1]);
	} else
		MSG("time_cholesky_logDet=[]\n");
	MSG("total_time_cholesky_logDet=%e\n\n", sum);
	fflush(stdout);

	nn = Stat.time_inv.size();
	sum = 0.0;
	if (nn > 0) {
		MSG("time_matrix_inversion=[");
		for (i = 0; i < nn - 1; ++i) {
			sum += Stat.time_inv[i];
			MSG("%e,", Stat.time_inv[i]);
		}
		sum += Stat.time_inv[nn - 1];
		MSG("%e]\n", Stat.time_inv[nn - 1]);
	} else
		MSG("time_matrix_inversion=[]\n");
	MSG("total_time_matrix_inversion=%e\n\n", sum);
	fflush(stdout);

	nn = Stat.time_lns.size();
	sum = 0.0;
	if (nn > 0) {
		MSG("time_line_search=[");
		for (i = 0; i < nn - 1; ++i) {
			sum += Stat.time_lns[i];
			MSG("%e,", Stat.time_lns[i]);
		}
		sum += Stat.time_lns[nn - 1];
		MSG("%e]\n", Stat.time_lns[nn - 1]);
	} else
		MSG("time_line_search=[]\n");
	MSG("total_time_line_search=%e\n\n", sum);
	fflush(stdout);

	nn = Stat.time_upd.size();
	sum = 0.0;
	if (nn > 0) {
		MSG("time_coordinate_update=[");
		for (i = 0; i < nn - 1; ++i) {
			sum += Stat.time_upd[i];
			MSG("%e,", Stat.time_upd[i]);
		}
		sum += Stat.time_upd[nn - 1];
		MSG("%e]\n", Stat.time_upd[nn - 1]);
	} else
		MSG("time_coordinate_update=[]\n");
	MSG("total_time_coordinate_update=%e\n\n", sum);
	fflush(stdout);
}

// convert COO matrix to COO_CustomeData (CSC like ..)
void SQUIC::COO_to_CustomeData(integer *X_i, integer *X_j, double *X_val, integer nnz, SparseMatrix *X) {

	// Intialilize structure
	X->nr = X->nc = p;
	X->nnz = nnz;

	X->ncol = (integer *)calloc(p, sizeof(integer)); // set it equal to zero .. there might be zero element in a row
	X->rowind = (integer **)malloc(p * sizeof(integer *));
	X->val = (double **)malloc(p * sizeof(double));

	integer inx_i, inx_j, pos;
	double val;
	double *temp_buff = (double *)calloc(p, sizeof(double));

	// count number of columns in each row
	for (integer k = 0; k < nnz; ++k) {
		inx_i = X_i[k];
		X->ncol[inx_i]++;
		temp_buff[inx_i]++;
	}

	// allocate memoery according to the number of columns in each row
	for (integer i = 0; i < p; i++) {
		X->rowind[i] = (integer *)malloc(X->ncol[i] * sizeof(integer));
		X->val[i] = (double *)malloc(X->ncol[i] * sizeof(double));
	} // end for j

	// Go though the dataset once and allocate values.
	for (integer k = 0; k < nnz; ++k) {

		inx_i = X_i[k];
		inx_j = X_j[k];
		val = X_val[k];

		pos = X->ncol[inx_i] - temp_buff[inx_i];

		X->val[inx_i][pos] = val;
		X->rowind[inx_i][pos] = inx_j;

		temp_buff[inx_i]--;
	}
};

// convert CSC matrix to CustomeData (CSC like ..)
void SQUIC::CSC_to_CustomeData(integer *X_row_index, integer *X_col_ptr, double *X_val, integer nnz, SparseMatrix *X) {

	// Intialilize structure
	X->nr = X->nc = p;
	X->nnz = nnz;

	integer temp_i = 0;

	X->ncol = (integer *)calloc(p, sizeof(integer)); // set it equal to zero .. there might be zero element in a row
	X->rowind = (integer **)malloc(p * sizeof(integer *));
	X->val = (double **)malloc(p * sizeof(double));

	// allocate memoery according to the number of columns in each row
	for (integer i = 0; i < p; i++) {
		X->ncol[i] = X_col_ptr[i + 1] - X_col_ptr[i];
		X->rowind[i] = (integer *)malloc(X->ncol[i] * sizeof(integer));
		X->val[i] = (double *)malloc(X->ncol[i] * sizeof(double));

		// copy in the information
		for (integer j = 0; j < X->ncol[i]; j++) {
			X->rowind[i][j] = X_row_index[temp_i];
			X->val[i][j] = X_val[temp_i];
			temp_i++;
		}
	}
};

extern "C"
{
#include "squic_interface.hpp"
}

#endif
