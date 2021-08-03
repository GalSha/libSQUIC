/* cholesky_matrix.cpp
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

#ifndef cholesky_matrix_cpp
#define cholesky_matrix_cpp

#include "squic.hpp"

#define MAX(A, B) (((A) > (B)) ? (A) : (B))

// #define PRINT_INFO_EXPORT

// The computed W does not satisfy |W - S|< Lambda.  Project it.
// we assume that the nonzero pattern of S covers at least the one of W (or even more)
double SQUIC::projLogDet(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol)
{

	/*
	S      sparse sample covariance matrix with entries at least at the positions of W
	W      sparse covariance matrix
	Lambda Lagrangian parameter
	*/

	integer i, j, k, l, m, p = S.nr, nnz = W.nnz, cnt, info;
	double tmp, Wij, Sij, logdet;
	integer ierr;

	// Copy data to contiguous structure
	integer *ia = new integer[p + 1];
	integer *ja = new integer[nnz];
	double *a = new double[nnz];

	ia[0] = 0;
	cnt = 0;
	integer ii, kk, ll, *pi;
	double myLambda;
	for (j = 0; j < p; j++)
	{

		// logical pointer to column j+1
		ia[j + 1] = ia[j] + W.ncol[j];
		l = 0;
		// abbrevations for the current column of LambdaMatrix
		pi = LambdaMatrix->rowind[j];
		ll = LambdaMatrix->ncol[j];
		kk = 0;
		for (k = 0; k < W.ncol[j]; k++)
		{
			i = W.rowind[j][k];
			Wij = W.val[j][k];
			ja[cnt] = i;
			// scan S until S_ij is found, S contains the pattern of W
			// both matrices are set up such that their row indices are taken in increasing
			// order
			while (S.rowind[j][l] < i)
			{
				l++;
			}
			Sij = S.val[j][l];

			// check for a different value of Lambda rather than the default value
			myLambda = Lambda;
			for (; kk < ll; kk++)
			{
				ii = pi[kk];
				if (ii == i)
				{
					myLambda = LambdaMatrix->val[j][kk];
					break;
				}
				else if (ii >= i)
					break;
			} // end for kk

			tmp = Wij;
			if (RunTimeConfig.off_diagonal)
			{
				if (i == j)
				{
					tmp = Sij;
				}
				else
				{ // i!=j
					if (Sij - myLambda > tmp)
						tmp = Sij - myLambda;
					if (Sij + myLambda < tmp)
						tmp = Sij + myLambda;
				}
			} // end if RunTimeConfig.off_diagonal
			else
			{ // NOT RunTimeConfig.off_diagonal
				if (Sij - myLambda > tmp)
					tmp = Sij - myLambda;
				if (Sij + myLambda < tmp)
					tmp = Sij + myLambda;
			} // end if-else RunTimeConfig.off_diagonal

			// project prW_ij
			// since S and W are exactly symmetric, the same result will apply to prW_ji
			a[cnt++] = tmp;
		} // end for k
	}	  // end for j

	// --------------------------------------------------------------
	// compute log(det X) using an external function call from MATLAB
	// --------------------------------------------------------------
	// logdet    MATLAB function
	//           [ld,info]=logdet(A,drop_tol,params) returns log(det(A)) for a given sparse
	//           symmetric positive definite matrix A with up to some accuracy drop_tol
	//           using some internal parameters
	// A

	double *D_LDL = new double[p];
	integer err;

	// generic LDL driver in CSC format
	err = Factorize_LDL(ja, ia, a, W.nc, W.nr, W.nnz, NULL, NULL, D_LDL, drop_tol);
	// printf("err=%ld\n",err); fflush(stdout);


	logdet = 0.0;
	if (!err)
	{
			for (integer i = 0; i < p; ++i)
			{
				logdet += log(D_LDL[i]);
			}
	}
	else
	{
		logdet = 1e+15;
	}

	// --------------------------------------------------------------
	// compute log(det X) using an external function call from MATLAB
	// --------------------------------------------------------------

#ifdef PRINT_INFO
	printf("log(det(X))=%8.1le\n", logdet);
	fflush(stdout);
	printf("info=%8ld\n", info);
	fflush(stdout);
#endif

	//printf("logdet : %12.4le \n", logdet);

	delete[] ia;
	delete[] ja;
	delete[] a;

	delete[] D_LDL;

	return logdet;
}; // end projLogDet

#endif
