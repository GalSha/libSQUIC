/* squic_matrix.cpp
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

#ifndef squic_matrix_cpp
#define squic_matrix_cpp

// METIS
#include "metis.h"
#include "squic.hpp"

using namespace std;

// randomized version, entry point without initial guesses (warm start)
void SQUIC::Run(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol,
                int maxIter, double term_tol) {
  RunCore(Lambda, LambdaMatrix, drop_tol, maxIter, term_tol);
}  // end SQUIC::run

// randomized version, entry point WITH initial guesses
void SQUIC::Run(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol,
                int maxIter, double term_tol, SparseMatrix *X0,
                SparseMatrix *W0) {
  FreeSparse(&X);
  if (X0 == NULL)
    InitSparseMatrix(&X, 1.0);
  else
    CopySparse(&X, X0);
  FreeSparse(&W);
  if (W0 == NULL)
    InitSparseMatrix(&W, 1.0);
  else
    CopySparse(&W, W0);
  RunCore(Lambda, LambdaMatrix, drop_tol, maxIter, term_tol);
}  // end SQUIC::run

// core randomized active set routine after optionally providing initial guesses
// for X and W
void SQUIC::RunCore(double Lambda, SparseMatrix *LambdaMatrix, double drop_tol,
                    int maxIter, double term_tol) {
  // init runtime variables
  InitVars();
  // Statistics for output
  int nn;
  Stat.dgap = -1;  // size of 1
  Stat.time_total = -omp_get_wtime();

  if (RunTimeConfig.verbose > 0) {
    MSG("\n");
    MSG("----------------------------------------------------------------\n");
    MSG("                     SQUIC Version %.2f                         \n",
        SQUIC_VER);
    MSG("----------------------------------------------------------------\n");
    MSG("Input Matrices\n");
    MSG(" nnz(X0)/p:   %e\n", double(X.nnz) / double(p));
    MSG(" nnz(W0)/p:   %e\n", double(W.nnz) / double(p));
    MSG(" nnz(M)/p:    %e\n", double(LambdaMatrix->nnz) / double(p));
    MSG(" Y:         %d x %d \n", p, n);
    MSG("Runtime Configs   \n");
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
    MSG("#SQUIC Version %.2f : p=%d n=%d lambda=%.2e nnz(M)/p=%.2e max_iter=%d "
        "term_tol=%.2e drop_tol=%.2e",
        SQUIC_VER, integer(p), integer(n), double(Lambda),
        double(LambdaMatrix->nnz) / double(p), integer(maxIter),
        double(term_tol), double(drop_tol));
    fflush(stdout);
  }

  double timeBegin = omp_get_wtime();
  double D_LDL[p];
  // permutation vector for reordering
  integer perm[p];
  // diagonal scaling matrix
  double SL[p];

  integer maxNewtonIter = maxIter;
  double coord_dec_sweep_tol = 0.05;
  integer max_lineiter = LINE_SEARCH_ITER_MAX;
  double fX = 1e+15;
  double fX_update = 1e+15;
  double fX_last = 1e+15, subgrad;
  double sigma = 0.001;
  integer i, j, k, l, m, r, s, t, max_loop = MAX_LOOP, SX, flagld = 0;
  integer *idx, *idxpos;
  double *val, Wij, Sij, Xij, Dij, g;
  double temp_double;

  // tolerance used to compute log(det(X)) and X^{-1} approximately
  double current_drop_tol = MAX(DROP_TOL0, drop_tol);
  double *W_a;

  // parallel version requires omp_get_max_threads() local versions of idx,
  // idxpos to avoid memory conflicts between the threads
  k = omp_get_max_threads();
  // #pragma omp critical
  // printf("maximum number of available threads %ld\n",k);
  // used as stack and list of indices
  idx = (integer *)malloc((size_t)k * p * sizeof(integer));

  // used as check mark array for nonzero entries, must be initialized with 0
  idxpos = (integer *)calloc((size_t)k * p, sizeof(integer));

  // used as list of nonzero values, must be initialized with 0.0
  val = (double *)malloc((size_t)k * p * sizeof(double));
  for (j = 0; j < k * p; j++) val[j] = 0.0;

  integer ierr;
  double *pr;

  // empty container for sparse update matrix \Delta
  SparseMatrix D;
  InitSparseMatrix(&D, 0.0);

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
  // Compute sample covariance matrix
  //////////////////////////////////////////////////
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
  printf("use %d threads\n\n", omp_get_max_threads());
  printf("Compute initial S...\n");
  fflush(stdout);
#endif

  // Compute Sparse Sample Covariance Matrix
  // generate at least the diagonal part of S and those S_ij s.t. |S_ij|>Lambda
  // (RunTimeConfig.generate_scm==DETERMINISTIC)
  GenerateSXL3B_OMP(mu, Lambda, LambdaMatrix, nu, ind, &idx, idxpos);
  SX = -1;

#ifdef PRINT_MSG
  printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginS));
  fflush(stdout);
#endif

#ifdef PRINT_CHECK
  SortSparse(&S, idx);
  ierr = CheckSymmetry(&S);
  if (ierr) {
    printf("!!!SQUIC(genSparseCov): S is nonsymmetric!!!\n");
    fflush(stdout);
  }
#endif
#ifdef PRINT_MSG
  double timeBeginAS = omp_get_wtime();
  printf("Augment S with the pattern of X...\n");
  fflush(stdout);
#endif
  // AugmentS(mu, idx, idxpos, &X);
  AugmentS_OMP(mu, idx, idxpos, &X);
  // sort nonzero entries in each column of S in increasing order
  // Remark: it suffices to sort S once after GenerateS and AugmentS
  // SortSparse(&S, idx);
  SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
  printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
  fflush(stdout);
#endif

#ifdef PRINT_CHECK
  ierr = CheckSymmetry(&S);
  if (ierr) {
    printf("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
    fflush(stdout);
  }
#endif

  Stat.time_cov += omp_get_wtime();

  if (RunTimeConfig.verbose > 0) {
    printf("* compute sample covariance matrix S: nnz(S)/p=%.2e time=%.2e\n",
           double(S.nnz) / double(p), Stat.time_cov);
    fflush(stdout);
  }
  //////////////////////////////////////////////////
  // END computation of the sample covariance matrix
  //////////////////////////////////////////////////

  // initially use active Set of tentative size 2p
  integer nactiveSet = 2 * p;
  index_pair *activeSet =
      (index_pair *)malloc((size_t)nactiveSet * sizeof(index_pair));
  double *activeSetweight2 =
      (double *)malloc((size_t)nactiveSet * sizeof(double));

  // |\Lambda*X|_1
  double l1norm_X = 0.0;

  // trace(SX)
  double trSX = 0.0;

  // log(det(X))
  double logdetX = 0.0;

  // scan matrices S and X
  integer ii, kk, ll, *pi;
  double myLambda;
  for (j = 0; j < p; j++) {
    // counters for X_{:,j}, S_{:,j}
    k = l = 0;
    // abbrevations for the current column of LambdaMatrix
    pi = LambdaMatrix->rowind[j];
    ll = LambdaMatrix->ncol[j];
    kk = 0;
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

      // check for a different value of Lambda rather than the default value
      myLambda = Lambda;
      for (; kk < ll; kk++) {
        ii = pi[kk];
        if (ii == i) {
          myLambda = LambdaMatrix->val[j][kk];
          break;
        } else if (ii >= i)
          break;
      }  // end for kk

      // increment |X|_1
      if (RunTimeConfig.off_diagonal) {
        if (i != j) l1norm_X += myLambda * fabs(Xij);
      }  // end if
      else
        l1norm_X += myLambda * fabs(Xij);

      // increment trace(SX)
      trSX += Sij * Xij;

    }  // end while
  }    // end for j

  // number of active set elements
  integer numActive = 0;

  // counter for the number of computed path steps < pathLen
  integer pathIdx = 0;

  //////////////////////////////////////////////////////////////////////////////////
  // Newton Iteration
  // outer Newton iteration loop, at most maxIter iteration steps unless
  // convergence
  //////////////////////////////////////////////////////////////////////////////////

  // counter for the number of Newton iteration steps <= maxIter
  integer NewtonIter = 1;

  // outer Newton iteration loop, at most maxIter iteration steps unless
  // convergence
  for (; NewtonIter <= maxNewtonIter; NewtonIter++) {
    // printf("!!! current tolerance: %8.1le!!!\n", current_drop_tol);
    // fflush(stdout);

    if (RunTimeConfig.verbose > 1) {
      MSG("\n");
    }

    Stat.time_itr.push_back(-omp_get_wtime());

    current_iter++;

    double iterTime = omp_get_wtime();

    // |\Delta|_1
    double l1norm_D = 0.0;
    // |(\mu e_ie_j^T)_{i,j}|_1
    double diffD = 0.0;
    // tentative norm ~ |grad g|
    subgrad = 1e+15;

    // initial step and diagonal initial inverse covariance matrix X
    if (NewtonIter == 1 && IsDiag(&X)) {
      // Define \Delta on the pattern of S and return the objective value of
      // f(X) we assume that X AND its selected inverse W are diagonal we point
      // out that the initial S covers at least the diagonal part and those S_ij
      // s.t. |S_ij|>\Lambda
#ifdef PRINT_MSG
      double timeBeginDN = omp_get_wtime();
      printf("Diagonal Newton step...\n");
      fflush(stdout);
#endif

      temp_double = -omp_get_wtime();

      fX = DiagNewton(Lambda, LambdaMatrix, &D);

#ifdef PRINT_MSG
      printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginDN));
      fflush(stdout);
#endif

      temp_double += omp_get_wtime();
      if (RunTimeConfig.verbose > 1) {
        MSG("* Diagonal Newton Step: time=%e\n", temp_double);
        fflush(stdout);
      }
    } else {
      // Compute the active set and the minimum norm subgradient:
      numActive = 0;

      // update matrix \Delta=0
      ClearSparse(&D);

      // augment S by the pattern of W
#ifdef PRINT_MSG
      timeBeginAS = omp_get_wtime();
      printf("Augment S with the pattern of W...\n");
      fflush(stdout);
#endif
      // AugmentS(mu, idx, idxpos, &W);
      AugmentS_OMP(mu, idx, idxpos, &W);
      // SortSparse(&S, idx);
      SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
      printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
      fflush(stdout);
#endif
#ifdef PRINT_CHECK
      ierr = CheckSymmetry(&S);

      if (ierr) {
        printf("!!!SQUIC(Augment S with pattern of W): S is nonsymmetric!!!\n");
        fflush(stdout);
      }
#endif

      // if (RunTimeConfig.verbose > 1)
      //{
      //	MSG("+ Compute Active Set \n");
      //	fflush(stdout);
      // }
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

      // compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>Lambda
      // To do so, scan S, X and W
      for (j = 0; j < p; j++) {
        // counters for S_{:,j}, X_{:,j}, W_{:,j}
        k = l = m = 0;
        // abbrevations for the current column of LambdaMatrix
        pi = LambdaMatrix->rowind[j];
        ll = LambdaMatrix->ncol[j];
        kk = 0;
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

            // g_lb = fabs(fabs(Sij) - fabs(Wij));

            // check for a different value of Lambda rather than the default
            // value
            myLambda = Lambda;
            for (; kk < ll; kk++) {
              ii = pi[kk];
              if (ii == i) {
                myLambda = LambdaMatrix->val[j][kk];
                break;
              } else if (ii >= i)
                break;
            }  // end for kk

            // X_ij!=0 or |S_ij-W_ij|>Lambda
            // if (Xij != 0.0 || fabs(g) > Lambda) {
            if (fabs(Xij) > EPS || fabs(g) > myLambda) {
              // augment I_free
              if (numActive >= nactiveSet) {
                nactiveSet += p;
                activeSet = (index_pair *)realloc(
                    activeSet, nactiveSet * sizeof(index_pair));
                activeSetweight2 = (double *)realloc(
                    activeSetweight2, nactiveSet * sizeof(double));
              }

              activeSet[numActive].i = i;
              activeSet[numActive].j = j;

              // update |grad g|_1
              if (RunTimeConfig.off_diagonal) {
                if (i != j) {
                  if (Xij > 0.0)
                    g += myLambda;
                  else if (Xij < 0.0)
                    g -= myLambda;
                  else
                    g = fabs(g) - myLambda;
                }
              }  // end if
              else {
                // NOT RunTimeConfig.off_diagonal
                if (Xij > 0.0)
                  g += myLambda;
                else if (Xij < 0.0)
                  g -= myLambda;
                else
                  g = fabs(g) - myLambda;
              }  // end if-else RunTimeConfig.off_diagonal

              subgrad += fabs(g);

              // use sub gradient contribution as alternative weight
              activeSetweight2[numActive] = fabs(g);
              numActive++;
            }  // end if  Xij!=0 or |g|>Lambda
          }    // end if i>=j
        }      // end while
      }        // end for j

      if (RunTimeConfig.verbose > 1) {
        MSG("+ Active Set: size=%ld time=%e\n", numActive,
            omp_get_wtime() - temp_double);
        fflush(stdout);
      }
      ///////////////////////////////////////////////////////////////////
      // END Compute Active Set
      // compute active set I_free whenever X_ij!=0 or |S_ij-W_ij|>Lambda
      ///////////////////////////////////////////////////////////////////

      // augment the pattern of \Delta with the pattern from the active set
#ifdef PRINT_MSG
      double timeBeginAD = omp_get_wtime();
      printf("Augment D...\n");
      fflush(stdout);
#endif
      // AugmentD(&D, idx, idxpos, activeSet, numActive);
      AugmentD_OMP(&D, idx, idxpos, activeSet, numActive);
      // SortSparse(&D, idx);
      SortSparse_OMP(&D, idx);
#ifdef PRINT_MSG
      printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAD));
      fflush(stdout);
#endif
#ifdef PRINT_CHECK
      ierr = CheckSymmetry(&D);
      if (ierr) {
        printf("!!!SQUIC(Augment D): D is nonsymmetric!!!\n");
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

      }  // end if flagld
      // ------------------------------------------------------------------------
      // END compute refined W=inv(X) using an external function call from
      // MATLAB
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
          printf("Add (%2ld,%2ld) %ld to active set\n", k1, k2);
#endif
        }

        // update \Delta_ij  where
        // \Delta' differs from \Delta only in positions (i,j), (j,i)
        // l1norm_D , diffD will be updated
#ifdef PRINT_MSG
        double timeBeginCDU = omp_get_wtime();
        // (RunTimeConfig.block_upd == COLLECTIVE_CDU)
        printf("Collective coordinate descent update...\n");
        fflush(stdout);
#endif

        // (RunTimeConfig.block_upd == COLLECTIVE_CDU)
        BlockCoordinateDescentUpdate(Lambda, LambdaMatrix, &D, idx, idxpos, val,
                                     activeSet, 0, numActive - 1, l1norm_D,
                                     diffD);

#ifdef PRINT_MSG
        printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginCDU));
        fflush(stdout);
#endif

        if (diffD <= l1norm_D * coord_dec_sweep_tol) {
          break;
        }
      }

#ifdef PRINT_INFOX
      printf("D:\n");
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
    }

    ////////////////////////////////////////
    // Factorize log-det for fX == 1e+15
    ////////////////////////////////////////
    if (fX == 1e+15) {
      // (RunTimeConfig.inversion == BLOCK_FSPAI)
      if (RunTimeConfig.verbose > 1) {
        //	printf("Factorization [nnz(L_last)/p]: %8.1le\n", double(BL.nnz
        //+ BiD.nnz / 2.0) / double(p)); 	fflush(stdout);
      }

#ifdef PRINT_MSG
      double timeBeginLD = omp_get_wtime();

      printf("log(det(X))...\n");
      fflush(stdout);
#endif

      Stat.time_chol.push_back(-omp_get_wtime());
      temp_double = omp_get_wtime();

      // ierr = LogDet(D_LDL, &logdetX, current_drop_tol);
      ierr = LogDet(&logdetX, current_drop_tol);

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
            MSG("- Cholesky successful: time=%e nnz(L)/p=%e logdetX=%e\n",
                Stat.time_chol[nn - 1],
                double(Cholmod.common_parameters.lnz) / p, logdetX);
            fflush(stdout);
          }
        }
      }
      // ----------------------
      // END compute log(det X)
      // ----------------------

#ifdef PRINT_MSG
      printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginLD));
      fflush(stdout);
#endif
      // printf("LOGDET\n");
      // fflush(stdout);
      // printf("nnz(X)/p=%8.1le va=%lf\n", double(X.nnz) / double(p), logdetX);
      // fflush(stdout);
      // printf("-------------------------------\n");
      // fflush(stdout);

#ifdef PRINT_INFO
      printf("ierr=%8ld\n", ierr);
      fflush(stdout);
#endif
      if (ierr != 0) {
        printf("\n Error! Lack of positive definiteness!");
        //	iter[0] = -1;
        free(activeSet);
        free(activeSetweight2);
        DeleteVars();
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
      printf("log(det(X))=%8.1le\n", logdetX);
      fflush(stdout);
#endif

      // update objective function f(x)=-log(det(X))+trace(SX)+|\Lambda*X|_1
      fX = (trSX + l1norm_X) - logdetX;
    }  // end if (fX == 1e+15)
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

      }  // end while
    }    // end for j
#ifdef PRINT_INFO
    printf("trace((S-X^{-1})Delta)=%8.1le\n", trgradgD);
    fflush(stdout);
#endif

    // line search; we overwrite X by X + \alpha \Delta
    // if necessary, we downdate X back
    double alpha = 1.0;
    double l1norm_XD = 0.0;
    double fX1prev = 1e+15;
    // augment X with the pattern of \Delta

#ifdef PRINT_MSG
    double timeBeginSP = omp_get_wtime();
    printf("Augment X with pattern of Delta...\n");
    fflush(stdout);
#endif
    // AugmentSparse(&X, idx, idxpos, &D);
    AugmentSparse_OMP(&X, idx, idxpos, &D);
    // SortSparse(&X, idx);
    SortSparse_OMP(&X, idx);
#ifdef PRINT_MSG
    printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginSP));
    fflush(stdout);
#endif
#ifdef PRINT_CHECK
    ierr = CheckSymmetry(X);
    if (ierr) {
      printf(
          "!!!SQUIC(Augment X with pattern of Delta): X is nonsymmetric!!!\n");
      fflush(stdout);
    }
#endif
    // printf("augmented X:\n"); PrintSparse(X);

#ifdef PRINT_MSG
    timeBeginAS = omp_get_wtime();
    printf("Augment S with the pattern of X...\n");
    fflush(stdout);
#endif
    // AugmentS(mu, idx, idxpos, &X);
    AugmentS_OMP(mu, idx, idxpos, &X);
    // SortSparse(&S, idx);
    SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
    printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
    fflush(stdout);
#endif
#ifdef PRINT_CHECK
    ierr = CheckSymmetry(&S);
    if (ierr) {
      printf("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
      fflush(stdout);
    }
#endif

    // printf("-------------------------------\n");
    // fflush(stdout);
    // printf("Log Det\n");
    // fflush(stdout);
    // printf("nnz(X)/p=%8.1le\n", double(X.nnz) / double(p));
    // fflush(stdout);

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
      double l1norm_X_update = 0.0;
      double tr_SX_update = 0.0;
      double logdet_X_update = 0.0;

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
        // abbrevations for the current column of LambdaMatrix
        pi = LambdaMatrix->rowind[j];
        ll = LambdaMatrix->ncol[j];
        kk = 0;
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

          // check for a different value of Lambda rather than the default value
          myLambda = Lambda;
          for (; kk < ll; kk++) {
            ii = pi[kk];
            if (ii == i) {
              myLambda = LambdaMatrix->val[j][kk];
              break;
            } else if (ii >= i)
              break;
          }  // end for kk

          // |X|_1
          if (RunTimeConfig.off_diagonal) {
            if (i != j) l1norm_X_update += myLambda * fabs(Xij);
          }  // end if
          else
            l1norm_X_update += myLambda * fabs(Xij);

          // trace(SX)
          tr_SX_update += Sij * Xij;
        }  // end while
      }    // end for j

      ////////////////////////////////////////
      // Compute Cholesky+log(det X)
      ////////////////////////////////////////

      Stat.time_chol.push_back(-omp_get_wtime());
      temp_double = omp_get_wtime();
#ifdef PRINT_MSG
      double timeBeginLD = omp_get_wtime();
      printf("log(det(X))...\n");
      fflush(stdout);
#endif

      // ierr = LogDet(D_LDL, &logdet_X_update, loc_current_drop_tol);
      ierr = LogDet(&logdet_X_update, loc_current_drop_tol);

      nn = Stat.time_chol.size();
      if (nn > 0) {
        Stat.time_chol[nn - 1] += omp_get_wtime();

        if (ierr) {
          if (RunTimeConfig.verbose > 1) {
            MSG("- Factorization failed: iteration=%li time=%e alpha=%e\n",
                lineiter, Stat.time_chol[nn - 1], alpha);
            fflush(stdout);
          }
        } else {
          if (RunTimeConfig.verbose > 1) {
            // (RunTimeConfig.factorization==CHOL_SUITESPARSE)
            MSG("- Factorization successful: iteration=%li time=%e alpha=%e "
                "nnz(L)/p=%e logdetX=%e\n",
                lineiter, Stat.time_chol[nn - 1], alpha,
                double(Cholmod.common_parameters.lnz) / p, logdetX);
            fflush(stdout);
          }
        }
      }

#ifdef PRINT_MSG
      printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginLD));
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
          }  // end while
        }    // end for j

        // try line search with alpha/2 instead
        // alpha *= 0.5;  <---- do this at the very top of the iteration
        continue;
      } else {
        flagld = -1;
      }

      // update objective function f(X_1)=-log(det(X_1))+trace(SX_1)+|\Lambda
      // X_1|_1
      fX_update = (tr_SX_update + l1norm_X_update) - logdet_X_update;
      if (alpha == 1.0) {
        l1norm_XD = l1norm_X_update;
      }

      // line search successful, accept and break
      if (fX_update <= fX + alpha * SIGMA * (trgradgD + l1norm_XD - l1norm_X) ||
          l1norm_D == 0) {
        fX_last = fX;
        fX = fX_update;
        l1norm_X = l1norm_X_update;
        logdetX = logdet_X_update;
        trSX = tr_SX_update;

        accept_update = 1;
        continue;
      }

      // line search successful, accept and break
      // AE Edit - if (fX1prev < fX1) {
      if (fX_last < fX_update) {
        fX_last = fX;
        l1norm_X = l1norm_X_update;
        logdetX = logdet_X_update;
        trSX = tr_SX_update;

        accept_update = 1;
        continue;
      }

      fX1prev = fX_update;

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
        }  // end while
      }    // end for j

      // next try line search with alpha/2
      // alpha *= 0.5;  <---- do this at the very top of the iteration
    }  // end for lineiter

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
        Stat.time_itr[Stat.time_itr.size() - 1] += omp_get_wtime();
      }
      MSG("# Cholesky Failed: Exiting at current iterate. \n");

      break;
    }

    if (RunTimeConfig.verbose > 1) {
      MSG("+ Objective value decreased by = %e.\n", fX_last - fX);
      MSG("+ fX_last = %e.\n", fX_last);
      MSG("+ fX = %e.\n", fX);
      MSG("+ fabs((fX-fX_last) / fX = %e.\n", fabs((fX - fX_last) / fX));
    }
    // objlist.push_back(fX);

    ///////////////////////////////////////////
    /// export triangular factorization
    ///////////////////////////////////////////
#ifdef PRINT_MSG
    integer timeBeginEX = omp_get_wtime();
    printf("export factorization...\n");
    fflush(stdout);
#endif

    // export L,D to sparse matrix format along with the permutation
    // at the same time discard the Cholesky factorization
    // (RunTimeConfig.inversion == BLOCK_FSPAI)
    BlockExportLdl();
    L_nnz_per_row = double(BL.nnz + BiD.nnz / 2.0) / double(p);
    // MSG("+ nnz(L)/p=%8.1le\n", double(BL.nnz + BiD.nnz / 2.0) / double(p));
    // fflush(stdout);

    // printf("-------------------------------\n");
    // fflush(stdout);

    /*
    printf("L:\n");
    PrintSparse(&L);
    printf("1/D_LDL:\n");
    for (integer ii=0; ii<p; ii++)
        printf("%12.4le",1.0/D_LDL[ii]);
    printf("\n\n");
    */

    // next time we call Cholesky the pattern may change
    analyze_done = 0;

#ifdef PRINT_MSG
    printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginEX));
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
    // printf("-------------------------------\n");
    // fflush(stdout);
    // printf("SELINV/NINV\n");
    // fflush(stdout);
    // printf("nnz(X)/p=%8.1le\n", double(X.nnz) / double(p));
    // fflush(stdout);

#ifdef PRINT_MSG
    double timeBeginSI = omp_get_wtime();
#endif

    // compute W ~ X^{-1} using (block incomplete) Cholesky decomposition
    // (RunTimeConfig.inversion == BLOCK_FSPAI)
    BlockNeumannInv(current_drop_tol, NULL);

    // PrintSparse(&W);

#ifdef PRINT_MSG
    printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginSI));
    fflush(stdout);
#endif
    // printf("nnz(W)/p=%8.1le\n", double(W.nnz) / double(p));
    // fflush(stdout);
    // printf("-------------------------------\n");
    // fflush(stdout);

    // obsolete
    // // SortSparse(&W, idx);
    // SortSparse_OMP(&W, idx);

    nn = Stat.time_inv.size();
    if (nn > 0) {
      Stat.time_inv[nn - 1] += omp_get_wtime();
      if (RunTimeConfig.verbose > 1) {
        MSG("+ Approximate Matrix Inversion: time=%e nnz(W)/p=%8.1le \n",
            Stat.time_inv[nn - 1], double(W.nnz) / double(p));
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
                    std::cout << std::right << std::setw(space) << D_LDL[i] << "
    "; std::cout << std::endl; MSG("]\n"); fflush(stdout);

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

    nn = Stat.time_itr.size();
    if (nn > 0) {
      Stat.time_itr[nn - 1] += omp_get_wtime();

      if (RunTimeConfig.verbose > 0) {
        MSG("* iter=%d time=%.2e obj=%.2e |delta(obj)|/obj=%.2e "
            "nnz(X,L,W)/p=[%.2e %.2e %.2e] lns_iter=%d \n",
            NewtonIter, Stat.time_itr[nn - 1], fX, fabs((fX - fX_last) / fX),
            double(X.nnz) / double(p), L_nnz_per_row, double(W.nnz) / double(p),
            lineiter);
        fflush(stdout);
      }
    } else {
      if (RunTimeConfig.verbose > 0) {
        MSG("* iter=%d obj=%.2e |delta(obj)|/obj=%.2e nnz(X,L,W)/p=[%.2e %.2e "
            "%.2e] lns_iter=%d \n",
            NewtonIter, fX, fabs((fX - fX_last) / fX),
            double(X.nnz) / double(p), L_nnz_per_row, double(W.nnz) / double(p),
            lineiter);
        fflush(stdout);
      }
    }
    Stat.opt.push_back(fX);

    // if (RunTimeConfig.verbose)
    //{
    //	MSG("=================\n");
    //	fflush(stdout);
    // }

    // printf("++++++++++++++++++++++++++++++++++++\n");

    // Check for convergence.

    // not yet converged
    // we use at least the relative residual between two consecutive calls
    // + some upper/lower bound
    current_drop_tol = DROP_TOL_GAP * fabs((fX - fX_last) / fX);
    current_drop_tol = MAX(current_drop_tol, drop_tol);
    current_drop_tol = MIN(MAX(DROP_TOL0, drop_tol), current_drop_tol);
    // if (subgrad * alpha >= l1norm_X * term_tol && (fabs((fX - fX_last) / fX)
    // >= EPS))
    if (fabs((fX - fX_last) / fX) >= term_tol) {
      continue;
    }

    break;
  }  // end for(; NewtonIter < maxNewtonIter; .....

  opt = fX;

  // The computed W does not satisfy |W - S|< Lambda.  Project it.
  // now the meaning of U changes to case (c) and U is used as a buffer
  // only, since we are interested in log(det(project(W))), which in turn
  // requires its Cholesky factorization
  // compute U<-projected(W)
#ifdef PRINT_MSG
  timeBeginAS = omp_get_wtime();
  printf("augment S with the pattern of refined W...\n");
  fflush(stdout);
#endif
  //////////////////////////////////////////////////////////////////////////////////
  // END Newton Iteration
  // outer Newton iteration loop, at most maxIter iteration steps unless
  // convergence
  //////////////////////////////////////////////////////////////////////////////////

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
  }  // end if flagld
  // ------------------------------------------------------------------------
  // END compute refined W=inv(X) using an external function call from MATLAB
  // ------------------------------------------------------------------------

  // AugmentS(mu, idx, idxpos, &W);
  AugmentS_OMP(mu, idx, idxpos, &W);
  // SortSparse(&S, idx);
  SortSparse_OMP(&S, idx);
#ifdef PRINT_MSG
  printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginAS));
  fflush(stdout);
#endif
#ifdef PRINT_CHECK
  ierr = CheckSymmetry(&S);
  if (ierr) {
    printf("!!!SQUIC(Augment S with pattern of X): S is nonsymmetric!!!\n");
    fflush(stdout);
  }
#endif
#ifdef PRINT_MSG
  double timeBeginPLD = omp_get_wtime();
  printf("projected Log Det\n");
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
  double logdetW = ProjLogDet(Lambda, LambdaMatrix, current_drop_tol);

  nn = Stat.time_chol.size();
  if (nn > 0) Stat.time_chol[nn - 1] += omp_get_wtime();

  if (RunTimeConfig.verbose == 4) {
    //(RunTimeConfig.factorization==CHOL_SUITESPARSE)
    MSG("* project log det: time=%.2e nnz(X,L,W)/p=[%0.2e %0.2e %0.2e] \n",
        (omp_get_wtime() - temp_double), double(X.nnz) / double(p),
        double(Cholmod.common_parameters.lnz) / double(p),
        double(W.nnz) / double(p)

    );
    fflush(stdout);
  }
  //////////////////////////////////////////////////////////////////////////////////
  // END Project Log Det
  //////////////////////////////////////////////////////////////////////////////////

  Stat.time_total += omp_get_wtime();

  if (RunTimeConfig.verbose > 0) {
    MSG("#SQUIC Finished: time=%0.2e nnz(X,W)/p=[%0.2e %0.2e]\n\n",
        Stat.time_total, double(X.nnz) / double(p), double(W.nnz) / double(p));
    fflush(stdout);
  }
  if (RunTimeConfig.verbose == 0) {
    MSG("time=%0.2e nnz(X,W)/p=[%0.2e %0.2e]\n", Stat.time_total,
        double(X.nnz) / double(p), double(W.nnz) / double(p));
    fflush(stdout);
  }

#ifdef PRINT_MSG
  printf("...done %8.1le [sec]\n", (omp_get_wtime() - timeBeginPLD));
  fflush(stdout);
#endif

  double gap = -logdetW - p - logdetX + trSX + l1norm_X;
  dGap = gap;

  Stat.dgap = gap;
  Stat.trSX = trSX;
  Stat.logdetX = logdetX;

  free(activeSet);
  free(activeSetweight2);

  // DeleteVars();    // you have to call this manually!!!!!!
  //	FreeSparse(&S);
  FreeSparse(&D);

  free(mu);
  free(nu);
  free(ind);
  free(idx);
  free(idxpos);
  free(val);

  // if (RunTimeConfig.verbose)
  //	{
  //	print_stats();
  //	fflush(stdout);
  // }

  return;
};  // end run

// Define D on the pattern of S and return the objective value of f(X)
double SQUIC::DiagNewton(const double Lambda, SparseMatrix *LambdaMatrix,
                         SparseMatrix *D) {
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
  D->nnz = S.nnz;

  // init log(det(X))
  logdet = 0.0;
  // init |\Lambda*X|_1
  l1normX = 0.0;
  // init trace(SX)
  trSX = 0.0;
  integer ii, kk, ll, *pi;
  double myLambda;
  for (j = 0; j < p; j++) {
    // W is diagonal
    Wjj = W.val[j][0];

    m = S.ncol[j];
    D->ncol[j] = m;
    D->rowind[j] = (integer *)malloc(m * sizeof(integer));
    D->val[j] = (double *)malloc(m * sizeof(double));
    // scan S_{:,j}
    // abbrevations for the current column of LambdaMatrix
    pi = LambdaMatrix->rowind[j];
    ll = LambdaMatrix->ncol[j];
    kk = 0;
    for (k = 0; k < m; k++) {
      // row index i of column j
      i = S.rowind[j][k];
      Sij = S.val[j][k];
      D->rowind[j][k] = i;
      D->val[j][k] = 0.0;

      // check for a different value of Lambda rather than the default value
      myLambda = Lambda;
      for (; kk < ll; kk++) {
        ii = pi[kk];
        if (ii == i) {
          myLambda = LambdaMatrix->val[j][kk];
          break;
        } else if (ii >= i)
          break;
      }  // end for kk

      // off-diagonal case
      if (i != j) {
        if (fabs(Sij) > myLambda) {
          // W is diagonal
          Wii = W.val[i][0];
          // off-diagonal case a = W_ii W_jj
          a = Wii * Wjj;
          // off-diagonal case b = S_ij, W is diagonal => W_ij=0
          b = Sij;

          // parameters for optimization parameter mu = -s(-f,valw) (c=0?)
          // where s(z,r)=sign(z)*max{|z|-r,0}
          f = b / a;
          valw = myLambda / a;
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
        }  // end if |Sij|>Lambda
      } else {
        // i=j
        // handle diagonal part for \Delta_jj
        // X is diagonal
        Xjj = fabs(X.val[j][0]);
        // update log(det(X))
        logdet += log(Xjj);
        // update |Lambda*X|_1
        if (RunTimeConfig.off_diagonal) l1normX += myLambda * Xjj;

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
          valw = myLambda / a;
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

  // objective value f(X) = -log(det(X)) + trace(SX) + |\Lambda*X|_1
  double fX = -logdet + trSX + l1normX;

  return fX;
};  // end DiagNewton

#endif
