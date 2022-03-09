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

// METIS
#include "squic.hpp"

#include "metis.h"

using namespace std;

SQUIC::~SQUIC() {
  delete[] Y;
  FreeSparse(&W);
  FreeSparse(&X);
};

SQUIC::SQUIC(integer Y_n, integer Y_p, double *Y_value) {
  n = Y_n;
  p = Y_p;
  Y = new double[Y_p * Y_n];
  memcpy(Y, Y_value, Y_p * Y_n * sizeof(double));

  InitSparseMatrix(&X, 1.0);
  InitSparseMatrix(&W, 1.0);
};

void SQUIC::InitVars() {
  Stat.opt.clear();
  Stat.time_itr.clear();
  Stat.time_chol.clear();
  Stat.time_inv.clear();
  Stat.time_lns.clear();
  Stat.time_upd.clear();

  InitSparseMatrix(&L, 0.0);
  InitSparseMatrix(&P, 1.0);
  InitSparseMatrix(&D, 0.0);

  InitSparseBlockMatrix(&BL);
  InitSparseBlockMatrix(&BiD);

  mu.resize(p);

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
  // Cholmod.common_parameters.method[0].ordering = CHOLMOD_GIVEN ;
  // Cholmod.common_parameters.postorder = true ;
  // Cholmod.common_parameters.grow0 = 10.0;
  Cholmod.common_parameters.print = -1;
  Cholmod.pA = new cholmod_sparse();
  // Be carefull this means that the matrix is (symmetric) but only the lower
  // triangular part is stored
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
};  // end InitVars

void SQUIC::DeleteVars() {
  if (Cholmod.pL != NULL)
    CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
  CHOLMOD_FINISH(&Cholmod.common_parameters);
  delete Cholmod.pA;

  FreeSparse(&S);
  FreeSparse(&L);
  FreeSparse(&P);

  FreeBlockSparse(&BL);
  FreeBlockSparse(&BiD);
};  // end SQUIC_DeleteVars

// Entry point without initial guesses (warm start)
void SQUIC::Run(double lambda, double drop_tol, int max_iter, double term_tol) {
  RunCore(lambda, drop_tol, max_iter, term_tol);
}  // end SQUIC::run

// Entry point WITH initial guesses
void SQUIC::Run(double lambda, double drop_tol, int max_iter, double term_tol,
                SparseMatrix *X0, SparseMatrix *W0) {
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

  RunCore(lambda, drop_tol, max_iter, term_tol);
}  // end SQUIC::run

// core randomized active set routine after optionally providing initial guesses
// for X and W
void SQUIC::RunCore(double lambda, double drop_tol, int max_iter,
                    double term_tol) {
  // init runtime variables
  InitVars();

  Stat.time_total = -omp_get_wtime();

  if (RunTimeConfig.verbose > 0) {
    MSG("----------------------------------------------------------------\n");
    MSG("                     SQUIC Version %.2f                         \n",
        SQUIC_VER);
    MSG("----------------------------------------------------------------\n");
    MSG("Input Matrices\n");
    MSG(" nnz(X0)/p:   %e\n", double(X.nnz) / double(p));
    MSG(" nnz(W0)/p:   %e\n", double(W.nnz) / double(p));
    MSG(" nnz(M)/p:    ignored\n");
    MSG(" Y:           %d x %d \n", p, n);
    MSG("Parameters       \n");
    MSG(" verbose:     %d \n", RunTimeConfig.verbose);
    MSG(" lambda:      %e \n", lambda);
    MSG(" max_iter:    %d \n", max_iter);
    MSG(" term_tol:    %e \n", term_tol);
    MSG(" inv_tol:     %e \n", drop_tol);
    MSG(" threads:     %d \n", omp_get_max_threads());
    MSG("\n");

    MSG("#SQUIC Started \n");
    fflush(stdout);
  } else {
    MSG("#SQUIC Version %.3f : p=%g n=%g lambda=%g max_iter=%g term_tol=%g "
        "drop_tol=%g ",
        SQUIC_VER, double(p), double(n), double(lambda), double(max_iter),
        double(term_tol), double(drop_tol));
    fflush(stdout);
  }

  integer max_newton_iter = max_iter;
  integer line_search_iter_max = LINE_SEARCH_ITER_MAX;
  double coord_dec_sweep_tol = 0.05;

  // tolerance used to compute log(det(X)) and X^{-1} approximately
  double current_drop_tol = MAX(DROP_TOL0, drop_tol);

  // parallel version requires omp_get_max_threads() local versions of idx,
  // idxpos to avoid memory conflicts between the threads
  integer k = omp_get_max_threads();

  // used as stack and list of indices
  integer *idx = (integer *)malloc((size_t)k * p * sizeof(integer));

  // used as check mark array for nonzero entries, must be initialized with 0
  integer *idxpos = (integer *)calloc((size_t)k * p, sizeof(integer));

  // used as list of nonzero values, must be initialized with 0.0
  double *val = (double *)malloc((size_t)k * p * sizeof(double));
  for (integer j = 0; j < k * p; j++) val[j] = 0.0;

  //////////////////////////////////////////////////
  // ! START:compute sample covariance matrix
  //////////////////////////////////////////////////
  Stat.time_cov = -omp_get_wtime();
  // Compute mean value of Y
  double *pY;
  for (integer i = 0; i < p; i++) {
    mu[i] = 0.0;
    double *pY = &Y[i];
    for (integer j = 0; j < n; j++) {
      mu[i] += pY[j * p];
    }
    mu[i] /= (double)n;
  }

  // generate at least the diagonal part of S and those S_ij s.t. |S_ij|>lambda
  // sort nonzero entries in each column of S in increasing order
  // Remark: it suffices to sort S once after GenerateS and AugmentS
  // nu, ind carry the S_ii in decreasing order
  double nu_not_used[p];
  integer ind_not_used[p];
  GenerateSXL3B_OMP(&mu[0], lambda, nu_not_used, ind_not_used, &idx, idxpos);
  AugmentS_OMP(&mu[0], idx, idxpos, &X);
  SortSparse_OMP(&S, idx);
  Stat.time_cov += omp_get_wtime();
  if (RunTimeConfig.verbose > 0) {
    MSG("* sample covariance matrix S: time=%.3e nnz(S)/p=%.3e\n",
        Stat.time_cov, double(S.nnz) / double(p));
    fflush(stdout);
  }
  //////////////////////////////////////////////////
  // END:compute sample covariance matrix
  //////////////////////////////////////////////////

  //////////////////////////////////////////////////
  // ! START: Initial Objective Value
  //////////////////////////////////////////////////
  double fX = 1e+15;
  double l1norm_X = 0.0;
  double tr_SX = 0.0;
  double logdet_X = 0;  // logdet of Identity is

  bool chol_failed = LogDet(&logdet_X, current_drop_tol);

  // choleksy failed, the intial guess is not posdef
  if (chol_failed) {
    MSG("* Initial Guess is not positive definate.\n");
    fflush(stdout);

    FreeSparse(&D);
    free(idx);
    free(idxpos);
    free(val);
    return;
  }

  // scan matrices S and X
  for (integer j = 0; j < p; j++) {
    // counters for X_{:,j}, S_{:,j}
    integer k = 0;
    integer l = 0;
    while (k < X.ncol[j] || l < S.ncol[j]) {
      // set row indices to a value larger than possible
      integer r = p;
      integer s = p;
      // row index r of X_rj
      if (k < X.ncol[j]) {
        r = X.rowind[j][k];
      }
      // row index s of S_sj
      if (l < S.ncol[j]) {
        s = S.rowind[j][l];
      }

      // determine smallest index i=min{r,s}<p
      integer i = r;
      if (s < i) {
        i = s;
      }

      // only load the values of the smallest index
      double Xij = 0.0;
      double Sij = 0.0;
      if (r == i) {
        Xij = X.val[j][k++];
      }
      if (s == i) {
        Sij = S.val[j][l++];
      }
      l1norm_X += fabs(Xij);
      tr_SX += Sij * Xij;
    }
  }
  l1norm_X *= lambda;
  fX = -logdet_X + tr_SX + l1norm_X;
  //////////////////////////////////////////////////
  // END: Initial Objective Value
  //////////////////////////////////////////////////

  //////////////////////////////////////////////////
  // ! START: Newton Iteration
  //////////////////////////////////////////////////
  // initially use active Set of tentative size 2*p
  vector<index_pair> active_index_set;
  double gamma = atof(std::getenv("GAMMA"));
  // std::cout << "GAMMA = " << gamma << endl;

  double relative_objective_diff = INF;

  double relative_objective_avg = 0.0;

  double gamma_base = gamma;

  // outer Newton iteration loop, at most max_iter iteration steps unless
  // convergence
  for (integer newton_iter = 1;
       newton_iter <= max_newton_iter && relative_objective_diff > term_tol;
       ++newton_iter) {
    Stat.time_itr.push_back(-omp_get_wtime());

    // pattern of S overlaps W
    AugmentS_OMP(&mu[0], idx, idxpos, &W);
    SortSparse_OMP(&S, idx);

    if (relative_objective_diff < term_tol * 10) {
      gamma = gamma * .9;
      std::cout << "GAMMA = " << gamma << endl;
    }

    cout << "out_gamma(end+1)=" << gamma << ";" << endl;

    //////////////////////////////////////////////////
    // ! START: Compute Active Set X_ij!=0 or |S_ij-W_ij|>lambda
    //////////////////////////////////////////////////
    // Initial index set is always diagonal
    if (newton_iter == 1) {
      active_index_set.reserve(X.nr);
      for (integer i = 0; i < X.nr; ++i) {
        index_pair temp_index_pair;
        temp_index_pair.i = i;
        temp_index_pair.j = i;
        active_index_set.push_back(temp_index_pair);
      }
    } else {
      SelectedActiveIndexSet(lambda, gamma, active_index_set);
    }

    cout << "out_abs_grad{end+1}=sparse(" << p << "," << p << ");" << endl;
    for (integer i = 0; i < active_index_set.size(); i++) {
      cout << "out_active_inx{end}(" << active_index_set[i].i + 1 << ","
           << active_index_set[i].j + 1 << ")=" << 1 << ";" << endl;
      cout << "out_active_inx{end}(" << active_index_set[i].j + 1 << ","
           << active_index_set[i].i + 1 << ")=" << 1 << ";" << endl;
    }

    //////////////////////////////////////////////////
    // END: Compute Active Set X_ij!=0 or |S_ij-W_ij|>lambda
    //////////////////////////////////////////////////

    // augment the pattern of \Delta with the pattern from the active set
    ClearSparse(&D);
    AugmentD_OMP(&D, idx, idxpos, &active_index_set[0],
                 active_index_set.size());
    SortSparse_OMP(&D, idx);
    /// ierr = CheckSymmetry(&D);

    ////////////////////////////////////////
    // START: Coordinate Decent Update
    ////////////////////////////////////////
    Stat.time_upd.push_back(-omp_get_wtime());
    double l1norm_D = 0.0;  // |\Delta|_1
    double diff_D = 0.0;    // |(\mu e_ie_j^T)_{i,j}|_1
    srand(1);

    for (integer cdSweep = 1; cdSweep <= 1 + newton_iter / 3; cdSweep++) {
      diff_D = 0.0;

      // random swap order of elements in the active set
      for (integer i = 0; i < active_index_set.size(); i++) {
        integer j = i + rand() % (active_index_set.size() - i);

        integer k1 = active_index_set[i].i;
        integer k2 = active_index_set[i].j;

        active_index_set[i].i = active_index_set[j].i;
        active_index_set[i].j = active_index_set[j].j;
        active_index_set[j].i = k1;
        active_index_set[j].j = k2;
      }

      // update \Delta_ij  where
      // \Delta' differs from \Delta only in positions (i,j), (j,i)
      // l1norm_D , diffD will be updated
      BlockCoordinateDescentUpdate(
          lambda, &D, idx, idxpos, val, &active_index_set[0], 0,
          active_index_set.size() - 1, l1norm_D, diff_D);

      if (diff_D <= l1norm_D * coord_dec_sweep_tol) break;
    }

    Stat.time_upd.back() += omp_get_wtime();
    ////////////////////////////////////////
    // END: coordinate decent update
    ////////////////////////////////////////

    ////////////////////////////////////////
    // ! START: compute trace((S-X^{-1})\Delta)
    ////////////////////////////////////////
    double trace_gradD = 0.0;
    for (integer j = 0; j < p; j++) {
      // counters for S_{:,j}, W_{:,j}, D_{:,j}
      integer k = 0;
      integer l = 0;
      integer m = 0;

      while (k < S.ncol[j] || l < W.ncol[j] || m < D.ncol[j]) {
        // set row indices to a value larger than possible
        integer r = p;
        integer s = p;
        integer t = p;
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
        integer i = r;
        if (s < i) {
          i = s;
        }
        if (t < i) {
          i = t;
        }

        // only load the values of the smallest index
        double Sij = 0.0;
        double Wij = 0.0;
        double Dij = 0.0;
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
        trace_gradD += (Sij - Wij) * Dij;
      }
    }
    // augment X with the pattern of \Delta
    AugmentSparse_OMP(&X, idx, idxpos, &D);
    SortSparse_OMP(&X, idx);
    AugmentS_OMP(&mu[0], idx, idxpos, &X);
    SortSparse_OMP(&S, idx);
    ////////////////////////////////////////
    // END: compute trace((S-X^{-1})\Delta)
    ////////////////////////////////////////

    ////////////////////////////////////////
    // ! START: Line Search & Factorization
    ////////////////////////////////////////
    Stat.time_lns.push_back(-omp_get_wtime());

    // Components in the objective function
    double fX_last = fX;
    double l1norm_XD = 0.0;

    // Coefficent
    double alpha = 1.0 / 0.5;
    double loc_current_drop_tol = current_drop_tol / SHRINK_DROP_TOL;
    integer line_search_iter_count = 0;
    double L_nnz_per_row = 0;

    // Checks
    bool chol_failed = true;
    bool armijo_failed = true;

    // Start of line search
    for (integer line_search_iter = 0;
         line_search_iter < line_search_iter_max &&
         (chol_failed || armijo_failed);
         line_search_iter++) {
      // possibly the LDL decomposition is not accurate enough

      double logdet_X_update = 0.0;
      double l1norm_X_update = 0.0;
      double tr_SX_update = 0.0;

      line_search_iter_count++;
      alpha *= 0.5;
      loc_current_drop_tol *= SHRINK_DROP_TOL;

      // update X <- X + \alpha \Delta
      for (integer j = 0; j < p; j++) {
        // counters for S_{:,j}, X_{:,j}, D_{:,j}
        integer k = 0;
        integer l = 0;
        integer m = 0;
        while (k < S.ncol[j] || l < X.ncol[j] || m < D.ncol[j]) {
          // set row indices to a value larger than possible
          integer r = p;
          integer s = p;
          integer t = p;
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
          integer i = r;
          if (s < i) {
            i = s;
          }
          if (t < i) {
            i = t;
          }
          // only load the values of the smallest index
          double Sij = 0.0;
          double Dij = 0.0;
          double Xij = 0.0;
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

          l1norm_X_update += fabs(Xij);
          tr_SX_update += Sij * Xij;
        }
      }

      // adjust by \lambda
      l1norm_X_update *= lambda;

      // We just weant to \lamdba*|X + D| ... not  \lamdba*|X +\alpha*D|
      if (alpha == 1.0) {
        l1norm_XD = l1norm_X_update;
      }

      // Compute Cholesky+log(det X)
      chol_failed = LogDet(&logdet_X_update, loc_current_drop_tol);

      // If choleksy did not fail
      // Check Armijo’s rule (Bertsekas, 1995; Tseng and Yun, 2007)
      // f(X+alpha*D)<=f(X)+alpha*sigma*(tr[grad * D]+|X+D|_1-|Lambda * X|_1)
      // update objective function
      // f(X_update)=-log(det[X_update])+tra[SX_update]+|\lambda*X_update|_1
      double fX_update = -logdet_X_update + tr_SX_update + l1norm_X_update;
      double del = trace_gradD + l1norm_XD - l1norm_X;

      armijo_failed =
          !(fX_update <= (fX + alpha * SIGMA * del) || l1norm_D < EPS);

      // Cholesky failed or armijio wasnt sufficent, retry line
      // search with smaller alpha
      if (chol_failed || armijo_failed) {
        // downdate X <- X - \alpha \Delta
        for (integer j = 0; j < p; j++) {
          // counters for X_{:,j}, D_{:,j}
          integer l = 0;
          integer m = 0;
          while (l < X.ncol[j] || m < D.ncol[j]) {
            // set row indices to a value larger than possible
            integer s = p;
            integer t = p;
            if (l < X.ncol[j]) {
              s = X.rowind[j][l];
            }
            if (m < D.ncol[j]) {
              t = D.rowind[j][m];
            }
            // compute smallest index i=min{s,t}<p
            integer i = s;
            if (t < i) {
              i = t;
            }
            // only load the values of the smallest index
            double Xij = 0.0;
            double Dij = 0.0;
            if (i == t) {
              Dij = D.val[j][m++];
            }
            // load and downdate X_ij <- X_ij + \alpha \Delta_ij
            if (i == s) {
              Xij = X.val[j][l];
              Xij -= Dij * alpha;
              X.val[j][l++] = Xij;
            }
          }
        }
      } else {
        // Accepted update
        fX = fX_update;
        l1norm_X = l1norm_X_update;
        logdet_X = logdet_X_update;
        tr_SX = tr_SX_update;
      }
    }

    // Export Factor
    // export L,D to sparse matrix format along with the permutation
    // at the same time discard the Cholesky factorization
    BlockExportLdl();
    L_nnz_per_row = double(BL.nnz + BiD.nnz) / double(p);
    // next time we call Cholesky the pattern may change (analyze_done=0)
    analyze_done = 0;

    Stat.time_lns.back() += omp_get_wtime();
    ////////////////////////////////////////
    // END: Line Search & Factorization
    ////////////////////////////////////////

    ///////////////////////////////////////
    // ! START: Matrix Inversion
    ///////////////////////////////////////
    Stat.time_inv.push_back(-omp_get_wtime());
    BlockNeumannInv(current_drop_tol, NULL);
    Stat.time_inv.back() += omp_get_wtime();
    ///////////////////////////////////////
    // End: Matrix Inversion
    ///////////////////////////////////////

    ///////////////////////////////////////
    // ! START: Matrix Cleanup
    ///////////////////////////////////////
    X.nnz = 0;
    for (integer j = 0; j < p; ++j) {
      // Temp buffers
      vector<double> buffer_val;
      vector<integer> buffer_rowind;
      buffer_val.reserve(X.ncol[j]);
      buffer_rowind.reserve(X.ncol[j]);

      // Loop over each column to count nnz
      for (integer k = 0; k < X.ncol[j]; ++k) {
        if (fabs(X.val[j][k]) > EPS) {
          buffer_val.push_back(X.val[j][k]);
          buffer_rowind.push_back(X.rowind[j][k]);
        }
      }

      // Update and reallocate values of X
      X.ncol[j] = buffer_val.size();
      X.nnz += X.ncol[j];
      X.rowind[j] =
          (integer *)realloc(X.rowind[j], (size_t)X.ncol[j] * sizeof(integer));
      X.val[j] =
          (double *)realloc(X.val[j], (size_t)X.ncol[j] * sizeof(double));

      // Copy over buffer
      for (integer k = 0; k < X.ncol[j]; ++k) {
        X.val[j][k] = buffer_val[k];
        X.rowind[j][k] = buffer_rowind[k];
      }
    }
    ///////////////////////////////////////
    // End: Matrix Cleanup
    ///////////////////////////////////////

    // we use at least the relative residual between two consecutive calls
    // + some upper/lower bound
    relative_objective_diff = fabs((fX - fX_last) / fX);
    current_drop_tol = DROP_TOL_GAP * relative_objective_diff;
    current_drop_tol = MAX(current_drop_tol, drop_tol);
    current_drop_tol = MIN(MAX(DROP_TOL0, drop_tol), current_drop_tol);

    Stat.time_itr.back() += omp_get_wtime();
    Stat.opt.push_back(fX);

    if (RunTimeConfig.verbose > 0) {
      MSG("* iter=%d time=%.3e obj=%.3e |delta(obj)|/obj=%.3e "
          "nnz(X,L,W)/p=[%.3e %.3e %.3e] lns_iter=%d \n",
          newton_iter, Stat.time_itr.back(), fX, relative_objective_diff,
          double(X.nnz) / double(p), L_nnz_per_row, double(W.nnz) / double(p),
          line_search_iter_count);
      fflush(stdout);
    }

    cout << "out_X_nnz(end+1)=" << double(X.nnz) << ";" << endl;
    cout << "out_L_nnz(end+1)=" << double(L_nnz_per_row) << ";" << endl;
    cout << "out_W_nnz(end+1)=" << double(W.nnz) << ";" << endl;
    cout << "out_obj(end+1)=" << fX << ";" << endl;
    cout << "out_time(end+1)=" << Stat.time_itr.back() << ";" << endl;
    cout << "out_ll(end+1)=" << -1.0 * (-logdet_X + tr_SX) << ";" << endl;
  }

  Stat.time_total += omp_get_wtime();
  Stat.trSX = tr_SX;
  Stat.logdetX = logdet_X;

  if (RunTimeConfig.verbose > 0) {
    MSG("#SQUIC Finished: time=%0.3e nnz(X,W)/p=[%0.3e %0.3e] \n\n",
        Stat.time_total, double(X.nnz) / double(p), double(W.nnz) / double(p));
    fflush(stdout);
  } else {
    MSG("time=%0.3e nnz(X,W)/p=[%0.3e %0.3e] \n", Stat.time_total,
        double(X.nnz) / double(p), double(W.nnz) / double(p));
    fflush(stdout);
  }

  FreeSparse(&D);
  free(idx);
  free(idxpos);
  free(val);

  return;
};  // end run

void SQUIC::FreeSparse(SparseMatrix *S) {
  integer j, p;

  if (S == NULL) return;
  p = S->nc;
  if (S->rowind != NULL) {
    for (j = 0; j < p; j++) {
      if (S->rowind[j] != NULL) free(S->rowind[j]);
    }  // end for j
    free(S->rowind);
    S->rowind = NULL;
  }  // end if
  if (S->val != NULL) {
    for (j = 0; j < p; j++) {
      if (S->val[j] != NULL) free(S->val[j]);
    }  // end for j
    free(S->val);
    S->val = NULL;
  }  // end if
  if (S->ncol != NULL) {
    free(S->ncol);
    S->ncol = NULL;
  }  // end if
  S->nr = S->nc = S->nnz = 0;
};  // end FreeSparse

void SQUIC::FreeBlockSparse(SparseBlockMatrix *S) {
  integer j, p;

  if (S == NULL) return;
  p = S->nc;
  if (S->rowind != NULL) {
    for (j = 0; j < p; j++)
      if (S->rowind[j] != NULL) free(S->rowind[j]);
    free(S->rowind);
    S->rowind = NULL;
  }  // end if

  if (S->colind != NULL) {
    for (j = 0; j < p; j++)
      if (S->colind[j] != NULL) free(S->colind[j]);
    free(S->colind);
    S->colind = NULL;
  }  // end if

  if (S->valD != NULL) {
    for (j = 0; j < p; j++)
      if (S->valD[j] != NULL) free(S->valD[j]);
    free(S->valD);
    S->valD = NULL;
  }  // end if

  if (S->valE != NULL) {
    for (j = 0; j < p; j++)
      if (S->valE[j] != NULL) free(S->valE[j]);
    free(S->valE);
    S->valE = NULL;
  }  // end if

  if (S->nblockrow != NULL) {
    free(S->nblockrow);
    S->nblockrow = NULL;
  }  // end if

  if (S->nblockcol != NULL) {
    free(S->nblockcol);
    S->nblockcol = NULL;
  }  // end if

  S->nr = S->nc = S->nblocks = S->nnz = 0;
};  // end FreeBlockSparse

void SQUIC::NullSparse(SparseMatrix *S) {
  integer j, p = S->nc;
  for (j = 0; j < p; j++) {
    free(S->rowind[j]);
    free(S->val[j]);
    S->rowind[j] = NULL;
    S->val[j] = NULL;
    S->ncol[j] = 0;
  }  // end for j

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
    }  // end for k
  }    // end for j

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
    }  // end if
    // reset nz counter
    idx[i] = 0;
  }  // end for i
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
    }  // end for k
  }    // end for j
  for (j = 0; j < p; j++) {
    for (k = 0; k < S->ncol[j]; k++) {
      if (S->rowind[j][k] != ST.rowind[j][k] || ST.val[j][k] != S->val[j][k]) {
        ierr = -1;
      }  // end if
    }    // end for k,l
    // reset nz counter
    idx[j] = 0;
  }  // end for j

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
    }  // end for k

    MSG("\n");
    fflush(stdout);
    for (k = 0; k < S->ncol[j]; k++) {
      val = S->val[j][k];
      MSG("%lf ", val);
    }  // end for k

    MSG("\n");
    fflush(stdout);
  }  // end for j
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
};  // end ClearSparse

void SQUIC::InitSparseMatrix(SparseMatrix *M, double diagVal) {
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
    }  // end for j
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
    }  // end for j
  }
};

void SQUIC::InitSparseBlockMatrix(SparseBlockMatrix *M) {
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
  }  // end for j
};   // end InitSparseBlockMatrix

void SQUIC::ResetSparseBlockMatrix(SparseBlockMatrix *M) {
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
  }  // end for j
};   // end reset_SparseBlockMatrix

void SQUIC::DeleteSparseBlockMatrix(SparseBlockMatrix *M) {
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
  }  // end for j
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
};  // end delete_SparseBlockMatrix

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
  }  // end for j
};

// Augment sparse matrix S with the pattern of W init with 0
void SQUIC::AugmentSparse(SparseMatrix *S, integer *idx, integer *idxpos,
                          SparseMatrix *W) {
  /*
  S          sparse matrix
  idx   \    pre-allocated arrays to hold the list of indices and the check
  marks idxpos/    idxpos must be initialized with 0 W          sparse matrix
  */

  integer i, j, k, p = W->nc,
                   cnt;  // counter for the number of nonzeros per column

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
    }  // end for k

    // now check if W has any additional nonzeros
    for (k = 0; k < W->ncol[j]; k++) {
      // index i of W_ij
      i = W->rowind[j][k];
      // this is an additional fill-in for S
      if (!idxpos[i]) {
        idx[cnt++] = i;
        // flag them negative to indicate fill-in
        idxpos[i] = -cnt;
      }  // end if
    }    // end for k

    // re-allocate memory for column j
    S->rowind[j] =
        (integer *)realloc(S->rowind[j], (size_t)cnt * sizeof(integer));
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
      }  // end if
    }    // end for k
    // new number of nonzeros in S
    S->ncol[j] = cnt;
    S->nnz += cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idx[k];
      idxpos[i] = 0;
    }  // end for k
  }    // end for j
};     // end AugmentSparse

// Augment sparse matrix S with the pattern of W init with 0
void SQUIC::AugmentSparse_OMP(SparseMatrix *S, integer *idx, integer *idxpos,
                              SparseMatrix *W) {
  /*
  S          sparse matrix
  idx   \    pre-allocated arrays to hold the list of indices and the check
  marks idxpos/    idxpos must be initialized with 0 W          sparse matrix
  */

  integer i, j, k, p = W->nc, *idxl, *idxposl, Sncolj, *Srowindj, Wncolj,
                   *Wrowindj,
                   mythreadnum,  // individual thread number
      cnt;  // counter for the number of nonzeros per column
  double *Svalj;

// -----------------------------------------------------------
#pragma omp parallel for shared(p, idx, idxpos, S, W) private(                 \
    mythreadnum, idxl, idxposl, cnt, Sncolj, Srowindj, k, i, Wncolj, Wrowindj, \
    Svalj)
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
        MSG("AugmentSparse_OMP, thread %ld: idxposl[%ld]!=0\n", mythreadnum,
            k + 1);
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
    }  // end for k

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
      }  // end if
    }    // end for k

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
      }  // end if
    }    // end for k
    // new number of nonzeros in S
    S->ncol[j] = cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idxl[k];
      idxposl[i] = 0;
    }  // end for k
  }    // end for j
  // end omp parallel for
  // -----------------------------------------------------------

  S->nnz = 0;
  for (j = 0; j < p; j++) S->nnz += S->ncol[j];
};  // end AugmentSparse_OMP

// Define D on the pattern of S and return the objective value of f(X)
double SQUIC::DiagNewton(const double lambda, SparseMatrix *D) {
  /*
  lambda     Lagrangian parameter
  S          sparse representation of the sample covariance matrix such that
  at least its diagonal entries and S_ij s.t. |S_ij|>\lambda are stored
  X          sparse DIAGONAL constrained inverse covariance matrix
  W          W ~ X^{-1} sparse DIAGONAL covariance matrix
  D          define sparse update matrix \Delta with same pattern as S
  D is defined as sparse matrix 0
  */

  integer i, j, k, l, m, r, s, p;

  double Sij, Wii, Wjj, Xjj,
      logdet,                // log(det(X))
      l1normX,               // |\lambda*X|_1
      trSX,                  // trace(SX)
      a, b, c, f, valw, mu;  // parameters for optimization function

  p = S.nr;
  // new number of nonzeros for \Delta
  D->nnz = S.nnz;

  // init log(det(X))
  logdet = 0.0;
  // init |\lambda*X|_1
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
        if (fabs(Sij) > lambda) {
          // W is diagonal
          Wii = W.val[i][0];
          // off-diagonal case a = W_ii W_jj
          a = Wii * Wjj;
          // off-diagonal case b = S_ij, W is diagonal => W_ij=0
          b = Sij;

          // parameters for optimization parameter mu = -s(-f,valw) (c=0?)
          // where s(z,r)=sign(z)*max{|z|-r,0}
          f = b / a;
          valw = lambda / a;
          // S_ij<0
          if (0.0 > f) {
            // \mu=(|S_ij|-\lambda)/a
            mu = -f - valw;
            // mu>0 <=> |Sij|>\lambda
            // |S_ij|<\lambda_ij?
            if (mu < 0.0) {
              mu = 0.0;
            }
          } else {
            // S_ij>0
            // \mu=(-|S_ij|+\lambda)/a
            mu = -f + valw;
            // mu<0 <=> |Sij|>\lambda
            // |S_ij|<\lambda_ij?
            if (mu > 0.0) {
              mu = 0.0;
            }
          }
          // since S is exactly symmetric, the same update will take place
          // for \Delta_ji
          D->val[j][k] = mu;
        }  // end if |Sij|>lambda
      } else {
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
          valw = lambda / a;
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
  // adjust |\lambda X|_1
  l1normX *= lambda;

  // objective value f(X) = -log(det(X)) + trace(SX) + |\lambda*X|_1
  double fX = -logdet + trSX + l1normX;

  return fX;
};  // end DiagNewton

// check symmetric positive definite matrix A for off-diagonal contributions
// We assume that the diagonal part is always included.
// A matrix is considered to be non-diagonal if every column consists of more
// than one entry
integer SQUIC::IsDiag(const SparseMatrix *A) {
  for (integer i = 0; i < A->nr; i++)
    if (A->ncol[i] > 1) return (integer)0;

  return (integer)1;
};

// to set up S properly, its row indices should be sorted in increasing order
void SQUIC::SortSparse(SparseMatrix *S, integer *stack) {
  integer j, k, p = S->nc;

  // #pragma omp parallel for schedule(static,1) shared(p,S,mystack) private(k)
  // stack has to be of size p*omp_get_max_threads() such that each tread
  // gets its local chunk of size p, i.e. integer*
  // my_stack=stack+p*omp_get_thread_num();
  for (j = 0; j < p; j++) {
    // start column j
    k = S->ncol[j];
    // sort S_{:,j} with quicksort such that the indices are taken in increasing
    // order
    qsort1_(S->val[j], S->rowind[j], stack, &k);
  }  // end for j
};

// to set up sparse matrices properly, its row indices should be sorted in
// increasing order OpenMP version
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
    // sort S_{:,j} with quicksort such that the indices are taken in increasing
    // order
    qsort1_(S->val[j], S->rowind[j], stackl, &k);
  }  // end for j
     // end omp parallel
};   // end SortSparse_OMP

// Augment S with the pattern of W and generate the associated entries
// void SQUIC::AugmentS(double *mu, integer *idx, integer *idxpos, SparseMatrix
// *T) {
void SQUIC::AugmentS(double *mu, integer *idx, integer *idxpos,
                     SparseMatrix *W) {
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
  idx   \    pre-allocated arrays to hold the list of indices and the check
  marks idxpos/    idxpos must be initialized with 0 W          W ~ X^{-1}
  sparse matrix
  */

  integer i, j, k, r,
      cnt;  // counter for the number of nonzeros per column

  double *pY,  // pointer to Y_{i,:}
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
    }  // end for k

    // now check if W has any additional nonzeros
    for (k = 0; k < W->ncol[j]; k++) {
      // index i of W_ij
      i = W->rowind[j][k];
      // this is an additional fill-in for S
      if (!idxpos[i]) {
        idx[cnt++] = i;
        // flag them negative to indicate fill-in
        idxpos[i] = -cnt;
      }  // end if
    }    // end for k

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

      }  // end if
    }    // end for k
    // new number of nonzeros in S
    S.ncol[j] = cnt;
    S.nnz += cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idx[k];
      idxpos[i] = 0;
    }  // end for k
  }    // end for j
};

// Augment S with the pattern of W and generate the associated entries
void SQUIC::AugmentS_OMP(double *mu, integer *idx, integer *idxpos,
                         SparseMatrix *W) {
  /*
  Y          n samples Y=[y_1,...y_n] from which the low-rank empirical
                     covariance matrix S=1/NM1(n) sum_i (y_i-mu)(y_i-mu)^T can
  be built, where mu=1/n sum_i y_i is the mean value Y is stored by columns p
  size parameter space n          number of samples mu         mean value S
  sparse representation of the sample covariance matrix idx   \    pre-allocated
  arrays to hold the list of indices and the check marks idxpos/    idxpos must
  be initialized with 0 W          W ~ X^{-1} sparse matrix
  */

  integer i, j, k, r, *idxl, *idxposl,
      mythreadnum,  // individual thread number
      cnt;          // counter for the number of nonzeros per column

  double *pY,  // pointer to Y_{i,:}
      Sij;

#pragma omp parallel for shared(p, idx, idxpos, S, W, Y, n, mu) private( \
    mythreadnum, idxl, idxposl, cnt, k, i, Sij, r, pY)
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
    }  // end for k

    // now check if W has any additional nonzeros
    for (k = 0; k < W->ncol[j]; k++) {
      // index i of W_ij
      i = W->rowind[j][k];
      // this is an additional fill-in for S
      if (!idxposl[i]) {
        idxl[cnt++] = i;
        // flag them negative to indicate fill-in
        idxposl[i] = -cnt;
      }  // end if
    }    // end for k

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
      }  // end if
    }    // end for k
    // new number of nonzeros in S
    S.ncol[j] = cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idxl[k];
      idxposl[i] = 0;
    }  // end for k
  }    // end for j
  // end omp parallel for

  // finally count number of nonzeros
  S.nnz = 0;
  for (j = 0; j < p; j++) S.nnz += S.ncol[j];
};  // end AugmentS_OMP

// augment the pattern of \Delta with the pattern from the active set
void SQUIC::AugmentD(SparseMatrix *D, integer *idx, integer *idxpos,
                     index_pair *activeSet, integer numActive) {
  /*
  D          sparse symmetric update matrix
  idx   \    pre-allocated arrays to hold the list of indices and the check
  marks idxpos/    idxpos must be initialized with 0 activeSet  list of acitve
  pairs numActive  number of pairs
  */

  integer i, j, k, l, p = D->nr,
                      cnt;  // counter for the number of nonzeros per column
  SparseMatrix Active;

  // convert Active set list into a sparse matrix in order to augment D
  // efficiently
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
  }  // end for k
  // allocate index memory
  Active.nnz = 0;
  for (j = 0; j < p; j++) {
    l = Active.ncol[j];
    Active.rowind[j] = (integer *)malloc(l * sizeof(integer));
    Active.nnz += l;
    // reset counter
    Active.ncol[j] = 0;
  }  // end for j
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
    }  // end if
  }    // end for k

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
    }  // end for k

    // now check if ActiveSet has any additional nonzeros
    for (k = 0; k < Active.ncol[j]; k++) {
      // index i of Active_ij
      i = Active.rowind[j][k];
      // this is an additional fill-in for \Delta
      if (!idxpos[i]) {
        idx[cnt++] = i;
        // flag them negative to indicate fill-in
        idxpos[i] = -cnt;
      }  // end if
    }    // end for k

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
      }  // end if
    }    // end for k
    // new number of nonzeros in \Delta
    D->ncol[j] = cnt;
    // update new fill in column j
    D->nnz += cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idx[k];
      idxpos[i] = 0;
    }  // end for k
  }    // end for j

  // release index memory
  for (j = 0; j < p; j++) {
    free(Active.rowind[j]);
  }
  free(Active.ncol);
  free(Active.rowind);
};  // end AugmentD

// augment the pattern of \Delta with the pattern from the active set
void SQUIC::AugmentD_OMP(SparseMatrix *D, integer *idx, integer *idxpos,
                         index_pair *activeSet, integer numActive) {
  /*
  D          sparse symmetric update matrix
  idx   \    pre-allocated arrays to hold the list of indices and the check
  marks idxpos/    idxpos must be initialized with 0 activeSet  list of acitve
  pairs numActive  number of pairs
  */

  integer i, j, k, l, p = D->nr, *idxl, *idxposl,
                      mythreadnum,  // individual thread number
      cnt;  // counter for the number of nonzeros per column
  SparseMatrix Active;

  // convert Active set list into a sparse matrix in order to augment D
  // efficiently
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
    if (i != j) Active.ncol[i]++;
  }  // end for k
  // allocate index memory
  Active.nnz = 0;
  for (j = 0; j < p; j++) {
    l = Active.ncol[j];
    Active.rowind[j] = (integer *)malloc(l * sizeof(integer));
    Active.nnz += l;
    // reset counter
    Active.ncol[j] = 0;
  }  // end for j
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
    }  // end if
  }    // end for k

// now we may supplement \Delta with the nonzeros from the active set
#pragma omp parallel for shared(p, idx, idxpos, D, Active) private( \
    mythreadnum, idxl, idxposl, cnt, k, i)
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
    }  // end for k

    // now check if ActiveSet has any additional nonzeros
    for (k = 0; k < Active.ncol[j]; k++) {
      // index i of Active_ij
      i = Active.rowind[j][k];
      // this is an additional fill-in for \Delta
      if (!idxposl[i]) {
        idxl[cnt++] = i;
        // flag them negative to indicate fill-in
        idxposl[i] = -cnt;
      }  // end if
    }    // end for k

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
      }  // end if
    }    // end for k
    // new number of nonzeros in \Delta
    D->ncol[j] = cnt;

    // clear check marks
    for (k = 0; k < cnt; k++) {
      i = idxl[k];
      idxposl[i] = 0;
    }  // end for k
  }    // end for j
  // end omp parallel for

  // count final number of nonzeros
  D->nnz = 0;
  for (j = 0; j < p; j++) D->nnz += D->ncol[j];

  // release index memory
  for (j = 0; j < p; j++) free(Active.rowind[j]);
  free(Active.ncol);
  free(Active.rowind);
}  // end AugmentD_OMP

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
void SQUIC::CooToCustomeData(integer *X_i, integer *X_j, double *X_val,
                             integer nnz, SparseMatrix *X) {
  // Intialilize structure
  X->nr = X->nc = p;
  X->nnz = nnz;

  X->ncol =
      (integer *)calloc(p, sizeof(integer));  // set it equal to zero .. there
                                              // might be zero element in a row
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
  }  // end for j

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
void SQUIC::CscToCustomeData(integer *X_row_index, integer *X_col_ptr,
                             double *X_val, integer nnz, SparseMatrix *X) {
  // Intialilize structure
  X->nr = X->nc = p;
  X->nnz = nnz;

  integer temp_i = 0;

  X->ncol =
      (integer *)calloc(p, sizeof(integer));  // set it equal to zero .. there
                                              // might be zero element in a row
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

extern "C" {
#include "squic_interface.hpp"
}

#endif
