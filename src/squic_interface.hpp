/* squic_interface.hpp
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

#include "squic.hpp"
/**
 * @brief      Creat buffer at given location
 *
 * @param      buffer  The buffer
 * @param[in]  length  The length of the buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memset_integer(
    integer *&buffer, integer length) {
  buffer = new integer[length]();
}

/**
 * @brief      Creat buffer at given location
 *
 * @param      buffer  The buffer
 * @param[in]  length  The length of the buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memset_double(
    double *&buffer, integer length) {
  buffer = new double[length]();
}

/**
 * @brief      Creat a new copy of the buffer
 *
 * @param      buffer  The copy buffer
 * @param      values  The buffer to be copied
 * @param[in]  length  The length of the buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memcopy_integer(
    integer *&buffer, integer *values, integer length) {
  buffer = new integer[length]();

  for (integer i = 0; i < length; ++i) {
    buffer[i] = values[i];
  }
}

/**
 * @brief      Creat a new copy of the buffer
 *
 * @param      buffer  The copy buffer
 * @param      values  The buffer to be copied
 * @param[in]  length  The length of the buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memcopy_double(
    double *&buffer, double *values, integer length) {
  buffer = new double[length]();

  for (integer i = 0; i < length; ++i) {
    buffer[i] = values[i];
  }
}

/**
 * @brief      Free the buffer
 *
 * @param      buffer  The buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memfree_integer(
    integer *&buffer) {
  delete[] buffer;
}

/**
 * @brief      Free the buffer
 *
 * @param      buffer  The buffer
 */
__attribute__((visibility("default"))) void SQUIC_CPP_UTIL_memfree_double(
    double *&buffer) {
  delete[] buffer;
}

/**
 * @brief      SQUIC CPP Interface
 *
 * @param[in]        mode            Runtime mode values [0,1,2,3,4] use the
 * "block" algorithem and [5,6,7,8,9] use the "scalar" algorithem <<NOTE:
 * Recommended "0">>.
 * @param[in]        p               Number of random variables
 * @param[in]        n               Number of samples
 * @param[in]        Y               Data pointer
 * @param[in]        lambda          Scalar tunning parameter
 * @param[in]        M_rinx          M matrix row index
 * @param[in]        M_cptr          M matrix column pointer
 * @param[in]        M_val           M matrix value
 * @param[in]        M_nnz           M matrix number of nonzeros
 * @param[in]        max_iter        Maximum Netwon iterations <<NOTE:
 * max_iter=0 will return sample covaraince matrix in ouput iC >>
 * @param[in]        inv_tol         Inversion tolerance for approximate
 * inversion <<NOTE: inv_tol>0 , Recommended inv_tol=term_tol >>
 * @param[in]        term_tol        Termination tolerance <<NOTE: term_tol>0 ,
 * Recommended inv_tol=term_tol >>
 * @param[in]        verbose         Verbose level 0 or 1
 * @param[in/out]    X_rinx          Percision matrix row index       <<NOTE:
 * Intial value X0_rinx is passed in here >>
 * @param[in/out]    X_cptr          Percision matrix column pointer  <<NOTE:
 * Intial value X0_cptr is passed in here >>
 * @param[in/out]    X_val           Percision matrix value           <<NOTE:
 * Intial value X0_val is passed in here >>
 * @param[in/out]    X_nnz           Percision matrix nnz             <<NOTE:
 * Intial value X_nnz is passed in here >>
 * @param[in/out]    W_rinx          Covariance matrix row index      <<NOTE:
 * Intial value W0_rinx is passed in here >>
 * @param[in/out]    W_cptr          Covariance matrix column pointer <<NOTE:
 * Intial value W0_rinx is passed in here >>
 * @param[in/out]    W_val           Covariance matrix value          <<NOTE:
 * Intial value W0_rinx is passed in here >>
 * @param[in/out]    W_nnz           Covariance matrix nnz            <<NOTE:
 * Intial value W0_rinx is passed in here >>
 * @param[out]       info_num_iter   Information number of newton iterations
 * performed
 * @param[out]       info_times      Information 6 element array of times for
 * computing 1)total 2)sample covaraince 3)optimization 4)factorization
 * 5)approximate inversion 6)coordinate update
 * @param[out]       info_objective  Objective function value         <<NOTE:
 * this array must be of length max_iter when passed in. Upon ouput only
 * info_num_iter element will be written to>>
 * @param[out]       info_logdetX    Log determinant of X
 * @param[out]       info_trSX       Trace of SX
 */
__attribute__((visibility("default"))) void SQUIC_CPP(
    int mode, integer p, integer n, double *Y, double lambda, integer *M_rinx,
    integer *M_cptr, double *M_val, integer M_nnz, int max_iter, double inv_tol,
    double term_tol, int verbose, integer *&X_rinx, integer *&X_cptr,
    double *&X_val, integer &X_nnz, integer *&W_rinx, integer *&W_cptr,
    double *&W_val, integer &W_nnz, int &info_num_iter,
    double *&
        info_times,  // length must be 6:
                     // [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
    double *&info_objective,  // length must be size max_iter
    double &info_logdetX, double &info_trSX) {
  assert(p > 2);
  assert(n > 1);

  assert(X_nnz >= p);
  assert(W_nnz >= p);

  assert(lambda > 0.0);
  assert(max_iter >= 0);
  assert(inv_tol >= 0.0);
  assert(term_tol >= 0.0);
  assert(mode >= 0 && mode <= 9);

  verbose = min(2, max(0, verbose));

  // kind of (incomplete) Cholesky decomposition: 0 (CHOLMOD), 1 (PARDISO), 2
  // (JANUS)
  int cholesky = 0;
  int coord_update;
  int inversion;

  bool use_ZERO_M = M_nnz < 1;
  bool use_block = mode < 5;

  double *mu = new double[p];
  double *sd = new double[p];

  // block variant
  SQUIC *sq = new SQUIC(n, p, Y);
  sq->RunTimeConfig.verbose = verbose;
  sq->RunTimeConfig.generate_scm = DETERMINISTIC;
  sq->RunTimeConfig.block_upd = COLLECTIVE_CDU;
  sq->RunTimeConfig.factorization = CHOL_SUITESPARSE;
  sq->RunTimeConfig.inversion = BLOCK_FSPAI;

  // Using scalar lambda and I intial guess

  if (use_ZERO_M) {
    SQUIC::SparseMatrix X0;
    SQUIC::SparseMatrix W0;
    sq->CscToCustomeData(X_rinx, X_cptr, X_val, X_nnz, &X0);
    sq->CscToCustomeData(W_rinx, W_cptr, W_val, W_nnz, &W0);
    sq->Run(lambda, inv_tol, max_iter, term_tol, &X0, &W0);
    sq->FreeSparse(&X0);
    sq->FreeSparse(&W0);
  } else {
    SQUIC::SparseMatrix X0;
    SQUIC::SparseMatrix W0;
    SQUIC::SparseMatrix LambdaMatrix;
    sq->CscToCustomeData(X_rinx, X_cptr, X_val, X_nnz, &X0);
    sq->CscToCustomeData(W_rinx, W_cptr, W_val, W_nnz, &W0);
    sq->CscToCustomeData(M_rinx, M_cptr, M_val, M_nnz, &LambdaMatrix);
    sq->Run(lambda, &LambdaMatrix, inv_tol, max_iter, term_tol, &X0, &W0);
    sq->FreeSparse(&X0);
    sq->FreeSparse(&W0);
    sq->FreeSparse(&LambdaMatrix);
  }

  // Re allocate the memory
  delete[] X_rinx;
  delete[] X_cptr;
  delete[] X_val;
  delete[] W_rinx;
  delete[] W_cptr;
  delete[] W_val;

  if (max_iter == 0) {
    X_nnz = sq->X.nnz;
    W_nnz = sq->S.nnz;  // nnz for S!!!

    X_rinx = new integer[X_nnz];
    X_cptr = new integer[p + 1];
    X_val = new double[X_nnz];

    W_rinx = new integer[W_nnz];
    W_cptr = new integer[p + 1];
    W_val = new double[W_nnz];

    X_cptr[0] = 0;
    W_cptr[0] = 0;

    integer nnz_count_X = 0;
    integer nnz_count_W = 0;
    double val;

    for (integer j = 0; j < p; j++) {
      // Copy value of X
      for (integer k = 0; k < sq->X.ncol[j]; k++) {
        val = sq->X.val[j][k];
        if (fabs(val) > EPS) {
          X_rinx[nnz_count_X] = sq->X.rowind[j][k];
          X_val[nnz_count_X] = val;
          nnz_count_X++;
        }
      }
      X_cptr[j + 1] = nnz_count_X;

      // Copy values of S to W!!!
      for (integer k = 0; k < sq->S.ncol[j]; k++) {
        val = sq->S.val[j][k];
        if (fabs(val) > EPS) {
          W_rinx[nnz_count_W] = sq->S.rowind[j][k];
          W_val[nnz_count_W] = val;
          nnz_count_W++;
        }
      }
      W_cptr[j + 1] = nnz_count_W;
    }

    // The memory footprint is the same but the nnz values in the front of the
    // array
    X_nnz = nnz_count_X;
    W_nnz = nnz_count_W;  // nnz for S!!!

    // Passout out stats
    info_num_iter = sq->Stat.time_itr.size();
    info_times[0] = sq->Stat.time_total;
    info_times[1] = sq->Stat.time_cov;
    info_times[2] = 0.0;
    info_times[3] = 0.0;
    info_times[4] = 0.0;
    info_times[5] = 0.0;

    // we do nothing with info_objective[]
    // for (integer i = 0; i < sq->Stat.time_itr.size(); ++i)
    //{
    //	info_objective[i] = sq->Stat.opt[i];
    //}

    // info_logdetX = do nothing ;
    // info_trSX = do nothing ;
  } else {
    // General case: we passout both X and W
    X_nnz = sq->X.nnz;
    W_nnz = sq->W.nnz;

    X_rinx = new integer[X_nnz];
    X_cptr = new integer[p + 1];
    X_val = new double[X_nnz];

    W_rinx = new integer[W_nnz];
    W_cptr = new integer[p + 1];
    W_val = new double[W_nnz];

    X_cptr[0] = 0;
    W_cptr[0] = 0;

    integer nnz_count_X = 0;
    integer nnz_count_W = 0;
    double val;

    for (integer j = 0; j < p; j++) {
      // For column j we go
      for (integer k = 0; k < sq->X.ncol[j]; k++) {
        val = sq->X.val[j][k];
        if (fabs(val) > EPS) {
          X_rinx[nnz_count_X] = sq->X.rowind[j][k];
          X_val[nnz_count_X] = val;
          nnz_count_X++;
        }
      }
      X_cptr[j + 1] = nnz_count_X;

      // For column j we go
      for (integer k = 0; k < sq->W.ncol[j]; k++) {
        val = sq->W.val[j][k];
        if (fabs(val) > EPS) {
          W_rinx[nnz_count_W] = sq->W.rowind[j][k];
          W_val[nnz_count_W] = val;
          nnz_count_W++;
        }
      }
      W_cptr[j + 1] = nnz_count_W;
    }

    // The memory footprint is the same but the nnz values in the front of the
    // array.
    X_nnz = nnz_count_X;
    W_nnz = nnz_count_W;

    // Passout out stats
    info_num_iter =
        sq->Stat.opt.size();  // Number of objective values evalutions is the
                              // number of iterations
    info_times[0] = sq->Stat.time_total;
    info_times[1] = sq->Stat.time_cov;

    info_times[2] =
        accumulate(sq->Stat.time_itr.begin(), sq->Stat.time_itr.end(), 0.0);
    info_times[3] =
        accumulate(sq->Stat.time_lns.begin(), sq->Stat.time_lns.end(), 0.0);
    info_times[4] =
        accumulate(sq->Stat.time_inv.begin(), sq->Stat.time_inv.end(), 0.0);
    info_times[5] =
        accumulate(sq->Stat.time_upd.begin(), sq->Stat.time_upd.end(), 0.0);

    for (integer i = 0; i < info_num_iter; ++i) {
      info_objective[i] = sq->Stat.opt[i];
    }
    info_logdetX = sq->Stat.logdetX;
    info_trSX = sq->Stat.trSX;
  }

  sq->DeleteVars();
  sq->FreeSparse(&sq->X);
  sq->FreeSparse(&sq->W);

  delete sq;
};
