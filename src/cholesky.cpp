/* cholesky.cpp
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

#ifndef cholesky_cpp
#define cholesky_cpp

#include "squic.hpp"

#define MAX(A, B) (((A) > (B)) ? (A) : (B))

// #define PRINT_INFO_EXPORT

// The computed W does not satisfy |W - S|< Lambda.  Project it.
// we assume that the nonzero pattern of S covers at least the one of W (or even
// more)
double SQUIC::ProjLogDet(double Lambda, double drop_tol) {
  /*
  S      sparse sample covariance matrix with entries at least at the positions
  of W W      sparse covariance matrix Lambda Lagrangian parameter
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
  for (j = 0; j < p; j++) {
    // logical pointer to column j+1
    ia[j + 1] = ia[j] + W.ncol[j];
    l = 0;
    for (k = 0; k < W.ncol[j]; k++) {
      i = W.rowind[j][k];
      Wij = W.val[j][k];
      ja[cnt] = i;
      // scan S until S_ij is found, S contains the pattern of W
      // both matrices are set up such that their row indices are taken in
      // increasing order
      while (S.rowind[j][l] < i) {
        l++;
      }
      Sij = S.val[j][l];

      tmp = Wij;
      if (RunTimeConfig.off_diagonal) {
        if (i == j) {
          tmp = Sij;
        } else {  // i!=j
          if (Sij - Lambda > tmp) tmp = Sij - Lambda;
          if (Sij + Lambda < tmp) tmp = Sij + Lambda;
        }
      }       // end if RunTimeConfig.off_diagonal
      else {  // NOT RunTimeConfig.off_diagonal
        if (Sij - Lambda > tmp) tmp = Sij - Lambda;
        if (Sij + Lambda < tmp) tmp = Sij + Lambda;
      }  // end if-else RunTimeConfig.off_diagonal

      // project prW_ij
      // since S and W are exactly symmetric, the same result will apply to
      // prW_ji
      a[cnt++] = tmp;
    }  // end for k
  }    // end for j

  // --------------------------------------------------------------
  // compute log(det X) using an external function call from MATLAB
  // --------------------------------------------------------------
  // logdet    MATLAB function
  //           [ld,info]=logdet(A,drop_tol,params) returns log(det(A)) for a
  //           given sparse symmetric positive definite matrix A with up to some
  //           accuracy drop_tol using some internal parameters
  // A

  double *D_LDL = new double[p];
  integer err;

  // generic LDL driver in CSC format
  err = FactorizeLdl(ja, ia, a, W.nc, W.nr, W.nnz, NULL, NULL, D_LDL, drop_tol);
  // printf("err=%ld\n",err); fflush(stdout);

  logdet = 0.0;
  if (!err) {
    for (integer i = 0; i < p; ++i) {
      logdet += log(D_LDL[i]);
    }
  } else {
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

  // printf("logdet : %12.4le \n", logdet);

  delete[] ia;
  delete[] ja;
  delete[] a;

  delete[] D_LDL;

  return logdet;
};  // end projLogDet

// integer SQUIC::LogDet(double *D, double *ld, double drop_tol)
bool SQUIC::LogDet(double *ld, double drop_tol) {
  double D[p];

  integer err = FactorizeLdl(&X, &L, &P, D, drop_tol);

  *ld = 0;
  if (!err) {
    for (integer i = 0; i < p; ++i) {
      ld[0] += log(D[i]);
    }
    return false;
  } else {
    ld[0] = 1e+15;
    return true;
  }

};  // end logDet

// generic entry point for LDL factorization if the matrix is given in CSC
// format if successful, return the Cholesky factor along with the reordering in
// L_factor, D_LDL and L_perm, where L_factor is unit lower triangular, D_LDL is
// the diagonal part s.t. A(perm,perm)=LDL'
integer SQUIC::FactorizeLdl(integer *A_rowind, integer *A_p, double *A_val,
                            integer nc, integer nr, integer nnz,
                            SparseMatrix *L_factor, SparseMatrix *L_perm,
                            double *D_LDL, double drop_tol) {
  integer i, j, k, l;

  // check diagonal part in advance
  for (j = 0; j < nc; j++) {
    l = A_p[j + 1];
    for (k = A_p[j]; k < l; k++) {
      i = A_rowind[k];
      if (i >= j) break;
    }  // end for k
    // did we find the diagonal entry?
    // no, no diagonal part or sub-diagonal part available
    if (k >= l) return 1;
    // no, at least the diagonal entry is missing
    if (i > j) return 1;
    // yes, but zero or negative diagonal entry
    if (A_val[k] <= 0.0) return 1;
  }  // end for j

  // (CHOL_SUITESPARSE==RunTimeConfig.factorization)
  // ---- deprecated CHOLMOD driver ----
  // FactorizeCholmod(ja , ia, a, W.nc, W.nr, W.nnz , D_LDL);
  // generic CHOLMOD driver
  return FactorizeCholmod(A_rowind, A_p, A_val, nc, nr, nnz, L_factor, L_perm,
                          D_LDL);
};  // end FactorizeLDL

// generic entry point for LDL factorization if the matrix is given in
// SparseMatrix format if successful, return the Cholesky factor along with the
// reordering in L_factor, D_LDL and L_perm, where L_factor is unit lower
// triangular, D_LDL is the diagonal part s.t. A(perm,perm)=LDL'
integer SQUIC::FactorizeLdl(SparseMatrix *A, SparseMatrix *L_factor,
                            SparseMatrix *L_perm, double *D_LDL,
                            double drop_tol) {
  integer i, j, k, l, *pi;

  // check diagonal part in advance
  for (j = 0; j < A->nc; j++) {
    l = A->ncol[j];
    pi = A->rowind[j];
    for (k = 0; k < l; k++) {
      i = pi[k];
      if (i >= j) break;
    }  // end for k
    // did we find the diagonal entry?
    // no, no diagonal part or sub-diagonal part available
    if (k >= l) return 1;
    // no, at least the diagonal entry is missing
    if (i > j) return 1;
    // yes, but zero or negative diagonal entry
    if (A->val[j][k] <= 0.0) return 1;
  }  // end for j

  // Copy data to contiguous structure traditional CSC format for CHOLMOD
  integer *A_p = new integer[A->nc + 1];
  integer *A_rowind = new integer[A->nnz];
  double *A_val = new double[A->nnz];

  integer counter = 0;

  for (integer i = 0; i < A->nc; ++i) {
    A_p[i] = counter;
    for (integer j = 0; j < A->ncol[i]; ++j) {
      A_val[counter] = A->val[i][j];
      A_rowind[counter] = A->rowind[i][j];
      counter++;
    }
  }
  A_p[A->nc] = (A->nnz);

  integer rtrn;
  // (CHOL_SUITESPARSE==RunTimeConfig.factorization)
  rtrn = FactorizeCholmod(A_rowind, A_p, A_val, A->nc, A->nr, A->nnz, L_factor,
                          L_perm, D_LDL);

  delete[] A_p;
  delete[] A_rowind;
  delete[] A_val;

  return rtrn;

};  // end FactorizeLDL

// export LDL factorization generated from a Cholesky factorization along with
// the reordering where L_factor is unit lower triangular, D_LDL is the diagonal
// part s.t. A(perm,perm)=LDL'
void SQUIC::ExportLdl(double *D_LDL) {
  // (CHOL_SUITESPARSE==RunTimeConfig.factorization)
  ExportCholmod(D_LDL);
};  // end Export_LDL

// export block LDL factorization generated from a Cholesky factorization along
// with the reordering where L_factor is unit lower triangular, D_LDL is the
// diagonal part s.t. A(perm,perm)=LDL'
void SQUIC::BlockExportLdl() {
  // (CHOL_SUITESPARSE==RunTimeConfig.factorization)
  BlockExportCholmod();
};  // end BlockExport_LDL

// CHOLMOD entry point if the matrix is given in CSC format
// if successful, return the Cholesky factor along with the reordering in
// L_factor, D_LDL and L_perm, where L_factor is unit lower triangular, D_LDL is
// the diagonal part s.t. A(perm,perm)=LDL'
integer SQUIC::FactorizeCholmod(integer *A_rowind, integer *A_p, double *A_val,
                                integer nc, integer nr, integer nnz,
                                SparseMatrix *L_factor, SparseMatrix *L_perm,
                                double *D_LDL) {
#pragma omp critical
  {
    // Initialize CHOLMOD sparse matrix in CSC format
    Cholmod.pA->ncol = nc;
    Cholmod.pA->nrow = nr;
    Cholmod.pA->nzmax = nnz;

    Cholmod.pA->p = &A_p[0];
    Cholmod.pA->i = &A_rowind[0];
    Cholmod.pA->x = &A_val[0];
    Cholmod.pA->itype = CHOLMOD_INTGER_TYPE;

    // printf ("norm (A,1)   = %g\n", CHOLMOD_NORM_SPARSE(Cholmod.pA, 1,
    // &Cholmod.common_parameters)) ; printf ("nz: %ld\n",
    // CHOLMOD_NNZ(Cholmod.pA, &Cholmod.common_parameters));

    // release memory of the previous numerical factorization
    if (factorization_done) {
      if (RunTimeConfig.verbose > 2) {
        MSG("- Free existing numerical Cholesky decomposition\n");
        fflush(stdout);
      }
      CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
      Cholmod.pL = NULL;
      // the numerical factorization is given away
      factorization_done = 0;
    }  // end if

    // symbolic analysis has been performed previously
    if (analyze_done && L_perm != NULL) {
      if (RunTimeConfig.verbose > 2) {
        MSG("- Cholesky decomposition WITHOUT symbolic analysis\n");
        fflush(stdout);
      }
      // Store Permutation
      integer *perm = new integer[Cholmod.pA->ncol];
      integer *f = new integer[Cholmod.pA->ncol];
      for (integer i = 0; i < Cholmod.pA->ncol; ++i) {
        perm[i] = integer(L_perm->rowind[i][0]);
        f[i] = i;
      }

      Cholmod.common_parameters.method[0].ordering = CHOLMOD_GIVEN;
      Cholmod.pL = CHOLMOD_ANALYZE_P(Cholmod.pA, perm, f, Cholmod.pA->ncol,
                                     &Cholmod.common_parameters);
      // CHOLMOD_CHANGE_FACTOR(CHOLMOD_REAL, 1, 0, 0, 1, Cholmod.pL,
      // &Cholmod.common_parameters);

      double beta[2];
      beta[0] = 0;
      beta[1] = 0;

      CHOLMOD_FACTORIZE_P(Cholmod.pA, beta, f, Cholmod.pA->ncol, Cholmod.pL,
                          &Cholmod.common_parameters);

      delete[] perm;
      delete[] f;
    } else {  // symbolic analysis is not yet available

      if (RunTimeConfig.verbose > 2) {
        MSG("- Cholesky decomposition INCLUDING symbolic analysis\n");
        fflush(stdout);
      }

      Cholmod.common_parameters.method[0].ordering = CHOLMOD_AMD;
      Cholmod.pL = CHOLMOD_ANALYZE(Cholmod.pA, &Cholmod.common_parameters);
      // // CHOLMOD_CHANGE_FACTOR(CHOLMOD_REAL, 1, 0, 0, 1, Cholmod.pL,
      // &Cholmod.common_parameters);
      // CHOLMOD_PRINT_SPARSE(Cholmod.pA,"Cholmod.pA",&Cholmod.common_parameters);
      // CHOLMOD_PRINT_FACTOR(Cholmod.pL,"Cholmod.pL
      // pre",&Cholmod.common_parameters);
      CHOLMOD_FACTORIZE(Cholmod.pA, Cholmod.pL, &Cholmod.common_parameters);
      // CHOLMOD_PRINT_FACTOR(Cholmod.pL,"Cholmod.pL
      // post",&Cholmod.common_parameters);
    }
    if (RunTimeConfig.verbose > 2) {
      // CHOLMOD_PRINT_COMMON("Cholmod.common_parameters",
      // &Cholmod.common_parameters);

      if (Cholmod.common_parameters.status == CHOLMOD_OK) {
        if (Cholmod.pL->is_super) {
          printf("    number of supernodes: %d\n", Cholmod.pL->nsuper);
          integer maxcols = 0, maxsize = 0;
          integer *Super = (integer *)Cholmod.pL->super;
          integer *Lpi = (integer *)Cholmod.pL->pi;
          for (integer s = 0; s < Cholmod.pL->nsuper; s++) {
            // number of columns in a supernode panel
            integer nscols = Super[s + 1] - Super[s];
            // number of rows in a supernode panel
            integer nsrow = Lpi[s + 1] - Lpi[s];
            // number of columns in a supernode panel
            maxcols = MAX(maxcols, nscols);
            maxsize = MAX(maxsize, nsrow * nscols);
          }  // end for s
          printf("    maximum number of columns in a supernode: %d\n", maxcols);
          printf("    maximum size of a supernode: %d\n\n", maxsize);
        }  // end if
      }    // end if
      fflush(stdout);
    }  // end if

    bool is_pos_def = (Cholmod.common_parameters.status == CHOLMOD_OK);

    // export the permutation vector if requested by the calling routine
    // otherwise the analysis phase is ignored
    if (!analyze_done && L_perm != NULL) {
      // Store Permutation
      for (integer i = 0; i < Cholmod.pL->n; ++i) {
        L_perm->rowind[i][0] = integer(((integer *)Cholmod.pL->Perm)[i]);
      }

      analyze_done = 1;
    }

    // Cholesky factor complete
    if (is_pos_def) {
      // the numerical factorization has been successfully computed
      factorization_done = 1;

      if (!Cholmod.pL->is_super) {
        // export diagonal part of LDL
        for (integer j = 0; j < Cholmod.pL->n; j++) {
          integer nz = ((integer *)Cholmod.pL->nz)[j];
          integer p = ((integer *)Cholmod.pL->p)[j];

          for (integer i = 0; i < nz; ++i) {
            integer rowind = ((integer *)Cholmod.pL->i)[p + i];
            double val_temp = ((double *)Cholmod.pL->x)[p + i];

            // Store all the diagonal entries of L
            if (rowind == j) {
              D_LDL[j] = val_temp;
              break;
            }  // end if
          }    // end for
        }      // end for j

        // D_LDL = diag(L)^2
        for (integer i = 0; i < p; ++i) {
          D_LDL[i] = D_LDL[i] * D_LDL[i];
        }  // end for i
      }    // end if !is_super
      else {
        integer nblocks = Cholmod.pL->nsuper;

        // printf("number of supernodes %d\n",nblocks); fflush(stdout);

        integer cnt = 0;
        integer nscol;
        integer *Super = (integer *)Cholmod.pL->super;
        // cout << "Super: " << Super << endl; fflush(stdout);
        integer *Lpi = (integer *)Cholmod.pL->pi;
        // cout << "Lpi: " << Lpi << endl; fflush(stdout);
        integer *Lpx = (integer *)Cholmod.pL->px;
        // cout << "Lpx: " << Lpx << endl; fflush(stdout);
        integer *Ls = (integer *)Cholmod.pL->s;
        // cout << "Ls: " << Ls << endl; fflush(stdout);
        double *Lx = (double *)Cholmod.pL->x;
        // cout << "Lx: " << Lx << endl; fflush(stdout);

        for (integer s = 0; s < nblocks; s++) {
          // start of supernode panel s
          integer k1 = Super[s];
          // printf("k1=%d\n",k1); fflush(stdout);
          // start of supernode panel s+1
          integer k2 = Super[s + 1];
          // printf("k2=%d\n",k2); fflush(stdout);
          // start indices of supernode panel s relative to pL->s
          integer psi = Lpi[s];
          // printf("psi=%d\n",psi); fflush(stdout);
          // start indices of supernode panel s+1 relative to pL->s
          integer psend = Lpi[s + 1];
          // printf("psend=%d\n",psend); fflush(stdout);
          // start numerical values of supernode panel s relative to pL->x
          integer psx = Lpx[s];
          // printf("psx=%d\n",psx); fflush(stdout);
          // number of rows in a supernode panel
          integer nsrow = psend - psi;
          // printf("nsrow=%d\n",nsrow); fflush(stdout);
          // number of columns in a supernode panel
          nscol = k2 - k1;
          // printf("nscol=%d\n",nscol); fflush(stdout);
          // number of rows in the sub-diagonal part of a supernode panel
#ifdef PRINT_INFO_EXPORT
          printf("supernode %d\n", s + 1);
          fflush(stdout);
          printf("   number of columns %d\n", nscol);
          fflush(stdout);
          printf("   number of rows %d\n", nsrow);
          fflush(stdout);
#endif

          // extract diagonal part to D_LDL
          integer j = 0;
          for (integer i = 0; i < nscol; i++, j += nsrow + 1) {
            D_LDL[cnt + i] = Lx[psx + j] * Lx[psx + j];
            // cout << "D_LDL[" << cnt+i+1<< "]: " << D_LDL[cnt+i] << endl;
            // fflush(stdout);
          }  // end for i
          cnt += nscol;
        }  // end for s
      }    // end if-else !is_super
    }      // if is_pos_def

    if (!is_pos_def || L_perm == NULL || L_factor == NULL) {
      // release memory of the factorization
      if (RunTimeConfig.verbose > 2) {
        MSG("- Discard numerical Cholesky decomposition\n");
        fflush(stdout);
      }
      CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
      Cholmod.pL = NULL;
      // the numerical factorization is given away
      factorization_done = 0;
    }  // end if

  }  // pragma omp critical

  return Cholmod.common_parameters.status != CHOLMOD_OK;
};  // end Factorize_CHOLMOD

// CHOLMOD entry point
// if successful, return the Cholesky factor along with the reordering in
// L_factor, D_LDL and L_perm, where L_factor is unit lower triangular, D_LDL is
// the diagonal part s.t. A(perm,perm)=LDL'
void SQUIC::ExportCholmod(double *D_LDL) {
  if (RunTimeConfig.verbose > 2) {
    MSG("- Export Cholesky factorization\n");
    fflush(stdout);
  }
#ifdef PRINT_MSG
  CHOLMOD_PRINT_FACTOR(Cholmod.pL, "L", &Cholmod.common_parameters);
#endif

  // export the permutation vector if requested by the calling routine
  // otherwise the analysis phase is ignored
  if (!analyze_done) {
    // Store Permutation
    for (integer i = 0; i < Cholmod.pL->n; ++i) {
      P.rowind[i][0] = integer(((integer *)Cholmod.pL->Perm)[i]);
    }  // end for i

  }  // end if

  // Re-allocate and store L_factor
  ClearSparse(&L);
  L.nr = L.nc = Cholmod.pL->n;

  // For some reason this is NOT the nnz of the Factor ....
  // L_factor->nnz = Cholmod.pL->nzmax;
  // Thus we manually count the nnz, using index

  L.ncol = (integer *)realloc(L.ncol, (size_t)p * sizeof(integer));
  L.rowind = (integer **)realloc(L.rowind, (size_t)p * sizeof(integer *));
  L.val = (double **)realloc(L.val, (size_t)p * sizeof(double *));

  // CHOLMOD did not use supernodal format
  if (!Cholmod.pL->is_super) {
    // (partially) export Cholesky factor in SparseMatrix format as well as the
    // diagonal part of LDL
    integer index = 0;
    for (integer j = 0; j < Cholmod.pL->n; j++) {
      integer nz = ((integer *)Cholmod.pL->nz)[j];
      integer p = ((integer *)Cholmod.pL->p)[j];

      L.ncol[j] = nz;
      L.rowind[j] =
          (integer *)realloc(L.rowind[j], (size_t)nz * sizeof(integer));
      L.val[j] = (double *)realloc(L.val[j], (size_t)nz * sizeof(double));

      for (integer i = 0; i < nz; ++i) {
        integer rowind = ((integer *)Cholmod.pL->i)[p + i];
        double val_temp = ((double *)Cholmod.pL->x)[p + i];

        // Store all the diagonal entries of L
        if (rowind == j) {
          D_LDL[j] = val_temp;
        }

        L.rowind[j][i] = rowind;
        // Normalize L with diagonal
        L.val[j][i] = val_temp / D_LDL[j];

        index++;
      }
    }  // end for j

    L.nnz = index;

    // D_LDL = diag(L)^2
    for (integer i = 0; i < p; ++i) {
      D_LDL[i] = D_LDL[i] * D_LDL[i];
    }     // end for i
  }       // end if !is_super
  else {  // is_super
    integer nblocks = Cholmod.pL->nsuper;

    // printf("number of supernodes %d\n",nblocks); fflush(stdout);

    integer index = 0;
    integer cnt = 0;
    integer nscol;
    integer *Super = (integer *)Cholmod.pL->super;
    // cout << "Super: " << Super << endl; fflush(stdout);
    integer *Lpi = (integer *)Cholmod.pL->pi;
    // cout << "Lpi: " << Lpi << endl; fflush(stdout);
    integer *Lpx = (integer *)Cholmod.pL->px;
    // cout << "Lpx: " << Lpx << endl; fflush(stdout);
    integer *Ls = (integer *)Cholmod.pL->s;
    // cout << "Ls: " << Ls << endl; fflush(stdout);
    double *Lx = (double *)Cholmod.pL->x;
    // cout << "Lx: " << Lx << endl; fflush(stdout);
    for (integer s = 0; s < nblocks; s++) {
      // start of supernode panel s
      integer k1 = Super[s];
      // printf("k1=%d\n",k1); fflush(stdout);
      // start of supernode panel s+1
      integer k2 = Super[s + 1];
      // printf("k2=%d\n",k2); fflush(stdout);
      // start indices of supernode panel s relative to pL->s
      integer psi = Lpi[s];
      // printf("psi=%d\n",psi); fflush(stdout);
      // start indices of supernode panel s+1 relative to pL->s
      integer psend = Lpi[s + 1];
      // printf("psend=%d\n",psend); fflush(stdout);
      // printf("k2=%d\n",k2); fflush(stdout);
      // start numerical values of supernode panel s relative to pL->x
      integer psx = Lpx[s];
      // printf("psx=%d\n",psx); fflush(stdout);
      // number of rows in a supernode panel
      integer nsrow = psend - psi;
      // printf("nsrow=%d\n",nsrow); fflush(stdout);
      // number of columns in a supernode panel
      nscol = k2 - k1;
      // printf("nscol=%d\n",nscol); fflush(stdout);
      // number of rows in the sub-diagonal part of a supernode panel
#ifdef PRINT_INFO_EXPORT
      printf("supernode %d\n", s + 1);
      fflush(stdout);
      printf("   number of columns %d\n", nscol);
      fflush(stdout);
      printf("   number of rows %d\n", nsrow);
      fflush(stdout);
#endif

      integer j = 0;
      for (integer i = 0; i < nscol; i++, j += nsrow + 1) {
        // extract (squared) diagonal part to D_LDL
        double val_temp = Lx[psx + j];
        D_LDL[cnt + i] = val_temp * val_temp;
        // cout << "D_LDL[" << cnt+i+1<< "]: " << D_LDL[cnt+i] << endl;
        // fflush(stdout);

        L.ncol[cnt + i] = nsrow - i;
        L.rowind[cnt + i] = (integer *)realloc(
            L.rowind[cnt + i], (size_t)(nsrow - i) * sizeof(integer));
        L.val[cnt + i] = (double *)realloc(
            L.val[cnt + i], (size_t)(nsrow - i) * sizeof(double));

        for (integer k = i; k < nsrow; ++k) {
          L.rowind[cnt + i][k - i] = Ls[psi + k];
          // Normalize L with diagonal
          L.val[cnt + i][k - i] = Lx[psx + k + i * nsrow] / val_temp;
          index++;
        }  // end for k
      }    // end for i
      cnt += nscol;
    }  // end for s

    // Set factor nnz
    L.nnz = index;

    /*
printf("permutation vector\n");
for (integer i = 0; i < p; ++i)
printf("%8ld",P.rowind[i][0]+1);
printf("\n"); fflush(stdout);
for (integer i=0; i<p; i++)
printf("%8.1le",D_LDL[i]);
printf("\n"); fflush(stdout);
PrintSparse(&L);
*/
  }  // end if-else !is_super

  // release memory of the factorization
  if (RunTimeConfig.verbose > 2) {
    MSG("- Discard numerical Cholesky decomposition\n");
    fflush(stdout);
  }
  CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
  Cholmod.pL = NULL;
  // the numerical factorization is given away
  factorization_done = 0;

};  // end Export_CHOLMOD

#include <omp.h>

// CHOLMOD entry point
// if successful, return the Cholesky factor along with the reordering in
// L_factor, D_LDL and L_perm, where L_factor is BLOCK unit lower triangular,
// BiD is inverse BLOCK diagonal part s.t. A(perm,perm)=LDL'
void SQUIC::BlockExportCholmod() {
  double time = omp_get_wtime();

  if (RunTimeConfig.verbose > 10) {
    MSG("- Export Cholesky block factorization\n");
    fflush(stdout);
  }

  // export the permutation vector if requested by the calling routine
  // otherwise the analysis phase is ignored
  if (!analyze_done) {
    // Store Permutation
    for (integer i = 0; i < Cholmod.pL->n; ++i) {
      P.rowind[i][0] = integer(((integer *)Cholmod.pL->Perm)[i]);
    }  // end for i

  }  // end if

  integer nnzBL = 0, nnzBiD = 0;
  // CHOLMOD did not use supernodal format
  if (!Cholmod.pL->is_super) {
    // printf("Cholmod.pL->is_super == FALSE \n");
    // fflush(stdout);

    // no, everything is stored in simplicial format
    // (partially) export Cholesky factor in SparseMatrix format as well as the
    // diagonal part of LDL
    integer nblocks = Cholmod.pL->n;
    BiD.nblocks = nblocks;
    BL.nblocks = nblocks;

    integer *Lnz = (integer *)Cholmod.pL->nz;
    // cout << "Lnz: " << Lnz << endl; fflush(stdout);
    integer *Lp = (integer *)Cholmod.pL->p;
    // cout << "Lp: " << Lp << endl; fflush(stdout);
    integer *Li = (integer *)Cholmod.pL->i;
    // cout << "Li: " << Li << endl; fflush(stdout);
    double *Lx = (double *)Cholmod.pL->x;
    // cout << "Lx: " << Lx << endl; fflush(stdout);
    for (integer s = 0; s < nblocks; s++) {
      integer nz = Lnz[s];
      // cout << "nz: " << nz << endl; fflush(stdout);
      integer nz2 = nz - 1;
      // cout << "nz2: " << nz2 << endl; fflush(stdout);
      integer p = Lp[s];
      // cout << "p: " << p << endl; fflush(stdout);
      integer p2 = p + 1;
      // cout << "p2: " << p2 << endl; fflush(stdout);

      // number of columns in block s
      BiD.nblockcol[s] = 1;

      // (re-)allocate memory for column indices
      if (BiD.colind[s] != NULL)
        BiD.colind[s] =
            (integer *)realloc(BiD.colind[s], (size_t)1 * sizeof(integer));
      else
        BiD.colind[s] = (integer *)malloc((size_t)1 * sizeof(integer));

      // (re-)allocate memory for numerical values in the diagonal block
      if (BiD.valD[s] != NULL)
        BiD.valD[s] =
            (double *)realloc(BiD.valD[s], (size_t)1 * 1 * sizeof(double));
      else
        BiD.valD[s] = (double *)malloc((size_t)1 * 1 * sizeof(double));

      // insert associated column indices
      // start position column indices in block s
      BiD.colind[s][0] = s;

      // copy numerical values row-by-row
      // start position numerical values of the block-diagonal part in block s
      BiD.valD[s][0] = 1.0 / (Lx[p] * Lx[p]);
      nnzBiD++;

      // number of columns in block s
      BL.nblockcol[s] = 1;
      // number of sub-diagonal rows in block s
      BL.nblockrow[s] = nz2;

      // (re-)allocate memory for column indices
      if (BL.colind[s] != NULL)
        BL.colind[s] =
            (integer *)realloc(BL.colind[s], (size_t)1 * sizeof(integer));
      else
        BL.colind[s] = (integer *)malloc((size_t)1 * sizeof(integer));

      // (re-)allocate memory for sub-diagonal row indices
      if (BL.rowind[s] != NULL)
        BL.rowind[s] =
            (integer *)realloc(BL.rowind[s], (size_t)nz2 * sizeof(integer));
      else
        BL.rowind[s] = (integer *)malloc((size_t)nz2 * sizeof(integer));

      // (re-)allocate memory for numerical values in the sub-diagonal block
      if (BL.valE[s] != NULL)
        BL.valE[s] =
            (double *)realloc(BL.valE[s], (size_t)nz2 * 1 * sizeof(double));
      else
        BL.valE[s] = (double *)malloc((size_t)nz2 * 1 * sizeof(double));
      nnzBL += nz2;

      // insert associated column indices
      BL.colind[s][0] = s;

      if (Cholmod.pL->is_monotonic) {
        // insert associated sub-block-diagonal row indices and values
        // also copy numerical values row-by-row
        // start position row indices in block s
        integer *pi = BL.rowind[s];
        // start position numerical values of the sub-block-diagonal part in
        // block s
        double *pval = BL.valE[s];
        double dgl = Lx[p];
        for (integer i = 0; i < nz2; i++) {
          pi[i] = Li[p2 + i];
          // BLAS dcopy Lx -> pval
          *pval++ = Lx[p2 + i] / dgl;
        }       // end for i
      } else {  // indices are not sorted in increasing order
        // extract diagonal entry first
        double dgl;
        for (integer i = 0; i < nz; i++)
          if (Li[p + i] == s) {
            dgl = Lx[p + i];
            break;
          }
        // insert associated sub-block-diagonal row indices and values
        // also copy numerical values row-by-row
        // start position row indices in block s
        integer *pi = BL.rowind[s];
        // start position numerical values of the sub-block-diagonal part in
        // block s
        double *pval = BL.valE[s];
        for (integer i = 0; i < nz; i++)
          if (Li[p + i] != s) {
            pi[i] = Li[p + i];
            *pval++ = Lx[p + i] / dgl;
          }
      }  // end if-else

    }  // end for s
  } else {
    // printf("Cholmod.pL->is_super == TURE \n");
    // fflush(stdout);

    // (partially) export Cholesky factor in SparseMatrix format as well as the
    // diagonal part of LDL
    integer nblocks = Cholmod.pL->nsuper;
    //#ifdef PRINT_INFO_EXPORT
    // printf("number of supernodes %d\n", nblocks);
    // fflush(stdout);
    //#endif
    BiD.nblocks = nblocks;
    BL.nblocks = nblocks;

    // cout << "is_ll: " << Cholmod.pL->is_ll << endl; fflush(stdout);
    // cout << "is_super: " << Cholmod.pL->is_super << endl; fflush(stdout);
    // cout << "is_monotonic: " << Cholmod.pL->is_monotonic << endl;
    // fflush(stdout);

    integer cnt = 0;
    integer nscol;
    integer *Super = (integer *)Cholmod.pL->super;
    // cout << "Super: " << Super << endl; fflush(stdout);
    integer *Lpi = (integer *)Cholmod.pL->pi;
    // cout << "Lpi: " << Lpi << endl; fflush(stdout);
    integer *Lpx = (integer *)Cholmod.pL->px;
    // cout << "Lpx: " << Lpx << endl; fflush(stdout);
    integer *Ls = (integer *)Cholmod.pL->s;
    // cout << "Ls: " << Ls << endl; fflush(stdout);
    double *Lx = (double *)Cholmod.pL->x;
    // cout << "Lx: " << Lx << endl; fflush(stdout);
    for (integer s = 0; s < nblocks; s++) {
      // start of supernode panel s
      integer k1 = Super[s];
      // printf("k1=%d\n",k1); fflush(stdout);
      // start of supernode panel s+1
      integer k2 = Super[s + 1];
      // printf("k2=%d\n",k2); fflush(stdout);
      // start indices of supernode panel s relative to pL->s
      integer psi = Lpi[s];
      // printf("psi=%d\n",psi); fflush(stdout);
      // start indices of supernode panel s+1 relative to pL->s
      integer psend = Lpi[s + 1];
      // printf("psend=%d\n",psend); fflush(stdout);
      // start numberical values of supernode panel s relative to pL->x
      integer psx = Lpx[s];
      // printf("psx=%d\n",psx); fflush(stdout);
      // number of rows in a supernode panel
      integer nsrow = psend - psi;
      // printf("nsrow=%d\n",nsrow); fflush(stdout);
      // number of columns in a supernode panel
      nscol = k2 - k1;
      // printf("nscol=%d\n",nscol); fflush(stdout);
      // number of rows in the sub-diagonal part of a supernode panel
      integer nsrow2 = nsrow - nscol;
      // printf("nsrow2=%d\n",nsrow2); fflush(stdout);
      // starting point of the strict lower triangular part in the supernode
      // panel relative to L->s, L->x
      integer ps2 = psi + nscol;
      // printf("ps2=%d\n",ps2); fflush(stdout);
#ifdef PRINT_INFO_EXPORT
      printf("supernode %d\n", s + 1);
      fflush(stdout);
      printf("   number of columns %d\n", nscol);
      fflush(stdout);
      printf("   number of rows %d\n", nsrow);
      fflush(stdout);
#endif

      // number of columns in block s
      BiD.nblockcol[s] = nscol;

      // (re-)allocate memory for column indices
      if (BiD.colind[s] != NULL)
        BiD.colind[s] =
            (integer *)realloc(BiD.colind[s], (size_t)nscol * sizeof(integer));
      else
        BiD.colind[s] = (integer *)malloc((size_t)nscol * sizeof(integer));

      // (re-)allocate memory for numerical values in the diagonal block
      if (BiD.valD[s] != NULL)
        BiD.valD[s] = (double *)realloc(BiD.valD[s],
                                        (size_t)nscol * nscol * sizeof(double));
      else
        BiD.valD[s] = (double *)malloc((size_t)nscol * nscol * sizeof(double));

      // insert associated column indices
      // start position column indices in block s
      integer *pi = BiD.colind[s];
      for (integer i = 0; i < nscol; i++) pi[i] = cnt + i;

      // copy numerical values row-by-row
      // start position numerical values of the block-diagonal part in block s
      double *pval = BiD.valD[s];
      for (integer i = 0; i < nscol; i++, pval++) {
        // BLAS dcopy Lx -> pval
#ifdef _BLAS_LAPACK_32_
        int mynscol = nscol, mynsrow = nsrow;
        dcopy_(&mynscol, Lx + psx + i, &mynsrow, pval, &mynscol);
#else
        dcopy_(&nscol, Lx + psx + i, &nsrow, pval, &nscol);
#endif
      }  // end for i

      // invert diagonal block
      pval = BiD.valD[s];
      integer info;
#ifdef _BLAS_LAPACK_32_
      int mynscol = nscol, myinfo = info;
      dpotri_("l", &mynscol, pval, &mynscol, &myinfo, 1);
      info = myinfo;
#else
      dpotri_("l", &nscol, pval, &nscol, &info, 1);
#endif
      if (info) {
        if (info < 0)
          printf("LAPACK's dpotri_: %ld-th argument had an illegal value\n",
                 -info);
        else
          printf(
              "LAPACK's dpotri_ routine encountered zero column in step %ld\n",
              info);
      }

      // copy strict lower triangular part to strict upper triangular part
      double *pb = pval + 1;
      integer mm = 1;
      for (integer ii = 1; ii < nscol; pb++, ii++) {
        double *pD = pval + nscol * ii;
#ifdef _BLAS_LAPACK_32_
        int myii = ii, mynscol = nscol, mymm = mm;
        dcopy_(&myii, pb, &mynscol, pD, &mymm);
#else
        dcopy_(&ii, pb, &nscol, pD, &mm);
#endif
      }  // end for ii
      nnzBiD += nscol * nscol;

      // number of columns in block s
      BL.nblockcol[s] = nscol;
      // number of sub-diagonal rows in block s
      BL.nblockrow[s] = nsrow2;

      // (re-)allocate memory for column indices
      if (BL.colind[s] != NULL)
        BL.colind[s] =
            (integer *)realloc(BL.colind[s], (size_t)nscol * sizeof(integer));
      else
        BL.colind[s] = (integer *)malloc((size_t)nscol * sizeof(integer));

      // (re-)allocate memory for sub-diagonal row indices
      if (BL.rowind[s] != NULL)
        BL.rowind[s] =
            (integer *)realloc(BL.rowind[s], (size_t)nsrow2 * sizeof(integer));
      else
        BL.rowind[s] = (integer *)malloc((size_t)nsrow2 * sizeof(integer));

      // (re-)allocate memory for numerical values in the sub-diagonal block
      if (BL.valE[s] != NULL)
        BL.valE[s] = (double *)realloc(BL.valE[s],
                                       (size_t)nsrow2 * nscol * sizeof(double));
      else
        BL.valE[s] = (double *)malloc((size_t)nsrow2 * nscol * sizeof(double));

      // insert associated column indices
      // start position column indices in block s
      pi = BL.colind[s];
      for (integer i = 0; i < nscol; i++) pi[i] = cnt + i;

      // insert associated sub-block-diagonal row indices and values
      // also copy numerical values row-by-row
      // start position row indices in block s
      pi = BL.rowind[s];
      // copy entries initially to a buffer in order to efficiently use GEMM
      // later
      pval = BL.valE[s];
      for (integer i = 0; i < nsrow2; i++, pval++) {
        pi[i] = Ls[ps2 + i];
        // BLAS dcopy Lx -> pval
#ifdef _BLAS_LAPACK_32_
        int mynscol = nscol, mynsrow = nsrow, mynsrow2 = nsrow2;
        dcopy_(&mynscol, Lx + psx + nscol + i, &mynsrow, pval, &mynsrow2);
#else
        dcopy_(&nscol, Lx + psx + nscol + i, &nsrow, pval, &nsrow2);
#endif
      }  // end for i

      // valE <- valE*valD^{-1}
      if (nsrow2 > 0) {
        double alpha = 1.0;
#ifdef _BLAS_LAPACK_32_
        int mynsrow2 = nsrow2, mynscol = nscol, mynsrow = nsrow;
        dtrsm_("r", "l", "n", "n", &mynsrow2, &mynscol, &alpha, Lx + psx,
               &mynsrow, BL.valE[s], &mynsrow2, 1, 1, 1, 1);
#else
        dtrsm_("r", "l", "n", "n", &nsrow2, &nscol, &alpha, Lx + psx, &nsrow,
               BL.valE[s], &nsrow2, 1, 1, 1, 1);
#endif
      }  // end if
      nnzBL += nsrow2 * nscol;

      /*
// extract diagonal part to D_LDL
integer j=0;
for (integer i=0; i<nscol; i++,j+=nsrow+1) {
D_LDL[cnt+i]=Lx[psx+j]*Lx[psx+j];
// cout << "D_LDL[" << cnt+i+1<< "]: " << D_LDL[cnt+i] << endl; fflush(stdout);
} // end for i
*/
      cnt += nscol;
    }  // end for s
  }    // end if-else is_super
  BiD.nnz = nnzBiD;
  BL.nnz = nnzBL;

  /*
printf("exported BL\n");
PrintBlockSparse(BL);
printf("exported BiD\n");
PrintBlockSparse(BiD);
printf("\n");
*/

  // release memory of the factorization
  if (RunTimeConfig.verbose > 10) {
    MSG("- Discard numerical Cholesky decomposition\n");
    fflush(stdout);
  }
  CHOLMOD_FREE_FACTOR(&Cholmod.pL, &Cholmod.common_parameters);
  Cholmod.pL = NULL;
  // the numerical factorization is given away
  factorization_done = 0;

  time = omp_get_wtime() - time;

  // MSG("- time =%f \n", time);
  // fflush(stdout);

};  // end BlockExport_CHOLMOD

#endif
