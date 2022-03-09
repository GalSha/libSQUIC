// core randomized active set routine after optionally providing initial guesses
// for X and W
void SQUIC::RunCore(double lambda, double drop_tol, int max_iter,
                    double term_tol) {
  // init runtime variables
  InitVars();
  // Statistics for output
  int nn;

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
  }

  if (RunTimeConfig.verbose == 0) {
    MSG("#SQUIC Version %.2f : p=%g n=%g lambda=%g max_iter=%g term_tol=%g "
        "drop_tol=%g ",
        SQUIC_VER, double(p), double(n), double(lambda), double(max_iter),
        double(term_tol), double(drop_tol));
    fflush(stdout);
  }

  double timeBegin = omp_get_wtime();

  integer max_newton_iter = max_iter;
  integer line_search_iter_max = LINE_SEARCH_ITER_MAX;

  double coord_dec_sweep_tol = 0.05;

  // integer i, j, k, l, m, r, s, t;
  integer *idx, *idxpos;
  double *val, Wij, Sij, Xij, Dij, g;
  double temp_double;

  integer ierr;
  double *pr;

  // tolerance used to compute log(det(X)) and X^{-1} approximately
  double current_drop_tol = MAX(DROP_TOL0, drop_tol);

  // parallel version requires omp_get_max_threads() local versions of idx,
  // idxpos to avoid memory conflicts between the threads
  integer k = omp_get_max_threads();

  // used as stack and list of indices
  idx = (integer *)malloc((size_t)k * p * sizeof(integer));

  // used as check mark array for nonzero entries, must be initialized with 0
  idxpos = (integer *)calloc((size_t)k * p, sizeof(integer));

  // used as list of nonzero values, must be initialized with 0.0
  val = (double *)malloc((size_t)k * p * sizeof(double));
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
    printf("* sample covariance matrix S: time=%.3e nnz(S)/p=%.3e\n",
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
  integer stage_iter_max = atof(std::getenv("STAGE"));
  std::cout << "GAMMA,STAGE = " << gamma << "," << stage_iter_max << endl;

  double relative_objective_newton = INF;

  // outer Newton iteration loop, at most max_iter iteration steps unless
  // convergence
  for (integer newton_iter = 1;
       newton_iter <= max_newton_iter && relative_objective_newton > term_tol;
       ++newton_iter) {
    Stat.time_itr.push_back(-omp_get_wtime());

    // update matrix \Delta=0
    // pattern of S overlaps W
    AugmentS_OMP(&mu[0], idx, idxpos, &W);
    SortSparse_OMP(&S, idx);
    // ierr = CheckSymmetry(&S);

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
    //////////////////////////////////////////////////
    // END: Compute Active Set X_ij!=0 or |S_ij-W_ij|>lambda
    //////////////////////////////////////////////////

    double fX_newton = fX;
    double relative_objective = INF;
    double L_nnz_per_row = 0;
    integer line_search_iter_count = 0;
    integer stage_iter = 0;

    while (relative_objective > term_tol && stage_iter <= stage_iter_max) {
      stage_iter++;

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
      double diffD = 0.0;     // |(\mu e_ie_j^T)_{i,j}|_1
      srand(1);

      for (integer cdSweep = 1; cdSweep <= 1 + newton_iter / 3; cdSweep++) {
        diffD = 0.0;

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
            active_index_set.size() - 1, l1norm_D, diffD);

        if (diffD <= l1norm_D * coord_dec_sweep_tol) break;
      }

      Stat.time_upd[Stat.time_upd.size() - 1] += omp_get_wtime();
      ////////////////////////////////////////
      // END: coordinate decent update
      ////////////////////////////////////////

      ////////////////////////////////////////
      // ! START: compute trace((S-X^{-1})\Delta)
      ////////////////////////////////////////
      double trgradgD = 0.0;
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
          trgradgD += (Sij - Wij) * Dij;
        }
      }
      // augment X with the pattern of \Delta
      AugmentSparse_OMP(&X, idx, idxpos, &D);
      SortSparse_OMP(&X, idx);
      AugmentS_OMP(&mu[0], idx, idxpos, &X);
      SortSparse_OMP(&S, idx);
      // ierr = CheckSymmetry(X);
      // ierr = CheckSymmetry(&S);
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
      line_search_iter_count = 0;

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
        Stat.time_chol.push_back(-omp_get_wtime());
        chol_failed = LogDet(&logdet_X_update, loc_current_drop_tol);
        Stat.time_chol.back() = Stat.time_chol.back() + omp_get_wtime();

        // If choleksy did not fail
        // Check Armijo’s rule (Bertsekas, 1995; Tseng and Yun, 2007)
        // f(X+\alpha *D ) <=
        // f(X)+\alpha \sigma *(tr[grad * D]+|X+D|_1-|\Lambda * X|_1)
        // update objective function
        // f(X_1)=-log(det[X_1])+tra[SX_1]+|\lambda*X_1|_1

        double fX_update = -logdet_X_update + tr_SX_update + l1norm_X_update;
        double del = trgradgD + l1norm_XD - l1norm_X;

        armijo_failed =
            !(fX_update <= (fX + alpha * SIGMA * del) || l1norm_D < EPS);

        //  MSG("logdet_X_update=%e, tr_SX_update=%e, "
        //    "l1norm_X_update=%e,logdet_X=%e,tr_SX=%e, l1norm_X=%e\n ",
        //    logdet_X_update, tr_SX_update, l1norm_X_update, logdet_X, tr_SX,
        //    l1norm_X);

        // cout << ">>>> " << fX << " " << fX_update << " " << chol_failed <<
        // " "
        //     << armijo_failed << std::endl;

        // cout << "++++" << alpha << ")"
        //      << " " << fX_update << "<=" << (fX + alpha * SIGMA * del)
        //      << std::endl;

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

      // EXPORT FACTOR
      // export L,D to sparse matrix format along with the permutation
      // at the same time discard the Cholesky factorization
      BlockExportLdl();
      L_nnz_per_row = double(BL.nnz + BiD.nnz / 2.0) / double(p);
      // next time we call Cholesky the pattern may change (analyze_done=0)
      analyze_done = 1;

      Stat.time_lns.back() = Stat.time_lns.back() + omp_get_wtime();
      ////////////////////////////////////////
      // END: Line Search & Factorization
      ////////////////////////////////////////

      ///////////////////////////////////////
      // ! START: Matrix Inversion
      ///////////////////////////////////////
      Stat.time_inv.push_back(-omp_get_wtime());
      BlockNeumannInv(current_drop_tol, NULL);
      Stat.time_inv.back() = Stat.time_inv.back() + omp_get_wtime();
      ///////////////////////////////////////////////////
      // End: Matrix Inversion
      ///////////////////////////////////////////////////

      //  Clean up X ... remove zero values
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
        X.rowind[j] = (integer *)realloc(X.rowind[j],
                                         (size_t)X.ncol[j] * sizeof(integer));
        X.val[j] =
            (double *)realloc(X.val[j], (size_t)X.ncol[j] * sizeof(double));

        // Copy over buffer
        for (integer k = 0; k < X.ncol[j]; ++k) {
          X.val[j][k] = buffer_val[k];
          X.rowind[j][k] = buffer_rowind[k];
        }
      }

      relative_objective = fabs((fX - fX_last) / fX);

      MSG("     obj=%.3e |delta(obj)|/obj=%.3e "
          "nnz(X,L,W)/p=[%.3e %.3e %.3e] lns_iter=%d \n",
          fX, relative_objective, double(X.nnz) / double(p), L_nnz_per_row,
          double(W.nnz) / double(p), line_search_iter_count);
      fflush(stdout);
    }

    // we use at least the relative residual between two consecutive calls
    // + some upper/lower bound
    relative_objective_newton = fabs((fX - fX_newton) / fX);
    current_drop_tol = DROP_TOL_GAP * relative_objective_newton;
    current_drop_tol = MAX(current_drop_tol, drop_tol);
    current_drop_tol = MIN(MAX(DROP_TOL0, drop_tol), current_drop_tol);

    Stat.time_itr.back() = Stat.time_itr.back() + omp_get_wtime();
    Stat.opt.push_back(fX);

    MSG("* iter=%d time=%.3e obj=%.3e |delta(obj)|/obj=%.3e "
        "nnz(X,L,W)/p=[%.3e %.3e %.3e] lns_iter=%d \n",
        newton_iter, Stat.time_itr.back(), fX, relative_objective_newton,
        double(X.nnz) / double(p), L_nnz_per_row, double(W.nnz) / double(p),
        line_search_iter_count);
    fflush(stdout);

    // cout << "X.nnz=" << X.nnz << endl;

    //  cout << "output_time(end+1)=" << Stat.time_itr.back() << endl;
    //  cout << "output_obj(end+1)=" << Stat.opt.back() << endl;
    //  cout << "output_nnzprX(end+1)=" << X.nnz / double(p) << endl;
    //  cout << "output_nnzprL(end+1)=" << L_nnz_per_row << endl;
    //  cout << "output_nnzprW(end+1)=" << double(W.nnz) / double(p) << endl;
  }

  // The computed W does not satisfy |W - S|< lambda.  Project it.
  // now the meaning of U changes to case (c) and U is used as a buffer
  // only, since we are interested in log(det(project(W))), which in turn
  // requires its Cholesky factorization
  // compute U<-projected(W)
  AugmentS_OMP(&mu[0], idx, idxpos, &W);
  SortSparse_OMP(&S, idx);
  // ierr = CheckSymmetry(&S);

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
  double logdet_W = ProjLogDet(lambda, current_drop_tol);

  nn = Stat.time_chol.size();
  if (nn > 0) Stat.time_chol[nn - 1] += omp_get_wtime();

  if (RunTimeConfig.verbose == 4) {
    //(RunTimeConfig.factorization==CHOL_SUITESPARSE)
    MSG("* project log det: time=%0.2e nnz(X,L,W)/p=[%0.2e %0.2e %0.2e]\n",
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

  cout << "logdet_W " << logdet_W << endl;
  cout << "double(p) " << double(p) << endl;
  cout << "logdet_X " << logdet_X << endl;
  cout << "tr_SX " << tr_SX << endl;
  cout << "l1norm_X " << l1norm_X << endl;

  Stat.dgap = -logdet_W - double(p) - logdet_X + tr_SX + l1norm_X;
  Stat.trSX = tr_SX;
  Stat.logdetX = logdet_X;

  if (RunTimeConfig.verbose > 0) {
    MSG("#SQUIC Finished: time=%0.3e nnz(X,W)/p=[%0.3e %0.3e] dGap=%0.3e \n\n",
        Stat.time_total, double(X.nnz) / double(p), double(W.nnz) / double(p),
        Stat.dgap);
    fflush(stdout);
  }
  if (RunTimeConfig.verbose == 0) {
    MSG("time=%0.3e nnz(X,W)/p=[%0.3e %0.3e] dGap=%0.3e \n", Stat.time_total,
        double(X.nnz) / double(p), double(W.nnz) / double(p), Stat.dgap);
    fflush(stdout);
  }

  FreeSparse(&D);
  free(idx);
  free(idxpos);
  free(val);

  return;
};  // end run