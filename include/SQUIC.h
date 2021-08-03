/* SQUIC.h
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

#define integer long int

extern "C"
{
	/**
	 * @brief      Create buffer at given location
	 *
	 * @param      buffer  The buffer
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memset_integer(integer *&buffer, integer length);

	/**
	 * @brief      Create buffer at given location
	 *
	 * @param      buffer  The buffer
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memset_double(double *&buffer, integer length);


	/**
	 * @brief      Create a new copy of the buffer
	 *
	 * @param      buffer  The copy buffer
	 * @param      values  The buffer to be copied
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memcopy_integer(integer *&buffer, integer *values, integer length);

	/**
	 * @brief      Create a new copy of the buffer
	 *
	 * @param      buffer  The copy buffer
	 * @param      values  The buffer to be copied
	 * @param[in]  length  The length of the buffer
	 */
	void SQUIC_CPP_UTIL_memcopy_double(double *&buffer, double *values, integer length);

	/**
	 * @brief      Free the buffer
	 *
	 * @param      buffer  The buffer
	 */
	void SQUIC_CPP_UTIL_memfree_integer(integer *&buffer);

	/**
	 * @brief      Free the buffer
	 *
	 * @param      buffer  The buffer
	 */
	void SQUIC_CPP_UTIL_memfree_double(double *&buffer);


	/**
	 * @brief      SQUIC CPP Interface
	 *
	 * @param[in]        mode            Runtime mode values [0,1,2,3,4] use the "block" algorithem and [5,6,7,8,9] use the "scalar" algorithem <<NOTE: Recommended "0">>.
	 * @param[in]        p               Number of random variables
	 * @param[in]        n               Number of samples
	 * @param[in]        Y               Data pointer
	 * @param[in]        lambda          Scalar tunning parameter
	 * @param[in]        M_rinx          M matrix row index
	 * @param[in]        M_cptr          M matrix column pointer
	 * @param[in]        M_val           M matrix value
	 * @param[in]        M_nnz           M matrix number of nonzeros
	 * @param[in]        max_iter        Maximum Netwon iterations <<NOTE: max_iter=0 will return sample covaraince matrix in ouput iC >>
	 * @param[in]        inv_tol         Inversion tolerance for approximate inversion <<NOTE: inv_tol>0 , Recommended inv_tol=term_tol >>
	 * @param[in]        term_tol        Termination tolerance <<NOTE: term_tol>0 , Recommended inv_tol=term_tol >>
	 * @param[in]        verbose         Verbose level 0 or 1
	 * @param[in/out]    X_rinx          Percision matrix row index       <<NOTE: Intial value X0_rinx is passed in here >>
	 * @param[in/out]    X_cptr          Percision matrix column pointer  <<NOTE: Intial value X0_cptr is passed in here >>
	 * @param[in/out]    X_val           Percision matrix value           <<NOTE: Intial value X0_val is passed in here >>
	 * @param[in/out]    X_nnz           Percision matrix nnz             <<NOTE: Intial value X_nnz is passed in here >>
	 * @param[in/out]    W_rinx          Covariance matrix row index      <<NOTE: Intial value W0_rinx is passed in here >>
	 * @param[in/out]    W_cptr          Covariance matrix column pointer <<NOTE: Intial value W0_rinx is passed in here >>
	 * @param[in/out]    W_val           Covariance matrix value          <<NOTE: Intial value W0_rinx is passed in here >>
	 * @param[in/out]    W_nnz           Covariance matrix nnz            <<NOTE: Intial value W0_rinx is passed in here >>
	 * @param[out]       info_num_iter   Information number of newton iterations performed
	 * @param[out]       info_times      Information 6 element array of times for computing 1)total 2)sample covaraince 3)optimization 4)factorization 5)approximate inversion 6)coordinate update
	 * @param[out]       info_objective  Objective function value         <<NOTE: this array must be of length max_iter when passed in. Upon ouput only info_num_iter element will be written to>>
	 * @param[out]       info_logdetX    Log determinant of X
	 * @param[out]       info_trSX       Trace of SX
	 */
	void SQUIC_CPP(
	    int mode,
	    integer p, integer n, double *Y,
	    double lambda,
	    integer *M_rinx, integer *M_cptr, double *M_val, integer M_nnz,
	    int max_iter, double inv_tol, double term_tol, int verbose,
	    integer *&X_rinx, integer *&X_cptr, double *&X_val, integer &X_nnz,
	    integer *&W_rinx, integer *&W_cptr, double *&W_val, integer &W_nnz,
	    int &info_num_iter,
	    double *&info_times,	 //length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
	    double *&info_objective, // length must be size max_iter
	    double &info_logdetX,
	    double &info_trSX);
};
