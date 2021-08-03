/* mymkl.h
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

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* useful MKL extensions to sparse matrices
   $Id: mymkl.h 3203 2017-04-16 22:07:48Z bolle $ */
#ifndef _MYMKL_H_
#define _MYMKL_H
  
#define MKL_DOMAIN_ALL      0
#define MKL_DOMAIN_BLAS     1
#define MKL_DOMAIN_FFT      2
#define MKL_DOMAIN_VML      3
#define MKL_DOMAIN_PARDISO  4




extern "C" {

int mkl_get_max_threads();
int mkl_get_dynamic();
int mkl_domain_get_max_threads(int domain);
int mkl_domain_set_num_threads(int nt, int domain);

#ifdef _BLAS_LAPACK_32_
void cblas_dsctr(int N, double *X, int *indx, 
		 double *Y);

void cblas_dgthr(int N, double *Y, double *X, 
		 int *indx);

void cblas_daxpyi(int N, double alpha, double *X, 
		  int *indx, double *Y);

double cblas_ddoti(int N, double *X, int *indx,
			    double *Y);
#else
void cblas_dsctr(integer N, double *X, integer *indx, 
		 double *Y);

void cblas_dgthr(integer N, double *Y, double *X, 
		 integer *indx);

void cblas_daxpyi(integer N, double alpha, double *X, 
		  integer *indx, double *Y);

double cblas_ddoti(integer N, double *X, integer *indx,
			    double *Y);
#endif
}
#endif

