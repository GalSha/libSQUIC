/* long_integer.h
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

#ifndef _LONG_INTEGER_H
#define _LONG_INTEGER_H

// #ifndef integer

#ifdef _LONG_INTEGER_
#define integer long

#define CHOLMOD_INTGER_TYPE CHOLMOD_LONG
#define CHOLMOD_ANALYZE_P cholmod_l_analyze_p
#define CHOLMOD_CHANGE_FACTOR cholmod_l_change_factor
#define CHOLMOD_FACTORIZE_P cholmod_l_factorize_p
#define CHOLMOD_ANALYZE cholmod_l_analyze
#define CHOLMOD_FACTORIZE cholmod_l_factorize
#define CHOLMOD_FREE_FACTOR cholmod_l_free_factor
#define CHOLMOD_START cholmod_l_start
#define CHOLMOD_FINISH cholmod_l_finish
#define CHOLMOD_PRINT_FACTOR cholmod_l_print_factor
#define CHOLMOD_NORM_SPARSE cholmod_l_norm_sparse
#define CHOLMOD_NNZ cholmod_l_nnz
#define CHOLMOD_CHECK_COMMON cholmod_l_check_common
#define CHOLMOD_PRINT_COMMON cholmod_l_print_common
#define CHOLMOD_CHECK_SPARSE cholmod_l_check_sparse
#define CHOLMOD_PRINT_SPARSE cholmod_l_print_sparse
#define CHOLMOD_CHECK_FACTOR cholmod_l_check_factor
#define CHOLMOD_PRINT_FACTOR cholmod_l_print_factor

#else

#define integer int

#define CHOLMOD_INTGER_TYPE CHOLMOD_INT
#define CHOLMOD_ANALYZE_P cholmod_analyze_p
#define CHOLMOD_CHANGE_FACTOR cholmod_change_factor
#define CHOLMOD_FACTORIZE_P cholmod_factorize_p
#define CHOLMOD_ANALYZE cholmod_analyze
#define CHOLMOD_FACTORIZE cholmod_factorize
#define CHOLMOD_FREE_FACTOR cholmod_free_factor
#define CHOLMOD_START cholmod_start
#define CHOLMOD_FINISH cholmod_finish
#define CHOLMOD_PRINT_FACTOR cholmod_print_factor
#define CHOLMOD_NORM_SPARSE cholmod_norm_sparse
#define CHOLMOD_NNZ cholmod_nnz
#define CHOLMOD_CHECK_COMMON cholmod_check_common
#define CHOLMOD_PRINT_COMMON cholmod_print_common
#define CHOLMOD_CHECK_SPARSE cholmod_check_sparse
#define CHOLMOD_PRINT_SPARSE cholmod_print_sparse
#define CHOLMOD_CHECK_FACTOR cholmod_check_factor
#define CHOLMOD_PRINT_FACTOR cholmod_print_factor

// #endif

#endif

#endif
