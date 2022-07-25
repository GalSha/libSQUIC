cmake_minimum_required(VERSION 3.9)

#Project
project(SQUIC
VERSION 1.0
DESCRIPTION "Large-Scale Sparse Precision Matrix Estimation Package"
LANGUAGES CXX C Fortran)

# Fortran\CPP
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
unset(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Check for MKL root
if("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/extern/suitesparse")
    message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/extern/suitesparse does not exist. Please run make_cholmod.sh.")
endif()

# Set directories
set(SS_DIR "${CMAKE_SOURCE_DIR}/extern/suitesparse")
set(MKL_DIR $ENV{MKLROOT})
set(INSTALL_DIR "$ENV{HOME}/SQUIC/" CACHE STRING "INSTALL DIR")
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)


# Base Flags
set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}-w -O3 -D_LONG_INTEGER_")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}-w -O3 -fimplicit-none")

# Intel Flags 
# Link flags & Compiler Options from https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -DMKL_ILP64  -m64")
set(CMAKE_Fortran_FLAGS	"${CMAKE_Fortran_FLAGS} -fdefault-integer-8 -m64")

# Flags for static linking ... as much as possible. 
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -static-libgfortran")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

#Create shared library SQUIC
add_library(SQUIC
	src/squic.cpp  
	src/bninv.cpp 
	src/cholesky.cpp 
	src/coordinatedescentupdate.cpp 
	src/generates.cpp  
	src/squic_matrix.cpp  
	src/cholesky_matrix.cpp 
	src/coordinatedescentupdate_matrix.cpp 
	src/generates_matrix.cpp 
	src/qsort2.f 
	src/qsort1.f 
	src/qqsorti.f
)

# Find OMP
if ( "${OMP_LIB}" STREQUAL "" AND  "${OMP_INC}" STREQUAL "" AND  "${OMP_FLAGS}" STREQUAL "") 

	find_package(OpenMP REQUIRED)

	message(STATUS "OpenMP_CXX_LIBRARIES    : " ${OpenMP_CXX_LIBRARIES})
	message(STATUS "OpenMP_Fortran_LIBRARIES: " ${OpenMP_Fortran_LIBRARIES})

	if(OpenMP_CXX_FOUND)
    	target_link_libraries(SQUIC PRIVATE  OpenMP::OpenMP_CXX)
    	set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
	endif()

else() 
# This is a manual search approach - it  is only used for testing - not recommended
# e.g., -DOMP_LIB=/usr/local/lib/libomp.a -DOMP_INC=/usr/local/include -DOMP_FLAGS=-fopenmp 
	
	message("Using predifined location and flags for OpenMP")

	message(STATUS "OMP_LIB   : " ${OMP_LIB})
	message(STATUS "OMP_INC   : " ${OMP_INC})
	message(STATUS "OMP_FLAGS : " ${OMP_FLAGS})

	target_link_libraries(SQUIC PRIVATE ${OMP_LIB})
	target_include_directories(SQUIC PRIVATE ${OMP_INC})
	set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} ${OMP_FLAGS}")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OMP_FLAGS}")
endif()

# CHOLMOD
target_link_libraries(SQUIC PRIVATE 
	${SS_DIR}/lib/libcholmod.a 
	${SS_DIR}/lib/libsuitesparseconfig.a  
	${SS_DIR}/lib/libcolamd.a 
	${SS_DIR}/lib/libamd.a  
	${SS_DIR}/lib/libcamd.a 
	${SS_DIR}/lib/libccolamd.a 
	${SS_DIR}/lib/libmetis.a  
)
target_include_directories(SQUIC PRIVATE ${SS_DIR}/include)
##

# INTEL
find_library(libmkl_core_lib 
	NAMES libmkl_core.a
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	)

find_library(libmkl_intel_ilp64_lib 
	NAMES libmkl_intel_ilp64.a 
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	) 

find_library(libmkl_intel_thread_lib 
	NAMES libmkl_intel_thread.a
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
	) 

find_library(libmkl_gnu_thread_lib 
	NAMES libmkl_gnu_thread.a
	PATHS ${MKL_DIR} ${MKL_DIR}/lib ${MKL_DIR}/mkl/lib ${MKL_DIR}/lib/intel64 
	NO_DEFAULT_PATH
) 

if(APPLE)
	target_link_libraries(SQUIC PRIVATE  ${libmkl_intel_ilp64_lib} ${libmkl_intel_thread_lib} ${libmkl_core_lib})
elseif(UNIX AND NOT APPLE)
	target_link_libraries(SQUIC PRIVATE  ${MKL_DIR}/lib/intel64/libmkl_blas95_ilp64.a ${MKL_DIR}/lib/intel64/libmkl_lapack95_ilp64.a  ${MKL_DIR}/lib/intel64/libmkl_lapack95_ilp64.a -Wl,--start-group ${libmkl_intel_ilp64_lib} ${libmkl_gnu_thread_lib} ${libmkl_core_lib} -Wl,--end-group)
else()
    message(FATAL_ERROR "Operating system not supported.")
endif()

target_include_directories(SQUIC PRIVATE ${MKL_DIR}/include ${MKL_DIR}/include/intel64/ilp64)
##

# Disable setting rpath
set_target_properties(SQUIC PROPERTIES
	MACOSX_RPATH OFF
)

# Install library by default in the home directory
install(TARGETS SQUIC LIBRARY DESTINATION ${INSTALL_DIR}
)
#DESTINATION ${INSTALL_DIR}/lib
message("===============================================================================")
message("==========================Configuration Summary================================")
message("===============================================================================")
message("CMAKE_CXX_FLAGS          : " ${CMAKE_CXX_FLAGS})
message("CMAKE_Fortran_FLAGS      : " ${CMAKE_Fortran_FLAGS})
message("Install Location          : " ${INSTALL_DIR})
message("INTEL MKL Location       : " ${MKL_DIR})
message("SuitSparse Location      : " ${SS_DIR})
message("Build Sys.               : " "${CMAKE_SYSTEM_NAME}${CMAKE_SYSTEM_VERSION}-CXX${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}-F${CMAKE_Fortran_COMPILER_ID}${CMAKE_Fortran_COMPILER_VERSION}")
