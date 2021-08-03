# SQUIC Release Source

SQUIC is a second-order, L1-regularized maximum likelihood method for performant large-scale sparse precision matrix estimation. The presented source code is the SQUIC shared library (libSQUIC) intended for Linux and Mac OS.

The following interface packages for SQUIC are available:

- R package SQUIC_R (see )
- Python package SQUIC_Python (see )
- Matlab interface SQUIC_Matlab (see )

_Note: For all note interface packages, libSQUIC is required. Precompiled versions of libSQUIC for Mac and Linux are available and are ready to use._ 


#### References

[1] [Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.](https://epubs.siam.org/doi/abs/10.1137/17M1147615?journalCode=sjoce3)

[2] [Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation on high-performance computing architectures. In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.](https://dl.acm.org/doi/10.5555/3291656.3291683)

[3] [Eftekhari, A., Pasadakis, D., Bollhöfer, M., Scheidegger, S. and Schenk, O., 2021. Block-Enhanced PrecisionMatrix                 Estimation for Large-Scale Datasets. Journal of Computational Science, p. 101389.](https://www.sciencedirect.com/science/article/pii/S1877750321000776)


## Precompiled libSQUIC

The precompiled libSQUIC distributions are listed in the folder ``/precompiled/``. All distributions require OpenMP to be installed, which can be done via a standard package manager; e.g., for Ubuntu: ``sudo apt-get install libomp-dev``, or  Mac: ``brew install libomp``.  

_Note: The default location of libSQUIC for the different interface packages is the home directory; i.e., ``~/``. For further details, refer to the respective interface package._

## Installation from Source

### Requirements :

For Mac:
- CMake (>3.9): For example using homebrew ``brew install cmake``
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=mac&distributions=webdownload&options=online).
- OpenMP: For example using homebrew ``brew install libomp`` 
- C++/C Compilers (Clang): This is the default; nothing is required.
- Fortran Compiler (GNU): For example, using homebrew ``brew install gcc``

For Linux:
- CMake (>3.9)
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=online).
- OpenMP 
- C++/C/Fortran Compilers (GNU)


### Compile & Install :

Step 0: Git clone the repository.

Step 1: After installing Intel MKL, the evironment variable MKLROOT must be set via the command:
```angular2
source /opt/intel/mkl/bin/mklvars.sh intel64  # or 
source /opt/intel/oneapi/mkl/latest/env/vars.sh  intel64
```

Step 2: Compile CHOLMOD and related libraries from SuiteSparse (version 4.5.4 is provided) by the script:
	
```angular2
sh make_cholmod.sh 
```
_Note: This script will copy the relevant files in the ``/extern/`` folder._

Step 3: Create the build directory and compile the project using cmake with the following commands:  
```angular2
mkdir build
cd build
cmake ..
make -j12
```

Step 4: Install the libSQUIC library in the default location (i.e., ``~/``) using the command:

```angular2
make install 
```
