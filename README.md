# Sparse QUadratic Inverse Covariance matrix estimation (SQUIC)
This repository contains the official code for SQUIC, a second-order $`\ell_1`$-regularized maximum likelihood method 
for performant large-scale sparse precision matrix estimation. The code is packaged as the shared library 
``libSQUIC``, intended for Linux and Mac OS. It is written in C++ and is parallelized with OpenMP. This 
work is a synthesis of the previous publications listed in [References](#References).

The shared library can be downloaded and used directly in a precompiled version, see [`precompiled/`](precompiled/) 
directory, or compiled from source. Additionally a [Python API](https://www.gitlab.ci.inf.usi.ch/SQUIC/SQUIC_Python), 
and an [R API](https://www.gitlab.ci.inf.usi.ch/SQUIC/SQUIC_R) are available. Note that for all APIs, the shared library ``libSQUIC`` is required.


![image](https://drive.google.com/uc?id=14ob4yMPKd6NMcCnxsRT2Otxulpksb0Ay)
<div align="center">
<b>The optimized algorithmic components present in SQUIC.<b>
</div>


## Precompiled libSQUIC
The simplest way to use ``libSQUIC`` is to download the precompiled shared library. The 
precompiled distributions are listed in the folder [`precompiled/`](precompiled/). All distributions, are 
self-contained except for OpenMP, which needs to be installed. This can be done via a standard package 
managers.

## Compile from Source
The shared library can be compiled from source by first fulfilling the prerequisites listed below and following the instructions. 

##### Prerequisites

For Mac:
- CMake (>3.9).
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=mac&distributions=webdownload&options=online).
- OpenMP.
- C++/C Compilers (Clang): This is the default; nothing is required.

For Linux:
- CMake (>3.9). 
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=online).
- OpenMP.
- C++/C/Fortran Compilers (GNU): This is the default; nothing is required.

***

##### Step 0
Git clone the repository.

##### Step 1
After installing Intel MKL, the environment variable ``MKLROOT`` must be set, for example via the command:
```angular2
source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64
```
_Note: This command may differ depending on the version of MKL that you are using._

##### Step 2
Compile CHOLMOD and related libraries from SuiteSparse (version 4.5.4 is provided) by the script:
	
```angular2
sh make_cholmod.sh 
```
_Note: This script will copy the relevant files in the [extern/](extern/) folder._

##### Step 3
Create the build directory and compile the project using cmake with the following commands:  
```angular2
mkdir build
cd build
cmake ..
make -j12
```
_Note: By default a dynamic library will be generated, if you want to generate a static library modify the
cmake command as follows: ``cmake ../ -DBUILD_SHARED_LIBS=OFF``_

##### Step 4
Install the ``libSQUIC`` library (default location: ``~/``, use ``-DINSTALL_DIR`` to change it)
using the command:

```angular2
make install 
```

### Publications & References

Please cite our publication if it helps your research:

***Paper under review.***

For related research on SQUIC please see:

- [Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.](https://epubs.siam.org/doi/abs/10.1137/17M1147615)

- [Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation on high-performance computing architectures. In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.](https://dl.acm.org/doi/10.5555/3291656.3291683)

- [Eftekhari, A., Pasadakis, D., Bollhöfer, M., Scheidegger, S. and Schenk, O., 2021. Block-Enhanced PrecisionMatrix Estimation for Large-Scale Datasets. Journal of Computational Science, p. 101389.](https://www.sciencedirect.com/science/article/pii/S1877750321000776)
