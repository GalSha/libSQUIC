# SQUIC Release Source
This repository contains the official code for SQUIC, a second-order L1-regularized maximum likelihood method 
for performant large-scale sparse precision matrix estimation. The code is packaged as the shared library 
``libSQUIC**``, intended for Linux and Mac OS. It is written in C++ and is parallelized with OpenMP. This 
work is a synthesis of the previous publications listed in [References](#References).

The shared library can be downloaded and used directly in a precompiled version, see [`precompiled/`]
(precompiled/) directory, or compiled from source. Additionally a Python API can be found [here](https://www.
gitlab.ci.inf.usi.ch/SQUIC/SQUIC_Python), and an R API [here](https://www.gitlab.ci.inf.usi.ch/SQUIC/
SQUIC_R) . Note, for all APIs, the shared library ``libSQUIC.*`` is required.


<p align="center">
  <img align="middle" src="(https://drive.google.com/uc?id=14m72E-rGeM4dEBOfo6UGK6dAZ-n4Pnic)" alt="Location of samples" width="400"/>
</p>
<center>
The optimized algorithmic components present in SQUIC.
</center>


## Precompiled libSQUIC
The simplest way to start using SQUIC is to use the precompiled shared library ``libSQUIC.*``. The 
precompiled distributions are listed in the folder [`precompiled/`](precompiled/). All distributions, are 
self-contained except for OpenMP, which needs to be installed. This can be done via a standard package 
manager; e.g., for Ubuntu: ``sudo apt-get install libomp-dev``, or  Mac: ``brew install libomp``. The shared 
library exposes a single function called ``SQUIC_CPP``, see [`include\SQUIC.h`](include\SQUIC.h) for further details. 

## Compile from Source
The shared library can be compiled from source by first fulfilling the [Prerequisites](#Prerequisites) listed below and following the [Compilation & Instaation](#Compile&Install) instruction. 

### Prerequisites

For Mac:
- CMake (>3.9): For example using homebrew ``brew install cmake``.
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=mac&distributions=webdownload&options=online).
- OpenMP: For example using homebrew ``brew install libomp`` .
- C++/C Compilers (Clang): This is the default; nothing is required.

For Linux:
- CMake (>3.9)
- Intel MKL libraries: [Download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=online).
- OpenMP 
- C++/C/Fortran Compilers (GNU)


### Compile & Install

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
By default a dynamic library will be generated, if you want to generate a static library plase modify
cmake command with:
```aungular2
cmake ../ -DBUILD_SHARED_LIBS=OFF
```
Step 4: Install the libSQUIC library (default location: ``~/SQUIC``, use -DINSTALL_DIR to change it)
using the command:

```angular2
make install 
```

#### References

[1] [Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.](https://epubs.siam.org/doi/abs/10.1137/17M1147615)

[2] [Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation on high-performance computing architectures. In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.](https://dl.acm.org/doi/10.5555/3291656.3291683)

[3] [Eftekhari, A., Pasadakis, D., Bollhöfer, M., Scheidegger, S. and Schenk, O., 2021. Block-Enhanced PrecisionMatrix Estimation for Large-Scale Datasets. Journal of Computational Science, p. 101389.](https://www.sciencedirect.com/science/article/pii/S1877750321000776)


#### Citation
Please cite our publications if it helps your research:

```
@article{doi:10.1137/17M1147615,
	author = {Bollh\"{o}fer, Matthias and Eftekhari, Aryan and Scheidegger, Simon and Schenk, Olaf},
	journal = {SIAM Journal on Scientific Computing},
	number = {1},
	pages = {A380-A401},
	title = {Large-scale Sparse Inverse Covariance Matrix Estimation},
	url = {https://doi.org/10.1137/17M1147615},
	volume = {41},
	year = {2019}
}

@inproceedings{10.5555/3291656.3291683,
    author = {Eftekhari, Aryan and Bollh\"{o}fer, Matthias and Schenk, Olaf},
    title = {Distributed Memory Sparse Inverse Covariance Matrix Estimation on High-Performance Computing Architectures},
    year = {2018},
    publisher = {IEEE Press},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis},
    articleno = {20},
    numpages = {12},
    location = {Dallas, Texas},
    series = {SC '18}
}

@article{EFTEKHARI2021101389,
	author = {Aryan Eftekhari and Dimosthenis Pasadakis and Matthias Bollh{\"o}fer and Simon Scheidegger and Olaf Schenk},
	doi = {https://doi.org/10.1016/j.jocs.2021.101389},
	issn = {1877-7503},
	journal = {Journal of Computational Science},
	pages = {101389},
	title = {Block-enhanced precision matrix estimation for large-scale datasets},
	volume = {53},
	year = {2021}
}
```

