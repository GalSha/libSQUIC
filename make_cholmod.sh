#!/bin/ bash

RED='\033[0;31m'
GREEN='\033[92m'
NC='\033[0m' 

CC=${1:-gcc}
CXX=${2:-g++}

INTEL_LIB=$MKLROOT


if [ -z "$INTEL_LIB" ]
then
      echo -e "${GREEN}>> The environment variable MKLROOT=${MKLROOT} is not set ... Please source the Intel MKL mklvars.sh file. ${NC}"
      exit 1
fi

echo -e "${GREEN}>> Compiling CHOLMOD Using: ${RED} CC=${CC} CXX=${CXX}  INTEL_LIB=${INTEL_LIB} ${NC}"
read -p "Press any key to resume ..."

INSTDIR=$(pwd)/extern/suitesparse

LIBDIR=$INSTDIR/lib
INCDIR=$INSTDIR/include

mkdir -p $LIBDIR
mkdir -p $INCDIR

echo -e "${GREEN}>> Unzip SuiteSparse-4.5.4 ${NC}"
read -p "Press any key to resume ..."
unzip SuiteSparse-4.5.4.zip

echo -e "${GREEN}>> Make metis static library first  ${NC}"
read -p "Press any key to resume ..."
cd SuiteSparse-4.5.4/metis-5.1.0
make config  cc=$CC prefix=$INSTDIR 
make install
mv lib/*.a $LIBDIR
cp include/*.h $INCDIR

echo -e "${GREEN}>> Make CHOLMOD staic library with static METIS  ${NC}"
cd ..

make config AUTOCC="no" CC="$CC" CXX="$CXX" JOBS="1" INSTALL="$INSTDIR" MY_METIS_LIB="$LIBDIR/libmetis.a" MY_METIS_INC="$INCDIR" CUDA="no"  
read -p "Press any key to resume ..."
make library AUTOCC="no" CC="$CC" CXX="$CXX" JOBS="1" INSTALL="$INSTDIR" MY_METIS_LIB="$LIBDIR/libmetis.a" MY_METIS_INC="$INCDIR" CUDA="no"      

echo -e "${GREEN}>> Copy files ..  ${NC}"

cp CHOLMOD/Lib/*.a $LIBDIR
cp CHOLMOD/Include/*.h $INCDIR

cp SuiteSparse_config/*.a $LIBDIR
cp SuiteSparse_config/*.h $INCDIR

echo -e "${GREEN}>> Cleaningup  ${NC}"
cd ..
rm -rf SuiteSparse-4.5.4
