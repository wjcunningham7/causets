AC_PREREQ([2.63])
AC_INIT([CausalSet], [1.1], [w.cunningham@northeastern.edu])
AM_INIT_AUTOMAKE

if [[ -e /home/$USER/.causet ]] ; then
  source /home/$USER/.causet
fi

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/Causet.cu])
AC_CONFIG_HEADERS([inc/config.h])

AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB

AC_LANG([C++])
AX_CXX_COMPILE_STDCXX([11])

AM_CONDITIONAL(GCC_GE_481, test `g++ -dumpversion | awk '{print $1>=4.8.1?"1":"0"}'` = 1)
AX_PATH_GSL([1.13],, AC_MSG_ERROR([Could not find the required version of GSL.]))
BOOST_REQUIRE([1.55.0])
AC_OPENMP

NVCC=$(which nvcc)
if [[ "$NVCC" != "" ]] ; then
  echo "checking for nvcc... yes"
  if [[ -z ${CUDA_HOME+x} ]] ; then
    CUDA_HOME=$(dirname $(dirname $NVCC))
  fi
  CUDA_SDK_PATH=$CUDA_HOME/samples

  AC_DEFINE([HAVE_CUDA_INSTRUCTIONS], [1], [Cuda is detected and will be enabled.])
  AC_DEFINE_UNQUOTED([NVCC], ["$NVCC"], [Path for nvcc compiler.])
  AC_DEFINE_UNQUOTED([CUDA_HOME], ["$CUDA_HOME"], [CUDA installation directory.])
  AC_DEFINE_UNQUOTED([CUDA_SDK_PATH], ["$CUDA_SDK_PATH"], [CUDA SDK directory.])

  AC_SUBST([NVCC], ["$NVCC"])
  AC_SUBST([CUDA_HOME], ["$CUDA_HOME"])
  AC_SUBST([CUDA_SDK_PATH], ["$CUDA_SDK_PATH"])

  if [[ -z ${CUDA_ARCH+x} ]] ; then
    CUDA_ARCH=sm_20
  fi
  echo "checking for nvcc arch flag... -arch="$CUDA_ARCH
  AC_DEFINE_UNQUOTED([CUDA_ARCH], ["$CUDA_ARCH"], [GPU architecture passed to nvcc with -arch flag.])

  NVCCFLAGS="-O3 -G -g -DBOOST_NOINLINE='__attribute__ ((noinline))' -DCUDA_ENABLED -arch="$CUDA_ARCH" --std=c++11 -DBOOST_NO_FENV_H"
  GLOB_MEM_MIB=$(nvidia-smi -q | grep -A3 "FB Memory Usage" | awk '/Total/ {print $3}')
  GLOB_MEM_BYTES=$(($(echo "scale=8; $GLOB_MEM_MIB * 1.049 / 1000" | bc | awk '{printf("%d\n", $1)}') * 1000000000))
  AC_DEFINE_UNQUOTED([GLOB_MEM], ["$GLOB_MEM_BYTES"], [GPU global memory size.])
else
  echo "checking for nvcc... no"
fi

AX_GCC_X86_CPU_SUPPORTS([avx2],, AC_MSG_RESULT([AVX2 features will be disabled.]))
AX_GCC_X86_CPU_SUPPORTS([popcnt],, AC_MSG_ERROR([CPU does not support the popcnt instruction.]))
AX_GCC_ARCHFLAG([no],, AC_MSG_ERROR([CPU architecture could not be identified.]))

CXXFLAGS="-O3 -g -Wall -x c++ -DBOOST_NO_FENV_H $ax_cv_gcc_archflag"
if [[ -z ${HAVE_AVX2_INSTRUCTIONS+x} ]] ; then
  CXXFLAGS+=" -mavx2 -DAVX2_ENABLED"
  NVCCFLAGS+=" -Xcompiler '-mavx2 $ax_cv_gcc_archflag -DAVX2_ENABLED'"
fi
if [[ -z ${HAVE_POPCNT_INSTRUCTIONS+x} ]] ; then
  CXXFLAGS+=" -mpopcnt"
  NVCCFLAGS+=" -Xcompiler -mpopcnt"
fi

AC_SUBST([NVCCFLAGS], ["$NVCCFLAGS"])

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
