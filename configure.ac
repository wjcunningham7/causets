AC_PREREQ([2.63])
AC_INIT([CausalSet], [1.1], [w.cunningham@northeastern.edu])
AM_INIT_AUTOMAKE

if [[ -e /home/$USER/.causet ]] ; then
  source /home/$USER/.causet
fi

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/Causet.cu])
AC_CONFIG_HEADERS([inc/config.h])

if [[[ -z ${CXXFLAGS+x} ]]] ; then
  CXXFLAGS=""
fi

AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB

AC_LANG_PUSH([C++])

AX_COMPILER_VERSION
AX_COMPARE_VERSION([$ax_cv_cxx_compiler_version], [ge], [4.8.1],, AC_MSG_ERROR([GNU compiler must be at least version 4.8.1]))
AX_CXX_COMPILE_STDCXX([11])

AX_FASTMATH([$FASTMATH_HOME], [mandatory])
AX_CHECK_CUDA([$CUDA_HOME], [$CUDA_SDK_PATH], [$CUDA_ARCH])
AX_PATH_GSL([1.13],, AC_MSG_ERROR([Could not find the required version of GSL.]))
BOOST_REQUIRE([1.55.0])
AX_PLATFORM([$PLATFORM])
AC_OPENMP

AC_CHECK_LIB([m], [cos],, AC_MSG_ERROR([Could not find the library "libm"]))
AC_CHECK_LIB([gslcblas], [cblas_dgemm],, AC_MSG_ERROR([Could not find the library "libgslcblas"]))
AC_CHECK_LIB([gsl], [gsl_blas_dgemm],, AC_MSG_ERROR([Could not find the library "libgsl"]))
AC_CHECK_LIB([stdc++], [main],, AC_MSG_ERROR([Could not find the library "libstdc++"]))
AC_CHECK_LIB([pthread], [fork],, AC_MSG_ERROR([Could not find the library "pthread"]))

AC_CHECK_FILES([src/autocorr2.cu src/Causet.cu src/CuResources.cu src/Measurements.cu src/NetworkCreator.cu src/Subroutines.cu src/Validate.cu],, AC_MSG_ERROR([Could not find CausalSet source files.]))
AC_CHECK_FILES([inc/autocorr2.h inc/Causet.h inc/Constants.h inc/Coordinates.h inc/CuResources.h inc/Geodesics.h inc/Measurements.h inc/NetworkCreator.h inc/Operations.h inc/Subroutines.h inc/Validate.h],, AC_MSG_ERROR([Could not find CausalSet header files.]))

AC_CHECK_HEADERS([cassert cmath cstring exception fstream getopt.h inttypes.h iomanip iostream limits math.h pthread.h sstream stdarg.h stddef.h stdint.h stdio.h string sys/io.h unordered_set vector],, AC_MSG_ERROR([Could not find C++ header file.]))
AC_CHECK_HEADERS([boost/functional/hash/hash.hpp boost/random/mersenne_twister.hpp boost/random/normal_distribution.hpp boost/random/poisson_distribution.hpp boost/random/uniform_real.hpp boost/random/variate_generator.hpp boost/unordered_map.hpp],, AC_MSG_ERROR([Could not find Boost header file.]))
AC_CHECK_HEADER([gsl/gsl_sf_lambert.h],, AC_MSG_ERROR([Could not find GSL header file.]))

if [[[ "$CUDA_ENABLED" == "yes" ]]] ; then
  AC_CHECK_FILES([src/NetworkCreator_GPU.cu src/Operations_GPU.cu src/Subroutines_GPU.cu],, AC_MSG_ERROR([Could not find CausalSet GPU source files.]))
  AC_CHECK_FILES([inc/NetworkCreator_GPU.h inc/Operations_GPU.h inc/Subroutines_GPU.h],, AC_MSG_ERROR([Could not find CausalSet GPU header files.]))
fi

AC_FUNC_MALLOC
AC_TYPE_SIZE_T
AC_TYPE_INT64_T
AC_TYPE_UINT64_T
AC_C_INLINE
AC_C_CONST
AC_C_VOLATILE

CXXFLAGS="$CXXFLAGS -O3 -g -Wall -x c++ -DBOOST_NO_FENV_H -mpopcnt"
LDFLAGS="-lstdc++ -lpthread -lm -lgsl -lgslcblas"
AX_GCC_X86_CPU_SUPPORTS([avx2], [AVXFLAGS="-mavx2 -DAVX2_ENABLED"], AC_MSG_RESULT([AVX2 features will be disabled.]))
AX_GCC_X86_CPU_SUPPORTS([popcnt],, AC_MSG_ERROR([CPU does not support the popcnt instruction.]))
AX_GCC_ARCHFLAG([no],, AC_MSG_ERROR([CPU architecture could not be identified.]))
AC_SUBST([ARCHFLAGS], [$ax_cv_gcc_archflag])

dnl echo "CXXFLAGS:     "$CXXFLAGS
dnl echo "CUDA_FLAGS:   "$CUDA_FLAGS
dnl echo "FAST_FLAGS:   "$FASTMATH_FLAGS
dnl echo "ARCHFLAGS:    "$ARCHFLAGS
dnl echo "AVXFLAGS:     "$AVXFLAGS
dnl echo "LDFLAGS:      "$LDFLAGS
dnl echo "CUDA_LDFLAGS: "$CUDA_LDFLAGS
dnl echo "FAST_LDFLAGS: "$FASTMATH_LDFLAGS

AX_PROG_MPICXX([$MPI_HOME])
BINARY_NAME=CausalSet$PLATFORM
if [[[ "$CUDA_ENABLED" == "yes" ]]] ; then
COMPILE_STR="$NVCC -dc $CUDA_FLAGS $FASTMATH_FLAGS -Xcompiler '$AVXFLAGS $ARCHFLAGS $OPENMP_CXXFLAGS $MPI_FLAGS'"
LNK_STR="$NVCC src/*.o -dlink $CUDA_FLAGS -o src/linked.o"
BIN_STR="$CXX src/*.o $OPENMP_CXXFLAGS $LDFLAGS $CUDA_LDFLAGS $FASTMATH_LDFLAGS -o bin/$BINARY_NAME"
else
COMPILE_STR="$CXX -c $CXXFLAGS $FASTMATH_FLAGS $AVXFLAGS $ARCHFLAGS $OPENMP_CXXFLAGS $MPI_FLAGS"
LNK_STR="$CXX src/*.o $OPENMP_CXXFLAGS $LDFLAGS $FASTMATH_LDFLAGS -o bin/$BINARY_NAME"
BIN_STR=""
fi
AC_SUBST([COMPILE_STR], [$COMPILE_STR])
AC_SUBST([LNK_STR], [$LNK_STR])
AC_SUBST([BIN_STR], [$BIN_STR])
AC_SUBST([BINARY_NAME], [$BINARY_NAME])
AM_CONDITIONAL(USE_GPU, [test "x$CUDA_ENABLED" == "xyes"])

AC_CONFIG_FILES([Makefile src/Makefile inc/Makefile printcolor/Makefile])
AC_OUTPUT
