AC_DEFUN([AX_CHECK_CUDA],
[
AC_PROVIDE([AX_CHECK_CUDA])
AC_REQUIRE([AC_PROG_SED])
AC_REQUIRE([AC_PROG_GREP])
AC_REQUIRE([AC_PROG_AWK])

AC_ARG_ENABLE([cuda], AS_HELP_STRING([--enable-cuda], [Enable CUDA capabilities (default no)]), [USE_CUDA=$enable_cuda], [USE_CUDA=no])

if [[[ $USE_CUDA == "yes" ]]] ; then
  AC_ARG_WITH([cuda], AS_HELP_STRING([--with-cuda=PREFIX], [Prefix of your CUDA installation]),
    [cuda_prefix=$with_cuda],
    [ if [[[ "$#" -ge 1 && "$1" != "" ]]] ; then
        cuda_prefix="$1"
      else
        cuda_prefix=$(which nvcc | sed "s/\/bin\/nvcc//")
      fi
    ])

  if [[[ $cuda_prefix != "" ]]] ; then
    AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
    if [[[ -e $cuda_prefix/bin/nvcc ]]] ; then
      AC_MSG_RESULT([found])
      CUDA_HOME=$cuda_prefix
      NVCC=$cuda_prefix/bin/nvcc
      VALID_CUDA=yes
    else
      AC_MSG_RESULT([not found!])
      AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
      VALID_CUDA=no
    fi
  else
    AC_MSG_WARN([Could not find installation directory for CUDA.])
    VALID_CUDA=no
  fi
fi

if [[[ $USE_CUDA == "yes" && $VALID_CUDA == "yes" ]]] ; then
  AC_ARG_WITH([cuda-sdk], AS_HELP_STRING([--with-cuda-sdk=PREFIX], [Prefix of your CUDA SDK installation]),
    [cuda_sdk_prefix=$with_cuda_sdk],
    [ if [[[ "$#" -ge 2 && "$2" != "" ]]] ; then
        cuda_sdk_prefix="$2"
      else
        cuda_sdk_prefix=$cuda_prefix/samples
      fi
    ])

  if [[[ $cuda_sdk_prefix != "" ]]] ; then
    AC_MSG_CHECKING([CUDA SDK in $cuda_sdk_prefix])
    if [[[ -d $cuda_sdk_prefix ]]] ; then
      AC_MSG_RESULT([found])
      CUDA_SDK_PATH=$cuda_sdk_prefix
    else
      AC_MSG_RESULT([not found!])
      AC_MSG_WARN([CUDA SDK was not found in $cuda_sdk_prefix])
      VALID_CUDA=no
    fi
  else
    AC_MSG_WARN([Could not find installation directory for CUDA SDK.])
    VALID_CUDA=no
  fi
fi

if [[[ $USE_CUDA == "yes" && $VALID_CUDA == "yes" ]]] ; then
  AC_MSG_CHECKING([CUDA Toolkit version])
  ax_cuda_version=$(nvcc --version | awk -F "[[, ]]" 'NR==4 {print [$]6}')
  AC_MSG_RESULT([$ax_cuda_version])
  AX_COMPARE_VERSION([$ax_cuda_version], [ge], [7.0],, AC_MSG_WARN([CUDA Toolkit must be at least version 7.0]); VALID_CUDA=no)
fi

if [[[ $USE_CUDA == "yes" && $VALID_CUDA == "yes" ]]] ; then
  AC_ARG_WITH([cuda-arch], AS_HELP_STRING([--with-cuda-arch=ARCH], [CUDA Compute Capability (default sm_30)]),
    [cuda_arch=$with_cuda_arch],
    [ if [[[ "$#" -ge 3 && "$3" != "" ]]] ; then
        cuda_arch="$3"
      else
        cuda_arch="sm_30"
      fi
    ])

  AC_ARG_WITH([nvidia-smi], AS_HELP_STRING([--with-nvidia-smi=PREFIX], [Prefix of the nvidia-smi command]),
    [nvidia_smi_prefix=$with_nvidia_smi],
    [nvidia_smi_prefix=$(which nvidia-smi | sed "s/\/nvidia-smi//")])

  if [[[ $nvidia_smi_prefix != "" ]]] ; then
    AC_MSG_CHECKING([size of global GPU memory])
    glob_mem_mib=$($nvidia_smi_prefix/nvidia-smi -q | grep -A3 "FB Memory Usage" | awk '/Total/ {print [$]3; exit;}')
    glob_mem_bytes=$(($(echo "scale=8; $glob_mem_mib * 1.049 / 1000" | bc | awk '{printf("%d\n", [$]1)}') * 1000000000))
    glob_mem_gb=$(($glob_mem_bytes/1000000000))
    AC_MSG_RESULT([$glob_mem_gb GB])
  else
    AC_MSG_WARN([Could not find the location of nvidia-smi.])
    VALID_CUDA=no
  fi
fi

if [[[ $USE_CUDA == "yes" && $VALID_CUDA == "yes" ]]] ; then
  CUDA_INCLUDE="-I $CUDA_HOME/include -I $CUDA_SDK_PATH/common/inc"
  CUDA_LDFLAGS="-L $CUDA_HOME/lib64 -L $CUDA_SDK_PATH/common/lib -lcuda -lcudart"

  cxx_flags=$CXXFLAGS
  ld_flags=$LDFLAGS

  CXXFLAGS="$CXXFLAGS $CUDA_INCLUDE"
  LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"
  AC_CHECK_HEADER([cuda.h],, AC_MSG_WARN([Could not find cuda.h]); VALID_CUDA=no, [#include <cuda.h>])
  AC_CHECK_LIB([cuda], [cuInit],, AC_MSG_WARN([Could not find libcuda]); VALID_CUDA=no)
  AC_CHECK_LIB([cudart], [cudaGetDevice],, AC_MSG_WARN([Could not find libcudart]); VALID_CUDA=no)
  LIBS=""

  CXXFLAGS=$cxx_flags
  LDFLAGS=$ld_flags
fi

if [[[ $USE_CUDA == "yes" ]]] ; then
  AC_MSG_CHECKING([if CUDA will be used])
  if [[[ $VALID_CUDA == "yes" ]]] ; then
    AC_MSG_RESULT([yes])
    AC_SUBST([CUDA_HOME], [$CUDA_HOME])
    AC_SUBST([CUDA_SDK_PATH], [$CUDA_SDK_PATH])
    AC_SUBST([NVCC], [$NVCC])

    CUDA_FLAGS="$CUDA_INCLUDE -O3 -G -g -DBOOST_NOINLINE='__attribute__ ((noinline))' -DCUDA_ENABLED -arch=$cuda_arch --std=c++11 -DBOOST_NO_FENV_H"
    AC_SUBST([CUDA_FLAGS], [$CUDA_FLAGS])
    AC_SUBST([CUDA_LDFLAGS], [$CUDA_LDFLAGS])
    AC_SUBST([CUDA_ENABLED], [yes])
    AC_DEFINE_UNQUOTED([GLOB_MEM], [$glob_mem_bytes], [GPU global memory size (in bytes)])
  else
    AC_MSG_RESULT([no])
    AC_SUBST([CUDA_ENABLED], [no])
    AC_MSG_ERROR([Could not properly enable CUDA.])
  fi
fi
])

