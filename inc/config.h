/* inc/config.h.  Generated from config.h.in by configure.  */
/* inc/config.h.in.  Generated from configure.ac by autoheader.  */

/* GPU architecture passed to nvcc with -arch flag. */
#define CUDA_ARCH "sm_35"

/* CUDA installation directory. */
#define CUDA_HOME "/shared/apps/cuda7.0"

/* CUDA SDK directory. */
#define CUDA_SDK_PATH "/shared/apps/cuda7.0/samples"

/* Define if avx2 instructions are supported */
#define HAVE_AVX2_INSTRUCTIONS 1

/* Defined if the requested minimum BOOST version is satisfied */
#define HAVE_BOOST 1

/* Cuda is detected and will be enabled. */
#define HAVE_CUDA_INSTRUCTIONS 1

/* define if the compiler supports basic C++11 syntax */
#define HAVE_CXX11 1

/* Define if popcnt instructions are supported */
#define HAVE_POPCNT_INSTRUCTIONS 1

/* Path for nvcc compiler. */
#define NVCC "/shared/apps/cuda7.0/bin/nvcc"

/* Name of package */
#define PACKAGE "causalset"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "w.cunningham@northeastern.edu"

/* Define to the full name of this package. */
#define PACKAGE_NAME "CausalSet"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "CausalSet 1.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "causalset"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.1"

/* Version number of package */
#define VERSION "1.1"
