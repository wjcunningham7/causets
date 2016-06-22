AUTOMAKE_OPTIONS = foreign

ACLOCAL_AMFLAGS = -I m4 --install
SUFFIXES = .cu

INCD = -I $(top_builddir)/inc -I $(top_builddir)/opt -I $(top_builddir)/opt/fastmath/inc -I $(top_builddir)/opt/fastmath/opt
CUDA_INCD = -I $(CUDA_SDK_PATH)/common/inc -I $(CUDA_HOME)/include

noinst_LIBRARIES = libcauset.a
libcauset_a_SOURCES = autocorr2.cu CuResources.cu Causet.cu Subroutines_GPU.cu Subroutines.cu Operations_GPU.cu NetworkCreator_GPU.cu Validate.cu NetworkCreator.cu Measurements.cu

.cu.$(OBJEXT):
	$(NVCC) $(NVCCFLAGS) -dc $(INCD) $(CUDA_INCD) -o $@ $<
