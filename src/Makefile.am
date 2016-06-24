AUTOMAKE_OPTIONS = foreign

ACLOCAL_AMFLAGS = -I m4 --install
SUFFIXES = .cu

INCD = -I $(top_builddir)/inc -I $(top_builddir)/printcolor

noinst_LIBRARIES = libcauset.a
if USE_GPU
libcauset_a_SOURCES = autocorr2.cu CuResources.cu Causet.cu Subroutines_GPU.cu Subroutines.cu Operations_GPU.cu NetworkCreator_GPU.cu Validate.cu NetworkCreator.cu Measurements.cu
else
libcauset_a_SOURCES = autocorr2.cu CuResources.cu Causet.cu Subroutines.cu Validate.cu NetworkCreator.cu Measurements.cu
endif

.cu.$(OBJEXT):
	$(COMPILE_STR) $(INCD) -o $@ $<
