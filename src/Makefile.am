###########################
#(C) Will Cunningham 2017 #
#         DK Lab          #
# Northeastern University #
###########################

AUTOMAKE_OPTIONS = foreign

ACLOCAL_AMFLAGS = -I m4 --install
SUFFIXES = .cu

INCD = -I $(top_builddir)/inc

noinst_LIBRARIES = libcauset.a
if USE_GPU
libcauset_a_SOURCES = CuResources.cu Causet.cu Subroutines_GPU.cu Subroutines.cu MarkovChain.cu Operations_GPU.cu NetworkCreator_GPU.cu Validate.cu NetworkCreator.cu Measurements.cu Anneal.cu
else
libcauset_a_SOURCES = CuResources.cu Causet.cu Subroutines.cu MarkovChain.cu Validate.cu NetworkCreator.cu Measurements.cu
endif

.cu.$(OBJEXT):
	$(COMPILE_STR) $(INCD) -o $@ $<
