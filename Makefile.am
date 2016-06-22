AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = src

INCD = -I $(top_builddir)/inc -I $(top_builddir)/opt -I $(top_builddir)/opt/fastmath/inc -I $(top_builddir)/opt/fastmath/opt
CUDA_INCD = -I $(CUDA_SDK_PATH)/common/inc -I $(CUDA_HOME)/include

LIBS = -lstdc++ -lpthread -lm -lgsl -lgslcblas
LOCAL_LIBS = -L $(top_builddir)/opt/fastmath/lib -lfastmath -L $(top_builddir)/opt/printcolor -lprintcolor
CUDA_LIBS = -L $(CUDA_HOME)/lib64 -L $(CUDA_SDK_PATH)/common/lib -lcuda -lcudart

.PHONY: $(SUBDIRS) linkgpu all cleanobj

$(SURDIRS):
	$(MAKE) --directory $@
	
linkgpu:
	@rm -f obj/linked.o
	$(NVCC) $(NVCCFLAGS) -dlink obj/*.o -o obj/linked.o

all: $(SUBDIRS) cleanobj linkgpu
	$(CXX) obj/*.o -o bin/CausalSet $(INCD) $(INCD_CUDA) $(LIBS) $(LOCAL_LIBS) $(CUDA_LIBS)

cleanobj:
	@mkdir -p obj
	mv src/*.o obj/
	@rm -f src/*.a
