BINDIR 	:= ./bin
INCDIR		:= ./inc
SRCDIR		:= ./src
DATDIR		:= ./dat
 
CUDA_SDK_PATH 	?= /usr/local/cuda-5.0/samples
CUDA_HOME 	?= /usr/local/cuda
 
NVCC 		?= $(CUDA_HOME)/bin/nvcc
CXX 		?= /usr/bin/g++
INCD 		 = -I"$(CUDA_SDK_PATH)/common/inc" -I"$(CUDA_HOME)/include" -I"./" -I$(INCDIR)
LIBS 		 = -L/usr/lib/nvidia-current/ -lcuda -L$(LD_LIBRARY_PATH) -L$(CUDA_HOME)/lib64/ -lcudart -lcurand -L"$(CUDA_SDK_PATH)/common/lib" -lstdc++ -lpthread -lm -lGLU -lglut

NVCCFLAGS 	:= -arch=sm_30 -O3 -G -g

SOURCES		:= $(SRCDIR)/Causet.cu
HEADERS		:= $(INCDIR)/shrQATest.h $(INCDIR)/shrUtils.h $(INCDIR)/stopwatch.h $(INCDIR)/ran2.h $(INCDIR)/CuResources.h $(INCDIR)/Causet.h $(INCDIR)/Subroutines.cu $(INCDIR)/Operations.cu $(INCDIR)/GPUSubroutines.cu $(INCDIR)/NetworkCreator.cu $(INCDIR)/Measurements.cu 
OBJS		:= $(patsubst %.cu, %.cu_o, $(SOURCES))
 
%.cu_o : %.cu 
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<
 
$(BINDIR): $(OBJS) $(HEADERS)
	$(CXX) -o $(BINDIR)/CausalSet $(OBJS) $(INCD) $(LIBS)
 
clean:
	rm -f $(SRCDIR)/*.cu_o ./*.pyc ./*.mek

cleandata:
	rm -f $(DATDIR)/*.cset $(DATDIR)/pos/*.cset $(DATDIR)/edg/*.cset $(DATDIR)/dst/*.cset $(DATDIR)/cls/*.cset $(DATDIR)/cdk/*.cset
