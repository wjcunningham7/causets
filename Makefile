BINDIR		:= ./bin
INCDIR		:= ./inc
SRCDIR		:= ./src
OBJDIR		:= ./obj
DATDIR		:= ./dat
 
CUDA_SDK_PATH 	?= /usr/local/cuda-5.0/samples
CUDA_HOME 	?= /usr/local/cuda
 
GCC		?= /usr/bin/gcc
CXX 		?= /usr/bin/g++
NVCC 		?= $(CUDA_HOME)/bin/nvcc
INCD 		 = -I"$(CUDA_SDK_PATH)/common/inc" -I"$(CUDA_HOME)/include" -I$(INCDIR)
LIBS 		 = -L/usr/lib/nvidia-current/ -lcuda -L$(LD_LIBRARY_PATH) -L$(CUDA_HOME)/lib64/ -lcudart -lcurand -L"$(CUDA_SDK_PATH)/common/lib" -lstdc++ -lpthread -lm -lGLU -lglut

CXXFLAGS	:= -O3 -g
NVCCFLAGS 	:= -arch=sm_30 -O3 -G -g

CSOURCES	:= $(SRCDIR)/stopwatch.cpp $(SRCDIR)/ran2.cpp $(SRCDIR)/autocorr2.cpp $(SRCDIR)/FastMath.cpp
CHEADERS	:= $(INCDIR)/stopwatch.h $(INCDIR)/ran2.h $(INCDIR)/autocorr2.h $(INCDIR)/fastapprox.h $(INCDIR)/FastMath.h
COBJS		:= $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CSOURCES))

SOURCES		:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Operations.cu $(SRCDIR)/GPUSubroutines.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
HEADERS		:= $(INCDIR)/shrQATest.h $(INCDIR)/shrUtils.h $(INCDIR)/CuResources.h $(INCDIR)/Causet.h $(INCDIR)/Subroutines.h $(INCDIR)/Operations.h $(INDIR)/GPUSubroutines.h $(INCDIR)/NetworkCreator.h $(INCDIR)/Measurements.h
OBJS		:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.cu_o, $(SOURCES))

all: $(COBJS) $(OBJS) $(BINDIR) clean
 
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -I$(INCDIR) -o $@ $<

$(OBJDIR)/%.cu_o : $(SRCDIR)/%.cu 
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<
 
$(BINDIR): $(OBJS)
	$(CXX) -o $(BINDIR)/CausalSet $(COBJS) $(OBJS) $(INCD) $(LIBS)
 
clean:
	rm -f $(OBJDIR)/*.cu_o $(OBJDIR)/*.o

cleandata:
	rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/data_keys.key
