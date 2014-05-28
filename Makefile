BINDIR		:= ./bin
INCDIR		:= ./inc
SRCDIR		:= ./src
OBJDIR		:= ./obj
DATDIR		:= ./dat

FASTSRC		:= /usr/local/src/fastmath
 
CUDA_SDK_PATH 	?= /usr/local/cuda-5.0/samples
CUDA_HOME 	?= /usr/local/cuda
 
GCC		?= /usr/bin/gcc
CXX 		?= /usr/bin/g++
GFOR		?= /usr/bin/gfortran
NVCC 		?= $(CUDA_HOME)/bin/nvcc
INCD 		 = -I $(CUDA_SDK_PATH)/common/inc -I $(CUDA_HOME)/include -I $(INCDIR)
LIBS 		 = -L /usr/lib/nvidia-current/ -lcuda -L $(LD_LIBRARY_PATH) -L $(CUDA_HOME)/lib64/ -lcudart -lcurand -L $(CUDA_SDK_PATH)/common/lib -lstdc++ -lpthread -lm -lGLU -lglut -lnint -lgsl -lgslcblas -lfastmath

CXXFLAGS	:= -O3 -g
NVCCFLAGS 	:= -arch=sm_30 -O3 -G -g

CSOURCES	:= $(SRCDIR)/autocorr2.cpp
CEXTSOURCES	:= $(FASTSRC)/ran2.cpp $(FASTSRC)/stopwatch.cpp 
SOURCES		:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Operations.cu $(SRCDIR)/GPUSubroutines.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
FSOURCES1	:= $(SRCDIR)/ds3.F
FSOURCES2	:= $(SRCDIR)/Matter_Dark_Energy_downscaled.f

COBJS		:= $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CSOURCES))
CEXTOBJS	:= $(patsubst $(FASTSRC)/%.cpp, $(OBJDIR)/%.o, $(CEXTSOURCES))
OBJS		:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.cu_o, $(SOURCES))

all : $(COBJS) $(CEXTOBJS) $(OBJS) bindir bin fortran cleanlog

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -I $(INCDIR) -o $@ $<

$(COBJS) : | $(OBJDIR)

$(OBJDIR) :
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o : $(FASTSRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c -I $(INCDIR) -o $@ $<

$(OBJDIR)/%.cu_o : $(SRCDIR)/%.cu 
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<
 
bin : $(COBJS) $(CEXTOBJS) $(OBJS)
	$(CXX) -o $(BINDIR)/CausalSet $(COBJS) $(CEXTOBJS) $(OBJS) $(INCD) $(LIBS)

bindir : 
	mkdir -p $(BINDIR)

fortran : fortran1 fortran2

fortran1 : $(FSOURCES1)
	$(GFOR) $(FSOURCES1) -o $(BINDIR)/ds3

fortran2 : $(FSOURCES2)
	$(GFOR) $(FSOURCES2) -o $(BINDIR)/universe

clean : cleanbin
	rm -f $(OBJDIR)/*.cu_o $(OBJDIR)/*.o

cleanbin :
	rm -f $(BINDIR)/*

cleanlog :
	rm -f ./causet.log ./causet.err

cleandata :
	rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/data_keys.key
