###########################
#(C) Will Cunningham 2014 #
# Krioukov Research Group #
# Northeastern University #
###########################

BINDIR		:= ./bin
INCDIR		:= ./inc
SRCDIR		:= ./src
OBJDIR		:= ./obj
DATDIR		:= ./dat
ETCDIR		:= ./etc

LOCAL_DIR	:= /home/cunningham.wi/local

FASTSRC		:= $(LOCAL_DIR)/src/fastmath
 
CUDA_SDK_PATH 	?= /shared/apps/cuda6.0/samples
CUDA_HOME 	?= /shared/apps/cuda6.0
 
GCC		?= /usr/bin/gcc
CXX 		?= /usr/bin/g++
MPI		?= /opt/ibm/platform_mpi/bin/mpicc
GFOR		?= /usr/bin/gfortran
NVCC 		?= $(CUDA_HOME)/bin/nvcc
#COMPILER	?= $(CXX)

INCD 		 = -I $(INCDIR) -I $(LOCAL_DIR)/inc/
CUDA_INCD	 = -I $(CUDA_SDK_PATH)/common/inc -I $(CUDA_HOME)/include
LIBS		 = -L $(LOCAL_DIR)/lib64 -lstdc++ -lpthread -lm -lgsl -lgslcblas -lfastmath -lnint -lgomp -lprintcolor
CUDA_LIBS	 = -L /usr/lib/nvidia-current -L $(CUDA_HOME)/lib64/ -L $(CUDA_SDK_PATH)/common/lib -lcuda -lcudart

CXXFLAGS	:= -O3 -g -Wall -x c++
NVCCFLAGS 	:= -arch=sm_35 -m64 -O3 -G -g --use_fast_math -DBOOST_NOINLINE='__attribute__ ((noinline))' -DCUDA_ENABLED
OMPFLAGS	:=
MPIFLAGS	:=
#COMPILER_FLAGS	:= $(CXXFLAGS)

USE_CUDA	:= 0
USE_OMP		:= 0
USE_MPI		:= 0

ifneq ($(USE_CUDA), 0)
#	COMPILER = $(NVCC)
#	COMPILER_FLAGS = $(NVCCFLAGS)
	INCD += $(CUDA_INCD)
	LIBS += $(CUDA_LIBS)
endif

ifneq ($(USE_OMP), 0)
   	OMP_FLAGS += -Xcompiler -fopenmp
endif

ifneq ($(USE_MPI), 0)
	CXX=$(MPI)
	MPIFLAGS += -DMPI_ENABLED -Xcompiler -Wno-deprecated
endif

CSOURCES	:= $(SRCDIR)/autocorr2.cpp
CEXTSOURCES	:= $(FASTSRC)/ran2.cpp $(FASTSRC)/stopwatch.cpp
CUDASOURCES	:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines_GPU.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Operations_GPU.cu $(SRCDIR)/NetworkCreator_GPU.cu $(SRCDIR)/Validate.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
SOURCES		:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Validate.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
FSOURCES1	:= $(SRCDIR)/ds3.F
FSOURCES2	:= $(SRCDIR)/Matter_Dark_Energy_downscaled.f

COBJS		:= $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CSOURCES))
CEXTOBJS	:= $(patsubst $(FASTSRC)/%.cpp, $(OBJDIR)/%.o, $(CEXTSOURCES))
CUDAOBJS	:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%_cu.o, $(CUDASOURCES))
OBJS		:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SOURCES))

all : $(COBJS) $(CEXTOBJS) $(OBJS) link

gpu : $(COBJS) $(CEXTOBJS) $(CUDAOBJS) linkgpu bin

$(COBJS) : | dirs

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -I $(INCDIR) -o $@ $<

dirs : objdir bindir

objdir :
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o : $(FASTSRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $(INCD) -o $@ $<

$(OBJDIR)/%_cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc $(INCD) $(CUDA_INCD) -o $@ $<

$(OBJDIR)/%.o : $(SRCDIR)/%.cu
	$(CXX) $(CXXFLAGS) -c $(INCD) -o $@ $<

link :
	$(CXX) -o $(BINDIR)/CausalSet $(OBJDIR)/*.o $(LIBS)

linkgpu : 
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJDIR)/*_cu.o -o $(OBJDIR)/linked.o

bin : $(COBJS) $(CEXTOBJS) $(CUDAOBJS)
	$(CXX) -o $(BINDIR)/CausalSet $(OBJDIR)/*.o $(INCD) $(CUDA_INCD) $(LIBS) $(CUDA_LIBS)

bindir : 
	mkdir -p $(BINDIR)

fortran : bindir fortran1 fortran2

fortran1 : $(FSOURCES1)
	$(GFOR) $(FSOURCES1) -o $(BINDIR)/ds3

fortran2 : $(FSOURCES2)
	$(GFOR) $(FSOURCES2) -o $(BINDIR)/universe

clean : cleanbin cleanobj cleanlog

cleanbin :
	rm -rf $(BINDIR)

cleanobj :
	rm -rf $(OBJDIR)

cleanlog :
	rm -f causet.log

cleanscratch :
	rm -rf /scratch/cunningham

cleandata :
	rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/emb/*.cset.emb.dat $(DATDIR)/emb/tn/*.cset.emb_tn.dat $(DATDIR)/emb/fp/*.cset.emb_fp.dat $(ETCDIR)/data_keys.cset.key $(DATDIR)/ref/*.ref $(DATDIR)/idf/*.cset.idf.dat $(DATDIR)/odf/*.cset.odf.dat
