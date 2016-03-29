###########################
#(C) Will Cunningham 2014 #
#         DK Lab          #
# Northeastern University #
###########################

#####################
# Local Directories #
#####################

BINDIR		:= ./bin
INCDIR		:= ./inc
SRCDIR		:= ./src
ASMDIR		:= ./asm
OBJDIR		:= ./obj
DATDIR		:= ./dat
ETCDIR		:= ./etc

LOCAL_DIR	:= $(CAUSET_LOCAL_DIR)
FASTSRC		:= $(FAST_LOCAL_DIR)/src/fastmath

#############################
# CUDA Resource Directories #
#############################

CUDA_SDK_PATH 	?= /shared/apps/cuda7.0/samples
CUDA_HOME 	?= /shared/apps/cuda7.0

#############
# Compilers #
#############

GCC		?= gcc
CXX 		?= g++
MPI		?= mpiCC
GFOR		?= gfortran
NVCC 		?= /shared/apps/cuda7.0/bin/nvcc

#########################
# Headers and Libraries #
#########################

INCD 		 = -I $(INCDIR) -I $(LOCAL_DIR)/include/
CUDA_INCD	 = -I /shared/apps/cuda7.0/samples/common/inc -I /shared/apps/cuda7.0/include
LIBS		 = -L $(LOCAL_DIR)/lib64 -lstdc++ -lpthread -lm -lgsl -lgslcblas
LOCAL_LIBS	 = -lnint -lprintcolor
CUDA_LIBS	 = -L /usr/lib/nvidia-current -L /shared/apps/cuda7.0/lib64 -L /shared/apps/cuda7.0/samples/common/lib -lcuda -lcudart

##################
# Compiler Flags #
##################

CXXFLAGS	:= -O3 -g -Wall -x c++
NVCCFLAGS 	:= -O3 -G -g -DBOOST_NOINLINE='__attribute__ ((noinline))' -DCUDA_ENABLED -arch=sm_35

################
# Acceleration #
################

USE_AVX2	:= 1
USE_OMP		:= 1
USE_MPI		:= 0

ifneq ($(USE_OMP), 0)
CXXFLAGS += -fopenmp
NVCCFLAGS += -Xcompiler -fopenmp
LIBS += -lgomp
endif

ifneq ($(USE_MPI), 0)
CXX=$(MPI)
CXXFLAGS += -DMPI_ENABLED -Wno-deprecated
NVCCFLAGS += -DMPI_ENABLED -Xcompiler -Wno-deprecated
endif

ifneq ($(USE_AVX2), 0)
CXXFLAGS += -mavx2 -march=core-avx2 -mtune=core-avx2 -mpopcnt -DAVX2_ENABLED
NVCCFLAGS += -Xcompiler "-mavx2 -march=core-avx2 -mtune=core-avx2 -mpopcnt -DAVX2_ENABLED"
LIBS += -lfastmathavx
else
LIBS += -lfastmath
endif

###############
# Source Code #
###############

CSOURCES	:= $(SRCDIR)/autocorr2.cpp
CEXTSOURCES	:= $(FASTSRC)/ran2.cpp $(FASTSRC)/stopwatch.cpp
CUDASOURCES	:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines_GPU.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Operations_GPU.cu $(SRCDIR)/NetworkCreator_GPU.cu $(SRCDIR)/Validate.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
SOURCES		:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Validate.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements.cu
FSOURCES1	:= $(SRCDIR)/ds3.F
FSOURCES2	:= $(SRCDIR)/Matter_Dark_Energy_downscaled.f

################
# Object Files #
################

COBJS		:= $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CSOURCES))
CEXTOBJS	:= $(patsubst $(FASTSRC)/%.cpp, $(OBJDIR)/%.o, $(CEXTSOURCES))
CUDAOBJS	:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%_cu.o, $(CUDASOURCES))
OBJS		:= $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SOURCES))

#################
# Phony Targets #
#################

.PHONY : check-env cpu gpu link linkgpu bin fortran fortran1 fortran2 dirs objdir bindir clean cleanall cleanbin cleanobj cleanlog cleanscratch cleandata

###################################
# Top-Level Compilation Sequences #
###################################

all : gpu

cpu : check-env $(COBJS) $(OBJS) link

gpu : check-env dirs $(COBJS) $(CUDAOBJS) linkgpu bin

###############################
# Check Environment Variables #
###############################

check-env :
	@ if test "$(LOCAL_DIR)" = "" ; then \
		echo "LOCAL_DIR not set!"; \
		exit 1; \
	fi

######################
# Source Compilation #
######################

$(COBJS) : | dirs

$(OBJS) : | dirs

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -I $(INCDIR) -o $@ $<

$(OBJDIR)/%.o : $(FASTSRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $(INCD) -o $@ $<

$(OBJDIR)/%.o : $(SRCDIR)/%.cu
	$(CXX) $(CXXFLAGS) -c $(INCD) -o $@ $<

$(OBJDIR)/%_cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc $(INCD) $(CUDA_INCD) -o $@ $<

##################
# Object Linkage #
##################

link :
	$(CXX) -o $(BINDIR)/CausalSet_debug $(OBJDIR)/*.o $(LIBS) $(LOCAL_LIBS)

linkgpu : 
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJDIR)/*_cu.o -o $(OBJDIR)/linked.o

###################
# Binary Creation #
###################

bin : $(COBJS) $(CUDAOBJS)
	$(CXX) -o $(BINDIR)/CausalSet_debug $(OBJDIR)/*.o $(INCD) $(CUDA_INCD) $(LIBS) $(LOCAL_LIBS) $(CUDA_LIBS)

#######################
# Fortran Compilation #
#######################

fortran : bindir fortran1 fortran2

fortran1 : $(FSOURCES1)
	$(GFOR) $(FSOURCES1) -o $(BINDIR)/ds3

fortran2 : $(FSOURCES2)
	$(GFOR) $(FSOURCES2) -o $(BINDIR)/universe

######################
# Directory Creation #
######################

dirs : objdir bindir

asmdir :
	@ mkdir -p $(ASMDIR)

objdir :
	@ mkdir -p $(OBJDIR)

bindir : 
	@ mkdir -p $(BINDIR)

######################
# Cleaning Sequences #
######################

cleanall : clean cleanscratch cleandata cleandbg

clean : cleanasm cleanobj cleanlog

cleanbin :
	@ rm -rf $(BINDIR)

cleanasm :
	@ rm -rf $(ASMDIR)

cleanobj :
	@ rm -rf $(OBJDIR)

cleanlog :
	@ rm -f causet.log

cleanscratch :
	@ rm -rf /gss_gpfs_scratch/cunningham/*

cleandata :
	@ rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/emb/*.cset.emb.dat $(DATDIR)/emb/tn/*.cset.emb_fn.dat $(DATDIR)/emb/fp/*.cset.emb_fp.dat $(ETCDIR)/data_keys.cset.key $(DATDIR)/ref/*.ref $(DATDIR)/idf/*.cset.idf.dat $(DATDIR)/odf/*.cset.odf.dat $(DATDIR)/act/*.cset.act.dat

cleandbg :
	@ rm -f *.dbg.dat
