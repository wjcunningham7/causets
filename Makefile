###########################
#(C) Will Cunningham 2014 #
# Krioukov Research Group #
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

# Reference for directories on Northeastern's Discovery cluster
HOST0=compute
# Reference for your personal directories (edit as needed)
# NOTE: If you are using HOST1 then make sure to edit all entries
# which reference it below with the correct directories
HOST1=tiberius

FASTSRC		:= $(LOCAL_DIR)/src/fastmath

#############################
# CUDA Resource Directories #
#############################

ifneq (, $(findstring $(HOST0), $(HOSTNAME)))
CUDA_SDK_PATH 	?= /shared/apps/cuda7.0/samples
CUDA_HOME 	?= /shared/apps/cuda7.0
else ifneq (, $(findstring $(HOST1), $(HOSTNAME)))
CUDA_SDK_PATH	?= /usr/local/cuda-7.5/samples
CUDA_HOME	?= /usr/local/cuda
else
$(error Cannot find CUDA directories!)
endif

#############
# Compilers #
#############

GCC		?= gcc
CXX 		?= g++
ifneq (, $(findstring $(HOST0), $(HOSTNAME)))
MPI		?= mpiCC
else ifneq (, $(findstring $(HOST1), $(HOSTNAME)))
MPI		?= /usr/lib64/openmpi/bin/mpicc
else
$(error Cannot find MPI compiler!)
endif
GFOR		?= gfortran
NVCC 		?= $(CUDA_HOME)/bin/nvcc

#########################
# Headers and Libraries #
#########################

INCD 		 = -I $(INCDIR) -I $(LOCAL_DIR)/include/
CUDA_INCD	 = -I $(CUDA_SDK_PATH)/common/inc -I $(CUDA_HOME)/include
LIBS		 = -L $(LOCAL_DIR)/lib64 -lstdc++ -lpthread -lm -lgsl -lgslcblas -lfastmath -lnint -lprintcolor -lgomp
CUDA_LIBS	 = -L /usr/lib/nvidia-current -L $(CUDA_HOME)/lib64/ -L $(CUDA_SDK_PATH)/common/lib -lcuda -lcudart

##################
# Compiler Flags #
##################

CXXFLAGS	:= -O3 -g -Wall -x c++
NVCCFLAGS 	:= -O3 -G -g --use_fast_math -DBOOST_NOINLINE='__attribute__ ((noinline))' -DCUDA_ENABLED #--keep --keep-dir $(ASMDIR)
ifneq (, $(findstring $(HOST0), $(HOSTNAME)))
NVCCFLAGS += -arch=sm_35
else ifneq (, $(findstring $(HOST1), $(HOSTNAME)))
NVCCFLAGS += -arch=sm_30
else
endif
OMPFLAGS1	:=
OMPFLAGS2	:=
MPIFLAGS1	:=
MPIFLAGS2	:=

##############################
# OpenMP or MPI Acceleration #
##############################

USE_OMP		:= 0
USE_MPI		:= 0

ifneq ($(USE_OMP), 0)
OMPFLAGS1 += -fopenmp
OMPFLAGS2 += -Xcompiler -fopenmp
endif

ifneq ($(USE_MPI), 0)
CXX=$(MPI)
MPIFLAGS1 += -DMPI_ENABLED -Wno-deprecated
MPIFLAGS2 += -DMPI_ENABLED -Xcompiler -Wno-deprecated
endif

CXXFLAGS += $(OMPFLAGS1) $(MPIFLAGS1)
NVCCFLAGS += $(OMPFLAGS2) $(MPIFLAGS2)

###############
# Source Code #
###############

CSOURCES	:= $(SRCDIR)/autocorr2.cpp
CEXTSOURCES	:= $(FASTSRC)/ran2.cpp $(FASTSRC)/stopwatch.cpp
CUDASOURCES	:= $(SRCDIR)/CuResources.cu $(SRCDIR)/Causet.cu $(SRCDIR)/Subroutines_GPU.cu $(SRCDIR)/Subroutines.cu $(SRCDIR)/Operations_GPU.cu $(SRCDIR)/NetworkCreator_GPU.cu $(SRCDIR)/Validate.cu $(SRCDIR)/NetworkCreator.cu $(SRCDIR)/Measurements_GPU.cu $(SRCDIR)/Measurements.cu
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

#cpu : check-env $(COBJS) $(CEXTOBJS) $(OBJS) link
cpu : check-env $(COBJS) $(OBJS) link

#gpu : check-env $(COBJS) $(CEXTOBJS) $(CUDAOBJS) linkgpu bin
gpu : check-env $(COBJS) $(CUDAOBJS) linkgpu bin

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
	$(CXX) -o $(BINDIR)/CausalSet $(OBJDIR)/*.o $(LIBS)

linkgpu : 
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJDIR)/*_cu.o -o $(OBJDIR)/linked.o

###################
# Binary Creation #
###################

bin : $(COBJS) $(CUDAOBJS)
	$(CXX) -o $(BINDIR)/CausalSet $(OBJDIR)/*.o $(INCD) $(CUDA_INCD) $(LIBS) $(CUDA_LIBS)

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

dirs : asmdir objdir bindir

asmdir :
	@ mkdir -p $(ASMDIR)

objdir :
	@ mkdir -p $(OBJDIR)

bindir : 
	@ mkdir -p $(BINDIR)

######################
# Cleaning Sequences #
######################

cleanall : clean cleanscratch cleandata

clean : cleanbin cleanasm cleanobj cleanlog

cleanbin :
	@ rm -rf $(BINDIR)

cleanasm :
	@ rm -rf $(ASMDIR)

cleanobj :
	@ rm -rf $(OBJDIR)

cleanlog :
	@ rm -f causet.log

cleanscratch :
	@ rm -rf /scratch/cunningham

cleandata :
	@ rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/emb/*.cset.emb.dat $(DATDIR)/emb/tn/*.cset.emb_fn.dat $(DATDIR)/emb/fp/*.cset.emb_fp.dat $(ETCDIR)/data_keys.cset.key $(DATDIR)/ref/*.ref $(DATDIR)/idf/*.cset.idf.dat $(DATDIR)/odf/*.cset.odf.dat $(DATDIR)/act/*.cset.act.dat
