#ifndef CU_RESOURCES_H_
#define CU_RESOURCES_H_

#include <cassert>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

#ifdef CUDA_ENABLED
#include <cuda.h>
#include <drvapi_error_string.h>
#endif

#include <fastmath/stopwatch.h>
#include "Subroutines.h"

#define CU_DEBUG false

#ifdef CUDA_ENABLED
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)
#else
typedef int CUdevice;
typedef int CUcontext;
#endif

enum Status {
	FAILED = 0,
	PASSED = 1,
	WAIVED = 2
};

struct Resources {
	Resources() : cuDevice(0), cuContext(0), gpuID(0), hostMemUsed(0), maxHostMemUsed(0), devMemUsed(0), maxDevMemUsed(0) {}

	//CUDA Driver API Variables
	CUdevice cuDevice;
	CUcontext cuContext;

	//GPU Identification Number
	int gpuID;

	//Memory Allocated (in bytes)
	size_t hostMemUsed;
	size_t maxHostMemUsed;
	size_t devMemUsed;
	size_t maxDevMemUsed;
};		

#ifdef CUDA_ENABLED
void __checkCudaErrors(CUresult err, const char *file, const int line);
void __getLastCudaError(const char *errorMessage, const char *file, const int line);
#endif

int printStart(const char **argv, const int &rank);
void printFinish(const char **argv, const int &exename_start, const int &rank, int iStatus);
int findExeNameStart(const char *exec_name);
void printCPUInfo();

void printMemUsed(char const * chkPoint, size_t hostMem, size_t devMem, const int &rank);
void memoryCheckpoint(const size_t &hostMemUsed, size_t &maxHostMemUsed, const size_t &devMemUsed, size_t &maxDevMemUsed);
void printChk();

#ifdef CUDA_ENABLED
void connectToGPU(Resources *resources, int argc, char **argv, const int &rank);
CUdevice findCudaDevice(int id, const int &rank);
#endif

#endif
