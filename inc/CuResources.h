#ifndef CU_RESOURCES_H_
#define CU_RESOURCES_H_

#include <assert.h>
#include <iostream>
#include <string>

#include <cuda.h>
#include <drvapi_error_string.h>
#include <shrQATest.h>

#include <fastmath/stopwatch.h>

#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)

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

void __checkCudaErrors(CUresult err, const char *file, const int line);
void __getLastCudaError( const char *errorMessage, const char *file, const int line );
void printMemUsed(char *chkPoint, size_t hostMem, size_t devMem);
void memoryCheckpoint(const size_t &hostMemUsed, size_t &maxHostMemUsed, const size_t &devMemUsed, size_t &maxDevMemUsed);
void connectToGPU(Resources *resources, int argc, char **argv);
CUdevice findCudaDevice(int id);

#endif
