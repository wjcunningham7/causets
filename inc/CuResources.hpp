#ifndef CU_RESOURCES_H_
#define CU_RESOURCES_H_

#include <cuda.h>
#include <drvapi_error_string.h>
#include <iostream>
#include <string>

#include "shrQATest.h"
//#include "shrUtils.h"
#include "stopwatch.h"

#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line);
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line );
void printMemUsed(char *chkPoint, size_t &hostMem, size_t &devMem);
void memoryCheckpoint(void);
void connectToGPU(int argc, char **argv);
inline CUdevice findCudaDevice();

//CUDA Driver API Variables
CUdevice cuDevice;
CUcontext cuContext;
int gpuID = 0;

//Memory Allocated (in bytes)
size_t hostMemUsed = 0, maxHostMemUsed = 0;
size_t devMemUsed = 0, maxDevMemUsed = 0;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err) {
    		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
               	 err, getCudaDrvErrorString(err), file, line );
    		exit(-1);
	}
}

// This will output the proper error string when calling cudaGetLastError
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
    		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d).\n",
        	        file, line, errorMessage, (int)err);
       	exit(-1);
	}
}

//Print the Total Memory to the Terminal
void printMemUsed(char *chkPoint, size_t &hostMem, size_t &devMem)
{
	if (chkPoint != NULL)
		printf("\nTotal Memory Used %s.........\n", chkPoint);
	else
		printf("\nMax Memory Used.........\n");
	printf("--------------------------------------------------------------\n");
	size_t bytes = 0, KBytes = 0, MBytes = 0, GBytes = 0;

	if (hostMem >= pow(2.0, 30))
		GBytes = hostMem / pow(2.0, 30);
	if (hostMem - (GBytes * pow(2.0, 30)) >= pow(2.0, 20))
		MBytes = (hostMem - (GBytes * pow(2.0, 30))) / pow(2.0, 20);
	if (hostMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) >= pow(2.0, 10))
		KBytes = (hostMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20))) / pow(2.0, 10);
	if (hostMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) - (KBytes * pow(2.0, 10)) > 0)
		bytes = hostMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) - (KBytes * pow(2.0, 10));

	if (GBytes > 0)
		printf("%i GB\t", GBytes);
	if (MBytes > 0)
		printf("%i MB\t", MBytes);
	if (KBytes > 0)
		printf("%i KB\t", KBytes);
	if (bytes  > 0)
		printf("%i bytes\t", bytes);
	if (GBytes > 0 || MBytes > 0 || KBytes > 0 || bytes > 0)
		printf(" [HOST]\n\n");

	bytes = 0;	KBytes = 0;	MBytes = 0;	GBytes = 0;
	if (devMem >= pow(2.0, 30))
		GBytes = devMem / pow(2.0, 30);
	if (devMem - (GBytes * pow(2.0, 30)) >= pow(2.0, 20))
		MBytes = (devMem - (GBytes * pow(2.0, 30))) / pow(2.0, 20);
	if (devMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) >= pow(2.0, 10))
		KBytes = (devMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20))) / pow(2.0, 10);
	if (devMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) - (KBytes * pow(2.0, 10)) > 0)
		bytes = devMem - (GBytes * pow(2.0, 30)) - (MBytes * pow(2.0, 20)) - (KBytes * pow(2.0, 10));

	if (GBytes > 0)
		printf("%i GB\t", GBytes);
	if (MBytes > 0)
		printf("%i MB\t", MBytes);
	if (KBytes > 0)
		printf("%i KB\t", KBytes);
	if (bytes  > 0)
		printf("%i bytes\t", bytes);
	if (GBytes > 0 || MBytes > 0 || KBytes > 0 || bytes > 0)
		printf("[DEVICE]\n\n");
}

//Used to keep track of max memory used
void memoryCheckpoint(void)
{
	if (hostMemUsed > maxHostMemUsed)
		maxHostMemUsed = hostMemUsed;
	if (devMemUsed > maxDevMemUsed)
		maxDevMemUsed = devMemUsed;
}

//Initialize connection to GPU
void connectToGPU(int argc, char **argv)
{
	//Pick CUDA Device
	CUresult status;
	int devCount, major = 0, minor = 0;

	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGetCount(&devCount));
	assert(gpuID > -1 && gpuID < devCount);
	cuDevice = findCudaDevice();

	checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));

	//Statistics about the Device
	printf("> GPU device has SM %d.%d compute capabilities\n\n", major, minor);
	int version = (major * 0x10 + minor);
	if (version < 0x11) {
		printf("Program requires a minimum CUDA compute capability of 1.1\n");
		shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}

	//Create Context
	status = cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
	if (status != CUDA_SUCCESS) {
		printf("Could not create CUDA context!\n");
		cuCtxDetach(cuContext);
		shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}
}

//Find available GPUs
inline CUdevice findCudaDevice()
{
    	CUdevice device;
      	char name[100];

       checkCudaErrors(cuDeviceGet(&device, gpuID));
	cuDeviceGetName(name, 100, device);
       printf("> Using CUDA Device [%d]: %s\n", gpuID, name);

    	return device;
}

#endif
