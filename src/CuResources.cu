#include "CuResources.h"

#ifdef CUDA_ENABLED

//This will output the proper CUDA error strings in the event that a CUDA host call returns an error
void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err) {
    		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, getCudaDrvErrorString(err), file, line);
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 3);
		#else
    		exit(3);
		#endif
	}
}

//This will output the proper error string when calling cudaGetLastError
void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
    		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d).\n", file, line, errorMessage, (int)err);
		fprintf(stderr, "CUDA Error Message: %s\n", cudaGetErrorString(err));
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 4);
		#else
       		exit(4);
		#endif
	}
}

#endif

//These next three functions are loosely derived from shrQATest.h in the CUDA SDK

int printStart(const char **argv, const int &rank)
{
	int exename_start = findExeNameStart(argv[0]);
	printf_mpi(rank, "[%s] starting...\n\n", &(argv[0][exename_start]));
	fflush(stdout);
	if (rank == 0)
		printCPUInfo();
	return exename_start;
}

void printFinish(const char **argv, const int &exename_start, const int &rank, int iStatus)
{
	const char *sStatus[] = { "FAILED", "PASSED", "WAIVED", NULL };
	printf_mpi(rank, "[%s] results...\n%s\n\n", &(argv[0][exename_start]), sStatus[iStatus]);
	fflush(stdout);
}

int findExeNameStart(const char *exec_name)
{
	int exename_start = (int)strlen(exec_name);

	while (exename_start > 0 && 
	       exec_name[exename_start] != '\\' && 
	       exec_name[exename_start] != '/')
		exename_start--;

	if (exec_name[exename_start] == '\\' ||
	    exec_name[exename_start] == '/')
		return exename_start + 1;
	else
		return exename_start;
}

//Print Info about the CPU
void printCPUInfo()
{
	std::string line;
	std::ifstream finfo("/proc/cpuinfo");

	while (getline(finfo, line)) {
		std::stringstream str(line);
		std::string itype;
		std::string info;

		if (getline(str, itype, ':') &&
		    getline(str, info) &&
		    itype.substr(0, 10) == "model name") {
			printf("> Using Processor: %s\n", info.c_str());
			fflush(stdout);
			break;
		}
	}
}

//Print the Total Memory to the Terminal
void printMemUsed(char const * chkPoint, size_t hostMem, size_t devMem, const int &rank)
{
	if (chkPoint != NULL)
		printf_mpi(rank, "\nTotal Memory Used %s.........\n", chkPoint);
	else
		printf_mpi(rank, "\nMax Memory Used.........\n");
	printf_mpi(rank, "--------------------------------------------------------------\n");
	fflush(stdout);
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
		printf_mpi(rank, "%zd GB\t", GBytes);
	if (MBytes > 0)
		printf_mpi(rank, "%zd MB\t", MBytes);
	if (KBytes > 0)
		printf_mpi(rank, "%zd KB\t", KBytes);
	if (bytes  > 0)
		printf_mpi(rank, "%zd bytes\t", bytes);
	if (GBytes > 0 || MBytes > 0 || KBytes > 0 || bytes > 0)
		printf_mpi(rank, " [HOST]\n\n");
	fflush(stdout);

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
		printf_mpi(rank, "%zd GB\t", GBytes);
	if (MBytes > 0)
		printf_mpi(rank, "%zd MB\t", MBytes);
	if (KBytes > 0)
		printf_mpi(rank, "%zd KB\t", KBytes);
	if (bytes  > 0)
		printf_mpi(rank, "%zd bytes\t", bytes);
	if (GBytes > 0 || MBytes > 0 || KBytes > 0 || bytes > 0)
		printf_mpi(rank, "[DEVICE]\n\n");
	fflush(stdout);
}

//Used to keep track of max memory used
void memoryCheckpoint(const size_t &hostMemUsed, size_t &maxHostMemUsed, const size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (hostMemUsed > maxHostMemUsed)
		maxHostMemUsed = hostMemUsed;
	if (devMemUsed > maxDevMemUsed)
		maxDevMemUsed = devMemUsed;
}

//Print 'CHECKPOINT' for debugging
void printChk()
{
	printf("CHECKPOINT\n");
}

#ifdef CUDA_ENABLED

//Initialize connection to GPU
void connectToGPU(Resources *resources, int argc, char **argv, const int &rank)
{
	//No null pointers
	if (CU_DEBUG) assert (resources != NULL);

	//Pick CUDA Device
	CUresult status;
	int devCount, major = 0, minor = 0;

	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGetCount(&devCount));
	if (CU_DEBUG) assert(resources->gpuID > -1 && resources->gpuID < devCount);
	resources->cuDevice = findCudaDevice(resources->gpuID, rank);

	checkCudaErrors(cuDeviceComputeCapability(&major, &minor, resources->cuDevice));

	//Statistics about the Device
	printf_mpi(rank, "> GPU device has SM %d.%d compute capabilities\n", major, minor);
	int version = (major * 0x10 + minor);
	if (version < 0x11) {
		printf("Program requires a minimum CUDA compute capability of 1.1\n");
		printFinish((const char**)argv, findExeNameStart(argv[0]), 0, FAILED);
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 1);
		#else
		exit(1);
		#endif
	}
	fflush(stdout);

	//Create Context
	status = cuCtxCreate(&resources->cuContext, CU_CTX_SCHED_SPIN, resources->cuDevice);
	if (status != CUDA_SUCCESS) {
		printf("Could not create CUDA context!\n");
		cuCtxDetach(resources->cuContext);
		printFinish((const char**)argv, findExeNameStart(argv[0]), 0, FAILED);
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 2);
		#else
		exit(2);
		#endif
	}
	fflush(stdout);
}

//Find available GPUs
CUdevice findCudaDevice(int id, const int &rank)
{
    	CUdevice device;
      	char name[100];

	checkCudaErrors(cuDeviceGet(&device, id));
	cuDeviceGetName(name, 100, device);
	printf_mpi(rank, "> Using CUDA Device [%d]: %s\n", id, name);
	fflush(stdout);

    	return device;
}

#endif
