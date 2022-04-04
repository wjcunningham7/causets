/////////////////////////////
//(C) Will Cunningham 2017 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "CuResources.h"

//Print the Total Memory to the Terminal
void printMemUsed(char const * chkPoint, size_t hostMem, size_t devMem, const int &rank)
{
	if (chkPoint != NULL)
		printf_mpi(rank, "\nTotal Memory Used %s.........\n", chkPoint);
	else
		printf_mpi(rank, "\nMax Memory Used.........\n");
	printf_mpi(rank, "--------------------------------------------------------------\n");
	fflush(stdout);
	size_t GB = pow(10.0, 9.0);
	size_t MB = pow(10.0, 6.0);
	size_t KB = pow(10.0, 3.0);
	size_t bytes = 0, KBytes = 0, MBytes = 0, GBytes = 0;

	if (hostMem >= GB) {
		GBytes = hostMem / GB;
		hostMem -= GBytes * GB;
	}
	if (hostMem >= MB) {
		MBytes = hostMem / MB;
		hostMem -= MBytes * MB;
	}
	if (hostMem >= KB) {
		KBytes = hostMem / KB;
		hostMem -= KBytes * KB;
	}
	bytes = hostMem;

	if (GBytes > 0) printf_mpi(rank, "%zd GB\t", GBytes);
	if (MBytes > 0) printf_mpi(rank, "%zd MB\t", MBytes);
	if (KBytes > 0) printf_mpi(rank, "%zd KB\t", KBytes);
	if (bytes  > 0) printf_mpi(rank, "%zd bytes\t", bytes);
	if (GBytes > 0 || MBytes > 0 || KBytes > 0 || bytes > 0)
		printf_mpi(rank, " [HOST]\n\n");
	fflush(stdout);

	GBytes = MBytes = KBytes = bytes = 0;

	if (devMem >= GB) {
		GBytes = devMem / GB;
		devMem -= GBytes * GB;
	}
	if (devMem >= MB) {
		MBytes = devMem / MB;
		devMem -= MBytes * MB;
	}
	if (devMem >= KB) {
		KBytes = devMem / KB;
		devMem -= KBytes * KB;
	}
	bytes = devMem;

	if (GBytes > 0) printf_mpi(rank, "%zd GB\t", GBytes);
	if (MBytes > 0) printf_mpi(rank, "%zd MB\t", MBytes);
	if (KBytes > 0) printf_mpi(rank, "%zd KB\t", KBytes);
	if (bytes  > 0) printf_mpi(rank, "%zd bytes\t", bytes);
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
