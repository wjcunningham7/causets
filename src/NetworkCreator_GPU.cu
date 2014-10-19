#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists_v2(float4 *nodes0, float4 *nodes1, int *k_in, int *k_out, bool *edges, int diag)
{
	__shared__ float4 shr_n1[THREAD_SIZE];
	__shared__ int n[BLOCK_SIZE][THREAD_SIZE];

	float4 n0, n1;

	unsigned int tid = threadIdx.y;
	unsigned int i = blockIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int k;

	if (!tid)
		for (k = 0; k < THREAD_SIZE; k++)
			shr_n1[k] = nodes1[i*THREAD_SIZE+k];
	__syncthreads();

	float dt[THREAD_SIZE];
	float dx[THREAD_SIZE];

	for (k = 0; k < THREAD_SIZE; k++) {
		if (!diag || j < i * THREAD_SIZE + k) {
			n0 = nodes0[j];
			n1 = shr_n1[k];

			dt[k] = n1.w - n0.w;
			dx[k] = acosf(sphProduct_GPU(n0, n1));
		}
	}

	bool edge[THREAD_SIZE];
	int in = 0;
	for (k = 0; k < THREAD_SIZE; k++) {
		edge[k] = (!diag || j < i * THREAD_SIZE + k) && dx[k] < dt[k];
		n[tid][k] = (int)edge[k];
		in += (int)edge[k];
	}
	__syncthreads();

	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
		if (!(tid % (stride << 1)))
			for (k = 0; k < THREAD_SIZE; k++)
				n[tid][k] += n[tid+stride][k];
		__syncthreads();
	}

	//Write values to global memory
	for (k = 0; k < THREAD_SIZE; k++)
		if (!diag || j < i * THREAD_SIZE + k)
			edges[(i*THREAD_SIZE+k)*blockDim.y*gridDim.y+j] = edge[k];
	atomicAdd(&k_in[j], in);

	if (!tid)
		for (k = 0; k < THREAD_SIZE; k++)
			if (!diag || j < i * THREAD_SIZE + k)
				atomicAdd(&k_out[i*THREAD_SIZE+k], n[0][k]);
}

__global__ void DecodeFutureEdges(uint64_t *edges, int *future_edges, int elements, int offset)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < elements) {
		//Decode Future Edges
		uint64_t key = edges[idx + offset];
		unsigned int i = key >> 32;
		unsigned int j = key & 0x00000000FFFFFFFF;

		//Write Future Edges
		future_edges[idx] = j;

		//Encode Past Edges
		edges[idx+offset] = ((uint64_t)j) << 32 | ((uint64_t)i);
	}
}

__global__ void DecodePastEdges(uint64_t *edges, int *past_edges, int elements, int offset)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < elements) {
		//Decode Past Edges
		uint64_t key = edges[idx + offset];

		//Write Past Edges
		past_edges[idx] = key & 0x00000000FFFFFFFF;
	}
}

__global__ void ResultingProps(int *k_in, int *k_out, int *N_res, int *N_deg2, int elements)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < elements) {
		int k = k_in[idx] + k_out[idx];
		if (k <= 1) {
			atomicAdd(N_deg2, 1);
			if (!k)
				atomicAdd(N_res, 1);
		}
	}
}

bool linkNodesGPU(const Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		assert (N_tar > 0);
		assert (k_tar > 0);
		assert (a > 0.0);
		if (universe)
			assert (alpha > 0.0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (edge_buffer >= 0);
	}

	Stopwatch sGPUOverhead = Stopwatch();
	Stopwatch sGenAdjList = Stopwatch();
	Stopwatch sBitonic0 = Stopwatch();
	Stopwatch sDecode0 = Stopwatch();
	Stopwatch sBitonic1 = Stopwatch();
	Stopwatch sDecode1 = Stopwatch();
	Stopwatch sScan0 = Stopwatch();
	Stopwatch sScan1 = Stopwatch();
	Stopwatch sDecode2 = Stopwatch();
	Stopwatch sProps = Stopwatch();

	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	CUdeviceptr d_past_edge_row_start, d_future_edge_row_start;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_buf, d_buf_scanned;
	CUdeviceptr d_N_res, d_N_deg2;

	uint64_t *h_edges;
	int *g_idx;
	int i, j, k;

	stopwatchStart(&sLinkNodesGPU);
	stopwatchStart(&sGPUOverhead);
	
	size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar / 2 + edge_buffer)));

	//Allocate Overhead on Host
	try {
		h_edges = (uint64_t*)malloc(sizeof(uint64_t) * d_edges_size);
		if (h_edges == NULL)
			throw std::bad_alloc();
		memset(h_edges, 0, sizeof(uint64_t) * d_edges_size);
		hostMemUsed += sizeof(uint64_t) * d_edges_size;

		g_idx = (int*)malloc(sizeof(int));
		if (g_idx == NULL)
			throw std::bad_alloc();
		memset(g_idx, 0, sizeof(int));
		hostMemUsed += sizeof(int);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	checkCudaErrors(cuMemsetD32(d_edges, 0, d_edges_size << 1));
	devMemUsed += sizeof(uint64_t) * d_edges_size;
	
	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGPUOverhead);
	stopwatchStart(&sGenAdjList);

	//if (!generateLists(nodes, h_edges, g_idx, N_tar, d_edges_size, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
	if (!generateLists_v2(nodes, h_edges, g_idx, N_tar, d_edges_size, ctx, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
	//	return false;

	stopwatchStop(&sGenAdjList);
	stopwatchStart(&sGPUOverhead);

	try {
		if (*g_idx + 1 >= N_tar * k_tar / 2 + edge_buffer)
			throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	checkCudaErrors(cuMemcpyHtoD(d_edges, h_edges, sizeof(uint64_t) * d_edges_size));
	checkCudaErrors(cuCtxSynchronize());

	free(h_edges);
	h_edges = NULL;
	hostMemUsed -= sizeof(uint64_t) * d_edges_size;

	stopwatchStop(&sGPUOverhead);

	//Decode past and future edge lists using Bitonic Sort

	//CUDA Grid Specifications
	unsigned int gridx_bitonic = d_edges_size / BLOCK_SIZE;
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_bitonic(gridx_bitonic, 1, 1);

	stopwatchStart(&sBitonic0);

	//Execute Kernel
	for (k = 2; k <= d_edges_size; k <<= 1) {
		for (j = k >> 1; j > 0; j >>= 1) {
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, j, k);
			getLastCudaError("Kernel 'Subroutines_GPU.BitonicSort' Failed to Execute!\n");
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	stopwatchStop(&sBitonic0);
	stopwatchStart(&sGPUOverhead);

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_future_edges, sizeof(int) * d_edges_size));
	devMemUsed += sizeof(int) * d_edges_size;

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_future_edges, 0, d_edges_size));

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<float>(*g_idx) / BLOCK_SIZE));
	dim3 blocks_per_grid_decode(gridx_decode, 1, 1);

	stopwatchStart(&sDecode0);

	//Execute Kernel
	DecodeFutureEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_future_edges, *g_idx, d_edges_size - *g_idx);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeFutureEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sDecode0);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.future_edges, d_future_edges, sizeof(int) * *g_idx));

	//Free Device Memory
	cuMemFree(d_future_edges);
	d_future_edges = NULL;
	devMemUsed -= sizeof(int) * d_edges_size;

	stopwatchStop(&sGPUOverhead);
	stopwatchStart(&sBitonic1);

	//Resort Edges with New Encoding
	for (k = 2; k <= d_edges_size; k <<= 1) {
		for (j = k >> 1; j > 0; j >>= 1) {
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, j, k);
			getLastCudaError("Kernel 'Subroutines_GPU.BitonicSort' Failed to Execute!\n");
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	stopwatchStop(&sBitonic1);
	stopwatchStart(&sGPUOverhead);

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edges, sizeof(int) * d_edges_size));
	devMemUsed += sizeof(int) * d_edges_size;

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_past_edges, 0, d_edges_size));

	stopwatchStop(&sGPUOverhead);
	stopwatchStart(&sDecode1);

	//Execute Kernel
	DecodePastEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_past_edges, *g_idx, d_edges_size - *g_idx);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePastEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sDecode1);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.past_edges, d_past_edges, sizeof(int) * *g_idx));
	
	//Free Device Memory
	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * d_edges_size;

	cuMemFree(d_past_edges);
	d_past_edges = NULL;
	devMemUsed -= sizeof(int) * d_edges_size;

	//Parallel Prefix Sum of 'k_in' and 'k_out' and Write to Edge Pointers

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edge_row_start, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_future_edge_row_start, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_buf, sizeof(int) * (BLOCK_SIZE << 1)));
	devMemUsed += sizeof(int) * (BLOCK_SIZE << 1);

	checkCudaErrors(cuMemAlloc(&d_buf_scanned, sizeof(int) * (BLOCK_SIZE << 1)));
	devMemUsed += sizeof(int) * (BLOCK_SIZE << 1);
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Prefix Sum", hostMemUsed, devMemUsed);

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_past_edge_row_start, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_future_edge_row_start, 0, N_tar));

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_k_in, nodes.k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_k_out, nodes.k_out, sizeof(int) * N_tar));

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_scan = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE << 1)));
	dim3 blocks_per_grid_scan(gridx_scan, 1, 1);

	stopwatchStart(&sScan0);

	//Execute Kernels
	Scan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_k_in, (int*)d_past_edge_row_start, (int*)d_buf, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	Scan<<<dim3(1,1,1), threads_per_block>>>((int*)d_buf, (int*)d_buf_scanned, NULL, BLOCK_SIZE << 1);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	PostScan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_past_edge_row_start, (int*)d_buf_scanned, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.PostScan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sScan0);
	stopwatchStart(&sScan1);

	Scan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_k_out, (int*)d_future_edge_row_start, (int*)d_buf, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	Scan<<<dim3(1,1,1), threads_per_block>>>((int*)d_buf, (int*)d_buf_scanned, NULL, BLOCK_SIZE << 1);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	PostScan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_future_edge_row_start, (int*)d_buf_scanned, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.PostScan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sScan1);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.past_edge_row_start, d_past_edge_row_start, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(edges.future_edge_row_start, d_future_edge_row_start, sizeof(int) * N_tar));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGPUOverhead);

	//Formatting
	for (i = N_tar - 1; i > 0; i--) {
		edges.past_edge_row_start[i] = edges.past_edge_row_start[i-1];
		edges.future_edge_row_start[i] = edges.future_edge_row_start[i-1];
	}

	edges.past_edge_row_start[0] = -1;
	edges.future_edge_row_start[0] = 0;

	int pv = edges.past_edge_row_start[N_tar-1];
	int fv = edges.future_edge_row_start[N_tar-1];

	for (i = N_tar-2; i >= 0; i--) {
		if (pv == edges.past_edge_row_start[i])
			edges.past_edge_row_start[i] = -1;
		else
			pv = edges.past_edge_row_start[i];

		if (fv == edges.future_edge_row_start[i])
			edges.future_edge_row_start[i] = -1;
		else
			fv = edges.future_edge_row_start[i];
	}

	edges.future_edge_row_start[N_tar-1] = -1;
	
	stopwatchStart(&sGPUOverhead);

	//Free Device Memory
	cuMemFree(d_past_edge_row_start);
	d_past_edge_row_start = NULL;
	devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_future_edge_row_start);
	d_future_edge_row_start = NULL;
	devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_buf);
	d_buf = NULL;
	devMemUsed -= sizeof(int) * (BLOCK_SIZE << 1);

	cuMemFree(d_buf_scanned);
	d_buf_scanned = NULL;
	devMemUsed -= sizeof(int) * (BLOCK_SIZE << 1);

	//Resulting Network Properties

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_N_res, sizeof(int)));
	devMemUsed += sizeof(int);

	checkCudaErrors(cuMemAlloc(&d_N_deg2, sizeof(int)));
	devMemUsed += sizeof(int);
	
	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_N_res, 0, 1));
	checkCudaErrors(cuMemsetD32(d_N_deg2, 0, 1));

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_res_prop = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / BLOCK_SIZE));
	dim3 blocks_per_grid_res_prop(gridx_res_prop, 1, 1);

	stopwatchStart(&sProps);

	//Execute Kernel
	ResultingProps<<<gridx_res_prop, threads_per_block>>>((int*)d_k_in, (int*)d_k_out, (int*)d_N_res, (int*)d_N_deg2, N_tar);
	getLastCudaError("Kernel 'NetworkCreator_GPU.ResultingProps' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sProps);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(nodes.k_in, d_k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(nodes.k_out, d_k_out, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(&N_res, d_N_res, sizeof(int)));
	checkCudaErrors(cuMemcpyDtoH(&N_deg2, d_N_deg2, sizeof(int)));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	N_res = N_tar - N_res;
	N_deg2 = N_tar - N_deg2;
	k_res = static_cast<float>(*g_idx << 1) / N_res;

	if (DEBUG) {
		assert (N_res > 0);
		assert (N_deg2 > 0);
		assert (k_res > 0.0);
	}

	//Free Device Memory
	cuMemFree(d_k_in);
	d_k_in = NULL;
	devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_k_out);
	d_k_out = NULL;
	devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_N_res);
	d_N_res = NULL;
	devMemUsed -= sizeof(int);

	cuMemFree(d_N_deg2);
	d_N_deg2 = NULL;
	devMemUsed -= sizeof(int);

	stopwatchStop(&sGPUOverhead);
	stopwatchStop(&sLinkNodesGPU);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tUndirected Links:         %d\n", *g_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", (k_res * N_res) / N_tar);
		printf_std();
		fflush(stdout);
	}
	
	//Free Host Memory
	free(g_idx);
	g_idx = NULL;
	hostMemUsed -= sizeof(int);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodesGPU.elapsedTime);
		printf("\t\t\tGPU Overhead Time: %5.6f sec\n", sGPUOverhead.elapsedTime);
		printf("\t\t\tAdjacency List Kernel Time: %5.6f sec\n", sGenAdjList.elapsedTime);
		printf("\t\t\tBitonic Sort 0 Kernel Time: %5.6f sec\n", sBitonic0.elapsedTime);
		printf("\t\t\tFuture Edge Decode Time: %5.6f sec\n", sDecode0.elapsedTime);
		printf("\t\t\tBitonic Sort 1 Kernel Time: %5.6f sec\n", sBitonic1.elapsedTime);
		printf("\t\t\tPast Edge Decode Time: %5.6f sec\n", sDecode1.elapsedTime);
		printf("\t\t\tScan 0 Kernel Time: %5.6f sec\n", sScan0.elapsedTime);
		printf("\t\t\tScan 1 Kernel Time: %5.6f sec\n", sScan1.elapsedTime);
		printf("\t\t\tPost Scan Decode Time: %5.6f sec\n", sDecode2.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Uses multiple buffers and asynchronous operations
bool generateLists_v2(const Node &nodes, uint64_t * const &edges, int * const &g_idx, const int &N_tar, const size_t &d_edges_size, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		//No Null Pointers
		assert (nodes.c.sc != NULL);
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (edges != NULL);
		assert (g_idx != NULL);

		//Values in correct ranges
		assert (N_tar > 0);
	}

	//CUDA Streams
	CUstream stream[NBUFFERS];

	//Arrays of Buffers
	int *h_k_in[NBUFFERS];
	int *h_k_out[NBUFFERS];
	bool *h_edges[NBUFFERS];

	CUdeviceptr d_nodes0[NBUFFERS];
	CUdeviceptr d_nodes1[NBUFFERS];
	CUdeviceptr d_k_in[NBUFFERS];
	CUdeviceptr d_k_out[NBUFFERS];
	CUdeviceptr d_edges[NBUFFERS];

	unsigned int i, j, m;
	unsigned int diag;
	
	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (2 * BLOCK_SIZE * GROUP_SIZE)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

	//Create Streams
	for (i = 0; i < NBUFFERS; i++)
		checkCudaErrors(cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING));

	//Allocate Memory
	for (i = 0; i < NBUFFERS; i++) {
		checkCudaErrors(cuMemHostAlloc((void**)&h_k_in[i], sizeof(int) * mthread_size, CU_MEMHOSTALLOC_PORTABLE));
		hostMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemHostAlloc((void**)&h_k_out[i], sizeof(int) * mthread_size, CU_MEMHOSTALLOC_PORTABLE));
		hostMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemHostAlloc((void**)&h_edges[i], sizeof(bool) * m_edges_size, CU_MEMHOSTALLOC_PORTABLE));
		hostMemUsed += sizeof(bool) * m_edges_size;

		checkCudaErrors(cuMemAlloc(&d_nodes0[i], sizeof(float4) * mthread_size));
		devMemUsed += sizeof(float4) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_nodes1[i], sizeof(float4) * mthread_size));
		devMemUsed += sizeof(float4) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_k_in[i], sizeof(int) * mthread_size));
		devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_k_out[i], sizeof(int) * mthread_size));
		devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_edges[i], sizeof(bool) * m_edges_size));
		devMemUsed += sizeof(bool) * m_edges_size;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Linking Nodes on GPU", hostMemUsed, devMemUsed);

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil(static_cast<float>(mthread_size) / THREAD_SIZE));
	unsigned int gridy = mblock_size;
	dim3 threads_per_block(1, BLOCK_SIZE, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);
	
	//Stopwatch gtest = Stopwatch();
	//stopwatchStart(&gtest);

	//Adjacency matrix block C and non-diagonal blocks of AB
	for (i = 0; i < 2 * GROUP_SIZE; i++) {
		for (j = 0; j < 2 * GROUP_SIZE / NBUFFERS; j++) {
			#ifdef _OPENMP
			#pragma omp parallel num_threads(NBUFFERS)
			{
				checkCudaErrors(cuCtxSetCurrent(ctx));

				#pragma omp for schedule(dynamic, 1)
			#endif
				for (m = 0; m < NBUFFERS; m++) {
					if (i > j * NBUFFERS + m)
						continue;

					diag = (unsigned int)(i == j * NBUFFERS + m);

					//Clear Device Buffers
					checkCudaErrors(cuMemsetD32Async(d_k_in[m], 0, mthread_size, stream[m]));
					checkCudaErrors(cuMemsetD32Async(d_k_out[m], 0, mthread_size, stream[m]));
					checkCudaErrors(cuMemsetD8Async(d_edges[m], 0, m_edges_size, stream[m]));					
			
					//Transfer Nodes to Device Buffers
					checkCudaErrors(cuMemcpyHtoDAsync(d_nodes0[m], nodes.c.sc + i * mthread_size, sizeof(float4) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_nodes1[m], nodes.c.sc + (j*NBUFFERS+m) * mthread_size, sizeof(float4) * mthread_size, stream[m]));

					//Execute Kernel
					GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block, 0, stream[m]>>>((float4*)d_nodes0[m], (float4*)d_nodes1[m], (int*)d_k_in[m], (int*)d_k_out[m], (bool*)d_edges[m], diag);
					getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists_v2' Failed to Execute!\n");

					//Copy Memory to Host Buffers
					checkCudaErrors(cuMemcpyDtoHAsync(h_k_in[m], d_k_in[m], sizeof(int) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyDtoHAsync(h_k_out[m], d_k_out[m], sizeof(int) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyDtoHAsync(h_edges[m], d_edges[m], sizeof(bool) * m_edges_size, stream[m]));
				}
			#ifdef _OPENMP
			}
			#endif

			for (m = 0; m < NBUFFERS; m++) {
				if (!((j * NBUFFERS + m) % (2 * GROUP_SIZE + 1)))
					continue;

				//Synchronize
				checkCudaErrors(cuStreamSynchronize(stream[m]));

				//Read Data from Buffers
				readDegrees(nodes.k_in, h_k_in[m], i, mthread_size);
				readDegrees(nodes.k_out, h_k_out[m], j*NBUFFERS+m, mthread_size);
				readEdges(edges, h_edges[m], g_idx, d_edges_size, mthread_size, i, j*NBUFFERS+m);
			}				
		}
	}
	
	//stopwatchStop(&gtest);
	//printf_cyan();
	//printf("Time Elapsed: %.6f\n", gtest.elapsedTime);
	//printf_std();
	//fflush(stdout);

	//Free Buffers
	for (i = 0; i < NBUFFERS; i++) {
		cuMemFreeHost(h_k_in[i]);
		h_k_in[i] = NULL;
		hostMemUsed -= sizeof(int) * mthread_size;

		cuMemFreeHost(h_k_out[i]);
		h_k_out[i] = NULL;
		hostMemUsed -= sizeof(int) * mthread_size;

		cuMemFreeHost(h_edges[i]);
		h_edges[i] = NULL;
		hostMemUsed -= sizeof(bool) * m_edges_size;

		cuMemFree(d_nodes0[i]);
		d_nodes0[i] = NULL;
		devMemUsed -= sizeof(float4) * mthread_size;

		cuMemFree(d_nodes1[i]);
		d_nodes1[i] = NULL;
		devMemUsed -= sizeof(float4) * mthread_size;

		cuMemFree(d_k_in[i]);
		d_k_in[i] = NULL;
		devMemUsed -= sizeof(int) * mthread_size;

		cuMemFree(d_k_out[i]);
		d_k_out[i] = NULL;
		devMemUsed -= sizeof(int) * mthread_size;

		cuMemFree(d_edges[i]);
		d_edges[i] = NULL;
		devMemUsed -= sizeof(bool) * m_edges_size;
	}

	//Destroy Streams
	for (i = 0; i < NBUFFERS; i++)
		checkCudaErrors(cuStreamDestroy(stream[i]));

	//Final Synchronization
	checkCudaErrors(cuCtxSynchronize());

	return true;
}

bool generateLists(const Node &nodes, uint64_t * const &edges, int * const &g_idx, const int &N_tar, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		//No Null Pointers
		assert (nodes.c.sc != NULL);
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (edges != NULL);
		assert (g_idx != NULL);

		//Values in correct ranges
		assert (N_tar > 0);
	}

	//Stopwatch gtest = Stopwatch();
	//stopwatchStart(&gtest);

	//Temporary Buffers
	CUdeviceptr d_nodes0, d_nodes1;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_edges;

	int *h_k_in;
	int *h_k_out;
	bool *h_edges;

	unsigned int i, j;
	unsigned int diag;

	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (2 * BLOCK_SIZE * GROUP_SIZE)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

	//DEBUG
	//printf_red();
	//printf("THREAD  SIZE: %d\n", THREAD_SIZE);
	//printf("BLOCK   SIZE: %d\n", BLOCK_SIZE);
	//printf("GROUP   SIZE: %d\n", GROUP_SIZE);
	//printf("MBLOCK  SIZE: %zd\n", mblock_size);
	//printf("MTHREAD SIZE: %zd\n", mthread_size);
	//printf("\nNumber of Times Kernel is Executed: %d\n", (GROUP_SIZE*GROUP_SIZE));
	//printf_std();
	//fflush(stdout);

	//Allocate Buffers on Host
	try {
		h_k_in = (int*)malloc(sizeof(int) * mthread_size);
		if (h_k_in == NULL)
			throw std::bad_alloc();
		memset(h_k_in, 0, sizeof(int) * mthread_size);
		hostMemUsed += sizeof(int) * mthread_size;

		h_k_out = (int*)malloc(sizeof(int) * mthread_size);
		if (h_k_out == NULL)
			throw std::bad_alloc();
		memset(h_k_out, 0, sizeof(int) * mthread_size);
		hostMemUsed += sizeof(int) * mthread_size;

		h_edges = (bool*)malloc(sizeof(bool) * m_edges_size);
		if (h_edges == NULL)
			throw std::bad_alloc();
		memset(h_edges, 0, sizeof(bool) * m_edges_size);
		hostMemUsed += sizeof(bool) * m_edges_size;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	//Allocate Node Buffers on Device
	checkCudaErrors(cuMemAlloc(&d_nodes0, sizeof(float4) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_nodes0, 0, 4 * mthread_size));
	devMemUsed += sizeof(float4) * mthread_size;

	checkCudaErrors(cuMemAlloc(&d_nodes1, sizeof(float4) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_nodes1, 0, 4 * mthread_size));
	devMemUsed += sizeof(float4) * mthread_size;

	//Allocate Degree Buffers on Device
	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
	devMemUsed += sizeof(int) * mthread_size;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));
	devMemUsed += sizeof(int) * mthread_size;

	//Allocate Edge Buffer on Device
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(bool) * m_edges_size));
	checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));
	devMemUsed += sizeof(bool) * m_edges_size;

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Linking Nodes on GPU", hostMemUsed, devMemUsed);

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil(static_cast<float>(mthread_size) / THREAD_SIZE));
	unsigned int gridy = mblock_size;
	dim3 threads_per_block(1, BLOCK_SIZE, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);

	//DEBUG
	//printf_red();
	//printf("Grid X: %u\n", gridx);
	//printf("Grid Y: %u\n", gridy);
	//printf_std();
	//fflush(stdout);

	//Block C and non-diagonal groups of block AB
	for (i = 0; i < 2 * GROUP_SIZE; i++) {
		for (j = 0; j < 2 * GROUP_SIZE; j++) {
			if (i > j)
				continue;

			diag = (unsigned int)(i == j);

			//Copy node values to device buffers
			checkCudaErrors(cuMemcpyHtoD(d_nodes0, nodes.c.sc + i * mthread_size, sizeof(float4) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_nodes1, nodes.c.sc + j * mthread_size, sizeof(float4) * mthread_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Execute Kernel
			GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block>>>((float4*)d_nodes0, (float4*)d_nodes1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, diag);
			getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists_v2' Failed to Execute!\n");

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Copy edges to host
			checkCudaErrors(cuMemcpyDtoH(h_edges, d_edges, sizeof(bool) * m_edges_size));

			//Copy degrees to host
			checkCudaErrors(cuMemcpyDtoH(h_k_in, d_k_in, sizeof(int) * mthread_size));
			checkCudaErrors(cuMemcpyDtoH(h_k_out, d_k_out, sizeof(int) * mthread_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Transfer data from buffers
			readEdges(edges, h_edges, g_idx, d_edges_size, mthread_size, i, j);
			readDegrees(nodes.k_in, h_k_in, i, mthread_size);
			readDegrees(nodes.k_out, h_k_out, j, mthread_size);

			//Clear Device Memory
			checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));
			checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
			checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());			
		}
	}

	cuMemFree(d_nodes0);
	d_nodes0 = NULL;
	devMemUsed -= sizeof(float4) * mthread_size;

	cuMemFree(d_nodes1);
	d_nodes1 = NULL;
	devMemUsed -= sizeof(float4) * mthread_size;

	cuMemFree(d_k_in);
	d_k_in = NULL;
	devMemUsed -= sizeof(int) * mthread_size;

	cuMemFree(d_k_out);
	d_k_out = NULL;
	devMemUsed -= sizeof(int) * mthread_size;

	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(bool) * m_edges_size;

	free(h_k_in);
	h_k_in = NULL;
	hostMemUsed -= sizeof(int) * mthread_size;

	free(h_k_out);
	h_k_out = NULL;
	hostMemUsed -= sizeof(int) * mthread_size;

	free(h_edges);
	h_edges = NULL;
	hostMemUsed -= sizeof(bool) * m_edges_size;

	//stopwatchStop(&gtest);
	//printf_cyan();
	//printf("Time Elapsed: %.6f\n", gtest.elapsedTime);
	//printf_std();
	//fflush(stdout);

	return true;
}
