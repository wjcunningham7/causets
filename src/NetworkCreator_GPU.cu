#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists(float4 *nodes, uint64_t *edges, int *k_in, int *k_out, int *g_idx, int width)
{
	///////////////////////////////////////
	// Identify Node Pair with Thread ID //
	///////////////////////////////////////

	__shared__ float4 shr_node0_c;
	__shared__ int n_a[BLOCK_SIZE];
	__shared__ int n_b[BLOCK_SIZE];
	__shared__ int n_c[BLOCK_SIZE];
	float4 node0_ab, node0_c, node1_ab, node1_c;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.y;
	unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
	int do_map = i >= j;

	// a -> upper triangle of upper left block (do_map == 0)
	// b -> lower triangle of upper left block (do_map == 1)
	// c -> upper right block

	unsigned int i_ab = i + do_map * (((width - i) << 1) - 1);
	unsigned int j_ab = j + do_map * ((width - j) << 1);

	unsigned int i_c = i;
	unsigned int j_c = j + width;

	if (!tid)
		shr_node0_c = nodes[i_c];
	__syncthreads();

	float dt_ab = 0.0f, dt_c = 0.0f, dx_ab = 0.0f, dx_c = 0.0f;
	if (j < width) {
		node0_c = shr_node0_c;
		node1_c = nodes[j_c];

		node0_ab = do_map ? nodes[i_ab] : node0_c;
		node1_ab = !j ? node0_ab : nodes[j_ab];

		//////////////////////////////////
		// Identify Causal Relationship //
		//////////////////////////////////

		//Calculate dt (assumes nodes are already temporally ordered)
		dt_ab = node1_ab.w - node0_ab.w;
		dt_c  = node1_c.w  - node0_c.w;

		//Calculate dx
		dx_ab = acosf(sphProduct_GPU(node0_ab, node1_ab));
		dx_c = acosf(sphProduct_GPU(node0_c, node1_c));
	}

	//Reduction in Shared Memory
	int edge_ab = dx_ab < dt_ab;
	int edge_c = dx_c < dt_c;
	n_a[tid] = edge_ab * !do_map;
	n_b[tid] = edge_ab * do_map;
	n_c[tid] = edge_c;
	__syncthreads();

	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
		if (!(tid % (stride << 1))) {
			n_a[tid] += n_a[tid+stride];
			n_b[tid] += n_b[tid+stride];
			n_c[tid] += n_c[tid+stride];
		}
		__syncthreads();
	}

	//Write Degrees to Global Memory
	if (edge_ab)
		atomicAdd(&k_in[j_ab], 1);
	if (edge_c)
		atomicAdd(&k_in[j_c], 1);
	if (!tid) {
		if (n_a[0])
			atomicAdd(&k_out[i], n_a[0]);
		if (n_b[0])
			atomicAdd(&k_out[i_ab], n_b[0]);
		if (n_c[0])
			atomicAdd(&k_out[i_c], n_c[0]);
	}

	//Write Edges to Global Memory
	int idx = 0;
	if (edge_ab | edge_c)
		idx = atomicAdd(g_idx, edge_ab + edge_c);
	if (edge_ab)
		edges[idx++] = ((uint64_t)i_ab) << 32 | ((uint64_t)j_ab);
	if (edge_c)
		edges[idx] = ((uint64_t)i_c) << 32 | ((uint64_t)j_c);
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

__global__ void DecodePostScan(int *edge_row_start, int elements)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < elements && !edge_row_start[idx])
		edge_row_start[idx] = -1;
}

__global__ void ResultingProps(int *k_in, int *k_out, int *N_res, int *N_deg2, int elements)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < elements) {
		int k = k_in[idx] + k_out[idx];
		if (k <= 1) {
			atomicAdd(N_deg2, 1);
			if (k)
				atomicAdd(N_res, 1);
		}
	}
}

bool linkNodesGPU(Node &nodes, int * const &past_edges, int * const &future_edges, int * const &past_edge_row_start, int * const &future_edge_row_start, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (past_edges != NULL);
		assert (future_edges != NULL);
		assert (past_edge_row_start != NULL);
		assert (future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		assert (N_tar > 0);
		assert (k_tar > 0);
		assert (a > 0.0);
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

	CUdeviceptr d_nodes;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	CUdeviceptr d_past_edge_row_start, d_future_edge_row_start;
	CUdeviceptr d_buf, d_buf_scanned;
	CUdeviceptr d_N_res, d_N_deg2;
	CUdeviceptr d_g_idx;
	int *g_idx;
	int j,k;

	stopwatchStart(&sLinkNodesGPU);
	stopwatchStart(&sGPUOverhead);

	//Allocate Overhead on Host
	try {
		g_idx = (int*)malloc(sizeof(int));
		if (g_idx == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_nodes, sizeof(float4) * N_tar));
	devMemUsed += sizeof(float4) * N_tar;

	size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar / 2 + edge_buffer)));
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	devMemUsed += sizeof(uint64_t) * d_edges_size;
	
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(int)));
	devMemUsed += sizeof(int);
	
	//Allocate Mapped Pinned Memory
	checkCudaErrors(cuMemHostGetDevicePointer(&d_k_in, (void*)nodes.k_in, 0));
	checkCudaErrors(cuMemHostGetDevicePointer(&d_k_out, (void*)nodes.k_out, 0));

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", hostMemUsed, devMemUsed);

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_nodes, nodes.sc, sizeof(float4) * N_tar));

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_edges, 0, d_edges_size << 1));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 1));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_GAL = static_cast<unsigned int>(ceil((static_cast<float>(N_tar) / 2) / BLOCK_SIZE));
	unsigned int gridy_GAL = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / 2));
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_GAL(gridx_GAL, gridy_GAL, 1);
	
	stopwatchStart(&sGenAdjList);

	//Execute Kernel
	GenerateAdjacencyLists<<<blocks_per_grid_GAL, threads_per_block>>>((float4*)d_nodes, (uint64_t*)d_edges, (int*)d_k_in, (int*)d_k_out, (int*)d_g_idx, N_tar >> 1);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGenAdjList);
	stopwatchStart(&sGPUOverhead);

	//Check Number of Connections
	checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(int)));
	checkCudaErrors(cuCtxSynchronize());

	//Free Device Memory
	cuMemFree(d_nodes);
	d_nodes = NULL;
	devMemUsed -= sizeof(float4) * N_tar;
	
	cuMemFree(d_g_idx);
	d_g_idx = NULL;
	devMemUsed -= sizeof(int);

	stopwatchStop(&sGPUOverhead);

	//Decode past and future edge lists using Bitonic Sort

	//CUDA Grid Specifications
	unsigned int gridx_bitonic = d_edges_size / BLOCK_SIZE;
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

	//Allocate Mapped Pinned Memory
	checkCudaErrors(cuMemHostGetDevicePointer(&d_past_edges, (void*)past_edges, 0));
	checkCudaErrors(cuMemHostGetDevicePointer(&d_future_edges, (void*)future_edges, 0));

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	//unsigned int gridx_decode = gridx_bitonic;
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<float>(*g_idx) / BLOCK_SIZE));
	dim3 blocks_per_grid_decode(gridx_decode, 1, 1);

	stopwatchStart(&sDecode0);

	//Execute Kernel
	DecodeFutureEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_future_edges, *g_idx, d_edges_size - *g_idx + 1);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeFutureEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sDecode0);
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
	stopwatchStart(&sDecode1);

	//Execute Kernel
	DecodePastEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_past_edges, *g_idx, d_edges_size - *g_idx + 1);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePastEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sDecode1);
	stopwatchStart(&sGPUOverhead);
	
	//Free Device Memory
	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * d_edges_size;

	//Parallel Prefix Sum of 'k_in' and 'k_out' and Write to Edge Pointers

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edge_row_start, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_future_edge_row_start, sizeof(int) * N_tar));
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

	//CUDA Grid Specifications
	unsigned int gridx_scan_decode = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / BLOCK_SIZE));
	dim3 blocks_per_grid_scan_decode(gridx_scan_decode, 1, 1);

	stopwatchStart(&sDecode2);

	//Execute Kernel
	DecodePostScan<<<gridx_scan_decode, threads_per_block>>>((int*)d_past_edge_row_start, N_tar);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePostScan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	DecodePostScan<<<gridx_scan_decode, threads_per_block>>>((int*)d_future_edge_row_start, N_tar);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePostScan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sDecode2);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(past_edge_row_start, d_past_edge_row_start, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(future_edge_row_start, d_future_edge_row_start, sizeof(int) * N_tar));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

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
	unsigned int gridx_res_prop = gridx_scan_decode;
	dim3 blocks_per_grid_res_prop(gridx_res_prop, 1, 1);

	stopwatchStart(&sProps);

	//Execute Kernel
	ResultingProps<<<gridx_res_prop, threads_per_block>>>((int*)d_k_in, (int*)d_k_out, (int*)d_N_res, (int*)d_N_deg2, N_tar);
	getLastCudaError("Kernel 'NetworkCreator_GPU.ResultingProps' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sProps);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(&N_res, d_N_res, sizeof(int)));
	checkCudaErrors(cuMemcpyDtoH(&N_deg2, d_N_deg2, sizeof(int)));

	N_res = N_tar - N_res;
	N_deg2 = N_tar - N_deg2;
	k_res = static_cast<float>(*g_idx << 1) / N_res;

	if (DEBUG) {
		assert (N_res > 0);
		assert (N_deg2 > 0);
		assert (k_res > 0.0);
	}

	//Free Device Memory
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
		//printf("\t\tUndirected Links: %d\n", *g_idx);
		printf("\t\tResulting Network Size: %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
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
