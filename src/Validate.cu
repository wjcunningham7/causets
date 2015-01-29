#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Debug:  Future vs Past Edges in Adjacency List
//O(1) Efficiency
void compareAdjacencyLists(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int i, j;
	for (i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Forward Connections:\n");
		if (edges.future_edge_row_start[i] == -1)
			printf("\tNo future connections.\n");
		else {
			for (j = 0; j < nodes.k_out[i] && j < 10; j++)
				printf("%d ", edges.future_edges[edges.future_edge_row_start[i]+j]);
			printf("\n");
		}

		printf("Backward Connections:\n");
		if (edges.past_edge_row_start[i] == -1)
			printf("\tNo past connections.\n");
		else {
			for (j = 0; j < nodes.k_in[i] && j < 10; j++)
				printf("%d ", edges.past_edges[edges.past_edge_row_start[i]+j]);
			printf("\n");
		}
	
		fflush(stdout);
	}
}

//Debug:  Future and Past Adjacency List Indices
//O(1) Effiency
void compareAdjacencyListIndices(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int max1 = 20;
	int max2 = 100;
	int i, j;

	printf("\nFuture Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.past_edge_row_start[i]);
	fflush(stdout);

	int next_future_idx = -1;
	int next_past_idx = -1;

	for (i = 0; i < max1; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %d\n", nodes.k_out[i]);
		if (edges.future_edge_row_start[i] == -1) {
			printf("Pointer: 0\n");
		} else {
			for (j = 1; j < max2; j++) {
				if (edges.future_edge_row_start[i+j] != -1) {
					next_future_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.future_edge_row_start[i+next_future_idx] - edges.future_edge_row_start[i]));
		}

		printf("In-Degrees: %d\n", nodes.k_in[i]);
		if (edges.past_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (j = 1; j < max2; j++) {
				if (edges.past_edge_row_start[i+j] != -1) {
					next_past_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.past_edge_row_start[i+next_past_idx] - edges.past_edge_row_start[i]));
		}
		fflush(stdout);
	}
}

#ifdef CUDA_ENABLED
//Note that core_edge_exists has not been implemented in this version of the linkNodesGPU subroutine.
bool linkNodesGPU_v1(Node &nodes, const Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (nodes.crd->getDim() == 4);
		assert (!nodes.crd->isNull());
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);	
		assert (nodes.crd->z() != NULL);
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);
		assert (N_tar > 0);
		assert (k_tar > 0);
		assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
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

	CUdeviceptr d_w;
	CUdeviceptr d_x;
	CUdeviceptr d_y;
	CUdeviceptr d_z;
	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	CUdeviceptr d_past_edge_row_start, d_future_edge_row_start;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_buf, d_buf_scanned;
	CUdeviceptr d_N_res, d_N_deg2;
	CUdeviceptr d_g_idx;

	int *g_idx;
	int i, j, k;

	stopwatchStart(&sLinkNodesGPU);
	stopwatchStart(&sGPUOverhead);

	//Allocate Overhead on Host
	try {
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
	checkCudaErrors(cuMemAlloc(&d_w, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_x, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_y, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_z, sizeof(float) * N_tar));
	devMemUsed += sizeof(float) * N_tar * 4;

	size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar / 2 + edge_buffer)));
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	devMemUsed += sizeof(uint64_t) * d_edges_size;

	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;
	
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(int)));
	devMemUsed += sizeof(int);

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", hostMemUsed, devMemUsed);

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_w, nodes.crd->w(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_x, nodes.crd->x(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_y, nodes.crd->y(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_z, nodes.crd->z(), sizeof(float) * N_tar));

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_edges, 0, d_edges_size << 1));
	checkCudaErrors(cuMemsetD32(d_k_in, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_k_out, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 1));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_GAL = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / 2));
	unsigned int gridy_GAL = static_cast<unsigned int>(ceil((static_cast<float>(N_tar) / 2) / BLOCK_SIZE));
	dim3 blocks_per_grid_GAL(gridx_GAL, gridy_GAL, 1);
	dim3 threads_per_block_GAL(1, BLOCK_SIZE, 1);
	
	stopwatchStart(&sGenAdjList);

	//Execute Kernel
	GenerateAdjacencyLists_v1<<<blocks_per_grid_GAL, threads_per_block_GAL>>>((float*)d_w, (float*)d_x, (float*)d_y, (float*)d_z, (uint64_t*)d_edges, (int*)d_k_in, (int*)d_k_out, (int*)d_g_idx, N_tar >> 1, compact);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGenAdjList);
	stopwatchStart(&sGPUOverhead);

	//Check Number of Connections
	checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(int)));
	checkCudaErrors(cuCtxSynchronize());

	//Free Device Memory
	cuMemFree(d_w);
	d_w = NULL;

	cuMemFree(d_x);
	d_x = NULL;

	cuMemFree(d_y);
	d_y = NULL;

	cuMemFree(d_z);
	d_z = NULL;

	devMemUsed -= sizeof(float) * N_tar * 4;

	cuMemFree(d_g_idx);
	d_g_idx = NULL;
	devMemUsed -= sizeof(int);

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

	stopwatchStop(&sGPUOverhead);

	//Decode past and future edge lists using Bitonic Sort

	//CUDA Grid Specifications
	unsigned int gridx_bitonic = d_edges_size / BLOCK_SIZE;
	dim3 blocks_per_grid_bitonic(gridx_bitonic, 1, 1);
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);

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
	
	//Print Results
	/*if (!printDegrees(nodes, N_tar, "in-degrees_GPU_v1.cset.dbg.dat", "out-degrees_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, *g_idx, "past-edges_GPU_v1.cset.dbg.dat", "future-edges_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_GPU_v1.cset.dbg.dat", "future-edge-pointers_GPU_v1.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	exit(EXIT_SUCCESS);*/

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

bool generateLists_v1(Node &nodes, uint64_t * const &edges, bool * const core_edge_exists, int * const &g_idx, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose)
{
	if (DEBUG) {
		assert (nodes.crd->getDim() == 4);
		assert (!nodes.crd->isNull());
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (edges != NULL);
		assert (core_edge_exists != NULL);
		assert (g_idx != NULL);
		assert (N_tar > 0);
		assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	}

	//Temporary Buffers
	CUdeviceptr d_w0, d_x0, d_y0, d_z0;
	CUdeviceptr d_w1, d_x1, d_y1, d_z1;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_edges;

	int *h_k_in;
	int *h_k_out;
	bool *h_edges;

	unsigned int i, j;
	unsigned int diag;
	unsigned int core_limit = static_cast<unsigned int>(core_edge_fraction * N_tar);

	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (2 * BLOCK_SIZE * GROUP_SIZE)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

	//DEBUG
	/*if (DEBUG) {
		printf_red();
		printf("\nTHREAD  SIZE: %d\n", THREAD_SIZE);
		printf("BLOCK   SIZE: %d\n", BLOCK_SIZE);
		printf("GROUP   SIZE: %d\n", GROUP_SIZE);
		printf("MBLOCK  SIZE: %zd\n", mblock_size);
		printf("MTHREAD SIZE: %zd\n", mthread_size);
		printf("Number of Times Kernel is Executed: %d\n\n", (GROUP_SIZE*GROUP_SIZE));
		printf_std();
		fflush(stdout);
	}*/

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
	checkCudaErrors(cuMemAlloc(&d_w0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_x0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_z0, sizeof(float) * mthread_size));
	devMemUsed += sizeof(float) * mthread_size * 4;

	checkCudaErrors(cuMemAlloc(&d_w1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_x1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_z1, sizeof(float) * mthread_size));
	devMemUsed += sizeof(float) * mthread_size * 4;

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
		printMemUsed("for Generating Lists on GPU", hostMemUsed, devMemUsed);

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

	//Index 'i' marks the row and 'j' marks the column
	for (i = 0; i < 2 * GROUP_SIZE; i++) {
		for (j = 0; j < 2 * GROUP_SIZE; j++) {
			if (i > j)
				continue;

			diag = (unsigned int)(i == j);

			//Copy node values to device buffers
			checkCudaErrors(cuMemcpyHtoD(d_w0, nodes.crd->w() + i * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_x0, nodes.crd->x() + i * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_y0, nodes.crd->y() + i * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_z0, nodes.crd->z() + i * mthread_size, sizeof(float) * mthread_size));

			checkCudaErrors(cuMemcpyHtoD(d_w1, nodes.crd->w() + j * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_x1, nodes.crd->x() + j * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_y1, nodes.crd->y() + j * mthread_size, sizeof(float) * mthread_size));
			checkCudaErrors(cuMemcpyHtoD(d_z1, nodes.crd->z() + j * mthread_size, sizeof(float) * mthread_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Execute Kernel
			GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, diag, compact);
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
			readDegrees(nodes.k_in, h_k_in, j, mthread_size);
			readDegrees(nodes.k_out, h_k_out, i, mthread_size);
			readEdges(edges, h_edges, core_edge_exists, g_idx, core_limit, d_edges_size, mthread_size, i, j);

			//Clear Device Memory
			checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));
			checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
			checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());			
		}
	}

	cuMemFree(d_w0);
	d_w0 = NULL;

	cuMemFree(d_x0);
	d_x0 = NULL;

	cuMemFree(d_y0);
	d_y0 = NULL;

	cuMemFree(d_z0);
	d_z0 = NULL;

	devMemUsed -= sizeof(float) * mthread_size * 4;

	cuMemFree(d_w1);
	d_w1 = NULL;

	cuMemFree(d_x1);
	d_x1 = NULL;

	cuMemFree(d_y1);
	d_y1 = NULL;

	cuMemFree(d_z1);
	d_z1 = NULL;

	devMemUsed -= sizeof(float) * mthread_size * 4;

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

	return true;
}

//Decode past and future edge lists using Bitonic Sort
bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (h_edges != NULL);
		assert (g_idx != NULL);

		assert (d_edges_size > 0);
	}

	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	int j, k;

	//printMemUsed("at Checkpoint 1", hostMemUsed, devMemUsed);

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	devMemUsed += sizeof(uint64_t) * d_edges_size;

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_edges, h_edges, sizeof(uint64_t) * d_edges_size));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//CUDA Grid Specifications
	unsigned int gridx_bitonic = d_edges_size / BLOCK_SIZE;
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_bitonic(gridx_bitonic, 1, 1);

	//Execute Kernel
	for (k = 2; k <= d_edges_size; k <<= 1) {
		for (j = k >> 1; j > 0; j >>= 1) {
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, j, k);
			getLastCudaError("Kernel 'Subroutines_GPU.BitonicSort' Failed to Execute!\n");
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	//printMemUsed("at Checkpoint 2", hostMemUsed, devMemUsed);

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_future_edges, sizeof(int) * d_edges_size));
	devMemUsed += sizeof(int) * d_edges_size;

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Bitonic Sorting", hostMemUsed, devMemUsed);

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_future_edges, 0, d_edges_size));

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<float>(*g_idx) / BLOCK_SIZE));
	//printf("Grid X Decode: %u\n", gridx_decode);
	dim3 blocks_per_grid_decode(gridx_decode, 1, 1);

	//Execute Kernel
	DecodeFutureEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_future_edges, *g_idx, d_edges_size - *g_idx);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeFutureEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.future_edges, d_future_edges, sizeof(int) * *g_idx));

	//Free Device Memory
	cuMemFree(d_future_edges);
	d_future_edges = NULL;
	devMemUsed -= sizeof(int) * d_edges_size;

	//Resort Edges with New Encoding
	for (k = 2; k <= d_edges_size; k <<= 1) {
		for (j = k >> 1; j > 0; j >>= 1) {
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, j, k);
			getLastCudaError("Kernel 'Subroutines_GPU.BitonicSort' Failed to Execute!\n");
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edges, sizeof(int) * d_edges_size));
	devMemUsed += sizeof(int) * d_edges_size;

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_past_edges, 0, d_edges_size));

	//Execute Kernel
	DecodePastEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_past_edges, *g_idx, d_edges_size - *g_idx);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePastEdges' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.past_edges, d_past_edges, sizeof(int) * *g_idx));
	
	//Free Device Memory
	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * d_edges_size;

	cuMemFree(d_past_edges);
	d_past_edges = NULL;
	devMemUsed -= sizeof(int) * d_edges_size;

	return true;
}
#endif

//Generate confusion matrix for geodesic distances in universe with matter
//Save matrix values as well as d_theta and d_eta to file
bool validateEmbedding(EVData &evd, Node &nodes, const Edge &edges, const int &N_tar, const double &N_emb, const int &N_res, const float &k_res, const int &dim, const Manifold &manifold, const double &a, const double &alpha, long &seed, const int &num_mpi_threads, const int &mpi_rank, Stopwatch &sValidateEmbedding, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &compact, const bool &verbose)
{
	if (DEBUG) {
		assert (nodes.crd->getDim() == 4);
		assert (!nodes.crd->isNull());
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (N_tar > 0);
		assert (dim == 3);
		#ifdef MPI_ENABLED
		assert (num_mpi_threads > 0);
		assert (mpi_rank >= 0);
		#endif
		assert (manifold == DE_SITTER);
		assert (universe);	//Just for now
	}

	uint64_t stride = static_cast<uint64_t>(static_cast<double>(N_tar) * (N_tar - 1) / (N_emb * 2));
	uint64_t npairs = static_cast<uint64_t>(N_emb);
	uint64_t k;

	stopwatchStart(&sValidateEmbedding);

	//printf("Number of paths to test: %" PRIu64 "\n", static_cast<uint64_t>(N_emb));
	//printf("Stride: %" PRIu64 "\n", stride);

	try {
		evd.confusion = (uint64_t*)malloc(sizeof(uint64_t) * 4);
		if (evd.confusion == NULL)
			throw std::bad_alloc();
		memset(evd.confusion, 0, sizeof(uint64_t) * 4);
		hostMemUsed += sizeof(uint64_t) * 4;

		evd.tn = (float*)malloc(sizeof(float) * npairs * 2);
		if (evd.tn == NULL)
			throw std::bad_alloc();
		memset(evd.tn, 0, sizeof(float) * npairs * 2);
		hostMemUsed += sizeof(float) * npairs * 2;

		evd.fp = (float*)malloc(sizeof(float) * npairs * 2);
		if (evd.fp == NULL)
			throw std::bad_alloc();
		memset(evd.fp, 0, sizeof(float) * npairs * 2);
		hostMemUsed += sizeof(float) * npairs * 2;

		memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
		if (verbose)
			printMemUsed("for Embedding Validation", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	//Broadcast:
	// > nodes.crd.w
	// > nodes.crd.x
	// > nodes.crd.y
	// > nodes.crd.z
	// > nodes.id.tau
	#endif

	unsigned int start = 0;
	unsigned int finish = npairs;

	#ifdef MPI_ENABLED
	//Define start based on num_mpi_threads and mpi_rank
	//Define end based on num_mpi_threads and mpi_rank
	#endif

	#ifdef _OPENMP
	//Assuming omp_get_max_threads() returns 32
	#pragma omp parallel num_threads(32)
	{
	#pragma omp for
	#endif
	for (k = start; k < finish; k++) {
		//Choose a pair (i,j) from a single index k
		uint64_t vec_idx = k * stride + 1;
		int i = static_cast<int>(vec_idx / (N_tar - 1));
		int j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		int do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		//Embedded distance
		double distance = distanceEmbFLRW(nodes.crd->getFloat4(i), nodes.id.tau[i], nodes.crd->getFloat4(j), nodes.id.tau[j], dim, manifold, a, alpha, universe, compact);

		if (distance == INF) continue;

		//Check light cone condition for 4D vs 5D
		//Null hypothesis is the nodes are not connected
		double d_eta = ABS(static_cast<double>(node_b.w - node_a.w), STL);
		double d_theta = ACOS(static_cast<double>(DIST_V2 ? sphProduct_v2(node_a, node_b) : sphProduct_v1(node_a, node_b)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

		if (d_theta < d_eta) {	//Actual Timelike
			if (distance > 0) {
				//False Negative (both timelike)
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				evd->confusion[1]++;
			} else {
				//True Negative
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				evd->confusion[2]++;

				#ifdef _OPENMP
				#pragma omp critical (tn)
				{
				#endif
				evd->tn[evd->tn_idx++] = static_cast<float>(d_eta);
				evd->tn[evd->tn_idx++] = static_cast<float>(d_theta);
				#ifdef _OPENMP
				}
				#endif
			}
		} else {	//Actual Spacelike
			if (distance > 0) {
				//False Positive
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				evd->confusion[3]++;

				#ifdef _OPENMP
				#pragma omp critical (fp)
				{
				#endif
				evd->fp[evd->fp_idx++] = static_cast<float>(d_eta);
				evd->fp[evd->fp_idx++] = static_cast<float>(d_theta);
				#ifdef _OPENMP
				}
				#endif
			} else {
				//True Positive (both spacelike)
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				evd->confusion[0]++;
			}
		}		
	}
	#ifdef _OPENMP
	}
	#endif

	#ifdef MPI_ENABLED
	//Reduce:
	// > evd.confusion
	//Gatherv:
	// > evd.fp
	// > evd.tn
	#endif

	//Number of timelike distances in 4-D native FLRW spacetime
	evd.A1T = static_cast<double>(N_res * k_res / 2);
	//Number of spacelike distances in 4-D native FLRW spacetime
	evd.A1S = static_cast<double>(N_tar) * (N_tar - 1) / 2 - A1T;

	stopwatchStop(&sValidateEmbedding);

	printf("\tCalculated Confusion Matrix.\n");
	printf_cyan();
	printf("\t\tTrue  Positives: %f\n", static_cast<double>(evd.confusion[0]) / evd.A1S);
	printf("\t\tFalse Negatives: %f\n", static_cast<double>(evd.confusion[1]) / evd.A1T);
	printf_red();
	printf("\t\tTrue  Negatives: %f\n", static_cast<double>(evd.confusion[2]) / evd.A1T);
	printf("\t\tFalse Positives: %f\n", static_cast<double>(evd.confusion[3]) / evd.A1S);
	printf_std();
	fflush(stdout);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sValidateEmbedding.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(Node &nodes, const int num_vals, const char *filename, const char *coord)
{
	if (DEBUG) {
		//No null pointers
		assert (filename != NULL);
		assert (coord != NULL);

		//Variables in correct range
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream;
		outputStream.open(filename);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			if (strcmp(coord, "tau") == 0)
				outputStream << nodes.id.tau[i] << std::endl;
			else if (strcmp(coord, "eta") == 0)
				outputStream << nodes.crd->w(i) << std::endl;	//Use for dim = 3
				//outputStream << nodes.crd->x(i) << std::endl;	//Use for dim = 1
			else if (strcmp(coord, "theta1") == 0)
				outputStream << nodes.crd->x(i) << std::endl;
			else if (strcmp(coord, "theta2") == 0)
				outputStream << nodes.crd->y(i) << std::endl;
			else if (strcmp(coord, "theta3") == 0)
				outputStream << nodes.crd->z(i) << std::endl;
			else
				throw CausetException("Unrecognized value in 'coord' parameter!\n");
		}
	
		outputStream.flush();
		outputStream.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out)
{
	if (DEBUG) {
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (filename_in != NULL);
		assert (filename_out != NULL);
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream_in;
		outputStream_in.open(filename_in);
		if (!outputStream_in.is_open())
			throw CausetException("Failed to open in-degree file in 'printDegrees' function!\n");

		std::ofstream outputStream_out;
		outputStream_out.open(filename_out);
		if (!outputStream_out.is_open())
			throw CausetException("Failed to open out-degree file in 'printDegrees' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			outputStream_in << nodes.k_in[i] << std::endl;
			outputStream_out << nodes.k_out[i] << std::endl;
		}

		outputStream_in.flush();
		outputStream_in.close();

		outputStream_out.flush();
		outputStream_out.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

bool printEdgeLists(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (filename_past != NULL);
		assert (filename_future != NULL);
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream_past;
		outputStream_past.open(filename_past);
		if (!outputStream_past.is_open())
			throw CausetException("Failed to open past-edge file in 'printEdgeLists' function!\n");

		std::ofstream outputStream_future;
		outputStream_future.open(filename_future);
		if (!outputStream_future.is_open())
			throw CausetException("Failed to open future-edges file in 'printEdgeLists' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			outputStream_past << edges.past_edges[i] << std::endl;
			outputStream_future << edges.future_edges[i] << std::endl;
		}

		outputStream_past.flush();
		outputStream_past.close();

		outputStream_future.flush();
		outputStream_future.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future)
{
	if (DEBUG) {
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (filename_past != NULL);
		assert (filename_future != NULL);
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream_past;
		outputStream_past.open(filename_past);
		if (!outputStream_past.is_open())
			throw CausetException("Failed to open past-edge-pointer file in 'printEdgeLists' function!\n");

		std::ofstream outputStream_future;
		outputStream_future.open(filename_future);
		if (!outputStream_future.is_open())
			throw CausetException("Failed to open future-edge-pointer file in 'printEdgeLists' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			outputStream_past << edges.past_edge_row_start[i] << std::endl;
			outputStream_future << edges.future_edge_row_start[i] << std::endl;
		}

		outputStream_past.flush();
		outputStream_past.close();

		outputStream_future.flush();
		outputStream_future.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}
