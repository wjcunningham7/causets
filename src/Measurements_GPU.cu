#include "Measurements_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

__global__ void MeasureAction(bool *edges0, bool *edges1, unsigned int *N_ij)
{
	__shared__ bool shr_N[BLOCK_SIZE];

	//Define Useful Indices
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
	unsigned int m = blockIdx.z;
	unsigned int n = blockIdx.y;
	unsigned int N = gridDim.x * BLOCK_SIZE;
	unsigned int idx0 = m * N + i;
	unsigned int idx1 = n * N + i;

	//Global Read
	bool x = edges0[idx0];
	bool y = edges1[idx1];

	//Binary Multiplication
	bool z = x & y;

	//Write to Shared Memory
	shr_N[tid] = z;

	//Synchronize Threads among Block
	__syncthreads();

	//Parallel Reduction
	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
		if (!(tid % (stride << 1)))
			shr_N[tid] += shr_N[tid+stride];
		__syncthreads();
	}

	//Global Write (Atomic)
	if (!tid)
		atomicAdd(&N_ij[n * gridDim.z + m], shr_N[0]);
}

//Measure Causal Set Action
//This algorithm has been parallelized on the GPU
//It is assumed the edge list has already been generated
//Here we calculate the smeared action
bool measureActionGPU(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, const int &N_tar, const int &stdim, const Manifold &manifold, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &use_bit, const bool &compact, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!use_bit);
	assert (!nodes.crd->isNull());
	assert (stdim == 2 || stdim == 4);
	assert (manifold == DE_SITTER);

	if (stdim == 2)
		assert (nodes.crd->getDim() == 2);
	else if (stdim == 4) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
	}
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);

	assert (link ^ relink);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (ca != NULL);

	assert (N_tar > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif

	//cusp::csr_matrix<int,float,cusp::host_memory> csr_host(5,8,12);
	//func1();

	//Adjacency List Tiles
	CUdeviceptr d_adj0, d_adj1;
	bool *h_adj0, *h_adj1;

	//Interval Tiles
	CUdeviceptr d_N_ij;
	unsigned int *h_N_ij;

	//Buffers for Overhead
	int *idx_buf0, *idx_buf1;

	long double N2 = static_cast<long double>(N_tar) * N_tar + 1.8E10;
	int l = static_cast<int>(exp2(floor(log2(static_cast<double>((sqrt(N2) - N_tar) / 4.0)))));

	size_t d_adj_size = l * l * N_tar;
	size_t num_groups = N_tar / l + 1;
	int i, j;

	stopwatchStart(&sMeasureAction);

	//Allocate Overhead on Host
	try {
		cardinalities = (int*)malloc(sizeof(int) * (N_tar - 1));
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(int) * (N_tar - 1));
		ca->hostMemUsed += sizeof(int) * (N_tar - 1);

		h_adj0 = (bool*)malloc(sizeof(bool) * d_adj_size);
		if (h_adj0 == NULL)
			throw std::bad_alloc();
		memset(h_adj0, 0, sizeof(bool) * d_adj_size);
		ca->hostMemUsed += sizeof(bool) * d_adj_size;

		h_adj1 = (bool*)malloc(sizeof(bool) * d_adj_size);
		if (h_adj1 == NULL)
			throw std::bad_alloc();
		memset(h_adj1, 0, sizeof(bool) * d_adj_size);
		ca->hostMemUsed += sizeof(bool) * d_adj_size;

		h_N_ij = (unsigned int*)malloc(sizeof(unsigned int) * l * l);
		if (h_N_ij == NULL)
			throw std::bad_alloc();
		memset(h_N_ij, 0, sizeof(unsigned int) * l * l);
		ca->hostMemUsed += sizeof(unsigned int) * l * l;

		idx_buf0 = (int*)malloc(sizeof(int) * l);
		if (idx_buf0 == NULL)
			throw std::bad_alloc();
		memset(idx_buf0, 0, sizeof(int) * l);
		ca->hostMemUsed += sizeof(int) * l;

		idx_buf1 = (int*)malloc(sizeof(int) * l);
		if (idx_buf1 == NULL)
			throw std::bad_alloc();
		memset(idx_buf1, 0, sizeof(int) * l);
		ca->hostMemUsed += sizeof(int) * l;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_adj0, sizeof(bool) * d_adj_size));
	ca->devMemUsed += sizeof(bool) * d_adj_size;

	checkCudaErrors(cuMemAlloc(&d_adj1, sizeof(bool) * d_adj_size));
	ca->devMemUsed += sizeof(bool) * d_adj_size;

	checkCudaErrors(cuMemAlloc(&d_N_ij, sizeof(unsigned int) * l * l));
	ca->devMemUsed += sizeof(unsigned int) * l * l;

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Measuring Smeared Action", ca->hostMemUsed, ca->devMemUsed, 0);

	//CUDA Grid Specifications
	unsigned int gridx = N_tar / BLOCK_SIZE;
	unsigned int gridy = l;
	unsigned int gridz = l;
	dim3 blocks_per_grid(gridx, gridy, gridz);
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);

	//Tiling Algorithm
	for (i = 0; i < num_groups / 2; i++) {
		for (j = 0; j < num_groups; j++) {
			//Generate portion of adjacency matrix
			remakeAdjMatrix(h_adj0, h_adj1, nodes.k_in, nodes.k_out, edges.past_edges, edges.future_edges, edges.past_edge_row_start, edges.future_edge_row_start, idx_buf0, idx_buf1, N_tar, i, j, l);

			//Copy Memory from Host to Device
			checkCudaErrors(cuMemcpyHtoD(d_adj0, h_adj0, sizeof(bool) * d_adj_size));
			checkCudaErrors(cuMemcpyHtoD(d_adj1, h_adj1, sizeof(bool) * d_adj_size));

			//Initialize Memory on Device
			checkCudaErrors(cuMemsetD8(d_N_ij, 0, sizeof(unsigned int) * l * l));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Execute Kernel
			MeasureAction<<<blocks_per_grid, threads_per_block>>>((bool*)d_adj0, (bool*)d_adj1, (unsigned int*)d_N_ij);

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());
			
			//Copy Results to Host
			checkCudaErrors(cuMemcpyDtoH(h_N_ij, d_N_ij, sizeof(unsigned int) * l * l));

			//Interpret Results
			readIntervals(cardinalities, h_N_ij, l);
		}
	}

	//Free Host Memory
	free(h_adj0);
	h_adj0 = NULL;
	ca->hostMemUsed -= sizeof(bool) * d_adj_size;

	free(h_adj1);
	h_adj1 = NULL;
	ca->hostMemUsed -= sizeof(bool) * d_adj_size;

	free(h_N_ij);
	h_N_ij = NULL;
	ca->hostMemUsed -= sizeof(unsigned int) * l * l;

	free(idx_buf0);
	idx_buf0 = NULL;
	ca->hostMemUsed -= sizeof(int) * l;

	free(idx_buf1);
	idx_buf1 = NULL;
	ca->hostMemUsed -= sizeof(int) * l;

	//Free Device Memory
	cuMemFree(d_adj0);
	d_adj0 = 0;
	ca->devMemUsed -= sizeof(bool) * d_adj_size;

	cuMemFree(d_adj1);
	d_adj1 = 0;
	ca->devMemUsed -= sizeof(bool) * d_adj_size;

	cuMemFree(d_N_ij);
	d_N_ij = 0;
	ca->devMemUsed -= sizeof(unsigned int) * l * l;

	//Format Results
	cardinalities[0] = N_tar;

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf("\tCalculated Smeared Action.\n");
		printf_cyan();
		printf("\t\tCausal Set Action: %f\n", action);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}
