#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

/*__global__ void Generate(Node *nodes, int N_tar, long seed)
{
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int j = blockDim.y * blockIdx.y + threadIdx.y;
	//if ((j * width) + i > N_tar)
	//	return;

	//Implement CURAND package here for random number generation
}*/

__global__ void GenerateAdjacencyLists(float4 *nodes, int *past_edges, int *future_edges, int *g_idx, int width, int map)
{
	///////////////////////////////////////
	// Identify Node Pair with Thread ID //
	///////////////////////////////////////

	int tid = threadIdx.x;
	int i = blockIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ float4 node0;
	if (!tid)
		node0 = nodes[i];
	__syncthreads();

	if (j >= width)
		return;

	float4 node1 = nodes[j+width];

	//Global Thread ID (unique among all threads)
	/*uint64_t gid = (static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i = gid / width;
	int j = gid % width;

	if (!(map | j) | i >= width)
		return;

	int do_map = (i >= j) * !map;
	i += do_map * (2 * (width - i) - 1);
	j += map * width + do_map * (2 * (width - j));

	//Read Coordinates from Global Memory
	float4 node0 = nodes[i];
	float4 node1 = nodes[j];*/
	
	//////////////////////////////////
	// Identify Causal Relationship //
	//////////////////////////////////

	//Calculate dt (assumes nodes are already temporally ordered)
	float dt = node1.w - node0.w;

	//Calculate dx
	float dx = acosf(X1_GPU(node0.y) * X1_GPU(node1.y) +
			 X2_GPU(node0.y, node0.z) * X2_GPU(node1.y, node1.z) +
			 X3_GPU(node0.y, node0.z, node0.x) * X3_GPU(node1.y, node1.z, node1.x) +
			 X4_GPU(node0.y, node0.z, node0.x) * X4_GPU(node1.y, node1.z, node1.x));

	if (dx >= dt)
		return;

	//Write to Global Memory
	int idx = atomicAdd(g_idx, 1);
	past_edges[idx] = i;
	future_edges[idx] = j;
}

__global__ void DecodeAdjacencyLists(int *past_edges, int *future_edges, int *past_edge_row_start, int *future_edge_row_start, int n_links)
{
	//
}

__global__ void FindNodeDegrees(int *past_edge_row_start, int *future_edge_row_start, int *k_in, int *k_out)
{
	//
}

/*bool generateNodesGPU(Node * const &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodesGPU, const bool &universe, const bool &verbose, const bool &bench)
{
	//CURAND
	curandGenerator_t prng;
	
	try {
		if (CURAND_STATUS_SUCCESS != curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT))
			throw CausetException("Failed to create curand generator.\n");
		if (CURAND_STATUS_SUCCESS != curandSetPseudoRandomGeneratorSeed(prng, (int)network->network_properties.seed))
			throw CausetException("Failed to set curand seed.\n");

		//Need to redesign Node for GPU so memory for points is contiguous
		//Lots of thought should go into this...
		//if (CURAND_STATUS_SUCCESS != curandGenerateUniform(prng, (float*)d_points, network->network_properties.N_tar))
		//	throw CausetException("Failed to generate curand uniform number distribution.\n");

		if (CURAND_STATUS_SUCCESS != curandDestroyGenerator(prng))
			throw CausetException("Failed to destroy curand generator.\n");
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	}

	//Invoke Kernel
	Generate<<<network->network_properties.network_exec.blocks_per_grid, network->network_properties.network_exec.threads_per_block>>>((Node*)network->d_nodes, network->network_properties.N_tar, network->network_properties.seed);
	getLastCudaError("Kernel 'Generate' Failed to Execute!");
	checkCudaErrors(cuCtxSynchronize());

	//Copy Values to Host
	checkCudaErrors(cuMemcpyDtoH(network->nodes, network->d_nodes, sizeof(Node) * network->network_properties.N_tar));

	return true;
}*/

bool linkNodesGPU(Node * const &nodes, CUdeviceptr d_nodes, int * const &past_edges, CUdeviceptr d_past_edges, int * const &future_edges, CUdeviceptr d_future_edges, int * const &past_edge_row_start, CUdeviceptr d_past_edge_row_start, int * const &future_edge_row_start, CUdeviceptr d_future_edge_row_start, bool * const &core_edge_exists, CUdeviceptr d_k_in, CUdeviceptr d_k_out, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench)
{
	//Add assert statements

	float4 *coord;
	int *k_in;
	int *k_out;
	int *g_idx;
	int i;

	stopwatchStart(&sLinkNodesGPU);

	//Allocate Overhead on Host
	try {
		coord = (float4*)malloc(sizeof(float4) * N_tar);
		if (coord == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(float4) * N_tar;

		k_in = (int*)malloc(sizeof(int) * N_tar);
		if (k_in == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		k_out = (int*)malloc(sizeof(int) * N_tar);
		if (k_out == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		g_idx = (int*)malloc(sizeof(int));
		if (g_idx == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Copy Node Coordinates to Contiguous Memory on Host
	for (i = 0; i < N_tar; i++) {
		coord[i].w = nodes[i].eta;
		coord[i].x = nodes[i].theta;
		coord[i].y = nodes[i].phi;
		coord[i].z = nodes[i].chi;
	}

	//Allocate Global Index on Device
	CUdeviceptr d_g_idx;
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(int)));
	devMemUsed += sizeof(int);

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", hostMemUsed, devMemUsed);

	//Initialize Memory on Device
	int max = N_tar * k_tar / 2 + edge_buffer;
	checkCudaErrors(cuMemsetD32(d_past_edges, 0, max));
	checkCudaErrors(cuMemsetD32(d_future_edges, 0, max));
	checkCudaErrors(cuMemsetD32(d_past_edge_row_start, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_future_edge_row_start, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 1));

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_nodes, coord, sizeof(float4) * N_tar));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	free(coord);
	coord = NULL;
	hostMemUsed -= sizeof(float4) * N_tar;

	//CUDA Grid Specifications
	//unsigned int grid_size = static_cast<unsigned int>(ceil(N_tar / (2 * sqrt(static_cast<float>(BLOCK_SIZE)))));
	unsigned int gridx = static_cast<unsigned int>(ceil((static_cast<float>(N_tar) / 2) / BLOCK_SIZE));
	unsigned int gridy = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / 2));
	//printf("Grid Size: %u\n", grid_size);
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	//dim3 blocks_per_grid(grid_size, grid_size, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);

	//Execute Kernel for Upper Left / Lower Right Adjacency Pairs
	/*GenerateAdjacencyLists<<<blocks_per_grid, threads_per_block>>>((float4*)d_nodes, (int*)d_past_edges, (int*)d_future_edges, (int*)d_g_idx, N_tar / 2, 0);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());*/

	//Execute Kernel for Upper Right Adjacency Pairs
	GenerateAdjacencyLists<<<blocks_per_grid, threads_per_block>>>((float4*)d_nodes, (int*)d_past_edges, (int*)d_future_edges, (int*)d_g_idx, N_tar / 2, 1);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Check Number of Connections
	//if (DEBUG) {
		checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(int)));
		checkCudaErrors(cuCtxSynchronize());
		//assert (*g_idx < max);
		printf("links: %d\n", *g_idx);
	//}

	cuMemFree(d_g_idx);
	g_idx = NULL;
	devMemUsed -= sizeof(int);

	//Execute Kernel
	/*DecodeAdjacencyLists<<<blocks_per_grid, threads_per_block>>>(d_past_edges, d_future_edges, d_past_edge_row_start, d_future_edge_row_start, *g_idx);
	getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());*/
	
	free(g_idx);
	g_idx = NULL;
	hostMemUsed -= sizeof(int);

	//Copy adjacency lists from Device to Host
	checkCudaErrors(cuMemcpyDtoH(past_edges, d_past_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer)));
	checkCudaErrors(cuMemcpyDtoH(future_edges, d_future_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer)));

	//Execute kernel to increment in-degrees and out-degrees from adjacency list pointers
	/*FindNodeDegrees<<<blocks_per_grid, threads_per_block>>>(d_past_edge_row_start, d_future_edge_row_start, d_k_in, d_k_out);

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Copy adjacency list pointers from Device to Host
	checkCudaErrors(cuMemcpyDtoH(past_edge_row_start, d_past_edge_row_start, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(future_edge_row_start, d_future_edge_row_start, sizeof(int) * N_tar));

	//Copy in-degree and out-degree counters from Device to Host
	checkCudaErrors(cuMemcpyDtoH(k_in, d_k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(k_out, d_k_out, sizeof(int) * N_tar));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Write contiguous degree counters back to 'nodes'
	for (i = 0; i < N_tar; i++) {
		nodes[i].k_in = k_in[i];
		nodes[i].k_out = k_out[i];
	}*/

	stopwatchStop(&sLinkNodesGPU);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodesGPU.elapsedTime);
		fflush(stdout);
	}

	return true;
}
