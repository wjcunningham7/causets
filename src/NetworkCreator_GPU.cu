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

__global__ void GenerateAdjacencyLists(float4 *nodes, int *past_edges, int *future_edges, int *g_idx, int N_tar, uint64_t N_pairs, int max)
{
	///////////////////////////////////////
	// Identify Node Pair with Thread ID //
	///////////////////////////////////////

	//Thread ID (unique within each block)
	//int tid = threadIdx.x;

	//Global Thread ID (unique among all threads)
	uint64_t gid = (static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i = static_cast<int>(gid / N_tar);
	int j = static_cast<int>(gid % N_tar);
	//bool valid = gid < N_pairs;
	bool valid = i < j;

	if (!valid)
		return;

	//Unique Key
	uint64_t key;

	//if (valid) {
		//Identify Node Pair
		//int i, j;
		//uint64_t k = valid ? vec2MatIdx(N_tar, gid) : 0;
		//__syncthreads();

		//i = static_cast<int>(k / N_tar);
		//j = static_cast<int>(k % N_tar);

		//Read Coordinates from Global Memory
		float4 node0 = valid ? nodes[i] : nodes[0];
		float4 node1 = valid ? nodes[j] : nodes[0];
	
		//////////////////////////////////
		// Identify Causal Relationship //
		//////////////////////////////////

		//Calculate dt (assumes nodes are already temporally ordered)
		float dt = valid ? node1.w - node0.w : 0.0;

		//Calculate dx
		float dx = valid ? acosf(X1_GPU(node0.y) * X1_GPU(node1.y) +
				 X2_GPU(node0.y, node0.z) * X2_GPU(node1.y, node1.z) +
				 X3_GPU(node0.y, node0.z, node0.x) * X3_GPU(node1.y, node1.z, node1.x) +
				 X4_GPU(node0.y, node0.z, node0.x) * X4_GPU(node1.y, node1.z, node1.x)) : 0.0;
		__syncthreads();

		//Calculate Pair Key
		key = (valid && dx < dt) ? i * static_cast<uint64_t>(N_tar) + j : 0;
	//}

	//int idx;
	if (valid && key) {
		//idx = atomicAdd(g_idx, key & 0x0001);
		//if (key & 0x0001) future_edges[idx] = key;
		future_edges[atomicAdd(g_idx, 1)] = key;
	}

	/////////////////////////////////
	// Operations in Shared Memory //
	/////////////////////////////////

	/*__shared__ int l_key[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ int d;
	__shared__ int l_idx;

	//Prefix Sum
	l_key[tid] = !key;
	__syncthreads();

	int diff = 0;
	int m;
	for (m = 0; m < tid; m++)
		diff += l_key[m];
	__syncthreads();

	//Reduction
	int lstride;
	for (lstride = 1; lstride < BLOCK_SIZE * BLOCK_SIZE; lstride <<= 1) {
		if (!(tid % (lstride << 1)))
			l_key[tid] += l_key[tid + lstride];
		__syncthreads();
	}

	//Number of Disconnected Pairs
	if (!tid)
		d = l_key[0];
	__syncthreads();

	//Compaction
	if (key)
		l_key[tid - diff] = key;

	////////////////////////////
	// Write to Global Memory //
	////////////////////////////

	if (!tid)
		l_idx = atomicAdd(g_idx, BLOCK_SIZE * BLOCK_SIZE - d);
	__syncthreads();

	if (tid < BLOCK_SIZE * BLOCK_SIZE - d) {
		//Read Keys from Local to Register Memory
		int f_r_key = l_key[tid];
		int p_r_key = (f_r_key % N_tar) * N_tar + (f_r_key / N_tar);

		//Write Keys from Register to Global Memory
		future_edges[l_idx + tid] = f_r_key;
		past_edges[l_idx + tid] = p_r_key;
	}*/
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
	//unsigned int grid_size = static_cast<unsigned int>(ceil(sqrt(static_cast<long double>(N_tar) * (N_tar - 1) / (2 * BLOCK_SIZE))));
	//unsigned int grid_size = static_cast<unsigned int>(ceil(sqrt(static_cast<long double>(N_tar) * N_tar / BLOCK_SIZE)));
	unsigned int grid_size = 884;
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid(grid_size, grid_size, 1);
	uint64_t N_pairs = static_cast<uint64_t>(N_tar) * (N_tar - 1) / 2;

	//Execute Kernel
	GenerateAdjacencyLists<<<blocks_per_grid, threads_per_block>>>((float4*)d_nodes, (int*)d_past_edges, (int*)d_future_edges, (int*)d_g_idx, N_tar, N_pairs, max);
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
