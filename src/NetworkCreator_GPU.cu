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

__global__ void GenerateAdjacencyLists(float4 *nodes, uint64_t *edges, int *g_idx, int width)
{
	///////////////////////////////////////
	// Identify Node Pair with Thread ID //
	///////////////////////////////////////

	__shared__ float4 shr_node0_c;
	float4 node0_ab, node0_c, node1_ab, node1_c;

	int tid = threadIdx.x;
	int i = blockIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int do_map = i >= j;

	// a -> upper triangle of upper left block (do_map == 0)
	// b -> lower triangle of upper left block (do_map == 1)
	// c -> upper right block

	int i_ab = i + do_map * (2 * (width - i) - 1);
	int j_ab = j + do_map * (2 * (width - j));

	int i_c = i;
	int j_c = j + width;

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
		dx_ab = acosf(X1_GPU(node0_ab.y) * X1_GPU(node1_ab.y) +
			      X2_GPU(node0_ab.y, node0_ab.z) * X2_GPU(node1_ab.y, node1_ab.z) +
			      X3_GPU(node0_ab.y, node0_ab.z, node0_ab.x) * X3_GPU(node1_ab.y, node1_ab.z, node1_ab.x) +
			      X4_GPU(node0_ab.y, node0_ab.z, node0_ab.x) * X4_GPU(node1_ab.y, node1_ab.z, node1_ab.x));

		dx_c = acosf(X1_GPU(node0_c.y) * X1_GPU(node1_c.y) +
			     X2_GPU(node0_c.y, node0_c.z) * X2_GPU(node1_c.y, node1_c.z) +
			     X3_GPU(node0_c.y, node0_c.z, node0_c.x) * X3_GPU(node1_c.y, node1_c.z, node1_c.x) +
			     X4_GPU(node0_c.y, node0_c.z, node0_c.x) * X4_GPU(node1_c.y, node1_c.z, node1_c.x));
	}

	//Write to Global Memory
	int idx = atomicAdd(g_idx, (dx_ab < dt_ab) + (dx_c < dt_c));
	if (dx_ab < dt_ab)
		edges[idx++] = ((uint64_t)i_ab) << 32 | ((uint64_t)j_ab);
	if (dx_c < dt_c)
		edges[idx] = ((uint64_t)i_c) << 32 | ((uint64_t)j_c);
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

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_nodes, sizeof(float4) * N_tar));
	devMemUsed += sizeof(float4) * N_tar;

	CUdeviceptr d_edges;
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer)));
	devMemUsed += sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer);

	CUdeviceptr d_g_idx;
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(int)));
	devMemUsed += sizeof(int);

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", hostMemUsed, devMemUsed);

	//Copy Node Coordinates to Contiguous Memory on Host
	for (i = 0; i < N_tar; i++) {
		coord[i].w = nodes[i].eta;
		coord[i].x = nodes[i].theta;
		coord[i].y = nodes[i].phi;
		coord[i].z = nodes[i].chi;
	}

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_nodes, coord, sizeof(float4) * N_tar));

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_edges, 0, N_tar * k_tar + 2 * edge_buffer));
	checkCudaErrors(cuMemsetD32(d_past_edge_row_start, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_future_edge_row_start, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 1));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Free Host Memory
	free(coord);
	coord = NULL;
	hostMemUsed -= sizeof(float4) * N_tar;

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil((static_cast<float>(N_tar) / 2) / BLOCK_SIZE));
	unsigned int gridy = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / 2));
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);

	//Execute Kernel
	GenerateAdjacencyLists<<<blocks_per_grid, threads_per_block>>>((float4*)d_nodes, (uint64_t*)d_edges, (int*)d_g_idx, N_tar / 2);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Check Number of Connections
	checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(int)));
	checkCudaErrors(cuCtxSynchronize());
	printf("links: %d\n", *g_idx);

	//Free Host Memory
	free(g_idx);
	g_idx = NULL;
	hostMemUsed -= sizeof(int);

	//Free Device Memory
	cuMemFree(d_nodes);
	d_nodes = NULL;
	devMemUsed -= sizeof(float4) * N_tar;

	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer);

	cuMemFree(d_g_idx);
	g_idx = NULL;
	devMemUsed -= sizeof(int);

	//Copy adjacency lists from Device to Host
	/*checkCudaErrors(cuMemcpyDtoH(past_edges, d_past_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer)));
	checkCudaErrors(cuMemcpyDtoH(future_edges, d_future_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer)));*/

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
