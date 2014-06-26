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

/*__global__ void GenerateAdjacencyLists(CUtexObject t_nodes, float4 *nodes, uint64_t *edges, int *g_idx, int width)
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
		node1_c = tex1Dfetch<float4>(t_nodes, j_c);

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
}*/

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

__global__ void BitonicSort(uint64_t *edges, int n_links, int j, int k)
{
	//Sorting Parameters
	unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned ixj = i^j;

	//Threads with the lowest IDs sort the list
	if (ixj > i) {
		if (!(i & k))
			//Sort Ascending
			if (edges[i] > edges[ixj])
				swap(edges, i, ixj);

		if (i & k)
			//Sort Descending
			if (edges[i] < edges[ixj])
				swap(edges, i, ixj);
	}
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

bool linkNodesGPU(Node &nodes, CUdeviceptr d_nodes, int * const &past_edges, CUdeviceptr d_past_edges, int * const &future_edges, CUdeviceptr d_future_edges, int * const &past_edge_row_start, CUdeviceptr d_past_edge_row_start, int * const &future_edge_row_start, CUdeviceptr d_future_edge_row_start, bool * const &core_edge_exists, CUdeviceptr d_k_in, CUdeviceptr d_k_out, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench)
{
	//Add assert statements

	stopwatchStart(&sLinkNodesGPU);

	//Allocate Overhead on Host
	int *g_idx;
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

	CUdeviceptr d_edges;
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer)));
	devMemUsed += sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer);
	
	CUdeviceptr d_g_idx;
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(int)));
	devMemUsed += sizeof(int);
	
	//Allocate Mapped Pinned Memory
	checkCudaErrors(cuMemHostGetDevicePointer(&d_k_in, (void*)nodes.k_in, 0));
	checkCudaErrors(cuMemHostGetDevicePointer(&d_k_out, (void*)nodes.k_out, 0));
	//checkCudaErrors(cuMemHostGetDevicePointer(&d_past_edges, (void*)past_edges, 0));
	//checkCudaErrors(cuMemHostGetDevicePointer(&d_future_edges, (void*)future_edges, 0));

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", hostMemUsed, devMemUsed);

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_nodes, nodes.sc, sizeof(float4) * N_tar));

	//Create Texture Object
	/*CUDA_RESOURCE_DESC res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = CU_RESOURCE_TYPE_LINEAR;
	res_desc.res.linear.devPtr = d_nodes;
	res_desc.res.linear.format = CU_AD_FORMAT_FLOAT;
	res_desc.res.linear.numChannels = 4;
	res_desc.res.linear.sizeInBytes = sizeof(float4) * N_tar;
	res_desc.flags = 0;

	CUDA_TEXTURE_DESC tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));

	CUtexObject t_nodes = 0;
	checkCudaErrors(cuTexObjectCreate(&t_nodes, &res_desc, &tex_desc, NULL));*/

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 1));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//CUDA Grid Specifications
	unsigned int gridx_GAL = static_cast<unsigned int>(ceil((static_cast<float>(N_tar) / 2) / BLOCK_SIZE));
	unsigned int gridy_GAL = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / 2));
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_GAL(gridx_GAL, gridy_GAL, 1);

	//Execute Kernel
	GenerateAdjacencyLists<<<blocks_per_grid_GAL, threads_per_block>>>((float4*)d_nodes, (uint64_t*)d_edges, (int*)d_k_in, (int*)d_k_out, (int*)d_g_idx, N_tar / 2);
	//GenerateAdjacencyLists<<<blocks_per_grid_GAL, threads_per_block>>>(t_nodes, (float4*)d_nodes, (uint64_t*)d_edges, (int*)d_g_idx, N_tar / 2);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Check Number of Connections
	checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(int)));
	checkCudaErrors(cuCtxSynchronize());
	printf("links: %d\n", *g_idx);

	//Free Texture Object
	//cuTexObjectDestroy(t_nodes);

	//Free Device Memory
	cuMemFree(d_nodes);
	d_nodes = NULL;
	devMemUsed -= sizeof(float4) * N_tar;

	//Bitonic Sort
	//Borrowed from https://gist.github.com/mre/1392067
	unsigned int gridx_bitonic = static_cast<unsigned int>(ceil(static_cast<double>(*g_idx) / BLOCK_SIZE));
	dim3 blocks_per_grid_bitonic(gridx_bitonic, 1, 1);
	int j, k;

	for (k = 2; k <= *g_idx; k <<= 1)
		for (j = k >> 1; j > 0; j >>= 1)
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, *g_idx, j, k);
	getLastCudaError("Kernel 'NetworkCreator_GPU.BitonicSort' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Decode 'edges' and Write to 'past_edges' and 'future_edges'
	
	//Free Host Memory
	free(g_idx);
	g_idx = NULL;
	hostMemUsed -= sizeof(int);
	
	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * (N_tar * k_tar / 2 + edge_buffer);
	
	cuMemFree(d_g_idx);
	d_g_idx = NULL;
	devMemUsed -= sizeof(int);

	//Parallel Prefix Sum of 'k_in' and 'k_out' and Write to Edge Pointers

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
