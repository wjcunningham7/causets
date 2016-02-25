#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

//Debug:  Future vs Past Edges in Adjacency List
//O(1) Efficiency
void compareAdjacencyLists(const Node &nodes, const Edge &edges)
{
	#if DEBUG
	//No null pointers
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	#endif

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
	#if DEBUG
	//No null pointers
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	#endif

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

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int * const future_edge_row_start, const Bitset adj, const int &N_tar, const float &core_edge_fraction)
{
	#if DEBUG
	assert (k_out != NULL);
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);
	assert (N_tar > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif
	
	int core_limit = static_cast<int>(core_edge_fraction * N_tar);
	int idx1, idx2;
	uint64_t idx12, idx21;
	int i, j;

	try {
		for (i = 0; i < core_limit; i++) {
			idx1 = i;

			#if DEBUG
			assert (!(future_edge_row_start[idx1] == -1 && k_out[idx1] > 0));
			assert (!(future_edge_row_start[idx1] != -1 && k_out[idx1] == 0));
			#endif

			for (j = 0; j < k_out[idx1]; j++) {
				idx2 = future_edges[future_edge_row_start[idx1]+j];

				if (idx2 >= core_limit)
					continue;

				idx12 = static_cast<uint64_t>(idx1) * core_limit + idx2;
				idx21 = static_cast<uint64_t>(idx2) * core_limit + idx1;

				//printf("idx12: %" PRIu64 "\tidx21: %" PRIu64 "\n", idx12, idx21);

				if (!adj[idx12] || !adj[idx21])
					throw CausetException("Adjacency matrix does not match sparse list!\n");
			}
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

#ifdef CUDA_ENABLED
__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, unsigned long long int *g_idx, int width, bool compact)
{
	///////////////////////////////////////
	// Identify Node Pair with Thread ID //
	///////////////////////////////////////

	__shared__ float shr_w0_c;
	__shared__ float shr_x0_c;
	__shared__ float shr_y0_c;
	__shared__ float shr_z0_c;
	__shared__ int n_a[BLOCK_SIZE];
	__shared__ int n_b[BLOCK_SIZE];
	__shared__ int n_c[BLOCK_SIZE];
	float4 node0_ab, node0_c, node1_ab, node1_c;

	unsigned int tid = threadIdx.y;
	unsigned int i = blockIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	int do_map = i >= j;

	// a -> upper triangle of upper left block (do_map == 0)
	// b -> lower triangle of upper left block (do_map == 1)
	// c -> upper right block

	unsigned int i_ab = i + do_map * (((width - i) << 1) - 1);
	unsigned int j_ab = j + do_map * ((width - j) << 1);

	unsigned int i_c = i;
	unsigned int j_c = j + width;

	if (!tid) {
		shr_w0_c = w[i_c];
		shr_x0_c = x[i_c];
		shr_y0_c = y[i_c];
		shr_z0_c = z[i_c];
	}
	__syncthreads();

	float dt_ab = 0.0f, dt_c = 0.0f, dx_ab = 0.0f, dx_c = 0.0f;
	if (j < width) {
		node0_c.w = shr_w0_c;
		node0_c.x = shr_x0_c;
		node0_c.y = shr_y0_c;
		node0_c.z = shr_z0_c;

		node1_c.w = w[j_c];
		node1_c.x = x[j_c];
		node1_c.y = y[j_c];
		node1_c.z = z[j_c];

		node0_ab.w = do_map ? w[i_ab] : node0_c.w;
		node0_ab.x = do_map ? x[i_ab] : node0_c.x;
		node0_ab.y = do_map ? y[i_ab] : node0_c.y;
		node0_ab.z = do_map ? z[i_ab] : node0_c.z;

		node1_ab.w = !j ? node0_ab.w : w[j_ab];
		node1_ab.x = !j ? node0_ab.x : x[j_ab];
		node1_ab.y = !j ? node0_ab.y : y[j_ab];
		node1_ab.z = !j ? node0_ab.z : z[j_ab];

		//////////////////////////////////
		// Identify Causal Relationship //
		//////////////////////////////////

		//Calculate dt (assumes nodes are already temporally ordered)
		dt_ab = node1_ab.w - node0_ab.w;
		dt_c  = node1_c.w  - node0_c.w;

		//Calculate dx
		if (compact) {
			if (DIST_V2) {
				dx_ab = acosf(sphProduct_GPU_v2(node0_ab, node1_ab));
				dx_c = acosf(sphProduct_GPU_v2(node0_c, node1_c));
			} else {
				dx_ab = acosf(sphProduct_GPU_v1(node0_ab, node1_ab));
				dx_c = acosf(sphProduct_GPU_v1(node0_c, node1_c));
			}
		} else {
			if (DIST_V2) {
				dx_ab = sqrtf(flatProduct_GPU_v2(node0_ab, node1_ab));
				dx_c = sqrtf(flatProduct_GPU_v2(node0_c, node1_c));
			} else {
				dx_ab = sqrtf(flatProduct_GPU_v1(node0_ab, node1_ab));
				dx_c = sqrtf(flatProduct_GPU_v1(node0_c, node1_c));
			}
		}
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

//Note that adj has not been implemented in this version of the linkNodesGPU subroutine.
bool linkNodesGPU_v1(Node &nodes, const Edge &edges, Bitset &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, CaResources * const ca, Stopwatch &sLinkNodesGPU, const bool &verbose, const bool &bench)
{
	#if DEBUG
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
	assert (ca != NULL);

	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	#if EMBED_NODES
	fprintf(stderr, "linkNodesGPU_v2 not implemented for EMBED_NODES=true.  Find me on line %d in %s.\n", __LINE__, __FILE__);
	#endif

	Stopwatch sGPUOverhead = Stopwatch();
	Stopwatch sGenAdjList = Stopwatch();
	Stopwatch sBitonic0 = Stopwatch();
	Stopwatch sDecode0 = Stopwatch();
	Stopwatch sBitonic1 = Stopwatch();
	Stopwatch sDecode1 = Stopwatch();
	Stopwatch sProps = Stopwatch();

	CUdeviceptr d_w;
	CUdeviceptr d_x;
	CUdeviceptr d_y;
	CUdeviceptr d_z;
	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_N_res, d_N_deg2;
	CUdeviceptr d_g_idx;

	unsigned long long int *g_idx;
	int j, k;
	
	bool compact = get_curvature(spacetime) & POSITIVE;

	stopwatchStart(&sLinkNodesGPU);
	stopwatchStart(&sGPUOverhead);

	//Allocate Overhead on Host
	try {
		g_idx = (unsigned long long int*)malloc(sizeof(unsigned long long int));
		if (g_idx == NULL)
			throw std::bad_alloc();
		memset(g_idx, 0, sizeof(unsigned long long int));
		ca->hostMemUsed += sizeof(unsigned long long int);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_w, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_x, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_y, sizeof(float) * N_tar));
	checkCudaErrors(cuMemAlloc(&d_z, sizeof(float) * N_tar));
	ca->devMemUsed += sizeof(float) * N_tar * 4;

	size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar * (1.0 + edge_buffer) / 2)));
	//printf("d_edges_size: %zd\n", d_edges_size);
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	ca->devMemUsed += sizeof(uint64_t) * d_edges_size;

	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;
	
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(unsigned long long int)));
	ca->devMemUsed += sizeof(unsigned long long int);

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", ca->hostMemUsed, ca->devMemUsed, 0);

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_w, nodes.crd->w(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_x, nodes.crd->x(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_y, nodes.crd->y(), sizeof(float) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_z, nodes.crd->z(), sizeof(float) * N_tar));

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_edges, 0, d_edges_size << 1));
	checkCudaErrors(cuMemsetD32(d_k_in, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_k_out, 0, N_tar));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 2));

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
	GenerateAdjacencyLists_v1<<<blocks_per_grid_GAL, threads_per_block_GAL>>>((float*)d_w, (float*)d_x, (float*)d_y, (float*)d_z, (uint64_t*)d_edges, (int*)d_k_in, (int*)d_k_out, (unsigned long long int*)d_g_idx, N_tar >> 1, compact);
	getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists' Failed to Execute!\n");

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGenAdjList);
	stopwatchStart(&sGPUOverhead);

	//Check Number of Connections
	checkCudaErrors(cuMemcpyDtoH(g_idx, d_g_idx, sizeof(unsigned long long int)));
	checkCudaErrors(cuCtxSynchronize());

	//Free Device Memory
	cuMemFree(d_w);
	d_w = 0;

	cuMemFree(d_x);
	d_x = 0;

	cuMemFree(d_y);
	d_y = 0;

	cuMemFree(d_z);
	d_z = 0;

	ca->devMemUsed -= sizeof(float) * N_tar * 4;

	cuMemFree(d_g_idx);
	d_g_idx = 0;
	ca->devMemUsed -= sizeof(unsigned long long int);

	try {
		if (*g_idx + 1 >= static_cast<uint64_t>(N_tar) * k_tar * (1.0 + edge_buffer) / 2)
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
	ca->devMemUsed += sizeof(int) * d_edges_size;

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_future_edges, 0, d_edges_size));

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<double>(*g_idx) / BLOCK_SIZE));
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
	d_future_edges = 0;
	ca->devMemUsed -= sizeof(int) * d_edges_size;

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
	ca->devMemUsed += sizeof(int) * d_edges_size;

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
	d_edges = 0;
	ca->devMemUsed -= sizeof(uint64_t) * d_edges_size;

	cuMemFree(d_past_edges);
	d_past_edges = 0;
	ca->devMemUsed -= sizeof(int) * d_edges_size;

	//Resulting Network Properties

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_N_res, sizeof(int)));
	ca->devMemUsed += sizeof(int);

	checkCudaErrors(cuMemAlloc(&d_N_deg2, sizeof(int)));
	ca->devMemUsed += sizeof(int);
	
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

	//Prefix Sum of 'k_in' and 'k_out'
	scan(nodes.k_in, nodes.k_out, edges.past_edge_row_start, edges.future_edge_row_start, N_tar);

	N_res = N_tar - N_res;
	N_deg2 = N_tar - N_deg2;
	k_res = static_cast<float>(static_cast<double>(*g_idx) * 2 / N_res);

	#if DEBUG
	assert (N_res > 0);
	assert (N_deg2 > 0);
	assert (k_res > 0.0);
	#endif

	//Free Device Memory
	cuMemFree(d_k_in);
	d_k_in = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_k_out);
	d_k_out = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_N_res);
	d_N_res = 0;
	ca->devMemUsed -= sizeof(int);

	cuMemFree(d_N_deg2);
	d_N_deg2 = 0;
	ca->devMemUsed -= sizeof(int);

	stopwatchStop(&sGPUOverhead);
	stopwatchStop(&sLinkNodesGPU);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tUndirected Links:         %" PRId64 "\n", *g_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", (k_res * N_res) / N_tar);
		printf_red();
		printf("\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		printf_std();
		fflush(stdout);
	}
	
	//if(!compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction))
	//	return false;

	//Print Results
	/*if (!printDegrees(nodes, N_tar, "in-degrees_GPU_v1.cset.dbg.dat", "out-degrees_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, *g_idx, "past-edges_GPU_v1.cset.dbg.dat", "future-edges_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_GPU_v1.cset.dbg.dat", "future-edge-pointers_GPU_v1.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	printChk();*/

	//Free Host Memory
	free(g_idx);
	g_idx = NULL;
	ca->hostMemUsed -= sizeof(unsigned long long int);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodesGPU.elapsedTime);
		printf("\t\t\tGPU Overhead Time: %5.6f sec\n", sGPUOverhead.elapsedTime);
		printf("\t\t\tAdjacency List Kernel Time: %5.6f sec\n", sGenAdjList.elapsedTime);
		printf("\t\t\tBitonic Sort 0 Kernel Time: %5.6f sec\n", sBitonic0.elapsedTime);
		printf("\t\t\tFuture Edge Decode Time: %5.6f sec\n", sDecode0.elapsedTime);
		printf("\t\t\tBitonic Sort 1 Kernel Time: %5.6f sec\n", sBitonic1.elapsedTime);
		printf("\t\t\tPast Edge Decode Time: %5.6f sec\n", sDecode1.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool generateLists_v1(Node &nodes, uint64_t * const &edges, Bitset &adj, int64_t * const &g_idx, const unsigned int &spacetime, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &use_bit, const bool &verbose)
{
	#if DEBUG
	assert (nodes.crd->getDim() == 4);
	assert (!nodes.crd->isNull());
	assert (nodes.crd->w() != NULL);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.crd->z() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (edges != NULL);
	assert (g_idx != NULL);
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	if (use_bit)
		assert (core_edge_fraction == 1.0f);
	#endif

	//Temporary Buffers
	CUdeviceptr d_w0, d_x0, d_y0, d_z0;
	CUdeviceptr d_w1, d_x1, d_y1, d_z1;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_edges;

	int *h_k_in;
	int *h_k_out;
	bool *h_edges;

	unsigned int core_limit = static_cast<unsigned int>(core_edge_fraction * N_tar);
	unsigned int i, j;
	bool diag;

	bool compact = get_curvature(spacetime) & POSITIVE;

	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE * group_size)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

	//DEBUG
	/*#if DEBUG
	printf_red();
	printf("\nTHREAD  SIZE: %d\n", THREAD_SIZE);
	printf("BLOCK   SIZE: %d\n", BLOCK_SIZE);
	printf("GROUP   SIZE: %d\n", group_size);
	printf("MBLOCK  SIZE: %zd\n", mblock_size);
	printf("MTHREAD SIZE: %zd\n", mthread_size);
	printf("Number of Times Kernel is Executed: %d\n\n", (group_size*group_size));
	printf_std();
	fflush(stdout);
	#endif*/

	//Allocate Buffers on Host
	try {
		h_k_in = (int*)malloc(sizeof(int) * mthread_size);
		if (h_k_in == NULL)
			throw std::bad_alloc();
		memset(h_k_in, 0, sizeof(int) * mthread_size);
		ca->hostMemUsed += sizeof(int) * mthread_size;

		h_k_out = (int*)malloc(sizeof(int) * mthread_size);
		if (h_k_out == NULL)
			throw std::bad_alloc();
		memset(h_k_out, 0, sizeof(int) * mthread_size);
		ca->hostMemUsed += sizeof(int) * mthread_size;

		h_edges = (bool*)malloc(sizeof(bool) * m_edges_size);
		if (h_edges == NULL)
			throw std::bad_alloc();
		memset(h_edges, 0, sizeof(bool) * m_edges_size);
		ca->hostMemUsed += sizeof(bool) * m_edges_size;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	//Allocate Node Buffers on Device
	checkCudaErrors(cuMemAlloc(&d_w0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_x0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_z0, sizeof(float) * mthread_size));
	ca->devMemUsed += sizeof(float) * mthread_size * 4;

	checkCudaErrors(cuMemAlloc(&d_w1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_x1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_z1, sizeof(float) * mthread_size));
	ca->devMemUsed += sizeof(float) * mthread_size * 4;

	//Allocate Degree Buffers on Device
	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
	ca->devMemUsed += sizeof(int) * mthread_size;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * mthread_size));
	checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));
	ca->devMemUsed += sizeof(int) * mthread_size;

	//Allocate Edge Buffer on Device
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(bool) * m_edges_size));
	checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));
	ca->devMemUsed += sizeof(bool) * m_edges_size;

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Generating Lists on GPU", ca->hostMemUsed, ca->devMemUsed, 0);

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil(static_cast<float>(mthread_size) / THREAD_SIZE));
	unsigned int gridy = mblock_size;
	dim3 threads_per_block(1, BLOCK_SIZE, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);

	//DEBUG
	/*printf_red();
	printf("Grid X: %u\n", gridx);
	printf("Grid Y: %u\n", gridy);
	printf_std();
	fflush(stdout);*/

	size_t final_size = N_tar - mthread_size * (group_size - 1);
	size_t size0, size1;

	//Index 'i' marks the row and 'j' marks the column
	for (i = 0; i < group_size; i++) {
		for (j = 0; j < group_size; j++) {
			if (i > j)
				continue;

			diag = (i == j);

			size0 = (i < group_size - 1) ? mthread_size : final_size;
			size1 = (j < group_size - 1) ? mthread_size : final_size;

			//Copy node values to device buffers
			checkCudaErrors(cuMemcpyHtoD(d_w0, nodes.crd->w() + i * mthread_size, sizeof(float) * size0));
			checkCudaErrors(cuMemcpyHtoD(d_x0, nodes.crd->x() + i * mthread_size, sizeof(float) * size0));
			checkCudaErrors(cuMemcpyHtoD(d_y0, nodes.crd->y() + i * mthread_size, sizeof(float) * size0));
			checkCudaErrors(cuMemcpyHtoD(d_z0, nodes.crd->z() + i * mthread_size, sizeof(float) * size0));

			checkCudaErrors(cuMemcpyHtoD(d_w1, nodes.crd->w() + j * mthread_size, sizeof(float) * size1));
			checkCudaErrors(cuMemcpyHtoD(d_x1, nodes.crd->x() + j * mthread_size, sizeof(float) * size1));
			checkCudaErrors(cuMemcpyHtoD(d_y1, nodes.crd->y() + j * mthread_size, sizeof(float) * size1));
			checkCudaErrors(cuMemcpyHtoD(d_z1, nodes.crd->z() + j * mthread_size, sizeof(float) * size1));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Execute Kernel
			GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, diag, compact);
			getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists_v2' Failed to Execute!\n");

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Copy edges to host
			checkCudaErrors(cuMemcpyDtoH(h_edges, d_edges, sizeof(bool) * m_edges_size));

			//Copy degrees to host
			checkCudaErrors(cuMemcpyDtoH(h_k_in, d_k_in, sizeof(int) * size1));
			checkCudaErrors(cuMemcpyDtoH(h_k_out, d_k_out, sizeof(int) * size0));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Transfer data from buffers
			readDegrees(nodes.k_in, h_k_in, j * mthread_size, size1);
			readDegrees(nodes.k_out, h_k_out, i * mthread_size, size0);
			readEdges(edges, h_edges, adj, g_idx, core_limit, d_edges_size, mthread_size, size0, size1, i, j, use_bit);

			//Clear Device Memory
			checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
			checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));
			checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());			
		}
	}

	cuMemFree(d_w0);
	d_w0 = 0;

	cuMemFree(d_x0);
	d_x0 = 0;

	cuMemFree(d_y0);
	d_y0 = 0;

	cuMemFree(d_z0);
	d_z0 = 0;

	ca->devMemUsed -= sizeof(float) * mthread_size * 4;

	cuMemFree(d_w1);
	d_w1 = 0;

	cuMemFree(d_x1);
	d_x1 = 0;

	cuMemFree(d_y1);
	d_y1 = 0;

	cuMemFree(d_z1);
	d_z1 = 0;

	ca->devMemUsed -= sizeof(float) * mthread_size * 4;

	cuMemFree(d_k_in);
	d_k_in = 0;
	ca->devMemUsed -= sizeof(int) * mthread_size;

	cuMemFree(d_k_out);
	d_k_out = 0;
	ca->devMemUsed -= sizeof(int) * mthread_size;

	cuMemFree(d_edges);
	d_edges = 0;
	ca->devMemUsed -= sizeof(bool) * m_edges_size;

	free(h_k_in);
	h_k_in = NULL;
	ca->hostMemUsed -= sizeof(int) * mthread_size;

	free(h_k_out);
	h_k_out = NULL;
	ca->hostMemUsed -= sizeof(int) * mthread_size;

	free(h_edges);
	h_edges = NULL;
	ca->hostMemUsed -= sizeof(bool) * m_edges_size;

	return true;
}

//Decode past and future edge lists using Bitonic Sort
bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int64_t * const g_idx, const size_t &d_edges_size, CaResources * const ca, const bool &verbose)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (h_edges != NULL);
	assert (g_idx != NULL);
	assert (ca != NULL);
	assert (d_edges_size > 0);
	#endif

	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	int j, k;

	//Allocate Global Device Memory
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	ca->devMemUsed += sizeof(uint64_t) * d_edges_size;

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

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_future_edges, sizeof(int) * d_edges_size));
	ca->devMemUsed += sizeof(int) * d_edges_size;

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Bitonic Sorting", ca->hostMemUsed, ca->devMemUsed, 0);

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_future_edges, 0, d_edges_size));

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<double>(*g_idx) / BLOCK_SIZE));
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
	d_future_edges = 0;
	ca->devMemUsed -= sizeof(int) * d_edges_size;

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
	ca->devMemUsed += sizeof(int) * d_edges_size;

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
	d_edges = 0;
	ca->devMemUsed -= sizeof(uint64_t) * d_edges_size;

	cuMemFree(d_past_edges);
	d_past_edges = 0;
	ca->devMemUsed -= sizeof(int) * d_edges_size;

	return true;
}
#endif

//Generate confusion matrix for geodesic distances
//Compares timelike/spacelike in 4D/5D
bool validateEmbedding(EVData &evd, Node &nodes, const Edge &edges, const Bitset adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const double &N_emb, const int &N_res, const float &k_res, const double &a, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sValidateEmbedding, const bool &verbose)
{
	#if DEBUG
	assert (nodes.crd->getDim() == 4);
	assert (!nodes.crd->isNull());
	assert (nodes.crd->w() != NULL);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.crd->z() != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (ca != NULL);

	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (get_stdim(spacetime) == 4);
	assert (get_manifold(spacetime) & (DE_SITTER | FLRW));
	assert (a > 0.0);
	if (get_manifold(spacetime) & FLRW)
		assert (alpha > 0.0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	uint64_t max_pairs = static_cast<uint64_t>(N_tar) * (N_tar - 1) / 2;
	uint64_t stride = max_pairs / static_cast<uint64_t>(N_emb);
	uint64_t npairs = static_cast<uint64_t>(N_emb);
	uint64_t start = 0;
	uint64_t finish = npairs;
	int rank = cmpi.rank;

	#ifdef MPI_ENABLED
	uint64_t core_edges_size = static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT));
	int edges_size = static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer) / 2);
	#endif

	stopwatchStart(&sValidateEmbedding);

	//printf("Number of paths to test: %" PRIu64 "\n", static_cast<uint64_t>(N_emb));
	//printf("Stride: %" PRIu64 "\n", stride);

	try {
		evd.confusion = (uint64_t*)malloc(sizeof(uint64_t) * 4);
		if (evd.confusion == NULL) {
			cmpi.fail = 1;
			goto ValEmbPoint;
		}
		memset(evd.confusion, 0, sizeof(uint64_t) * 4);
		ca->hostMemUsed += sizeof(uint64_t) * 4;

		ValEmbPoint:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}

		memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
		if (verbose)
			printMemUsed("for Embedding Validation", ca->hostMemUsed, ca->devMemUsed, rank);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->w(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->x(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->y(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->z(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.id.tau, N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Bcast(nodes.k_out, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edge_row_start, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	//MPI_Bcast(adj, core_edges_size, MPI_C_BOOL, 0, MPI_COMM_WORLD);

	uint64_t mpi_chunk = npairs / cmpi.num_mpi_threads;
	start = rank * mpi_chunk;
	finish = start + mpi_chunk;
	#endif

	uint64_t c0 = evd.confusion[0];
	uint64_t c1 = evd.confusion[1];
	uint64_t c2 = evd.confusion[2];
	uint64_t c3 = evd.confusion[3];

	#ifdef _OPENMP
	unsigned int seed = static_cast<unsigned int>(mrng.rng() * 4000000000);
	#pragma omp parallel reduction (+ : c0, c1, c2, c3)
	{
	Engine eng(seed ^ omp_get_thread_num());
	UDistribution dst(0.0, 1.0);
	UGenerator rng(eng, dst);
	#pragma omp for schedule (dynamic, 1)
	#else
	UGenerator &rng = mrng.rng;
	#endif
	for (uint64_t k = start; k < finish; k++) {
		//Choose a pair
		uint64_t vec_idx;

		if (VE_RANDOM) {
			#ifdef _OPENMP
			vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
			#else
			vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
			#endif
		} else
			vec_idx = k * stride;

		int i = static_cast<int>(vec_idx / (N_tar - 1));
		int j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		int do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		//Embedded distance
		double distance = distanceEmb(nodes.crd->getFloat4(i), nodes.id.tau[i], nodes.crd->getFloat4(j), nodes.id.tau[j], spacetime, a, alpha);

		//Check light cone condition for 4D vs 5D
		//Null hypothesis is the nodes are not connected
		if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, i, j)) {
			if (distance > 0)
				//True Negative (both timelike)
				c1++;
			else
				//False Positive
				c2++;
		} else {	//Actual Spacelike (Positive)
			if (distance > 0)
				//False Negative
				c3++;
			else
				//True Positive (both spacelike)
				c0++;
		}		
	}

	#ifdef _OPENMP
	}
	#endif

	evd.confusion[0] = c0;
	evd.confusion[1] = c1;
	evd.confusion[2] = c2;
	evd.confusion[3] = c3;

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank)
		MPI_Reduce(MPI_IN_PLACE, evd.confusion, 4, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(evd.confusion, NULL, 4, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	#endif

	//Number of timelike distances
	evd.A1T = (N_res * k_res / 2) * ((double)npairs / max_pairs);
	//Number of spacelike distances
	evd.A1S = npairs - evd.A1T;

	stopwatchStop(&sValidateEmbedding);

	printf_mpi(rank, "\tCalculated Embedding Confusion Matrix.\n");
	if (!rank) printf_cyan();
	printf_mpi(rank, "\t\tTrue  Positives: %f\t(4D spacelike, 5D spacelike)\n", static_cast<double>(evd.confusion[0]) / evd.A1S);
	printf_mpi(rank, "\t\tTrue  Negatives: %f\t(4D timelike,  5D timelike)\n", static_cast<double>(evd.confusion[1]) / evd.A1T);
	if (!rank) printf_red();
	printf_mpi(rank, "\t\tFalse Positives: %f\t(4D timelike,  5D spacelike)\n", static_cast<double>(evd.confusion[2]) / evd.A1T);
	printf_mpi(rank, "\t\tFalse Negatives: %f\t(4D spacelike, 5D timelike)\n", static_cast<double>(evd.confusion[3]) / evd.A1S);
	if (!rank) printf_std();
	fflush(stdout);

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sValidateEmbedding.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Used to compare distance calculated with embedding to
//distances calculated with exact formula
//NOTE: This only works with de Sitter since there is not a known
//formula for the embedded FLRW distance.
bool validateDistances(DVData &dvd, Node &nodes, const unsigned int &spacetime, const int &N_tar, const double &N_dst, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sValidateDistances, const bool &verbose)
{
	#if DEBUG
	assert (nodes.crd->getDim() == 4);
	assert (!nodes.crd->isNull());
	assert (nodes.crd->w() != NULL);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.crd->z() != NULL);
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (get_stdim(spacetime) == 4);
	assert (get_manifold(spacetime) & DE_SITTER);
	assert (a > 0.0);
	#endif

	bool DST_DEBUG = false;

	uint64_t max_pairs = static_cast<uint64_t>(N_tar) * (N_tar - 1) / 2;
	uint64_t stride = max_pairs / static_cast<uint64_t>(N_dst);
	uint64_t npairs = static_cast<uint64_t>(N_dst);
	uint64_t start = 0;
	uint64_t finish = npairs;

	double tol = 1.0e-2;

	stopwatchStart(&sValidateDistances);

	try {
		dvd.confusion = (uint64_t*)malloc(sizeof(uint64_t) * 2);
		if (dvd.confusion == NULL)
			throw std::bad_alloc();
		memset(dvd.confusion, 0, sizeof(uint64_t) * 2);
		ca->hostMemUsed += sizeof(uint64_t) * 2;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Validate de Sitter Distance Algorithm", ca->hostMemUsed, ca->devMemUsed, 0);

	uint64_t c0 = dvd.confusion[0];
	uint64_t c1 = dvd.confusion[1];

	#ifdef _OPENMP
	unsigned int seed = static_cast<unsigned int>(mrng.rng() * 4000000000);
	#pragma omp parallel reduction(+ : c0, c1)
	{
	Engine eng(seed ^ omp_get_thread_num());
	UDistribution dst(0.0, 1.0);
	UGenerator rng(eng, dst);
	#pragma omp for schedule (dynamic, 1)
	#else
	UGenerator &rng = mrng.rng;
	#endif
	for (uint64_t k = start; k < finish; k++) {
		//Choose a pair
		uint64_t vec_idx;
		if (VD_RANDOM) {
			#ifdef _OPENMP
			vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
			#else
			vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
			#endif
		} else
			vec_idx = k * stride;

		int i = static_cast<int>(vec_idx / (N_tar - 1));
		int j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		int do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		//Distance using embedding
		double embeddedDistance = ABS(distanceEmb(nodes.crd->getFloat4(i), nodes.id.tau[i], nodes.crd->getFloat4(j), nodes.id.tau[j], spacetime, a, alpha), STL);
		//if (embeddedDistance + 1.0 < INF)
		//	printf("\n\tEmbedded Distance: %f\n", embeddedDistance);

		//Distance using exact formula
		double exactDistance = 0.0;
		if (get_curvature(spacetime) & POSITIVE)
			exactDistance = distanceDeSitterSph(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, i, j);
		else if (get_curvature(spacetime) & FLAT)
			exactDistance = distanceDeSitterFlat(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, i, j);
		//if (exactDistance + 1.0 < INF)
		//	printf("\tExactDistance:     %f\n", exactDistance);

		double abserr = ABS(embeddedDistance - exactDistance, STL) / embeddedDistance;

		if (exactDistance != -1 && abserr < tol) {
			if (DST_DEBUG) {
				printf_cyan();
				printf("SUCCESS\n");
			}
			c0++;
		} else {
			if (DST_DEBUG) {
				printf_red();
				printf("FAILURE\t%f\n", abserr);
				printf("\t%f\t%f\n", embeddedDistance, exactDistance);
			}
			c1++;
		}

		if (DST_DEBUG) {
			printf_std();
			fflush(stdout);
		}
		
		/*if (embeddedDistance < 100) {
			printf("\nTesting Lookup Table...\n");
			testOmega12(nodes.id.tau[i], nodes.id.tau[j], ACOS(sphProduct_v2(nodes.crd->getFloat4(i), nodes.crd->getFloat4(j)), STL, VERY_HIGH_PRECISION), -0.15, 0.5, 0.001, manifold);
			printf("\n");
			fflush(stdout);
			#ifndef _OPENMP
			break;
			#endif
		}*/
	}

	#ifdef _OPENMP
	}
	#endif

	dvd.confusion[0] = c0;
	dvd.confusion[1] = c1;
	dvd.norm = static_cast<double>(npairs);

	stopwatchStop(&sValidateDistances);

	printf("\tCalculated Distances Confusion Matrix.\n");
	printf_cyan();
	printf("\t\tMatching    Pairs: %f\n", static_cast<double>(dvd.confusion[0]) / dvd.norm);
	printf_red();
	printf("\t\tConflicting Pairs: %f\n", static_cast<double>(dvd.confusion[1]) / dvd.norm);
	printf_std();
	printf("\t\tNumber of Samples: %" PRIu64 "\n", npairs);
	fflush(stdout);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sValidateDistances.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(Node &nodes, const unsigned int &spacetime, const int num_vals, const char *filename, const char *coord)
{
	#if DEBUG
	//No null pointers
	assert (filename != NULL);
	assert (coord != NULL);

	//Variables in correct range
	assert (num_vals > 0);
	#endif

	try {
		char full_name[80] = "ST-";
		snprintf(&full_name[3], 80, "%d", spacetime);
		strcat(full_name, "_");
		strcat(full_name, filename);

		std::ofstream outputStream;
		outputStream.open(full_name);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		outputStream << std::setprecision(10);
		for (i = 0; i < num_vals; i++) {
			if (!strcmp(coord, "tau"))
				outputStream << nodes.id.tau[i] << std::endl;
			else if (!strcmp(coord, "eta")) {
				if (get_stdim(spacetime) == 2)
					outputStream << nodes.crd->x(i) << std::endl;
				else if (get_stdim(spacetime) == 4)
					#if EMBED_NODES
					outputStream << nodes.crd->v(i) << std::endl;
					#else
					outputStream << nodes.crd->w(i) << std::endl;
					#endif
			} else if (!strcmp(coord, "theta1"))
				outputStream << nodes.crd->x(i) << std::endl;
			else if (!strcmp(coord, "theta2"))
				outputStream << nodes.crd->y(i) << std::endl;
			else if (!strcmp(coord, "theta3"))
				outputStream << nodes.crd->z(i) << std::endl;
			else if (!strcmp(coord, "u")) {
				if (get_stdim(spacetime) == 2)
					outputStream << (nodes.crd->x(i) + nodes.crd->y(i)) / sqrt(2.0) << std::endl;
				else if (get_stdim(spacetime) == 4)
					outputStream << (nodes.crd->w(i) + nodes.crd->x(i)) / sqrt(2.0) << std::endl;
			} else if (!strcmp(coord, "v")) {
				if (get_stdim(spacetime) == 2)
					outputStream << (nodes.crd->x(i) - nodes.crd->y(i)) / sqrt(2.0) << std::endl;
				else if (get_stdim(spacetime) == 4)
					outputStream << (nodes.crd->w(i) - nodes.crd->x(i)) / sqrt(2.0) << std::endl;
			} else
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
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (filename_in != NULL);
	assert (filename_out != NULL);
	assert (num_vals > 0);
	#endif

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

bool printEdgeLists(const Edge &edges, const int64_t num_vals, const char *filename_past, const char *filename_future)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (filename_past != NULL);
	assert (filename_future != NULL);
	assert (num_vals >= 0);
	#endif

	try {
		std::ofstream outputStream_past;
		outputStream_past.open(filename_past);
		if (!outputStream_past.is_open())
			throw CausetException("Failed to open past-edge file in 'printEdgeLists' function!\n");

		std::ofstream outputStream_future;
		outputStream_future.open(filename_future);
		if (!outputStream_future.is_open())
			throw CausetException("Failed to open future-edges file in 'printEdgeLists' function!\n");

		for (int64_t i = 0; i < num_vals; i++) {
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
	#if DEBUG
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (filename_past != NULL);
	assert (filename_future != NULL);
	assert (num_vals > 0);
	#endif

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

//Searches a range of lambdas for a match to omega12
bool testOmega12(float tau1, float tau2, const double &omega12, const double min_lambda, const double max_lambda, const double lambda_step, const unsigned int &spacetime)
{
	#if DEBUG
	assert (tau1 > 0.0f);
	assert (tau2 > 0.0f);
	assert (omega12 > 0.0);
	assert (min_lambda < max_lambda);
	assert (lambda_step > 0.0);
	assert (get_manifold(spacetime) & (DE_SITTER | FLRW));
	#endif

	printf("\tTesting Geodesic Lookup Algorithm...\n");
	fflush(stdout);

	if (!(get_manifold(spacetime) & (DE_SITTER | FLRW)))
		return false;

	if (tau1 > tau2) {
		float temp = tau1;
		tau1 = tau2;
		tau2 = temp;
	}

	bool DS_EXACT = false;

	double (*kernel)(double x, void *params) = (get_manifold(spacetime) & FLRW) ? &flrwLookupKernel : &deSitterLookupKernel;

	IntData idata = IntData();
	idata.limit = 50;
	idata.tol = 1e-5;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	double tol = 1e-3;
	int n_lambda = static_cast<int>((max_lambda - min_lambda) / lambda_step);

	double omega_val;
	int i;

	printf("tau1: %f\ttau2: %f\tomega12: %f\n", tau1, tau2, omega12);
	fflush(stdout);

	for (i = 0; i < n_lambda; i++) {
		double lambda = i * lambda_step + min_lambda;
		double tau_m = geodesicMaxTau(get_manifold(spacetime), lambda);
		printf("tau_m:   %f\t", tau_m);
		fflush(stdout);

		if (tau1 >= tau2 || lambda == 0.0)
			omega_val = 0.0;
		else if (lambda > 0) {
			if (DS_EXACT) {
				double ov0 = deSitterLookupExact(static_cast<double>(tau1), lambda);
				double ov1 = deSitterLookupExact(static_cast<double>(tau2), lambda);
				if (ov0 == 0.0 || ov1 == 0.0)
					omega_val = 0.0;
				else
					omega_val = ov1 - ov0;
			} else {
				idata.lower = tau1;
				idata.upper = tau2;
				omega_val = integrate1D(kernel, (void*)&lambda, &idata, QAGS);
			}
		} else if (lambda < 0 && (tau1 < tau_m && tau2 < tau_m)) {

			if (DS_EXACT) {
				double ov0 = deSitterLookupExact(tau_m, lambda);
				double ov1 = deSitterLookupExact(static_cast<double>(tau1), lambda);
				double ov2 = deSitterLookupExact(static_cast<double>(tau2), lambda);
				if (ov0 == 0.0 || ov1 == 0.0 || ov2 == 0.0)
					omega_val = 0.0;
				else
					omega_val = 2.0 * ov0 - ov1 - ov2;
			} else {
				idata.lower = tau1;
				idata.upper = tau_m;
				omega_val = integrate1D(kernel, (void*)&lambda, &idata, QAGS);

				idata.lower = tau2;
				double omega_val2 = integrate1D(kernel, (void*)&lambda, &idata, QAGS);
				omega_val += omega_val2;
			}
		} else
			omega_val = 0.0;

		printf("lambda: %f\tomega12: %f\n", lambda, omega_val);
		fflush(stdout);

		if (ABS(omega12 - omega_val, STL) / omega12 < tol) {
			printf("MATCH!\n");
			fflush(stdout);
			break;
		}
	}
	
	gsl_integration_workspace_free(idata.workspace);

	printf("\tTask Completed.\n");
	fflush(stdout);

	return true;
}

//Generates the lookup tables
bool generateGeodesicLookupTable(const char *filename, const double max_tau, const double min_lambda, const double max_lambda, const double tau_step, const double lambda_step, const unsigned int &spacetime, const bool &verbose)
{
	#if DEBUG
	assert (filename != NULL);
	assert (max_tau > 0.0);
	assert (min_lambda < max_lambda);
	assert (tau_step > 0.0);
	assert (lambda_step > 0.0);
	assert (get_manifold(spacetime) & (DE_SITTER | FLRW));
	#endif

	if (get_manifold(spacetime) & FLRW)
		printf("\tGenerating FLRW geodesic lookup table...\n");
	else if (get_manifold(spacetime) & DE_SITTER)
		printf("\tGenerating de Sitter geodesic lookup table...\n");
	else
		return false;
	fflush(stdout);

	//Only set this to 'true' when a de Sitter table is being generated
	//and the two methods (integration v. exact) should be compared
	bool DS_EXACT = false;

	double (*kernel)(double x, void *params) = (get_manifold(spacetime) & FLRW) ? &flrwLookupKernel : &deSitterLookupKernel;

	Stopwatch sLookup = Stopwatch();
	IntData idata = IntData();
	idata.limit = 50;
	idata.tol = 1e-5;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	int n_tau = static_cast<int>(max_tau / tau_step);
	int n_lambda = static_cast<int>((max_lambda - min_lambda) / lambda_step);

	double tau1, tau2, tau_m, omega12, lambda;
	int i, j, k;

	try {
		FILE *table = fopen(filename, "wb");
		if (table == NULL)
			throw CausetException("Failed to open geodesic lookup table!\n");

		stopwatchStart(&sLookup);

		double zero = 0.0;
		double n_tau_d = static_cast<double>(n_tau);
		double n_lambda_d = static_cast<double>(n_lambda);
		fwrite(&zero, sizeof(double), 1, table);
		fwrite(&zero, sizeof(double), 1, table);
		fwrite(&n_tau_d, sizeof(double), 1, table);
		fwrite(&n_lambda_d, sizeof(double), 1, table);

		//printf("tau1\t\ttau2\t\tomega12\tlambda\n");
		for (i = 0; i < n_tau; i++) {
			tau1 = i * tau_step;
			for (j = 0; j < n_tau; j++) {
				tau2 = j * tau_step;

				for (k = 0; k < n_lambda; k++) {
					lambda = k * lambda_step + min_lambda;
					tau_m = geodesicMaxTau(get_manifold(spacetime), lambda);

					if (tau1 >= tau2 || lambda == 0.0)
						omega12 = 0.0;
					else if (lambda > 0) {
						if (DS_EXACT) {
							double ov0 = deSitterLookupExact(static_cast<double>(tau1), lambda);
							double ov1 = deSitterLookupExact(static_cast<double>(tau2), lambda);

							if (ov0 == 0.0 || ov1 == 0.0)
								omega12 = 0.0;
							else
								omega12 = ov1 - ov0;
						} else {
							idata.lower = tau1;
							idata.upper = tau2;
							omega12 = integrate1D(kernel, (void*)&lambda, &idata, QAGS);
						}
					} else if (lambda < 0 && (tau1 < tau_m && tau2 < tau_m)) {
						if (DS_EXACT) {
							double ov0 = deSitterLookupExact(tau_m, lambda);
							double ov1 = deSitterLookupExact(static_cast<double>(tau1), lambda);
							double ov2 = deSitterLookupExact(static_cast<double>(tau2), lambda);

							if (ov0 == 0.0 || ov1 == 0.0 || ov2 == 0.0)
								omega12 = 0.0;
							else
								omega12 = 2.0 * ov0 - ov1 - ov2;
						} else {
							idata.lower = tau1;
							idata.upper = tau_m;
							omega12 = integrate1D(kernel, (void*)&lambda, &idata, QAGS);

							idata.lower = tau2;
							omega12 += integrate1D(kernel, (void*)&lambda, &idata, QAGS);
						}
					} else
						omega12 = 0.0;

					//Write to file
					fwrite(&tau1, sizeof(double), 1, table);
					fwrite(&tau2, sizeof(double), 1, table);
					fwrite(&omega12, sizeof(double), 1, table);
					fwrite(&lambda, sizeof(double), 1, table);
				}
			}
		}

		stopwatchStop(&sLookup);

		fclose(table);
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	gsl_integration_workspace_free(idata.workspace);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLookup.elapsedTime);
		fflush(stdout);
	}
	
	printf("\tTask Completed.\n");
	fflush(stdout);

	return true;
}

//Debug and validate distance approximation algorithm
bool validateDistApprox(const Node &nodes, const Edge &edges, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha)
{
	//This line is VERY important if this code should be portable
	assert (sizeof(long double) == 16);

	//printf("\nBeginning Timelike Analysis:\n");
	printf("\nBeginning Spacelike Analysis:\n");
	printf("----------------------------\n");

	int past_idx = 0;
	//int future_idx = edges.future_edges[0];
	int future_idx = 1;
	printf_cyan();
	//printf("Studying timelike relation [%d - %d]\n", past_idx, future_idx);
	printf("Studying spacelike relation [%d - %d]\n", past_idx, future_idx);
	printf_std();

	double omega12;
	double dt = nodes.crd->w(future_idx) - nodes.crd->w(past_idx);
	nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);
	printf("dt: %f\tdx: %f\n", dt, omega12);

	double x1 = POW(SINH(1.5 * nodes.id.tau[past_idx], STL), 1.0 / 3.0, STL);
	double x2 = POW(SINH(1.5 * nodes.id.tau[future_idx], STL), 1.0 / 3.0, STL);
	printf("x1: %f\tx2: %f\n", x1, x2);

	IntData idata = IntData();
	idata.tol = 1.0e-10;
	idata.lower = x1;
	idata.upper = x2;

	double lambda, omega_res;	
	/*printf("\nAttempting large lambda approximation.\n");
	lambda = omega12 / (2.0 * integrate1D(&flrwLookupApprox, NULL, &idata, QNG));
	printf("lambda:\t\t%f\n", lambda);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QNG);
	printf("Resulting dx:\t%f\n", omega_res);
	if (ABS(omega_res - omega12, STL) / omega12 > 1e-3)
		printf("Attempt failed.\n");
	else
		printf("Attempt succeeded.\n");*/

	printf("\nStudying omegaRegion1.\n");
	double om1;
	double err = 1.0E-10;
	int nterms = 10;

	printf("Testing single point.\n");
	//lambda = 0.05;
	lambda = -0.05;
	double z1 = -1.0 * lambda * x1 * x1 * x1 * x1;
	double z2 = -1.0 * lambda * x2 * x2 * x2 * x2;
	double mz = 1.0;
	om1 = omegaRegion1(x1, lambda, z1, &err, &nterms);
	printf("omega(x = %f, lambda = %f) = %f\n", x1, lambda, om1);
	printf("Resulting Error: %.8e\n", err);
	printf("Terms Used: %d\n", nterms);

	printf("\nAttempting to find lambda via bisection method.\n");
	double x0;
	double res = 1.0, tol = 1e-5;
	//double lower = 0.0, upper = 10000.0;
	double lower = -1.0 / POW2(POW2(x2, EXACT), EXACT), upper = 0.0;
	double mx;
	int iter = 0, max_iter = 10000;
	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		z1 = -1.0 * x0 * x1 * x1 * x1 * x1;
		z2 = -1.0 * x0 * x2 * x2 * x2 * x2;
		//Added for spacelike study
		err = 1.0E-10;
		nterms = 10;
		mx = geodesicMaxX(x0);
		res = 2.0 * omegaRegion1(mx, x0, mz, &err, &nterms);
		//End addition
		err = 1.0E-10;
		nterms = 10;
		//Changed from = to -= for spacelike
		res -= omegaRegion1(x2, x0, z2, &err, &nterms);
		assert (res == res);
		err = 1.0E-10;
		nterms = 10;
		res -= omegaRegion1(x1, x0, z1, &err, &nterms);
		assert (res == res);
		//printf("omega(x1 = %f, x2 = %f, lambda = %f) = %f\n", x1, x2, x0, res);
		res -= omega12;
		//NOTE: This is > for timelike, < for spacelike!
		if (res < 0.0)
			lower = x0;
		else
			upper = x0;
		iter++;
	}
	lambda = x0;
	printf("Finished bisection after %d iterations.\n", iter);
	printf("Identified lambda = %f\n", lambda);

	printf("\nInserting lambda into original integral.\n");
	//This block has been revised for spacelike distances
	idata.upper = mx;
	idata.limit = 60;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	idata.lower = x2;
	omega_res += 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Resulting dx:\t%.8e\n", omega_res);
	printf("Error:\t\t%.8e\n", ABS(omega_res - omega12, STL) / omega12);	

	printf("\nCalculating Distance.\n");
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	idata.limit = 60;
	double tm = geodesicMaxTau(get_manifold(spacetime), lambda);
	idata.lower = nodes.id.tau[past_idx];
	idata.upper = tm;
	double distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	idata.lower = nodes.id.tau[future_idx];
	distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Distance: %.8e\n", distance);
	//printf("Completed timelike study of omegaRegion1.\n");
	printf("Completed spacelike study of omegaRegion1.\n");

	//Time algorithm
	Stopwatch sdb = Stopwatch();
	int ntests = 1000;
	stopwatchStart(&sdb);
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	for (int i = 0; i < ntests; i++) {
		idata.lower = x1;
		idata.upper = mx;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
		idata.lower = x2;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	}
	gsl_integration_workspace_free(idata.workspace);
	stopwatchStop(&sdb);
	double intTime = sdb.elapsedTime;

	stopwatchReset(&sdb);
	stopwatchStart(&sdb);
	for (int i = 0; i < ntests; i++) {
		//omegaRegion1(mx, x0, &err, &nterms);
		omegaRegion1(x2, x0, z2, &err, &nterms);
		omegaRegion1(x1, x0, z2, &err, &nterms);
	}
	stopwatchStop(&sdb);
	double approxTime = sdb.elapsedTime;

	printf("Time for integration:   %.6f\n", intTime);
	printf("Time for approximation: %.6f\n", approxTime);

	past_idx = 1500;
	//future_idx = 5935;
	future_idx = 1501;
	printf_cyan();
	//printf("\nStudying timelike relation [%d - %d]\n", past_idx, future_idx);
	printf("\nStudying spacelike relation [%d - %d]\n", past_idx, future_idx);
	printf_std();
	dt = nodes.crd->w(future_idx) - nodes.crd->w(past_idx);
	nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);
	printf("dt: %f\tdx: %f\n", dt, omega12);

	x1 = POW(SINH(1.50 * nodes.id.tau[past_idx], STL), 1.0 / 3.0, STL);
	x2 = POW(SINH(1.5 * nodes.id.tau[future_idx], STL), 1.0 / 3.0, STL);
	printf("x1: %f\tx2: %f\n", x1, x2);

	/*printf("\nAttempting large lambda approximation.\n");
	idata.lower = x1;
	idata.upper = x2;
	lambda = omega12 / (2.0 * integrate1D(&flrwLookupApprox, NULL, &idata, QNG));
	printf("lambda:\t\t%f\n", lambda);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QNG);
	printf("Resulting dx:\t%f\n", omega_res);
	if (ABS(omega_res - omega12, STL) / omega12 > 1e-3)
		printf("Attempt failed.\n");
	else
		printf("Attempt succeeded.\n");*/

	//printf("\nStudying omegaRegion2a.\n");
	printf("\nStudying omegaRegion2b.\n");
	double om2;

	printf("Testing single point.\n");
	//lambda = 0.05;
	lambda = -0.05;
	err = 1.0E-10;
	nterms = 15;
	//om2 = omegaRegion2a(x2, lambda, &err, &nterms);
	om2 = omegaRegion2b(x1, lambda, &err, &nterms);
	printf("omega(x = %f, lambda = %f) = %f\n", x1, lambda, om2);
	printf("Terms Used: %d\n", nterms);

	printf("\nAttempting to find lambda via the bisection method.\n");
	res = 1.0;
	tol = 1.0e-5;
	//lower = 0.0;
	//upper = 10000.0;
	lower = -1.0 / POW2(POW2(x2, EXACT), EXACT);
	upper = 0.0;
	iter = 0;
	max_iter = 10000;
	while (ABS(res, STL) > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		//Changed 2a to 2b for spacelike
		mx = geodesicMaxX(x0);
		res = 2.0 * omegaRegion2b(mx, x0, &err, &nterms);
		assert (res == res);
		//Changed = to -= for spacelike
		res -= omegaRegion2b(x2, x0, &err, &nterms);
		assert (res == res);
		res -= omegaRegion2b(x1, x0, &err, &nterms);
		assert (res == res);
		//printf("omega(x1 = %f, x2 = %f, lambda = %f) = %f\n", x1, x2, x0, res);
		res -= omega12;
		//Changed > to < for spacelike
		if (res < 0.0)
			lower = x0;
		else
			upper = x0;
		iter++;
	}
	lambda = x0;
	printf("Finished bisection after %d iterations.\n", iter);
	printf("Identifed lambda = %f\n", lambda);

	printf("\nInserting lambda into original integral.\n");
	//This block has been revised for spacelike distances
	idata.lower = x1;
	idata.upper = mx;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	idata.lower = x2;
	omega_res += 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Resulting dx:\t%.8e\n", omega_res);
	printf("Error:\t\t%.8e\n", ABS(omega_res - omega12, STL) / omega12);

	printf("\nCalculating Distance.\n");
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	tm = geodesicMaxTau(get_manifold(spacetime), lambda);
	idata.lower = nodes.id.tau[past_idx];
	idata.upper = tm;
	distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	idata.lower = nodes.id.tau[future_idx];
	distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Distance: %.8e\n", distance);
	//printf("Completed timelike study of omegaRegion2a.\n");
	printf("Completed spacelike study of omegaRegion2b.\n");

	stopwatchReset(&sdb);
	stopwatchStart(&sdb);
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	for (int i = 0; i < ntests; i++) {
		idata.lower = x1;
		idata.upper = mx;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
		idata.lower = x2;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	}
	gsl_integration_workspace_free(idata.workspace);
	stopwatchStop(&sdb);
	intTime = sdb.elapsedTime;

	stopwatchReset(&sdb);
	nterms = 6;
	stopwatchStart(&sdb);
	for (int i = 0; i < ntests; i++) {
		omegaRegion2b(mx, lambda, &err, &nterms);
		omegaRegion2b(x1, lambda, &err, &nterms);
		omegaRegion2b(x2, lambda, &err, &nterms);
	}
	stopwatchStop(&sdb);
	approxTime = sdb.elapsedTime;

	printf("Time for integration:   %.6f\n", intTime);
	printf("Time for approximation: %.6f\n", approxTime);

	past_idx = 7530;
	//future_idx = 9705;
	future_idx = 7531;
	printf_cyan();
	//printf("\nStudying timelike relation [%d - %d]\n", past_idx, future_idx);
	printf("\nStudying spacelike relation [%d - %d]\n", past_idx, future_idx);
	printf_std();

	dt = nodes.crd->w(future_idx) - nodes.crd->w(past_idx);
	nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);
	printf("dt: %f\tdx: %f\n", dt, omega12);

	x1 = POW(SINH(1.5 * nodes.id.tau[past_idx], STL), 1.0 / 3.0, STL);
	x2 = POW(SINH(1.5 * nodes.id.tau[future_idx], STL), 1.0 / 3.0, STL);
	printf("x1: %f\tx2: %f\n", x1, x2);

	/*printf("\nAttempting large lambda approximation.\n");
	idata.lower = x1;
	idata.upper = x2;
	lambda = omega12 / (2.0 * integrate1D(&flrwLookupApprox, NULL, &idata, QNG));
	printf("lambda:\t\t%f\n", lambda);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QNG);
	printf("Resulting dx:\t%f\n", omega_res);
	if (ABS(omega_res - omega12, STL) / omega12 > 1e-3)
		printf("Attempt failed.\n");
	else
		printf("Attempt succeeded.\n");*/

	printf("\nStudying omegaRegion3.\n");
	double *table;
	double om3;
	long size = 0L;
	getLookupTable("./etc/tables/partial_fraction_coefficients.cset.bin", &table, &size);
	assert (table != NULL);

	printf("Testing single pair.\n");
	//lambda = 3.0;
	lambda = -0.4;
	z1 = -1.0 * lambda * x1 * x1 * x1 * x1;
	z2 = -1.0 * lambda * x2 * x2 * x2 * x2;
	err = 1.0E-10;
	nterms = 10;
	om3 = omegaRegion3(table, x1, x2, lambda, z1, z2, &err, &nterms, size);
	printf("omega(x1 = %f, x2 = %f, lambda = %f) = %e\n", x1, x2, lambda, om3);
	printf("Resulting Error: %.8e\n", err);
	printf("Terms Used: %d\n", nterms);

	printf("\nAttempting to find lambda via the bisection method.\n");
	res = 1.0;
	tol = 1.0e-5;
	//lower = 0.0;
	//upper = 1000.0;
	lower = -1.0 / POW2(POW2(x2, EXACT), EXACT);
	upper = 0.0;
	iter = 0;
	max_iter = 10000;
	while(ABS(res, STL) > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		z1 = -1.0 * x0 * x1 * x1 * x1 * x1;
		z2 = -1.0 * x0 * x2 * x2 * x2 * x2;
		mx = geodesicMaxX(x0);
		err = 1.0E-10;
		nterms = 10;
		//Changed x2 to mx for spacelike
		res = omegaRegion3(table, x1, mx, x0, z1, mz, &err, &nterms, size);
		assert (res == res);
		//Added for spacelike
		err = 1.000E-10;
		nterms = 10;
		res += omegaRegion3(table, x2, mx, x0, z2, mz, &err, &nterms, size);
		assert (res == res);
		//printf("omega(x1 = %f, x2 = %f, lambda = %f) = %f\n", x1, x2, x0, res);
		res -= omega12;
		//Changed > to < for spacelike
		if (res < 0.0)
			lower = x0;
		else
			upper = x0;
		iter++;
	}
	lambda = x0;
	printf("Finished bisection after %d iterations.\n", iter);
	printf("Identified lambda = %f\n", lambda);

	printf("\nInserting lambda into original integral.\n");
	//This block has been revised for spacelike distances
	idata.lower = x1;
	idata.upper = mx;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	omega_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	idata.lower = x2;
	omega_res += 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Resulting dx:\t%.8e\n", omega_res);
	printf("Error:\t\t%.8e\n", ABS(omega_res - omega12, STL) / omega12);

	printf("\nCalculating Distance.\n");
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	tm = geodesicMaxTau(get_manifold(spacetime), lambda);
	idata.lower = nodes.id.tau[past_idx];
	idata.upper = tm;
	distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	idata.lower = nodes.id.tau[future_idx];
	distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	gsl_integration_workspace_free(idata.workspace);
	printf("Distance: %.8e\n", distance);
	//printf("Completed timelike study of omegaRegion3.\n");
	printf("Completed spacelike study of omegaRegion3.\n");

	stopwatchReset(&sdb);
	stopwatchStart(&sdb);
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	for (int i = 0; i < ntests; i++) {
		idata.lower = x1;
		idata.upper = mx;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
		idata.lower = x2;
		integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QAGS);
	}
	gsl_integration_workspace_free(idata.workspace);
	stopwatchStop(&sdb);
	intTime = sdb.elapsedTime;

	stopwatchReset(&sdb);
	z1 = -1.0 * lambda * x1 * x1 * x1 * x1;
	z2 = -1.0 * lambda * x2 * x2 * x2 * x2;
	stopwatchStart(&sdb);
	for (int i = 0; i < ntests; i++) {
		err = 1.0E-10;
		nterms = 10;
		omegaRegion3(table, x1, mx, lambda, z1, mz, &err, &nterms, size);
		//omegaRegion3a(mx, lambda, zm, &err, &nterms);
		//omegaRegion3b(table, mx, lambda, &err, &nterms, size);
		err = 1.0E-10;
		nterms = 10;
		omegaRegion3(table, x2, mx, lambda, z2, mz, &err, &nterms, size);
		//omegaRegion3a(x1, lambda, z1, &err, &nterms);
		//omegaRegion3b(table, mx, lambda, &err, &nterms, size);
	}
	stopwatchStop(&sdb);
	approxTime = sdb.elapsedTime;

	printf("Time for integration:   %.6f\n", intTime);
	printf("Time for approximation: %.6f\n", approxTime);

	/*printf("\nHypergeometric Test.\n");
	double f;
	double f_err = 1.0e-10;
	int f_nt = -1;
	lambda = -0.501845;
	double z = -1.0 * lambda * POW2(POW2(x1, EXACT), EXACT);
	double w = 1.0 - z;
	_2F1(-9.0, 1.0, 1.5, w, &f, &f_err, &f_nt, false);
	printf("2F1(-9, 1, 1.5, %f) = %f\n", w, f);
	printf("Error: %.8e\n", f_err);

	printf("\nCoefficient Test.\n");
	double n0 = _2F1_An(-9.0, 1.0, 1.5, 8);
	double n1 = _2F1_An(-9.0, 1.0, 1.5, 9);
	double n2 = _2F1_An(-9.0, 1.0, 1.5, 10);
	double n3 = _2F1_An(-9.0, 1.0, 1.5, 11);
	printf("n0: %f\nn1: %f\nn2: %f\nn3: %f\n", n0, n1, n2, n3);

	printf("\nPochhammer Test.\n");
	double p0 = POCHHAMMER(-9.0, 9);
	double p1 = POCHHAMMER(1.0, 9);
	double p2 = POCHHAMMER(1.5, 9);
	double p3 = GAMMA(10.0, STL);
	double p4 = p0 * p1 / (p2 * p3);
	printf("p0: %f\np1: %f\np2: %f\np3: %f\np4: %f\n", p0, p1, p2, p3, p4);*/

	return true;
}

//Node Traversal Algorithm
//Not accelerated with OpenMP
//Uses geodesic distances
bool traversePath_v1(const Node &nodes, const Edge &edges, const Bitset adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, int source, int dest, bool &success, bool &past_horizon)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW | HYPERBOLIC));

	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);

	if (get_stdim(spacetime) == 2) {
		assert (nodes.crd->getDim() == 2);
		assert (get_manifold(spacetime) & (DE_SITTER | HYPERBOLIC));
	} else if (get_stdim(spacetime) == 4) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (used != NULL);

	assert (N_tar > 0);
	if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
		assert (a > 0.0);
		if (get_manifold(spacetime) & (DUST | FLRW)) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (get_curvature(spacetime) & POSITIVE) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (get_curvature(spacetime) & FLAT) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}
	if (get_curvature(spacetime) & FLAT)
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (source >= 0 && source < N_tar);
	assert (dest >= 0 && dest < N_tar);
	#endif

	bool TRAV_DEBUG = false;

	float min_dist = 0.0f;
	int loc = source;
	int idx_a = source;
	int idx_b = dest;

	float dist;
	int next;
	int m;

	//Check if source and destination can be connected by any geodesic
	if (get_manifold(spacetime) & (DE_SITTER | FLRW)) {
		if ((get_manifold(spacetime) & DE_SITTER) && (get_curvature(spacetime) & POSITIVE))
			dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
		else
			dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
	} else if (get_manifold(spacetime) & DUST)
		dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
	else if (get_manifold(spacetime) & HYPERBOLIC)
		dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
	else
		return false;

	if (dist == -1)
		return false;

	if (dist + 1.0 > INF) {
		past_horizon = true;
		return true;
	}

	if (TRAV_DEBUG) {
		printf_cyan();
		printf("Beginning at [%d : %.4f].\tLooking for [%d : %.4f].\n", source, nodes.id.tau[source], dest, nodes.id.tau[dest]);
		printf_std();
		fflush(stdout);
	}

	//While the current location (loc) is not equal to the destination (dest)
	while (loc != dest) {
		next = loc;
		dist = INF;
		min_dist = INF;
		used[loc] = true;

		//These indicate corrupted data
		#if DEBUG
		assert (!(edges.past_edge_row_start[loc] == -1 && nodes.k_in[loc] > 0));
		assert (!(edges.past_edge_row_start[loc] != -1 && nodes.k_in[loc] == 0));
		assert (!(edges.future_edge_row_start[loc] == -1 && nodes.k_out[loc] > 0));
		assert (!(edges.future_edge_row_start[loc] != -1 && nodes.k_out[loc] == 0));
		#endif

		//(1) Check past relations
		for (m = 0; m < nodes.k_in[loc]; m++) {
			idx_a = edges.past_edges[edges.past_edge_row_start[loc]+m];
			/*if (TRAV_DEBUG) {
				printf_cyan();
				printf("\tConsidering past neighbor %d.\n", idx_a);
				printf_std();
				fflush(stdout);
			}*/

			//(A) If the current location's past neighbor is the destination, return true
			if (idx_a == idx_b) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : %.4f].\n", idx_a, nodes.id.tau[idx_a]);
					printf_red();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				success = true;
				return true;
			}

			//(B) If the current location's past neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, idx_a, idx_b)) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : %.4f].\n", idx_a, nodes.id.tau[idx_a]);
					printf("Moving to [%d : %.4f].\n", idx_b, nodes.id.tau[idx_b]);
					printf_red();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				success = true;
				return true;
			}

			//(C) Otherwise find the past neighbor closest to the destination
			if (get_manifold(spacetime) & (DE_SITTER | FLRW)) {
				if ((get_manifold(spacetime) & DE_SITTER) && (get_curvature(spacetime) & POSITIVE))
					dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
				else
					dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			} else if (get_manifold(spacetime) & DUST)
				dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (get_manifold(spacetime) & HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
			else
				return false;

			//Check for errors in the 'distance' function
			if (dist == -1)
				return false;

			//Save the minimum distance
			if (dist < min_dist) {
				min_dist = dist;
				next = idx_a;
			}
		}

		//(2) Check future relations
		for (m = 0; m < nodes.k_out[loc]; m++) {
			idx_a = edges.future_edges[edges.future_edge_row_start[loc]+m];
			/*if (TRAV_DEBUG) {
				printf_cyan();
				printf("\tConsidering future neighbor %d.\n", idx_a);
				printf_std();
				fflush(stdout);
			}*/

			//(D) If the current location's future neighbor is the destination, return true
			if (idx_a == idx_b) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : %.4f].\n", idx_a, nodes.id.tau[idx_a]);
					printf_red();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				success = true;
				return true;
			}

			//(E) If the current location's future neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, idx_a, idx_b)) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : %.4f].\n", idx_a, nodes.id.tau[idx_a]);
					printf("Moving to [%d : %.4f].\n", idx_b, nodes.id.tau[idx_b]);
					printf_red();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				success = true;
				return true;
			}

			//(F) Otherwise find the future neighbor closest to the destination
			if (get_manifold(spacetime) & (DE_SITTER | FLRW)) {
				if ((get_manifold(spacetime) & DE_SITTER) && (get_curvature(spacetime) & POSITIVE))
					dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
				else
					dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			} else if (get_manifold(spacetime) & DUST)
				dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (get_manifold(spacetime) & HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
			else
				return false;

			//Check for errors in the 'distance' function
			if (dist == -1)
				return false;

			//Save the minimum distance
			if (dist < min_dist) {
				min_dist = dist;
				next = idx_a;
			}
		}

		if (TRAV_DEBUG && min_dist + 1.0 < INF) {
			printf_cyan();
			printf("Moving to [%d : %.4f].\n", next, nodes.id.tau[next]);
			printf_std();
			fflush(stdout);
		}

		if (!used[next] && min_dist + 1.0 < INF)
			loc = next;
		else {
			if (min_dist + 1.0 > INF)
				past_horizon = true;

			if (TRAV_DEBUG) {
				printf_red();
				printf("FAILURE\n");
				printf_std();
				fflush(stdout);
			}
			break;
		}
	}

	success = false;
	return true;
}

//Measure Causal Set Action
//O(N*k^2*ln(k)) Efficiency (Linked)
//O(N^2*k) Efficiency (No Links)
bool measureAction_v1(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const Bitset adj, const unsigned int &spacetime, const int &N_tar, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	if (!no_pos)
		assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) == 4);
	assert (get_manifold(spacetime) & DE_SITTER);

	if (!no_pos) {
		if (get_stdim(spacetime) == 2)
			assert (nodes.crd->getDim() == 2);
		else if (get_stdim(spacetime) == 4) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->z() != NULL);
		}

		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
	}

	if (!use_bit && (link || relink)) {
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}
	assert (ca != NULL);
		
	assert (N_tar > 0);
	assert (max_cardinality > 0);
	assert (a > 0.0);
	if (get_curvature(spacetime) & POSITIVE) {
		assert (zeta > 0.0);
		assert (zeta < HALF_PI);
	} else if (get_curvature(spacetime) & FLAT) {
		assert (zeta > HALF_PI);
		assert (zeta1 > HALF_PI);
		assert (zeta > zeta1);
	} 
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif

	int core_limit = static_cast<int>(core_edge_fraction * N_tar);
	int elements;
	int fstart, pstart;
	int i, j, k;
	bool too_many;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality data
	try {
		cardinalities = (int*)malloc(sizeof(int) * max_cardinality);
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(int) * max_cardinality);
		ca->hostMemUsed += sizeof(int) * max_cardinality;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Action", ca->hostMemUsed, ca->devMemUsed, 0);

	cardinalities[0] = N_tar;

	if (max_cardinality == 1)
		goto ActionExit;

	too_many = false;
	for (i = 0; i < N_tar - 1; i++) {
		for (j = i + 1; j < N_tar; j++) {
			elements = 0;
			if (!use_bit && (link || relink)) {
				if (!nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, i, j))
					continue;

				//These indicate corrupted data
				#if DEBUG
				assert (!(edges.past_edge_row_start[j] == -1 && nodes.k_in[j] > 0));
				assert (!(edges.past_edge_row_start[j] != -1 && nodes.k_in[j] == 0));
				assert (!(edges.future_edge_row_start[i] == -1 && nodes.k_out[i] > 0));
				assert (!(edges.future_edge_row_start[i] != -1 && nodes.k_out[i] == 0));
				#endif

				if (core_limit == N_tar) {
					int col0 = static_cast<uint64_t>(i) * core_limit;
					int col1 = static_cast<uint64_t>(j) * core_limit;
					for (k = i + 1; k < j; k++)
						elements += (int)(adj[col0+k] & adj[col1+k]);
					if (elements >= max_cardinality - 1)
						too_many = true;
				} else {
					pstart = edges.past_edge_row_start[j];
					fstart = edges.future_edge_row_start[i];
					//printf("\nLooking at %d future neighbors of [node %d] and %d past neighbors of [node %d].\n", nodes.k_out[i], i, nodes.k_in[j], j);
					causet_intersection_v2(elements, edges.past_edges, edges.future_edges, nodes.k_in[j], nodes.k_out[i], max_cardinality, pstart, fstart, too_many);
				}
			} else {
				if (!nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, i, j, NULL))
					continue;

				for (k = i + 1; k < j; k++) {
					if (nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, alpha, r_max, i, k, NULL) && nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, k, j, NULL))
						elements++;
					if (elements >= max_cardinality - 1) {
						too_many = true;
						break;
					}
				}
			}

			if (!too_many)
				cardinalities[elements+1]++;

			too_many = false;
		}
	}

	if (max_cardinality < 5)
		goto ActionExit;

	action = static_cast<float>(cardinalities[0] - cardinalities[1] + 9 * cardinalities[2] - 16 * cardinalities[3] + 8 * cardinalities[4]);
	action *= 4.0f / sqrtf(6.0f);
	
	ActionExit:
	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf("\tCalculated Action.\n");
		printf("\t\tTerms Used: %d\n", max_cardinality);
		printf_cyan();
		printf("\t\tCausal Set Action: %f\n", action);
		if (max_cardinality < 10)
			for (i = 0; i < max_cardinality; i++)
				printf("\t\t\tN%d: %d\n", i, cardinalities[i]);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

void validateCoordinates(const Node &nodes, const unsigned int &spacetime, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const int &i)
{
	#if EMBED_NODES
	float tol = 1.0e-4;
	double r;
	#endif
	switch (spacetime) {
	case (2 | MINKOWSKI | DIAMOND | FLAT | ASYMMETRIC):
		//printf("eta: %.10f\n", nodes.crd->x(i));
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < eta0);
		assert (iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0));
		break;
	case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
		//printf("eta: %.8f\n", nodes.crd->x(i));
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < eta0);
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
		#else
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
		#endif
		break;
	case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
		assert (fabs(nodes.crd->x(i)) < HALF_PI - zeta);
		assert (nodes.id.tau[i] > -tau0 && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
		#else
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
		#endif
		break;
	case (2 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
		fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
		assert (false);
		break;
	case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < HALF_PI - zeta);
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		//y, z
		#else
		//y
		#endif
		break;
	case (2 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
		fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
		assert (false);
		break;
	case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > HALF_PI - zeta && nodes.crd->v(i) < HALF_PI - zeta1);
		assert (POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < POW2(r_max, EXACT));
		#else
		assert (nodes.crd->w(i) > HALF_PI - zeta && nodes.crd->w(i) < HALF_PI - zeta1);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
		assert (nodes.id.tau[i] > -tau0 && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (fabs(nodes.crd->v(i)) < HALF_PI - zeta);
		assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
		#else
		assert (fabs(nodes.crd->v(i)) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		assert (iad(nodes.crd->v(i), r, HALF_PI - zeta, HALF_PI - zeta1));
		#else
		assert (nodes.crd->w(i) > HALF_PI - zeta && nodes.crd->w(i) < HALF_PI - zeta1);
		assert (iad(nodes.crd->w(i), nodes.crd->x(i), HALF_PI - zeta, HALF_PI - zeta1));
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		r = acos(nodes.crd->w(i));
		assert (iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta));
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta));
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
		fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
		assert (false);
		break;
	case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		assert (POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < r_max);
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		assert (iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta));
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta));
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		assert (POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < r_max);;
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
		assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
		#if EMBED_NODES
		assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		assert (iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta));
		#else
		assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
		assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
		assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
		assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
		#endif
		break;
	default:
		fprintf(stderr, "Spacetime parameters not supported!\n");
		assert (false);
	}
}
