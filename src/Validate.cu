/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "Validate.h"

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
		printf("%" PRId64 "\n", edges.future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%" PRId64 "\n", edges.past_edge_row_start[i]);
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
			printf("Pointer: %" PRId64 "\n", (edges.future_edge_row_start[i+next_future_idx] - edges.future_edge_row_start[i]));
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
			printf("Pointer: %" PRId64 "\n", (edges.past_edge_row_start[i+next_past_idx] - edges.past_edge_row_start[i]));
		}
		fflush(stdout);
	}
}

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N, const float &core_edge_fraction)
{
	#if DEBUG
	assert (k_out != NULL);
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);
	assert (N > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif
	
	int core_limit = static_cast<int>(core_edge_fraction * N);
	int idx1, idx2;
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

				if (!adj[idx1].read(idx2) || !adj[idx2].read(idx1))
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
bool linkNodesGPU_v1(Node &nodes, const Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, CaResources * const ca, Stopwatch &sLinkNodesGPU, const bool &link_epso, const bool &has_exact_k, const bool &verbose, const bool &bench)
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

	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
	assert (N > 0);
	assert (k_tar > 0.0f);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	assert (!link_epso);
	#endif

	#if EMBED_NODES
	fprintf(stderr, "linkNodesGPU_v1 not implemented for EMBED_NODES=true.  Find me on line %d in %s.\n", __LINE__, __FILE__);
	#endif

	if (verbose || bench)
		printf_dbg("Using Version 1 (linkNodesGPU).\n");

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
	
	//bool compact = get_curvature(spacetime) & POSITIVE;
	bool compact = spacetime.curvatureIs("Positive");

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
	checkCudaErrors(cuMemAlloc(&d_w, sizeof(float) * N));
	checkCudaErrors(cuMemAlloc(&d_x, sizeof(float) * N));
	checkCudaErrors(cuMemAlloc(&d_y, sizeof(float) * N));
	checkCudaErrors(cuMemAlloc(&d_z, sizeof(float) * N));
	ca->devMemUsed += sizeof(float) * N * 4;

	size_t d_edges_size = pow(2.0, ceil(log2(N * k_tar * (1.0 + edge_buffer) / 2)));
	checkCudaErrors(cuMemAlloc(&d_edges, sizeof(uint64_t) * d_edges_size));
	ca->devMemUsed += sizeof(uint64_t) * d_edges_size;

	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N));
	ca->devMemUsed += sizeof(int) * N;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N));
	ca->devMemUsed += sizeof(int) * N;
	
	checkCudaErrors(cuMemAlloc(&d_g_idx, sizeof(unsigned long long int)));
	ca->devMemUsed += sizeof(unsigned long long int);

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Node Linking", ca->hostMemUsed, ca->devMemUsed, 0);

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_w, nodes.crd->w(), sizeof(float) * N));
	checkCudaErrors(cuMemcpyHtoD(d_x, nodes.crd->x(), sizeof(float) * N));
	checkCudaErrors(cuMemcpyHtoD(d_y, nodes.crd->y(), sizeof(float) * N));
	checkCudaErrors(cuMemcpyHtoD(d_z, nodes.crd->z(), sizeof(float) * N));

	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_edges, 0, d_edges_size << 1));
	checkCudaErrors(cuMemsetD32(d_k_in, 0, N));
	checkCudaErrors(cuMemsetD32(d_k_out, 0, N));
	checkCudaErrors(cuMemsetD32(d_g_idx, 0, 2));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sGPUOverhead);

	//CUDA Grid Specifications
	unsigned int gridx_GAL = static_cast<unsigned int>(ceil(static_cast<float>(N) / 2));
	unsigned int gridy_GAL = static_cast<unsigned int>(ceil((static_cast<float>(N) / 2) / BLOCK_SIZE));
	dim3 blocks_per_grid_GAL(gridx_GAL, gridy_GAL, 1);
	dim3 threads_per_block_GAL(1, BLOCK_SIZE, 1);
	
	stopwatchStart(&sGenAdjList);

	//Execute Kernel
	GenerateAdjacencyLists_v1<<<blocks_per_grid_GAL, threads_per_block_GAL>>>((float*)d_w, (float*)d_x, (float*)d_y, (float*)d_z, (uint64_t*)d_edges, (int*)d_k_in, (int*)d_k_out, (unsigned long long int*)d_g_idx, N >> 1, compact);
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

	ca->devMemUsed -= sizeof(float) * N * 4;

	cuMemFree(d_g_idx);
	d_g_idx = 0;
	ca->devMemUsed -= sizeof(unsigned long long int);

	try {
		if (*g_idx + 1 >= static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2))
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
	unsigned int gridx_res_prop = static_cast<unsigned int>(ceil(static_cast<float>(N) / BLOCK_SIZE));
	dim3 blocks_per_grid_res_prop(gridx_res_prop, 1, 1);

	stopwatchStart(&sProps);

	//Execute Kernel
	ResultingProps<<<gridx_res_prop, threads_per_block>>>((int*)d_k_in, (int*)d_k_out, (int*)d_N_res, (int*)d_N_deg2, N);
	getLastCudaError("Kernel 'NetworkCreator_GPU.ResultingProps' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	stopwatchStop(&sProps);
	stopwatchStart(&sGPUOverhead);

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(nodes.k_in, d_k_in, sizeof(int) * N));
	checkCudaErrors(cuMemcpyDtoH(nodes.k_out, d_k_out, sizeof(int) * N));
	checkCudaErrors(cuMemcpyDtoH(&N_res, d_N_res, sizeof(int)));
	checkCudaErrors(cuMemcpyDtoH(&N_deg2, d_N_deg2, sizeof(int)));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	//Prefix Sum of 'k_in' and 'k_out'
	scan(nodes.k_in, nodes.k_out, edges.past_edge_row_start, edges.future_edge_row_start, N);

	N_res = N - N_res;
	N_deg2 = N - N_deg2;
	k_res = static_cast<float>(static_cast<long double>(*g_idx) * 2 / N_res);

	#if DEBUG
	assert (N_res > 0);
	assert (N_deg2 > 0);
	assert (k_res > 0.0);
	#endif

	//Free Device Memory
	cuMemFree(d_k_in);
	d_k_in = 0;
	ca->devMemUsed -= sizeof(int) * N;

	cuMemFree(d_k_out);
	d_k_out = 0;
	ca->devMemUsed -= sizeof(int) * N;

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
		//printf("\t\tUndirected Links:         %" PRIu64 "\n", *g_idx);
		printf("\t\tUndirected Links:         %llu\n", *g_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", k_res * (N_res / N));
		if (has_exact_k) {
			printf_red();
			printf("\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		}
		printf_std();
		fflush(stdout);
	}
	
	//if(!compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction))
	//	return false;

	//Print Results
	/*if (!printDegrees(nodes, N, "in-degrees_GPU_v1.cset.dbg.dat", "out-degrees_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, *g_idx, "past-edges_GPU_v1.cset.dbg.dat", "future-edges_GPU_v1.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N, "past-edge-pointers_GPU_v1.cset.dbg.dat", "future-edge-pointers_GPU_v1.cset.dbg.dat")) return false;
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

bool generateLists_v1(Node &nodes, uint64_t * const &edges, Bitvector &adj, int64_t * const &g_idx, const Spacetime &spacetime, const int &N, const double &r_max, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &link_epso, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (edges != NULL);
	assert (g_idx != NULL);
	assert (ca != NULL);
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
	assert (N > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (!link_epso);
	if (use_bit) {
		assert (adj.size() >= N);
		assert (core_edge_fraction == 1.0f);
	}
	#endif

	if (verbose || bench)
		printf_dbg("Using Version 1 (generateLists).\n");

	//Temporary Buffers
	CUdeviceptr d_w0, d_x0, d_y0, d_z0;
	CUdeviceptr d_w1, d_x1, d_y1, d_z1;
	CUdeviceptr d_k_in, d_k_out;
	CUdeviceptr d_edges;

	int *h_k_in;
	int *h_k_out;
	bool *h_edges;

	unsigned int core_limit = static_cast<unsigned int>(core_edge_fraction * N);
	unsigned int i, j;
	bool diag;

	unsigned int stdim = atoi(Spacetime::stdims[spacetime.get_stdim()]);
	bool compact = spacetime.curvatureIs("Positive");

	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N) / (BLOCK_SIZE * group_size)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

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
	checkCudaErrors(cuMemAlloc(&d_x0, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y0, sizeof(float) * mthread_size));
	if (stdim == 4) {
		checkCudaErrors(cuMemAlloc(&d_w0, sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z0, sizeof(float) * mthread_size));
	}
	ca->devMemUsed += sizeof(float) * mthread_size * stdim;

	checkCudaErrors(cuMemAlloc(&d_x1, sizeof(float) * mthread_size));
	checkCudaErrors(cuMemAlloc(&d_y1, sizeof(float) * mthread_size));
	if (stdim == 4) {
		checkCudaErrors(cuMemAlloc(&d_w1, sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z1, sizeof(float) * mthread_size));
	}
	ca->devMemUsed += sizeof(float) * mthread_size * stdim;

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

	size_t final_size = N - mthread_size * (group_size - 1);
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
			checkCudaErrors(cuMemcpyHtoD(d_x0, nodes.crd->x() + i * mthread_size, sizeof(float) * size0));
			checkCudaErrors(cuMemcpyHtoD(d_y0, nodes.crd->y() + i * mthread_size, sizeof(float) * size0));
			if (stdim == 4) {
				checkCudaErrors(cuMemcpyHtoD(d_w0, nodes.crd->w() + i * mthread_size, sizeof(float) * size0));
				checkCudaErrors(cuMemcpyHtoD(d_z0, nodes.crd->z() + i * mthread_size, sizeof(float) * size0));
			}

			checkCudaErrors(cuMemcpyHtoD(d_x1, nodes.crd->x() + j * mthread_size, sizeof(float) * size1));
			checkCudaErrors(cuMemcpyHtoD(d_y1, nodes.crd->y() + j * mthread_size, sizeof(float) * size1));
			if (stdim == 4) {
				checkCudaErrors(cuMemcpyHtoD(d_w1, nodes.crd->w() + j * mthread_size, sizeof(float) * size1));
				checkCudaErrors(cuMemcpyHtoD(d_z1, nodes.crd->z() + j * mthread_size, sizeof(float) * size1));
			}

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());

			//Execute Kernel
			int flags = ((int)diag << 4) | ((int)compact << 3) | stdim;
			switch (flags) {
			case 2:
				GenerateAdjacencyLists_v2<false, false, false, false, 2><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 4:
				GenerateAdjacencyLists_v2<false, false, false, false, 4><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 10:
				GenerateAdjacencyLists_v2<true, false, false, false, 2><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 12:
				GenerateAdjacencyLists_v2<true, false, false, false, 4><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 18:
				GenerateAdjacencyLists_v2<false, false, false, true, 2><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 20:
				GenerateAdjacencyLists_v2<false, false, false, true, 4><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 26:
				GenerateAdjacencyLists_v2<true, false, false, true, 2><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			case 28:
				GenerateAdjacencyLists_v2<true, false, false, true, 4><<<blocks_per_grid, threads_per_block>>>((float*)d_w0, (float*)d_x0, (float*)d_y0, (float*)d_z0, (float*)d_w1, (float*)d_x1, (float*)d_y1, (float*)d_z1, (int*)d_k_in, (int*)d_k_out, (bool*)d_edges, size0, size1, r_max);
				break;
			default:
				fprintf(stderr, "Invalid flag value: %d\n", flags);
				return false;
			}

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
			readEdges(edges, h_edges, adj, g_idx, core_limit, core_limit, d_edges_size, mthread_size, size0, size1, i, j, use_bit, false);

			//Clear Device Memory
			checkCudaErrors(cuMemsetD32(d_k_in, 0, mthread_size));
			checkCudaErrors(cuMemsetD32(d_k_out, 0, mthread_size));
			checkCudaErrors(cuMemsetD8(d_edges, 0, m_edges_size));

			//Synchronize
			checkCudaErrors(cuCtxSynchronize());			
		}
	}

	cuMemFree(d_x0);
	d_x0 = 0;

	cuMemFree(d_y0);
	d_y0 = 0;

	if (stdim == 4) {
		cuMemFree(d_w0);
		d_w0 = 0;

		cuMemFree(d_z0);
		d_z0 = 0;
	}

	ca->devMemUsed -= sizeof(float) * mthread_size * stdim;

	cuMemFree(d_x1);
	d_x1 = 0;

	cuMemFree(d_y1);
	d_y1 = 0;

	if (stdim == 4) {
		cuMemFree(d_w1);
		d_w1 = 0;

		cuMemFree(d_z1);
		d_z1 = 0;
	}

	ca->devMemUsed -= sizeof(float) * mthread_size * stdim;

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

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(Node &nodes, const Spacetime &spacetime, const int num_vals, const char *filename, const char *coord)
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
		snprintf(&full_name[3], 77, "%s", spacetime.toHexString());
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
				if (spacetime.stdimIs("2") || spacetime.stdimIs("3"))
					outputStream << nodes.crd->x(i) << std::endl;
				else if (spacetime.stdimIs("4"))
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
			else if (!strcmp(coord, "w"))
				outputStream << nodes.crd->w(i) << std::endl;
			else if (!strcmp(coord, "x"))
				outputStream << nodes.crd->x(i) << std::endl;
			else if (!strcmp(coord, "y"))
				outputStream << nodes.crd->y(i) << std::endl;
			else if (!strcmp(coord, "z"))
				outputStream << nodes.crd->z(i) << std::endl;			
			else if (!strcmp(coord, "u")) {
				if (spacetime.stdimIs("2") || spacetime.stdimIs("3"))
					outputStream << (nodes.crd->x(i) + nodes.crd->y(i)) / sqrt(2.0) << std::endl;
				else if (spacetime.stdimIs("4"))
					outputStream << (nodes.crd->w(i) + nodes.crd->x(i)) / sqrt(2.0) << std::endl;
				else if (spacetime.stdimIs("5"))
					outputStream << (nodes.crd->v(i) + nodes.crd->w(i)) / sqrt(2.0) << std::endl;
			} else if (!strcmp(coord, "v")) {
				if (spacetime.stdimIs("2") || spacetime.stdimIs("3"))
					outputStream << (nodes.crd->x(i) - nodes.crd->y(i)) / sqrt(2.0) << std::endl;
				else if (spacetime.stdimIs("4"))
					outputStream << (nodes.crd->w(i) - nodes.crd->x(i)) / sqrt(2.0) << std::endl;
				else if (spacetime.stdimIs("5"))
					outputStream << (nodes.crd->v(i) - nodes.crd->w(i)) / sqrt(2.0) << std::endl;
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

bool printAdjMatrix(const Bitvector &adj, const int N, const char *filename, const int num_mpi_threads, const int rank)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (N > 0);
	assert (filename != NULL);
	assert (num_mpi_threads >= 1);
	assert (rank >= 0);
	#endif

	try {
		std::ofstream os;
		for (int i = 0; i < num_mpi_threads; i++) {
			#ifdef MPI_ENABLED
			MPI_Barrier(MPI_COMM_WORLD);
			#endif

			if (i == rank) {
				os.open(filename, std::ios::app);
				if (!os.is_open())
					throw CausetException("Failed to open adjacency matrix file in 'printEdgeLists' function!\n");

				for (int j = 0; j < N / num_mpi_threads; j++) {
					/*if (j == N / (num_mpi_threads << 1)) {
						for (int k = 0; k < N; k++)
							printf("--");
						printf("\n");
					}*/
					for (int k = 0; k < N; k++) {
						os << adj[j].read(k);
						//printf("%" PRIu64 " ", adj[j].read(k));
					}
					os << "\n";
					//printf(" [%d]\n", rank);
				}

				/*if (i != num_mpi_threads - 1) {
					for (int j = 0; j < N; j++)
						printf("==");
					printf("\n");
				}
				fflush(stdout);*/
				sleep(1);

				os.flush();
				os.close();
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

//Node Traversal Algorithm
//Not accelerated with OpenMP
//Uses geodesic distances
bool traversePath_v1(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const bool &strict_routing, int source, int dest, int &nsteps, bool &success, bool &success2, bool &past_horizon)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic"));

	if (spacetime.manifoldIs("Hyperbolic"))
		assert (spacetime.stdimIs("2"));

	if (spacetime.stdimIs("2")) {
		assert (nodes.crd->getDim() == 2);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Hyperbolic"));
	} else if (spacetime.stdimIs("4")) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
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

	assert (N > 0);
	if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		assert (a > 0.0);
		if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (spacetime.curvatureIs("Positive")) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (spacetime.curvatureIs("Flat")) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (!strict_routing);
	assert (source >= 0 && source < N);
	assert (dest >= 0 && dest < N);
	#endif

	bool TRAV_DEBUG = false;

	float min_dist = 0.0f;
	int loc = source;
	int idx_a = source;
	int idx_b = dest;
	nsteps = 0;

	float dist;
	int next;
	int m;

	//Check if source and destination can be connected by any geodesic
	if (spacetime.manifoldIs("De_Sitter")) {
		if (spacetime.curvatureIs("Positive"))
			dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
		else if (spacetime.curvatureIs("Flat"))
			dist = distanceDeSitterFlat(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
		else
			return false;
	} else if (spacetime.manifoldIs("FLRW"))
			dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
	else if (spacetime.manifoldIs("Dust"))
		dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
	else if (spacetime.manifoldIs("Hyperbolic"))
		dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
	else
		return false;

	if (dist == -1)
		return false;

	if (dist + 1.0 > INF) {
		past_horizon = true;
		success2 = false;
	}

	if (TRAV_DEBUG) {
		printf_cyan();
		printf("Beginning at [%d : %.4f].\tLooking for [%d : %.4f].\n", source, nodes.id.tau[source], dest, nodes.id.tau[dest]);
		if (past_horizon) {
			printf_red();
			printf("Past horizon at start.\n");
		}
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
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_a, nodes.id.tau[idx_a], nodes.crd->z(idx_a));
					printf_cyan();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				nsteps++;
				success = true;
				return true;
			}

			//(B) If the current location's past neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, idx_a, idx_b)) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_a, nodes.id.tau[idx_a], nodes.crd->z(idx_a));
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_b, nodes.id.tau[idx_b], nodes.crd->z(idx_b));
					printf_cyan();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				nsteps += 2;
				success = true;
				return true;
			}

			//(C) Otherwise find the past neighbor closest to the destination
			if (spacetime.manifoldIs("De_Sitter")) {
				if (spacetime.curvatureIs("Positive"))
					dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
				else if (spacetime.curvatureIs("Flat"))
					dist = distanceDeSitterFlat(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
				else
					return false;
			} else if (spacetime.manifoldIs("FLRW"))
					dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (spacetime.manifoldIs("Dust"))
				dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (spacetime.manifoldIs("Hyperbolic"))
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
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_a, nodes.id.tau[idx_a], nodes.crd->z(idx_a));
					printf_cyan();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				nsteps++;
				success = true;
				return true;
			}

			//(E) If the current location's future neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, idx_a, idx_b)) {
				if (TRAV_DEBUG) {
					printf_cyan();
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_a, nodes.id.tau[idx_a], nodes.crd->z(idx_a));
					printf("Moving to [%d : (%.4f, %.4f)].\n", idx_b, nodes.id.tau[idx_b], nodes.crd->z(idx_b));
					printf_cyan();
					printf("SUCCESS\n");
					printf_std();
					fflush(stdout);
				}
				nsteps += 2;
				success = true;
				return true;
			}

			//(F) Otherwise find the future neighbor closest to the destination
			if (spacetime.manifoldIs("De_Sitter")) {
				if (spacetime.curvatureIs("Positive"))
					dist = fabs(distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], spacetime, a, alpha));
				else if (spacetime.curvatureIs("Flat"))
					dist = distanceDeSitterFlat(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
				else
					return false;
			} else if (spacetime.manifoldIs("FLRW"))
					dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (spacetime.manifoldIs("Dust"))
				dist = distanceDust(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b);
			else if (spacetime.manifoldIs("Hyperbolic"))
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
			printf("Moving to [%d : (%.4f, %.4f)].\n", next, nodes.id.tau[next], nodes.crd->z(next));
			printf_std();
			fflush(stdout);
		}

		if (!used[next] && min_dist + 1.0 < INF) {
			loc = next;
			nsteps++;
		} else {
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
	success2 = false;
	return true;
}

//Measure Causal Set Action
//O(N*k^2*ln(k)) Efficiency (Linked)
//O(N^2*k) Efficiency (No Links)
bool measureAction_v1(uint64_t *& cardinalities, float &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int &N, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool no_pos, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	if (!no_pos)
		assert (!nodes.crd->isNull());

	if (!no_pos) {
		if (spacetime.stdimIs("2"))
			assert (nodes.crd->getDim() == 2);
		else if (spacetime.stdimIs("4")) {
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
		
	assert (N > 0);
	assert (max_cardinality > 0);
	assert (a > 0.0);
	if (spacetime.curvatureIs("Positive")) {
		assert (zeta > 0.0);
		assert (zeta < HALF_PI);
	} else if (spacetime.curvatureIs("Flat")) {
		assert (zeta > HALF_PI);
		assert (zeta1 > HALF_PI);
		assert (zeta > zeta1);
	} 
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif

	if (verbose || bench)
		printf_dbg("Using Version 1 (measureAction).\n");

	double lk = 2.0;
	bool smeared = max_cardinality == N;
	int core_limit = static_cast<int>(core_edge_fraction * N);
	int elements;
	int64_t fstart, pstart;
	int i, j, k;
	bool too_many;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality data
	try {
		cardinalities = (uint64_t*)malloc(sizeof(uint64_t) * max_cardinality);
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(uint64_t) * max_cardinality);
		ca->hostMemUsed += sizeof(uint64_t) * max_cardinality * omp_get_max_threads();
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, 0);

	cardinalities[0] = N;

	if (max_cardinality == 1)
		goto ActionExit;

	too_many = false;
	for (i = 0; i < N - 1; i++) {
		for (j = i + 1; j < N; j++) {
			elements = 0;
			if (!use_bit && (link || relink)) {
				if (!nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, i, j))
					continue;

				//These indicate corrupted data
				#if DEBUG
				assert (!(edges.past_edge_row_start[j] == -1 && nodes.k_in[j] > 0));
				assert (!(edges.past_edge_row_start[j] != -1 && nodes.k_in[j] == 0));
				assert (!(edges.future_edge_row_start[i] == -1 && nodes.k_out[i] > 0));
				assert (!(edges.future_edge_row_start[i] != -1 && nodes.k_out[i] == 0));
				#endif

				if (core_limit == N) {
					uint64_t col0 = static_cast<uint64_t>(i) * core_limit;
					uint64_t col1 = static_cast<uint64_t>(j) * core_limit;
					for (k = i + 1; k < j; k++)
						elements += (int)(adj[col0].read(k) & adj[col1].read(k));
					if (elements >= max_cardinality - 1)
						too_many = true;
				} else {
					pstart = edges.past_edge_row_start[j];
					fstart = edges.future_edge_row_start[i];
					//printf("\nLooking at %d future neighbors of [node %d] and %d past neighbors of [node %d].\n", nodes.k_out[i], i, nodes.k_in[j], j);
					causet_intersection_v2(elements, edges.past_edges, edges.future_edges, nodes.k_in[j], nodes.k_out[i], max_cardinality, pstart, fstart, too_many);
				}
			} else {
				if (!nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j, NULL))
					continue;

				for (k = i + 1; k < j; k++) {
					if (nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, alpha, r_max, i, k, NULL) && nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, k, j, NULL))
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

	action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), lk, smeared);
	assert (action == action);
	
	ActionExit:
	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf("\tCalculated Action.\n");
		printf("\t\tTerms Used: %d\n", max_cardinality);
		printf_cyan();
		printf("\t\tCausal Set Action: %f\n", action);
		if (max_cardinality < 10)
			for (i = 0; i < max_cardinality; i++)
				printf("\t\t\tN%d: %" PRIu64 "\n", i, cardinalities[i]);
		printf_std();
		fflush(stdout);
	} 

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Measure Causal Set Action
//Algorithm has been parallelized on the CPU
bool measureAction_v2(uint64_t *& cardinalities, float &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	if (!no_pos)
		assert (!nodes.crd->isNull());

	if (!no_pos) {
		if (spacetime.stdimIs("2"))
			assert (nodes.crd->getDim() == 2);
		else if (spacetime.stdimIs("4")) {
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

	assert (N > 0);
	assert (k_tar > 0.0f);
	assert (max_cardinality > 0);
	assert (a > 0.0);
	if (spacetime.curvatureIs("Positive")) {
		assert (zeta > 0.0);
		assert (zeta < HALF_PI);
	} else if (spacetime.curvatureIs("Flat")) {
		assert (zeta > HALF_PI);
		assert (zeta1 > HALF_PI);
		assert (zeta > zeta1);
	} 
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer > 0.0f);
	#endif

	if (verbose || bench)
		printf_dbg("Using Version 2 (measureAction).\n");

	int n = N + N % 2;
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	uint64_t start = 0;
	uint64_t finish = npairs;
	int core_limit = static_cast<int>(core_edge_fraction * N);
	int rank = cmpi.rank;
	bool smeared = (max_cardinality == N);
	double lk = 2.0;

	#ifdef MPI_ENABLED
	assert (false);	//MPI code not maintained
	//uint64_t core_edges_size = static_cast<uint64_t>(POW2(core_edge_fraction * N, EXACT));
	//uint64_t edges_size = static_cast<uint64_t>(N) * k_tar * (1.0 + edge_buffer) / 2;
	#endif

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements
	try {
		cardinalities = (uint64_t*)malloc(sizeof(uint64_t) * max_cardinality * omp_get_max_threads());
		if (cardinalities == NULL) {
			cmpi.fail = 1;
			goto ActPoint;
		}
		memset(cardinalities, 0, sizeof(uint64_t) * max_cardinality * omp_get_max_threads());
		ca->hostMemUsed += sizeof(uint64_t) * max_cardinality * omp_get_max_threads();

		ActPoint:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, rank);

	//The first element will be N
	cardinalities[0] = N;

	#ifdef MPI_ENABLED
	/*MPI_Barrier(MPI_COMM_WORLD);
	if (!use_bit && (link || relink)) {
		MPI_Bcast(nodes.k_in, N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(nodes.k_out, N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(edges.past_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(edges.future_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(edges.past_edge_row_start, 2 * N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(edges.future_edge_row_start, 2 * N, MPI_INT, 0, MPI_COMM_WORLD);
		//MPI_Bcast(adj, core_edges_size, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(nodes.crd->x(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(nodes.crd->y(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (get_stdim(spacetime) == 4) {
			MPI_Bcast(nodes.crd->w(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Bcast(nodes.crd->z(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
	}

	uint64_t mpi_chunk = npairs / cmpi.num_mpi_threads;
	start = rank * mpi_chunk;
	finish = start + mpi_chunk;*/
	#endif

	if (max_cardinality == 1)
		goto ActionExit;

	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 4)
	#endif
	for (uint64_t v = start; v < finish; v++) {
		//Choose a pair
		int i = static_cast<int>(v / (n - 1));
		int j = static_cast<int>(v % (n - 1) + 1);
		int do_map = i >= j;

		if (j < n >> 1) {
			i = i + do_map * ((((n >> 1) - i) << 1) - 1);
			j = j + do_map * (((n >> 1) - j) << 1);
		}

		if (j == N) continue;

		int elements = 0;
		bool too_many = false;

		if (!use_bit && (link || relink)) {
			//If the nodes have been linked, use edge lists / adjacency matrix
			if (!nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, i, j))
				continue;

			#if DEBUG
			assert (!(edges.past_edge_row_start[j] == -1 && nodes.k_in[j] > 0));
			assert (!(edges.past_edge_row_start[j] != -1 && nodes.k_in[j] == 0));
			assert (!(edges.future_edge_row_start[i] == -1 && nodes.k_out[i] > 0));
			assert (!(edges.future_edge_row_start[i] != -1 && nodes.k_out[i] == 0));
			#endif

			if (core_limit == N) {
				uint64_t col0 = static_cast<uint64_t>(i) * core_limit;
				uint64_t col1 = static_cast<uint64_t>(j) * core_limit;

				for (int k = i + 1; k < j; k++)
					elements += (int)(adj[col0].read(k) & adj[col1].read(k));

				if (elements >= max_cardinality - 1)
					too_many = true;
			} else {
				//Index of first past neighbor of the 'future element j'
				int64_t pstart = edges.past_edge_row_start[j];
				//Index of first future neighbor of the 'past element i'
				int64_t fstart = edges.future_edge_row_start[i];

				//Intersection of edge lists
				causet_intersection_v2(elements, edges.past_edges, edges.future_edges, nodes.k_in[j], nodes.k_out[i], max_cardinality, pstart, fstart, too_many);
			}
		} else {
			//If nodes have not been linked, do each comparison
			if (!nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j, NULL))
				continue;

			for (int k = i + 1; k < j; k++) {
				if (nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, i, k, NULL) && nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, k, j, NULL))
					elements++;

				if (elements >= max_cardinality - 1) {
					too_many = true;
					break;
				}
			}
		}

		if (!too_many)
			cardinalities[omp_get_thread_num()*max_cardinality+elements+1]++;
	}

	//Reduction used when OpenMP has been used
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < max_cardinality; j++)
			cardinalities[j] += cardinalities[i*max_cardinality+j];

	#ifdef MPI_ENABLED
	/*MPI_Barrier(MPI_COMM_WORLD);
	if (!rank)
		MPI_Reduce(MPI_IN_PLACE, cardinalities, max_cardinality, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(cardinalities, NULL, max_cardinality, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);*/
	#endif

	if (max_cardinality < 5)
		goto ActionExit;

	action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), lk, smeared);
	assert (action == action);

	ActionExit:
	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Action.\n");
		printf_mpi(rank, "\t\tTerms Used: %d\n", max_cardinality);
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tCausal Set Action: %f\n", action);
		if (max_cardinality < 10)
			for (int i = 0; i < max_cardinality; i++)
				printf_mpi(rank, "\t\t\tN%d: %d\n", i, cardinalities[i]);
		if (!rank) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool validateCoordinates(const Node &nodes, const Spacetime &spacetime, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const int &i)
{
	#if EMBED_NODES
	float tol = 1.0e-4;
	double r;
	#endif
	if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Positive", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(nodes.crd->y(i) > 0.0 && nodes.crd->y(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_T1", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(fabs(nodes.crd->y(i)) < eta0)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_S1", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < r_max)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_TS", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < 1.0)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_N3", "Flat", "None")) {
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < 1.5)) return false;
		if (!(fabs(nodes.crd->y(i)) < 0.5)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Diamond", "Flat", "None")) {
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < eta0)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_T", "Flat", "Temporal")) {
		#if SPECIAL_SAUCER
		if (!(nodes.crd->x(i) > -1.0 && nodes.crd->x(i) < 1.0)) return false;
		if (!(nodes.crd->y(i) > -1.5 && nodes.crd->y(i) < 1.5)) return false;
		#else
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
		#endif
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_S", "Flat", "Temporal")) {
		#if SPECIAL_SAUCER
		if (!(nodes.crd->x(i) > -1.5 && nodes.crd->x(i) < 1.5)) return false;
		if (!(nodes.crd->y(i) > -1.0 && nodes.crd->y(i) < 1.0)) return false;
		#else
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
		#endif
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_T", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(nodes.crd->y(i) > 0.0 && nodes.crd->y(i) < r_max)) return false;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_S", "Flat", "None")) {
		if (!(nodes.crd->x(i) > 0.0 && nodes.crd->x(i) < eta0)) return false;
		if (!(fabs(nodes.crd->y(i)) < r_max)) return false;
	} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "None")) {
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < eta0)) return false;
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol)) return false;
		#else
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < HALF_PI - zeta)) return false;
		if (!(nodes.id.tau[i] > -tau0 && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol)) return false;
		#else
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Negative", "None")) {
		if (!(nodes.id.tau[i] < tau0)) return false;
		if (!(nodes.crd->x(i) < eta0)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] < tau0)) return false;
		if (!(nodes.crd->x(i) > HALF_PI - zeta && nodes.crd->x(i) < HALF_PI - zeta1)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), HALF_PI - zeta, HALF_PI - zeta1)) return false;
	} else if (spacetime.spacetimeIs("2", "Dust", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] < tau0)) return false;
		if (!(nodes.crd->x(i) > 0.0 && nodes.crd->x(i) < eta0)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0)) return false;
	} else if (spacetime.spacetimeIs("2", "Hyperbolic", "Slab", "Positive", "None")) {
		if (!(nodes.id.tau[i] > 0.0 && nodes.id.tau[i] <= tau0)) return false;
		if (!(nodes.crd->x(i) > 0.0 && nodes.crd->x(i) <= r_max)) return false;
		if (!(nodes.crd->y(i) > 0.0 && nodes.crd->y(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("2", "Polycone", "Slab", "Positive", "None")) {
		if (!(nodes.id.tau[i] > 0.0 && nodes.id.tau[i] <= tau0)) return false;
		if (!(nodes.crd->x(i) <= eta0)) return false;
		if (!(nodes.crd->y(i) > 0.0 && nodes.crd->y(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("3", "Minkowski", "Slab", "Flat", "Temporal")) {
		if (!(fabs(nodes.crd->x(i)) < eta0)) return false;
		if (!(nodes.crd->y(i) < r_max)) return false;
		if (!(nodes.crd->z(i) >= 0.0 && nodes.crd->z(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("3", "Minkowski", "Diamond", "Flat", "None")) {
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < eta0)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0)) return false;
		if (!(nodes.crd->z(i) > 0.0 && nodes.crd->z(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("3", "Minkowski", "Cube", "Flat", "None")) {
		if (!(nodes.crd->x(i) >= 0.0 && nodes.crd->x(i) <= eta0)) return false;
		if (!(nodes.crd->y(i) >= 0.0 && nodes.crd->y(i) <= r_max)) return false;
		if (!(nodes.crd->z(i) >= 0.0 && nodes.crd->z(i) <= r_max)) return false;
	} else if (spacetime.spacetimeIs("3", "De_Sitter", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] < tau0)) return false;
		if (!(nodes.crd->x(i) > -1.0f && nodes.crd->x(i) < HALF_PI - zeta1)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), -1.0, HALF_PI - zeta1)) return false;
		if (!(nodes.crd->z(i) > 0.0 && nodes.crd->z(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("3", "Dust", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] < tau0)) return false;
		if (!(nodes.crd->x(i) > 0.0 && nodes.crd->x(i) < eta0)) return false;
		if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0)) return false;
		if (!(nodes.crd->z(i) > 0.0 && nodes.crd->z(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("4", "Minkowski", "Diamond", "Flat", "None")) {
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < eta0)) return false;
		if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, eta0)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > HALF_PI - zeta && nodes.crd->v(i) < HALF_PI - zeta1)) return false;
		if (!(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < POW2(r_max, EXACT))) return false;
		#else
		if (!(nodes.crd->w(i) > HALF_PI - zeta && nodes.crd->w(i) < HALF_PI - zeta1)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		if (!(fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "Temporal")) {
		if (!(nodes.id.tau[i] > -tau0 && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(fabs(nodes.crd->v(i)) < HALF_PI - zeta)) return false;
		if (!(fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol)) return false;
		#else
		if (!(fabs(nodes.crd->v(i)) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		if (!iad(nodes.crd->v(i), r, HALF_PI - zeta, HALF_PI - zeta1)) return false;
		#else
		if (!(nodes.crd->w(i) > HALF_PI - zeta && nodes.crd->w(i) < HALF_PI - zeta1)) return false;
		if (!iad(nodes.crd->w(i), nodes.crd->x(i), HALF_PI - zeta, HALF_PI - zeta1)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Positive", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		r = acos(nodes.crd->w(i));
		if (!iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "Dust", "Slab", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		if (!(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < r_max)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "Dust", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		if (!iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		if (!(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) < r_max)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Positive", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		if (!(fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol)) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("4", "FLRW", "Diamond", "Flat", "None")) {
		if (!(nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0)) return false;
		#if EMBED_NODES
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta)) return false;
		r = sqrt(POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT));
		if (!(iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta))) return false;
		#else
		if (!(nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
		#endif
	} else if (spacetime.spacetimeIs("5", "Minkowski", "Diamond", "Flat", "None")) {
		if (!(nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < eta0)) return false;
		if (!iad(nodes.crd->v(i), nodes.crd->w(i), 0.0, eta0)) return false;
		if (!(nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI)) return false;
		if (!(nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI)) return false;
		if (!(nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI)) return false;
	} else {
		fprintf(stderr, "Spacetime parameters not supported!\n");
		assert (false);
	}

	return true;
}

void printDot(Bitvector &adj, const int * const k_out, int N, const char *filename)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (k_out != NULL);
	assert (N > 0);
	assert (filename != NULL);
	#endif

	std::ofstream data;

	try {
		data.open(filename);
		if (!data.is_open())
			throw CausetException("Failed to open dot file!\n");

		data << "digraph \"causet\" {\n";
		data << "rankdir=BT; concentrate=true;\n";
		for (int i = 0; i < N; i++) {
			data << i << " [shape=plaintext];\n";
			for (int j = i + 1; j < N; j++)
				if (adj[i].read(j))
					data << i << "->" << j << "; ";
			if (!!k_out[i])
				data << "\n";
		}
		data << "}\n";
		data.flush();
		data.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
	}
}
