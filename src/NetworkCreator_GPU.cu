#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists_v2(float *w0, float *x0, float *y0, float *z0, float *w1, float *x1, float *y1, float *z1, int *k_in, int *k_out, bool *edges, int diag, bool compact)
{
	__shared__ float shr_w1[THREAD_SIZE];
	__shared__ float shr_x1[THREAD_SIZE];
	__shared__ float shr_y1[THREAD_SIZE];
	__shared__ float shr_z1[THREAD_SIZE];
	__shared__ int n[BLOCK_SIZE][THREAD_SIZE];

	float4 n0, n1;

	unsigned int tid = threadIdx.y;
	//Index 'i' marks the row and 'j' marks the column
	unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int j = blockIdx.x;
	unsigned int k;

	//Each thread compares 1 node in 'nodes0' to 'THREAD_SIZE' nodes in 'nodes1'
	if (!tid) {
		for (k = 0; k < THREAD_SIZE; k++) {
			shr_w1[k] = w1[j*THREAD_SIZE+k];
			shr_x1[k] = x1[j*THREAD_SIZE+k];
			shr_y1[k] = y1[j*THREAD_SIZE+k];
			shr_z1[k] = z1[j*THREAD_SIZE+k];
		}
	}
	__syncthreads();

	float dt[THREAD_SIZE];
	float dx[THREAD_SIZE];

	for (k = 0; k < THREAD_SIZE; k++) {
		if (!diag || i < j * THREAD_SIZE + k) {
			//Identify nodes to compare
			n0.w = w0[i];
			n0.x = x0[i];
			n0.y = y0[i];
			n0.z = z0[i];

			n1.w = shr_w1[k];
			n1.x = shr_x1[k];
			n1.y = shr_y1[k];
			n1.z = shr_z1[k];

			//BEGIN COMPACT EQUATIONS (Completed)

			//Identify spacetime interval
			dt[k] = n1.w - n0.w;

			if (compact)
				dx[k] = acosf(sphProduct_GPU(n0, n1));
			else
				dx[k] = sqrtf(flatProduct_GPU(n0, n1));

			//END COMPACT EQUATIONS
		}
	}

	bool edge[THREAD_SIZE];
	int out = 0;
	for (k = 0; k < THREAD_SIZE; k++) {
		//Mark if edge is present (register memory)
		edge[k] = (!diag || i < j * THREAD_SIZE + k) && dx[k] < dt[k];
		//Copy to shared memory to prepare for reduction
		n[tid][k] = (int)edge[k];
		//Identify number of out-degrees found by a single thread
		out += (int)edge[k];
	}
	__syncthreads();

	//Reduction algorithm (used to optimize atomic operations below)
	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
		if (!(tid % (stride << 1)))
			for (k = 0; k < THREAD_SIZE; k++)
				n[tid][k] += n[tid+stride][k];
		__syncthreads();
	}

	//Global Memory Operations

	//Write edges to global memory
	for (k = 0; k < THREAD_SIZE; k++)
		if (!diag || i < j * THREAD_SIZE + k)
			edges[(i*THREAD_SIZE*gridDim.x)+(j*THREAD_SIZE)+k] = edge[k];

	//Write out-degrees
	atomicAdd(&k_out[i], out);

	//Wrtie in-degrees
	if (!tid)
		for (k = 0; k < THREAD_SIZE; k++)
			if (!diag || i < j * THREAD_SIZE + k)
				atomicAdd(&k_in[j*THREAD_SIZE+k], n[0][k]);
}

__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, int *g_idx, int width, bool compact)
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

		//BEGIN COMPACT EQUATIONS (Completed)

		//Calculate dx
		if (compact) {
			dx_ab = acosf(sphProduct_GPU(node0_ab, node1_ab));
			dx_c = acosf(sphProduct_GPU(node0_c, node1_c));
		} else {
			dx_ab = sqrtf(flatProduct_GPU(node0_ab, node1_ab));
			dx_c = sqrtf(flatProduct_GPU(node0_c, node1_c));
		}

		//END COMPACT EQUATIONS
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

bool linkNodesGPU_v2(Node &nodes, const Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose, const bool &bench)
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
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (edge_buffer >= 0);
	}

	Stopwatch sGenAdjList = Stopwatch();
	Stopwatch sDecodeLists = Stopwatch();
	Stopwatch sScanLists = Stopwatch();
	Stopwatch sProps = Stopwatch();

	CUdeviceptr d_k_in, d_k_out;
	uint64_t *h_edges;
	int *g_idx;

	size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar / 2 + edge_buffer)));

	stopwatchStart(&sLinkNodesGPU);

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

	stopwatchStart(&sGenAdjList);
	if (GEN_ADJ_LISTS_GPU_V2) {
		if (!generateLists_v2(nodes, h_edges, g_idx, N_tar, d_edges_size, ctx, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, compact, verbose))
			return false;
	} else {
		if (!generateLists_v1(nodes, h_edges, g_idx, N_tar, d_edges_size, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, compact, verbose))
			return false;
	}
	stopwatchStop(&sGenAdjList);

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

	//Decode Adjacency Lists
	stopwatchStart(&sDecodeLists);
	if (DECODE_LISTS_GPU_V2) {
		if (!decodeLists_v2(edges, h_edges, g_idx, d_edges_size, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
			return false;
	} else {
		if (!decodeLists_v1(edges, h_edges, g_idx, d_edges_size, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
			return false;
	}
	stopwatchStop(&sDecodeLists);

	//Free Host Memory
	free(h_edges);
	h_edges = NULL;
	hostMemUsed -= sizeof(uint64_t) * d_edges_size;
	
	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N_tar));
	devMemUsed += sizeof(int) * N_tar;

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_k_in, nodes.k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_k_out, nodes.k_out, sizeof(int) * N_tar));

	//Parallel Prefix Scan of Degrees
	stopwatchStart(&sScanLists);
	if (!scanLists(edges, d_k_in, d_k_out, N_tar, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
		return false;
	stopwatchStop(&sScanLists);

	//Identify Resulting Network Properties
	stopwatchStart(&sProps);
	if (!identifyListProperties(nodes, d_k_in, d_k_out, g_idx, N_tar, N_res, N_deg2, k_res, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, verbose))
		return false;
	stopwatchStop(&sProps);	

	//Free Device Memory
	cuMemFree(d_k_in);
	d_k_in = NULL;
	devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_k_out);
	d_k_out = NULL;
	devMemUsed -= sizeof(int) * N_tar;

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
	/*if (!printDegrees(nodes, N_tar, "in-degrees_GPU_v2.cset.dbg.dat", "out-degrees_GPU_v2.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, *g_idx, "past-edges_GPU_v2.cset.dbg.dat", "future-edges_GPU_v2.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_GPU_v2.cset.dbg.dat", "future-edge-pointers_GPU_v2.cset.dbg.dat")) return false;
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
		printf("\t\t\tAdjacency List Function Time: %5.6f sec\n", sGenAdjList.elapsedTime);
		printf("\t\t\tDecode Lists Function Time: %5.6f sec\n", sDecodeLists.elapsedTime);
		printf("\t\t\tScan Lists Function Time: %5.6f sec\n", sScanLists.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Uses multiple buffers and asynchronous operations
bool generateLists_v2(Node &nodes, uint64_t * const &edges, int * const &g_idx, const int &N_tar, const size_t &d_edges_size, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose)
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
		assert (g_idx != NULL);

		assert (N_tar > 0);
	}

	//CUDA Streams
	CUstream stream[NBUFFERS];

	//Arrays of Buffers
	int *h_k_in[NBUFFERS];
	int *h_k_out[NBUFFERS];
	bool *h_edges[NBUFFERS];

	CUdeviceptr d_w0[NBUFFERS];
	CUdeviceptr d_x0[NBUFFERS];
	CUdeviceptr d_y0[NBUFFERS];
	CUdeviceptr d_z0[NBUFFERS];

	CUdeviceptr d_w1[NBUFFERS];
	CUdeviceptr d_x1[NBUFFERS];
	CUdeviceptr d_y1[NBUFFERS];
	CUdeviceptr d_z1[NBUFFERS];

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

		checkCudaErrors(cuMemAlloc(&d_w0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_x0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_y0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z0[i], sizeof(float) * mthread_size));
		devMemUsed += sizeof(float) * mthread_size * 4;

		checkCudaErrors(cuMemAlloc(&d_w1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_x1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_y1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z1[i], sizeof(float) * mthread_size));
		devMemUsed += sizeof(float) * mthread_size * 4;

		checkCudaErrors(cuMemAlloc(&d_k_in[i], sizeof(int) * mthread_size));
		devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_k_out[i], sizeof(int) * mthread_size));
		devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_edges[i], sizeof(bool) * m_edges_size));
		devMemUsed += sizeof(bool) * m_edges_size;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Generating Lists on GPU", hostMemUsed, devMemUsed);

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil(static_cast<float>(mthread_size) / THREAD_SIZE));
	unsigned int gridy = mblock_size;
	dim3 threads_per_block(1, BLOCK_SIZE, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);
	
	//Index 'i' marks the row and 'j' marks the column
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
					checkCudaErrors(cuMemcpyHtoDAsync(d_w0[m], nodes.crd->w() + i * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_x0[m], nodes.crd->x() + i * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_y0[m], nodes.crd->y() + i * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_z0[m], nodes.crd->z() + i * mthread_size, sizeof(float) * mthread_size, stream[m]));

					checkCudaErrors(cuMemcpyHtoDAsync(d_w1[m], nodes.crd->w() + (j*NBUFFERS+m) * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_x1[m], nodes.crd->x() + (j*NBUFFERS+m) * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_y1[m], nodes.crd->y() + (j*NBUFFERS+m) * mthread_size, sizeof(float) * mthread_size, stream[m]));
					checkCudaErrors(cuMemcpyHtoDAsync(d_z1[m], nodes.crd->z() + (j*NBUFFERS+m) * mthread_size, sizeof(float) * mthread_size, stream[m]));

					//Execute Kernel
					GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block, 0, stream[m]>>>((float*)d_w0[m], (float*)d_x0[m], (float*)d_y0[m], (float*)d_z0[m], (float*)d_w1[m], (float*)d_x1[m], (float*)d_y1[m], (float*)d_z1[m], (int*)d_k_in[m], (int*)d_k_out[m], (bool*)d_edges[m], diag, compact);
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
				if (i > j * NBUFFERS + m)
					continue;

				//Synchronize
				checkCudaErrors(cuStreamSynchronize(stream[m]));

				//Read Data from Buffers
				readDegrees(nodes.k_in, h_k_in[m], j*NBUFFERS+m, mthread_size);
				readDegrees(nodes.k_out, h_k_out[m], i, mthread_size);
				readEdges(edges, h_edges[m], g_idx, d_edges_size, mthread_size, i, j*NBUFFERS+m);
			}				
		}
	}
	
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

		cuMemFree(d_w0[i]);
		d_w0[i] = NULL;

		cuMemFree(d_x0[i]);
		d_x0[i] = NULL;

		cuMemFree(d_y0[i]);
		d_y0[i] = NULL;

		cuMemFree(d_z0[i]);
		d_z0[i] = NULL;

		devMemUsed -= sizeof(float) * mthread_size * 4;

		cuMemFree(d_w1[i]);
		d_w1[i] = NULL;

		cuMemFree(d_x1[i]);
		d_x1[i] = NULL;

		cuMemFree(d_y1[i]);
		d_y1[i] = NULL;

		cuMemFree(d_z1[i]);
		d_z1[i] = NULL;

		devMemUsed -= sizeof(float) * mthread_size * 4;

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

//Decode past and future edge lists using Bitonic Sort
bool decodeLists_v2(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (h_edges != NULL);
		assert (g_idx != NULL);

		assert (*g_idx > 0);
		assert (d_edges_size > 0);
	}

	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	int cpy_size;
	int i, j, k;

	size_t g_mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(*g_idx) / (BLOCK_SIZE * GROUP_SIZE)));
	size_t g_mthread_size = g_mblock_size * BLOCK_SIZE;

	//DEBUG
	//printf_red();
	//printf("G_MBLOCK SIZE:  %zu\n", g_mblock_size);
	//printf("G_MTHREAD_SIZE: %zu\n", g_mthread_size);
	//printf_std();
	//fflush(stdout);

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

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_future_edges, sizeof(int) * g_mthread_size));
	devMemUsed += sizeof(int) * g_mthread_size;

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Bitonic Sorting", hostMemUsed, devMemUsed);

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<float>(g_mthread_size) / BLOCK_SIZE));
	dim3 blocks_per_grid_decode(gridx_decode, 1, 1);

	for (i = 0; i < GROUP_SIZE; i++) {
		//Clear Device Buffers
		checkCudaErrors(cuMemsetD32(d_future_edges, 0, g_mthread_size));

		//Execute Kernel
		DecodeFutureEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_future_edges, *g_idx - i * g_mthread_size, d_edges_size - (*g_idx - i * g_mthread_size));
		getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeFutureEdges' Failed to Execute!\n");

		//Synchronize
		checkCudaErrors(cuCtxSynchronize());

		//Copy Memory from Device to Host
		cpy_size = *g_idx - static_cast<int>(i * g_mthread_size) >= 0 ? g_mthread_size : static_cast<int>(i * g_mthread_size) - *g_idx;
		checkCudaErrors(cuMemcpyDtoH(edges.future_edges + i * g_mthread_size, d_future_edges, sizeof(int) * cpy_size));
	}

	//Free Device Memory
	cuMemFree(d_future_edges);
	d_future_edges = NULL;
	devMemUsed -= sizeof(int) * g_mthread_size;

	//Resort Edges with New Encoding
	for (k = 2; k <= d_edges_size; k <<= 1) {
		for (j = k >> 1; j > 0; j >>= 1) {
			BitonicSort<<<blocks_per_grid_bitonic, threads_per_block>>>((uint64_t*)d_edges, j, k);
			getLastCudaError("Kernel 'Subroutines_GPU.BitonicSort' Failed to Execute!\n");
			checkCudaErrors(cuCtxSynchronize());
		}
	}

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edges, sizeof(int) * g_mthread_size));
	devMemUsed += sizeof(int) * g_mthread_size;

	for (i = 0; i < GROUP_SIZE; i++) {
		//Clear Device Buffers
		checkCudaErrors(cuMemsetD32(d_past_edges, 0, g_mthread_size));

		//Execute Kernel
		DecodePastEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_past_edges, *g_idx - i * g_mthread_size, d_edges_size - (*g_idx - i * g_mthread_size));
		getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePastEdges' Failed to Execute!\n");

		//Synchronize
		checkCudaErrors(cuCtxSynchronize());

		//Copy Memory from Device to Host
		cpy_size = *g_idx - static_cast<int>(i * g_mthread_size) >= 0 ? g_mthread_size : static_cast<int>(i * g_mthread_size) - *g_idx;
		checkCudaErrors(cuMemcpyDtoH(edges.past_edges + i * g_mthread_size, d_past_edges, sizeof(int) * cpy_size));
	}

	//Free Device Memory
	cuMemFree(d_past_edges);
	d_past_edges = NULL;
	devMemUsed -= sizeof(int) * g_mthread_size;

	cuMemFree(d_edges);
	d_edges = NULL;
	devMemUsed -= sizeof(uint64_t) * d_edges_size;

	return true;
}

//Parallel Prefix Sum of 'k_in' and 'k_out' and Write to Edge Pointers
bool scanLists(const Edge &edges, const CUdeviceptr &d_k_in, const CUdeviceptr d_k_out, const int &N_tar, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);

		assert (N_tar > 0);
	}

	CUdeviceptr d_past_edge_row_start, d_future_edge_row_start;
	CUdeviceptr d_buf, d_buf_scanned;
	int i;

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

	//CUDA Grid Specifications
	unsigned int gridx_scan = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE << 1)));
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_scan(gridx_scan, 1, 1);

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

	Scan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_k_out, (int*)d_future_edge_row_start, (int*)d_buf, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	Scan<<<dim3(1,1,1), threads_per_block>>>((int*)d_buf, (int*)d_buf_scanned, NULL, BLOCK_SIZE << 1);
	getLastCudaError("Kernel 'Subroutines_GPU.Scan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	PostScan<<<blocks_per_grid_scan, threads_per_block>>>((int*)d_future_edge_row_start, (int*)d_buf_scanned, N_tar);
	getLastCudaError("Kernel 'Subroutines_GPU.PostScan' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(edges.past_edge_row_start, d_past_edge_row_start, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(edges.future_edge_row_start, d_future_edge_row_start, sizeof(int) * N_tar));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

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

	return true;
}

bool identifyListProperties(const Node &nodes, const CUdeviceptr &d_k_in, const CUdeviceptr &d_k_out, const int *g_idx, const int &N_tar, int &N_res, int &N_deg2, float &k_res, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose)
{
	if (DEBUG) {
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (g_idx != NULL);

		assert (N_tar > 0);
	}

	CUdeviceptr d_N_res, d_N_deg2;

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_N_res, sizeof(int)));
	devMemUsed += sizeof(int);

	checkCudaErrors(cuMemAlloc(&d_N_deg2, sizeof(int)));
	devMemUsed += sizeof(int);
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("for Identifying List Properties", hostMemUsed, devMemUsed);
	
	//Initialize Memory on Device
	checkCudaErrors(cuMemsetD32(d_N_res, 0, 1));
	checkCudaErrors(cuMemsetD32(d_N_deg2, 0, 1));

	//CUDA Grid Specifications
	unsigned int gridx_res_prop = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / BLOCK_SIZE));
	dim3 threads_per_block(BLOCK_SIZE, 1, 1);
	dim3 blocks_per_grid_res_prop(gridx_res_prop, 1, 1);

	//Execute Kernel
	ResultingProps<<<gridx_res_prop, threads_per_block>>>((int*)d_k_in, (int*)d_k_out, (int*)d_N_res, (int*)d_N_deg2, N_tar);
	getLastCudaError("Kernel 'NetworkCreator_GPU.ResultingProps' Failed to Execute!\n");
	checkCudaErrors(cuCtxSynchronize());

	//Copy Memory from Device to Host
	checkCudaErrors(cuMemcpyDtoH(nodes.k_in, d_k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(nodes.k_out, d_k_out, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyDtoH(&N_res, d_N_res, sizeof(int)));
	checkCudaErrors(cuMemcpyDtoH(&N_deg2, d_N_deg2, sizeof(int)));

	//Synchronize
	checkCudaErrors(cuCtxSynchronize());

	N_res = N_tar - N_res;
	N_deg2 = N_tar - N_deg2;
	k_res = static_cast<float>(*g_idx * 2) / N_res;

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

	return true;
}
