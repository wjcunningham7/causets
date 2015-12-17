#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists_v2(float *w0, float *x0, float *y0, float *z0, float *w1, float *x1, float *y1, float *z1, int *k_in, int *k_out, bool *edges, size_t size0, size_t size1, bool diag, bool compact)
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
			if (j * THREAD_SIZE + k < size1) {
				shr_w1[k] = w1[j*THREAD_SIZE+k];
				shr_x1[k] = x1[j*THREAD_SIZE+k];
				shr_y1[k] = y1[j*THREAD_SIZE+k];
				shr_z1[k] = z1[j*THREAD_SIZE+k];
			}
		}
	}
	__syncthreads();

	float dt[THREAD_SIZE];
	float dx[THREAD_SIZE];

	for (k = 0; k < THREAD_SIZE; k++) {
		if ((!diag || i < j * THREAD_SIZE + k) && (i < size0 && j * THREAD_SIZE + k < size1)) {
			//Identify nodes to compare
			n0.w = w0[i];
			n0.x = x0[i];
			n0.y = y0[i];
			n0.z = z0[i];

			n1.w = shr_w1[k];
			n1.x = shr_x1[k];
			n1.y = shr_y1[k];
			n1.z = shr_z1[k];

			//Identify spacetime interval
			dt[k] = n1.w - n0.w;

			if (compact) {
				if (DIST_V2)
					dx[k] = acosf(sphProduct_GPU_v2(n0, n1));
				else
					dx[k] = acosf(sphProduct_GPU_v1(n0, n1));
			} else {
				if (DIST_V2)
					dx[k] = sqrtf(flatProduct_GPU_v2(n0, n1));
				else
					dx[k] = sqrtf(flatProduct_GPU_v1(n0, n1));
			}
		}
	}

	bool edge[THREAD_SIZE];
	int out = 0;
	for (k = 0; k < THREAD_SIZE; k++) {
		//Mark if edge is present (register memory)
		edge[k] = (!diag || i < j * THREAD_SIZE + k) && (i < size0 && j * THREAD_SIZE + k < size1) && dx[k] < dt[k];
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
		if ((!diag || i < j * THREAD_SIZE + k) && (i < size0 && j * THREAD_SIZE + k < size1))
			edges[(i*THREAD_SIZE*gridDim.x)+(j*THREAD_SIZE)+k] = edge[k];

	//Write out-degrees
	atomicAdd(&k_out[i], out);

	//Wrtie in-degrees
	if (!tid)
		for (k = 0; k < THREAD_SIZE; k++)
			if ((!diag || i < j * THREAD_SIZE + k) && (i < size0 && j * THREAD_SIZE + k < size1))
				atomicAdd(&k_in[j*THREAD_SIZE+k], n[0][k]);
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

bool linkNodesGPU_v2(Node &nodes, const Edge &edges, std::vector<bool> &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, const int &group_size, CaResources * const ca, Stopwatch &sLinkNodesGPU, const CUcontext &ctx, const bool &decode_cpu, const bool &use_bit, const bool &compact, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (nodes.crd->getDim() == 4);
	assert (!nodes.crd->isNull());
	assert (nodes.crd->w() != NULL);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.crd->z() != NULL);
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	} else
		assert (core_edge_fraction == 1.0f);
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	Stopwatch sGenAdjList = Stopwatch();
	Stopwatch sDecodeLists = Stopwatch();
	Stopwatch sScanLists = Stopwatch();
	Stopwatch sProps = Stopwatch();

	CUdeviceptr d_k_in, d_k_out;
	uint64_t *h_edges;
	int *g_idx;

	size_t d_edges_size = use_bit ? 1 : pow(2.0, ceil(log2(N_tar * k_tar * (1.0 + edge_buffer) / 2)));

	stopwatchStart(&sLinkNodesGPU);

	//Allocate Overhead on Host
	try {
		h_edges = (uint64_t*)malloc(sizeof(uint64_t) * d_edges_size);
		if (h_edges == NULL)
			throw std::bad_alloc();
		memset(h_edges, 0, sizeof(uint64_t) * d_edges_size);
		ca->hostMemUsed += sizeof(uint64_t) * d_edges_size;

		g_idx = (int*)malloc(sizeof(int));
		if (g_idx == NULL)
			throw std::bad_alloc();
		memset(g_idx, 0, sizeof(int));
		ca->hostMemUsed += sizeof(int);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	stopwatchStart(&sGenAdjList);
	#if GEN_ADJ_LISTS_GPU_V2
	if (!generateLists_v2(nodes, h_edges, core_edge_exists, g_idx, N_tar, core_edge_fraction, d_edges_size, group_size, ca, ctx, use_bit, compact, verbose))
		return false;
	#else
	if (!generateLists_v1(nodes, h_edges, core_edge_exists, g_idx, N_tar, core_edge_fraction, d_edges_size, group_size, ca, compact, verbose))
		return false;
	#endif
	stopwatchStop(&sGenAdjList);

	if (!use_bit) {
		try {
			if (*g_idx + 1 >= static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer) / 2))
				throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
		} catch (CausetException c) {
			fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
			return false;
		} catch (std::exception e) {
			fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
			return false;
		}

		/*if (!printDegrees(nodes, N_tar, "in-degrees_GPU_v2.cset.dbg.dat", "out-degrees_GPU_v2.cset.dbg.dat")) return false;
		printf_red();
		printf("Check files now.\n");
		printf_std();
		fflush(stdout);
		printChk();*/

		//Decode Adjacency Lists
		stopwatchStart(&sDecodeLists);
		if (decode_cpu) {
			if (!decodeListsCPU(edges, h_edges, g_idx))
				return false;
		} else {
			#if DECODE_LISTS_GPU_V2
			if (!decodeLists_v2(edges, h_edges, g_idx, d_edges_size, group_size, ca, verbose))
				return false;
			#else
			if (!decodeLists_v1(edges, h_edges, g_idx, d_edges_size, ca, verbose))
				return false;
			#endif
		}
		stopwatchStop(&sDecodeLists);
	}

	//Free Host Memory
	free(h_edges);
	h_edges = NULL;
	ca->hostMemUsed -= sizeof(uint64_t) * d_edges_size;
	
	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_k_in, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_k_out, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;

	//Copy Memory from Host to Device
	checkCudaErrors(cuMemcpyHtoD(d_k_in, nodes.k_in, sizeof(int) * N_tar));
	checkCudaErrors(cuMemcpyHtoD(d_k_out, nodes.k_out, sizeof(int) * N_tar));

	//Identify Resulting Network Properties
	stopwatchStart(&sProps);
	if (!identifyListProperties(nodes, d_k_in, d_k_out, g_idx, N_tar, N_res, N_deg2, k_res, ca, verbose))
		return false;
	stopwatchStop(&sProps);	

	if (!use_bit) {
		//Prefix Scan of Degrees
		stopwatchStart(&sScanLists);
		scan(nodes.k_in, nodes.k_out, edges.past_edge_row_start, edges.future_edge_row_start, N_tar);
		stopwatchStop(&sScanLists);
	}

	//Free Device Memory
	cuMemFree(d_k_in);
	d_k_in = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_k_out);
	d_k_out = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	stopwatchStop(&sLinkNodesGPU);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tUndirected Links:         %d\n", *g_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", (k_res * N_res) / N_tar);
		printf_red();
		printf("\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		printf_std();
		fflush(stdout);
	}

	//if(!compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction))
	//	return false;

	//Print Results
	/*if (!printDegrees(nodes, N_tar, "in-degrees_GPU_v2.cset.dbg.dat", "out-degrees_GPU_v2.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, *g_idx, "past-edges_GPU_v2.cset.dbg.dat", "future-edges_GPU_v2.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_GPU_v2.cset.dbg.dat", "future-edge-pointers_GPU_v2.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	printChk();*/

	//Free Host Memory
	free(g_idx);
	g_idx = NULL;
	ca->hostMemUsed -= sizeof(int);

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
bool generateLists_v2(Node &nodes, uint64_t * const &edges, std::vector<bool> &core_edge_exists, int * const &g_idx, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const CUcontext &ctx, const bool &use_bit, const bool &compact, const bool &verbose)
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
	assert (g_idx != NULL);
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	if (use_bit)
		assert (core_edge_fraction == 1.0f);
	#endif

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

	unsigned int core_limit = static_cast<unsigned int>(core_edge_fraction * N_tar);
	unsigned int i, j, m;
	bool diag;

	//Thread blocks are grouped into "mega" blocks
	size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE * group_size)));
	size_t mthread_size = mblock_size * BLOCK_SIZE;
	size_t m_edges_size = mthread_size * mthread_size;

	//Create Streams
	for (i = 0; i < NBUFFERS; i++)
		checkCudaErrors(cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING));

	//Allocate Memory
	for (i = 0; i < NBUFFERS; i++) {
		checkCudaErrors(cuMemHostAlloc((void**)&h_k_in[i], sizeof(int) * mthread_size, CU_MEMHOSTALLOC_PORTABLE));
		ca->hostMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemHostAlloc((void**)&h_k_out[i], sizeof(int) * mthread_size, CU_MEMHOSTALLOC_PORTABLE));
		ca->hostMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemHostAlloc((void**)&h_edges[i], sizeof(bool) * m_edges_size, CU_MEMHOSTALLOC_PORTABLE));
		ca->hostMemUsed += sizeof(bool) * m_edges_size;

		checkCudaErrors(cuMemAlloc(&d_w0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_x0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_y0[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z0[i], sizeof(float) * mthread_size));
		ca->devMemUsed += sizeof(float) * mthread_size * 4;

		checkCudaErrors(cuMemAlloc(&d_w1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_x1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_y1[i], sizeof(float) * mthread_size));
		checkCudaErrors(cuMemAlloc(&d_z1[i], sizeof(float) * mthread_size));
		ca->devMemUsed += sizeof(float) * mthread_size * 4;

		checkCudaErrors(cuMemAlloc(&d_k_in[i], sizeof(int) * mthread_size));
		ca->devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_k_out[i], sizeof(int) * mthread_size));
		ca->devMemUsed += sizeof(int) * mthread_size;

		checkCudaErrors(cuMemAlloc(&d_edges[i], sizeof(bool) * m_edges_size));
		ca->devMemUsed += sizeof(bool) * m_edges_size;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Generating Lists on GPU", ca->hostMemUsed, ca->devMemUsed, 0);

	//CUDA Grid Specifications
	unsigned int gridx = static_cast<unsigned int>(ceil(static_cast<float>(mthread_size) / THREAD_SIZE));
	unsigned int gridy = mblock_size;
	dim3 threads_per_block(1, BLOCK_SIZE, 1);
	dim3 blocks_per_grid(gridx, gridy, 1);

	size_t final_size = N_tar - mthread_size * (group_size - 1);
	size_t size0, size1;

	//Index 'i' marks the row and 'j' marks the column
	for (i = 0; i < group_size; i++) {
		for (j = 0; j < group_size / NBUFFERS; j++) {
			for (m = 0; m < NBUFFERS; m++) {
				if (i > j * NBUFFERS + m)
					continue;

				diag = (i == j * NBUFFERS + m);

				size0 = (i < group_size - 1) ? mthread_size : final_size;
				size1 = (j * NBUFFERS + m < group_size - 1) ? mthread_size : final_size;

				//Clear Device Buffers
				checkCudaErrors(cuMemsetD32Async(d_k_in[m], 0, mthread_size, stream[m]));
				checkCudaErrors(cuMemsetD32Async(d_k_out[m], 0, mthread_size, stream[m]));
				checkCudaErrors(cuMemsetD8Async(d_edges[m], 0, m_edges_size, stream[m]));					
			
				//Transfer Nodes to Device Buffers
				checkCudaErrors(cuMemcpyHtoDAsync(d_w0[m], nodes.crd->w() + i * mthread_size, sizeof(float) * size0, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_x0[m], nodes.crd->x() + i * mthread_size, sizeof(float) * size0, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_y0[m], nodes.crd->y() + i * mthread_size, sizeof(float) * size0, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_z0[m], nodes.crd->z() + i * mthread_size, sizeof(float) * size0, stream[m]));

				checkCudaErrors(cuMemcpyHtoDAsync(d_w1[m], nodes.crd->w() + (j * NBUFFERS + m) * mthread_size, sizeof(float) * size1, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_x1[m], nodes.crd->x() + (j * NBUFFERS + m) * mthread_size, sizeof(float) * size1, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_y1[m], nodes.crd->y() + (j * NBUFFERS + m) * mthread_size, sizeof(float) * size1, stream[m]));
				checkCudaErrors(cuMemcpyHtoDAsync(d_z1[m], nodes.crd->z() + (j * NBUFFERS + m) * mthread_size, sizeof(float) * size1, stream[m]));

				//Execute Kernel
				GenerateAdjacencyLists_v2<<<blocks_per_grid, threads_per_block, 0, stream[m]>>>((float*)d_w0[m], (float*)d_x0[m], (float*)d_y0[m], (float*)d_z0[m], (float*)d_w1[m], (float*)d_x1[m], (float*)d_y1[m], (float*)d_z1[m], (int*)d_k_in[m], (int*)d_k_out[m], (bool*)d_edges[m], size0, size1, diag, compact);
				getLastCudaError("Kernel 'NetworkCreator_GPU.GenerateAdjacencyLists_v2' Failed to Execute!\n");

				//Copy Memory to Host Buffers
				checkCudaErrors(cuMemcpyDtoHAsync(h_k_in[m], d_k_in[m], sizeof(int) * size1, stream[m]));
				checkCudaErrors(cuMemcpyDtoHAsync(h_k_out[m], d_k_out[m], sizeof(int) * size0, stream[m]));
				checkCudaErrors(cuMemcpyDtoHAsync(h_edges[m], d_edges[m], sizeof(bool) * m_edges_size, stream[m]));

				//Synchronize
				checkCudaErrors(cuStreamSynchronize(stream[m]));

				//Read Data from Buffers
				readDegrees(nodes.k_in, h_k_in[m], (j * NBUFFERS + m) * mthread_size, size1);
				readDegrees(nodes.k_out, h_k_out[m], i * mthread_size, size0);
				readEdges(edges, h_edges[m], core_edge_exists, g_idx, core_limit, d_edges_size, mthread_size, size0, size1, i, j*NBUFFERS+m, use_bit);
			}				
		}
	}

	//Free Buffers
	for (i = 0; i < NBUFFERS; i++) {
		cuMemFreeHost(h_k_in[i]);
		h_k_in[i] = NULL;
		ca->hostMemUsed -= sizeof(int) * mthread_size;

		cuMemFreeHost(h_k_out[i]);
		h_k_out[i] = NULL;
		ca->hostMemUsed -= sizeof(int) * mthread_size;

		cuMemFreeHost(h_edges[i]);
		h_edges[i] = NULL;
		ca->hostMemUsed -= sizeof(bool) * m_edges_size;

		cuMemFree(d_w0[i]);
		d_w0[i] = 0;

		cuMemFree(d_x0[i]);
		d_x0[i] = 0;

		cuMemFree(d_y0[i]);
		d_y0[i] = 0;

		cuMemFree(d_z0[i]);
		d_z0[i] = 0;

		ca->devMemUsed -= sizeof(float) * mthread_size * 4;

		cuMemFree(d_w1[i]);
		d_w1[i] = 0;

		cuMemFree(d_x1[i]);
		d_x1[i] = 0;

		cuMemFree(d_y1[i]);
		d_y1[i] = 0;

		cuMemFree(d_z1[i]);
		d_z1[i] = 0;

		ca->devMemUsed -= sizeof(float) * mthread_size * 4;

		cuMemFree(d_k_in[i]);
		d_k_in[i] = 0;
		ca->devMemUsed -= sizeof(int) * mthread_size;

		cuMemFree(d_k_out[i]);
		d_k_out[i] = 0;
		ca->devMemUsed -= sizeof(int) * mthread_size;

		cuMemFree(d_edges[i]);
		d_edges[i] = 0;
		ca->devMemUsed -= sizeof(bool) * m_edges_size;
	}

	//Destroy Streams
	for (i = 0; i < NBUFFERS; i++)
		checkCudaErrors(cuStreamDestroy(stream[i]));

	//Final Synchronization
	checkCudaErrors(cuCtxSynchronize());

	return true;
}

//Decode past and future edge lists using Bitonic Sort
bool decodeLists_v2(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &verbose)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (h_edges != NULL);
	assert (g_idx != NULL);
	assert (ca != NULL);
	assert (*g_idx > 0);
	assert (d_edges_size > 0);
	#endif

	CUdeviceptr d_edges;
	CUdeviceptr d_past_edges, d_future_edges;
	int cpy_size;
	int i, j, k;

	size_t g_mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(*g_idx) / (BLOCK_SIZE * group_size)));
	size_t g_mthread_size = g_mblock_size * BLOCK_SIZE;

	//DEBUG
	/*printf_red();
	printf("G_IDX:          %d\n", *g_idx);
	printf("BLOCK_SIZE:     %d\n", BLOCK_SIZE);
	printf("GROUP_SIZE:     %d\n", group_size);
	printf("G_MBLOCK_SIZE:  %zu\n", g_mblock_size);
	printf("G_MTHREAD_SIZE: %zu\n", g_mthread_size);
	printf_std();
	fflush(stdout);*/

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
	checkCudaErrors(cuMemAlloc(&d_future_edges, sizeof(int) * g_mthread_size));
	ca->devMemUsed += sizeof(int) * g_mthread_size;

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Bitonic Sorting", ca->hostMemUsed, ca->devMemUsed, 0);

	//CUDA Grid Specifications
	unsigned int gridx_decode = static_cast<unsigned int>(ceil(static_cast<float>(g_mthread_size) / BLOCK_SIZE));
	dim3 blocks_per_grid_decode(gridx_decode, 1, 1);

	for (i = 0; i < group_size; i++) {
		//Clear Device Buffers
		checkCudaErrors(cuMemsetD32(d_future_edges, 0, g_mthread_size));

		//Execute Kernel
		DecodeFutureEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_future_edges, *g_idx - i * g_mthread_size, d_edges_size - (*g_idx - i * g_mthread_size));
		getLastCudaError("Kernel 'NetworkCreator_GPU.DecodeFutureEdges' Failed to Execute!\n");

		//Synchronize
		checkCudaErrors(cuCtxSynchronize());

		//Copy Memory from Device to Host
		if (*g_idx > g_mthread_size)
			cpy_size = *g_idx - static_cast<int>(i * g_mthread_size) >= 0 ? g_mthread_size : static_cast<int>(i * g_mthread_size) - *g_idx;
		else
			cpy_size = *g_idx;
		checkCudaErrors(cuMemcpyDtoH(edges.future_edges + i * g_mthread_size, d_future_edges, sizeof(int) * cpy_size));

		if (cpy_size < g_mthread_size)
			break;
	}

	//Free Device Memory
	cuMemFree(d_future_edges);
	d_future_edges = 0;
	ca->devMemUsed -= sizeof(int) * g_mthread_size;

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
	ca->devMemUsed += sizeof(int) * g_mthread_size;

	for (i = 0; i < group_size; i++) {
		//Clear Device Buffers
		checkCudaErrors(cuMemsetD32(d_past_edges, 0, g_mthread_size));

		//Execute Kernel
		DecodePastEdges<<<blocks_per_grid_decode, threads_per_block>>>((uint64_t*)d_edges, (int*)d_past_edges, *g_idx - i * g_mthread_size, d_edges_size - (*g_idx - i * g_mthread_size));
		getLastCudaError("Kernel 'NetworkCreator_GPU.DecodePastEdges' Failed to Execute!\n");

		//Synchronize
		checkCudaErrors(cuCtxSynchronize());

		//Copy Memory from Device to Host
		if (*g_idx > g_mthread_size)
			cpy_size = *g_idx - static_cast<int>(i * g_mthread_size) >= 0 ? g_mthread_size : static_cast<int>(i * g_mthread_size) - *g_idx;
		else
			cpy_size = *g_idx;
		checkCudaErrors(cuMemcpyDtoH(edges.past_edges + i * g_mthread_size, d_past_edges, sizeof(int) * cpy_size));

		if (cpy_size < g_mthread_size)
			break;
	}

	//Free Device Memory
	cuMemFree(d_past_edges);
	d_past_edges = 0;
	ca->devMemUsed -= sizeof(int) * g_mthread_size;

	cuMemFree(d_edges);
	d_edges = 0;
	ca->devMemUsed -= sizeof(uint64_t) * d_edges_size;

	return true;
}

bool decodeListsCPU(const Edge &edges, uint64_t *h_edges, const int * const g_idx)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (h_edges != NULL);
	assert (g_idx != NULL);
	assert (*g_idx > 0);
	#endif

	uint64_t key;
	unsigned int idx0, idx1;
	int i;

	quicksort(h_edges, 0, *g_idx - 1);

	for (i = 0; i < *g_idx; i++) {
		key = h_edges[i];
		idx0 = key >> 32;
		idx1 = key & 0x00000000FFFFFFFF;
		edges.future_edges[i] = idx1;
		h_edges[i] = ((uint64_t)idx1) << 32 | ((uint64_t)idx0);
	}

	quicksort(h_edges, 0, *g_idx - 1);

	for (i = 0; i < *g_idx; i++) {
		key = h_edges[i];
		idx0 = key >> 32;
		idx1 = key & 0x00000000FFFFFFFF;
		edges.past_edges[i] = idx1;
	}

	return true;
}

//Parallel Prefix Sum of 'k_in' and 'k_out' and Write to Edge Pointers
//This function works, but has been deprecated since it doesn't provide much speedup
bool scanLists(const Edge &edges, const CUdeviceptr &d_k_in, const CUdeviceptr d_k_out, const int &N_tar, CaResources * const ca, const bool &verbose)
{
	#if DEBUG
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (ca != NULL);
	assert (N_tar > 0);
	#endif

	CUdeviceptr d_past_edge_row_start, d_future_edge_row_start;
	CUdeviceptr d_buf, d_buf_scanned;
	int i;

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_past_edge_row_start, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_future_edge_row_start, sizeof(int) * N_tar));
	ca->devMemUsed += sizeof(int) * N_tar;

	checkCudaErrors(cuMemAlloc(&d_buf, sizeof(int) * (BLOCK_SIZE << 1)));
	ca->devMemUsed += sizeof(int) * (BLOCK_SIZE << 1);

	checkCudaErrors(cuMemAlloc(&d_buf_scanned, sizeof(int) * (BLOCK_SIZE << 1)));
	ca->devMemUsed += sizeof(int) * (BLOCK_SIZE << 1);
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Parallel Prefix Sum", ca->hostMemUsed, ca->devMemUsed, 0);

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
	d_past_edge_row_start = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_future_edge_row_start);
	d_future_edge_row_start = 0;
	ca->devMemUsed -= sizeof(int) * N_tar;

	cuMemFree(d_buf);
	d_buf = 0;
	ca->devMemUsed -= sizeof(int) * (BLOCK_SIZE << 1);

	cuMemFree(d_buf_scanned);
	d_buf_scanned = 0;
	ca->devMemUsed -= sizeof(int) * (BLOCK_SIZE << 1);

	return true;
}

bool identifyListProperties(const Node &nodes, const CUdeviceptr &d_k_in, const CUdeviceptr &d_k_out, const int *g_idx, const int &N_tar, int &N_res, int &N_deg2, float &k_res, CaResources * const ca, const bool &verbose)
{
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (g_idx != NULL);
	assert (N_tar > 0);
	#endif

	CUdeviceptr d_N_res, d_N_deg2;

	//Allocate Device Memory
	checkCudaErrors(cuMemAlloc(&d_N_res, sizeof(int)));
	ca->devMemUsed += sizeof(int);

	checkCudaErrors(cuMemAlloc(&d_N_deg2, sizeof(int)));
	ca->devMemUsed += sizeof(int);
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("for Identifying List Properties", ca->hostMemUsed, ca->devMemUsed, 0);
	
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

	#if DEBUG
	assert (N_res >= 0);
	assert (N_deg2 >= 0);
	assert (k_res >= 0.0);
	#endif

	//Free Device Memory
	cuMemFree(d_N_res);
	d_N_res = 0;
	ca->devMemUsed -= sizeof(int);

	cuMemFree(d_N_deg2);
	d_N_deg2 = 0;
	ca->devMemUsed -= sizeof(int);

	return true;
}
