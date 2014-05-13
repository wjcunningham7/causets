#include "GPUSubroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

__global__ void Generate(Node *nodes, int N_tar, long seed)
{
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int j = blockDim.y * blockIdx.y + threadIdx.y;
	//if ((j * width) + i > N_tar)
	//	return;

	//Implement CURAND package here for random number generation
}

bool generateNodesGPU(Network *network)
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
}
