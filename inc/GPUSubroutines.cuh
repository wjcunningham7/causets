#ifndef GPU_SUBROUTINES_CUH_
#define GPU_SUBROUTINES_CUH_

#include "GPUSubroutines.hpp"

__global__ void Generate(Node *nodes, unsigned int N, long seed)
{
	//unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	//unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	//if ((j * width) + i > N)
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
		if (CURAND_STATUS_SUCCESS != curandSetPseudoRandomGeneratorSeed(prng, (unsigned int)network->network_properties.seed))
			throw CausetException("Failed to set curand seed.\n");

		//Need to redesign Node for GPU so memory for points is contiguous
		//Lots of thought should go into this...
		//if (CURAND_STATUS_SUCCESS != curandGenerateUniform(prng, (float*)d_points, network->network_properties.N))
		//	throw CausetException("Failed to generate curand uniform number distribution.\n");

		if (CURAND_STATUS_SUCCESS != curandDestroyGenerator(prng))
			throw CausetException("Failed to destroy curand generator.\n");
	} catch (CausetException e) {
		fprintf(stderr, e.what());
		exit(EXIT_FAILURE);
	}

	//Invoke Kernel
	Generate<<<network->network_properties.network_exec.blocks_per_grid, network->network_properties.network_exec.threads_per_block>>>((Node*)network->d_nodes, network->network_properties.N, network->network_properties.seed);
	getLastCudaError("Kernel 'Generate' Failed to Execute!");
	checkCudaErrors(cuCtxSynchronize());

	//Copy Values to Host
	checkCudaErrors(cuMemcpyDtoH(network->nodes, network->d_nodes, sizeof(Node) * network->network_properties.N));

	return true;
}

#endif