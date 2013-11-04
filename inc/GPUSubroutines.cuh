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

bool generateNodesGPU(Network *network, float &eta0)
{
	//Invoke Kernel
	Generate<<<network->network_properties.network_exec.blocks_per_grid, network->network_properties.network_exec.threads_per_block>>>((Node*)network->d_nodes, network->network_properties.N, network->network_properties.seed);
	getLastCudaError("Kernel 'Generate' Failed to Execute!");
	checkCudaErrors(cuCtxSynchronize());

	//Copy Values to Host
	checkCudaErrors(cuMemcpyDtoH(network->nodes, network->d_nodes, sizeof(Node) * network->network_properties.N));

	return true;
}

#endif