#ifndef GPU_SUBROUTINES_HPP_
#define GPU_SUBROUTINES_HPP_

#include "Causet.hpp"

__global__ void Generate(Node *nodes, unsigned int N, long seed);

bool generateNodesGPU(Network *network);

#endif