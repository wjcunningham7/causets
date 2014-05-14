#ifndef GPU_SUBROUTINES_H_
#define GPU_SUBROUTINES_H_

#include "Causet.h"
#include "CuResources.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

__global__ void Generate(Node *nodes, int N_tar, long seed);

bool generateNodesGPU(Network *network);

#endif
