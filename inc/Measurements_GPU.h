#ifndef MEASUREMENTS_GPU_H_
#define MEASUREMENTS_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"
//#include "Cusp.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

__global__ void MeasureAction(bool *edges0, bool *edges1, unsigned int *N_ij);

bool measureActionGPU(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, const int &N_tar, const int &stdim, const Manifold &manifold, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &use_bit, const bool &compact, const bool &verbose, const bool &bench);

#endif
