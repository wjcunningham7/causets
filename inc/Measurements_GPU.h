#ifndef MEASUREMENTS_GPU_H_
#define MEASUREMENTS_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void MeasureAction(bool *edges0, bool *edges1, unsigned int *N_ij);

bool measureActionGPU(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const bool * const core_edge_exists, const int &N_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &compact, const bool &verbose, const bool &bench);

#endif
