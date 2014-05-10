#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include "Operations.h"
#include "GPUSubroutines.h"

bool createNetwork(Network *network, CausetPerformance *cp, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
bool generateNodes(Network *network, CausetPerformance *cp, bool &use_gpu);
bool linkNodes(Network *network, CausetPerformance *cp, bool &use_gpu);

void compareAdjacencyLists(Node *nodes, int *future_edges, int *future_edge_row_start, int *past_edges, int *past_edge_row_start);
void compareAdjacencyListIndices(Node *nodes, int *future_edges, int *future_edge_row_start, int *past_edges, int *past_edge_row_start);

void printValues(Node *values, int num_vals, char *filename, char *coord);
void printSpatialDistances(Node *nodes, Manifold manifold, int N_tar, int dim);

#endif
