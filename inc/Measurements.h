#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include "Causet.h"
#include "CuResources.h"

void measureClustering(Network *network, CausetPerformance *cp, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
bool nodesAreConnected(Network *network, int past_idx, int future_idx);

#endif
