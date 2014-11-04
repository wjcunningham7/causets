#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "Causet.h"
#include "CuResources.h"
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool validateEmbedding(EVData &evd, const Node &nodes, const Edge &edges, const int &N_tar, const double &N_emb, const int &N_res, const float &k_res, const int &dim, const Manifold &manifold, const double &a, const double &alpha, long &seed, Stopwatch &sValidateEmbedding, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose);

bool printValues(const Node &nodes, const int num_vals, const char *filename, const char *coord);

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out);

bool printEdgeLists(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

#endif
