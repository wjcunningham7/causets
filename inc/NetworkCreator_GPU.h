#ifndef NETWORK_CREATOR_GPU_H_
#define NETWORK_CREATOR_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

#define BLOCK_SIZE 128

__global__ void Generate(Node *nodes, int N_tar, long seed);

//__global__ void GenerateAdjacencyLists(cudaTextureObject_t t_nodes, float4 *nodes, int *edges, int *g_idx, int width);

__global__ void GenerateAdjacencyLists(float4 *nodes, int *edges, int *k_in, int *k_out, int *g_idx, int width);

__global__ void BitonicSort(uint64_t edges, int n_links, int j, int k);

__global__ void FindNodeDegrees(int *past_edge_row_start, int *future_edge_row_start, int *k_in, int *k_out);

bool generateNodesGPU(Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodesGPU, const bool &universe, const bool &verbose, const bool &bench);

bool linkNodesGPU(Node &nodes, CUdeviceptr d_nodes, int * const &past_edges, CUdeviceptr d_past_edges, int * const &future_edges, CUdeviceptr d_future_edges, int * const &past_edge_row_start, CUdeviceptr d_past_edge_row_start, int * const &future_edge_row_start, CUdeviceptr d_future_edge_row_start, bool * const &core_edge_exists, CUdeviceptr d_k_in, CUdeviceptr d_k_out, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench);

#endif
