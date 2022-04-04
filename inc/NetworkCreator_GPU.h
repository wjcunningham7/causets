/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef NETWORK_CREATOR_GPU_H_
#define NETWORK_CREATOR_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"
#include "Validate.h"

template<bool compact, bool hyp, bool epso, bool diag, int stdim>
__global__ void GenerateAdjacencyLists_v2(float *w0, float *x0, float *y0, float *z0, float *w1, float *x1, float *y1, float *z1, int *k_in, int *k_out, bool *edges, size_t size0, size_t size1, float hyp_r);

__global__ void DecodeFutureEdges(uint64_t *edges, int *future_edges, int64_t elements, int64_t offset);

__global__ void DecodePastEdges(uint64_t *edges, int *past_edges, int64_t elements, int64_t offset);

__global__ void ResultingProps(int *k_in, int *k_out, int *N_res, int *N_deg2, int elements);

bool linkNodesGPU_v2(Node &nodes, const Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int N, const float k_tar, int &N_res, float &k_res, int &N_deg2, const double r_max, const float core_edge_fraction, const float edge_buffer, CausetMPI &cmpi, const int group_size, CaResources * const ca, Stopwatch &sLinkNodesGPU, CUcontext &ctx, const bool decode_cpu, const bool link_epso, const bool has_exact_k, const bool use_bit, const bool mpi_split, const bool verbose, const bool bench);

bool generateLists_v3(Node &nodes, Bitvector &adj, int64_t * const &g_idx, const Spacetime &spacetime, const int &N, const float &r_max, const int &group_size, CausetMPI &cmpi, CaResources * const ca, const bool &link_epso, const bool &use_bit, const bool &mpi_split, const bool &verbose, const bool &bench);

bool generateLists_v2(Node &nodes, uint64_t * const &edges, Bitvector &adj, int64_t * const &g_idx, const Spacetime &spacetime, const int &N, const double r_max, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const CUcontext &ctx, const bool &link_epso, const bool &use_bit, const bool &verbose, const bool &bench);

bool decodeLists_v2(const Edge &edges, const uint64_t * const h_edges, const int64_t * const g_idx, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &verbose);

bool decodeListsCPU(const Edge &edges, uint64_t *h_edges, const int64_t * const g_idx);

bool scanLists(const Edge &edges, const CUdeviceptr &d_k_in, const CUdeviceptr d_k_out, const int &N, CaResources * const ca, const bool &verbose);

bool identifyListProperties(const Node &nodes, const CUdeviceptr &d_k_in, const CUdeviceptr &d_k_out, const int64_t *g_idx, const int &N, int &N_res, int &N_deg2, float &k_res, CausetMPI &cmpi, CaResources * const ca, const bool &verbose);

#endif
