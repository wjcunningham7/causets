/////////////////////////////
//(C) Will Cunningham 2020 //
//    Perimeter Institute  //
/////////////////////////////

#ifndef ANNEAL_H
#define ANNEAL_H

#include "Causet.h"

//typedef curandStatePhilox4_32_10_t RNGState;
typedef curandState_t RNGState;

#include "AnnealKernels.h"

struct CausetReplicas {
	CUdeviceptr adj;		//Adjacency (causal) matrix
	CUdeviceptr link;		//Link matrix (irreducible relations)
	CUdeviceptr action;		//Action of the replica
	CUdeviceptr offspring;		//Replica's number of offspring
	CUdeviceptr partial_sums;	//Partial sums
};

__global__ void RNGInit(RNGState *rng, uint64_t rng_seed, unsigned R);
__global__ void ReplicaInit(unsigned *adj, uint64_t rng_seed, uint64_t initial_sequence, const int N);
__global__ void ReplicaInitPacked(unsigned *adj, unsigned *families, RNGState *rng, const int N, const size_t bitblocks);
template<size_t TBSIZE>
__global__ void ReplicaInit2D(unsigned *U, unsigned *V, unsigned *families, RNGState *rng, const int N, const int fam_offset);

template<size_t TBSIZE>
__global__ void OrderToMatrix(unsigned *adj, unsigned *U, unsigned *V, unsigned *nrel, const int N, const size_t bitblocks);
template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaCountPacked(unsigned *adj, unsigned *cnt, const int N, const size_t bitblocks);

template<size_t ROW_CACHE>
__global__ void SymmetrizePacked(unsigned *adj, const int N, const size_t bitblocks);
template<size_t ROW_CACHE>
__global__ void ReplicaClosure(unsigned *adj, const int N, const int iter);
template<size_t ROW_CACHE>
__global__ void ReplicaClosurePacked(unsigned *adj, const int N, const size_t bitblocks);
template<size_t ROW_CACHE>
__global__ void ReplicaReduction(unsigned *adj, const int N, const int iter);
template<size_t ROW_CACHE>
__global__ void ReplicaReductionPacked(unsigned *adj, const int N, const size_t bitblocks);

__global__ void RelationAction(double *action, double *observables, unsigned *nrel, unsigned *nlink, const unsigned R, const double A0, const double A1);
template<size_t ROW_CACHE>
__global__ void ReplicaActionPair(unsigned *adj, double *action, double *averages, const int N, const double A, const int iter);
template<size_t ROW_CACHE>
__global__ void ReplicaActionPairPacked(double *action, double *observables, unsigned *adj, unsigned *nrel, const int N, const size_t bitblocks, const double A, const double B);

__global__ void QKernel(double *action, int R, double dB, double mean_action, double *Q);
__global__ void TauKernel(double *action, unsigned *offspring, unsigned *partial_sums, unsigned R0, unsigned R, double lnQ, double dB, unsigned *Rnew, uint64_t rng_seed, uint64_t initial_sequence);
__global__ void TauKernel_v2(double *action, unsigned *offspring, unsigned *partial_sums, RNGState *rng, unsigned R0, unsigned R, unsigned Rloc, double lnQ, double dB, unsigned *Rnew, double *Reff);
__global__ void PartialSum(unsigned *offspring, unsigned *partial_sums, unsigned R);

__global__ void ResampleReplicas(unsigned *adj_new, unsigned *adj, unsigned *link_new, unsigned *link, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned R, unsigned N);
__global__ void ResampleReplicas2D(unsigned *U_new, unsigned *U, unsigned *V_new, unsigned *V, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned *families_new, unsigned *families, unsigned R, unsigned N);
__global__ void ResampleReplicasPacked(unsigned *adj_new, unsigned *adj, unsigned *link_new, unsigned *link, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned *families_new, unsigned *families, unsigned R, unsigned N, size_t bitblocks);

template<size_t ROW_CACHE>
__global__ void RelaxReplicas(unsigned *adj, unsigned *link, double *action, double *averages, unsigned sweeps, uint64_t rng_seed, uint64_t initial_sequence, unsigned N, double beta, double A);
template<GraphType gt, WeightFunction wf, size_t ROW_CACHE_SIZE>
__global__ void RelaxReplicasPacked(unsigned *adj, unsigned *link, unsigned *U, unsigned *V, unsigned *nrel, unsigned *nlink, double *action, RNGState *rng, unsigned sweeps, unsigned steps, unsigned N, double beta, double A0, double A1, size_t bitblocks);

__global__ void MeasureObservables(double *observables, double *biased_action, double *action, unsigned *nrel, unsigned *nlink, unsigned R, unsigned npairs, unsigned stride);

__global__ void HistogramOverlap(double *action, unsigned R0, unsigned R, unsigned Rloc, double lnQ, double dB, double *overlap, double mean_energy);
double calc_histogram_overlap(CUdeviceptr &d_action, CUdeviceptr &d_Q, CUdeviceptr &d_overlap, const unsigned R0, const unsigned R, const unsigned Rloc, const double dB, const double mean_action, const dim3 blocks_per_grid, const dim3 threads_per_block, unsigned rank);
void adaptive_beta_stepsize(double &dB, CUdeviceptr d_action, CUdeviceptr d_Q, CUdeviceptr d_overlap, const unsigned R0, const unsigned R, const unsigned Rloc, const double beta, const double beta_final, const double min_overlap, const double max_overlap, const double mean_action, const dim3 blocks_per_grid, const dim3 threads_per_block, unsigned rank);

//double autocorrelation(double *h_action_fine, double *acorr, unsigned theta, unsigned theta_max, unsigned &max_lag, unsigned rank);

#ifdef MPI_ENABLED
void balance_replicas(CUdeviceptr &d_adj, CUdeviceptr &d_link, CUdeviceptr &d_U, CUdeviceptr &d_V, CUdeviceptr &d_nrel, CUdeviceptr &d_nlink, CUdeviceptr &d_action, CUdeviceptr &d_offspring, CUdeviceptr &d_parsum, CUdeviceptr &d_families, GraphType gt, unsigned N, unsigned R, unsigned &Rloc, size_t uints_per_replica, size_t &nblocks_rep, size_t &nblocks_obs, dim3 &blocks_per_grid_rep, dim3 &blocks_per_grid_obs, unsigned rank, unsigned mpi_threads);

template<typename T>
MPI_Datatype getMPIDataType();

template<>
inline MPI_Datatype getMPIDataType<double>() { return MPI_DOUBLE; }

template<>
inline MPI_Datatype getMPIDataType<unsigned>() { return MPI_UNSIGNED; }
#endif

bool annealNetwork_v2(Network * const network, CuResources * const cu, CaResources * const ca);
bool annealNetwork(Network * const network, CuResources * const cu, CaResources * const ca);

template<typename T>
void gather_data(T *data, CUdeviceptr d_data, int * const sendcnt, int * const displs, unsigned Rloc, int nprocs, int rank);

template<typename T>
void annealH5Write(T *data, hid_t file, hid_t dset, hid_t dspace, hsize_t *rowdim, hsize_t *rowoffset, unsigned num_values);

#endif
