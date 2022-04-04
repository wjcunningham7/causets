/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"
#include "CuResources.h"

//Hashing function for std::pair<int,int>
namespace std
{
	template<>
	class hash<std::pair<int,int> >
	{
	public:
		size_t operator()(std::pair<int,int> const& p) const
		{
			size_t seed = 0;
			boost::hash_combine(seed, (size_t)std::get<0>(p));
			boost::hash_combine(seed, (size_t)std::get<1>(p));
			return seed;
		}
	};
};

//Lookup Table (Linear Interpolation w/ Table)
bool getLookupTable(const char *filename, double **lt, long *size);

double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing);

//Quicksort Algorithm
void quicksort(Node &nodes, const Spacetime &spacetime, int low, int high);

void quicksort(uint64_t *edges, int64_t low, int64_t high);

void quicksort(std::vector<unsigned> &U, std::vector<unsigned> &V, int64_t low, int64_t high);

void swap(Node &nodes, const Spacetime &spacetime, const int i, const int j);

void swap(uint64_t *edges, const int64_t i, const int64_t j);

void swap(const int * const *& list0, const int * const *& list1, int64_t &idx0, int64_t &idx1, int64_t &max0, int64_t &max1);

void swap(std::vector<unsigned> &U, std::vector<unsigned> &V, int64_t i, int64_t j);

void ordered_labeling(std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> &Unew, std::vector<unsigned> &Vnew);

//Cyclesort Algorithm
void cyclesort(unsigned int &writes, std::vector<unsigned int> elements, std::vector<std::pair<int,int> > *swaps);

//Bisection Algorithm
bool bisection(double (*solve)(const double x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3);

//Causal Relation Identification Algorithm
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N, const float &core_edge_fraction, int past_idx, int future_idx);

bool nodesAreConnected_v2(const Bitvector &adj, const int &N, int past_idx, int future_idx);

//Depth First Search Algorithm
void dfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements, int level);

void dfsearch_v2(const Node &nodes, const Bitvector &adj, const int &N, const int index, const int id, int &elements);

//Breadth First Search Algorithm
int shortestPath(const Node &nodes, const Edge &edges, const int &N, int * const distances, const int start, const int end);

//Array Intersection Algorithms
void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int64_t &pstart, const int64_t &fstart);

void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int64_t &pstart, const int64_t &fstart);

//Format Partial Adjacency Matrix Data
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size);

void readEdges(uint64_t * const &edges, const bool * const h_edges, Bitvector &adj, int64_t * const &g_idx, const unsigned int &core_limit_row, const unsigned int &core_limit_col, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y, const bool &use_bit, const bool &use_mpi);

//Prefix Sum Algorithm
void scan(const int * const k_in, const int * const k_out, int64_t * const &past_edge_pointers, int64_t * const &future_edge_pointers, const int &N);

//Overloaded Print Statements
//int printf_dbg(const char * format, ...);

//int printf_mpi(int rank, const char * format, ...);

//Check for MPI Errors
bool checkMpiErrors(CausetMPI &cmpi);

//MPI Trading: Permutation Algorithms
void init_mpi_permutations(std::unordered_set<FastBitset> &permutations, std::vector<unsigned int> elements);

void remove_bad_perms(std::unordered_set<FastBitset> &permutations, std::unordered_set<std::pair<int,int> > pairs);

void init_mpi_pairs(std::unordered_set<std::pair<int,int> > &pairs, const std::vector<unsigned int> elements);

void fill_mpi_similar(std::vector<std::vector<unsigned int> > &similar, std::vector<unsigned int> elements);

void get_most_similar(std::vector<unsigned int> &sim, unsigned int &nsteps, const std::vector<std::vector<unsigned int> > candidates, const std::vector<unsigned int> elements);

void relabel_vector(std::vector<unsigned int> &output, const std::vector<unsigned int> input);

void perm_to_binary(FastBitset &fb, const std::vector<unsigned int> perm);

void binary_to_perm(std::vector<unsigned int> &perm, const FastBitset &fb, const unsigned int len);

unsigned int loc_to_glob_idx(std::vector<unsigned int> perm, const unsigned int idx, const int N, const int num_mpi_threads, const int rank);

//MPI Trading: Signals and Swaps
#ifdef MPI_ENABLED
void mpi_swaps(const std::vector<std::pair<int,int> > swaps, Bitvector &adj, Bitvector &adj_buf, const int N, const int num_mpi_threads, const int rank);

void sendSignal(const MPISignal signal, const int rank, const int num_mpi_threads);
#endif

//Chain/Antichain Identification and Measurement Algorithms
std::vector<std::tuple<FastBitset, int, int> > getPossibleChains(Bitvector &adj, Bitvector &subadj, Bitvector &chains, Bitvector &chains2, FastBitset *excluded, std::vector<std::pair<int,int> > &endpoints, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, const int min_weight, int &max_weight, int &max_idx, int &end_idx);

std::pair<int,int> longestChainGuided_v2(FastBitset &longest_chain, Bitvector &adj, const Bitvector &subadj, Bitvector &fwork, Bitvector &fwork2, FastBitset &fwork3, FastBitset &fwork4, const std::vector<unsigned> &candidates, int * const iwork, std::pair<int,int> * const i2work, const int N, const int NS, int i, int j, const unsigned level);

std::pair<int,int> longestChainGuided(Bitvector &adj, Bitvector &subadj, Bitvector &chains, Bitvector &chains2, FastBitset &longest_chain, FastBitset *workspace, FastBitset *subworkspace, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, int i, int j, const unsigned int level);

int longestChain_v3(Bitvector &adj, Bitvector &chains, FastBitset &longest_chain, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level);

int longestChain_v2r(Bitvector &adj, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level);

int longestChain_v2(const Bitvector &adj, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level);

int longestChain_v1(Bitvector &adj, FastBitset *workspace, const int N, int i, int j, unsigned int level);

int rootChain(Bitvector &adj, Bitvector &chains, FastBitset &chain, const int * const k_in, const int * const k_out, int *lengths, const int N);

int findLongestMaximal(Bitvector &adj, const int *k_out, int *lengths, const int N, const int idx);

int findLongestMinimal(Bitvector &adj, const int *k_in, int *lengths, const int N, const int idx);

Network find_subgraph(const Network &network, std::vector<unsigned int> candidates, CaResources * const ca);

int maximalAntichain(FastBitset &antichain, const Bitvector &adj, const int N, const int seed);

//Transitive Closure and Reduction Algorithms
void closure(Bitvector &adj, Bitvector &links, const int N, const int source, const int v);

void transitiveClosure(Bitvector &adj, Bitvector &links, const int N);

void transitiveReduction(Bitvector &links, Bitvector &adj, const int N);

//Timelike Boundary Algorithms
void identifyTimelikeCandidates(std::vector<unsigned> &candidates, int *chaintime, int *iwork, Bitvector &fwork, const Node &nodes, const Bitvector &adj, const Spacetime &spacetime, const int N, Stopwatch &sMeasureExtrinsicCurvature, const bool verbose, const bool bench);

bool configureSubgraph(Network *subgraph, const Node &nodes, std::vector<unsigned int> candidates, CaResources * const ca, CUcontext &ctx);

void identifyBoundaryChains(std::vector<std::tuple<FastBitset, int, int>> &boundary_chains, std::vector<std::pair<int,int>> &pwork, int *iwork, std::pair<int,int> *i2work, Bitvector &fwork, Bitvector &fwork2, Network * const network, Network * const subnet, std::vector<unsigned> &candidates);

//Statistical Analysis for Monte Carlo
void autocorrelation(double *data, double *acorr, const double * const lags, unsigned nsamples, double &tau_exp, double &tau_exp_err, double &tau_int, double &tau_int_err);

double jackknife(double *jacksamples, double mean, unsigned nsamples);

void specific_heat(double &Cv, double &err, double *action, double mean, double stddev, double beta, unsigned nsamples, unsigned stride);

void free_energy(double &F, double &err, double *action, double mean, double beta, unsigned nsamples, unsigned stride);

void entropy(double &s, double &err, double action_mean, double action_stddev, double free_energy, double free_energy_stddev, double beta, uint64_t npairs, unsigned nsamples);

#endif
