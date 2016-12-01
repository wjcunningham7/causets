#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"
#include "CuResources.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

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

double lookupValue4D(const double *table, const long &size, const double &omega12, double t1, double t2);

//Quicksort Algorithm
void quicksort(Node &nodes, const unsigned int &spacetime, int low, int high);

void quicksort(uint64_t *edges, int64_t low, int64_t high);

void swap(Node &nodes, const unsigned int &spacetime, const int i, const int j);

void swap(uint64_t *edges, const int64_t i, const int64_t j);

void swap(const int * const *& list0, const int * const *& list1, int64_t &idx0, int64_t &idx1, int64_t &max0, int64_t &max1);

//Cyclesort Algorithm
void cyclesort(unsigned int &writes, std::vector<unsigned int> elements, std::vector<std::pair<int,int> > *swaps);

//Bisection Algorithm
bool bisection(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3);

//Causal Relation Identification Algorithm
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx);

bool nodesAreConnected_v2(const Bitvector &adj, const int &N_tar, int past_idx, int future_idx);

//Depth First Search Algorithm
void dfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements, int level);

void dfsearch_v2(const Node &nodes, const Bitvector &adj, const int &N_tar, const int index, const int id, int &elements);

//Breadth First Search Algorithm
int shortestPath(const Node &nodes, const Edge &edges, const int &N_tar, int * const distances, const int start, const int end);

//Array Intersection Algorithms
void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int64_t &pstart, const int64_t &fstart, bool &too_many);

void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int64_t &pstart, const int64_t &fstart, bool &too_many);

//Format Partial Adjacency Matrix Data
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size);

void readEdges(uint64_t * const &edges, const bool * const h_edges, Bitvector &adj, int64_t * const &g_idx, const unsigned int &core_limit_row, const unsigned int &core_limit_col, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y, const bool &use_bit, const bool &use_mpi);

void remakeAdjMatrix(bool * const adj0, bool * const adj1, const int * const k_in, const int * const k_out, const int * const past_edges, const int * const future_edges, const int64_t * const past_edge_row_start, const int64_t * const future_edge_row_start, int * const idx_buf0, int * const idx_buf1, const int &N_tar, const int &i, const int &j, const int64_t &l);

void readIntervals(int * const cardinalities, const unsigned int * const N_ij, const int &l);

//Prefix Sum Algorithm
void scan(const int * const k_in, const int * const k_out, int64_t * const &past_edge_pointers, int64_t * const &future_edge_pointers, const int &N_tar);

//Overloaded Print Statements
int printf_dbg(const char * format, ...);

int printf_mpi(int rank, const char * format, ...);

//Check for MPI Errors
bool checkMpiErrors(CausetMPI &cmpi);

//MPI Trading: Permutation Algorithms
void init_mpi_permutations(std::unordered_set<FastBitset> &permutations, std::vector<unsigned int> elements);

void remove_bad_perms(std::unordered_set<FastBitset> &permutations, std::unordered_set<std::pair<int,int> > pairs);

void init_mpi_pairs(std::unordered_set<std::pair<int,int> > &pairs, const std::vector<unsigned int> elements, bool &split_job);

void fill_mpi_similar(std::vector<std::vector<unsigned int> > &similar, std::vector<unsigned int> elements);

void get_most_similar(std::vector<unsigned int> &sim, unsigned int &nsteps, const std::vector<std::vector<unsigned int> > candidates, const std::vector<unsigned int> elements);

void relabel_vector(std::vector<unsigned int> &output, const std::vector<unsigned int> input);

void perm_to_binary(FastBitset &fb, const std::vector<unsigned int> perm);

void binary_to_perm(std::vector<unsigned int> &perm, const FastBitset &fb, const unsigned int len);

unsigned int loc_to_glob_idx(std::vector<unsigned int> perm, const unsigned int idx, const int N_tar, const int num_mpi_threads, const int rank);

//MPI Trading: Signals and Swaps
#ifdef MPI_ENABLED
void mpi_swaps(const std::vector<std::pair<int,int> > swaps, Bitvector &adj, Bitvector &adj_buf, const int N_tar, const int num_mpi_threads, const int rank);

void sendSignal(const MPISignal signal, const int rank, const int num_mpi_threads);
#endif

std::vector<std::tuple<FastBitset, int, int>> getPossibleChains(Bitvector &adj, Bitvector &subadj, Bitvector &chains, FastBitset *excluded, std::vector<std::pair<int,int>> &endpoints, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, const int min_weight, int &max_weight, int &max_idx, int &end_idx);

std::pair<int,int> longestChainGuided(Bitvector &adj, Bitvector &subadj, Bitvector &chains, FastBitset &longest_chain, FastBitset *workspace, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, int i, int j, const unsigned int level);

int longestChain_v3(Bitvector &adj, Bitvector &chains, FastBitset &longest_chain, FastBitset *workspace, int *lengths, const int N_tar, int i, int j, unsigned int level);

int longestChain_v2(Bitvector &adj, FastBitset *workspace, int *lengths, const int N_tar, int i, int j, unsigned int level);

int longestChain_v1(Bitvector &adj, FastBitset *workspace, const int N_tar, int i, int j, unsigned int level);

int rootChain(Bitvector &adj, Bitvector &workspace, FastBitset &chain, const int * const k_in, const int * const k_out, int *lengths, const int N_tar);

int findLongestMaximal(Bitvector &adj, const int *k_out, int *lengths, const int N_tar, const int idx);

int findLongestMinimal(Bitvector &adj, const int *k_in, int *lengths, const int N_tar, const int idx);

Network find_subgraph(const Network &network, std::vector<unsigned int> candidates, CaResources * const ca);

#endif
