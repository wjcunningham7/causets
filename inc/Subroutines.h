#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"
#include "CuResources.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

//Lookup Table (Linear Interpolation w/ Table)
bool getLookupTable(const char *filename, double **lt, long *size);

double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing);

double lookupValue4D(const double *table, const long &size, const double &omega12, double t1, double t2);

//Quicksort Algorithm
void quicksort(Node &nodes, const unsigned int &spacetime, int low, int high);

void quicksort(uint64_t *edges, int low, int high);

void swap(Node &nodes, const unsigned int &spacetime, const int i, const int j);

void swap(uint64_t *edges, const int i, const int j);

void swap(const int * const *& list0, const int * const *& list1, int &idx0, int &idx1, int &max0, int &max1);

//Bisection Algorithm
bool bisection(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3);

//Causal Relation Identification Algorithm
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const Bitset adj, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx);

bool nodesAreConnected_v2(const Bitset adj, const int &N_tar, int past_idx, int future_idx);

//Breadth First Search Algorithm
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements);

void bfsearch_v2(const Node &nodes, const Bitset &adj, const int &N_tar, const int index, const int id, int &elements);

//Array Intersection Algorithms
void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many);

void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many);

//Format Partial Adjacency Matrix Data
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size);

void readEdges(uint64_t * const &edges, const bool * const h_edges, Bitset &adj, int * const &g_idx, const unsigned int &core_limit, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y, const bool &use_bit);

void remakeAdjMatrix(bool * const adj0, bool * const adj1, const int * const k_in, const int * const k_out, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, int * const idx_buf0, int * const idx_buf1, const int &N_tar, const int &i, const int &j, const int &l);

void readIntervals(int * const cardinalities, const unsigned int * const N_ij, const int &l);

//Prefix Sum Algorithm
void scan(const int * const k_in, const int * const k_out, int * const &past_edge_pointers, int * const &future_edge_pointers, const int &N_tar);

//Overloaded Print Statements
int printf_dbg(const char * format, ...);

int printf_mpi(int rank, const char * format, ...);

//Check for MPI Errors
bool checkMpiErrors(CausetMPI &cmpi);

#endif
