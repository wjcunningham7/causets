#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Lookup Table (Linear Interpolation w/ Table)
bool getLookupTable(const char *filename, double **lt, long *size);
double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing);
double lookupValue4D(const double *table, const long &size, const double &omega12, double t1, double t2);

//Quicksort Algorithm
void quicksort(Node &nodes, const int &dim, const Manifold &manifold, int low, int high);
void quicksort(uint64_t *edges, int low, int high);
void swap(Node &nodes, const int &dim, const Manifold &manifold, const int i, const int j);
void swap(uint64_t *edges, const int i, const int j);
void swap(const int * const *& list0, const int * const *& list1, int &idx0, int &idx1, int &max0, int &max1);

//Bisection Algorithm
bool bisection(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3);

//Causal Relation Identification Algorithm
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx);

//Breadth First Search Algorithm
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements);

//Array Intersection Algorithm
void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many);

//Format Partial Adjacency Matrix Data
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size);
void readEdges(uint64_t * const &edges, const bool * const h_edges, bool * const core_edge_exists, int * const &g_idx, const unsigned int &core_limit, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y);

//Prefix Sum Algorithm
void scan(const int * const k_in, const int * const k_out, int * const &past_edge_pointers, int * const &future_edge_pointers, const int &N_tar);

//Print Statements for MPI
int printf_mpi(int rank, const char * format, ...);

//Check for MPI Errors
bool checkMpiErrors(CausetMPI &cmpi);

#endif
