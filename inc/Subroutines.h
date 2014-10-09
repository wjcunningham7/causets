#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Lookup Table (Linear Interpolation w/ Table)
bool getLookupTable(double **lt, const char *filename);
double lookupValue(const double *table, double *x, double *y, bool increasing);

//Quicksort Algorithm
void quicksort(Node &nodes, const int &dim, const Manifold &manifold, int low, int high);
void quicksort(uint64_t *edges, int low, int high);
static void swap(Node &nodes, const int &dim, const Manifold &manifold, const int i, const int j);
static void swap(uint64_t *edges, const int i, const int j);

//Bisection Algorithm
bool bisection(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * p6), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

//Edge Identification Algorithm
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx);

//Breadth First Search Algorithm
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements);

//Format Partial Adjacency Matrix Data
void readDegrees(int * const &degrees, const int * const h_k, const int &index, const size_t &offset_size);
void readEdges(uint64_t * const &edges, const bool * const h_edges, int * const &g_idx, const size_t &d_edges_size, const size_t &buffer_size, const int x, const int y);

#endif
