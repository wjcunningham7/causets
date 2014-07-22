#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Quicksort Algorithm
void quicksort(Node &nodes, const int &dim, const Manifold &manifold, int low, int high);
void quicksort(uint64_t *edges, int low, int high);
static void swap(Node &nodes, const int &dim, const Manifold &manifold, const int i, const int j);
static void swap(uint64_t *edges, const int i, const int j);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

//Breadth First Search Algorithm
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements);

#endif
