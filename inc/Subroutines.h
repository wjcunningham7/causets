#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Quicksort Algorithm
void quicksort(Node &nodes, int low, int high);
static void swap(Node &nodes, const int i, const int j);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

#endif
