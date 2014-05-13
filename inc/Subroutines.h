#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

#include "Causet.h"

//Quicksort Algorithm
void quicksort(Node *nodes, int low, int high);
void swap(Node *n, Node *m);

//Newton-Raphson Algorithm
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

#endif
