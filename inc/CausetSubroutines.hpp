#ifndef CAUSET_SUBROUTINES_HPP_
#define CAUSET_SUBROUTINES_HPP_

#include <GL/freeglut.h>

#include "Causet.hpp"
#include "GPUSubroutines.hpp"
#include "CausetOperations.hpp"

//Primary Causet Subroutines
bool initializeNetwork(Network *network);
bool createNetwork(Network *network);
bool generateNodes(Network *network, bool &use_gpu);
bool linkNodes(Network *network, bool &use_gpu);
bool displayNetwork(Node *nodes, unsigned int *links, int argc, char **argv);
bool printNetwork(Network network);
bool destroyNetwork(Network *network);

//Quicksort Algorithm
void quicksort(Node *nodes, double a, int low, int high);
void swap(Node *n, Node *m);

//Newton-Raphson Algorithm
void newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed);

//Display Function for OpenGL Instructions
void display();

#endif
