#ifndef CAUSET_SUBROUTINES_HPP_
#define CAUSET_SUBROUTINES_HPP_

#include <GL/glut.h>

#include "Causet.hpp"
#include "GPUSubroutines.hpp"

//Core Subroutines
bool initializeNetwork(Network *network);
bool createNetwork(Network *network);
bool generateNodes(Network *network, float &eta0, bool &use_gpu);
bool linkNodes(Network *network, bool &use_gpu);
bool displayNetwork(Node *nodes, unsigned int *links, int argc, char **argv);
bool printNetwork(Network network);
bool destroyNetwork(Network *network);

//Additional Subroutines
float newton(float guess, unsigned int &max_iter, unsigned int &N, unsigned int &k, unsigned int &dim);
void quicksort(Node *nodes, int low, int high);
void swap(Node *n, Node *m);

inline float f2D(float &x, unsigned int &N, unsigned int &k);
inline float fprime2D(float &x);

inline float f4D(float &x);
inline float fprime4D(float &x);

inline float Z0(float &a, float &eta);
inline float Z1(float &a, float &eta, float &phi);
inline float Z2(float &a, float &eta, float &phi, float &chi);
inline float Z3(float &a, float &eta, float &phi, float &chi, float &theta);
inline float Z4(float &a, float &eta, float &phi, float &chi, float &theta);

void display();

#endif