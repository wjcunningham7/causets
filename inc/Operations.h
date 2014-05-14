#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

//Primary Newton-Raphson Functions
double solveZeta(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);
double solveTau0(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);
double solveT(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);
double solveTau(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);
double solvePhi(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6);

//Secondary Newton-Raphson Functions
double eta02D(const double &x, const int &N_tar, const float &k_tar);
double eta0Prime2D(const double &x);

double zeta4D(const double &x, const int &N_tar, const float &k_tar);
double zetaPrime4D(const double &x);

double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta);
double tau0Prime(const double &x);

double t4D(const double &x, const double &zeta, const double &a, const double &rval);
double tPrime4D(const double &x, const double &zeta, const double &a);

double tau(const double &x, const double &tau0, const double &rval);
double tauPrime(const double &x, const double &tau0);

double phi4D(const double &x, const double &rval);
double phiPrime4D(const double &x);

//Math Function for 2F1
double _2F1_z(const double &tau);

//De Sitter Spatial Lengths
float X1(const float &phi);
float X2(const float &phi, const float &chi);
float X3(const float &phi, const float &chi, const float &theta);
float X4(const float &phi, const float &chi, const float &theta);

//Temporal Transformations
float etaToT(const float eta, const double a);
float tToEta(const float t, const double a);
float tauToEtaUniverse(const float &tau);

#endif
