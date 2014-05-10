#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

//Primary Newton-Raphson Functions
double solveZeta(NewtonProperties *np);
double solveTau0(NewtonProperties *np);
double solveT(NewtonProperties *np);
double solveTau(NewtonProperties *np);
double solvePhi(NewtonProperties *np);

//Secondary Newton-Raphson Functions
double eta02D(double &x, int &N_tar, float &k_tar);
double eta0Prime2D(double &x);

double zeta4D(double &x, int &N_tar, float &k_tar);
double zetaPrime4D(double &x);

double tau0(double &x, int &N_tar, double &alpha, double &delta);
double tau0Prime(double &x);

double t4D(double &x, double &zeta, double &a, double rval);
double tPrime4D(double &x, double &zeta, double &a);

double tau(double &x, double &tau0, double rval);
double tauPrime(double &x, double &tau0);

double phi4D(double &x, double rval);
double phiPrime4D(double &x);

//De Sitter Spatial Lengths
float X1(float &phi);
float X2(float &phi, float &chi);
float X3(float &phi, float &chi, float &theta);
float X4(float &phi, float &chi, float &theta);

//Temporal Transformations
float etaToT(float eta, double a);
float tToEta(float t, double a);

#endif
