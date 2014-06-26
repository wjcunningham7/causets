#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Sort nodes temporally by tau coordinate
//O(N*log(N)) Efficiency
void quicksort(Node &nodes, int low, int high)
{
	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) / 2;
		swap(nodes, low, k);
		key = nodes.tau[low];
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (nodes.tau[i] <= key))
				i++;
			while ((j >= low) && (nodes.tau[j] > key))
				j--;
			if (i < j)
				swap(nodes, i, j);
		}

		swap(nodes, low, j);
		quicksort(nodes, low, j - 1);
		quicksort(nodes, j + 1, high);
	}
}

//Exchange two nodes
static void swap(Node &nodes, const int i, const int j)
{
	float4 sc = nodes.sc[i];
	float tau = nodes.tau[i];
	int k_in = nodes.k_in[i];
	int k_out = nodes.k_out[i];
	
	nodes.sc[i] = nodes.sc[j];
	nodes.tau[i] = nodes.tau[j];
	nodes.k_in[i] = nodes.k_in[j];
	nodes.k_out[i] = nodes.k_out[j];

	nodes.sc[j] = sc;
	nodes.tau[j] = tau;
	nodes.k_in[j] = k_in;
	nodes.k_out[j] = k_out;
}

//Newton-Raphson Method
//Solves Transcendental Equations
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * p6)
{
	if (DEBUG) assert (solve != NULL);

	double res = 1.0;
	double x1;
	int iter = 0;

	try {
		while (ABS(res, STL) > tol && iter < max_iter) {
			res = (*solve)(*x, p1, p2, p3, p4, p5, p6);
			//printf("res: %E\n", res);
			if (res != res)
				throw CausetException("NaN Error in Newton-Raphson\n");
	
			x1 = *x + res;
			//printf("x1: %E\n", x1);
	
			*x = x1;
			iter++;

			fflush(stdout);
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	//printf("Newton-Raphson Results:\n");
	//printf("Tolerance: %E\n", tol);
	//printf("%d of %d iterations performed.\n", iter, max);
	//printf("Residual: %E\n", res);
	//printf("Solution: %E\n", *x);
	//fflush(stdout);

	return true;
}
