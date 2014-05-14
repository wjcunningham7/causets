#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Sort nodes temporally by tau coordinate
//O(N*log(N)) Efficiency
void quicksort(Node *nodes, int low, int high)
{
	//No null pointers
	assert (nodes != NULL);

	//Values in correct ranges
	assert (low >= 0);
	assert (high >= 0);

	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) / 2;
		swap(&nodes[low], &nodes[k]);
		key = nodes[low].t;
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (nodes[i].t <= key))
				i++;
			while ((j >= low) && (nodes[j].t > key))
				j--;
			if (i < j)
				swap(&nodes[i], &nodes[j]);
		}

		swap(&nodes[low], &nodes[j]);
		quicksort(nodes, low, j - 1);
		quicksort(nodes, j + 1, high);
	}
}

//Exchange two nodes
void swap(Node *n, Node *m)
{
	//No null pointers
	assert (n != NULL);
	assert (m != NULL);

	Node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}

//Newton-Raphson Method
//Solves Transcendental Equations
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * p6)
{
	assert (solve != NULL);

	double res = 1.0;
	double x1;
	int iter = 0;

	try {
		while (ABS(res, 0) > tol && iter < max_iter) {
			res = (*solve)(*x, p1, p2, p3, p4, p5, p6);
			//printf("res: %E\n", res);
			if (res != res)
				throw CausetException("NaN Error in Newton-Raphson\n");
	
			x1 = *x + res;
			//printf("x1: %E\n", x1);
	
			*x = x1;
			iter++;
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

	return true;
}

//Numerical Integration
//Uses Midpoint Riemann Sum
bool integrate(float (*solve)(const float &x), float &x, const float &lower, const float &upper, const float &dx)
{
	//No null pointers
	assert (solve != NULL);

	//Values in correct ranges
	assert (lower < upper);
	assert (dx > 0.0);

	float i;
	x = 0.0;

	try {
		for (i = lower; i < upper; i += dx) {
			x += solve(i);
			if (x != x)
				throw CausetException("NaN Error in Integrate\n");
		}
		x *= dx;
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}
