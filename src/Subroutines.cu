#ifndef SUBROUTINES_CU_
#define SUBROUTINES_CU_

#include "Subroutines.h"

//Sort nodes temporally by tau coordinate
//O(N*log(N)) Efficiency
void quicksort(Node *nodes, int low, int high)
{
	assert (nodes != NULL);
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
	assert (n != NULL);
	assert (m != NULL);

	Node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}

//Newton-Raphson Method
//Solves Transcendental Equations
bool newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed)
{
	assert (np != NULL);
	assert (solve != NULL);
	assert (seed != NULL);

	double x1;
	double res = 1.0;

	int iter = 0;
	try {
		while (fabs(res) > np->tol && iter < np->max) {
			res = (*solve)(np);
			//printf("res: %E\n", res);
			if (res != res)
				throw CausetException("NaN Error in Newton-Raphson\n");
	
			x1 = np->x + res;
			//printf("x1: %E\n", x1);
	
			np->x = x1;
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
	//printf("Tolerance: %E\n", np->tol);
	//printf("%d of %d iterations performed.\n", iter, np->max);
	//printf("Residual: %E\n", res);
	//printf("Solution: %E\n", np->x);

	return true;
}

#endif
