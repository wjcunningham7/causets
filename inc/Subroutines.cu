#ifndef SUBROUTINES_CU_
#define SUBROUTINES_CU_

#include "Operations.cu"

//Quicksort Algorithm
void quicksort(Node *nodes, int low, int high);
void swap(Node *n, Node *m);

//Newton-Raphson Algorithm
void newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed);

//Sort nodes temporally by tau coordinate
void quicksort(Node *nodes, int low, int high)
{
	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) / 2;
		swap(&nodes[low], &nodes[k]);
		key = nodes[low].tau;
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (nodes[i].tau <= key))
				i++;
			while ((j >= low) && (nodes[j].tau > key))
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
	Node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}

//Newton-Raphson Method
//Solves Transcendental Equations
void newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed)
{
	double x1;
	double res = 1.0;

	int iter = 0;
	while (fabs(res) > np->tol && iter < np->max) {
		res = (*solve)(np);
		//printf("res: %E\n", res);
		if (res != res) {
			printf("NaN Error in Newton-Raphson\n");
			exit(0);
		}

		x1 = np->x + res;
		//printf("x1: %E\n", x1);

		np->x = x1;
		iter++;
	}

	//printf("Newton-Raphson Results:\n");
	//printf("Tolerance: %E\n", np->tol);
	//printf("%d of %d iterations performed.\n", iter, np->max);
	//printf("Residual: %E\n", res);
	//printf("Solution: %E\n", np->x);
}

#endif
