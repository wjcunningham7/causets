#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Sort nodes temporally by tau coordinate
//O(N*log(N)) Efficiency
void quicksort(Node &nodes, const int &dim, const Manifold &manifold, int low, int high)
{
	if (DEBUG) {
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
	}

	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) >> 1;
		swap(nodes, dim, manifold, low, k);
		if (dim == 1)
			key = nodes.c.hc[low].x;
		else if (dim == 3)
			key = nodes.c.sc[low].w;
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && ((dim == 3 ? nodes.c.sc[i].w : nodes.c.hc[i].x) <= key))
				i++;
			while ((j >= low) && ((dim == 3 ? nodes.c.sc[j].w : nodes.c.hc[i].x) > key))
				j--;
			if (i < j)
				swap(nodes, dim, manifold, i, j);
		}

		swap(nodes, dim, manifold, low, j);
		quicksort(nodes, dim, manifold, low, j - 1);
		quicksort(nodes, dim, manifold, j + 1, high);
	}
}

//Exchange two nodes
static void swap(Node &nodes, const int &dim, const Manifold &manifold, const int i, const int j)
{
	if (DEBUG) {
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
	}

	if (dim == 1) {
		float2 hc = nodes.c.hc[i];
		nodes.c.hc[i] = nodes.c.hc[j];
		nodes.c.hc[j] = hc;
	} else if (dim == 3) {
		float4 sc = nodes.c.sc[i];
		nodes.c.sc[i] = nodes.c.sc[j];
		nodes.c.sc[j] = sc;
	}

	if (manifold == DE_SITTER) {
		float tau = nodes.id.tau[i];
		nodes.id.tau[i] = nodes.id.tau[j];
		nodes.id.tau[j] = tau;
	} else if (manifold == HYPERBOLIC) {
		int AS = nodes.id.AS[i];
		nodes.id.AS[i] = nodes.id.AS[j];
		nodes.id.AS[j] = AS;
	}
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

//Breadth First Search
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements)
{
	int ps = edges.past_edge_row_start[index];
	int fs = edges.future_edge_row_start[index];
	int i;

	nodes.cc_id[index] = id;
	elements++;

	for (i = 0; i < nodes.k_in[index]; i++)
		if (!nodes.cc_id[edges.past_edges[ps+i]])
			bfsearch(nodes, edges, edges.past_edges[ps+i], id, elements);

	for (i = 0; i < nodes.k_out[index]; i++)
		if (!nodes.cc_id[edges.future_edges[fs+i]])
			bfsearch(nodes, edges, edges.future_edges[fs+i], id, elements);
}
