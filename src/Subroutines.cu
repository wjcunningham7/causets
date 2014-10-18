#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Linear Interpolation using Lookup Table
bool getLookupTable(const char *filename, double **lt, long *size)
{
	if (DEBUG) {
		assert (lt != NULL);
		assert (filename != NULL);
	}

	double *table;
	std::ifstream ltable(filename, std::ios::in | std::ios::binary | std::ios::ate);

	try {
		if (ltable.is_open()) {
			//Find size of file
			*size = ltable.tellg();

			if (*size == 0)
				throw CausetException("Lookup table file is empty!\n");

			//Allocate Memory for Buffer
			char *memblock = (char*)malloc(*size);
			if (memblock == NULL)
				throw std::bad_alloc();

			//Allocate Memory for Lookup Table
			table = (double*)malloc(*size);
			if (table == NULL)
				throw std::bad_alloc();

			//Read File
			ltable.seekg(0, std::ios::beg);
			ltable.read(memblock, *size);
			memcpy(table, memblock, *size);

			//Free Memory
			free(memblock);
			memblock = NULL;

			//Close Stream
			ltable.close();
		} else
			throw CausetException("Failed to open lookup table file!\n");

		//Return Table
		*lt = table;
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (size > 0);
		assert (x == NULL ^ y == NULL);
	}
	
	//Identify which is being calculated
	bool first = (x == NULL);
	//Identify input value
	double input = first ? *y : *x;
	double output = 0.0;
	int t_idx = 0;
	int i;

	try {
		//Identify Value in Table
		//Assumes values are written (y, x)
		for (i = (int)(!first); i < size / sizeof(double); i += 2) {
			if (increasing && table[i] > input || !increasing && table[i] < input) {
				t_idx = i;
				break;
			}
		}

		//Check if Table is Insufficient
		if (t_idx == (int)(!first))
			throw CausetException("Values from lookup table do not include requested input.  Recreate table or change input.\n");

		//Linear Interpolation
		if (first)
			output = table[t_idx-1] + (table[t_idx+1] - table[t_idx-1]) * (input - table[t_idx-2]) / (table[t_idx] - table[t_idx-2]);
		else
			output = table[t_idx-3] + (table[t_idx-1] - table[t_idx-3]) * (input - table[t_idx-2]) / (table[t_idx] - table[t_idx-2]);
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		output = std::numeric_limits<double>::quiet_NaN();
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		output = std::numeric_limits<double>::quiet_NaN();
	}

	return output;
}

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
			while ((j >= low) && ((dim == 3 ? nodes.c.sc[j].w : nodes.c.hc[j].x) > key))
				j--;
			if (i < j)
				swap(nodes, dim, manifold, i, j);
		}

		swap(nodes, dim, manifold, low, j);
		quicksort(nodes, dim, manifold, low, j - 1);
		quicksort(nodes, dim, manifold, j + 1, high);
	}
}

//Sort edge list
void quicksort(uint64_t *edges, int low, int high)
{
	if (DEBUG)
		assert (edges != NULL);

	int i, j, k;
	uint64_t key;

	if (low < high) {
		k = (low + high) >> 1;
		swap(edges, low, k);
		key = edges[low];
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (edges[i] <= key))
				i++;
			while ((j >= low) && (edges[j] > key))
				j--;
			if (i < j)
				swap(edges, i, j);
		}

		swap(edges, low, j);
		quicksort(edges, low, j - 1);
		quicksort(edges, j + 1, high);
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

//Exchange two edges
static void swap(uint64_t *edges, const int i, const int j)
{
	uint64_t tmp = edges[i];
	edges[i] = edges[j];
	edges[j] = tmp;
}

//Bisection Method
//Use when Newton-Raphson fails
bool bisection(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) assert (solve != NULL);

	double res = 1.0;
	double a = lower;
	double b = upper;
	int iter = 0;

	try {
		if (b <= a)
			throw CausetException("Invalid Bounds in Bisection!\n");

		*x = (b + a) / 2;
		while (ABS(res, STL) > tol && iter < max_iter) {
			//printf("lower: %.16e\n", a);
			//printf("upper: %.16e\n", b);
			res = (*solve)(*x, p1, p2, p3, p4, p5, p6);
			//printf("res:   %.16e\n\n", res);
			if (res != res)
				throw CausetException("NaN Error in Bisection!\n");
			if (increasing) {
				if (res > 0)
					b = *x;
				else
					a = *x;
			} else {
				if (res > 0)
					a = *x;
				else
					b = *x;
			}

			*x = (b + a) / 2;
			iter++;
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}
	
	//printf("Bisection Results:\n");
	//printf("Tolerance: %E\n", tol);
	//printf("%d of %d iterations performed.\n", iter, max_iter);
	//printf("Residual: %E\n", y - res);
	//printf("Solution: %E\n", *x);
	//fflush(stdout);

	return true;
}

//Newton-Raphson Method
//Solves Transcendental Equations
bool newton(double (*solve)(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6), double *x, const int max_iter, const double tol, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
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
	//printf("%d of %d iterations performed.\n", iter, max_iter);
	//printf("Residual: %E\n", res);
	//printf("Solution: %E\n", *x);
	//fflush(stdout);

	return true;
}

//Returns true if two nodes are causally connected
//Note: past_idx must be less than future_idx
//O(1) Efficiency for Adjacency Matrix
//O(k) Efficiency for Adjacency List
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx)
{
	if (DEBUG) {
		//No null pointers
		assert (future_edges != NULL);
		assert (future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Parameters in correct ranges
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (past_idx >= 0 && past_idx < N_tar);
		assert (future_idx >= 0 && future_idx < N_tar);
		assert (past_idx < future_idx);
	}

	int core_limit = static_cast<int>((core_edge_fraction * N_tar));
	int i;

	//Check if the adjacency matrix can be used
	if (past_idx < core_limit && future_idx < core_limit)
		return (core_edge_exists[(past_idx * core_limit) + future_idx]);
	//Check if past node is not connected to anything
	else if (future_edge_row_start[past_idx] == -1)
		return false;
	//Check adjacency list
	else
		for (i = 0; i < nodes.k_out[past_idx]; i++)
			if (future_edges[future_edge_row_start[past_idx] + i] == future_idx)
				return true;

	return false;
}

//Breadth First Search
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements)
{
	//printf("IDX: %d\n", index);
	int ps = edges.past_edge_row_start[index];
	int fs = edges.future_edge_row_start[index];
	int i;

	//printf("PS: %d\n", ps);
	//printf("FS: %d\n", fs);

	nodes.cc_id[index] = id;
	elements++;

	//printf("IN\n");
	for (i = 0; i < nodes.k_in[index]; i++)
		if (!nodes.cc_id[edges.past_edges[ps+i]])
			bfsearch(nodes, edges, edges.past_edges[ps+i], id, elements);

	//printf("OUT\n");
	for (i = 0; i < nodes.k_out[index]; i++)
		if (!nodes.cc_id[edges.future_edges[fs+i]])
			bfsearch(nodes, edges, edges.future_edges[fs+i], id, elements);
}

void readDegrees(int * const &degrees, const int * const h_k, const int &index, const size_t &offset_size)
{
	unsigned int i;
	for (i = 0; i < offset_size; i++)
		degrees[index*offset_size+i] += h_k[i];
}

void readEdges(uint64_t * const &edges, const bool * const h_edges, int * const &g_idx, const size_t &d_edges_size, const size_t &buffer_size, const int x, const int y)
{
	unsigned int i, j;
	for (i = 0; i < buffer_size; i++)
		for (j = 0; j < buffer_size; j++)
			if (h_edges[i*buffer_size+j] && g_idx[0] < d_edges_size)
				edges[++g_idx[0]] = ((uint64_t)(x*buffer_size+i)) << 32 | ((uint64_t)(y*buffer_size+j));
}
