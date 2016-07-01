#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

//Linear Interpolation using Lookup Table
bool getLookupTable(const char *filename, double **lt, long *size)
{
	#if DEBUG
	assert (filename != NULL);
	assert (lt != NULL);
	assert (size != NULL);
	#endif

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

//Lookup value in table of (x, y) coordinates -> 2D parameter space
double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing)
{
	#if DEBUG
	assert (table != NULL);
	assert (size > 0);
	assert ((x == NULL) ^ (y == NULL));
	#endif
	
	//Identify which is being calculated
	bool first = (x == NULL);
	//Identify input value
	double input = first ? *y : *x;
	double output = 0.0;
	int t_idx = (int)(!first);
	int i;

	try {
		//Identify Value in Table
		//Assumes values are written (y, x)
		for (i = (int)(!first); i < size / (int)sizeof(double); i += 2) {
			if ((increasing && table[i] >= input) || (!increasing && table[i] <= input)) {
				t_idx = i;
				break;
			}
		}

		//Check if Table is Insufficient
		if (t_idx == (int)(!first) && input != table[i]) {
			//printf("%f\n", input);
			throw CausetException("Values from lookup table do not include requested input.  Recreate table or change input.\n");
		}

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

//Lookup value in table of (t1, t2, omega12, lambda) coordinates -> 4D parameter space
//Used for geodesic distance calculations
//Returns the transcendental integration parameter 'lambda'
double lookupValue4D(const double *table, const long &size, const double &omega12, double t1, double t2)
{
	#if DEBUG
	assert (table != NULL);
	assert (size > 0);
	assert (omega12 >= 0.0);
	assert (t1 >= 0.0);
	assert (t2 >= 0.0);
	#endif

	if (t2 < t1) {
		double temp = t1;
		t1 = t2;
		t2 = temp;
	}

	double lambda = 0.0;
	double tol = 1e-2;
	int tau_step, lambda_step, step;
	int counter;
	int i;

	try {
		//The first two table elements should be zero
		if (table[0] != 0.0 || table[1] != 0.0)
			throw CausetException("Corrupted lookup table!\n");

		tau_step = table[2];
		lambda_step = table[3];
		step = 4 * tau_step * lambda_step;
		counter = 0;

		//Identify Value in Table
		//Assumes values are written (tau1, tau2, omega12, lambda)
		for (i = 4; i < size / (int)sizeof(double); i += step) {
			counter++;

			if (step == 4 * tau_step * lambda_step && table[i] > t1) {
				i -= (step - 1);
				step = 4 * lambda_step;
				i -= step;
				counter = 0;
			} else if (step == 4 * lambda_step && table[i] > t2) {
				i -= (step - 1);
				step = 4;
				i -= step;
				counter = 0;
			} else if (step == 4 && ABS(table[i] - omega12, STL) / omega12 < tol && table[i] != 0.0) {
				i -= step;
				step = 1;
			} else if (step == 1) {
				lambda = table[i];
				break;
			}

			if ((step == 4 * tau_step * lambda_step && counter == tau_step) ||
			    (step == 4 * lambda_step && counter == tau_step) ||
			    (step == 4 && counter == lambda_step))
				break;
		}

		//Perhaps do some linear interpolation here?

		//If no value found
		if (lambda == 0.0) {
			if (step == 4 * tau_step * lambda_step)
				throw CausetException("tau1 value not found in geodesic lookup table.\n");
			else if (step == 4 * lambda_step)
				throw CausetException("tau2 value not found in geodesic lookup table.\n");
			else if (step == 4)
				throw CausetException("omega12 value not found in geodesic lookup table.\n");
			else if (step == 1)
				throw std::exception();
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		lambda = std::numeric_limits<double>::quiet_NaN();
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		lambda = std::numeric_limits<double>::quiet_NaN();
	}

	return lambda;
}

//Sort nodes temporally
//O(N*log(N)) Efficiency
void quicksort(Node &nodes, const unsigned int &spacetime, int low, int high)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW | HYPERBOLIC));
	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);
	#endif

	int i, j, k;
	float key = 0.0;
	#if EMBED_NODES
	float *& time = get_stdim(spacetime) == 2 ? nodes.crd->x() : nodes.crd->v();
	#else
	float *& time = get_stdim(spacetime) == 2 ? nodes.crd->x() : nodes.crd->w();
	#endif

	if (low < high) {
		k = (low + high) >> 1;
		swap(nodes, spacetime, low, k);
		key = time[low];
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (time[i] <= key))
				i++;
			while ((j >= low) && (time[j] > key))
				j--;
			if (i < j)
				swap(nodes, spacetime, i, j);
		}

		swap(nodes, spacetime, low, j);
		quicksort(nodes, spacetime, low, j - 1);
		quicksort(nodes, spacetime, j + 1, high);
	}
}

//Sort edge list
//O(N*log(N)) Efficiency
void quicksort(uint64_t *edges, int64_t low, int64_t high)
{
	#if DEBUG
	assert (edges != NULL);
	#endif

	int64_t i, j, k;
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
void swap(Node &nodes, const unsigned int &spacetime, const int i, const int j)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW | HYPERBOLIC));
	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);
	#endif

	#if EMBED_NODES
	if (get_stdim(spacetime) == 2) {
		float3 hc = nodes.crd->getFloat3(i);
		nodes.crd->setFloat3(nodes.crd->getFloat3(j), i);
		nodes.crd->setFloat3(hc, j);
	} else if (get_stdim(spacetime) == 4) {
		float5 sc = nodes.crd->getFloat5(i);
		nodes.crd->setFloat5(nodes.crd->getFloat5(j), i);
		nodes.crd->setFloat5(sc, j);
	}
	#else
	if (get_stdim(spacetime) == 2) {
		float2 hc = nodes.crd->getFloat2(i);
		nodes.crd->setFloat2(nodes.crd->getFloat2(j), i);
		nodes.crd->setFloat2(hc, j);
	} else if (get_stdim(spacetime) == 4) {
		float4 sc = nodes.crd->getFloat4(i);
		nodes.crd->setFloat4(nodes.crd->getFloat4(j), i);
		nodes.crd->setFloat4(sc, j);
	}
	#endif

	if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
		float tau = nodes.id.tau[i];
		nodes.id.tau[i] = nodes.id.tau[j];
		nodes.id.tau[j] = tau;
	} else if (get_manifold(spacetime) & HYPERBOLIC) {
		int AS = nodes.id.AS[i];
		nodes.id.AS[i] = nodes.id.AS[j];
		nodes.id.AS[j] = AS;
	}
}

//Exchange two edges
void swap(uint64_t *edges, const int64_t i, const int64_t j)
{
	#if DEBUG
	assert (edges != NULL);
	assert (i >= 0);
	assert (j >= 0);
	#endif

	uint64_t tmp = edges[i];
	edges[i] = edges[j];
	edges[j] = tmp;
}

//Exchange references to two lists
//as well as related indices (used in causet_intersection)
void swap(const int * const *& list0, const int * const *& list1, int64_t &idx0, int64_t &idx1, int64_t &max0, int64_t &max1)
{
	#if DEBUG
	assert (idx0 >= 0);
	assert (idx1 >= 0);
	assert (max0 >= 0);
	assert (max1 >= 0);
	#endif

	const int * const * tmp_list = list0;
	list0 = list1;
	list1 = tmp_list;

	//Bitwise swaps
	idx0 ^= idx1;
	idx1 ^= idx0;
	idx0 ^= idx1;

	max0 ^= max1;
	max1 ^= max0;
	max0 ^= max1;
}

//Cyclesort
//Comparison sort which performs minimum swaps
//This algorithm returns the number of writes, and the sequence
//of swaps if a non-null vector is passed as the third argument
//NOTE: It is important the integers in 'elements' are unique for this to work
//O(N^2) Efficiency
void cyclesort(unsigned int &writes, std::vector<unsigned int> elements, std::vector<std::pair<int,int> > *swaps)
{
	unsigned int it, p;
	size_t len = elements.size();
	writes = 0;
	for (size_t j = 0; j < len - 1; j++) {
		it = elements[j];
		p = j;

		for (unsigned int k = j + 1; k < len; k++)
			if (elements[k] < it)
				p++;

		if (j == p) continue;

		//Swap
		while (elements[p] == it)
			p++;
		it ^= elements[p];
		elements[p] ^= it;
		it ^= elements[p];
		writes += 2;
		if (swaps != NULL) {
			swaps->push_back(std::make_pair(p, -1));
			swaps->push_back(std::make_pair(j, p));
		}

		while (j != p) {
			p = j;
			for (size_t k = j + 1; k < len; k++)
				if (elements[k] < it)
					p++;
			
			while (elements[p] == it)
				p++;
			it ^= elements[p];
			elements[p] ^= it;
			it ^= elements[p];
			writes++;
			if (swaps != NULL)
				swaps->push_back(std::make_pair(-1, p));
		}
	}
}

//Bisection Method
//Use when Newton-Raphson fails
bool bisection(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (solve != NULL);
	#endif

	double res = 1.0;
	double a = lower;
	double b = upper;
	int iter = 0;

	try {
		if (b <= a)
			throw CausetException("Invalid Bounds in Bisection!\n");

		//Initial test point
		*x = (b + a) / 2;
		while (ABS(res, STL) > tol && iter < max_iter) {
			//Residual Value
			res = (*solve)(*x, p1, p2, p3);
			//printf("res:   %.16e\n\n", res);
			//printf("x: %.16e\n\n", *x);

			//Check for NaN
			if (res != res)
				throw CausetException("NaN Error in Bisection!\n");

			//Change bounds
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

			//New test point
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
bool newton(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (solve != NULL);
	#endif

	double res = 1.0;
	double x1;
	int iter = 0;

	try {
		while (ABS(res, STL) > tol && iter < max_iter) {
			//Residual Value
			res = (*solve)(*x, p1, p2, p3);
			//printf("res: %E\n", res);

			//Check for NaN
			if (res != res)
				throw CausetException("NaN Error in Newton-Raphson\n");
	
			//New test value
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
//O(<k>) Efficiency for Adjacency List
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx)
{
	#if DEBUG
	//No null pointers
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);

	//Parameters in correct ranges
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	assert (past_idx != future_idx);

	assert (!(future_edge_row_start[past_idx] == -1 && nodes.k_out[past_idx] > 0));
	assert (!(future_edge_row_start[past_idx] != -1 && nodes.k_out[past_idx] == 0));
	#endif

	int core_limit = static_cast<int>(core_edge_fraction * N_tar);
	int i;

	//Make sure past_idx < future_idx
	if (past_idx > future_idx) {
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	//Check if the adjacency matrix can be used
	if (past_idx < core_limit && future_idx < core_limit)
		return (bool)adj[past_idx].read(future_idx);
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

//Returns true if two nodes are causally connected
//Note: past_idx must be less than future_idx
//O(1) Efficiency
bool nodesAreConnected_v2(const Bitvector &adj, const int &N_tar, int past_idx, int future_idx)
{
	#if DEBUG
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	//assert (past_idx != future_idx);
	#endif

	return (bool)adj[past_idx].read(future_idx);
}

//Breadth First Search
//O(N+E) Efficiency
void bfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements)
{
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (nodes.cc_id != NULL);
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (index >= 0);
	assert (id >= 0);
	assert (elements >= 0);
	#endif

	int64_t ps = edges.past_edge_row_start[index];
	int64_t fs = edges.future_edge_row_start[index];
	int i;

	nodes.cc_id[index] = id;
	elements++;

	//Move to past nodes
	for (i = 0; i < nodes.k_in[index]; i++)
		if (!nodes.cc_id[edges.past_edges[ps+i]])
			bfsearch(nodes, edges, edges.past_edges[ps+i], id, elements);

	//Move to future nodes
	for (i = 0; i < nodes.k_out[index]; i++)
		if (!nodes.cc_id[edges.future_edges[fs+i]])
			bfsearch(nodes, edges, edges.future_edges[fs+i], id, elements);
}

//Breadth First Search
//Uses adjacency matrix only
//O(N+E) Efficiency
void bfsearch_v2(const Node &nodes, const Bitvector &adj, const int &N_tar, const int index, const int id, int &elements)
{
	#if DEBUG
	assert (nodes.cc_id != NULL);
	assert (N_tar >= 0);
	assert (index >= 0 && index < N_tar);
	assert (id >= 0 && id <= N_tar / 2);
	assert (elements >= 0 && elements < N_tar);
	#endif

	nodes.cc_id[index] = id;
	elements++;

	int i;
	for (i = 0; i < N_tar; i++)
		if (adj[index].read(i) && !nodes.cc_id[i])
			bfsearch_v2(nodes, adj, N_tar, i, id, elements);
}

//Modification of v1 which eliminates swaps
//O(k*log(k)) Efficiency
void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int64_t &pstart, const int64_t &fstart, bool &too_many)
{
	#if DEBUG
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (k_i >= 0);
	assert (k_o >= 0);
	assert (!(k_i == 0 && k_o == 0));
	assert (max_cardinality > 1);
	assert (pstart >= 0);
	assert (fstart >= 0);
	#endif

	if (k_i == 1 || k_o == 1) {
		elements = 0;
		return;
	}

	int64_t idx0 = pstart;	//Index of past neighbors of 'future element j'
	int64_t idx1 = fstart;	//Index of future neighbors of 'past element i'
	int64_t max0 = idx0 + k_i;
	int64_t max1 = idx1 + k_o;

	while (idx0 < max0 && idx1 < max1 && !too_many) {
		if (past_edges[idx0] > future_edges[idx1])
			idx1++;
		else if (past_edges[idx0] < future_edges[idx1])
			idx0++;
		else {
			elements++;

			if (elements >= max_cardinality - 1) {
				too_many = true;
				break;
			}

			idx0++;
			idx1++;
		}
	}
}

//Intersection of Sorted Lists
//Used to find the cardinality of an interval
//O(k*log(k)) Efficiency
void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int64_t &pstart, const int64_t &fstart, bool &too_many)
{
	#if DEBUG
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (k_i >= 0);
	assert (k_o >= 0);
	assert (!(k_i == 0 && k_o == 0));
	assert (max_cardinality > 1);
	assert (pstart >= 0);
	assert (fstart >= 0);
	#endif

	int64_t idx0 = pstart;
	int64_t idx1 = fstart;
	int64_t max0 = idx0 + k_i;
	int64_t max1 = idx1 + k_o;

	if (k_i == 1 || k_o == 1) {
		elements = 0;
		return;
	}

	//Pointers are used here so that 'primary' and 'secondary'
	//can be switched as needed.  References are static, so they
	//cannot be used.  The 'const' specifiers are kept since the
	//edge list values and their locations in memory should
	//not be modified in this algorithm.

	const int * const * primary = &past_edges;
	const int * const * secondary = &future_edges;

	while (idx0 < max0 && idx1 < max1) {
		if ((*secondary)[idx1] > (*primary)[idx0])
			swap(primary, secondary, idx0, idx1, max0, max1);

		while (idx1 < max1 && (*secondary)[idx1] < (*primary)[idx0])
			idx1++;

		if (idx1 == max1)
			break;

		if ((*primary)[idx0] == (*secondary)[idx1]) {
			elements++;
			if (elements >= max_cardinality - 1) {
				too_many = true;
				return;
			}
			idx0++;
			idx1++;
		}
	}
}

//Data formatting used when reading the degree
//sequences found on the GPU
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size)
{
	#if DEBUG
	assert (degrees != NULL);
	assert (h_k != NULL);
	#endif

	unsigned int i;
	for (i = 0; i < size; i++)
		degrees[offset+i] += h_k[i];
}

//Data formatting used when reading output of
//the adjacency list created by the GPU
void readEdges(uint64_t * const &edges, const bool * const h_edges, Bitvector &adj, int64_t * const &g_idx, const unsigned int &core_limit_row, const unsigned int &core_limit_col, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y, const bool &use_bit, const bool &use_mpi)
{
	#if DEBUG
	if (!use_bit)
		assert (edges != NULL);
	assert (h_edges != NULL);
	assert (g_idx != NULL);
	assert (*g_idx >= 0);
	assert (x >= 0);
	assert (y >= 0);
	//assert (x <= y);
	#endif

	//printf("x: %d\tsize0: %zd\n", x, size0);
	//printf("I have a bitvector of length %zd\n", adj.size());

	unsigned int i, j;
	for (i = 0; i < size0; i++) {
		for (j = 0; j < size1; j++) {
			if (h_edges[i*mthread_size+j] && (use_bit || g_idx[0] < (int64_t)d_edges_size)) {
				if (!use_bit)
					edges[g_idx[0]++] = (static_cast<uint64_t>(x*mthread_size+i)) << 32 | (static_cast<uint64_t>(y*mthread_size+j));
				else
					g_idx[0]++;
				if (x*mthread_size+i < core_limit_row && y*mthread_size+j < core_limit_col) {
					adj[x*mthread_size+i].set(y*mthread_size+j);
					if (!use_mpi)
						adj[y*mthread_size+j].set(x*mthread_size+i);
				}
			}
		}
	}
}

//Remake adjacency sub-matrix using 'l' rows, beginning at row 'i'
void remakeAdjMatrix(bool * const adj0, bool * const adj1, const int * const k_in, const int * const k_out, const int * const past_edges, const int * const future_edges, const int64_t * const past_edge_row_start, const int64_t * const future_edge_row_start, int * const idx_buf0, int * const idx_buf1, const int &N_tar, const int &i, const int &j, const int64_t &l)
{
	#if DEBUG
	assert (adj0 != NULL);
	assert (adj1 != NULL);
	assert (k_in != NULL);
	assert (k_out != NULL);
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (past_edge_row_start != NULL);
	assert (future_edge_row_start != NULL);
	assert (idx_buf0 != NULL);
	assert (idx_buf1 != NULL);
	assert (N_tar > 0);
	assert (i >= 0);
	assert (j >= 0);
	assert (l > 0);
	#endif

	//Map tile indices to global indices
	for (int m = 0; m < l; m++) {
		int M = m + i * l;
		for (int n = 0; n < l; n++) {
			int N = n + j * l;
			if (!N)
				continue;

			//Use triangular mapping
			int do_map = M >= N;
			if (N < N_tar >> 1) {
				M = M + do_map * ((((N_tar >> 1) - M) << 1) - 1);
				N = N + do_map * (((N_tar >> 1) - N) << 1);
			}

			idx_buf0[m] = M;
			idx_buf1[n] = N;
		}
	}

	//Fill Adjacency Submatrix 0
	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 1)
	#endif
	for (int m = 0; m < l; m++) {
		int M = idx_buf0[m];
		int element;

		//Past Neighbors
		int64_t start = past_edge_row_start[M];
		for (int p = 0; p < k_in[M]; p++) {
			element = past_edges[start+p];
			adj0[m*N_tar+element] = true;
		}

		//Future Neighbors
		start = future_edge_row_start[M];
		for (int p = 0; p < k_out[M]; p++) {
			element = future_edges[start+p];
			adj0[m*N_tar+element] = true;
		}
	}

	//Fill Adjacency Submatrix 1
	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 1)
	#endif
	for (int n = 0; n < l; n++) {
		int N = idx_buf1[n];
		int element;

		if (!N)
			continue;

		//Past Neighbors
		int64_t start = past_edge_row_start[N];
		for (int p = 0; p < k_in[N]; p++) {
			element = past_edges[start+p];
			adj1[n*N_tar+element] = true;
		}

		//Future Neighbors
		start = future_edge_row_start[N];
		for (int p = 0; p < k_out[N]; p++) {
			element = future_edges[start+p];
			adj1[n*N_tar+element] = true;
		}
	}
}

//Data formatting used when reading output of
//the interval matrix created by the GPU
void readIntervals(int * const cardinalities, const unsigned int * const N_ij, const int &l)
{
	#if DEBUG
	assert (cardinalities != NULL);
	assert (N_ij != NULL);
	assert (l > 0);
	#endif

	int i, j;
	for (i = 0; i < l; i++)
		for (j = 0; j < l; j++)
			cardinalities[N_ij[j*l+i]+1]++;
}

//Scanning algorithm used when decoding
//lists found using GPU algorithms
void scan(const int * const k_in, const int * const k_out, int64_t * const &past_edge_pointers, int64_t * const &future_edge_pointers, const int &N_tar)
{
	int64_t past_idx = 0, future_idx = 0;
	int i;

	for (i = 0; i < N_tar; i++) {
		if (k_in[i] != 0) {
			past_edge_pointers[i] = past_idx;
			past_idx += k_in[i];
		} else
			past_edge_pointers[i] = -1;

		if (k_out[i] != 0) {
			future_edge_pointers[i] = future_idx;
			future_idx += k_out[i];
		} else
			future_edge_pointers[i] = -1;
	}
}

//Debug Print Variadic Function
int printf_dbg(const char * format, ...)
{
	printf_mag();
	va_list argp;
	va_start(argp, format);
	vprintf(format, argp);
	va_end(argp);
	printf_std();
	fflush(stdout);

	return 0;
}

//MPI Print Variadic Function
//Allows only the master process to print to stdout
//If MPI is not enabled, rank == 0
int printf_mpi(int rank, const char * format, ...)
{
	int retval = 0;

	if (rank == 0) {
		va_list argp;
		va_start(argp, format);
		vprintf(format, argp);
		va_end(argp);
	}

	return retval;
}

//Update all processes regarding failure status
bool checkMpiErrors(CausetMPI &cmpi)
{
	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	if (!cmpi.rank)
		MPI_Reduce(MPI_IN_PLACE, &cmpi.fail, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&cmpi.fail, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cmpi.fail, cmpi.num_mpi_threads, MPI_INT, 0, MPI_COMM_WORLD);
	#endif

	return !!cmpi.fail;
}

//Enumerate permutations of the unique integers stored in 'elements'
//Permutations are saved as binary strings
//The goal is to find all permutations of 2 unlabeled elements in 'elements.size()/2' unlabeled bins
// > swapping two elements in the same bin is not a unique permutation
// > swapping two bins is not a unique permutation
// > swapping two elements in different bins is a unique permutation
void init_mpi_permutations(std::unordered_set<FastBitset> &permutations, std::vector<unsigned int> elements)
{
	#if DEBUG
	assert (elements.size() > 0);
	#endif

	//Ensure output is empty to begin
	permutations.clear();
	permutations.swap(permutations);

	//This is the length of the binary string
	uint64_t len = static_cast<uint64_t>(elements.size()) * (elements.size() - 1) >> 1;
	while (std::next_permutation(elements.begin(), elements.end())) {
		FastBitset fb(len);
		perm_to_binary(fb, elements);
		bool insert = true;

		//Check if a similar element has already been added
		//Unique binary strings will be completely orthogonal
		for (std::unordered_set<FastBitset>::iterator fb0 = permutations.begin(); fb0 != permutations.end(); fb0++)
			for (uint64_t i = 0; i < len; i++)
				if (fb.read(i) & fb0->read(i))
					insert = false;

		if (insert) permutations.insert(fb);
	}
}

//Enumerate all unique pairs, assuming the first element is smaller
//The ordered pairs are not included (e.g. (0,1) (2,3), etc.) 
//The variable 'nbuf' should be twice the number of computers used
//'Elements' is given as a sequence of natural numbers beginning at zero
void init_mpi_pairs(std::unordered_set<std::pair<int,int> > &pairs, const std::vector<unsigned int> elements)
{
	#if DEBUG
	assert (elements.size() > 0 && !(elements.size() % 2));
	#endif

	//Ensure output is empty
	pairs.clear();
	pairs.swap(pairs);

	for (size_t k = 0; k < elements.size(); k += 2) {
		//Elements are understood to be stored in pairs
		unsigned int i = elements[k];
		unsigned int j = elements[k+1];

		for (unsigned int m = 0; m < elements.size(); m++) {
			if (m == i || m == j) continue;
			pairs.insert(std::make_pair(std::min(i, m), std::max(i, m)));
			pairs.insert(std::make_pair(std::min(j, m), std::max(j, m)));
		}
	}

	//Remove the ordered pairs
	for (size_t k = 0; k < elements.size(); k += 2)
		pairs.erase(std::make_pair(k, k+1));
}

//Saves all permutations which are recognized as non-unique by
//the subroutine 'init_mpi_permutations'
void fill_mpi_similar(std::vector<std::vector<unsigned int> > &similar, std::vector<unsigned int> elements)
{
	#if DEBUG
	assert (elements.size() > 0);
	#endif

	//Ensure output is empty
	similar.clear();
	similar.swap(similar);

	//Translate initial elements to a binary string
	uint64_t len = static_cast<uint64_t>(elements.size()) * (elements.size() - 1) >> 1;
	FastBitset fb(len);
	perm_to_binary(fb, elements);
	similar.push_back(elements);

	//Compare each permutation's binary string to the original permutation's string
	while (std::next_permutation(elements.begin(), elements.end())) {
		FastBitset sim(len);
		perm_to_binary(sim, elements);
		//If their binary string is equal, they are the same permutation
		if (fb == sim)
			similar.push_back(elements);
	}
}

//Use the cyclesort algorithm to determine which permutation is "most similar" to
//the original, thereby minimizing the number of swaps done using MPI
void get_most_similar(std::vector<unsigned int> &sim, unsigned int &nsteps, const std::vector<std::vector<unsigned int> > candidates, const std::vector<unsigned int> elements)
{
	#if DEBUG
	assert (candidates.size() > 0);
	assert (elements.size() > 0);
	#endif

	unsigned int idx = 0;
	nsteps = 100000000;	//Taken to be integer-infinity

	//Check each candidate
	for (size_t i = 0; i < candidates.size(); i++) {
		size_t len = candidates[i].size();
		std::vector<unsigned int> c(len);

		//Create a local copy of the candidate, and relabel
		//it so when sorted, it will equal 'elements'
		c = candidates[i];
		relabel_vector(c, elements);
		
		unsigned int writes;
		//Perform cyclesort, record number of writes
		cyclesort(writes, c, NULL);
		if (writes < nsteps) {
			//Save minimum
			nsteps = writes;
			idx = i;
		}
	}

	sim = candidates[idx];
}

//Relabel the output vector so when sorted the elements will equal the input
//This allows a set of numbers to be 'sorted' to a permutation other than the truly sorted one
void relabel_vector(std::vector<unsigned int> &output, const std::vector<unsigned int> input)
{
	#if DEBUG
	assert (input.size() == output.size());
	#endif

	size_t len = input.size();
	std::vector<unsigned int> out(len);

	for (size_t i = 0; i < len; i++)
		for (size_t j = 0; j < len; j++)
			if (output[j] == input[i])
				//Save the index of the element rather than the element itself
				out[j] = i;

	output = out;
}

//Generate a binary string from a set of unique integers
//This is used to identify which permutations are equal
//Recall, we wish to identify all unique permutations of 2 unlabeled elements in '2N'
//unlabeled bins, where 'N' is the number of computers we have
void perm_to_binary(FastBitset &fb, const std::vector<unsigned int> perm)
{
	#if DEBUG
	assert (!(fb.size() % 2));
	assert (!(perm.size() % 2));
	#endif

	for (uint64_t k = 0; k < perm.size(); k += 2) {
		//Group numbers into pairs (2 unlabeled elements)
		unsigned int i = perm[k];
		unsigned int j = perm[k+1];

		//Swap them if the first is larger
		if (i > j) {
			i ^= j;
			j ^= i;
			i ^= j;
		}

		//This (i,j) is a row/column of a matrix
		//We wish to transform this into a 1-D index
		//identified with elements in an upper-triangular matrix
		unsigned int do_map = i >= perm.size() >> 1;
		i -= do_map * (((i - (perm.size() >> 1)) << 1) + 1);
		j -= do_map * ((j - (perm.size() >> 1)) << 1);

		//This transformed linear index is stored in 'm'
		unsigned int m = i * (perm.size() - 1) + j - 1;
		fb.set(m);
	}
}

//The inverse operation of perm_to_binary
void binary_to_perm(std::vector<unsigned int> &perm, const FastBitset &fb, const unsigned int len)
{
	#if DEBUG
	assert (!(perm.size() % 2));
	assert (!(fb.size() % 2));
	#endif

	//Transform each set bit to an (i,j) pair
	for (uint64_t k = 0; k < fb.size(); k++) {
		if (!fb.read(k)) continue;

		unsigned int i = k / (len - 1);
		unsigned int j = k % (len - 1) + 1;

		unsigned int do_map = i >= j;
		i += do_map * ((((len >> 1) - i) << 1) - 1);
		j += do_map * (((len >> 1) - j) << 1);

		//These (i,j) pairs are the pair associated with the permutation
		perm.push_back(i);
		perm.push_back(j);
	}
}

//When the adjacency matrix is broken across computers, local indices refer to those within
//the sub-matrix stored in a particular buffer. When these buffers are shuffled, this subroutine will
//return the global index, with respect to the whole unshuffled matrix, provided the current permutation
unsigned int loc_to_glob_idx(std::vector<unsigned int> perm, const unsigned int idx, const int N_tar, const int num_mpi_threads, const int rank)
{
	#if DEBUG
	assert (idx < (unsigned int)N_tar);
	assert (N_tar > 0);
	assert (num_mpi_threads > 1);
	assert (rank >= 0);
	#endif

	//The number of rows in a single buffer (two buffers per computer)
	int mpi_offset = N_tar / (num_mpi_threads << 1);
	//Index local to a single buffer, whereas idx spans two buffers
	int loc_idx = idx - (idx / mpi_offset) * mpi_offset;
	//The sequence index - i.e. which buffer loc_idx belongs to
	int seq_idx = perm[(rank << 1) + (idx / mpi_offset)];
	//The original global index, spanning [0, N_tar)
	int glob_idx = seq_idx * mpi_offset + loc_idx;

	return glob_idx;
}

#ifdef MPI_ENABLED
//Perform MPI trades across multiple computers
//When a swap is performed, the memory in one buffer is scattered to all other computers
//because the temporary storage is split across all computers. Then, the second buffer is moved to
//the first buffer. Finally, the temporary storage is moved back to the second buffer. The index
//used for "storage" is '-1'. The list of swaps needed is stored in 'swaps', and this variable
//is populated using the 'cyclesort' algorithm.
void mpi_swaps(const std::vector<std::pair<int,int> > swaps, Bitvector &adj, Bitvector &adj_buf, const int N_tar, const int num_mpi_threads, const int rank)
{
	#if DEBUG
	assert (swaps.size() > 0);
	assert (adj.size() > 0);
	assert (adj_buf.size() > 0);
	assert (N_tar > 0);
	assert (num_mpi_threads > 1);
	assert (rank >= 0);
	#endif

	//Number of rows per buffer
	int mpi_offset = N_tar / (num_mpi_threads << 1);
	//Number of rows per temporary buffer
	int cpy_offset = mpi_offset / num_mpi_threads;

	int loc_idx;
	int start, finish;
	MPI_Status status;

	//Perform all swaps requested
	for (size_t i = 0; i < swaps.size(); i++) {
		//The two swap indices
		int idx0 = std::get<0>(swaps[i]);
		int idx1 = std::get<1>(swaps[i]);

		MPI_Barrier(MPI_COMM_WORLD);
		//If this is a simple trade from one computer to a second, computers not involved
		//can continue to the next iteration and wait at the barrier
		if (idx0 != -1 && idx1 != -1 && idx0 >> 1 == idx1 >> 1 && (rank << 1) + (idx0 % 2) != idx0)
			continue;

		if (idx0 == -1) {	//Copy buffer to idx1
			//Distinguish between two local buffers
			int buf_offset = mpi_offset * (idx1 % 2);
			//Range of rows which will be copied
			start = rank * cpy_offset;
			finish = start + cpy_offset;
			//Iterate over all rows
			for (int j = start; j < finish; j++) {
				//Index internal to a temporary storage buffer
				loc_idx = j % cpy_offset;
				//All MPI processes have work to do
				for (int k = 0; k < num_mpi_threads; k++) {
					//Index of the row being addressed
					int adj_idx = buf_offset + k * cpy_offset + loc_idx;
					if ((rank << 1) + (idx1 % 2) == idx1) {	//Receiving data to buffer idx1
						if (k == rank)
							//Copy local buffer row 'loc_idx' to adj row 'adj_idx'
							memcpy(adj[adj_idx].getAddress(), adj_buf[loc_idx].getAddress(), sizeof(BlockType) * adj_buf[loc_idx].getNumBlocks());
						else
							//Receive foreign buffer in rank 'k' to adj row 'adj_idx'
							MPI_Recv(adj[adj_idx].getAddress(), adj[adj_idx].getNumBlocks(), BlockTypeMPI, k, 0, MPI_COMM_WORLD, &status);
					} else if ((k << 1) + (idx1 % 2) == idx1)	//Sending data to buffer idx1
						//Send local buffer row 'loc_idx' to foreign buffer in rank 'k'
						MPI_Send(adj_buf[loc_idx].getAddress(), adj_buf[loc_idx].getNumBlocks(), BlockTypeMPI, k, 0, MPI_COMM_WORLD);
				}
			}
		} else if (idx1 == -1) {	//Copy idx0 to buffer
			//Distinguish between two local buffers
			int buf_offset = mpi_offset * (idx0 % 2);
			//Range of rows which will be copied
			start = rank * cpy_offset;
			finish = start + cpy_offset;
			//Iterate over all rows
			for (int j = start; j < finish; j++) {
				//Index internal to a temporary storage buffer
				loc_idx = j % cpy_offset;
				//All MPI processes have work to do
				for (int k = 0; k < num_mpi_threads; k++) {
					//Index of the row being addressed
					int adj_idx = buf_offset + k * cpy_offset + loc_idx;
					if ((rank << 1) + (idx0 % 2) == idx0) {	//Sending data to buffer idx0
						if (k == rank)
							//Copy adj row 'adj_idx' to local buffer row 'loc_idx'
							memcpy(adj_buf[loc_idx].getAddress(), adj[adj_idx].getAddress(), sizeof(BlockType) * adj[adj_idx].getNumBlocks());
						else
							//Send adj row 'adj_idx' to foreign buffer in rank 'k'
							MPI_Send(adj[adj_idx].getAddress(), adj[adj_idx].getNumBlocks(), BlockTypeMPI, k, 0, MPI_COMM_WORLD);
					} else if ((k << 1) + (idx0 % 2) == idx0)	//Receiving data to buffer idx0
						//Receive foreign buffer in rank 'k' to local buffer row 'loc_idx'
						MPI_Recv(adj_buf[loc_idx].getAddress(), adj_buf[loc_idx].getNumBlocks(), BlockTypeMPI, k, 0, MPI_COMM_WORLD, &status);
				}
			}
		} else {	//Copy from idx0 to idx1
			//Distinguish between two local buffers
			int buf_offset0 = mpi_offset * (idx0 % 2);
			int buf_offset1 = mpi_offset * (idx1 % 2);
			//Range of rows which will be copied
			start = rank * mpi_offset;
			finish = start + mpi_offset;
			//Iterate over all rows
			for (int j = start; j < finish; j++) {
				//Index internal to a temporary storage buffer
				loc_idx = j % mpi_offset;
				if (idx0 >> 1 == idx1 >> 1)	//Sending data to buffer idx1, both on same computer
					//Copy adj row 'buf_offset0 + loc_idx' to adj row 'buf_offset1 + loc_idx'
					memcpy(adj[buf_offset1+loc_idx].getAddress(), adj[buf_offset0+loc_idx].getAddress(), sizeof(BlockType) * adj[buf_offset0+loc_idx].getNumBlocks());
				else if ((rank << 1) + (idx0 % 2) == idx0)	//Send data to buffer idx1
					//Send adj row 'buf_offset0 + loc_idx' to foreign rank 'idx1 / 2'
					MPI_Send(adj[buf_offset0+loc_idx].getAddress(), adj[buf_offset0+loc_idx].getNumBlocks(), BlockTypeMPI, idx1 / 2, 0, MPI_COMM_WORLD);
				else if ((rank << 1) + (idx1 % 2) == idx1)	//Receive data to buffer idx1
					//Receive adj row 'buf_offset1 + loc_idx' from foreign rank 'idx0 / 2'
					MPI_Recv(adj[buf_offset1+loc_idx].getAddress(), adj[buf_offset1+loc_idx].getNumBlocks(), BlockTypeMPI, idx0 / 2, 0, MPI_COMM_WORLD, &status);
			}
		}
	}
}

//Send a signal to all MPI nodes, indicating which action is requested
void sendSignal(const MPISignal signal, const int rank, const int num_mpi_threads)
{
	MPI_Request req;
	for (int i = 0; i < num_mpi_threads; i++) {
		//Don't send a signal to yourself, unless it's a lock request to the spinlock
		if (signal != REQUEST_LOCK && i == rank) continue;
		//Make sure it's an asynchronous call to avoid blocking
		MPI_Isend((void*)&signal, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
	}
}
#endif
