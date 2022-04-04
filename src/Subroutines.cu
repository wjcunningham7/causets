/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "Subroutines.h"

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
	} catch (const CausetException &c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	} catch (const std::exception &e) {
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
	} catch (const CausetException &c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		output = std::numeric_limits<double>::quiet_NaN();
	} catch (const std::exception &e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		output = std::numeric_limits<double>::quiet_NaN();
	}

	return output;
}

//Sort nodes temporally
//O(N*log(N)) Efficiency
void quicksort(Node &nodes, const Spacetime &spacetime, int low, int high)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") || spacetime.stdimIs("5"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"));
	if (spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"))
		assert (spacetime.stdimIs("2"));
	#endif

	int i, j, k;
	float key = 0.0;
	#if EMBED_NODES
	float *& time = spacetime.stdimIs("2") ? nodes.crd->x() : nodes.crd->v();
	#else
	float *& time = (spacetime.stdimIs("2") || spacetime.stdimIs("3")) ? nodes.crd->x() : (spacetime.stdimIs("4") ? nodes.crd->w() : nodes.crd->v());
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

void quicksort(std::vector<unsigned> &U, std::vector<unsigned> &V, int64_t low, int64_t high)
{
	int64_t i, j, k;
	uint64_t key;

	if (low < high) {
		k = (low + high) >> 1;
		swap(U, V, low, k);
		key = U[low] + V[low];
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (U[i] + V[i] <= key))
				i++;
			while ((j >= low) && (U[j] + V[j] > key))
				j--;
			if (i < j)
				swap(U, V, i, j);
		}

		swap(U, V, low, j);
		quicksort(U, V, low, j - 1);
		quicksort(U, V, j + 1, high);
	}
}

//Exchange two nodes
void swap(Node &nodes, const Spacetime &spacetime, const int i, const int j)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") | spacetime.stdimIs("5"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"));
	if (spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"))
		assert (spacetime.stdimIs("2"));
	#endif

	#if EMBED_NODES
	if (spacetime.stdimIs("2")) {
		float3 hc = nodes.crd->getFloat3(i);
		nodes.crd->setFloat3(nodes.crd->getFloat3(j), i);
		nodes.crd->setFloat3(hc, j);
	} else if (spacetime.stdimIs("4")) {
		float5 sc = nodes.crd->getFloat5(i);
		nodes.crd->setFloat5(nodes.crd->getFloat5(j), i);
		nodes.crd->setFloat5(sc, j);
	}
	#else
	if (spacetime.stdimIs("2")) {
		float2 hc = nodes.crd->getFloat2(i);
		nodes.crd->setFloat2(nodes.crd->getFloat2(j), i);
		nodes.crd->setFloat2(hc, j);
	} else if (spacetime.stdimIs("3")) {
		float3 sc = nodes.crd->getFloat3(i);
		nodes.crd->setFloat3(nodes.crd->getFloat3(j), i);
		nodes.crd->setFloat3(sc, j);
	} else if (spacetime.stdimIs("4")) {
		float4 sc = nodes.crd->getFloat4(i);
		nodes.crd->setFloat4(nodes.crd->getFloat4(j), i);
		nodes.crd->setFloat4(sc, j);
	} else if (spacetime.stdimIs("5")) {
		float5 sc = nodes.crd->getFloat5(i);
		nodes.crd->setFloat5(nodes.crd->getFloat5(j), i);
		nodes.crd->setFloat5(sc, j);
	}
	#endif

	if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Polycone")) {
		float tau = nodes.id.tau[i];
		nodes.id.tau[i] = nodes.id.tau[j];
		nodes.id.tau[j] = tau;
	} else if (spacetime.manifoldIs("Hyperbolic")) {
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

void swap(std::vector<unsigned> &U, std::vector<unsigned> &V, int64_t i, int64_t j)
{
	std::swap(U[i], U[j]);
	std::swap(V[i], V[j]);
}

void ordered_labeling(std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> &Unew, std::vector<unsigned> &Vnew)
{
	if (U != Unew)
		std::copy(U.begin(), U.begin() + U.size(), Unew.begin());
	if (V != Vnew)
		std::copy(V.begin(), V.begin() + V.size(), Vnew.begin());
	quicksort(Unew, Vnew, 0, U.size() - 1);
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
bool bisection(double (*solve)(const double x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3)
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
	} catch (const CausetException &c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (const std::exception &e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}
	
	return true;
}

//Newton-Raphson Method
//Solves Transcendental Equations
bool newton(double (*solve)(const double x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double tol, const double * const p1, const float * const p2, const int * const p3)
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
	} catch (const CausetException &c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (const std::exception &e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

//Returns true if two nodes are causally connected
//Note: past_idx must be less than future_idx
//O(1) Efficiency for Adjacency Matrix
//O(<k>) Efficiency for Adjacency List
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N, const float &core_edge_fraction, int past_idx, int future_idx)
{
	#if DEBUG
	//No null pointers
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);

	//Parameters in correct ranges
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (past_idx >= 0 && past_idx < N);
	assert (future_idx >= 0 && future_idx < N);
	assert (past_idx != future_idx);

	assert (!(future_edge_row_start[past_idx] == -1 && nodes.k_out[past_idx] > 0));
	assert (!(future_edge_row_start[past_idx] != -1 && nodes.k_out[past_idx] == 0));
	#endif

	int core_limit = static_cast<int>(core_edge_fraction * N);
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
bool nodesAreConnected_v2(const Bitvector &adj, const int &N, int past_idx, int future_idx)
{
	#if DEBUG
	assert (past_idx >= 0 && past_idx < N);
	assert (future_idx >= 0 && future_idx < N);
	#endif

	return (bool)adj[past_idx].read(future_idx);
}

//Depth First Search
//O(N+E) Efficiency
void dfsearch(const Node &nodes, const Edge &edges, const int index, const int id, int &elements, int level)
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

	#if DEBUG
	if (!!nodes.k_in[index]) assert (ps >= 0);
	if (!!nodes.k_out[index]) assert (fs >= 0);
	#endif

	nodes.cc_id[index] = id;
	elements++;

	//Move to past nodes
	for (i = 0; i < nodes.k_in[index]; i++) {
		if (!nodes.cc_id[edges.past_edges[ps+i]])
			dfsearch(nodes, edges, edges.past_edges[ps+i], id, elements, level + 1);
	}

	//Move to future nodes
	for (i = 0; i < nodes.k_out[index]; i++) {
		if (!nodes.cc_id[edges.future_edges[fs+i]])
			dfsearch(nodes, edges, edges.future_edges[fs+i], id, elements, level + 1);
	}
}

//Depth First Search
//Uses adjacency matrix only
//O(N+E) Efficiency
void dfsearch_v2(const Node &nodes, const Bitvector &adj, const int &N, const int index, const int id, int &elements)
{
	#if DEBUG
	assert (nodes.cc_id != NULL);
	assert (N >= 0);
	assert (index >= 0 && index < N);
	assert (id >= 0 && id <= N / 2);
	assert (elements >= 0 && elements < N);
	#endif

	nodes.cc_id[index] = id;
	elements++;

	int i;
	for (i = 0; i < N; i++)
		if (adj[index].read(i) && !nodes.cc_id[i])
			dfsearch_v2(nodes, adj, N, i, id, elements);
}

//BFS Shortest Path
int shortestPath(const Node &nodes, const Edge &edges, const int &N, int * const distances, const int start, const int end)
{
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (nodes.cc_id != NULL);
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (N > 0);
	assert (distances != NULL);
	assert (start >= 0 && start < N);
	#endif

	static const bool SP_DEBUG = false;

	if (nodes.cc_id[start] != nodes.cc_id[end] || !nodes.cc_id[start] || !nodes.cc_id[end])
		return -1;

	if (SP_DEBUG) {
		printf("Finding shortest path between [%d] and [%d].\n", start, end);
		fflush(stdout);
	}
	
	std::deque<unsigned int> next;
	int shortest = -1;

	memset(distances, -1, sizeof(int) * N);
	distances[start] = 0;
	next.push_back(start);

	while (!!next.size()) {
		int loc_min = INT_MAX, loc_min_idx = -1;
		for (size_t i = 0; i < next.size(); i++) {
			if (distances[next[i]] >= 0 && distances[next[i]] < loc_min) {
				loc_min = distances[next[i]];
				loc_min_idx = i;
			}
		}

		#if DEBUG
		assert (loc_min_idx != -1);
		#endif

		int v = next[loc_min_idx];
		next.erase(next.begin() + loc_min_idx);
		if (SP_DEBUG) {
			printf("\nCurrent location: %d\n", v);
			printf("Looking at past neighbors:\n");
			fflush(stdout);
		}
		for (int i = 0; i < nodes.k_in[v]; i++) {
			int w = edges.past_edges[edges.past_edge_row_start[v]+i];
			if (distances[w] == -1) {
				if (SP_DEBUG) {
					printf(" > [%d]\n", w);
					fflush(stdout);
				}

				distances[w] = distances[v] + 1;
				if (w != end)
					next.push_back(w);
				else
					goto PathExit;
			}
		}

		if (SP_DEBUG) {
			printf("Looking at future neighbors:\n");
			fflush(stdout);
		}
		for (int i = 0; i < nodes.k_out[v]; i++) {
			int w = edges.future_edges[edges.future_edge_row_start[v]+i];
			if (distances[w] == -1) {
				if (SP_DEBUG) {
					printf(" > [%d]\n", w);
					fflush(stdout);
				}
				distances[w] = distances[v] + 1;
				if (w != end)
					next.push_back(w);
				else
					goto PathExit;
			}
		}
	}

	PathExit:
	shortest = distances[end];
	if (SP_DEBUG) {
		printf_dbg("shortest path: %d\n", shortest);
		fflush(stdout);
	}

	#if DEBUG
	assert (shortest != -1);
	#endif

	next.clear();
	next.swap(next);

	return shortest;
}

//Modification of v1 which eliminates swaps
//O(k*log(k)) Efficiency
void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int64_t &pstart, const int64_t &fstart)
{
	#if DEBUG
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (k_i >= 0);
	assert (k_o >= 0);
	assert (!(k_i == 0 && k_o == 0));
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

	while (idx0 < max0 && idx1 < max1) {
		if (past_edges[idx0] > future_edges[idx1])
			idx1++;
		else if (past_edges[idx0] < future_edges[idx1])
			idx0++;
		else {
			elements++;
			idx0++;
			idx1++;
		}
	}
}

//Intersection of Sorted Lists
//Used to find the cardinality of an interval
//O(k*log(k)) Efficiency
void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int64_t &pstart, const int64_t &fstart)
{
	#if DEBUG
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (k_i >= 0);
	assert (k_o >= 0);
	assert (!(k_i == 0 && k_o == 0));
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

//Scanning algorithm used when decoding
//lists found using GPU algorithms
void scan(const int * const k_in, const int * const k_out, int64_t * const &past_edge_pointers, int64_t * const &future_edge_pointers, const int &N)
{
	int64_t past_idx = 0, future_idx = 0;
	int i;

	for (i = 0; i < N; i++) {
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
/*int printf_dbg(const char * format, ...)
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
}*/

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

void remove_bad_perms(std::unordered_set<FastBitset> &permutations, std::unordered_set<std::pair<int,int> > pairs)
{
	#if DEBUG
	assert (permutations.size() > 0);
	assert (pairs.size() > 0);
	#endif

	std::vector<unsigned int> v_pairs;
	v_pairs.reserve(pairs.size() << 1);
	for (std::unordered_set<std::pair<int,int> >::iterator it = pairs.begin(); it != pairs.end(); it++) {
		v_pairs.push_back(std::get<0>(*it));
		v_pairs.push_back(std::get<1>(*it));
	}

	uint64_t len = static_cast<uint64_t>(v_pairs.size()) * (v_pairs.size() - 1) >> 1;
	FastBitset fb(len);
	perm_to_binary(fb, v_pairs);

	for (std::unordered_set<FastBitset>::iterator it = permutations.begin(); it != permutations.end(); it++) {
		FastBitset fb2 = fb;
		fb2.setIntersection(*it);
		if (fb2.count_v3() == 0)
			permutations.erase(*it);
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
unsigned int loc_to_glob_idx(std::vector<unsigned int> perm, const unsigned int idx, const int N, const int num_mpi_threads, const int rank)
{
	#if DEBUG
	assert (idx < (unsigned int)N);
	assert (N > 0);
	assert (num_mpi_threads > 1);
	assert (rank >= 0);
	#endif

	//The number of rows in a single buffer (two buffers per computer)
	int mpi_offset = N / (num_mpi_threads << 1);
	//Index local to a single buffer, whereas idx spans two buffers
	int loc_idx = idx - (idx / mpi_offset) * mpi_offset;
	//The sequence index - i.e. which buffer loc_idx belongs to
	int seq_idx = perm[(rank << 1) + (idx / mpi_offset)];
	//The original global index, spanning [0, N)
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
void mpi_swaps(const std::vector<std::pair<int,int> > swaps, Bitvector &adj, Bitvector &adj_buf, const int N, const int num_mpi_threads, const int rank)
{
	#if DEBUG
	assert (swaps.size() > 0);
	assert (adj.size() > 0);
	assert (adj_buf.size() > 0);
	assert (N > 0);
	assert (num_mpi_threads > 1);
	assert (rank >= 0);
	#endif

	//Number of rows per buffer
	int mpi_offset = N / (num_mpi_threads << 1);
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

//Get all possible chains between a set of pairs of minimal/maximal elements
//Return values are tuples containing {chain, weight, length}
std::vector<std::tuple<FastBitset, int, int> > getPossibleChains(Bitvector &adj, Bitvector &subadj, Bitvector &chains, Bitvector &chains2, FastBitset *excluded, std::vector<std::pair<int,int> > &endpoints, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, const int min_weight, int &max_weight, int &max_idx, int &end_idx)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (subadj.size() > 0);
	assert (chains.size() > 0);
	assert (endpoints.size() > 0);
	assert (candidates.size() > 0);
	assert (lengths != NULL);
	assert (sublengths != NULL);
	assert (N > 0);
	assert (N_sub > 0);
	#endif

	static const bool CHAIN_DEBUG = false;

	std::vector<std::tuple<FastBitset, int, int> > possible_chains;
	possible_chains.reserve(100);

	FastBitset workspace(N);
	FastBitset subworkspace(N_sub);
	FastBitset cand(N);

	if (excluded != NULL)
		for (size_t i = 0; i < candidates.size(); i++)
			cand.set(candidates[i]);

	size_t i;
	//Parallelize this loop
	for (i = 0; i < endpoints.size(); i++) {
		if (CHAIN_DEBUG) {
			printf_cyan();
			printf("\nLooking in the interval (%d, %d)\n", std::get<0>(endpoints[i]), std::get<1>(endpoints[i]));
			printf_std();
		}

		FastBitset longest_chain(N);
		std::pair<int,int> lw = longestChainGuided_v2(longest_chain, adj, subadj, chains, chains2, workspace, subworkspace, candidates, lengths, sublengths, N, N_sub, std::get<0>(endpoints[i]), std::get<1>(endpoints[i]), 0);

		if (CHAIN_DEBUG) {
			printf("weight: %d\n", std::get<0>(lw));
			printf("length: %d\n", std::get<1>(lw));
			printf("Identified Chain:\n");
			printf_red();
			printf("START -> ");
			for (int j = 0; j < N; j++)
				if (longest_chain.read(j))
					printf("%d -> ", j);
			printf("END\n");
			printf_std();
			int w = longest_chain.count_bits();
			printf(" > chain size is %d\n", w);
		}

		if (excluded != NULL) {
			longest_chain.setDifference(*excluded);
			std::get<0>(lw) = longest_chain.partial_vecprod(cand, 0, N);
		}

		if (CHAIN_DEBUG) {
			printf("chain size after difference: %u\n", (unsigned)longest_chain.count_bits());
			printf("min_weight: %d\n", min_weight);
			fflush(stdout);
		}

		if (std::get<0>(lw) > std::min(min_weight, 10)) {
			if (std::get<0>(lw) > max_weight) {
				max_weight = std::get<0>(lw);
				max_idx = possible_chains.size();
				end_idx = i;
			}

			possible_chains.push_back(std::make_tuple(longest_chain, std::get<0>(lw), std::get<1>(lw)));
		}
	}

	return possible_chains;
}

std::pair<int,int> longestChainGuided_v2(FastBitset &longest_chain, Bitvector &adj, const Bitvector &subadj, Bitvector &fwork, Bitvector &fwork2, FastBitset &fwork3, FastBitset &fwork4, const std::vector<unsigned> &candidates, int * const iwork, std::pair<int,int> * const i2work, const int N, const int NS, int i, int j, const unsigned level)
{
	#if DEBUG
	if (!level) {
		assert (NS > 0);
		assert (N >= NS);
		assert (longest_chain.size() >= (size_t)N);
		assert (adj.size() >= (size_t)N);
		assert (adj[0].size() >= (size_t)N);
		assert (fwork.size() >= (size_t)N);
		assert (fwork[0].size() >= (size_t)N);
		assert (fwork2.size() >= (size_t)N);
		assert (fwork2[0].size() >= (size_t)N);
		assert (fwork3.size() >= (size_t)N);
		assert (fwork4.size() >= (size_t)NS);
		assert (candidates.size() > 0);
		assert (iwork != NULL);
		assert (i2work != NULL);
		assert (i >= 0 && i < NS);
		assert (j >= 0 && j < NS);
	}
	#endif

	static const bool DEBUG_GUIDED = false;

	//Distance between a node and itself is zero
	if (i == j) return std::make_pair(0, 0);
	//Enforce i < j
	if (i > j) std::swap(i, j);

	int longest = 0, heaviest = 0;
	uint64_t idx;
	std::pair<int,int> c;

	if (DEBUG_GUIDED) {
		for (unsigned k = 0; k < level; k++)
			printf("\t");
		printf("i, j = [%d, %d]\n", candidates[i], candidates[j]);
		fflush(stdout);
	}

	if (!level) {
		//Throw an error if this function is called for an unrelated pair
		if (!nodesAreConnected_v2(subadj, NS, i, j))
			assert (false);

		//The set of elements in the longest chain between i and j
		longest_chain.reset();
		longest_chain.set(candidates[i]);

		//The set of longest chains for each element to j
		for (int k = 0; k < N; k++) {
			fwork[k].reset();
			fwork[k].set(k);
		}

		//The set of elements between i and j in the subgraph
		subadj[i].clone(fwork4);
		fwork4.partial_intersection(subadj[j], i, j - i + 1);

		//The lengths and weights of all chains
		for (int k = 0; k < NS; k++)
			i2work[k] = std::make_pair(-1, -1);

		//fwork2, fwork3, and iwork are workspaces used
		//by longestChain_v3 (see below)
	}

	FastBitset test_chain(N);
	FastBitset work(NS);
	FastBitset internal(N);
	FastBitset internal2(N);

	//Iterate over all elements between i and j
	while (fwork4.any()) {
		fwork4.unset(idx = fwork4.next_bit());
		test_chain.reset();
		internal.reset();
		internal2.reset();

		if (DEBUG_GUIDED) {
			for (unsigned k = 0; k < level; k++)
				printf("\t");
			printf("Examining element [%u]:\n", (unsigned)candidates[idx]);
			fflush(stdout);
		}

		//Ignore element if it is not directly linked to i in the subgraph
		subadj[i].clone(work);
		work.partial_intersection(subadj[idx], i, idx - i + 1);
		if (work.any()) continue;

		if (std::get<0>(i2work[idx]) != -1) {
			//We already have calculated the length and weight
			//of the longest chain from idx to j
			c = i2work[idx];
			if (DEBUG_GUIDED) {
				for (unsigned k = 0; k < level; k++)
					printf("\t");
				printf("\tPath has already been traversed: W, L = [%d, %d]\n", std::get<0>(c), std::get<1>(c));
				fflush(stdout);
			}
			fwork[candidates[idx]].partial_clone(test_chain, candidates[idx] + 1, N - candidates[idx] - 1);
		} else {
			//We need to calculate the length and weight
			//of the path from idx to j

			//The set of elements between idx and j in the subgraph
			subadj[idx].clone(work);
			work.partial_intersection(subadj[j], idx, j - idx + 1);

			//If this set is empty, idx and j are directly linked in the subgraph
			if (!work.any()) {
				if (DEBUG_GUIDED) {
					for (unsigned k = 0; k < level; k++)
						printf("\t");
					printf("\tPair [%d, %d] is directly linked in the subgraph.\n", candidates[idx], candidates[j]);
					fflush(stdout);
				}
				c = std::make_pair(1, longestChain_v3(adj, fwork2, internal, &fwork3, iwork, N, candidates[idx], candidates[j], 0));
				if (DEBUG_GUIDED) {
					for (unsigned k = 0; k < level; k++)
						printf("\t");
					printf("\tPath has W, L = [%d, %d]\n", std::get<0>(c), std::get<1>(c));
					fflush(stdout);
				}
			//Otherwise we need to recurse to study the interval between idx and j
			} else {
				if (DEBUG_GUIDED) {
					for (unsigned k = 0; k < level; k++)
						printf("\t");
					printf("\tRecursing to next level to examine interval [%d, %d]\n", candidates[idx], candidates[j]);
					fflush(stdout);
				}
				c = longestChainGuided_v2(test_chain, adj, subadj, fwork, fwork2, fwork3, work, candidates, iwork, i2work, N, NS, idx, j, level + 1);
				if (DEBUG_GUIDED) {
					for (unsigned k = 0; k < level; k++)
						printf("\t");
					printf("\tPath has W, L = [%d, %d]\n", std::get<0>(c), std::get<1>(c));
					fflush(stdout);
				}
			}

			//Record the results so the path idx -> j is not checked again
			i2work[idx] = c;
		}

		int L_i_idx = longestChain_v3(adj, fwork2, internal2, &fwork3, iwork, N, candidates[i], candidates[idx], 0);
		internal.unset(candidates[idx]);
		internal.unset(candidates[j]);
		internal2.unset(candidates[i]);
		internal2.unset(candidates[idx]);

		if (DEBUG_GUIDED) {
			for (unsigned k = 0; k < level; k++)
				printf("\t");
			printf("\tDistance between [%d, %d] is %d.\n", candidates[i], candidates[idx], L_i_idx);
			fflush(stdout);
		}
	
		//Identify the longest path i-j via element idx
		if (std::get<0>(c) + 1 > heaviest || (std::get<0>(c) + 1 == heaviest && std::get<1>(c) + L_i_idx > longest)) {
			heaviest = std::get<0>(c) + 1;
			longest = std::get<1>(c) + L_i_idx;
			test_chain.setUnion(internal);
			test_chain.setUnion(internal2);
			test_chain.partial_clone(fwork[candidates[i]], candidates[i] + 1, N - candidates[i] - 1);
			fwork[candidates[i]].set(candidates[idx]);
			fwork[candidates[i]].partial_clone(longest_chain, candidates[i] + 1, N - candidates[i] - 1);
			if (internal.any())
				internal.partial_clone(fwork[candidates[idx]], candidates[idx] + 1, N - candidates[idx] - 1);

			if (DEBUG_GUIDED) {
				for (unsigned k = 0; k < level; k++)
					printf("\t");
				printf("Path [%d, %d] contains elements: ", candidates[i], candidates[j]);
				for (int J = 0; J < N; J++)
					if (fwork[candidates[i]].read(J))
						printf("%d -> ", J);
				printf("END\n");
				fflush(stdout);
			}
		}
	}

	if (!level)
		longest_chain.set(candidates[j]);

	return std::make_pair(heaviest, longest);
}

//Finds the longest chain using the candidates
//The indices (i,j) should refer to the subgraph of candidates
//Lengths has 'N' elements (only used to be passed to longestChain_v2)
//Sublengths has 'N_sub' elements; the first pair index is for the subgraph length,
// > and the second index is for the global graph length (used at the very end)
//The pair returned is (x,y) where x is the longest path in the subgraph
// > and y is the associated longest path in the graph
std::pair<int,int> longestChainGuided(Bitvector &adj, Bitvector &subadj, Bitvector &chains, Bitvector &chains2, FastBitset &longest_chain, FastBitset *workspace, FastBitset *subworkspace, const std::vector<unsigned int> &candidates, int * const lengths, std::pair<int,int> * const sublengths, const int N, const int N_sub, int i, int j, const unsigned int level)
{
	#if DEBUG
	if (!level) {
		assert (adj.size() > 0);
		assert (subadj.size() > 0);
		assert (chains.size() > 0);
		assert (longest_chain.size() > 0);
		assert (candidates.size() > 0);
		assert (lengths != NULL);
		assert (sublengths != NULL);
		assert (N > 0);
		assert (N_sub > 0);
		assert (N_sub <= N);
		assert (i >= 0 && i < N_sub);
		assert (j >= 0 && j < N_sub);
	}
	assert (workspace != NULL);
	assert (subworkspace != NULL);
	#endif

	assert (false);	//This method is not working; use v2 above instead

	//If the two indices are equal, they must be the same node
	//Define the distance between a node and itself to be zero
	if (i == j) { return std::make_pair(0, 0); }

	//Enforce the condition that i < j
	if (i > j) {
		i ^= j;
		j ^= i;
		i ^= j;
	}

	int longest = 0, slongest = 0;
	bool outer = false;

	uint64_t loc_idx, glob_idx;
	std::pair<int,int> c;

	//This should be executed on the outer-most recursion only
	if (!level) {
		//If the nodes are not even related, return -1
		//Define the distance between unrelated nodes to be -1
		if (!nodesAreConnected_v2(subadj, N_sub, i, j)) { printChk(1) ; return std::make_pair(-1, -1); }
		outer = true;

		longest_chain.reset();
		longest_chain.set(candidates[i]);

		//The workspace is used to identify which nodes to look at between i and j
		subadj[i].clone(*subworkspace);
		subworkspace->partial_intersection(subadj[j], i, j - i + 1);
		//The workspace now contains a bitstring containing '1's
		//for elements in the Alexandrov set (i,j)

		//Initialize:
		// > chains to 0
		// > sublengths to -1 (i.e. unrelated)
		for (int k = 0; k < N; k++) {
			chains[k].reset();
			chains[k].set(k);
		}
		for (int k = 0; k < N_sub; k++)
			sublengths[k] = std::make_pair(-1, -1);
	}

	//These will be passed to the next recursion
	FastBitset test_chain(N);
	FastBitset work(N_sub);
	FastBitset internal_chain(N);
	FastBitset internal_chain2(N);

	//Iterate over all bits set to 1
	for (uint64_t k = 0; k < subworkspace->getNumBlocks(); k++) {
		uint64_t block;
		while ((block = subworkspace->readBlock(k))) {	//If all bits are zero, continue to the next block
			int chain_length = 0, schain_length = 0;
			test_chain.reset();

			//Find the first bit set in this block
			asm volatile("bsfq %1, %0" : "=r" (loc_idx) : "r" (block));
			//This index is 'global' - it is the column index in the subadj matrix
			glob_idx = loc_idx + (k << BLOCK_SHIFT);

			//If this is true, it means this branch has not yet been traversed
			//Note here we use depth first search
			if (std::get<0>(sublengths[glob_idx]) == -1) {
				//These two lines make 'work' now contain a bitstring
				//which has 1's for elements between glob_idx and j
				//i.e. the Alexandrov set (glob_idx, j)
				subworkspace->clone(work);
				work.partial_intersection(subadj[glob_idx], glob_idx, j - glob_idx + 1);

				//If this set defined by 'work' is not the null set...
				if (work.any_in_range(glob_idx, j - glob_idx + 1)) {
					//Recurse to the next level to traverse elements between glob_idx and j
					c = longestChainGuided(adj, subadj, chains, chains2, test_chain, workspace, &work, candidates, lengths, sublengths, N, N_sub, glob_idx, j, level + 1);
					if (std::get<0>(c) == -1 || std::get<0>(c) == -1) printChk(5);

					//std::get<1>(c) += longestChain_v3(adj, chains2, internal_chain, workspace, lengths, N, candidates[i], candidates[glob_idx], 0);
					//The recursion returned the longest paths in the subgraph and graph, respectively
					sublengths[glob_idx] = c;

					//Chain lengths, for this particular path
					schain_length = std::get<0>(c);
					chain_length = std::get<1>(c);

					if (schain_length > slongest || (schain_length == slongest && chain_length > longest)) {
						test_chain.partial_clone(chains[candidates[i]], candidates[i] + 1, N - candidates[i] - 1);
						chains[candidates[i]].set(candidates[glob_idx]);
						slongest = schain_length;
						longest = chain_length;
						chains[candidates[i]].partial_clone(longest_chain, candidates[i] + 1, N - candidates[i] - 1);
					}
				} else {	//It is the null set - the elements are directly linked
					//Then glob_idx and j are directly linked in the subgraph
					schain_length = 1;
					//Check if there are other nodes in between glob_idx and j in the original graph
					chain_length = longestChain_v2(adj, workspace, lengths, N, candidates[glob_idx], candidates[j], 0);
					//chain_length = longestChain_v3(adj, chains2, internal_chain, workspace, lengths, N, candidates[glob_idx], candidates[j], 0);

					//Record these values in sublengths so this branch
					//is not traversed again
					sublengths[glob_idx] = std::make_pair(schain_length, chain_length);
					if (schain_length > slongest || (schain_length == slongest && chain_length > longest)) {
						for (int m = candidates[i] + 1; m < N; m++)
							chains[candidates[i]].unset(m);
						chains[candidates[i]].set(candidates[glob_idx]);
						slongest = 1;
						longest = chain_length;
						chains[candidates[i]].partial_clone(longest_chain, candidates[i] + 1, N - candidates[i] - 1);
					}
				}
			} else {
				//The branch has previously been traversed, so just use those values
				schain_length = std::get<0>(sublengths[glob_idx]);
				chain_length = std::get<1>(sublengths[glob_idx]);
				//chain_length += longestChain_v3(adj, chains2, internal_chain, workspace, lengths, N, candidates[i], candidates[glob_idx], 0);

				if (schain_length > slongest || (schain_length == slongest && chain_length > longest)) {
					chains[candidates[glob_idx]].partial_clone(chains[candidates[i]], candidates[i] + 1, N - candidates[i] - 1);
					slongest = schain_length;
					longest = chain_length;
					chains[candidates[i]].partial_clone(longest_chain, candidates[i] + 1, N - candidates[i] - 1);
				}
			}

			//Unset the bit to continue to the next one
			subworkspace->unset(glob_idx);
		}
	}
	
	slongest++;
	if (outer) {
		//slongest++; longest++;
		//longest += longestChain_v2(adj, workspace, lengths, N, candidates[i], candidates[j], 0);
		longest_chain.set(candidates[j]);
	}

	return std::make_pair(slongest, longest);
}

//Version 3 returns the chain of elements as well
//chains - for each chain, the bitset represents the longest path to the final node
//         by definition, chains[k].read(k) = 1 since node k is the first node in chain[k]
//longest_chain - the longest chain between the source and destination
//workspace - temporary workspace used for intersections
//lengths - array of longest lengths from each node to the destination
int longestChain_v3(Bitvector &adj, Bitvector &chains, FastBitset &longest_chain, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level)
{
	#if DEBUG
	if (!level) {
		assert (adj.size() > 0);
		assert (chains.size() > 0);
		assert (workspace != NULL);
		assert (lengths != NULL);
		assert (N > 0);
		assert (i >= 0 && i < N);
		assert (j >= 0 && j < N);
	}
	assert (workspace != NULL);
	#endif

	if (i == j) return 0;

	if (i > j) {
		i ^= j;
		j ^= i;
		i ^= j;
	}

	int longest = 0;
	bool outer = false;
	
	uint64_t loc_idx, glob_idx;
	int c;

	if (!level) {
		//If the nodes are not even related, return a distance of -1
		if (!nodesAreConnected_v2(adj, N, i, j)) { printChk(2); return -1; }
		outer = true;

		longest_chain.reset();
		longest_chain.set(i);

		//The workspace is used to identify which nodes to look at between i and j
		adj[i].clone(*workspace);
		workspace->partial_intersection(adj[j], i, j - i + 1);

		for (int k = 0; k < N; k++) {
			chains[k].reset();
			chains[k].set(k);
		}

		memset(lengths, -1, sizeof(int) * N);
	}

	FastBitset test_chain(N);
	FastBitset work(N);

	for (uint64_t k = 0; k < workspace->getNumBlocks(); k++) {
		uint64_t block;
		while ((block = workspace->readBlock(k))) {	//If all bits are zero, continue to the next block
			test_chain.reset();

			//Find the first bit set in this block
			asm volatile("bsfq %1, %0" : "=r" (loc_idx) : "r" (block));
			//This index is 'global' - it is the column index in the adj. matrix
			glob_idx = loc_idx + (k << BLOCK_SHIFT);

			int chain_length = 0;	//Length between glob_idx and j
			//Have not yet studied between glob_idx and j
			if (lengths[glob_idx] == -1) {
				workspace->clone(work);
				work.partial_intersection(adj[glob_idx], glob_idx, j - glob_idx + 1);
				//'work' now contains elements between glob_idx and j

				//If there are other nodes in between, continue search
				if (work.any_in_range(glob_idx, j - glob_idx + 1)) {
					c = longestChain_v3(adj, chains, test_chain, &work, lengths, N, glob_idx, j, level + 1);

					lengths[glob_idx] = c;
					chain_length = c;

					if (chain_length > longest) {
						test_chain.partial_clone(chains[i], i + 1, N - i - 1);
						chains[i].set(glob_idx);
						longest = chain_length;
						chains[i].partial_clone(longest_chain, i + 1, N - i - 1);
					}
				} else {
					chain_length = 1;
					lengths[glob_idx] = 1;
					if (chain_length > longest) {
						for (int m = i + 1; m < N; m++)
							chains[i].unset(m);
						chains[i].set(glob_idx);
						longest = 1;
						chains[i].partial_clone(longest_chain, i + 1, N - i - 1);
					}
				}
			} else {
				chain_length = lengths[glob_idx];	//longest path from glob_idx to j

				if (chain_length > longest) {
					chains[glob_idx].partial_clone(chains[i], i + 1, N - i - 1);
					longest = chain_length;
					chains[i].partial_clone(longest_chain, i + 1, N - i - 1);
				}
			}

			workspace->unset(glob_idx);
		}
	}

	longest++;
	if (outer) {
		longest++;
		longest_chain.set(j);
	}

	return longest;
}

//This version identifies paths in reverse, 
//building up from minimal to maximal elements
int longestChain_v2r(Bitvector &adj, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level)
{
	#if DEBUG
	if (!level) {
		assert (adj.size() > 0);
		assert (workspace != NULL);
		assert (lengths != NULL);
		assert (N > 0);
		assert (i >= 0 && i < N);
		assert (j >= 0 && j < N);
	} else
		assert (workspace != NULL);
	#endif

	if (i == j) return 0;

	if (i > j) {
		i ^= j;
		j ^= i;
		i ^= j;
	}

	int longest = 0;
	
	uint64_t loc_idx, glob_idx;
	int c;

	if (!level) {
		//If the nodes are not even related, return a distance of 0
		if (!nodesAreConnected_v2(adj, N, i, j)) { printChk(3); return 0; }

		//The workspace is used to identify which nodes to look at between i and j
		adj[i].clone(*workspace);
		workspace->partial_intersection(adj[j], i, j - i + 1);

		memset(lengths, -1, sizeof(int) * N);
	}

	FastBitset work(N);
	for (uint64_t k = workspace->getNumBlocks(); k-- > 0; ) {
		uint64_t block;
		while ((block = workspace->readBlock(k))) {	//If all bits are zero, continue to the next block
			int chain_length = 0;

			//Find the last bit set in this block
			asm volatile("bsrq %1, %0" : "=r" (loc_idx) : "r" (block));
			glob_idx = loc_idx + (k << BLOCK_SHIFT);

			if (lengths[glob_idx] == -1) {
				workspace->clone(work);
				work.partial_intersection(adj[glob_idx], i, glob_idx - i + 1);

				if (work.any_in_range(i, glob_idx - i + 1)) {
					c = longestChain_v2r(adj, &work, lengths, N, i, glob_idx, level + 1);
					lengths[glob_idx] = c;
					chain_length = c;
				} else {
					chain_length = 1;
					lengths[glob_idx] = 1;
				}
			} else
				chain_length = lengths[glob_idx];

			longest = std::max(chain_length, longest);
			workspace->unset(glob_idx);
		}
	}

	longest++;

	return longest;
}

//Version 2 uses 'lengths' which adds a memory to graph traversal
//so the same path is not traversed twice
int longestChain_v2(const Bitvector &adj, FastBitset *workspace, int *lengths, const int N, int i, int j, unsigned int level)
{
	#if DEBUG
	if (!level) {
		assert (adj.size() > 0);
		assert (workspace != NULL);
		assert (lengths != NULL);
		assert (N > 0);
		assert (i >= 0 && i < N);
		assert (j >= 0 && j < N);
	} else
		assert (workspace != NULL);
	#endif

	if (i == j) return 0;

	if (i > j) {
		i ^= j;
		j ^= i;
		i ^= j;
	}

	int longest = 0;
	
	uint64_t loc_idx, glob_idx;
	int c;

	if (!level) {
		//If the nodes are not even related, return a distance of 0
		if (!nodesAreConnected_v2(adj, N, i, j)) { printChk(3); return 0; }

		//The workspace is used to identify which nodes to look at between i and j
		adj[i].clone(*workspace);
		workspace->partial_intersection(adj[j], i, j - i + 1);

		memset(lengths, -1, sizeof(int) * N);
	}

	FastBitset work(N);
	for (uint64_t k = 0; k < workspace->getNumBlocks(); k++) {
		uint64_t block;
		while ((block = workspace->readBlock(k))) {	//If all bits are zero, continue to the next block
			//int chain_length = 0;

			//Find the first bit set in this block
			asm volatile("bsfq %1, %0" : "=r" (loc_idx) : "r" (block));
			//This index is 'global' - it is the column index in the adj. matrix
			glob_idx = loc_idx + (k << BLOCK_SHIFT);

			if (lengths[glob_idx] == -1) {
				workspace->clone(work);
				work.partial_intersection(adj[glob_idx], glob_idx, j - glob_idx + 1);

				//if (work.any_in_range(glob_idx, j - glob_idx + 1)) {
				if (work.any()) {
					c = longestChain_v2(adj, &work, lengths, N, glob_idx, j, level + 1);
					lengths[glob_idx] = c;
					//chain_length = c;
				} else {
					//chain_length = 1;
					lengths[glob_idx] = 1;
				}
			} //else
			//	chain_length = lengths[glob_idx];

			//longest = std::max(chain_length, longest);
			longest = std::max(lengths[glob_idx], longest);
			workspace->unset(glob_idx);
		}
	}

	longest++;

	return longest;
}

int longestChain_v1(Bitvector &adj, FastBitset *workspace, const int N, int i, int j, unsigned int level)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (workspace != NULL);
	assert (N > 0);
	assert (i >= 0 && i < N);
	assert (j >= 0 && j < N);
	#endif

	static const bool CHAIN_DEBUG = true;

	if (i == j) return 0;

	if (i > j) {
		i ^= j;
		j ^= i;
		i ^= j;
	}

	int longest = 0;
	
	uint64_t loc_idx, glob_idx;
	int c;

	//This only executes for the outer-most loop
	//of the recursion
	if (!level) {
		if (CHAIN_DEBUG)
			printf("Searching for the longest chain between [%d] and [%d]\n", i, j);

		//If the nodes are not even related, return a distance of -1
		if (!nodesAreConnected_v2(adj, N, i, j)) { printChk(4); return 0; }

		//The workspace is used to identify which nodes to look at between i and j
		adj[i].clone(*workspace);
		workspace->partial_intersection(adj[j], i, j - i + 1);
		//workspace->printBitset();
	}

	//These variables are used for the next
	//iteration of the recursion
	FastBitset work(N);
	FastBitset used(N);
	//The outer two loops will iterate through all blocks in the FastBitset
	for (uint64_t k = 0; k < workspace->getNumBlocks(); k++) {
		uint64_t block;
		while ((block = workspace->readBlock(k))) {	//If all bits are zero, continue to the next block
			workspace->clone(work);
			int chain_length = 0;

			//Find the first bit set in this block
			asm volatile("bsfq %1, %0" : "=r" (loc_idx) : "r" (block));
			//This index is 'global' - it is the column index in the adj. matrix
			glob_idx = loc_idx + (k << BLOCK_SHIFT);
			if (CHAIN_DEBUG)
				printf("\nConsidering bit [%" PRIu64 "] (LEVEL %d)\n", glob_idx, level);

			//Set it to zero and perform an intersection
			//This will make 'work' contain set bits for nodes related
			//to both glob_idx and j
			workspace->unset(glob_idx);
			work.partial_intersection(adj[glob_idx], glob_idx, j - glob_idx + 1);

			//If there are other elements, between glob_idx and j, find the longest chain
			if (work.partial_count(glob_idx, j - glob_idx + 1)) {
				c = longestChain_v1(adj, &work, N, glob_idx, j, level + 1);
				//This helps reduce some of the steps performed
				//workspace->setDisjointUnion(work);
				if (CHAIN_DEBUG) {
					for (unsigned int m = 0; m < level + 1; m++) printf("\t");
					printf("Longest chain between [%" PRIu64 "] and [%d] is %d.\n", glob_idx, j, c);
				}
				chain_length += c;
			//Otherwise, this was the last step in the chain
			} else {
				chain_length++;
				if (CHAIN_DEBUG) {
					for (unsigned int m = 0; m < level + 1; m++) printf("\t");
					printf("Longest chain between [%" PRIu64 "] and [%d] is %d.\n", glob_idx, j, chain_length);
				}
			}
			used.set(glob_idx);

			longest = std::max(chain_length, longest);
		}
	}
	longest++;

	workspace->setUnion(used);

	return longest;
}

int rootChain(Bitvector &adj, Bitvector &chains, FastBitset &chain, const int * const k_in, const int * const k_out, int *lengths, const int N)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (chain.size() > 0);
	assert (k_in != NULL);
	assert (k_out != NULL);
	assert (lengths != NULL);
	assert (N > 0);
	#endif

	if (!chain.any()) return 0;

	FastBitset workspace(N);
	FastBitset longest(N);
	FastBitset fb(N);

	uint64_t block, loc_idx;
	int first = -1, last = -1;
	int first_longest = 0, last_longest = 0;
	int new_elements = 0;

	for (uint64_t k = 0; k < chain.getNumBlocks(); k++) {
		if ((block = chain.readBlock(k))) {
			asm volatile("bsfq %1, %0" : "=r" (loc_idx) : "r" (block));
			first = static_cast<int>(loc_idx + (k << BLOCK_SHIFT));
			break;
		}
	}

	for (uint64_t k = chain.getNumBlocks() - 1; k-- > 0; ) {
		uint64_t block;
		if ((block = chain.readBlock(k))) {
			asm volatile("bsrq %1, %0" : "=r" (loc_idx) : "r" (block));
			last = static_cast<int>(loc_idx + (k << BLOCK_SHIFT));
			break;
		}
		
	}

	for (int i = 0; i < first; i++) {
		if (!!k_in[i]) continue;
		if (!nodesAreConnected_v2(adj, N, i, first)) continue;

		int first_length = longestChain_v3(adj, chains, fb, &workspace, lengths, N, i, first, 0);
		if (first_length > first_longest) {
			first_longest = first_length;
			fb.clone(longest);
		}
	}

	new_elements += longest.count_bits() - 1;
	chain.setUnion(longest);
	longest.reset();

	for (int i = last + 1; i < N; i++) {
		if (!!k_out[i]) continue;
		if (!nodesAreConnected_v2(adj, N, last, i)) continue;

		int last_length = longestChain_v3(adj, chains, fb, &workspace, lengths, N, last, i, 0);
		if (last_length > last_longest) {
			last_longest = last_length;
			fb.clone(longest);
		}
	}

	new_elements += longest.count_bits() - 1;
	chain.setUnion(longest);

	return new_elements;
}

//Find the farthest maximal element
int findLongestMaximal(Bitvector &adj, const int *k_out, int *lengths, const int N, const int idx)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (lengths != NULL);
	assert (N > 0);
	assert (idx >= 0 && idx < N);
	#endif

	FastBitset workspace = FastBitset(N);
	int longest = -1;
	int max_idx = 0;

	//#pragma omp parallel for schedule (dynamic, 1) if (N > 1000)
	for (int i = 0; i < N; i++) {
		if (k_out[i]) continue;
		if (!nodesAreConnected_v2(adj, N, idx, i)) continue;
		int length = longestChain_v2(adj, &workspace, lengths, N, idx, i, 0);
		//#pragma omp critical
		{
		if (length > longest) {
			longest = length;
			max_idx = i;
		}
		}
	}

	return max_idx;
}

//Find the farthest minimal element
int findLongestMinimal(Bitvector &adj, const int *k_in, int *lengths, const int N, const int idx)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (lengths != NULL);
	assert (N > 0);
	assert (idx >= 0 && idx < N);
	#endif

	FastBitset workspace = FastBitset(N);
	int longest = -1;
	int min_idx = N - 1;

	//#pragma omp parallel for schedule (dynamic, 1) if (N > 1000)
	for (int i = 0; i < N; i++) {
		if (k_in[i]) continue;
		if (!nodesAreConnected_v2(adj, N, i, idx)) continue;
		int length = longestChain_v2(adj, &workspace, lengths, N, i, idx, 0);
		//#pragma omp critical
		{
		if (length > longest) {
			longest = length;
			min_idx = i;
		}
		}
	}

	return min_idx;
}

int maximalAntichain(FastBitset &antichain, const Bitvector &adj, const int N, const int seed)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (seed >= 0 && seed < N);
	#endif

	FastBitset candidates(N);
	FastBitset workspace(N);
	FastBitset workspace2(N);

	adj[seed].clone(antichain);	//1's indicate relations to 'seed'
	antichain.flip();		//1's indicate elements unrelated to 'seed'

	antichain.clone(candidates);
	while (candidates.any()) {
		uint64_t max_left = 0;
		uint64_t remove_me = 0;
		antichain.clone(workspace);
		//Go through the antichain, and see which element decreases the set size the least when removed
		//This is a 'good' element of the antichain, which should maximize the size
		while (workspace.any()) {
			//This is your test element
			uint64_t element = workspace.next_bit();
			//If this one has already passed a test and been kept, ignore it
			if (!candidates.read(element)) { workspace.unset(element); continue; }
			antichain.clone(workspace2);
			//Remove the element
			workspace2.setDifference(adj[element]);
			//The size of the antichain afterwards
			uint64_t left = workspace2.count_bits();
			//Remember which one keeps the most
			if (left > max_left) { max_left = left; remove_me = element; }

			workspace.unset(element);
		}
		//Remove this candidate's neighbors from the antichain and candidates
		antichain.setDifference(adj[remove_me]);
		candidates.setDifference(adj[remove_me]);
		//Do not re-consider this candidate
		candidates.unset(remove_me);
	}

	return (int)antichain.count_bits();
}

//We always have source <= v < N
void closure(Bitvector &adj, Bitvector &links, const int N, const int source, const int v)
{
	//We have the relation [source -> v]
	adj[source].set(v);
	adj[v].set(source);
	//Take the forward closure
	for (int i = v + 1; i < N; i++)
		//If there is a link [v -> i], but not the relation [source -> i], take the closure
		if (links[v].read(i) && !adj[source].read(i))
			closure(adj, links, N, source, i);
}

//Produce causal matrix 'adj' given link matrix 'links'
//Note: we only need the upper triangular link matrix
void transitiveClosure(Bitvector &adj, Bitvector &links, const int N)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	#endif

	for (int i = 0; i < N; i++)
		adj[i].reset();
	for (int i = 0; i < N; i++)
		closure(adj, links, N, i, i);
	for (int i = 0; i < N; i++)
		adj[i].unset(i);
}

//Produce link matrix 'links' given causal matrix 'adj'
//Only the upper triangular portion is produced
void transitiveReduction(Bitvector &links, Bitvector &adj, const int N)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (links.size() >= (size_t)N);
	#endif

	//Copy the upper triangular portion of adj into links
	for (int i = 0; i < N; i++)
		adj[i].partial_clone(links[i], i, N - i);

	for (int i = 0; i < N - 1; i++)
		for (int j = i + 1; j < N; j++)
			//A[i,j]=1 indicates a possible transitive relation
			//The partial_vecprod returns non-zero if the relation is transitive
			//We can't use the partial_vecprod if 'adj' is not symmetric
			//if (adj[i].read(j) && adj[i].partial_vecprod(adj[j], i, j - i + 1))
			if (adj[i].read(j))
				for (int k = i + 1; k < j; k++)
					if (adj[i].read(k) && adj[k].read(j))
						links[i].unset(j);
}

void identifyTimelikeCandidates(std::vector<unsigned> &candidates, int *chaintime, int *iwork, Bitvector &fwork, const Node &nodes, const Bitvector &adj, const Spacetime &spacetime, const int N, Stopwatch &sMeasureExtrinsicCurvature, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (chaintime != NULL);
	assert (iwork != NULL);
	//assert (fwork.size() > 0 && fwork.size() == fwork[0].size());
	assert (nodes.crd != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (adj.size() > 0 && adj.size() == adj[0].size());
	assert (spacetime.stdimIs("2") && spacetime.manifoldIs("Minkowski") && spacetime.curvatureIs("Flat"));
	assert (N > 0);
	#endif

	candidates.clear();
	candidates.swap(candidates);

	memset(chaintime, -1, N * omp_get_max_threads());
	//memset(iwork, -1, N * omp_get_max_threads());

	fwork.clear();
	fwork.swap(fwork);
	fwork.reserve(omp_get_max_threads());
	FastBitset fb((uint64_t)N);
	for (int i = 0; i < omp_get_max_threads(); i++)
		fwork.push_back(fb);

	FastBitset visited(N);
	Bitvector frontier;
	for (int i = 0; i < omp_get_max_threads(); i++)
		frontier.push_back(fb);
	unsigned *layer_sizes = (unsigned*)malloc(N * omp_get_max_threads() * sizeof(unsigned));
	unsigned *min_k = (unsigned*)malloc(N * omp_get_max_threads() * sizeof(unsigned));

	//Identify antichain layers
	unsigned layer, num_visited, maximum_chain;
	unsigned cutoff = 10;
	float epsilon = 1.5;
	bool forward = true;
	//First element will always belong to layer 0
	frontier[0].set(0);
	chaintime[0] = 0;
	layer_sizes[0] = 1;
	min_k[0] = nodes.k_in[0] + nodes.k_out[0];
	layer = 0;
	visited.set(0);
	printf("\tForward Pass:\n"); fflush(stdout);
	#if !LAYER_ONLY
	Layers:
	#endif
	memset(layer_sizes, 0, sizeof(unsigned) * N * omp_get_max_threads());
	memset(min_k, UINT32_MAX, N * omp_get_max_threads() * sizeof(unsigned));
	num_visited = 0;
	stopwatchStart(&sMeasureExtrinsicCurvature);
	while (num_visited != (unsigned)N) {
		#pragma omp parallel for schedule (guided) if (N - num_visited > 1024)
		for (unsigned i = 0; i < (unsigned)N; i++) {
			if (visited.read(i)) continue;
			unsigned tid = omp_get_thread_num();
			adj[i].clone(fwork[tid]);
			fwork[tid].setDifference(visited);
			if ((forward && !fwork[tid].partial_count(0, i)) || (!forward && !fwork[tid].partial_count(i, N - i + 1))) {
				frontier[tid].set(i);
				chaintime[i] = layer;
				layer_sizes[N*tid+layer]++;
				min_k[N*tid+layer] = std::min(min_k[N*tid+layer], (unsigned)(nodes.k_in[i] + nodes.k_out[i]));
			}
		}

		for (unsigned i = 0; i < (unsigned)omp_get_max_threads(); i++) {
			visited.setUnion(frontier[i]);
			frontier[i].reset();
			if (!!i) {
				layer_sizes[layer] += layer_sizes[N*i+layer];
				min_k[layer] = std::min(min_k[layer], min_k[N*i+layer]);
			}
		}
		//printf("layer [%u] size: [%u] min_k: [%u]\n", layer, layer_sizes[layer], min_k[layer]);

		num_visited = visited.count_bits();
		if (forward)
			layer++;
		else
			layer--;
	}

	stopwatchStop(&sMeasureExtrinsicCurvature);
	printf_dbg("\tIdentified Antichains: %5.6f sec\n", sMeasureExtrinsicCurvature.elapsedTime);
	stopwatchReset(&sMeasureExtrinsicCurvature);
	stopwatchStart(&sMeasureExtrinsicCurvature);

	//Length of longest (maximum) chain
	maximum_chain = *std::max_element(chaintime, chaintime + N);

	//Identify the boundary candidates
	float threshold;
	for (unsigned i = 0; i < (unsigned)N; i++) {
		layer = chaintime[i];
		if (layer_sizes[layer] < cutoff) continue;
		threshold = epsilon * sqrt((float)min_k[layer]);
		if (layer >= maximum_chain * 0.4 && layer < maximum_chain * 0.6)
			threshold *= 5.0;
		threshold += min_k[layer];
		if ((float)(nodes.k_in[i] + nodes.k_out[i]) < threshold && ((forward && layer >= maximum_chain * 0.35) || (!forward && layer < maximum_chain * 0.65)))
			candidates.push_back(i);
	}

	stopwatchStop(&sMeasureExtrinsicCurvature);
	printf_dbg("\tIdentified Candidates: %5.6f sec\n", sMeasureExtrinsicCurvature.elapsedTime);

	#if !LAYER_ONLY
	if (forward) {
		forward = false;
		layer = maximum_chain;
		//Last element will always belong to layer 'maximum_chain'
		frontier[0].set(N - 1);
		chaintime[N-1] = layer;
		layer_sizes[layer] = 1;
		min_k[layer] = nodes.k_in[N-1] + nodes.k_out[N-1];
		visited.reset();
		visited.set(N-1);
		printf("\n\tBackward Pass:\n"); fflush(stdout);
		goto Layers;
	}

	candidates.push_back(0);
	candidates.push_back(N-1);
	std::sort(candidates.begin(), candidates.end());
	#endif

	free(layer_sizes);
	free(min_k);

	if (!bench) {
		printf("\tCalculated Timelike Boundary Candidates.\n");
		printf_cyan();
		printf("\t\tIdentified [%zd] Candidates.\n", candidates.size());
		printf_std();
		printf("\t\tLongest Maximal Chain: [%d]\n", maximum_chain);
	}
}

#include "NetworkCreator.h"

bool configureSubgraph(Network *subgraph, const Node &nodes, std::vector<unsigned int> candidates, CaResources * const ca, CUcontext &ctx)
{
	#ifdef DEBUG
	assert (candidates.size() > 0);
	#endif

	printf("\n\tPreparing to Generate Subgraph...\n");
	fflush(stdout);

	subgraph->network_properties.N = candidates.size();

	subgraph->network_properties.flags.use_gpu = false;	
	subgraph->network_properties.flags.has_exact_k = false;

	subgraph->network_properties.flags.use_bit = true;

	subgraph->network_properties.flags.calc_clustering = false;
	subgraph->network_properties.flags.calc_components = false;
	subgraph->network_properties.flags.calc_success_ratio = false;
	subgraph->network_properties.flags.calc_action = false;
	subgraph->network_properties.flags.calc_extrinsic_curvature = false;

	subgraph->network_properties.flags.verbose = true;
	subgraph->network_properties.flags.bench = false;

	subgraph->network_observables.clustering = NULL;
	subgraph->network_observables.cardinalities = NULL;

	CausetPerformance cp = CausetPerformance();
	if (!createNetwork(subgraph->nodes, subgraph->edges, subgraph->adj, subgraph->links, subgraph->U, subgraph->V, subgraph->network_properties.spacetime, subgraph->network_properties.N, subgraph->network_properties.k_tar, subgraph->network_properties.core_edge_fraction, subgraph->network_properties.edge_buffer, subgraph->network_properties.gt, subgraph->network_properties.cmpi, subgraph->network_properties.group_size, ca, cp.sCreateNetwork, subgraph->network_properties.flags.use_gpu, subgraph->network_properties.flags.decode_cpu, subgraph->network_properties.flags.link, subgraph->network_properties.flags.relink, subgraph->network_properties.flags.mcmc, subgraph->network_properties.flags.no_pos, subgraph->network_properties.flags.use_bit, subgraph->network_properties.flags.mpi_split, subgraph->network_properties.flags.verbose, subgraph->network_properties.flags.bench, subgraph->network_properties.flags.yes))
		return false;

	for (int i = 0; i < subgraph->network_properties.N; i++) {
		if (subgraph->nodes.id.tau != NULL)
			subgraph->nodes.id.tau[i] = nodes.id.tau[candidates[i]];
		if (subgraph->nodes.crd->v() != NULL)
			subgraph->nodes.crd->v(i) = nodes.crd->v(candidates[i]);
		if (subgraph->nodes.crd->w() != NULL)
			subgraph->nodes.crd->w(i) = nodes.crd->w(candidates[i]);
		if (subgraph->nodes.crd->x() != NULL)
			subgraph->nodes.crd->x(i) = nodes.crd->x(candidates[i]);
		if (subgraph->nodes.crd->y() != NULL)
			subgraph->nodes.crd->y(i) = nodes.crd->y(candidates[i]);
		if (subgraph->nodes.crd->z() != NULL)
			subgraph->nodes.crd->z(i) = nodes.crd->z(candidates[i]);
	}

	//int low = 0;
	//int high = subgraph->network_properties.N - 1;
	//quicksort(subgraph->nodes, subgraph->network_properties.spacetime, low, high);

	if (!linkNodes(subgraph, ca, &cp, ctx))
		return false;

	printf("\tGenerated Subgraph from Candidates.\n");
	fflush(stdout);

	return true;
}

void identifyBoundaryChains(std::vector<std::tuple<FastBitset, int, int>> &boundary_chains, std::vector<std::pair<int,int>> &pwork, int *iwork, std::pair<int,int> *i2work, Bitvector &fwork, Bitvector &fwork2, Network * const network, Network * const subnet, std::vector<unsigned> &candidates)
{
	#if DEBUG
	assert (network != NULL);
	assert (subnet != NULL);
	assert (candidates.size() > 0);
	#endif

	FastBitset excluded(network->network_properties.N);
	int min_weight = -1;

	boundary_chains.clear();
	boundary_chains.swap(boundary_chains);
	boundary_chains.reserve(100);

	pwork.clear();
	pwork.swap(pwork);
	pwork.reserve(1000);

	memset(iwork, -1, network->network_properties.N * omp_get_max_threads());
	for (int i = 0; i < subnet->network_properties.N; i++)
		i2work[i] = std::make_pair(-1, -1);

	fwork.clear();
	fwork.swap(fwork);
	fwork.reserve(network->network_properties.N);
	FastBitset fb((uint64_t)network->network_properties.N);
	for (int i = 0; i < network->network_properties.N; i++)
		fwork.push_back(fb);

	fwork2.clear();
	fwork2.swap(fwork2);
	fwork2.reserve(network->network_properties.N);
	for (int i = 0; i < network->network_properties.N; i++)
		fwork2.push_back(fb);

	//Make a list of pairs of endpoints (vector of pairs)
	/*for (int i = 0; i < subnet->network_properties.N; i++) {
		if (!!subnet->nodes.k_in[i]) continue;
		for (int j = 0; j < subnet->network_properties.N; j++) {
			if (!!subnet->nodes.k_out[j]) continue;
			if (!nodesAreConnected_v2(subnet->adj, subnet->network_properties.N, i, j)) continue;
			pwork.push_back(std::make_pair(std::min(i, j), std::max(i, j)));
		}
	}*/
	pwork.push_back(std::make_pair(0, subnet->network_properties.N - 1));

	std::unordered_set<unsigned> past;
	std::unordered_set<unsigned> future;
	for (unsigned i = 0; i < pwork.size(); i++) {
		past.emplace(pwork[i].first);
		future.emplace(pwork[i].second);
	}

	printf("Checking [%zd] possible chains.\n", pwork.size());
	printf(" > [%zd] past elements\n", past.size());
	printf(" > [%zd] future elements\n", future.size());
	fflush(stdout);
	
	while (pwork.size() > 0) {
		int max_weight = 0, max_idx = -1, end_idx = -1;
		std::vector<std::tuple<FastBitset,int,int>> possible_chains = getPossibleChains(network->adj, subnet->adj, fwork, fwork2, &excluded, pwork, candidates, iwork, i2work, network->network_properties.N, subnet->network_properties.N, min_weight, max_weight, max_idx, end_idx);
		if (end_idx == -1) break;

		boundary_chains.push_back(possible_chains[max_idx]);
		excluded.setUnion(std::get<0>(possible_chains[max_idx]));

		//If there are multiple paths with the same endpoints, check for that here (not sure how yet)

		pwork.erase(pwork.begin() + end_idx);
		if (min_weight == -1) { min_weight = 10; }

		if (network->network_properties.spacetime.stdimIs("2") && ((network->network_properties.spacetime.regionIs("Half_Diamond_T") && boundary_chains.size() == 1) || boundary_chains.size() == 2)) break;
	}

	//Extend chains to maximal and minimal elements
	//for (size_t i = 0; i < boundary_chains.size(); i++)
	//	std::get<2>(boundary_chains[i]) += rootChain(network->adj, fwork, std::get<0>(boundary_chains[i]), network->nodes.k_in, network->nodes.k_out, iwork, network->network_properties.N);

	printf("Identified [%zd] Boundary Chains.\n", boundary_chains.size());
	fflush(stdout);
}

//Estimate the integrated and exponential autocorrelations of 'data'
//data: the target data sequence
//acorr: workspace for the autocorrelation coefficients
//lags: workspace which should contain log({1,2,3,...,n-1})
//nsamples: number of samples in 'data'
//tau_exp: exponential autocorrelation time
//tau_int: integrated autocorrelation time
void autocorrelation(double *data, double *acorr, const double * const lags, unsigned nsamples, double &tau_exp, double &tau_exp_err, double &tau_int, double &tau_int_err)
{
	memset(acorr, 0, sizeof(double) * nsamples);
	acorr[0] = 1;
	tau_int = acorr[0] / 2.0;
	tau_int_err = 0.0;
	tau_exp_err = 0.0;

	unsigned max_lag = 0;

	for (unsigned lag = 1; lag < nsamples / 2; lag++) {
		//Calculate the autocorrelation coefficients
		acorr[lag] = gsl_stats_correlation(data, 1, data + lag, 1, nsamples - lag);
		if (acorr[lag] != acorr[lag])
			acorr[lag] = 0.0;

		//Sum them to estimate the integrated autocorrelation
		tau_int += acorr[lag];
		tau_int_err += POW2(1.0 - POW2(acorr[lag])) / (nsamples - lag);
		if (lag >= 6 * tau_int) {
			max_lag = lag;
			break;
		}
	}
	tau_int_err = sqrt(tau_int_err);

	if (!max_lag || max_lag == 1) {
		//There is not enough data to estimate these
		//tau_int = tau_exp = nsamples;
		tau_int = tau_exp = 1.0;
		tau_int_err = tau_exp_err = 1.0;
		return;
	}

	//Prepare data for a linear fit
	for (unsigned lag = 1; lag <= max_lag; lag++)
		acorr[lag] = log(acorr[lag] + 1.0);

	double c0, c1, cov00, cov01, cov11, sumsq;
	gsl_fit_linear(lags, 1, acorr, 1, max_lag + 1, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
	tau_exp = -1.0 / c1;
	tau_exp_err = sumsq;
}

double jackknife(double *jacksamples, double mean, unsigned nsamples)
{
	double sigma = 0.0;
	for (unsigned i = 0; i < nsamples; i++)
		sigma += POW2(jacksamples[i] - mean);
	sigma = sqrt(sigma * (nsamples - 1) / nsamples);
	return sigma;
}

void specific_heat(double &Cv, double &err, double *action, double mean, double stddev, double beta, unsigned nsamples, unsigned stride)
{
	if (beta == 0.0) {
		Cv = err = 0.0;
		return;
	}

	Cv = POW2(beta * stddev);

	std::vector<double> jacksamples(nsamples, 0.0);
	for (unsigned i = 0; i < nsamples; i++) {
		//jacksamples[i] = POW2(beta) * (POW2(stddev) * (nsamples - 1) - POW2(action[i*stride] - mean));
		for (unsigned j = 0; j < nsamples; j++)
			if (i != j)
				jacksamples[j] += POW2(action[i*stride] - mean);
	}
	for (unsigned i = 0; i < nsamples; i++)
		jacksamples[i] = POW2(beta) * jacksamples[i] / (nsamples - 1.0);
	err = jackknife(jacksamples.data(), Cv, nsamples);
}

//Returns dimensionless free energy, -beta*F
void free_energy(double &F, double &err, double *action, double mean, double beta, unsigned nsamples, unsigned stride)
{
	if (beta == 0.0) {
		F = err = 0.0;
		return;
	}

	F = 0.0;
	std::vector<double> jacksamples(nsamples, 0.0);
	for (unsigned i = 0; i < nsamples; i++) {
		double Fi = exp(-beta * (action[i*stride] - mean));
		F += Fi;
		for (unsigned j = 0; j < nsamples; j++)
			if (i != j)
				jacksamples[j] += Fi;
	}
	F = log(F) - beta * mean;
	for (unsigned i = 0; i < nsamples; i++)
		jacksamples[i] = log(jacksamples[i]) - beta * mean;
	err = jackknife(jacksamples.data(), F, nsamples);
}

void entropy(double &s, double &err, double action_mean, double action_stddev, double free_energy, double free_energy_stddev, double beta, uint64_t npairs, unsigned nsamples)
{
	if (beta == 0.0) {
		s = err = 0.0;
		return;
	}

	s = beta * (action_mean - free_energy) / npairs;
	err = sqrt((POW2(action_stddev) + POW2(free_energy_stddev)) / nsamples);
}
