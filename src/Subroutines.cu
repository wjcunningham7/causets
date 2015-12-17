#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
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
	int t_idx = 0;
	int i;

	try {
		//Identify Value in Table
		//Assumes values are written (y, x)
		for (i = (int)(!first); i < size / (int)sizeof(double); i += 2) {
			if ((increasing && table[i] > input) || (!increasing && table[i] < input)) {
				t_idx = i;
				break;
			}
		}

		//Check if Table is Insufficient
		if (t_idx == (int)(!first)) {
			printf("%f\n", input);
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
void quicksort(Node &nodes, const int &stdim, const Manifold &manifold, int low, int high)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (stdim == 2 || stdim == 4);
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW || manifold == HYPERBOLIC);
	if (manifold == HYPERBOLIC)
		assert (stdim == 2);
	#endif

	int i, j, k;
	float key = 0.0;

	if (low < high) {
		k = (low + high) >> 1;
		swap(nodes, stdim, manifold, low, k);
		if (stdim == 2)
			key = nodes.crd->x(low);
		else if (stdim == 4)
			key = nodes.crd->w(low);
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && ((stdim == 4 ? nodes.crd->w(i) : nodes.crd->x(i)) <= key))
				i++;
			while ((j >= low) && ((stdim == 4 ? nodes.crd->w(j) : nodes.crd->x(j)) > key))
				j--;
			if (i < j)
				swap(nodes, stdim, manifold, i, j);
		}

		swap(nodes, stdim, manifold, low, j);
		quicksort(nodes, stdim, manifold, low, j - 1);
		quicksort(nodes, stdim, manifold, j + 1, high);
	}
}

//Sort edge list
void quicksort(uint64_t *edges, int low, int high)
{
	#if DEBUG
	assert (edges != NULL);
	#endif

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
void swap(Node &nodes, const int &stdim, const Manifold &manifold, const int i, const int j)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (stdim == 2 || stdim == 4);
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW || manifold == HYPERBOLIC);
	if (manifold == HYPERBOLIC)
		assert (stdim == 2);
	#endif

	if (stdim == 2) {
		float2 hc = nodes.crd->getFloat2(i);
		nodes.crd->setFloat2(nodes.crd->getFloat2(j), i);
		nodes.crd->setFloat2(hc, j);
	} else if (stdim == 4) {
		float4 sc = nodes.crd->getFloat4(i);
		nodes.crd->setFloat4(nodes.crd->getFloat4(j), i);
		nodes.crd->setFloat4(sc, j);
	}

	if (manifold == DE_SITTER || manifold == DUST || manifold == FLRW) {
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
void swap(uint64_t *edges, const int i, const int j)
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
void swap(const int * const *& list0, const int * const *& list1, int &idx0, int &idx1, int &max0, int &max1)
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
//O(k) Efficiency for Adjacency List
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const std::vector<bool> core_edge_exists, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx)
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
		//int temp = past_idx;
		//past_idx = future_idx;
		//future_idx = temp;
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	//Check if the adjacency matrix can be used
	if (past_idx < core_limit && future_idx < core_limit)
		return core_edge_exists[static_cast<uint64_t>(past_idx) * core_limit + future_idx];
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

	int ps = edges.past_edge_row_start[index];
	int fs = edges.future_edge_row_start[index];
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

void causet_intersection_v2(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many)
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

	//int larger = k_i > k_o ? k_i : k_o;
	//int smaller = k_i <= k_o ? k_i : k_o;

	//if (larger + smaller > smaller * LOG(larger, APPROX ? FAST : STL)) {
		//Binary search
	//} else {
		int idx0 = pstart;	//Index of past neighbors of 'future element j'
		int idx1 = fstart;	//Index of future neighbors of 'past element i'
		int max0 = idx0 + k_i;
		int max1 = idx1 + k_o;

		while (idx0 < max0 && idx1 < max1 && !too_many) {
			if (past_edges[idx0] > future_edges[idx1])
				idx1++;
			else if (past_edges[idx0] < future_edges[idx1])
				idx0++;
			else {
				elements++;

				if (elements >= max_cardinality - 1) {
					too_many = true;
					//printChk();
					break;
				}

				idx0++;
				idx1++;
			}
		}
	//}

	//printf("(%d - %d):\t%d\n", past_edges[pstart], future_edges[fstart], elements);
}

//Intersection of Sorted Lists
//Used to find the cardinality of an interval
//Complexity: O(k*log(k))
void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many)
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

	int idx0 = pstart;
	int idx1 = fstart;
	int max0 = idx0 + k_i;
	int max1 = idx1 + k_o;

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

	/*printf("Future Edge List:\n");
	for (int i = 0; i < k_o; i++)
		printf("\t%d\n", (*secondary)[i+fstart]);
	printf("Past Edge List:\n");
	for (int i = 0; i < k_i; i++)
		printf("\t%d\n", (*primary)[i+pstart]);*/

	//printf("idx0: %d\tidx1: %d\n", idx0, idx1);

	while (idx0 < max0 && idx1 < max1) {
		if ((*secondary)[idx1] > (*primary)[idx0])
			swap(primary, secondary, idx0, idx1, max0, max1);

		/*if (*primary == past_edges)
			printf("Primary: PAST\n");
		else if (*primary == future_edges)
			printf("Primary: FUTURE\n");
		if (*secondary == past_edges)
			printf("Secondary: PAST\n");
		else if (*secondary == future_edges)
			printf("Secondary: FUTURE\n");*/

		while (idx1 < max1 && (*secondary)[idx1] < (*primary)[idx0])
			idx1++;

		if (idx1 == max1)
			//continue;
			break;

		//printf("idx0: %d\tidx1: %d\n", idx0, idx1);

		if ((*primary)[idx0] == (*secondary)[idx1]) {
			//printf_red();
			//printf("Element Found!\n");
			//printf_std();
			elements++;
			if (elements >= max_cardinality - 1) {
				too_many = true;
				//printf("TOO MANY!\n");
				return;
			}
			idx0++;
			idx1++;
		}
	}

	/*printf_red();
	printf("Found %d Elements.\n", elements);
	printf_std();*/
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
void readEdges(uint64_t * const &edges, const bool * const h_edges, std::vector<bool> &core_edge_exists, int * const &g_idx, const unsigned int &core_limit, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y, const bool &use_bit)
{
	#if DEBUG
	if (!use_bit)
		assert (edges != NULL);
	assert (h_edges != NULL);
	assert (g_idx != NULL);
	assert (*g_idx >= 0);
	assert (x >= 0);
	assert (y >= 0);
	assert (x <= y);
	#endif

	uint64_t idx1, idx2;
	unsigned int i, j;

	for (i = 0; i < size0; i++) {
		for (j = 0; j < size1; j++) {
			if (h_edges[i*mthread_size+j] && (use_bit || g_idx[0] < (int)d_edges_size)) {
				if (!use_bit)
					edges[g_idx[0]++] = (static_cast<uint64_t>(x*mthread_size+i)) << 32 | (static_cast<uint64_t>(y*mthread_size+j));
				else
					g_idx[0]++;
				if (x*mthread_size+i < core_limit && y*mthread_size+j < core_limit) {
					idx1 = static_cast<uint64_t>(x * mthread_size + i) * core_limit + y * mthread_size + j;
					idx2 = static_cast<uint64_t>(y * mthread_size + j) * core_limit + x * mthread_size + i;

					core_edge_exists[idx1] = true;
					core_edge_exists[idx2] = true;
				}
			}
		}
	}
}

//Remake adjacency sub-matrix using 'l' rows, beginning at row 'i'
void remakeAdjMatrix(bool * const adj0, bool * const adj1, const int * const k_in, const int * const k_out, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, int * const idx_buf0, int * const idx_buf1, const int &N_tar, const int &i, const int &j, const int &l)
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
		int start = past_edge_row_start[M];
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
		int start = past_edge_row_start[N];
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
void scan(const int * const k_in, const int * const k_out, int * const &past_edge_pointers, int * const &future_edge_pointers, const int &N_tar)
{
	int past_idx = 0, future_idx = 0;
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
