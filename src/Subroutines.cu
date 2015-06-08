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
		assert (filename != NULL);
		assert (lt != NULL);
		assert (size != NULL);
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

//Lookup value in table of (x, y) coordinates -> 2D parameter space
double lookupValue(const double *table, const long &size, double *x, double *y, bool increasing)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (size > 0);
		assert ((x == NULL) ^ (y == NULL));
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
	if (DEBUG) {
		assert (table != NULL);
		assert (size > 0);
		assert (omega12 >= 0.0);
		assert (t1 >= 0.0);
		assert (t2 >= 0.0);
	}

	if (t2 < t1) {
		double temp = t1;
		t1 = t2;
		t2 = temp;
	}

	//double tau1_val = 0.0;
	//double tau2_val = 0.0;
	//double omega12_val = 0.0;
	double lambda = 0.0;
	double tol = 1e-2;

	//NOTE: these values are currently HARD CODED.  This will later be changed,
	//but it requires re-generating the lookup table.

	//int tau_step = 200;	//For FLRW
	//int tau_step = 500;	//For de Sitter
	//int lambda_step = 1000;
	//int step = 4 * tau_step * lambda_step;
	int tau_step, lambda_step, step;
	int counter;
	int i;

	//DEBUG
	//printf("tau1: %f\ttau2: %f\tomega12: %f\n", t1, t2, omega12);
	//fflush(stdout);

	try {
		//The first two table elements should be zero
		if (table[0] != 0.0 || table[1] != 0.0)
			throw CausetException("Corrupted lookup table!\n");

		tau_step = table[2];
		lambda_step = table[3];
		step = 4 * tau_step * lambda_step;
		counter = 0;

		//printf("Looking for tau1.\n");

		//Identify Value in Table
		//Assumes values are written (tau1, tau2, omega12, lambda)
		for (i = 4; i < size / (int)sizeof(double); i += step) {
			//printf("i: %d\tvalue: %f\n", i, table[i]);
			counter++;

			if (step == 4 * tau_step * lambda_step && table[i] > t1) {
				//tau1_val = table[i-step];
				i -= (step - 1);
				step = 4 * lambda_step;
				//printf("Found tau1: %f\tat index: %d\n", tau1_val, i - 1);
				//printf("Looking for tau2 beginning at %d.\n", i);
				i -= step;
				counter = 0;
			} else if (step == 4 * lambda_step && table[i] > t2) {
				//tau2_val = table[i-step];
				i -= (step - 1);
				step = 4;
				//printf("Found tau2: %f\tat index: %d\n", tau2_val, i - 1);
				//printf("Looking for omega12 beginning at %d.\n", i);
				i -= step;
				counter = 0;
			} else if (step == 4 && ABS(table[i] - omega12, STL) / omega12 < tol && table[i] != 0.0) {
				//omega12_val = table[i-step];
				i -= step;
				step = 1;
				//printf("Found omega12: %f\tat index: %d\n", omega12_val, i);
				//printf("Identifying corresponding lambda at %d.\n", i + 1);
			} else if (step == 1) {
				lambda = table[i];
				//printf_red();
				//printf("Found lambda: %f\n\n", lambda);
				//printf_std();
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
			//else if (step == 4)
			//	throw CausetException("omega12 value not found in geodesic lookup table.\n");
			//else if (step == 1)
			//	throw std::exception();
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
void quicksort(Node &nodes, const int &dim, const Manifold &manifold, int low, int high)
{
	if (DEBUG) {
		assert (!nodes.crd->isNull());
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
	}

	int i, j, k;
	float key = 0.0;

	if (low < high) {
		k = (low + high) >> 1;
		swap(nodes, dim, manifold, low, k);
		if (dim == 1)
			key = nodes.crd->x(low);
		else if (dim == 3)
			key = nodes.crd->w(low);
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && ((dim == 3 ? nodes.crd->w(i) : nodes.crd->x(i)) <= key))
				i++;
			while ((j >= low) && ((dim == 3 ? nodes.crd->w(j) : nodes.crd->x(j)) > key))
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
void swap(Node &nodes, const int &dim, const Manifold &manifold, const int i, const int j)
{
	if (DEBUG) {
		assert (!nodes.crd->isNull());
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
	}

	if (dim == 1) {
		float2 hc = nodes.crd->getFloat2(i);
		nodes.crd->setFloat2(nodes.crd->getFloat2(j), i);
		nodes.crd->setFloat2(hc, j);
	} else if (dim == 3) {
		float4 sc = nodes.crd->getFloat4(i);
		nodes.crd->setFloat4(nodes.crd->getFloat4(j), i);
		nodes.crd->setFloat4(sc, j);
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
void swap(uint64_t *edges, const int i, const int j)
{
	uint64_t tmp = edges[i];
	edges[i] = edges[j];
	edges[j] = tmp;
}

//Exchanige references to two lists
//as well as related indices (used in causet_intersection)
void swap(const int * const *& list0, const int * const *& list1, int &idx0, int &idx1, int &max0, int &max1)
{
	const int * const * tmp_list = list0;
	list0 = list1;
	list1 = tmp_list;

	int tmp = idx0;
	idx0 = idx1;
	idx1 = tmp;

	tmp = max0;
	max0 = max1;
	max1 = tmp;
}

//Bisection Method
//Use when Newton-Raphson fails
bool bisection(double (*solve)(const double &x, const double * const p1, const float * const p2, const int * const p3), double *x, const int max_iter, const double lower, const double upper, const double tol, const bool increasing, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG)
		assert (solve != NULL);

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
	if (DEBUG)
		assert (solve != NULL);

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
bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, int past_idx, int future_idx)
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
		assert (past_idx != future_idx);

		assert (!(future_edge_row_start[past_idx] == -1 && nodes.k_out[past_idx] > 0));
		assert (!(future_edge_row_start[past_idx] != -1 && nodes.k_out[past_idx] == 0));
	}

	int core_limit = static_cast<int>((core_edge_fraction * N_tar));
	int i;

	//Make sure past_idx < future_idx
	if (past_idx > future_idx) {
		int temp = past_idx;
		past_idx = future_idx;
		future_idx = temp;
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
	if (DEBUG) {
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
	}

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

//Intersection of Sorted Lists
//Used to find the cardinality of an interval
//Complexity: O(k*log(k))
void causet_intersection(int &elements, const int * const past_edges, const int * const future_edges, const int &k_i, const int &k_o, const int &max_cardinality, const int &pstart, const int &fstart, bool &too_many)
{
	if (DEBUG) {
		assert (past_edges != NULL);
		assert (future_edges != NULL);
		assert (k_i >= 0);
		assert (k_o >= 0);
		assert (!(k_i == 0 && k_o == 0));
		assert (max_cardinality > 1);
		assert (pstart >= 0);
		assert (fstart >= 0);
	}

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
			continue;

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
	printf_std();
	if (elements == 2)
		exit(99);*/
}

//Data formatting used when reading the degree
//sequences found on the GPU
void readDegrees(int * const &degrees, const int * const h_k, const size_t &offset, const size_t &size)
{
	if (DEBUG) {
		assert (degrees != NULL);
		assert (h_k != NULL);
	}

	unsigned int i;
	for (i = 0; i < size; i++)
		degrees[offset+i] += h_k[i];
}

//Data formatting used when reading output of
//the adjacency list created by the GPU
void readEdges(uint64_t * const &edges, const bool * const h_edges, bool * const core_edge_exists, int * const &g_idx, const unsigned int &core_limit, const size_t &d_edges_size, const size_t &mthread_size, const size_t &size0, const size_t &size1, const int x, const int y)
{
	if (DEBUG) {
		assert (edges != NULL);
		assert (h_edges != NULL);
		assert (core_edge_exists != NULL);
		assert (g_idx != NULL);
		assert (*g_idx >= 0);
		assert (x >= 0);
		assert (y >= 0);
		assert (x <= y);
	}

	uint64_t idx1, idx2;
	unsigned int i, j;

	for (i = 0; i < size0; i++) {
		for (j = 0; j < size1; j++) {
			if (h_edges[i*mthread_size+j] && g_idx[0] < (int)d_edges_size) {
				edges[g_idx[0]++] = (static_cast<uint64_t>(x*mthread_size+i)) << 32 | (static_cast<uint64_t>(y*mthread_size+j));
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
	if (!cmpi.rank)
		MPI_Reduce(MPI_IN_PLACE, &cmpi.fail, cmpi.num_mpi_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&cmpi.fail, NULL, cmpi.num_mpi_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cmpi.fail, cmpi.num_mpi_threads, MPI_INT, 0, MPI_COMM_WORLD);
	#endif

	if (cmpi.fail)
		return true;
	else
		return false;
}
