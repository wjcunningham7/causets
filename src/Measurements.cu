#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const Bitvector &adj, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &calc_autocorr, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (N_deg2 > 0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	#endif

	float c_avg = 0.0f;

	stopwatchStart(&sMeasureClustering);

	try {
		clustering = (float*)malloc(sizeof(float) * N_tar);
		if (clustering == NULL)
			throw std::bad_alloc();
		memset(clustering, 0, sizeof(float) * N_tar);
		ca->hostMemUsed += sizeof(float) * N_tar;
	} catch (std::bad_alloc()) {
		fprintf(stderr, "Failed to allocate memory in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Clustering", ca->hostMemUsed, ca->devMemUsed, 0);

	//i represents the node we are calculating the clustering coefficient for (node #1 in triplet)
	//j represents the second node in the triplet
	//k represents the third node in the triplet
	//j and k are not interchanging or else the number of triangles would be doubly counted


	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 1) reduction(+ : c_avg)
	#endif
	for (int i = 0; i < N_tar; i++) {
		//printf("\nNode %d:\n", i);
		//printf("\tDegrees: %d\n", (nodes.k_in[i] + nodes.k_out[i]));
		//printf("\t\tIn-Degrees: %d\n", nodes.k_in[i]);
		//printf("\t\tOut-Degrees: %d\n", nodes.k_out[i]);
		//fflush(stdout);

		//Ingore nodes of degree 0 and 1
		if (nodes.k_in[i] + nodes.k_out[i] < 2) {
			clustering[i] = 0.0f;
			continue;
		}

		#if DEBUG
		assert (!(edges.past_edge_row_start[i] == -1 && nodes.k_in[i] > 0));
		assert (!(edges.past_edge_row_start[i] != -1 && nodes.k_in[i] == 0));
		assert (!(edges.future_edge_row_start[i] == -1 && nodes.k_out[i] > 0));
		assert (!(edges.future_edge_row_start[i] != -1 && nodes.k_out[i] == 0));
		#endif

		float c_i = 0.0f;
		float c_k = static_cast<float>((nodes.k_in[i] + nodes.k_out[i]));
		float c_max = c_k * (c_k - 1.0f) / 2.0f;

		//(1) Consider both neighbors in the past
		if (edges.past_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_in[i]; j++)
				//3 < 2 < 1
				for (int k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.past_edges[edges.past_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(2) Consider both neighbors in the future
		if (edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				//1 < 3 < 2
				for (int k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, edges.future_edges[edges.future_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(3) Consider one neighbor in the past and one in the future
		if (edges.past_edge_row_start[i] != -1 && edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				for (int k = 0; k < nodes.k_in[i]; k++)
					//3 < 1 < 2
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		#if DEBUG
		assert (c_max > 0.0f);
		#endif
		c_i = c_i / c_max;
		#if DEBUG
		assert (c_i <= 1.0f);
		#endif

		clustering[i] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
		//fflush(stdout);
	}

	average_clustering = c_avg / N_deg2;
	#if DEBUG
	assert (average_clustering >= 0.0f && average_clustering <= 1.0f);
	#endif

	stopwatchStop(&sMeasureClustering);

	if (!bench) {
		printf("\tCalculated Clustering Coefficients.\n");
		printf_cyan();
		printf("\t\tAverage Clustering: %f\n", average_clustering);
		printf_std();
		fflush(stdout);
		if (calc_autocorr) {
			autocorr2 acClust(5);
			for (int i = 0; i < N_tar; i++)
				acClust.accum_data(clustering[i]);
			acClust.analysis();
			std::ofstream fout("clustAutoCorr.dat");
			acClust.fout_txt(fout);
			fout.close();
			printf("\t\tCalculated Autocorrelation.\n");
			fflush(stdout);
		}
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureClustering.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Calculates the number of connected components in the graph
//as well as the size of the giant connected component
//Efficiency: O(xxx)
bool measureConnectedComponents(Node &nodes, const Edge &edges, const Bitvector &adj, const int &N_tar, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}
	assert (ca != NULL);
	assert (N_tar > 0);
	#endif

	int rank = cmpi.rank;
	int elements;
	int i;

	stopwatchStart(&sMeasureConnectedComponents);

	try {
		nodes.cc_id = (int*)malloc(sizeof(int) * N_tar);
		if (nodes.cc_id == NULL) {
			cmpi.fail = 1;
			goto MccPoint;
		}
		memset(nodes.cc_id, 0, sizeof(int) * N_tar);
		ca->hostMemUsed += sizeof(int) * N_tar;

		MccPoint:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Components", ca->hostMemUsed, ca->devMemUsed, rank);

	if (!rank) {
		for (i = 0; i < N_tar; i++) {
			elements = 0;
			if (!nodes.cc_id[i] && (nodes.k_in[i] + nodes.k_out[i]) > 0) {
				if (!use_bit)
					bfsearch(nodes, edges, i, ++N_cc, elements);
				else
					bfsearch_v2(nodes, adj, N_tar, i, ++N_cc, elements);
			}
			if (elements > N_gcc)
				N_gcc = elements;
		}
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(nodes.cc_id, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N_cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N_gcc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	#endif

	stopwatchStop(&sMeasureConnectedComponents);

	#if DEBUG
	assert (N_cc > 0);
	assert (N_gcc > 1);
	#endif

	if (!bench) {
		printf_mpi(rank, "\tCalculated Number of Connected Components.\n");
		if (rank == 0) printf_cyan();
		printf_mpi(rank, "\t\tIdentified %d Components.\n", N_cc);
		printf_mpi(rank, "\t\tSize of Giant Component: %d\n", N_gcc);
		if (rank == 0) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sMeasureConnectedComponents.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Calculates the Success Ratio using N_sr Unique Pairs of Nodes
//O(xxx) Efficiency (revise this)
bool measureSuccessRatio(const Node &nodes, const Edge &edges, const Bitvector &adj, float &success_ratio, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const long double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW | HYPERBOLIC));

	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);

	if (get_stdim(spacetime) == 2) {
		assert (nodes.crd->getDim() == 2);
		assert (false);	//No distance algorithms written for 1+1
	} else if (get_stdim(spacetime) == 4) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}
	assert (ca != NULL);

	assert (N_tar > 0);
	assert (N_sr > 0 && N_sr <= ((uint64_t)N_tar * (N_tar - 1)) >> 1);
	if (get_manifold(spacetime) & (DUST | FLRW))
		assert (alpha > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	static const bool SR_DEBUG = false;

	int n = N_tar + N_tar % 2;
	uint64_t max_pairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	#if !SR_RANDOM
	uint64_t stride = static_cast<uint64_t>(max_pairs / N_sr);
	#endif
	uint64_t npairs = static_cast<uint64_t>(N_sr);
	uint64_t n_trav = 0;
	uint64_t n_succ = 0;
	uint64_t start = 0;
	uint64_t finish = npairs;

	bool *used = NULL;
	size_t u_size = sizeof(bool) * N_tar * omp_get_max_threads();
	int rank = cmpi.rank;
	bool fail = false;

	#ifdef MPI_ENABLED
	assert (false);	//MPI code has not been maintained
	//uint64_t core_edges_size = static_cast<int>(POW2(core_edge_fraction * N_tar, EXACT));
	int edges_size = static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer) / 2);
	#endif

	stopwatchStart(&sMeasureSuccessRatio);

	try {
		used = (bool*)malloc(u_size);
		if (used == NULL) {
			cmpi.fail = 1;
			goto SrPoint1;
		}
		memset(used, 0, u_size);
		ca->hostMemUsed += u_size;

		SrPoint1:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Success Ratio", ca->hostMemUsed, ca->devMemUsed, rank);

	//Debugging geodesic distance approximations
	//validateDistApprox(nodes, edges, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, compact);
	//printChk();

	/*int idx0 = 0, idx1 = 0;
	for (int i = 0; i < N_tar; i++) {
		if (fabs(nodes.crd->y(i) - 0.01) < 1E-9)
			idx0 = i;
		if (fabs(nodes.crd->y(i) - HALF_PI / 12.0) < 1E-6)
			idx1 = i;
	}
	double omega12;
	nodesAreRelated(nodes.crd, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, compact, idx0, idx1, &omega12);
	printf("omega12: %f\n", omega12);
	bool s = false, p = false;
	if (!traversePath_v1(nodes, edges, adj, &used[0], N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, core_edge_fraction, compact, idx0, idx1, s, p))
		return false;
	printf("success: %d\n", (int)s);
	printf("past hz: %d\n", (int)p);
	printChk();*/

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->x(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->y(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (get_stdim(spacetime) == 4) {
		MPI_Bcast(nodes.crd->w(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(nodes.crd->z(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	if (get_manifold(spacetime) & (DE_SITTER | FLRW))
		MPI_Bcast(nodes.id.tau, N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.k_in, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.k_out, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.past_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.past_edge_row_start, 2 * N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edge_row_start, 2 * N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	//MPI_Bcast(adj, core_edges_size, MPI_C_BOOL, 0, MPI_COMM_WORLD);

	uint64_t mpi_chunk = npairs / cmpi.num_mpi_threads;
	start = rank * mpi_chunk;
	finish = start + mpi_chunk;
	#endif

	#ifdef _OPENMP
	unsigned int seed = static_cast<unsigned int>(mrng.rng() * 4000000000);
	omp_set_nested(1);
	unsigned int nthreads = omp_get_max_threads() / 2;
	#pragma omp parallel reduction (+ : n_trav, n_succ) if (finish - start > 1000) num_threads (nthreads)
	{
	Engine eng(seed ^ omp_get_thread_num());
	UDistribution dst(0.0, 1.0);
	UGenerator rng(eng, dst);
	#pragma omp for schedule (dynamic, 2)
	#else
	UGenerator &rng = mrng.rng;
	#endif
	for (uint64_t k = start; k < finish; k++) {
		//Pick Pair
		uint64_t vec_idx;
		#if SR_RANDOM
		vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
		#else
		vec_idx = k * stride;
		#endif

		int i = static_cast<int>(vec_idx / (n - 1));
		int j = static_cast<int>(vec_idx % (n - 1) + 1);
		int do_map = i >= j;

		if (j < n >> 1) {
			i = i + do_map * ((((n >> 1) - i) << 1) - 1);
			j = j + do_map * (((n >> 1) - j) << 1);
		}

		if (j == N_tar) continue;

		if (SR_DEBUG) {
			//printf("k: %" PRIu64 "    \ti: %d    \tj: %d    \t", k, i, j);
			printf("i: %d    \tj: %d    \t", i, j);
			fflush(stdout);
		}

		//If the nodes are in different components, continue
		if (nodes.cc_id[i] != nodes.cc_id[j]) {
			if (SR_DEBUG) {
				printf("  ---\n");
				fflush(stdout);
			}
			continue;
		}

		#ifdef _OPENMP
		#pragma omp flush (fail)
		#endif
		if (fail) continue;

		//Set all nodes to "not yet used"
		int offset = N_tar * omp_get_thread_num();
		memset(used + offset, 0, sizeof(bool) * N_tar);

		//Begin Traversal from i to j
		bool success = false;
		bool past_horizon = false;
		#if TRAVERSE_V2
		if (use_bit) {
			if (!traversePath_v3(nodes, adj, &used[offset], spacetime, N_tar, a, zeta, zeta1, r_max, alpha, i, j, success))
				fail = true;
		} else if (!traversePath_v2(nodes, edges, adj, &used[offset], spacetime, N_tar, a, zeta, zeta1, r_max, alpha, core_edge_fraction, i, j, success))
				fail = true;
		#else
		if (use_bit) {
			fprintf(stderr, "traversePath_v1 not implemented for use_bit=true.  Set TRAVERSE_V2=true in inc/Constants.h\n");
			fail = true;
		} else if (!traversePath_v1(nodes, edges, adj, &used[offset], spacetime, N_tar, a, zeta, zeta1, r_max, alpha, core_edge_fraction, i, j, success, past_horizon))
			fail = true;
		#endif

		if (SR_DEBUG) {
			if (success) {
				printf_cyan();
				printf("SUCCESS\n");
			} else {
				printf_red();
				printf("FAILURE\n");
			}
			printf_std();
			fflush(stdout);
		}

		if (!past_horizon) {
			n_trav++;
			if (success)
				n_succ++;
		}
	}

	#ifdef _OPENMP
	}
	#endif

	free(used);
	used = NULL;
	ca->hostMemUsed -= u_size;

	if (fail)
		cmpi.fail = 1;
	if (checkMpiErrors(cmpi))
		return false;

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank)
		MPI_Reduce(MPI_IN_PLACE, &n_succ, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&n_succ, NULL, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

	if (!rank)
		MPI_Reduce(MPI_IN_PLACE, &n_trav, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&n_trav, NULL, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	if (!rank && n_trav)
		success_ratio = static_cast<float>(static_cast<long double>(n_succ) / n_trav);

	stopwatchStop(&sMeasureSuccessRatio);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Success Ratio.\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tSuccess Ratio: %f\n", success_ratio);
		printf_mpi(rank, "\t\tTraversed Pairs: %" PRIu64 "\n", n_trav);
		if (!rank) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sMeasureSuccessRatio.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Node Traversal Algorithm
//Returns true if the modified greedy routing algorithm successfully links 'source' and 'dest'
//Uses version 2 of the algorithm - spatial distances instead of geodesics
bool traversePath_v2(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, int source, int dest, bool &success)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW | HYPERBOLIC));

	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);

	if (get_stdim(spacetime) == 2) {
		assert (nodes.crd->getDim() == 2);
		assert (get_manifold(spacetime) & (DE_SITTER | HYPERBOLIC));
	} else if (get_stdim(spacetime) == 4) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (used != NULL);
		
	assert (N_tar > 0);
	if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
		assert (a > 0.0);
		if (get_manifold(spacetime) & (DUST | FLRW)) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (get_curvature(spacetime) & POSITIVE) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (get_curvature(spacetime) & FLAT) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}	
	if (get_curvature(spacetime) & FLAT)
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (source >= 0 && source < N_tar);
	assert (dest >= 0 && dest < N_tar);
	assert (source != dest);
	#endif

	static const bool TRAV_DEBUG = false;

	double min_dist = 0.0;
	int loc = source;
	int idx_a = source;
	int idx_b = dest;

	double dist;
	int next;
	int m;

	if (TRAV_DEBUG) {
		printf_cyan();
		printf("Beginning at %d. Looking for %d.\n", source, dest);
		printf_std();
		fflush(stdout);
	}

	//While the current location (loc) is not equal to the destination (dest)
	while (loc != dest) {
		next = loc;
		dist = INF;
		min_dist = INF;
		used[loc] = true;

		//These indicate corrupted data
		#if DEBUG
		assert (!(edges.past_edge_row_start[loc] == -1 && nodes.k_in[loc] > 0));
		assert (!(edges.past_edge_row_start[loc] != -1 && nodes.k_in[loc] == 0));
		assert (!(edges.future_edge_row_start[loc] == -1 && nodes.k_out[loc] > 0));
		assert (!(edges.future_edge_row_start[loc] != -1 && nodes.k_out[loc] == 0));
		#endif

		//(1) Check if destination is a neigbhor
		if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N_tar, core_edge_fraction, loc, idx_b)) {
			if (TRAV_DEBUG) {
				printf_cyan();
				printf("Moving to %d.\n", idx_a);
				printf_yel();
				printf("SUCCESS\n");
				printf_std();
				fflush(stdout);
			}
			success = true;
			return true;
		}

		//(2) Check minimal past relations
		for (m = 0; m < nodes.k_in[loc]; m++) {
			idx_a = edges.past_edges[edges.past_edge_row_start[loc]+m];
			/*if (TRAV_DEBUG) {
				printf_cyan();
				printf("\tConsidering past neighbor %d.\n", idx_a);
				printf_std();
				fflush(stdout);
			}*/

			//Continue if not a minimal element
			if (!!nodes.k_in[idx_a])
				continue;

			//Otherwise find the minimal element closest to the destination
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW))
				nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b, &dist);
			else if (get_manifold(spacetime) & HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
			else
				return false;

			//Save the minimum distance
			if (dist <= min_dist) {
				min_dist = dist;
				next = idx_a;
			}
		}

		//(3) Check maximal future relations
		for (m = 0; m < nodes.k_out[loc]; m++) {
			idx_a = edges.future_edges[edges.future_edge_row_start[loc]+m];
			/*if (TRAV_DEBUG) {
				printf_cyan();
				printf("\tConsidering future neighbor %d.\n", idx_a);
				printf_std();
				fflush(stdout);
			}*/

			//Continue if not a maximal element
			if (!!nodes.k_out[idx_a])
				continue;

			//Otherwise find the minimal element closest to the destination
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW))
				nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, idx_a, idx_b, &dist);
			else if (get_manifold(spacetime) & HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), spacetime, zeta);
			else
				return false;

			//Save the minimum distance
			if (dist <= min_dist) {
				min_dist = dist;
				next = idx_a;
			}
		}

		if (TRAV_DEBUG && min_dist + 1.0 < INF) {
			printf_cyan();
			printf("Moving to %d.\n", next);
			printf_std();
			fflush(stdout);
		}

		if (!used[next] && min_dist + 1.0 < INF)
			loc = next;
		else {
			if (TRAV_DEBUG) {
				printf_red();
				printf("FAILURE\n");
				printf_std();
				fflush(stdout);
			}
			break;
		}
	}

	success = false;
	return true;
}

//Node Traversal Algorithm
//Returns true if the modified greedy routing algorithm successfully links 'source' and 'dest'
//Uses version 3 of the algorithm - this uses only the adjacency matrix
bool traversePath_v3(const Node &nodes, const Bitvector &adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int source, int dest, bool &success)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW | HYPERBOLIC));

	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);

	if (get_stdim(spacetime) == 2) {
		assert (nodes.crd->getDim() == 2);
		assert (get_manifold(spacetime) & (DE_SITTER | HYPERBOLIC));
	} else if (get_stdim(spacetime) == 4) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (used != NULL);
		
	assert (N_tar > 0);
	if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
		assert (a > 0.0);
		if (get_manifold(spacetime) & (DUST | FLRW)) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (get_curvature(spacetime) & POSITIVE) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (get_curvature(spacetime) & FLAT) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}	
	if (get_curvature(spacetime) & FLAT)
		assert (r_max > 0.0);
	assert (source >= 0 && source < N_tar);
	assert (dest >= 0 && dest < N_tar);
	//assert (source != dest);
	#endif

	static const bool TRAV_DEBUG = false;

	uint64_t row;//, block;
	double dist;
	int next;

	if (TRAV_DEBUG) {
		printf_cyan();
		printf("Beginning at %d. Looking for %d.\n", source, dest);
		printf_std();
		fflush(stdout);
	}

	//While the current location (loc) is not equal to the destination (dest)
	double min_dist = 0.0;
	int loc = source;
	bool retval = true;
	while (loc != dest) {
		next = loc;
		row = static_cast<uint64_t>(loc) * N_tar;
		//block = static_cast<uint64_t>(ceil(static_cast<long double>(row + 1ULL) / (sizeof(BlockType) * CHAR_BIT))) - 1ULL;
		dist = INF;
		min_dist = INF;
		used[loc] = true;

		//(1) Check if destination is a neigbhor
		if (nodesAreConnected_v2(adj, N_tar, loc, dest)) {
			if (TRAV_DEBUG) {
				printf_cyan();
				printf("Moving to %d.\n", dest);
				printf_yel();
				printf("SUCCESS\n");
				printf_std();
				fflush(stdout);
			}
			success = true;
			return true;
		}

		#ifdef _OPENMP
		#pragma omp parallel for firstprivate (dist) schedule (dynamic, 4) num_threads(4)
		#endif
		for (int m = 0; m < N_tar; m++) {
			//Continue if 'loc' is not connected to anything in the whole block
			/*if (!(m % (sizeof(BlockType) * CHAR_BIT))) {
				if (!adj.readBlock(block + m / (sizeof(BlockType) * CHAR_BIT)))
					continue;
			}*/

			//Continue if 'loc' is not connected to 'm'
			if (!adj[row].read(m))
				continue;

			//Continue if not a minimal/maximal element
			if ((m < loc && !!nodes.k_in[m]) || (m > loc && !!nodes.k_out[m]))
				continue;

			//Otherwise find the minimal/maximal element closest to the destination
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW))
				nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, m, dest, &dist);
			else if (get_manifold(spacetime) & HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(m), nodes.crd->getFloat2(dest), spacetime, zeta);
			else
				#ifdef _OPENMP
				#pragma omp critical
				#endif
				retval = false;

			//Save the minimum distance
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				if (dist <= min_dist) {
					min_dist = dist;
					next = m;
				}
			}
		}

		if (TRAV_DEBUG && min_dist + 1.0 < INF) {
			printf_cyan();
			printf("Moving to %d.\n", next);
			printf_std();
			fflush(stdout);
		}

		if (!used[next] && min_dist + 1.0 < INF)
			loc = next;
		else {
			if (TRAV_DEBUG) {
				printf_red();
				printf("FAILURE\n");
				printf_std();
				fflush(stdout);
			}
			break;
		}
	}

	success = false;
	return retval;
}

//Takes N_df measurements of in-degree and out-degree fields at time tau_m
//O(xxx) Efficiency (revise this)
bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const unsigned int &spacetime, const int &N_tar, int &N_df, const double &tau_m, const double &a, const double &zeta, const double &zeta1, const double &alpha, const double &delta, CaResources * const ca, Stopwatch &sMeasureDegreeField, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (c->getDim() == 4);
	assert (!c->isNull());
	assert (c->w() != NULL);
	assert (c->x() != NULL);
	assert (c->y() != NULL);
	assert (c->z() != NULL);
	assert (ca != NULL);

	assert (N_tar > 0);
	assert (N_df > 0);
	assert (tau_m > 0.0);
	assert (get_stdim(spacetime) == 4);
	assert (get_manifold(spacetime) & (DE_SITTER | FLRW));
	assert (a > 0.0);
	if (get_manifold(spacetime) & DE_SITTER) {
		if (get_curvature(spacetime) & POSITIVE) {
			assert (HALF_PI > 0.0);
			assert (HALF_PI < HALF_PI);
		} else if (get_curvature(spacetime) & FLAT) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (get_manifold(spacetime) & FLRW) {
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	#endif

	double *table;
	float4 test_node;
	double eta_m;
	double d_size/*, x, rval*/;
	float dt = 0.0f, dx = 0.0f;
	long size = 0L;
	int k_in, k_out;
	int i, j;

	//Numerical Integration Parameters
	double *params = NULL;

	//Calculate theoretical values
	double k_in_theory = 0.0;
	double k_out_theory = 0.0;
	bool theoretical = (get_manifold(spacetime) & FLRW) && verbose;

	//Modify number of samples
	N_df = 1;

	IntData idata = IntData();
	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-5;
	if (get_manifold(spacetime) & FLRW && (USE_GSL || theoretical))
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	stopwatchStart(&sMeasureDegreeField);

	//Allocate memory for data
	try {
		in_degree_field = (int*)malloc(sizeof(int) * N_df);
		if (in_degree_field == NULL)
			throw std::bad_alloc();
		memset(in_degree_field, 0, sizeof(int) * N_df);
		ca->hostMemUsed += sizeof(int) * N_df;

		out_degree_field = (int*)malloc(sizeof(int) * N_df);
		if (out_degree_field == NULL)
			throw std::bad_alloc();
		memset(out_degree_field, 0, sizeof(int) * N_df);
		ca->hostMemUsed += sizeof(int) * N_df;

		if (theoretical) {
			if (!getLookupTable("./etc/tables/ctuc_table.cset.bin", &table, &size))
				return false;
			ca->hostMemUsed += size;

			params = (double*)malloc(size + sizeof(double) * 4);
			if (params == NULL)
				throw std::bad_alloc();
			ca->hostMemUsed += size + sizeof(double) * 4;
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Degree Fields", ca->hostMemUsed, ca->devMemUsed, 0);
	
	//Calculate eta_m
	if (get_manifold(spacetime) & FLRW) {
		if (USE_GSL) {
			//Numerical Integration
			idata.upper = tau_m;
			eta_m = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * a / alpha;
		} else
			//Exact Solution
			eta_m = tauToEtaFLRWExact(tau_m, a, alpha);
	} else if (get_manifold(spacetime) & DE_SITTER) {
		if (get_curvature(spacetime) & POSITIVE)
			eta_m = tauToEtaSph(tau_m);
		else if (get_curvature(spacetime) & FLAT)
			eta_m = tauToEtaFlat(tau_m);
	} else
		eta_m = 0.0;
	test_node.w = static_cast<float>(eta_m);
	
	if (theoretical) {	
		d_size = static_cast<double>(size);
		memcpy(params, &eta_m, sizeof(double));
		memcpy(params + 1, &a, sizeof(double));
		memcpy(params + 2, &alpha, sizeof(double));
		memcpy(params + 3, &d_size, sizeof(double));
		memcpy(params + 4, table, size);
	
		idata.limit = 100;
		idata.tol = 1e-4;
	
		//Theoretical Average In-Degree
		idata.lower = 0.0;
		idata.upper = eta_m;
		k_in_theory = (4.0 * M_PI * delta * POW2(POW2(alpha, EXACT), EXACT) / 3.0) * integrate1D(&degreeFieldTheory, params, &idata, QAGS);

		//Theoretical Average Out-Degree
		idata.lower = eta_m;
		idata.upper = HALF_PI - zeta;
		k_out_theory = (4.0 * M_PI * delta * POW2(POW2(alpha, EXACT), EXACT) / 3.0) * integrate1D(&degreeFieldTheory, params, &idata, QAGS);

		free(params);
		params = NULL;
		ca->hostMemUsed -= size + sizeof(double) * 4;

		free(table);
		table = NULL;
	}

	//Take N_df measurements of the fields
	for (i = 0; i < N_df; i++) {
		test_node.x = 1.0f;
		test_node.y = 1.0f;
		test_node.z = 1.0f;

		k_in = 0;
		k_out = 0;

		//Compare test node to N_tar other nodes
		float4 new_node;
		for (j = 0; j < N_tar; j++) {
			//Calculate sign of spacetime interval
			new_node = c->getFloat4(j);
			dt = static_cast<float>(ABS(static_cast<double>(c->w(j) - test_node.w), STL));

			if (get_curvature(spacetime) & POSITIVE) {
				#if DIST_V2
					dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v2(new_node, test_node)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
				#else
					dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v1(new_node, test_node)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
				#endif
			} else if (get_curvature(spacetime) & FLAT) {
				#if DIST_V2
					dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v2(new_node, test_node)), APPROX ? BITWISE : STL));
				#else
					dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v1(new_node, test_node)), APPROX ? BITWISE : STL));
				#endif
			}

			if (dx < dt) {
				//They are connected
				if (new_node.w < test_node.w)
					k_in++;
				else
					k_out++;
			}
		}

		//Save measurements
		in_degree_field[i] = k_in;
		out_degree_field[i] = k_out;

		avg_idf += k_in;
		avg_odf += k_out;
	}

	//Normalize averages
	avg_idf /= N_df;
	avg_odf /= N_df;

	stopwatchStop(&sMeasureDegreeField);

	if (get_manifold(spacetime) & FLRW && (USE_GSL || theoretical))
		gsl_integration_workspace_free(idata.workspace);

	if (!bench) {
		printf("\tCalculated Degree Field Values.\n");
		printf_cyan();
		printf("\t\tMeasurement Time: %f\n", tau_m);
		printf("\t\tAverage In-Degree Field: %f\n", avg_idf);
		if (theoretical) {
			printf_red();
			printf("\t\t\tTheory: %f\n", k_in_theory);
			printf_cyan();
		}
		printf("\t\tAverage Out-Degree Field: %f\n", avg_odf);
		if (theoretical) {
			printf_red();
			printf("\t\t\tTheory: %f\n", k_out_theory);
			printf_std();
		}
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureDegreeField.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Measure Causal Set Action
//Supports MPI Parallelization
//Algorithm has been optimized using minimal bitwise operations
//Requires the existence of the whole adjacency matrix
//This will calculate all cardinality intervals by construction
bool measureAction_v4(int *& cardinalities, float &action, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, CausetMPI &cmpi, CaResources * const ca, Stopwatch sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER));
	assert (N_tar > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	#ifdef MPI_ENABLED
	if (!cmpi.rank) printf_mag();
	printf_mpi(cmpi.rank, "Using Version 4.\n");
	if (!cmpi.rank) printf_std();
	fflush(stdout); sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);
	#else
	printf_dbg("Using Version 4.\n");
	#endif

	static const bool ACTION_DEBUG = true;
	static const bool ACTION_DEBUG_VERBOSE = false;

	Bitvector workspace;
	uint64_t clone_length = adj[0].getNumBlocks();
	double lk = 2.0;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements and workspace
	try {
		cardinalities = (int*)malloc(sizeof(int) * (N_tar - 1) * omp_get_max_threads());
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(int) * (N_tar - 1) * omp_get_max_threads());
		ca->hostMemUsed += sizeof(int) * (N_tar - 1) * omp_get_max_threads();

		workspace.reserve(omp_get_max_threads());
		for (int i = 0; i < omp_get_max_threads(); i++) {
			FastBitset fb(static_cast<uint64_t>(N_tar));
			workspace.push_back(fb);
			ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Construct set of permutations
	unsigned int nbuf = cmpi.num_mpi_threads << 1;
	std::vector<unsigned int> ordered(nbuf);
	std::iota(ordered.begin(), ordered.end(), 0);
	std::vector<unsigned int> current = ordered;
	FastBitset fb_ordered(static_cast<uint64_t>(current.size())*(current.size()-1)>>1);
	perm_to_binary(fb_ordered, ordered);
	std::unordered_set<FastBitset> permutations;
	init_mpi_permutations(permutations, current);

	//Print permutations to stdout
	if (ACTION_DEBUG) {
		printf_mpi(cmpi.rank, "List of Permutations:\n");
		for (std::unordered_set<FastBitset>::iterator fb = permutations.begin(); fb != permutations.end(); fb++) {
			std::vector<unsigned int> p;
			binary_to_perm(p, *fb, nbuf);
			if (!cmpi.rank) print_pairs(p);
		}
		printf_mpi(cmpi.rank, "\n");
		fflush(stdout); sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//fflush(stdout);
	//sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);
	//printf("Rank [%d] Reporting!\n", cmpi.rank);
	//printChk();

	//DEBUG
	/*int length = N_tar / cmpi.num_mpi_threads;
	Bitvector adj_test;
	adj_test.reserve(length);
	for (int i = 0; i < length; i++) {
		FastBitset fb((uint64_t)length);
		adj_test.push_back(fb);
	}*/
	//END DEBUG

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Action", ca->hostMemUsed, ca->devMemUsed, cmpi.rank);

	int N_eff = N_tar / cmpi.num_mpi_threads;
	uint64_t npairs = static_cast<uint64_t>(N_eff) * (N_eff - 1) >> 1;
	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 64) if (npairs > 10000)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / (N_eff - 1);
		uint64_t j = k % (N_eff - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((N_eff >> 1) - i) << 1) - 1);
		j += do_map * (((N_eff >> 1) - j) << 1);

		uint64_t glob_i = i + cmpi.rank * N_eff;
		uint64_t glob_j = j + cmpi.rank * N_eff;
		if (glob_i == glob_j) continue;
		//printf("Rank [%d] is examining [%" PRIu64 " - %" PRIu64 "]\n", cmpi.rank, glob_i, glob_j);
		if (!nodesAreConnected_v2(adj, N_tar, static_cast<int>(i), static_cast<int>(glob_j))) continue;
		//printf("Nodes are connected!\n");

		//DEBUG
		//adj_test[i].set(glob_j);

		uint64_t length = glob_j - glob_i + 1;
		adj[i].clone(workspace[tid], 0ULL, clone_length);
		workspace[tid].partial_intersection(adj[j], glob_i, length);
		uint64_t cnt = workspace[tid].partial_count(glob_i, length);
		if (ACTION_DEBUG_VERBOSE)
			printf("\tCounted [%" PRIu64 "] elements in the interval [%d - %d]\n", cnt, glob_i, glob_j);
		//cardinalities[tid*(N_tar-1)+workspace[tid].partial_count(glob_i, length)+1]++;
		cardinalities[tid*(N_tar-1)+cnt+1]++;
	}
	/*sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Rank [%d] Reporting!\n", cmpi.rank);
	printChk();*/
	//fflush(stdout); sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);

	//goto DebugPoint;

	//MPI Swaps
	N_eff >>= 1;
	npairs = static_cast<uint64_t>(N_eff) * N_eff;
	while (permutations.size() > 0) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Current permutation:\t\t");
			for (size_t i = 0; i < current.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", current[i], current[i+1]);
			printf_mpi(cmpi.rank, "\n");
			if (!cmpi.rank) fflush(stdout);
		}

		std::vector<unsigned int> next;
		FastBitset fb = *permutations.begin();
		binary_to_perm(next, *permutations.begin(), nbuf);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Next permutation:\t\t");
			for (size_t i = 0; i < next.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", next[i], next[i+1]);
			printf_mpi(cmpi.rank, "\n");
		}

		if (fb == fb_ordered) {
			if (ACTION_DEBUG)
				printf_mpi(cmpi.rank, "Identified an ordered permutation. Skipping this one.\n");
			fflush(stdout); sleep(1);
			permutations.erase(permutations.begin());
			continue;
		}

		std::vector<std::vector<unsigned int> > similar;
		fill_mpi_similar(similar, next);
		/*printf_mpi(cmpi.rank, "Similar Configurations:\n");
		for (size_t i = 0; i < similar.size(); i++) {
			for (size_t j = 0; j < similar[i].size(); j += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", similar[i][j], similar[i][j+1]);
			printf_mpi(cmpi.rank, "\n");
		}
		printf_mpi(cmpi.rank, "\n");
		fflush(stdout); sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);*/

		unsigned int min_steps;
		get_most_similar(next, min_steps, similar, current);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Most Similar Configuration:\t");
			for (size_t i = 0; i < next.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", next[i], next[i+1]);
			printf_mpi(cmpi.rank, "\n\n");
			if (!cmpi.rank) fflush(stdout);
		}

		relabel_vector(current, next);
		/*printf("Relabeled current vector: ");
		for (size_t i = 0; i < current.size(); i += 2)
			printf("(%d, %d) ", current[i], current[i+1]);
		printf("\n\n");*/
	
		std::vector<std::pair<int,int> > swaps;
		cyclesort(min_steps, current, &swaps);
		/*printf_mpi(cmpi.rank, "Sequence of Swaps:\n");
		for (size_t i = 0; i < swaps.size(); i++)
			printf_mpi(cmpi.rank, "[%d] --> [%d]\n", std::get<0>(swaps[i]), std::get<1>(swaps[i]));
		printf_mpi(cmpi.rank, "\n");
		if (!cmpi.rank) fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);*/

		/*printAdjMatrix(adj, N_tar, "adj_matrix.cset.dbg.dat", cmpi.num_mpi_threads, cmpi.rank);
		sleep(1);
		printf_mpi(cmpi.rank, "\n");
		if (!cmpi.rank) fflush(stdout);
		sleep(1);*/
		mpi_swaps(swaps, adj, cmpi.adj_buf, N_tar, cmpi.num_mpi_threads, cmpi.rank);
		current = next;
		/*printf_mpi(cmpi.rank, "\n");
		if (!cmpi.rank) fflush(stdout);
		sleep(1);
		printAdjMatrix(adj, N_tar, "adj_matrix2.cset.dbg.dat", cmpi.num_mpi_threads, cmpi.rank);*/

		//Compare pairs between the two buffers
		//printf_mpi(cmpi.rank, "npairs: %" PRIu64 "\n", npairs);
		#ifdef _OPENMP
		#pragma omp parallel for schedule (dynamic, 64) if (npairs > 10000)
		#endif
		for (uint64_t k = 0; k < npairs; k++) {
			unsigned int tid = omp_get_thread_num();
			//Choose a pair
			uint64_t i = k / N_eff;
			uint64_t j = k % N_eff;
			j += N_eff;

			uint64_t glob_i = loc_to_glob_idx(next, i, N_tar, cmpi.num_mpi_threads, cmpi.rank);
			uint64_t glob_j = loc_to_glob_idx(next, j, N_tar, cmpi.num_mpi_threads, cmpi.rank);
			if (glob_i == glob_j) continue;

			if (glob_i > glob_j) {
				glob_i ^= glob_j;
				glob_j ^= glob_i;
				glob_i ^= glob_j;

				i ^= j;
				j ^= i;
				i ^= j;
			}

			//printf("Rank [%d] is examining [%" PRIu64 " - %" PRIu64 "]\n", cmpi.rank, glob_i, glob_j);
			if (!nodesAreConnected_v2(adj, N_tar, static_cast<int>(i), static_cast<int>(glob_j))) continue;
			//printf("Nodes are connected!\n");

			//DEBUG
			//adj_test[i].set(glob_j);

			uint64_t length = glob_j - glob_i + 1;
			adj[i].clone(workspace[tid], 0ULL, clone_length);
			workspace[tid].partial_intersection(adj[j], glob_i, length);
			uint64_t cnt = workspace[tid].partial_count(glob_i, length);
			if (ACTION_DEBUG_VERBOSE)
				printf("\tCounted [%" PRIu64 "] elements in the interval [%d - %d]\n", cnt, glob_i, glob_j);
			//cardinalities[tid*(N_tar-1)+workspace[tid].partial_count(glob_i, length)+1]++;
			cardinalities[tid*(N_tar-1)+cnt+1]++;
		}
		fflush(stdout); sleep(1);
		//MPI_Barrier(MPI_COMM_WORLD);
		permutations.erase(permutations.begin());
	}

	//DEBUG
	//printAdjMatrix(adj_test, N_tar, "adj_test.cset.dbg.dat", cmpi.num_mpi_threads, cmpi.rank);

	//Return to original configuration
	if (cmpi.num_mpi_threads > 1) {
		std::vector<std::pair<int,int> > swaps;
		unsigned int min_steps;
		cyclesort(min_steps, current, &swaps);
		/*printf_mpi(cmpi.rank, "Sequence of Swaps:\n");
		for (size_t i = 0; i < swaps.size(); i++)
			printf_mpi(cmpi.rank, "[%d] --> [%d]\n", std::get<0>(swaps[i]), std::get<1>(swaps[i]));
		printf_mpi(cmpi.rank, "\n");
		if (!cmpi.rank) fflush(stdout);
		sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);*/
		mpi_swaps(swaps, adj, cmpi.adj_buf, N_tar, cmpi.num_mpi_threads, cmpi.rank);
	}

	//DebugPoint:

	//Reduction for OpenMP
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N_tar - 1; j++)
			cardinalities[j] += cardinalities[i*(N_tar-1)+j];

	//Reduction for MPI
	MPI_Allreduce(MPI_IN_PLACE, cardinalities, N_tar - 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	//The first element will be N_tar
	cardinalities[0] = N_tar;

	/*for (int i = 0; i < N_tar - 1; i++)
		printf("N[%d] = %d (rank [%d])\n", i, cardinalities[i], cmpi.rank);
	fflush(stdout); sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);*/

	//Free Workspace
	ca->hostMemUsed -= sizeof(BlockType) * clone_length * omp_get_max_threads();
	workspace.clear();
	workspace.swap(workspace);

	action = calcAction(cardinalities, get_stdim(spacetime), lk, true);
	assert (action == action);

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf_mpi(cmpi.rank, "\tCalculated Action.\n");
		printf_mpi(cmpi.rank, "\t\tTerms Used: %d\n", N_tar - 1);
		if (!cmpi.rank) printf_cyan();
		printf_mpi(cmpi.rank, "\t\tCausal Set Action: %f\n", action);
		if (!cmpi.rank) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(cmpi.rank, "\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	/*fflush(stdout);
	sleep(1);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Rank [%d] Reporting!\n", cmpi.rank);
	printChk();*/

	return true;
}

//Measure Causal Set Action
//Algorithm has been optimized using minimal bitwise operations
//Requires the existence of the whole adjacency matrix
//This will calculate all cardinality intervals by construction
bool measureAction_v3(int *& cardinalities, float &action, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, CaResources * const ca, Stopwatch sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER));
	assert (N_tar > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	printf_dbg("Using Version 3.\n");

	Bitvector workspace;
	int n = N_tar + N_tar % 2;
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	uint64_t clone_length = adj[0].getNumBlocks();
	double lk = 2.0;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements and workspace
	try {
		cardinalities = (int*)malloc(sizeof(int) * (N_tar - 1) * omp_get_max_threads());
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(int) * (N_tar - 1) * omp_get_max_threads());
		ca->hostMemUsed += sizeof(int) * (N_tar - 1) * omp_get_max_threads();

		workspace.reserve(omp_get_max_threads());
		for (int i = 0; i < omp_get_max_threads(); i++) {
			FastBitset fb(static_cast<uint64_t>(N_tar));
			workspace.push_back(fb);
			ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Action", ca->hostMemUsed, ca->devMemUsed, 0);

	//The first element will be N_tar
	cardinalities[0] = N_tar;

	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 64) if (npairs > 10000)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;

		if (static_cast<int>(j) < n >> 1) {
			i = i + do_map * ((((n >> 1) - i) << 1) - 1);
			j = j + do_map * (((n >> 1) - j) << 1);
		}

		if (static_cast<int>(j) == N_tar) continue;
		if (!nodesAreConnected_v2(adj, N_tar, static_cast<int>(i), static_cast<int>(j))) continue;

		uint64_t length = j - i + 1;
		adj[i].clone(workspace[tid], 0ULL, clone_length);
		workspace[tid].partial_intersection(adj[j], i, length);
		cardinalities[tid*(N_tar-1)+workspace[tid].partial_count(i, length)+1]++;
	}

	//Reduction for OpenMP
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N_tar - 1; j++)
			cardinalities[j] += cardinalities[i*(N_tar-1)+j];

	//Free Workspace
	ca->hostMemUsed -= sizeof(BlockType) * clone_length * omp_get_max_threads();
	workspace.clear();
	workspace.swap(workspace);

	action = calcAction(cardinalities, get_stdim(spacetime), lk, true);
	assert (action == action);

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf("\tCalculated Action.\n");
		printf("\t\tTerms Used: %d\n", N_tar - 1);
		printf_cyan();
		printf("\t\tCausal Set Action: %f\n", action);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}
