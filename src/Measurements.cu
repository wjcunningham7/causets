/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "Measurements.h"

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const Bitvector &adj, float &average_clustering, const int &N, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (edges.past_edges != NULL);
	assert (edges.future_edges != NULL);
	assert (edges.past_edge_row_start != NULL);
	assert (edges.future_edge_row_start != NULL);
	assert (ca != NULL);
	assert (N > 0);
	assert (N_deg2 > 0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	#endif

	float c_avg = 0.0f;

	stopwatchStart(&sMeasureClustering);

	//Allocate memory for clustering coefficients
	try {
		clustering = (float*)malloc(sizeof(float) * N);
		if (clustering == NULL)
			throw std::bad_alloc();
		memset(clustering, 0, sizeof(float) * N);
		ca->hostMemUsed += sizeof(float) * N;
	} catch (const std::bad_alloc&) {
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
	#pragma omp parallel for schedule (dynamic, 1) reduction(+ : c_avg) if (N > 10000)
	#endif
	for (int i = 0; i < N; i++) {
		//Ingore nodes of degree 0 and 1
		if (nodes.k_in[i] + nodes.k_out[i] < 2) {
			clustering[i] = 0.0f;
			continue;
		}

		#if DEBUG
		//Sanity checks
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
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.past_edges[edges.past_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(2) Consider both neighbors in the future
		if (edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				//1 < 3 < 2
				for (int k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, edges.future_edges[edges.future_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(3) Consider one neighbor in the past and one in the future
		if (edges.past_edge_row_start[i] != -1 && edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				for (int k = 0; k < nodes.k_in[i]; k++)
					//3 < 1 < 2
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		#if DEBUG
		assert (c_max > 0.0f);
		#endif
		//Normalization
		c_i = c_i / c_max;
		#if DEBUG
		assert (c_i <= 1.0f);
		#endif

		clustering[i] = c_i;
		c_avg += c_i;
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
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureClustering.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Calculates the number of connected components in the graph
//as well as the size of the giant connected component
//Note: While this subroutine supports MPI, it does not break up the problem across
//the MPI processes - it simply shares the results from process 0
//O(N+E) Efficiency
bool measureConnectedComponents(Node &nodes, const Edge &edges, const Bitvector &adj, const int &N, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}
	assert (ca != NULL);
	assert (N > 0);
	#endif

	int rank = cmpi.rank;
	int elements;
	int i;

	stopwatchStart(&sMeasureConnectedComponents);

	try {
		//Allocate space for component labels
		nodes.cc_id = (int*)malloc(sizeof(int) * N);
		if (nodes.cc_id == NULL) {
			cmpi.fail = 1;
			goto MccPoint;
		}
		memset(nodes.cc_id, 0, sizeof(int) * N);
		ca->hostMemUsed += sizeof(int) * N;

		MccPoint:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Components", ca->hostMemUsed, ca->devMemUsed, rank);

	if (!rank) {	//Only process 0 does work
		for (i = 0; i < N; i++) {
			elements = 0;
			//Execute D.F. search if the node is not isolated
			//A component ID of 0 indicates an isolated node
			if (!nodes.cc_id[i] && (nodes.k_in[i] + nodes.k_out[i]) > 0) {
				if (!use_bit)
					dfsearch(nodes, edges, i, ++N_cc, elements, 0);
				else
					//Version 2 uses only the adjacency matrix (no sparse lists)
					dfsearch_v2(nodes, adj, N, i, ++N_cc, elements);
			}

			//Check if this is the giant component
			if (elements > N_gcc)
				N_gcc = elements;
		}
	}

	#ifdef MPI_ENABLED
	//Share the results across MPI processes
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(nodes.cc_id, N, MPI_INT, 0, MPI_COMM_WORLD);
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
//Calculate the success ratio using a greedy routing algorithm
//If TRAVERSE_V2 is false, it will use geodesic distances to route; otherwise
//it will use spatial distances.
//O(d*N^2) Efficiency, where d is the stretch
bool measureSuccessRatio(const Node &nodes, const Edge &edges, const Bitvector &adj, float &success_ratio, float &success_ratio2, float &stretch, const Spacetime &spacetime, const int &N, const float &k_tar, const long double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &link_epso, const bool &use_bit, const bool &calc_stretch, const bool &strict_routing, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic"));

	if (spacetime.manifoldIs("Hyperbolic"))
		assert (spacetime.stdimIs("2"));

	if (spacetime.stdimIs("2")) {
		assert (nodes.crd->getDim() == 2);
		assert (TRAVERSE_V2 && use_bit && TRAVERSE_VECPROD);
	} else if (spacetime.stdimIs("4")) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	if (use_bit)
		assert (adj.size() >= (size_t)N);
	else {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}
	assert (ca != NULL);

	assert (N > 0);
	assert (N_sr > 0 && N_sr <= ((uint64_t)N * (N - 1)) >> 1);
	if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
		assert (alpha > 0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif

	static const bool SR_DEBUG = false;

	//If we aren't studying all pairs, figure out how many
	int n = N + N % 2;
	uint64_t max_pairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	#if !SR_RANDOM
	uint64_t stride = static_cast<uint64_t>(max_pairs / N_sr);
	#endif
	uint64_t npairs = static_cast<uint64_t>(N_sr);
	uint64_t n_trav = 0;
	uint64_t n_succ = 0, n_succ2 = 0;
	uint64_t start = 0;
	uint64_t finish = npairs;

	int *distances = NULL;
	bool *used = NULL;
	size_t d_size = sizeof(int) * N * omp_get_max_threads();
	size_t u_size = sizeof(bool) * N * omp_get_max_threads();
	int rank = cmpi.rank;
	bool fail = false;
	float loc_stretch = 0.0;

	stopwatchStart(&sMeasureSuccessRatio);

	try {
		//Keep track of nodes already visited to avoid cycles
		//If using the strict routing protocol, this is not used
		used = (bool*)malloc(u_size);
		if (used == NULL) {
			cmpi.fail = 1;
			goto SrPoint1;
		}
		memset(used, 0, u_size);
		ca->hostMemUsed += u_size;

		//Used for stretch calculations
		if (calc_stretch) {
			distances = (int*)malloc(d_size);
			if (distances == NULL) {
				cmpi.fail = 1;
				goto SrPoint1;
			}
			memset(distances, -1, d_size);
			ca->hostMemUsed += d_size;
		}

		SrPoint1:
		if (checkMpiErrors(cmpi)) {
			if (!rank)
				throw std::bad_alloc();
			else
				return false;
		}
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Success Ratio", ca->hostMemUsed, ca->devMemUsed, rank);

	unsigned int seed;
	unsigned int nthreads;

	uint64_t min_samples = 500;
	SuccessRatioLoop:

	#ifdef _OPENMP
	seed = static_cast<unsigned int>(mrng.rng() * 4000000000);
	nthreads = omp_get_max_threads();
	#if TRAVERSE_V2
	if (use_bit) {
		//In this case, traversePath_v3() is used, and we
		//will use nested OpenMP parallelization
		omp_set_nested(1);
		nthreads >>= 2;
	}
	#endif
	omp_set_num_threads(nthreads);
	#pragma omp parallel reduction (+ : n_trav, n_succ, n_succ2, loc_stretch) if (finish - start > 1000)
	{
	//Use independent RNG engines
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
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);

		if (j == N) continue;

		if (SR_DEBUG) {
			printf("\ni: %d    \tj: %d    \t", i, j);
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

		//If an error has arisen, continue instead of doing work
		//This is the only appropriate way to exit an OpenMP loop
		#ifdef _OPENMP
		#pragma omp flush (fail)
		#endif
		if (fail) continue;

		//Set all nodes to "not yet used"
		int offset = N * omp_get_thread_num();
		memset(used + offset, 0, sizeof(bool) * N);

		//Begin Traversal from i to j
		int nsteps = 0;
		bool success = false;
		bool past_horizon = false;
		#if TRAVERSE_V2
		if (use_bit) {
			//Version 3 uses the adjacency matrix and does spatial or vector product routing
			if (!traversePath_v3(nodes, adj, &used[offset], spacetime, N, a, zeta, zeta1, r_max, alpha, link_epso, strict_routing, i, j, success))
				fail = true;
		//Version 2 uses sparse lists and does spatial routing
		} else if (!traversePath_v2(nodes, edges, adj, &used[offset], spacetime, N, a, zeta, zeta1, r_max, alpha, core_edge_fraction, strict_routing, i, j, success))
				fail = true;
		#else
		bool success2 = true;
		if (use_bit) {
			fprintf(stderr, "traversePath_v1 not implemented for use_bit=true.  Set TRAVERSE_V2=true in inc/Constants.h\n");
			fail = true;
		//Version 1 uses sparse lists and does geodesic routing
		} else if (!traversePath_v1(nodes, edges, adj, &used[offset], spacetime, N, a, zeta, zeta1, r_max, alpha, core_edge_fraction, strict_routing, i, j, nsteps, success, success2, past_horizon))
			fail = true;
		#endif

		if (SR_DEBUG) {
			if (success) {
				printf_cyan();
				printf("SUCCESS\n");
				printf(" > steps: %d\n", nsteps);
			} else {
				printf_red();
				printf("FAILURE\n");
			}
			printf_std();
			fflush(stdout);
		}

		//past_horizon = false;	//This line means we consider all pairs regardless of method
					//This can only be set to true when studying geodesics, where
					//for some manifolds the distance can be infinite
		if (!past_horizon) {
			n_trav++;
			if (success) {
				n_succ++;
				if (calc_stretch)
					loc_stretch += nsteps / static_cast<double>(shortestPath(nodes, edges, N, &distances[offset], i, j));
			}
			#if !TRAVERSE_V2
			if (success2)		//Trivially false if past_horizon = true
				n_succ2++;
			#endif
		}
	}

	#ifdef _OPENMP
	}
	#endif

	if (n_trav < min_samples)
		goto SuccessRatioLoop;

	free(used);
	used = NULL;
	ca->hostMemUsed -= u_size;

	if (calc_stretch) {
		free(distances);
		distances = NULL;
		ca->hostMemUsed -= d_size;
	}

	if (fail)
		cmpi.fail = 1;
	if (checkMpiErrors(cmpi))
		return false;

	if (!rank && n_trav) {
		success_ratio = static_cast<float>(static_cast<long double>(n_succ) / n_trav);
		success_ratio2 = static_cast<float>(static_cast<long double>(n_succ2) / n_trav);
		if (calc_stretch)
			stretch = loc_stretch / n_succ;
	}

	stopwatchStop(&sMeasureSuccessRatio);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Success Ratio.\n");
		#if !TRAVERSE_V2
		printf_mpi(rank, "\t\tUsed Geodesic Routing.\n");
		#else
		if (!use_bit) {
		printf_mpi(rank, "\t\tUsed Spatial Routing (v2).\n");
		} else {
			#if !TRAVERSE_VECPROD
			printf_mpi(rank, "\t\tUsed Spatial Routing (v3).\n");
			#else
			printf_mpi(rank, "\t\tUsed Vector Product Routing.\n");
			#endif
		}
		#endif
		if (strict_routing)
			printf_mpi(rank, "\t\tUsed Strict Routing Protocol.\n");
		else
			printf_mpi(rank, "\t\tUsed Standard Routing Protocol.\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tSuccess Ratio (Type 1): %f\n", success_ratio);
		#if !TRAVERSE_V2
		printf_mpi(rank, "\t\tSuccess Ratio (Type 2): %f\n", success_ratio2);
		#endif
		if (calc_stretch)
			printf_mpi(rank, "\t\tStretch: %f\n", stretch);
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
bool traversePath_v2(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const bool &strict_routing, int source, int dest, bool &success)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic"));

	if (spacetime.manifoldIs("Hyperbolic"))
		assert (spacetime.stdimIs("2"));

	if (spacetime.stdimIs("2")) {
		assert (nodes.crd->getDim() == 2);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Hyperbolic"));
	} else if (spacetime.stdimIs("4")) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
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
		
	assert (N > 0);
	if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		assert (a > 0.0);
		if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (spacetime.curvatureIs("Positive")) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (spacetime.curvatureIs("Flat")) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}	
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (!strict_routing);
	assert (source >= 0 && source < N);
	assert (dest >= 0 && dest < N);
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
		used[loc] = true;	//Record the current location has been used

		#if DEBUG
		//These are sanity checks to find corrupted data
		assert (!(edges.past_edge_row_start[loc] == -1 && nodes.k_in[loc] > 0));
		assert (!(edges.past_edge_row_start[loc] != -1 && nodes.k_in[loc] == 0));
		assert (!(edges.future_edge_row_start[loc] == -1 && nodes.k_out[loc] > 0));
		assert (!(edges.future_edge_row_start[loc] != -1 && nodes.k_out[loc] == 0));
		#endif

		//(1) Check if destination is a neigbhor
		if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, loc, idx_b)) {
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

			//Continue if not a minimal element
			if (!!nodes.k_in[idx_a])
				continue;

			//Otherwise find the minimal element closest to the destination
			if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
				nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b, &dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
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

			//Continue if not a maximal element
			if (!!nodes.k_out[idx_a])
				continue;

			//Otherwise find the minimal element closest to the destination
			if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
				nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, idx_a, idx_b, &dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
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

		//Check for cycles
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
bool traversePath_v3(const Node &nodes, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &link_epso, const bool &strict_routing, int source, int dest, bool &success)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic"));
	assert (adj.size() >= (size_t)N);

	if (spacetime.manifoldIs("Hyperbolic"))
		assert (spacetime.stdimIs("2"));

	if (spacetime.stdimIs("2")) {
		assert (nodes.crd->getDim() == 2);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Hyperbolic"));
	} else if (spacetime.stdimIs("4")) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"));
	}

	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	if (!strict_routing)
		assert (used != NULL);
		
	assert (N > 0);
	if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		assert (a > 0.0);
		if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
			assert (zeta < HALF_PI);
			assert (alpha > 0.0);
		} else {
			if (spacetime.curvatureIs("Positive")) {
				assert (zeta > 0.0);
				assert (zeta < HALF_PI);
			} else if (spacetime.curvatureIs("Flat")) {
				assert (zeta > HALF_PI);
				assert (zeta1 > HALF_PI);
				assert (zeta > zeta1);
			}
		}
	}	
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	#if TRAVERSE_VECPROD
	assert (spacetime.curvatureIs("Positive") && spacetime.stdimIs("2"));
	#endif
	assert (source >= 0 && source < N);
	assert (dest >= 0 && dest < N);
	assert (source != dest);
	#endif

	static const bool TRAV_DEBUG = false;

	uint64_t row;
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
	int length = 0;
	bool retval = true;
	while (loc != dest) {
		next = loc;
		row = loc;
		dist = INF;
		if (strict_routing) {
			//The strict routing protocol says a packet only moves to a node 
			//with a distance to the destination shorter than the distance
			//between the current node and the destination
			#if TRAVERSE_VECPROD	//Use vector products in the embedded spacetime
			if (spacetime.manifoldIs("De_Sitter"))
				deSitterInnerProduct(nodes, spacetime, N, loc, dest, &min_dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
				nodesAreRelatedHyperbolic(nodes, spacetime, N, zeta, r_max, link_epso, loc, dest, &min_dist);
			#else	//Use spatial distances
			if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
				nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, loc, dest, &min_dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
				min_dist = distanceH(nodes.crd->getFloat2(loc), nodes.crd->getFloat2(dest), spacetime, zeta);
			#endif
			else
				return false;
		} else {
			min_dist = INF;
			//Record the current location is used
			used[loc] = true;
		}

		//(1) Check if destination is a neigbhor
		if (nodesAreConnected_v2(adj, N, loc, dest)) {
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
		for (int m = 0; m < N; m++) {
			//Continue if 'loc' is not connected to 'm'
			if (!adj[row].read(m))
				continue;

			//Otherwise find the element closest to the destination
			#if TRAVERSE_VECPROD
			if (spacetime.manifoldIs("De_Sitter"))
				deSitterInnerProduct(nodes, spacetime, N, m, dest, &dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
				nodesAreRelatedHyperbolic(nodes, spacetime, N, zeta, r_max, link_epso, m, dest, &dist);
			#else
			//Continue if not a minimal/maximal element (feature of spatial routing)
			if ((m < loc && !!nodes.k_in[m]) || (m > loc && !!nodes.k_out[m]))
				continue;

			if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
				nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, m, dest, &dist);
			else if (spacetime.manifoldIs("Hyperbolic"))
				dist = distanceH(nodes.crd->getFloat2(m), nodes.crd->getFloat2(dest), spacetime, zeta);
			#endif
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

		//If strict routing, make sure a move was identified
		//Otherwise, make sure the next node has not been visited already
		if (((strict_routing && next != loc) || (!strict_routing && !used[next])) && min_dist + 1.0 < INF) {
			loc = next;
			length++;
		} else {
			if (TRAV_DEBUG) {
				printf_red();
				printf("FAILURE\n");
				printf_std();
				fflush(stdout);
			}
			break;
		}
	}

	if (length >= 5) return false;

	success = false;
	return retval;
}

//Action kernel used with PTHREAD
//This is called from measureAction_v5()
void* actionKernel(void *params)
{
	bool ACTION_DEBUG = false;

	action_params *p = (action_params*)params;
	if (ACTION_DEBUG)
		printf("Rank [%d] is executing the action kernel.\n", p->rank);
	unsigned int nthreads = omp_get_max_threads();
	#ifdef AVX2_ENABLED
	nthreads >>= 1;
	#endif

	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 64) if (p->npairs > 10000) num_threads (nthreads)
	#endif
	for (uint64_t k = 0; k < p->npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / p->N_eff;
		uint64_t j = k % p->N_eff;
		j += p->N_eff;

		//Recover the global index - this is the index of an element with respect
		//to the entire adjacency matrix
		uint64_t glob_i = loc_to_glob_idx(p->current[0], i, p->N, p->num_mpi_threads, p->rank);
		uint64_t glob_j = loc_to_glob_idx(p->current[0], j, p->N, p->num_mpi_threads, p->rank);
		if (glob_i == glob_j) continue;	//No diagonal elements
		if (glob_i >= static_cast<uint64_t>(p->N) || glob_j >= static_cast<uint64_t>(p->N)) continue;

		//Ensure glob_i is always the smaller one
		if (glob_i > glob_j) {
			glob_i ^= glob_j;
			glob_j ^= glob_i;
			glob_i ^= glob_j;

			i ^= j;
			j ^= i;
			i ^= j;
		}

		//If these elements aren't even connected, don't bother finding the cardinality
		if (!nodesAreConnected_v2(p->adj[0], p->N, static_cast<int>(i), static_cast<int>(glob_j))) continue;

		//Save the cardinality
		p->cardinalities[tid*p->N+p->adj[0][i].partial_vecprod(p->adj[0][j], glob_i, glob_j - glob_i + 1)+1]++;
	}
	if (ACTION_DEBUG)
		printf("Rank [%d] has completed the action kernel.\n", p->rank);

	p->busy[0] = false;	//Update this shared variable
	pthread_exit(NULL);
}

bool measureAction_v6(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int N, const double epsilon, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureAction, const bool use_bit, const bool split_mpi, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (ca != NULL);
	assert (use_bit);
	assert (!split_mpi);
	#endif

	int rank = cmpi.rank;
	if (verbose || bench) {
		#ifdef MPI_ENABLED
		if (!rank) printf_mag();
		printf_mpi(rank, "Using Version 6 (measureAction).\n");
		if (!rank) printf_std();
		fflush(stdout); sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);
		#else
		//printf_dbg("Using Version 6 (measureAction).\n");
		#endif
	}

	int n = N + N % (2 * cmpi.num_mpi_threads);
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / (2 * cmpi.num_mpi_threads);
	uint64_t start = rank * npairs;
	uint64_t finish = start + npairs;
	unsigned int nthreads = omp_get_max_threads();

	stopwatchStart(&sMeasureAction);

	if (cardinalities == NULL) {
		try {
			cardinalities = (uint64_t*)calloc(N * nthreads, sizeof(uint64_t));
			if (cardinalities == NULL)
				throw std::bad_alloc();
			ca->hostMemUsed += sizeof(uint64_t) * N * nthreads;
		} catch (const std::bad_alloc&) {
			fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
			return false;
		}
	} else
		memset(cardinalities, 0, sizeof(uint64_t) * N * nthreads);

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, rank);

	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 256) num_threads(nthreads) if (npairs >= 1024)
	#endif
	for (uint64_t k = start; k < finish; k++) {
		unsigned int tid = omp_get_thread_num();
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);

		if (static_cast<int>(i) >= N || static_cast<int>(j) >= N || !nodesAreConnected_v2(adj, N, static_cast<int>(i), static_cast<int>(j)))
			continue;

		cardinalities[tid*N+adj[i].partial_vecprod(adj[j], i, j - i + 1)+1]++;
	}

	for (unsigned int i = 1; i < nthreads; i++)
		for (int j = 0; j < N; j++)
			cardinalities[j] += cardinalities[i*N+j];

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, cardinalities, N, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	#endif

	cardinalities[0] = N;
	if (spacetime.is_set()) {
		action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), epsilon);
		assert (action == action);
	}

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Action.\n");
		printf_mpi(rank, "\t\tTerms Used: %d\n", N);
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tCausal Set Action: %f\n", action);
		if (!rank) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Measure Causal Set Action
//Supports Asynchronous MPI Parallelization
//Algorithm has been optimized for MPI load balancing
//Requires the existence of the whole adjacency matrix
//This will calculate all cardinality intervals by construction
bool measureAction_v5(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int &N, const double epsilon, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	#ifdef MPI_ENABLED
	if (verbose || bench) {
		if (!cmpi.rank) printf_mag();
		printf_mpi(cmpi.rank, "Using Version 5 (measureAction).\n");
		if (!cmpi.rank) printf_std();
		fflush(stdout); sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	static const bool ACTION_DEBUG = false;
	static const bool ACTION_DEBUG_VERBOSE = false;
	unsigned int nbuf = cmpi.num_mpi_threads << 1;
	#else
	if (verbose || bench)
		printf_dbg("Using Version 5 (measureAction).\n");
	#endif

	int rank = cmpi.rank;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements
	try {
		cardinalities = (uint64_t*)malloc(sizeof(uint64_t) * N * omp_get_max_threads());
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(uint64_t) * N * omp_get_max_threads());
		ca->hostMemUsed += sizeof(uint64_t) * N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, rank);

	int N_eff = N / cmpi.num_mpi_threads;
	uint64_t npairs = static_cast<uint64_t>(N_eff) * (N_eff - 1) >> 1;
	unsigned int nthreads = omp_get_max_threads();
	#ifdef AVX2_ENABLED
	nthreads >>= 1;
	#endif

	//First compare all pairs of elements on each computer
	//If only one computer is being used then this step completes the algorithm
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 256) if (npairs >= 1024) num_threads (nthreads)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / (N_eff - 1);
		uint64_t j = k % (N_eff - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((N_eff >> 1) - i) << 1) - 1);
		j += do_map * (((N_eff >> 1) - j) << 1);

		//Relate the index on a single computer to the
		//index with respect to the entire adjacency matrix
		uint64_t glob_i = i + cmpi.rank * N_eff;
		uint64_t glob_j = j + cmpi.rank * N_eff;

		if (glob_i == glob_j) continue;	//Ignore diagonal elements
		//Ignore certain indices which arise due to padding
		if (glob_i >= static_cast<uint64_t>(N) || glob_j >= static_cast<uint64_t>(N)) continue;
		//Ignore elements which aren't related
		if (!nodesAreConnected_v2(adj, N, static_cast<int>(i), static_cast<int>(glob_j))) continue;

		//Save the cardinality
		cardinalities[tid*N+adj[i].partial_vecprod(adj[j], glob_i, glob_j - glob_i + 1)+1]++;
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	N_eff >>= 1;
	npairs = static_cast<uint64_t>(N_eff) * N_eff;	//This is how many pairs are compared
							//once buffers are shuffled around

	//Find first global permutation
	std::vector<unsigned int> current(nbuf);
	std::iota(current.begin(), current.end(), 0);
	std::unordered_set<FastBitset> permutations;
	init_mpi_permutations(permutations, current);

	//Construct set of all pairs except for the 
	//initial configuration, such that the first element
	//is always smaller than the second
	int pair[2];
	std::unordered_set<std::pair<int,int> > new_pairs;
	init_mpi_pairs(new_pairs, current);
	MPI_Barrier(MPI_COMM_WORLD);

	remove_bad_perms(permutations, new_pairs);
	//printCardinalities(cardinalities, N, nthreads, current[rank<<1], current[(rank<<1)+1], 5);

	//These next four steps identify the swaps necessary
	//to obtain the first memory permutation
	std::vector<unsigned int> next;
	FastBitset fb = *permutations.begin();
	binary_to_perm(next, *permutations.begin(), nbuf);

	std::vector<std::vector<unsigned int> > similar;
	fill_mpi_similar(similar, next);

	unsigned int min_steps;
	get_most_similar(next, min_steps, similar, current);
	relabel_vector(current, next);

	std::vector<std::pair<int,int> > swaps;
	cyclesort(min_steps, current, &swaps);

	//The memory segments will now be arranged in the first permutation
	mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, cmpi.rank);
	MPI_Barrier(MPI_COMM_WORLD);
	current = next;

	//The spinlock is created and initialized so all locks are unlocked
	//and no computer owns the spinlock
	MPI_Request req;
	MPI_Status status;
	CausetSpinlock lock[cmpi.num_mpi_threads];
	memset(lock, 0, sizeof(CausetSpinlock) * cmpi.num_mpi_threads);

	//This vector is used to store available trades
	//Computers add elements when they finish their actionKernel
	//and remove elements when they begin a new actionKernel
	std::vector<int> avail;
	avail.reserve(nbuf);

	pthread_t thread;
	pthread_attr_t attr;
	bool busy = false;		//If busy, then actionKernel hasn't finished
	bool waiting = false;		//If waiting, actionKernel has finished, and the
					//computer is in a passive state, waiting for
					//another computer to initiate trades

	//Signal values:
	//0 - Unlock Request
	//1 - Lock Request
	//2 - Broadcast avail
	//3 - Broadcast new_pairs
	//4 - Broadcast swap
	MPISignal signal;
	int success;

	action_params p;	//Parameters passed to the actionKernel
				//These are needed to send variables to a pthread
	p.adj = &adj;
	p.current = &current;
	p.cardinalities = cardinalities;
	//p.clone_length = clone_length;
	p.npairs = npairs;
	p.N = N;
	p.N_eff = N_eff;
	p.rank = rank;
	p.num_mpi_threads = cmpi.num_mpi_threads;
	p.busy = &busy;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

	int lockcnt = -1, lockmax = -1;
	//Continue work until all pairs of buffers have been compared
	while (new_pairs.size() > 0) {
		std::vector<std::pair<int,int> > swaps;
		std::pair<int,int> swap = std::make_pair(-1,-1);
		if (!p.busy[0] && !waiting) {	//Then the computer is in an 'active' state
						//and will attempt to do more work
			if (new_pairs.find(std::make_pair(std::min(current[rank<<1], current[(rank<<1)+1]), std::max(current[rank<<1], current[(rank<<1)+1]))) != new_pairs.end() && std::find(avail.begin(), avail.end(), current[rank<<1]) == avail.end() && std::find(avail.begin(), avail.end(), current[(rank<<1)+1]) == avail.end()) {
				p.busy[0] = true;
				if (ACTION_DEBUG) {
					printf_red();
					printf("Rank [%d] master thread is creating a slave.\n", rank);
					printf_std();
					fflush(stdout); sleep(1);
				}
				sleep(2);
				//Launch a pthread to do work in actionKernel
				pthread_create(&thread, &attr, actionKernel, (void*)&p);
			} else {
				//Computer enters the passive (spin) state
				waiting = true;
				if (ACTION_DEBUG)
					printf("Rank [%d] in SPIN state.\n", rank);
			}
		}

		lockcnt = 0;
		pair[0] = -1;
		pair[1] = -1;
		while (true) {	//Master thread listens for MPI messages
			MPI_Irecv(&signal, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
			int recv = 0;
			while ((p.busy[0] || waiting) && !recv) {
				if (p.busy[0]) {	//If it's busy, it can receive
							//a message, or finish its work
					if (ACTION_DEBUG) {
						printf("Rank [%d] waiting for message (TEST).\n", rank);
						fflush(stdout); sleep(1);
					}
					//Use a non-blocking call in case the work finishes
					MPI_Test(&req, &recv, &status);
					sleep(1);
				} else {	//If the state is passive, wait for a message
					if (ACTION_DEBUG) {
						printf("Rank [%d] waiting for message (WAIT).\n", rank);
						fflush(stdout); sleep(1);
					}
					//Safe to use a blocking call here since there's no work
					MPI_Wait(&req, &status);
					recv = 1;
				}
			}

			//This occurs when 'busy' has just been set to false in the previous loop
			//We will attempt to obtain the lock and then go into the waiting state
			if (!p.busy[0] && !recv) {
				MPI_Cancel(&req);
				MPI_Request_free(&req);	//Needed to avoid memory leaks
				if (cmpi.lock == UNLOCKED && lockcnt == 0) {
					if (ACTION_DEBUG_VERBOSE) {
						printf("Rank [%d] will send a lock signal and wait.\n", rank);
						fflush(stdout);
					}
					//Request ownership of the spinlock
					sendSignal(REQUEST_LOCK, rank, cmpi.num_mpi_threads);
					lock[rank] = LOCKED;
				}

				if (new_pairs.find(std::make_pair(std::min(current[rank<<1], current[(rank<<1)+1]), std::max(current[rank<<1], current[(rank<<1)+1]))) != new_pairs.end()) {
					printf_dbg("Rank [%d] printing for [%d - %d]\n", rank, current[rank<<1], current[(rank<<1)+1]);
					//printCardinalities(cardinalities, N - 1, nthreads, current[rank<<1], current[(rank<<1)+1], 5);
				}

				//State is now passive - the signal (if sent) has been sent to itself as well
				waiting = true;
				continue;
			}

			if (!p.busy[0]) waiting = true;

			int source = status.MPI_SOURCE;
			if (ACTION_DEBUG) {
				printf("Rank [%d] received the signal [%d] from rank [%d]\n", rank, signal, source);
				fflush(stdout); sleep(1);
			}

			int first = 0;
			switch (signal) {
			case REQUEST_UNLOCK:
				//Unlock
				MPI_Barrier(MPI_COMM_WORLD);

				cmpi.lock = UNLOCKED;
				lock[source] = UNLOCKED;
				//If the current elements are available, wait for something else to happen - otherwise they will be added and a trade will be attempted
				if (std::find(avail.begin(), avail.end(), current[rank<<1]) != avail.end() && std::find(avail.begin(), avail.end(), current[(rank<<1)+1]) != avail.end()) {
					if (ACTION_DEBUG_VERBOSE)
						printf("Rank [%d] WAITING.\n", rank);
					waiting = true;
				} else {
					if (ACTION_DEBUG_VERBOSE)
						printf("Rank [%d] NOT WAITING.\n", rank);
					waiting = false;
				}
				lockcnt = 0;
				MPI_Barrier(MPI_COMM_WORLD);
				//This is the final exit sequence from the outer loop
				if (new_pairs.size() == 0) {
					pair[0] = pair[1] = 0;
					waiting = false;
					break;
				}
				if (rank << 1 == pair[0] || (rank << 1) + 1 == pair[0] || rank << 1 == pair[1] || (rank << 1) + 1 == pair[1]) {
					//This will go to the beginning of the outer loop, where new slaves may be created
					//This is the element which matches the master of this unlock operation when a trade has occurred
					waiting = false;
					if (ACTION_DEBUG_VERBOSE)
						printf("Rank [%d] NOT WAITING, RESETTING.\n", rank);
					break;
				} else {
					pair[0] = -1;
					pair[1] = -1;
				}
				if (ACTION_DEBUG_VERBOSE)
					printf("Rank [%d] CONTINUING.\n", rank);
				continue;
			case REQUEST_LOCK:
				//Lock
				//First figure out how many computers are requesting it simultaneously
				//using the allgather operation
				//Protocol dictates if multiple computers request it simultaenously, the
				//one with the lowest index is granted ownership
				MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, lock, 1, MPI_INT, MPI_COMM_WORLD);
				success = 0;
				for (int i = 0; i < cmpi.num_mpi_threads; i++) {
					if (lock[i] == LOCKED) {
						success++;
						if (success == 1)
							first = i;
					}
				}
				if (lockcnt == 0) lockmax = success;
				lockcnt++;
				if (ACTION_DEBUG_VERBOSE) {
					printf("Rank [%d] identified %d locks requested.\n", rank, success);
					printf("Rank [%d] has a lock-count of [%d].\n", rank, lockcnt);
					printf("Rank [%d] identified first as [%d]\n", rank, first);
				}
				for (int i = 0; i < cmpi.num_mpi_threads; i++) {
					if (success > 1 && i != first) {
						lock[i] = UNLOCKED;	//This array indicates the owner of the spinlock
						if (ACTION_DEBUG_VERBOSE)
							printf("Rank [%d] unlocked the lock in rank [%d]\n", rank, i);
					}
					if (i == rank && lock[i] == UNLOCKED)
						cmpi.lock = LOCKED;	//This indicates this thread's lock state
				}

				if (ACTION_DEBUG_VERBOSE) {
					printf("Locks in rank [%d]: (%d, %d, %d, %d)\n", rank, lock[0], lock[1], lock[2], lock[3]);
					fflush(stdout); sleep(1);
				}
				if (!p.busy[0] && !waiting) {
					waiting = true;
					continue;
				}
				if (lockcnt == lockmax && cmpi.lock == UNLOCKED) waiting = false;
				break;
			case REQUEST_UPDATE_AVAIL:
				//Update list of available trades
				if (cmpi.lock == UNLOCKED) {
					printf("ERROR (2)\n");
					break;
				}
				int size;
				MPI_Bcast(&size, 1, MPI_INT, source, MPI_COMM_WORLD);
				avail.resize(size);
				MPI_Bcast(&avail.front(), avail.size(), MPI_INT, source, MPI_COMM_WORLD);
				if (ACTION_DEBUG_VERBOSE) {
					printf("Rank [%d] has updated the list of available trades.\n", rank);
					printf("Rank [%d] sees: (", rank);
					for (int i = 0; i < size; i++)
						printf("%d, ", avail[i]);
					printf(")\n");
				}
				if (!p.busy[0]) waiting = true;
				break;
			case REQUEST_UPDATE_NEW:
				//Update list of unused pairs
				if (cmpi.lock == UNLOCKED) {
					printf("ERROR (3)\n");
					break;
				}
				MPI_Bcast(&pair, 2, MPI_INT, source, MPI_COMM_WORLD);
				new_pairs.erase(std::make_pair(std::min(pair[0], pair[1]), std::max(pair[0], pair[1])));
				pair[0] = -1;
				pair[1] = -1;
				if (ACTION_DEBUG_VERBOSE) {
					printf("Rank [%d] has updated the list of new pairs.\n", rank);
					if (rank == 1)
						for (std::unordered_set<std::pair<int,int> >::iterator it = new_pairs.begin(); it != new_pairs.end(); it++)
							printf("(%d, %d)\n", std::get<0>(*it), std::get<1>(*it));
				}
				if (!p.busy[0]) waiting = true;
				break;
			case REQUEST_EXCHANGE:
				//Memory exchange
				if (cmpi.lock == UNLOCKED) {
					printf("ERROR (4)\n");
					break;
				}
				MPI_Bcast(&pair, 2, MPI_INT, source, MPI_COMM_WORLD);
				swaps.push_back(std::make_pair(pair[0], -1));
				swaps.push_back(std::make_pair(pair[1], pair[0]));
				swaps.push_back(std::make_pair(-1, pair[1]));
				mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, rank);
				MPI_Barrier(MPI_COMM_WORLD);
				swaps.clear();
				swaps.swap(swaps);
				current[pair[0]] ^= current[pair[1]];
				current[pair[1]] ^= current[pair[0]];
				current[pair[0]] ^= current[pair[1]];
				if (!p.busy[0]) waiting = true;
				break;
			}
			if (!waiting && !p.busy[0]) break;
		}

		if (pair[0] != -1 && pair[1] != -1) continue;

		//At this point, a process has gained ownership of the lock and other
		//processes know this. It will tell other processes about its freed buffers
		//and if other memory is free for trading it will initiate a trade
		if (ACTION_DEBUG) {
			printf_mag();
			printf("Rank [%d] has locked the spinlock.\n", rank);
			printf_std();
			fflush(stdout); sleep(1);
		}
		avail.push_back(current[rank<<1]);
		avail.push_back(current[(rank<<1)+1]);
		if (ACTION_DEBUG_VERBOSE) {
			printf("The master made available: %d and %d\n", avail[avail.size() - 2], avail[avail.size() - 1]);
			printf("Available elements: (");
			for (size_t i = 0; i < avail.size(); i++)
				printf("%d, ", avail[i]);
			printf(")\n");
		}
		int avail_size = avail.size();

		sendSignal(REQUEST_UPDATE_AVAIL, rank, cmpi.num_mpi_threads);
		MPI_Bcast(&avail_size, 1, MPI_INT, rank, MPI_COMM_WORLD);
		MPI_Bcast(&avail.front(), avail.size(), MPI_INT, rank, MPI_COMM_WORLD);

		pair[0] = current[rank<<1];
		pair[1] = current[(rank<<1)+1];
		if (ACTION_DEBUG_VERBOSE)
			printf("The master removed %d and %d from new_pairs.\n", pair[0], pair[1]);
		new_pairs.erase(std::make_pair(std::min(pair[0], pair[1]), std::max(pair[0], pair[1])));
		if (ACTION_DEBUG) {
			for (std::unordered_set<std::pair<int,int> >::iterator it = new_pairs.begin(); it != new_pairs.end(); it++)
				printf("(%d, %d)\n", std::get<0>(*it), std::get<1>(*it));
		}

		sendSignal(REQUEST_UPDATE_NEW, rank, cmpi.num_mpi_threads);
		MPI_Bcast(pair, 2, MPI_INT, rank, MPI_COMM_WORLD);

		//Look and see if a trade is available
		for (size_t i = 0; i < avail.size(); i++) {
			if (avail[i] == pair[0] || avail[i] == pair[1]) continue;
			if (new_pairs.find(std::make_pair(std::min(avail[i], pair[0]), std::max(avail[i], pair[0]))) != new_pairs.end()) {
				swap = std::make_pair(std::min(avail[i], pair[1]), std::max(avail[i], pair[1]));
				break;
			}
			if (new_pairs.find(std::make_pair(std::min(avail[i], pair[1]), std::max(avail[i], pair[1]))) != new_pairs.end()) {
				swap = std::make_pair(std::min(avail[i], pair[0]), std::max(avail[i], pair[0]));
				break;
			}
		}

		if (swap != std::make_pair(-1,-1)) {	//A trade is available
			if (ACTION_DEBUG_VERBOSE)
				printf("Identified a swap: (%d, %d)\n", std::get<0>(swap), std::get<1>(swap));
			for (int i = 0; i < static_cast<int>(nbuf); i++)
				if (std::get<0>(swap) == static_cast<int>(current[i]))
					swaps.push_back(std::make_pair(i, -1));
			for (int i = 0; i < static_cast<int>(nbuf); i++)
				if (std::get<1>(swap) == static_cast<int>(current[i]))
					swaps.push_back(std::make_pair(-1, i));

			sendSignal(REQUEST_EXCHANGE, rank, cmpi.num_mpi_threads);
			pair[0] = std::get<0>(swaps[0]);
			pair[1] = std::get<1>(swaps[1]);
			swaps.insert(swaps.begin()+1, std::make_pair(pair[1], pair[0]));
			if (ACTION_DEBUG_VERBOSE)
				printf("Pair indices: (%d, %d)\n", pair[0], pair[1]);
			MPI_Bcast(pair, 2, MPI_INT, rank, MPI_COMM_WORLD);

			mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, rank);
			MPI_Barrier(MPI_COMM_WORLD);

			current[pair[0]] ^= current[pair[1]];
			current[pair[1]] ^= current[pair[0]];
			current[pair[0]] ^= current[pair[1]];
			//printf_cyan();
			//printf("\nCurrent permutation:\t");
			//print_pairs(current);
			//printf_std();
			//fflush(stdout);

			int r0 = pair[0] >> 1;
			int r1 = pair[1] >> 1;
			for (size_t i = 0; i < avail.size(); i++) {
				if (avail[i] == static_cast<int>(current[r0<<1]) || avail[i] == static_cast<int>(current[(r0<<1)+1]) || avail[i] == static_cast<int>(current[r1<<1]) || avail[i] == static_cast<int>(current[(r1<<1)+1])) {
					avail.erase(avail.begin()+i);
					i--;
				}
			}
			if (new_pairs.find(std::make_pair(std::min(current[r0<<1], current[(r0<<1)+1]), std::max(current[r0<<1], current[(r0<<1)+1]))) == new_pairs.end()) {
				avail.push_back(current[r0<<1]);
				avail.push_back(current[(r0<<1)+1]);
			}
			if (new_pairs.find(std::make_pair(std::min(current[r1<<1], current[(r1<<1)+1]), std::max(current[r1<<1], current[(r1<<1)+1]))) == new_pairs.end()) {
				avail.push_back(current[r1<<1]);
				avail.push_back(current[(r1<<1)+1]);
			}
			if (ACTION_DEBUG_VERBOSE) {
				printf("The master updated avail to (");
				for (size_t i = 0; i < avail.size(); i++) {
					printf("%d, ", avail[i]);
				}
				printf(")\n");
			}

			avail_size = avail.size();
			sendSignal(REQUEST_UPDATE_AVAIL, rank, cmpi.num_mpi_threads);
			MPI_Bcast(&avail_size, 1, MPI_INT, rank, MPI_COMM_WORLD);
			MPI_Bcast(&avail.front(), avail.size(), MPI_INT, rank, MPI_COMM_WORLD);
		} else {	//Move to passive state
			waiting = true;
			pair[0] = -1;
			pair[1] = -1;
		}

		//Unlock
		sendSignal(REQUEST_UNLOCK, rank, cmpi.num_mpi_threads);
		MPI_Barrier(MPI_COMM_WORLD);
		lock[rank] = UNLOCKED;
		if (ACTION_DEBUG) {
			if (waiting)
				printf("Rank [%d] in SPIN state.\n", rank);
			printf_mag();
			printf("Rank [%d] has unlocked the spinlock.\n", rank);
			printf_std();
			fflush(stdout); sleep(1);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	pthread_attr_destroy(&attr);

	//Return to the original configuration
	if (cmpi.num_mpi_threads > 1) {
		std::vector<std::pair<int,int> > swaps;
		cyclesort(min_steps, current, &swaps);
		mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, rank);
	}

	#endif

	//OpenMP Reduction
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			cardinalities[j] += cardinalities[i*N+j];

	//MPI Reduction
	#ifdef MPI_ENABLED
	MPI_Allreduce(MPI_IN_PLACE, cardinalities, N, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	#endif
	cardinalities[0] = N;

	action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), epsilon);
	assert (action == action);

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Action.\n");
		printf_mpi(rank, "\t\tTerms Used: %d\n", N);
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tCausal Set Action: %f\n", action);
		if (!rank) printf_std();
	}

	if (verbose)
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
	if (!rank) fflush(stdout);

	return true;
}


//Measure Causal Set Action
//Supports MPI Parallelization
//Algorithm has been optimized using minimal bitwise operations
//Requires the existence of the whole adjacency matrix
//This will calculate all cardinality intervals by construction
bool measureAction_v4(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int &N, const double epsilon, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	if (verbose || bench) {
		#ifdef MPI_ENABLED
		if (!cmpi.rank) printf_mag();
		printf_mpi(cmpi.rank, "Using Version 4 (measureAction).\n");
		if (!cmpi.rank) printf_std();
		fflush(stdout); sleep(1);
		MPI_Barrier(MPI_COMM_WORLD);
		#else
		printf_dbg("Using Version 4 (measureAction).\n");
		#endif
	}

	static const bool ACTION_DEBUG = false;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements and workspace
	try {
		cardinalities = (uint64_t*)malloc(sizeof(uint64_t) * N * omp_get_max_threads());
		if (cardinalities == NULL)
			throw std::bad_alloc();
		memset(cardinalities, 0, sizeof(uint64_t) * N * omp_get_max_threads());
		ca->hostMemUsed += sizeof(uint64_t) * N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
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
		}
		printf_mpi(cmpi.rank, "\n");
		fflush(stdout); sleep(1);
		#ifdef MPI_ENABLED
		MPI_Barrier(MPI_COMM_WORLD);
		#endif
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, cmpi.rank);

	int N_eff = N / cmpi.num_mpi_threads;
	uint64_t npairs = static_cast<uint64_t>(N_eff) * (N_eff - 1) >> 1;
	unsigned int nthreads = omp_get_max_threads();
	#ifdef AVX2_ENABLED
	nthreads >>= 1;
	#endif

	//First compare all pairs of elements on each computer
	//If only one computer is being used then this step completes the algorithm
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 256) if (npairs >= 1024) num_threads (nthreads)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / (N_eff - 1);
		uint64_t j = k % (N_eff - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((N_eff >> 1) - i) << 1) - 1);
		j += do_map * (((N_eff >> 1) - j) << 1);

		//Relate the index on a single computer to the
		//index with respect to the entire adjacency matrix
		uint64_t glob_i = i + cmpi.rank * N_eff;
		uint64_t glob_j = j + cmpi.rank * N_eff;
		if (glob_i == glob_j) continue;	//Ignore diagonal elements
		//Ignore certain indices which arise due to padding
		if (glob_i >= static_cast<uint64_t>(N) || glob_j >= static_cast<uint64_t>(N)) continue;
		//Ignore elements which aren't related
		if (!nodesAreConnected_v2(adj, N, static_cast<int>(i), static_cast<int>(glob_j))) continue;

		//Save the cardinality
		cardinalities[tid*N+adj[i].partial_vecprod(adj[j], glob_i, glob_j - glob_i + 1)+1]++;
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	//MPI Swaps
	#ifdef MPI_ENABLED
	N_eff >>= 1;
	npairs = static_cast<uint64_t>(N_eff) * N_eff;
	//All elements will be compared on each computer during each permutation
	//This is NOT a load-balanced operation (hence the need for version 5)
	while (permutations.size() > 0) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Current permutation:\t\t");
			for (size_t i = 0; i < current.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", current[i], current[i+1]);
			printf_mpi(cmpi.rank, "\n");
			if (!cmpi.rank) fflush(stdout);
		}

		//Determine the next permutation
		std::vector<unsigned int> next;
		FastBitset fb = *permutations.begin();
		binary_to_perm(next, *permutations.begin(), nbuf);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Next permutation:\t\t");
			for (size_t i = 0; i < next.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", next[i], next[i+1]);
			printf_mpi(cmpi.rank, "\n");
		}

		//If this is the ordered (original) permutation, ignore it
		if (fb == fb_ordered) {
			if (ACTION_DEBUG)
				printf_mpi(cmpi.rank, "Identified an ordered permutation. Skipping this one.\n");
			fflush(stdout); sleep(1);
			permutations.erase(permutations.begin());
			continue;
		}

		std::vector<std::vector<unsigned int> > similar;
		fill_mpi_similar(similar, next);

		//Find the most similar configuration of the next permutation to the current one
		unsigned int min_steps;
		get_most_similar(next, min_steps, similar, current);
		if (ACTION_DEBUG) {
			printf_mpi(cmpi.rank, "Most Similar Configuration:\t");
			for (size_t i = 0; i < next.size(); i += 2)
				printf_mpi(cmpi.rank, "(%d, %d) ", next[i], next[i+1]);
			printf_mpi(cmpi.rank, "\n\n");
			if (!cmpi.rank) fflush(stdout);
		}

		//Use a cycle sort to identify the swaps needed
		relabel_vector(current, next);
		std::vector<std::pair<int,int> > swaps;
		cyclesort(min_steps, current, &swaps);

		//Perform the MPI exchanges
		printf_mpi(cmpi.rank, "Preparing memory exchange...");
		fflush(stdout); sleep(1);
		mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, cmpi.rank);
		MPI_Barrier(MPI_COMM_WORLD);
		printf_mpi(cmpi.rank, "completed!\n");
		fflush(stdout); sleep(1);
		current = next;

		//Compare pairs between the two buffers
		//Pairs within a single buffer are not compared
		#ifdef _OPENMP
		#pragma omp parallel for schedule (static, 256) if (npairs >= 1024) num_threads (nthreads)
		#endif
		for (uint64_t k = 0; k < npairs; k++) {
			unsigned int tid = omp_get_thread_num();
			//Choose a pair
			uint64_t i = k / N_eff;
			uint64_t j = k % N_eff;
			j += N_eff;

			uint64_t glob_i = loc_to_glob_idx(next, i, N, cmpi.num_mpi_threads, cmpi.rank);
			uint64_t glob_j = loc_to_glob_idx(next, j, N, cmpi.num_mpi_threads, cmpi.rank);
			if (glob_i == glob_j) continue;
			if (glob_i >= static_cast<uint64_t>(N) || glob_j >= static_cast<uint64_t>(N)) continue;

			if (glob_i > glob_j) {
				glob_i ^= glob_j;
				glob_j ^= glob_i;
				glob_i ^= glob_j;

				i ^= j;
				j ^= i;
				i ^= j;
			}

			if (!nodesAreConnected_v2(adj, N, static_cast<int>(i), static_cast<int>(glob_j))) continue;

			cardinalities[tid*N+adj[i].partial_vecprod(adj[j], glob_i, glob_j - glob_i + 1)+1]++;
		}
		permutations.erase(permutations.begin());
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//Return to original configuration
	if (cmpi.num_mpi_threads > 1) {
		std::vector<std::pair<int,int> > swaps;
		unsigned int min_steps;
		cyclesort(min_steps, current, &swaps);
		mpi_swaps(swaps, adj, cmpi.adj_buf, N, cmpi.num_mpi_threads, cmpi.rank);
	}
	#endif

	//Reduction for OpenMP
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			cardinalities[j] += cardinalities[i*N+j];

	//Reduction for MPI
	#ifdef MPI_ENABLED
	MPI_Allreduce(MPI_IN_PLACE, cardinalities, N, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	#endif
	//The first element will be N
	cardinalities[0] = N;

	action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), epsilon);
	assert (action == action);

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf_mpi(cmpi.rank, "\tCalculated Action.\n");
		printf_mpi(cmpi.rank, "\t\tTerms Used: %d\n", N);
		if (!cmpi.rank) printf_cyan();
		printf_mpi(cmpi.rank, "\t\tCausal Set Action: %f\n", action);
		if (!cmpi.rank) printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(cmpi.rank, "\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Measure Causal Set Action
//Algorithm has been optimized using minimal bitwise operations
//Requires the existence of the whole adjacency matrix
//This will calculate all cardinality intervals by construction
bool measureAction_v3(uint64_t *& cardinalities, double &action, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int N, const double epsilon, const GraphType gt, CaResources * const ca, Stopwatch &sMeasureAction, const bool use_bit, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	if (verbose || bench) {
		printf_dbg("Using Version 3 (measureAction).\n");
		#ifdef AVX512_ENABLED
		printf_dbg(" > enabling AVX-512 features\n");
		#endif
	}

	int n = N + N % 2;
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / 2;

	stopwatchStart(&sMeasureAction);

	//Allocate memory for cardinality measurements and workspace
	try {
		cardinalities = (uint64_t*)calloc(N * omp_get_max_threads(), sizeof(uint64_t));
		if (cardinalities == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += sizeof(uint64_t) * N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure B-D Action", ca->hostMemUsed, ca->devMemUsed, 0);

	//The first element will be N
	cardinalities[0] = N;

	unsigned int nthreads = omp_get_max_threads();
	#ifdef AVX2_ENABLED
	//nthreads >>= 1;
	#endif

	//Compare all pairs of elements
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 256) num_threads (nthreads) if (npairs >= 1024)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);

		//Ignore pairs which are not connected
		if (static_cast<int>(j) == N) continue;	//Arises due to index padding
		if (!nodesAreConnected_v2(adj, N, static_cast<int>(i), static_cast<int>(j))) continue;

		//Save the cardinality
		#ifdef AVX512_ENABLED
		cardinalities[tid*N+adj[i].partial_vecprod_avx512(adj[j], i, j - i + 1)+1]++;
		#else
		cardinalities[tid*N+adj[i].partial_vecprod(adj[j], i, j - i + 1)+1]++;
		#endif
	}

	//Reduction for OpenMP
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			cardinalities[j] += cardinalities[i*N+j];

	if (gt == RGG) {
		action = calcAction(cardinalities, atoi(Spacetime::stdims[spacetime.get_stdim()]), epsilon);
		assert (action == action);
	}

	stopwatchStop(&sMeasureAction);

	if (!bench) {
		printf("\tCalculated Interval Abundances.\n");
		if (gt == RGG) {
			printf("\t\tTerms Used in Action: %d\n", N);
			printf_cyan();
			printf("\t\tCausal Set Action: %f\n", action);
			printf_std();
		}
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool timelikeActionCandidates(std::vector<unsigned int> &candidates, int *chaintime, const Node &nodes, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int &N, CaResources * const ca, Stopwatch sMeasureActionTimelike, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter"));
	assert (N > 0);
	assert (ca != NULL);
	assert (use_bit);
	#endif

	static const bool ACTION_DEBUG = true;
	static const bool ACTION_BENCH = true;

	printf("Preparing to measure timelike action...\n");
	fflush(stdout);

	std::ofstream stream;

	candidates.clear();
	candidates.swap(candidates);

	Bitvector workspaces;
	int n = N + N % 2;
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / 2;

	int *lengths = NULL;
	int nthreads = omp_get_max_threads();
	//int longest = 0;
	int maximum_chain;
	int shortest_maximal_chain;
	int studybin;

	int cutoff = 10;	//Minimum number of elements in a hypersurface to be 'usable'
	int usable = 0;
	int k_threshold = 0;	//Minimum total degree to be counted as a candidate

	stopwatchStart(&sMeasureActionTimelike);

	try {
		chaintime = (int*)malloc(sizeof(int) * N * omp_get_max_threads());
		if (chaintime == NULL)
			throw std::bad_alloc();
		//memset(chaintime, 0, sizeof(int) * N);
		memset(chaintime, -1, sizeof(int) * N * omp_get_max_threads());
		ca->hostMemUsed += sizeof(int) * N * omp_get_max_threads();

		lengths = (int*)malloc(sizeof(int) * N * nthreads);
		if (lengths == NULL)
			throw std::bad_alloc();
		memset(lengths, -1, sizeof(int) * N * nthreads);
		ca->hostMemUsed += sizeof(int) * N * nthreads;

		workspaces.reserve(nthreads);
		for (int i = 0; i < nthreads; i++) {
			FastBitset fb(static_cast<uint64_t>(N));
			workspaces.push_back(fb);
			ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
		}
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	Stopwatch sChainTimes = Stopwatch();
	stopwatchStart(&sChainTimes);

	//Re-write longestChain_v2 using reverse algorithm (top to bottom)
	//Should accept multiple endpoints (minimal nodes)

	//Measure chain times for all nodes
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 64) num_threads(nthreads) if (npairs > 10000)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);

		if (static_cast<int>(j) == N) continue;
		if (!!k_in[i]) continue;
		if (!nodesAreConnected_v2(adj, N, i, j)) continue;	//Continue if not related

		int length = longestChain_v2(adj, &workspaces[tid], lengths + N * tid, N, i, j, 0);
		//#pragma omp critical
		//chaintime[j] = std::max(chaintime[j], length);
		chaintime[tid*N+j] = std::max(chaintime[tid*N+j], length);
	}

	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			chaintime[j] = std::max(chaintime[j], chaintime[i*N+j]);

	for (int i = 0; i < N; i++)
		if (!k_in[i])
			chaintime[i] = 0;
	maximum_chain = *std::max_element(chaintime, chaintime + N);

	//printf("distance [%d - %d] = %d\n", 0, 111, longestChain_v2r(adj, &workspaces[0], lengths, N, 0, 111, 0));
	//printf("distance [%d - %d] = %d\n", 111, 3972, longestChain_v2r(adj, &workspaces[0], lengths, N, 111, 3972, 0));
	/*longestChain_v2r(adj, &workspaces[0], lengths, N, 0, 697, 0);
	longestChain_v2(adj, &workspaces[0], &lengths[N], N, 0, 697, 0);
	adj[0].clone(workspaces[0]);
	workspaces[0].partial_intersection(adj[697], 0, 698);
	uint64_t cnt = workspaces[0].partial_count(0, 698) + 2;
	for (int i = 0; i < N; i++)
		if (lengths[i] != -1 || lengths[N+i] != -1)
			printf("lengthsR[%d] = %d\tlengths[%d] = %d\ttrue[%d] = %d\n", i, lengths[i], i, lengths[N+i], i, longestChain_v2(adj, &workspaces[0], &lengths[N*2], N, 0, i, 0));
	printf("Bit count (inclusive): %" PRIu64 "\n", cnt);
	printChk(9);*/

	//#ifdef _OPENMP
	//#pragma omp parallel for schedule (dynamic, 8) num_threads(nthreads) if (npairs > 10000)
	//#endif
	/*for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);

		if (static_cast<int>(j) == N) continue;
		if (!!k_in[i] || !!k_out[j]) continue;
		if (!nodesAreConnected_v2(adj, N, i, j)) continue;	//Continue if not related

		int length = longestChain_v2r(adj, &workspaces[tid], lengths + N * tid, N, i, j, 0);
		lengths[N*tid+i] = 0;

		//#pragma omp critical
		{
		//printf("lengths in interval [%d - %d]:\n", i, j);
		for (int m = 0; m < N; m++) {
			if (lengths[N*tid+m] == -1) continue;
			//printf("lengths[%d] = %d\n", m, length - lengths[N*tid+m]);
			chaintime[m] = std::max(chaintime[m], length - lengths[N*tid+m]);
		}
		//printChk(8);
		}
	}
	maximum_chain = *std::max_element(chaintime, chaintime + N);*/

	stopwatchStop(&sChainTimes);
	if (ACTION_BENCH)
		printf_dbg("\tFound Chain Times: %5.6f sec\n", sChainTimes.elapsedTime);

	ca->hostMemUsed -= sizeof(BlockType) * workspaces[0].getNumBlocks() * nthreads;
	workspaces.clear();
	workspaces.swap(workspaces);

	Stopwatch sCandidates = Stopwatch();
	stopwatchStart(&sCandidates);

	if (ACTION_DEBUG) {
		stream.open("chaintimes.cset.act.dat");
		for (int i = 0; i < N; i++)
			stream << chaintime[i] << "\n";
		stream.flush();
		stream.close();
	}

	//printf_dbg("Maximum Chain: %d\n", maximum_chain);
	//printChk(7);

	//Scan slices
	int num[maximum_chain];
	for (int i = 0; i < maximum_chain; i++) {
		num[i] = 0;
		for (int j = 0; j < N; j++) {
			if (chaintime[j] == i)
				num[i]++;
		}
		//printf("Bin [%d] has %d elements.\n", i, num[i]);
	}

	//Look for edge elements
	for (int i = maximum_chain - 1; i >= 0; i--) {
		if (num[i] > cutoff) {
			usable = i;
			break;
		}
	}

	int P0 = 0, P1 = 0, F0 = 0, F1 = 0;
	for (int i = 0; i < N; i++) {
		if (k_in[i] == 0) P0++;
		if (k_in[i] == 1) P1++;
		if (k_out[i] == 0) F0++;
		if (k_out[i] == 1) F1++;
	}

	//bool convex = false;
	/*int stdim = atoi(Spacetime::stdims[spacetime.get_stdim()]);
	if (P0 - stdim * P1 < 0 && stdim * F1 - F0 < 0)
		convex = false;
	if (!convex)
		printChk();*/

	//Iterate over all bins
	if (ACTION_DEBUG)
		stream.open("candidates.cset.act.dat");
	for (int k = 0; k < usable; k++) {
		//Maximums
		float max_ki = 0.0;
		float max_ko = 0.0;
		float max_k = 0.0;
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k) {
				max_ki = std::max(max_ki, static_cast<float>(k_in[i]));
				max_ko = std::max(max_ko, static_cast<float>(k_out[i]));
				max_k = std::max(max_k, static_cast<float>(k_in[i] + k_out[i]));
			}
		}

		//Minimums
		float min_ki = INF;
		float min_ko = INF;
		float min_k = INF;
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k) {
				min_ki = std::min(min_ki, static_cast<float>(k_in[i]));
				min_ko = std::min(min_ko, static_cast<float>(k_out[i]));
				min_k = std::min(min_k, static_cast<float>(k_in[i] + k_out[i]));
			}
		}

		//Averages
		/*float avg_x = 0.0;
		float avg_ki = 0.0;
		float avg_ko = 0.0;
		float avg_k = 0.0;
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k) {
				avg_x += fabs(nodes.crd->y(i));
				avg_ki += k_in[i];
				avg_ko += k_out[i];
				avg_k += k_in[i] + k_out[i];
			}
		}

		if (num[k]) {
			avg_x /= num[k];
			avg_ki /= num[k];
			avg_ko /= num[k];
			avg_k /= num[k];
		}*/

		//Standard Deviations
		/*float stddev_x = 0.0;
		float stddev_ki = 0.0;
		float stddev_ko = 0.0;
		float stddev_k = 0.0;
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k) {
				stddev_x += POW2(fabs(nodes.crd->y(i)) - avg_x);
				stddev_ki += POW2(k_in[i] - avg_ki);
				stddev_ko += POW2(k_out[i] - avg_ko);
				stddev_k += POW2(k_in[i] + k_out[i] - avg_k);
			}
		}

		if (num[k]) {
			stddev_x = sqrt(stddev_x / num[k]);
			stddev_ki = sqrt(stddev_ki / num[k]);
			stddev_ko = sqrt(stddev_ko / num[k]);
			stddev_k = sqrt(stddev_k / num[k]);
		}*/

		//Person Coefficients
		/*float r_x_ki = 0.0;
		float r_x_ko = 0.0;
		float r_x_k = 0.0;
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k) {
				r_x_ki += (fabs(nodes.crd->y(i)) - avg_x) * (k_in[i] - avg_ki);
				r_x_ko += (fabs(nodes.crd->y(i)) - avg_x) * (k_out[i] - avg_ko);
				r_x_k += (fabs(nodes.crd->y(i)) - avg_x) * (k_in[i] + k_out[i] - avg_k);
			}
		}

		if (num[k]) {
			if (stddev_ki)
				r_x_ki /= num[k] * stddev_x * stddev_ki;
			if (stddev_ko)
				r_x_ko /= num[k] * stddev_x * stddev_ko;
			if (stddev_k)
				r_x_k /= num[k] * stddev_x * stddev_k;
		}*/

		//printf("Time [%d] Pearson Coefficients:\n", k);
		//printf(" > |x|,k_i:     \t%f\n", r_x_ki);
		//printf(" > |x|,k_o:     \t%f\n", r_x_ko);
		//printf(" > |x|,k:       \t%f\n\n", r_x_k);

		//if (!(k % 5)) continue;

		//Identify Candidates
		for (int i = 0; i < N; i++) {
			//if (!(i % 2)) continue;	//Reduces the number of candidates by half
			if (chaintime[i] == k) {
				//if ((!convex && k < usable / 5 && k_in[i] + k_out[i] > max_k - 1.5 * sqrt(max_k)) || (((!convex && k >= usable / 5) || convex) &&
				if (k_in[i] > k_threshold && k_out[i] > k_threshold && k_in[i] + k_out[i] < min_k + 1.5 * sqrt(min_k)) {
					candidates.push_back(i);
					if (ACTION_DEBUG)
						stream << nodes.crd->y(i) << " " << nodes.crd->x(i) << "\n";
						//stream << nodes.crd->y(i) << " " << nodes.crd->x(i) << " " << nodes.crd->z(i) << "\n";
				}
			}
		}
	}
	if (ACTION_DEBUG) {
		stream.flush();
		stream.close();
	}

	stopwatchStop(&sCandidates);
	if (ACTION_BENCH)
		printf_dbg("\tIdentified Candidates: %5.6f sec\n", sCandidates.elapsedTime);

	ca->hostMemUsed += sizeof(unsigned int) * candidates.size();
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Identify Timelike Boundaries", ca->hostMemUsed, ca->devMemUsed, 0);

	Stopwatch sShortestMaximal = Stopwatch();
	stopwatchStart(&sShortestMaximal);

	//Find the shortest maximal chain
	//Used to identify Type 2 Boundary
	shortest_maximal_chain = maximum_chain;
	for (int k = 1; k < usable; k++) {
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == k && !k_out[i]) {
				shortest_maximal_chain = k;
				k = usable;	//To break outer loop
				break;		//To break inner loop
			}
		}
	}

	stopwatchStop(&sShortestMaximal);
	if (ACTION_BENCH)
		printf_dbg("\tIdentified Shortest Maximal Chain: %5.6f sec\n", sShortestMaximal.elapsedTime);

	//studybin = usable / 2;
	studybin = 0;
	if (ACTION_DEBUG) {
		stream.open("kdist.cset.act.dat");
		for (int i = 0; i < N; i++) {
			if (chaintime[i] == studybin) {
				stream << nodes.crd->x(i) << " " << nodes.crd->y(i) << " " << k_in[i] + k_out[i] << std::endl;
			}
		}
		stream.flush();
		stream.close();
	}

	free(lengths);
	lengths = NULL;
	ca->hostMemUsed -= sizeof(int) * N * nthreads;

	stopwatchStop(&sMeasureActionTimelike);

	if (!bench) {
		printf("\tCalculated Timelike Boundary Candidates.\n");
		printf_cyan();
		printf("\t\tIdentified [%zd] Candidates.\n", candidates.size());
		printf_std();
		printf("\t\tLongest  Maximal Chain: [%d]\n", maximum_chain);
		printf("\t\t > Usable Bins: [%d]\n", usable);
		//printf("\t\t > Studying Bin [%d]\n", studybin);
		printf("\t\tShortest Maximal Chain: [%d]\n", shortest_maximal_chain);
		if (shortest_maximal_chain <= 10)
			printf("\t > Detected Class II(b) Joint!\n");
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureActionTimelike.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureTimelikeAction(Network * const graph, Network * const subgraph, const std::vector<unsigned int> &candidates, CaResources * const ca)
{
	#if DEBUG
	assert (graph != NULL);
	assert (subgraph != NULL);
	#endif

	static const bool ACTION_DEBUG = true;
	assert (false);

	Bitvector chains;
	std::unordered_set<int> minimal_elements;
	std::unordered_set<int> maximal_elements;
	std::ofstream stream;
	std::pair<int,int> *sublengths = NULL;
	int *lengths = NULL;
	int nthreads = omp_get_max_threads();

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	try {
		chains.reserve(graph->network_properties.N);
		for (int i = 0; i < graph->network_properties.N; i++) {
			FastBitset fb(graph->network_properties.N);
			chains.push_back(fb);
			ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
		}

		lengths = (int*)malloc(sizeof(int) * graph->network_properties.N * nthreads);
		if (lengths == NULL)
			throw std::bad_alloc();
		memset(lengths, -1, sizeof(int) * graph->network_properties.N * nthreads);
		ca->hostMemUsed += sizeof(int) * graph->network_properties.N * nthreads;

		sublengths = (std::pair<int,int>*)malloc(sizeof(std::pair<int,int>) * subgraph->network_properties.N);
		if (sublengths == NULL)
			throw std::bad_alloc();
		for (int i = 0; i < subgraph->network_properties.N; i++)
			sublengths[i] = std::make_pair(-1, -1);
		ca->hostMemUsed += sizeof(std::pair<int,int>) * subgraph->network_properties.N;
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//Generate chains for all pairs of endpoints in L (saved in vector of tuples)
	// > Remove any chains with weight < min(10, global_heaviest/20)
	//    > global_heaviest is the weight of the first accepted chain; its value does not change
	// > Accept the heaviest of the remaining chains, and make its elements (excluding endpoints) prohibited (saved in vector of tuples)
	//    > Repeat for the same endpoints until no valid chains left (this detects multiple chains with the same pair of endpoints)
	// > Remove endpoints from L
	//Result: vector of chains with their lengths and weights
	// > Plot chains and sum their lengths to approximate the length of the boundary

	std::vector<std::pair<int,int>> endpoints;
	std::vector<std::tuple<FastBitset, int, int>> accepted_chains;
	//int min_weight = -1;

	//Make a list L of pairs of endpoints (vector of pairs).
	endpoints.reserve(1000);
	for (int i = 0; i < subgraph->network_properties.N; i++) {
		if (!!subgraph->nodes.k_in[i]) continue;
		for (int j = 0; j < subgraph->network_properties.N; j++) {
			if (!!subgraph->nodes.k_out[j]) continue;
			if (!nodesAreConnected_v2(subgraph->adj, subgraph->network_properties.N, i, j)) continue;
			endpoints.push_back(std::make_pair(std::min(i, j), std::max(i, j)));
		}
	}

	FastBitset excluded(graph->network_properties.N);
	accepted_chains.reserve(100);

	//Haven't updated getPossibleChains yet
	/*while (endpoints.size() > 0) {
		int max_weight = 0, max_idx = -1, end_idx = -1;
		std::vector<std::tuple<FastBitset,int,int>> possible_chains = getPossibleChains(graph->adj, subgraph->adj, chains, &excluded, endpoints, candidates, lengths, sublengths, graph->network_properties.N, subgraph->network_properties.N, min_weight, max_weight, max_idx, end_idx);
		if (end_idx == -1) break;

		accepted_chains.push_back(possible_chains[max_idx]);
		excluded.setUnion(std::get<0>(possible_chains[max_idx]));

		//If there are multiple paths with the same endpoints, check for that here (not sure how yet)

		endpoints.erase(endpoints.begin() + end_idx);
		if (min_weight == -1) { min_weight = 10; }

		if (graph->network_properties.spacetime.stdimIs("2") && accepted_chains.size() == 2) break;
	}*/

	//Extend chains to maximal and minimal elements
	for (size_t i = 0; i < accepted_chains.size(); i++)
		std::get<2>(accepted_chains[i]) += rootChain(graph->adj, chains, std::get<0>(accepted_chains[i]), graph->nodes.k_in, graph->nodes.k_out, lengths, graph->network_properties.N);

	std::ofstream stream2;
	if (ACTION_DEBUG) {
		stream.open("chains.cset.act.dat");
		stream2.open("chain_lengths.cset.act.dat");
	}
	int timelike_volume = 0;
	for (size_t i = 0; i < accepted_chains.size(); i++) {
		timelike_volume += std::get<2>(accepted_chains[i]);
		if (ACTION_DEBUG) {
			printf_dbg("Chain %d has [%d] elements.\n", i, std::get<2>(accepted_chains[i]));
			stream2 << std::get<0>(accepted_chains[i]).count_bits() << std::endl;
			for (int j = 0; j < graph->network_properties.N; j++)
				if (std::get<0>(accepted_chains[i]).read(j))
					stream << graph->nodes.crd->y(j) << " " << graph->nodes.crd->x(j) << std::endl;
					//stream << graph->nodes.crd->x(j) << " " << graph->nodes.crd->y(j) << " " << graph->nodes.crd->z(j) << "\n";
		}
	}
	if (ACTION_DEBUG) {
		stream.flush();
		stream.close();
		stream2.flush();
		stream2.close();
	}

	int P0 = 0, P1 = 0, F0 = 0, F1 = 0;
	for (int i = 0; i < graph->network_properties.N; i++) {
		if (graph->nodes.k_in[i] == 0) P0++;
		if (graph->nodes.k_in[i] == 1) P1++;
		if (graph->nodes.k_out[i] == 0) F0++;
		if (graph->nodes.k_out[i] == 1) F1++;
	}

	//Free Memory	
	for (int i = 0; i < graph->network_properties.N; i++)
		ca->hostMemUsed -= sizeof(BlockType) * chains[i].getNumBlocks();
	chains.clear();
	chains.swap(chains);

	free(lengths);
	lengths = NULL;
	ca->hostMemUsed -= sizeof(int) * graph->network_properties.N * nthreads;

	free(sublengths);
	sublengths = NULL;
	ca->hostMemUsed -= sizeof(std::pair<int,int>) * subgraph->network_properties.N;

	stopwatchStop(&s);

	//float timelike = static_cast<float>(timelike_volume) / (2.0 * sqrtf(static_cast<float>(graph->network_properties.N) / 2.0));
	float theoretical_volume = 2.0 * graph->network_properties.tau0;	//Box
	//float theoretical_volume = TWO_PI * graph->network_properties.r_max * graph->network_properties.tau0;	//Cylinder
	//if (get_symmetry(graph->network_properties.spacetime) & SYMMETRIC)
	if (graph->network_properties.spacetime.symmetryIs("Temporal"))
		theoretical_volume *= 2.0;
	//float err = fabs(theoretical_volume - timelike) / theoretical_volume;

	if (!graph->network_properties.flags.bench) {
		printf("\n\tCalculated Timelike Action.\n");
		printf_cyan();
		printf("\t\tNumber of Chains: %zd\n", accepted_chains.size());
		printf("\t\tAverage Chain Length: %d\n", static_cast<int>(timelike_volume / accepted_chains.size()));
		printf_std();
		/*printf("\t\tTheoretical Volume: %f\n", theoretical_volume);
		printf_cyan();
		printf("\t\tTimelike Volume: %f\n", timelike);
		printf_red();
		printf("\t\tError: %f\n", err);
		printf_std();*/
		printf("\tSpacelike Boundary:\n");
		printf_cyan();
		printf("\t\tP0: %d\n", P0);
		printf("\t\tP1: %d\n", P1);
		printf("\t\tF0: %d\n", F0);
		printf("\t\tF1: %d\n", F1);
		printf_std();
		fflush(stdout);
	}

	if (graph->network_properties.flags.verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", s.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureTheoreticalAction(double *& actth, int N_actth, const Node &nodes, Bitvector &adj, const Spacetime &spacetime, const int N, const double eta0, const double delta, CaResources * const ca, Stopwatch &sMeasureThAction, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (N_actth > 0);
	assert (adj.size() > 0);
	assert (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "Temporal"));
	assert (N > 0);
	assert (delta > 0.0);
	assert (ca != NULL);
	#endif

	double volume = static_cast<double>(N) / delta;
	int n = N + N % 2;
	uint64_t npairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	int min = 10;
	int nsamples = 3 * (N_actth - min) + 1;

	stopwatchStart(&sMeasureThAction);

	try {
		actth = (double*)malloc(sizeof(double) * nsamples * omp_get_max_threads());
		if (actth == NULL)
			throw std::bad_alloc();
		memset(actth, 0, sizeof(double) * nsamples * omp_get_max_threads());
		ca->hostMemUsed += sizeof(double) * nsamples * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure inn %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	int M[nsamples];
	for (int i = 0; i < nsamples; i++)
		M[i] = static_cast<int>(exp2(static_cast<double>(i) / 3.0 + min));

	int cnt = 0;
	#pragma omp parallel for schedule (dynamic, 1) reduction(+ : cnt) if (npairs > 10000)
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		int i = static_cast<int>(k / (n - 1));
		int j = static_cast<int>(k % (n - 1) + 1);
		int do_map = i >= j;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);
		if (j == N) continue;
		if (!adj[i].read(j)) continue;

		//double reduced_volume = volume_9000114_3(nodes.crd->getFloat4(i), nodes.crd->getFloat4(j)) / volume;
		double reduced_volume = 0.0;
		if (reduced_volume == 0) continue;
		for (int m = 0; m < nsamples; m++) {
			double xi = M[m] * reduced_volume;
			actth[tid*nsamples+m] += exp(-xi) * (1.0 - xi * (9.0 - xi * (8.0 - 4.0 * xi / 3.0)));
		}
		cnt++;
	}

	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < nsamples; j++)
			actth[j] += actth[i*nsamples+j];
	//printf("Integral: %f\n", actth[nsamples-1]);
	//printf("pairs used: %d\n", cnt);

	double T1 = -51.0 * sin(eta0) + 7.0 * sin(3.0 * eta0) + 6.0 * sec(eta0) * (eta0 * (3.0 + POW2(sec(eta0))) + tan(eta0));
	double T2 = 3.0 * sin(eta0) + sin(3.0 * eta0);
	double Vp = (T1 / T2) * TWO_PI / 9.0;

	for (int i = 0; i < nsamples; i++) {
		//actth[i] = 4.0 * sqrt(volume / (6.0 * M[i])) * (M[i] - actth[i]);
		actth[i] = 4.0 * sqrt(M[i] * volume / 6.0) * (1.0 - (static_cast<double>(M[i]) / cnt) * (Vp / volume) * actth[i]);
		printf("%d\t%f\n", M[i], actth[i]);
	}

	stopwatchStop(&sMeasureThAction);

	if (!bench) {
		printf("\n\tCalculated Theoretical Action.\n");
		printf_cyan();
		printf("\t\tFinal Value: %f\n", actth[nsamples-1]);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureThAction.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureChain(Bitvector &longest_chains, int &chain_length, std::pair<int,int> &longest_pair, const Node &nodes, Bitvector &adj, const int N, CaResources * const ca, Stopwatch &sMeasureChain, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert ((int)adj.size() >= N);
	assert (N > 0);
	assert (ca != NULL);
	#endif

	//static const bool DEBUG_CHAIN = false;
	FastBitset workspace(N);
	FastBitset longest_chain(N);
	int *lengths = NULL;

	stopwatchStart(&sMeasureChain);

	try {
		lengths = (int*)malloc(sizeof(int) * N);
		if (lengths == NULL)
			throw std::bad_alloc();
		memset(lengths, -1, sizeof(int) * N);
		ca->hostMemUsed += sizeof(int) * N;
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	//This is the original algorithm
	int len = 0;
	for (int i = 0; i < N; i++) {
		if (!!nodes.k_in[i]) continue;
		for (int j = i + 1; j < N; j++) {
			if (!!nodes.k_out[j]) continue;
			if (!nodesAreConnected_v2(adj, N, i, j)) continue;
			len = longestChain_v2(adj, &workspace, lengths, N, i, j, 0);
			if (len > chain_length) {
				chain_length = len;
				longest_pair = std::make_pair(i, j);
			}
		}
	}

	//DEBUG longestChain_v3
	//printf("Studying Pair (%d, %d):\n", std::get<0>(longest_pair), std::get<1>(longest_pair));
	//printf("Length (v2): %d\n", chain_length);
	//fflush(stdout);

	//This is the version which saves the actual chain elements

	//This was added for Half_Diamond_T ONLY
	/*longest_pair = std::make_pair(0, N - 1);

	Bitvector chains;
	chains.reserve(N);
	workspace.reset();
	for (int i = 0; i < N; i++)
		chains.push_back(workspace);
	
	int lc3 = longestChain_v3(adj, chains, longest_chain, &workspace, lengths, N, std::get<0>(longest_pair), std::get<1>(longest_pair), 0);
	longest_chains.push_back(longest_chain);
	chain_length = lc3;

	if (DEBUG_CHAIN) {
		std::ofstream f("longest_chain.dat", std::ios::out);
		for (int i = 0; i < N; i++)
			if (longest_chain.read(i))
				f << i << "\n";
		f.flush();
		f.close();
	}*/

	//END DEBUG

	free(lengths);
	lengths = NULL;
	ca->hostMemUsed -= sizeof(int) * N;

	stopwatchStop(&sMeasureChain);

	if (!bench) {
		printf("\tCalculated Maximum Chain.\n");
		printf_cyan();
		printf("\tMaximum Chain Length: %d\n", chain_length);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureChain.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//NOTE: This assumes nodes have been ranked by degree already
bool measureHubDensity(float &hub_density, float *& hub_densities, Bitvector &adj, const int * const k_in, const int * const k_out, const int N, int N_hubs, CaResources * const ca, Stopwatch &sMeasureHubs, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() > 0);
	assert (k_in != NULL);
	assert (k_out != NULL);
	assert (N > 0);
	assert (N_hubs > 0);
	assert (ca != NULL);
	#endif

	stopwatchStart(&sMeasureHubs);

	if (N_hubs > N)
		N_hubs = N;

	try {
		hub_densities = (float*)calloc(N_hubs + 1, sizeof(float));
		if (hub_densities == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += sizeof(float) * (N_hubs + 1);
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Failed to allocate memory in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	std::vector<uint64_t> hubs;
	hubs.reserve(N_hubs);
	ca->hostMemUsed += sizeof(uint64_t) * N_hubs;
	
	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Hub Density", ca->hostMemUsed, ca->devMemUsed, 0);

	int k_max;
	int idx;
	int k;
	for (int i = 0; i < N_hubs; i++) {
		k_max = 0;
		idx = -1;
		for (int j = 0; j < N; j++) {
			if ((k = k_in[j] + k_out[j]) >= k_max && std::find(hubs.begin(), hubs.end(), j) == hubs.end()) {
				k_max = k;
				idx = j;
			}
		}
		if (idx == -1)
			printf("i: %d\n", i);
		hubs.push_back(idx);
	}

	hub_densities[0] = hub_densities[1] = 0.0;
	for (int k = 2; k <= N_hubs; k++) {
		uint64_t tot = 0;
		for (int i = 0; i < k - 1; i++)
			for (int j = i + 1; j < k; j++)
				if (adj[hubs[i]].read(hubs[j]))
					tot++;
		hub_densities[k] = static_cast<float>(static_cast<long double>(tot << 1) / (k * (k - 1)));
	}
	hub_density = hub_densities[N_hubs];

	hubs.clear();
	hubs.swap(hubs);
	ca->hostMemUsed -= sizeof(uint64_t) * N_hubs;

	stopwatchStop(&sMeasureHubs);

	if (!bench) {
		printf("\tCalculated Hub Density.\n");
		printf("\tUsed %d Hubs.\n", N_hubs);
		printf("\tHub 0 has degree %d\n", k_in[hubs[0]] + k_out[hubs[0]]);
		printf_cyan();
		printf("\tHub Density: %.6f\n", hub_density);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureHubs.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Measure fraction of geodesically disconnected pairs
bool measureGeoDis(float &geo_discon, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int N, const long double N_gd, const double a, const double zeta, const double zeta1, const double r_max, const double alpha, const float core_edge_fraction, MersenneRNG &mrng, Stopwatch &sMeasureGeoDis, const bool use_bit, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("FLRW"));
	assert (spacetime.curvatureIs("Flat"));
	assert (nodes.crd->getDim() == 4);
	assert (nodes.crd->w() != NULL);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.crd->z() != NULL);
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	} else
		assert (adj.size() >= (size_t)N);

	assert (N > 0);
	assert (N_gd > 0 && N_gd <= ((uint64_t)N * (N - 1)) >> 1);
	if (spacetime.manifoldIs("FLRW"))
		assert (alpha > 0.0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	#endif

	int n = N + N % 2;
	uint64_t max_pairs = static_cast<uint64_t>(n) * (n - 1) / 2;
	uint64_t npairs = static_cast<uint64_t>(N_gd);
	uint64_t ndis = 0;

	stopwatchStart(&sMeasureGeoDis);

	unsigned int seed;
	unsigned int nthreads;

	#ifdef _OPENMP
	seed = static_cast<unsigned int>(mrng.rng() * 4000000000);
	nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);
	#pragma omp parallel reduction (+ : ndis) if (npairs >= 1024)
	{
	Engine eng(seed ^ omp_get_thread_num());
	UDistribution dst(0.0, 1.0);
	UGenerator rng(eng, dst);
	#pragma omp for schedule (dynamic, 1)
	#else
	UGenerator &rng = mrng.rng;
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		uint64_t vec_idx = static_cast<uint64_t>(rng() * (max_pairs - 1)) + 1;
		int i = static_cast<int>(vec_idx / (n - 1));
		int j = static_cast<int>(vec_idx % (n - 1) + 1);
		int do_map = i >= j;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);
		if (j == N) continue;
		if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction, i, j))
			continue;

		float dist = 0.0;
		if (spacetime.manifoldIs("De_Sitter"))
			dist = distanceDeSitterFlat(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j);
		else if (spacetime.manifoldIs("FLRW"))
			dist = distanceFLRW(nodes.crd, nodes.id.tau, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j);

		if (dist + 1.0 > INF)
			ndis++;
	}
	#ifdef _OPENMP
	}
	#endif

	geo_discon = static_cast<float>(static_cast<long double>(ndis) / npairs);

	stopwatchStop(&sMeasureGeoDis);

	if (!bench) {
		printf("\tCalculated Fraction of Geodesically Disconnected Pairs.\n");
		printf_cyan();
		printf("\t\tFraction of Pairs: %f\n", geo_discon);
		printf("\t\tTraversed Pairs: %" PRIu64 "\n", npairs);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureGeoDis.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureFoliation(Bitvector &timelike_foliation, Bitvector &spacelike_foliation, std::vector<unsigned int> &ax_set, Bitvector &adj, const Node &nodes, const int N, CaResources * const ca, Stopwatch &sMeasureFoliation, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (N > 0);
	assert (ca != NULL);
	#endif

	Bitvector chains[omp_get_max_threads()];
	Bitvector workspace;
	int *lengths;
	float volume_fraction = 0.0;

	stopwatchStart(&sMeasureFoliation);
	
	try {
		for (int i = 0; i < omp_get_max_threads(); i++) {
			chains[i].reserve(N);
			for (int j = 0; j < N; j++) {
				FastBitset fb(N);
				chains[i].push_back(fb);
				//ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
			}
		}

		workspace.reserve(omp_get_max_threads());
		for (int i = 0; i < omp_get_max_threads(); i++) {
			FastBitset w(N);
			workspace.push_back(w);
			ca->hostMemUsed += sizeof(BlockType) * w.getNumBlocks();
		}

		lengths = (int*)malloc(sizeof(int) * N * omp_get_max_threads());
		if (lengths == NULL)
			throw std::bad_alloc();
		memset(lengths, -1, sizeof(int) * N * omp_get_max_threads());
		ca->hostMemUsed += sizeof(int) * N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	Bitvector chain_pairs;
	std::vector<int> chain_indices;
	std::vector<int> chain_lengths;
	chain_pairs.reserve(N);
	chain_lengths.reserve(N);
	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 8)
	#endif
	for (int i = 0; i < N; i++) {	//These will be minimal elements
		int tid = omp_get_thread_num();
		if (!!nodes.k_in[i]) continue;
		for (int j = 0; j < N; j++) {	//These will be maximal elements
			if (!!nodes.k_out[j]) continue;
			if (!adj[i].read(j)) continue;

			FastBitset fb(N);
			int c = longestChain_v3(adj, chains[tid], fb, &workspace[tid], lengths + tid * N, N, i, j, 0);
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
			chain_pairs.push_back(fb);
			chain_lengths.push_back(c);
			}
		}
	}
	chain_indices.resize(chain_pairs.size());
	std::iota(chain_indices.begin(), chain_indices.end(), 0);

	//DEBUG
	//printf("\nChain candidates:\n");
	/*for (unsigned int k = 0; k < chain_pairs.size(); k++) {
		uint64_t i = chain_pairs[k].next_bit();
		uint64_t j = chain_pairs[k].prev_bit();
		if (!!nodes.k_in[i]) printf("ERROR\n");
		if (!!nodes.k_out[j]) printf("ERROR\n");
		printf("Chain [%d]: (%d, %d) Size: [%d]\n", k, (int)i, (int)j, chain_lengths[k]);
	}*/

	//printf("\nAdding candidates to timelike foliation.\n");
	//printf("[%zd] chains will be considered.\n", chain_pairs.size());
	timelike_foliation.reserve(chain_pairs.size());
	while (!chain_indices.empty()) {
		int longest = -1, idx = -1;
		for (unsigned int i = 0; i < chain_indices.size(); i++) {
			int I = chain_indices[i];
			if (chain_lengths[I] > longest) {
				longest = chain_lengths[I];
				idx = I;
			}
		}

		//int i = chain_pairs[idx].next_bit();
		//int j = chain_pairs[idx].prev_bit();
		//printf("Adding chain [%d] with endpoints (%d, %d)\n", idx, i, j);

		timelike_foliation.push_back(chain_pairs[idx]);
		break;

		/*for (int k = chain_indices.size() - 1; k >= 0; k--) {
			//printf("\tk: %d\n", k);
			int K = chain_indices[k];
			//if (!chain_pairs[K].any()) printf("Error here.\n");
			int m = chain_pairs[K].next_bit();
			int n = chain_pairs[K].prev_bit();
			//if (chain_pairs[k].read(i) || chain_pairs[k].read(j)) {
			if (m == i || n == j) {
				//printf("\tRemoving chain [%d] with endpoints (%d, %d)\n", K, m, n);
				chain_indices.erase(chain_indices.begin() + k);
				//chain_lengths.erase(chain_lengths.begin() + K);
			} 
		}*/
		/*printf("\tFinished removing chains\n");
		printf("There are [%zd] candidates left.\n", chain_indices.size());
		printf("\tRemaining chain candidates:\n");
		for (int k = chain_indices.size() - 1; k >= 0; k--) {
			int K = chain_indices[k];
			if (!chain_pairs[K].any()) printf("ERROR!\n");
			uint64_t i = chain_pairs[K].next_bit();
			uint64_t j = chain_pairs[K].prev_bit();
			printf("\t\tChain [%d]: (%d, %d) Size: [%d]\n", K, (int)i, (int)j, chain_lengths[K]);
		}*/
		//break;
		//dbg_cnt++;
	}


	//DEBUG
	std::ofstream d("chain_dist.dat", std::ios::out | std::ios::app);
	for (int i = 0; i < N; i++) {
		if (timelike_foliation[0].read(i))
			d << nodes.crd->y(i) << std::endl;
	}
	d.flush();
	d.close();
	if (N) return true;
	//END DEBUG

	//std::ofstream os("ax_dbg.cset.dbg.dat");
	ax_set.reserve(timelike_foliation.size());
	for (unsigned int i = 0; i < timelike_foliation.size(); i++) {
		int m = timelike_foliation[i].next_bit();
		int n = timelike_foliation[i].prev_bit();
		//uint64_t x = adj[m].partial_vecprod(adj[n], m, N - n + 1);
		adj[m].clone(workspace[0]);
		workspace[0].setIntersection(adj[n]);
		uint64_t x = workspace[0].count_bits();
		ax_set.push_back((unsigned int)x);
		//printf("Endpoints at r = (%f, %f) has set size [%d] (chain length [%d])\n", nodes.crd->y(m), nodes.crd->y(n), (int)x, (int)timelike_foliation[i].count_bits());
		//os << nodes.crd->y(m) << " " << (int)x << "\n";
	}
	//os.flush();
	//os.close();

	/*printf("\nTimelike foliation:\n");
	for (unsigned int i = 0; i < timelike_foliation.size(); i++) {
		uint64_t start = timelike_foliation[i].next_bit();
		printf("[%d, x=%f]: ", (int)timelike_foliation[i].count_bits(), nodes.crd->y(start));
		timelike_foliation[i].unset(start);
		printf("(%d", (int)start);
		while (timelike_foliation[i].any()) {
			uint64_t next = timelike_foliation[i].next_bit();
			printf(" -> %d", (int)next);
			timelike_foliation[i].unset(next);
		}
		printf(")\n\n");
	}*/

	//Now do the spacelike foliation
	//printf("\nSpacelike foliation:\n");
	int max_antichain_size = 0;
	timelike_foliation[0].clone(workspace[0]);
	spacelike_foliation.reserve(timelike_foliation[0].count_bits());
	while (workspace[0].any()) {
		uint64_t seed = workspace[0].next_bit();
		//Find the antichain
		FastBitset fb(N);
		int x = maximalAntichain(fb, adj, N, seed);
		max_antichain_size = x > max_antichain_size ? x : max_antichain_size;
		//printf("[%d]\n", x);
		spacelike_foliation.push_back(fb);
		workspace[0].unset(seed);
	}

	//adj[timelike_foliation[0].next_bit()].clone(workspace[0]);
	//workspace[0].setIntersection(adj[timelike_foliation[0].prev_bit()]);
	//volume_fraction = static_cast<float>(workspace[0].count_bits()) / N;
	volume_fraction = static_cast<float>(ax_set[0]) / N;

	int P0 = 0, P1 = 0, F0 = 0, F1 = 0;
	for (int i = 0; i < N; i++) {
		if (nodes.k_in[i] == 0) P0++;
		if (nodes.k_in[i] == 1) P1++;
		if (nodes.k_out[i] == 0) F0++;
		if (nodes.k_out[i] == 1) F1++;
	}

	for (int i = 0; i < omp_get_max_threads(); i++) {
		//ca->hostMemUsed -= sizeof(BlockType) * chains[i].size() * chains[i][0].getNumBlocks();
		chains[i].clear();
		chains[i].swap(chains[i]);
	}

	ca->hostMemUsed -= sizeof(BlockType) * workspace.size() * workspace[0].getNumBlocks();
	workspace.clear();
	workspace.swap(workspace);
	
	free(lengths);
	lengths = NULL;
	ca->hostMemUsed -= sizeof(int) * N * omp_get_max_threads();

	stopwatchStop(&sMeasureFoliation);


	//DEBUG
	int ma_length = 0;
	FastBitset fb(N);
	for (int i = 0; i < N; i++) {
		int ma = maximalAntichain(fb, adj, N, i);
		if (ma > ma_length) ma_length = ma;
	}
	printf("Longest should be: [%d]\n", ma_length);


	if (!bench) {
		printf("\tConstructed Spacetime Foliation.\n");
		printf_cyan();
		printf("\t\tChains:     %d (Max: %d)\n", (int)timelike_foliation.size(), (int)timelike_foliation[0].count_bits());
		printf("\t\tAntichains: %d (Max: %d)\n", (int)spacelike_foliation.size(), max_antichain_size);
		printf("\t\tVolume:     %d\n", ax_set[0]);
		printf("\t\tVolume Fraction: %f\n", volume_fraction);
		printf_std();
		printf("\tSpacelike Boundary:\n");
		printf_cyan();
		printf("\t\tP0: %d\n", P0);
		printf("\t\tP1: %d\n", P1);
		printf("\t\tF0: %d\n", F0);
		printf("\t\tF1: %d\n", F1);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureFoliation.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureAntichain(const Bitvector &adj, const Node &nodes, const int N, MersenneRNG &mrng, Stopwatch &sMeasureAntichain, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	#endif

	FastBitset antichain(N);
	FastBitset candidates(N);
	FastBitset workspace(N);
	FastBitset workspace2(N);
	//unsigned int start = static_cast<unsigned int>(mrng.rng() * N);
	//unsigned int start = N / 2;
	//printf("Start: [%d]\n", start);
	uint64_t size = 0;

	//DEBUG
	int min_k = 10000000, min_idx = -1;
	for (int i = 0; i < N; i++) {
		uint64_t k = adj[i].count_bits();
		if ((int)k < min_k) {
			min_k = k;
			min_idx = i;
		}
	}
	unsigned int start = min_idx;
	printf("Minimum degree: %d\n", min_k);
	printf("X Coordinate: %f\n", nodes.crd->y(min_idx));

	//END DEBUG

	stopwatchStart(&sMeasureAntichain);

	adj[start].clone(antichain);	//1's indicate relations to 'start'
	antichain.flip();		//1's indicate elements unrelated to 'start'

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

	//We should be left with an antichain
	//We check to make sure no elements in this antichain are related
	#if DEBUG
	for (int i = 0; i < N; i++) {
		if (!antichain.read(i)) continue;
		for (int j = i + 1; j < N; j++) {
			if (!antichain.read(j)) continue;
			if (adj[i].read(j))
				printf("ERROR!\n");
		}
	}
	#endif

	std::ofstream f("output.txt", std::ios::out);
	for (int i = 0; i < N; i++)
		if (antichain.read(i))
			f << i << std::endl;
	f.flush();
	f.close();

	//DEBUG
	std::ofstream d("antichain_dist.dat", std::ios::out | std::ios::app);
	for (int i = 0; i < N; i++)
		if (antichain.read(i))
			d << nodes.crd->x(i) << std::endl;
	d.flush();
	d.close();
	//END DEBUG

	size = antichain.count_bits();
	stopwatchStop(&sMeasureAntichain);

	if (!bench) {
		printf("\tIdentified a Random Maximal Antichain.\n");
		printf_cyan();
		printf("\t\tSize: [%" PRIu64 "] Elements\n", size);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureAntichain.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureDimension(float &dimension, Bitvector &adj, const Spacetime &spacetime, const int N, std::pair<int,int> longest_pair, Stopwatch &sMeasureDimension, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (spacetime.manifoldIs("Minkowski"));
	assert (N > 0);
	#endif

	stopwatchStart(&sMeasureDimension);

	//Number of elements in the Alexandroff set
	//int i = std::get<0>(longest_pair);
	//int j = std::get<1>(longest_pair);
	//uint64_t elements = adj[i].partial_vecprod(adj[j], i, j - i + 1);
	int i = 0, j = N - 1;
	uint64_t elements = N;
	if (strcmp(Spacetime::regions[spacetime.get_region()], "Diamond")) {
		i = std::get<0>(longest_pair);
		j = std::get<1>(longest_pair);
		elements = adj[i].partial_vecprod(adj[j], i, j - i + 1);
	}

	uint64_t relations = 0;
	for (int k = i; k <= j; k++)
		relations += adj[k].partial_count(i, j - i + 1);
	relations >>= 1;

	//float stdim = atof(Spacetime::stdims[spacetime.get_stdim()]);
	double lhs = (double)relations / POW2(elements);
	//float rhs = GAMMA(stdim + 1) * GAMMA(stdim / 2) / (4.0 * GAMMA(1.5 * stdim));

	//printf("lhs: %f\n", lhs);
	//printf("rhs: %f\n", rhs);
	double x = 1.0;
	if (!bisection(&solve_mmdim_bisec, &x, 2000, 0.0, 5.0, TOL, false, &lhs, NULL, NULL))
		x = NAN;

	assert (x == x);
	dimension = x;

	stopwatchStop(&sMeasureDimension);

	if (!bench) {
		printf("\tCalculated Manifold Dimension.\n");
		printf("\t\tPrescription: Myrheim-Meyer\n");
		printf_cyan();
		printf("\t\tDimension: %.5f\n", dimension);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureDimension.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureSpacetimeMutualInformation(uint64_t *& smi, Bitvector &adj, const Node &nodes, const Spacetime &spacetime, const int N, const double eta0, const double r_max, CaResources * const ca, Stopwatch &sMeasureSMI, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	#endif

	//Partition the causal set
	FastBitset set0(N);	//Past elements
	FastBitset set1(N);	//Future elements

	//Partition it
	set0.set();
	for (int i = 0; i < N; i++)
		if (smi_hypersurface(nodes.crd, spacetime, eta0, r_max, i))
			set1.set(i);
	//set1.clone(set0);

	uint64_t s0size = set0.count_bits();
	uint64_t s1size = set1.count_bits();
	printf("s0size: %u\n", (unsigned)s0size);
	printf("s1size: %u\n", (unsigned)s1size);
	uint64_t npairs = s0size * s1size;

	std::vector<int> idxA(s0size);
	std::vector<int> idxB(s1size);

	for (unsigned i = 0; i < s0size; i++)
		set0.unset(idxA[i] = set0.next_bit());
	for (unsigned i = 0; i < s1size; i++)
		set1.unset(idxB[i] = set1.next_bit());

	std::ofstream os0("i.dat");
	for (unsigned i = 0; i < s0size; i++)
		os0 << nodes.crd->y(idxA[i]) << " " << nodes.crd->x(idxA[i]) << "\n";
	os0.flush();
	os0.close();

	std::ofstream os1("j.dat");
	for (unsigned i = 0; i < s1size; i++)
		os1 << nodes.crd->y(idxB[i]) << " " << nodes.crd->x(idxB[i]) << "\n";
	os1.flush();
	os1.close();
	//printChk();

	stopwatchStart(&sMeasureSMI);

	//Allocate memory for cardinality measurements and workspace
	try {
		smi = (uint64_t*)calloc(N * omp_get_max_threads(), sizeof(uint64_t));
		if (smi == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += sizeof(uint64_t) * N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Spacetime Mutual Information", ca->hostMemUsed, ca->devMemUsed, 0);

	//The first element will be N
	//smi[0] = N;
	smi[0] = s1size;

	unsigned int nthreads = omp_get_max_threads();
	#ifdef AVX2_ENABLED
	nthreads >>= 1;
	#endif

	//Compare all pairs of elements
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 256) num_threads (nthreads) if (npairs >= 1024)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned int tid = omp_get_thread_num();
		//Choose a pair
		uint64_t i = k / s1size;
		uint64_t j = k % s1size;
		i = idxA[i];
		j = idxB[j];
		if (i >= j) continue;

		//Ignore pairs which are not connected
		if (!nodesAreConnected_v2(adj, N, (int)i, (int)j)) continue;

		//Save the cardinality
		smi[tid*N+adj[i].partial_vecprod(adj[j], i, j - i + 1)+1]++;
	}

	//Reduction for OpenMP
	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			smi[j] += smi[i*N+j];

	stopwatchStop(&sMeasureSMI);

	if (!bench) {
		printf("\tCalculated Spacetime Mutual Information.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureSMI.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool measureExtrinsicCurvature(Network *network, CaResources * const ca, CUcontext ctx)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	#endif

	static const bool DEBUG_EXTRINSIC = false;

	int *iwork = NULL;
	Network subgraph;
	Bitvector fwork, fwork2, chains;
	std::vector<std::pair<int, int>> pwork;
	std::pair<int, int> *i2work = NULL;
	Stopwatch sIdentify;
	unsigned L;
	//float K = 0.0;

	try {
		network->network_observables.layers = (int*)malloc(sizeof(int) * network->network_properties.N * omp_get_max_threads());
		if (network->network_observables.layers == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += sizeof(int) * network->network_properties.N * omp_get_max_threads();

		iwork = (int*)malloc(sizeof(int) * network->network_properties.N * omp_get_max_threads());
		if (iwork == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += sizeof(int) * network->network_properties.N * omp_get_max_threads();
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (network->network_properties.flags.verbose)
		printMemUsed("to Identify Boundary Candidates", ca->hostMemUsed, ca->devMemUsed, 0);

	identifyTimelikeCandidates(network->network_observables.timelike_candidates, network->network_observables.layers, iwork, fwork, network->nodes, network->adj, network->network_properties.spacetime, network->network_properties.N, sIdentify, network->network_properties.flags.verbose, network->network_properties.flags.bench);
	if (LAYER_ONLY)
		goto ExtrinsicExit;

	if (DEBUG_EXTRINSIC) {
		//Print Coordinates
		printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N, "eta_dist.cset.dbg.dat", "x");
		printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N, "x_dist.cset.dbg.dat", "y");

		std::ofstream layers("layers.dat", std::ios::out);
		std::ofstream lds("layer_degrees.dat", std::ios::out);
		for (int i = 0; i < network->network_properties.N; i++) {
			bool found = false;
			for (int j = 0; j < network->network_properties.N; j++) {
				if (network->network_observables.layers[j] == i) {
					layers << j << " ";
					lds << (network->nodes.k_in[j] + network->nodes.k_out[j]) << " ";
					found = true;
				}
			}
			if (found) {
				layers << "\n";
				lds << "\n";
			}
		}
		layers.flush();
		layers.close();
		lds.flush();
		lds.close();

		std::ofstream cs("candidates.dat", std::ios::out);
		for (size_t i = 0; i < network->network_observables.timelike_candidates.size(); i++)
			cs << network->network_observables.timelike_candidates[i] << "\n";
		cs.flush();
		cs.close();
	}

	{
	Network subgraph = Network(*network);
	if (!configureSubgraph(&subgraph, network->nodes, network->network_observables.timelike_candidates, ca, ctx))
		return false;

	try {
		i2work = (std::pair<int,int>*)malloc(sizeof(std::pair<int,int>) * subgraph.network_properties.N);
		if (i2work == NULL)
			throw std::bad_alloc();
		for (int i = 0; i < subgraph.network_properties.N; i++)
			i2work[i] = std::make_pair(-1, -1);
		ca->hostMemUsed += sizeof(std::pair<int,int>) * subgraph.network_properties.N;
	} catch (const std::bad_alloc&) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
	if (network->network_properties.flags.verbose)
		printMemUsed("to Identify Boundary Chains", ca->hostMemUsed, ca->devMemUsed, 0);

	identifyBoundaryChains(network->network_observables.boundary_chains, pwork, iwork, i2work, fwork, fwork2, network, &subgraph, network->network_observables.timelike_candidates);

	if (DEBUG_EXTRINSIC) {
		//Print Boundary Elements
		std::ofstream f("chains.dat", std::ios::out);
		uint64_t IDX = 0;
		for (size_t i = 0; i < network->network_observables.boundary_chains.size(); i++) {
			printf("Chain [%zd] size: [%" PRIu64 "]\n", i, std::get<0>(network->network_observables.boundary_chains[i]).count_bits());
			printf(" > Length: [%d]\tWeight: [%d]\n", std::get<2>(network->network_observables.boundary_chains[i]), std::get<1>(network->network_observables.boundary_chains[i]));
			fflush(stdout);

			std::get<0>(network->network_observables.boundary_chains[i]).clone(fwork[0]);
			while (fwork[0].any()) {
				fwork[0].unset(IDX = fwork[0].next_bit());
				f << (unsigned)IDX << "\n";
			}
		}
		printf("\n"); fflush(stdout);
		f.flush();
		f.close();
	}

	//Estimate extrinsic curvature
	/*int min_length = 12, max_length = 32, tot_length = 0;
	float K = 0.0;
	for (size_t i = 0; i < boundary_chains.size(); i++) {
		printf("Examining chain [%zd] of length [%d] with weight [%d].\n", i, std::get<2>(boundary_chains[i]), std::get<1>(boundary_chains[i]));
		printf("bitset length: %" PRIu64 "\n", std::get<0>(boundary_chains[i]).count_bits());
		fflush(stdout);
		uint64_t idx = 0;
		std::get<0>(boundary_chains[i]).clone(fwork[0]);
		for (int j = 0; j < std::get<2>(boundary_chains[i]) + 1 - min_length; j++) {
			fwork[0].unset(idx = fwork[0].next_bit());
			std::get<0>(boundary_chains[i]).clone(fwork[1]);
			uint64_t jdx = 0;
			for (int k = 0; k < std::get<2>(boundary_chains[i]); k++) {
				fwork[1].unset(jdx = fwork[1].next_bit());
				int Lp = k - j;
				if (Lp < min_length || Lp > max_length) continue;
				//printChk(1);

				//Look at interval (idx, jdx)
				//Longest chain w.r.t. entire causal set
				//int L = longestChain_v2(network->adj, &fwork[2], iwork, network->network_properties.N, idx, jdx, 0);
				network->adj[idx].clone(fwork[2]);
				//Size of causal interval
				uint64_t N = fwork[2].partial_vecprod(network->adj[jdx], idx, jdx - idx + 1);

				//Extrinsic Curvature
				//float kK = (32.0 * N / POW2((double)L) - 1.0) * 6.0 * sqrt(2.0) / L;
				float kK = (4.0 * N / POW2((double)Lp) - 1.0) * 3.0 / Lp;
				//float kK = (32.0 * N / POW2((double)L) - 1.0) * 3.0 / Lp;
				kK *= sqrt(network->network_properties.delta);
				K += kK;
				//printf("Interval [%d, %d]\tL: [%d]\tK: %.6f\n", idx, jdx, Lp, kK);
				//fflush(stdout);

				tot_length++;
			}
		}
	}
	K /= tot_length;*/

	L = std::get<2>(network->network_observables.boundary_chains[0]);
	//K = (3.0 * sqrt(2.0) / L) * (1.0 - 8.0 * network->network_properties.N / POW2(L)) * sqrt(network->network_properties.delta);

	free(i2work);
	i2work = NULL;
	ca->hostMemUsed -= sizeof(std::pair<int,int>) * subgraph.network_properties.N;

	destroyNetwork(&subgraph, ca->hostMemUsed, ca->devMemUsed);
	}

	ExtrinsicExit:

	free(iwork);
	iwork = NULL;
	ca->hostMemUsed -= sizeof(int) * network->network_properties.N * omp_get_max_threads();

	pwork.clear();
	pwork.swap(pwork);

	fwork.clear();
	fwork.swap(fwork);

	printf("Measured Extrinsic Curvature.\n");
	printf_cyan();
	//printf("Minimum chain length used: %d\n", min_length);
	//printf("Maximum chain length used: %d\n", max_length);
	//printf("Mean extrinsic curvature: %.6f\n", K);
	printf("Longest Boundary Chain: %d\n", L);
	printf_std();
	fflush(stdout);

	return true;
}

/*bool measureEntanglementEntropy(const Bitvector &adj, const Spacetime &spacetime, const int N, Stopwatch &sMeasureEntanglementEntropy, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= N);
	assert (N > 0);
	#endif

	//These are just for now - I will remove them once the function
	//supports more general entanglement entropies
	assert (spacetime.stdimIs("2"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter"));

	cusolverDnHandle_t handle = NULL;
	cusolverStatus_t status = cusolverDnCreate(&handle);

	cuComplex *i_delta = (cuComplex*)malloc(sizeof(cuComplex) * POW2(N));
	assert (i_delta != NULL);
	for (unsigned int i = 0; i < N; i++) //Row
		for (unsigned int j = 0; j < N; j++) //Column
			i_delta[i*N+j] = make_cuComplex(0.0, static_cast<float>(adj[i].read(j)) * 0.5 * SGN(j - i));

	cuComplex *eigenvectors = (cuComplex*)malloc(sizeof(cuComplex) * POW2(N));
	assert (eigenvectors != NULL);
	float *eigenvalues = (float*)malloc(sizeof(float) * N);
	assert (eigenvalues != NULL);

	cuComplex *d_i_delta = NULL;
	cudaMalloc((void**)&d_i_delta, sizeof(cuComplex) * POW2(N));

	cuComplex *d_eigenvalues = NULL;
	cudaMalloc((void**)&d_eigenvalues, sizeof(cuComplex) * N);

	int *d_info = NULL;
	cudaMalloc((void**)&d_info, sizeof(int));

	cudaMemcpy(d_i_delta, i_delta, sizeof(cuComplex) * POW2(N), cudaMemcpyHostToDevice);

	//Tells to compute eigenvectors as well as eigenvalues
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	//Not sure why we need this... see CUSOLVER documentation
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	//Query workspace length
	int lwork = 0;
	status = cusolverDnCheevd_bufferSize(handle, jobz, uplo, N, d_i_delta, N, eigenvalues, &lwork);

	//Allocate workspace
	cuComplex *d_work = NULL;
	cudaMalloc((void**)&d_work, sizeof(cuComplex) * lwork);

	//Find the eigenvalues and eigenvectors!
	status = cusolverDnCheevd(handle, jobz, uplo, N, d_i_delta, N, eigenvalues, d_work, lwork, d_info);
	cudaDeviceSynchronize();

	cuMemcpy(eigenvalues, d_eigenvalues, sizeof(double) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(eigenvectors, d_i_delta, sizeof(double) * POW2(N), cudaMemcpyDeviceToHost);

	//Now we have the eigenvalues and eigenvectors

	return true;
}*/
