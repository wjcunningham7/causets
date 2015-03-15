#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, Stopwatch &sMeasureClustering, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &calc_autocorr, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		assert (N_tar > 0);
		assert (N_deg2 > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	}

	float c_avg = 0.0f;

	stopwatchStart(&sMeasureClustering);

	try {
		clustering = (float*)malloc(sizeof(float) * N_tar);
		if (clustering == NULL)
			throw std::bad_alloc();
		memset(clustering, 0, sizeof(float) * N_tar);
		hostMemUsed += sizeof(float) * N_tar;
	} catch (std::bad_alloc()) {
		fprintf(stderr, "Failed to allocate memory in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Clustering", hostMemUsed, devMemUsed, 0);

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

		float c_i = 0.0f;
		float c_k = static_cast<float>((nodes.k_in[i] + nodes.k_out[i]));
		float c_max = c_k * (c_k - 1.0f) / 2.0f;

		//(1) Consider both neighbors in the past
		if (edges.past_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_in[i]; j++)
				//3 < 2 < 1
				for (int k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.past_edges[edges.past_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(2) Consider both neighbors in the future
		if (edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				//1 < 3 < 2
				for (int k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.future_edges[edges.future_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		//(3) Consider one neighbor in the past and one in the future
		if (edges.past_edge_row_start[i] != -1 && edges.future_edge_row_start[i] != -1)
			for (int j = 0; j < nodes.k_out[i]; j++)
				for (int k = 0; k < nodes.k_in[i]; k++)
					//3 < 1 < 2
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0f;

		if (DEBUG) assert (c_max > 0.0f);
		c_i = c_i / c_max;
		if (DEBUG) assert (c_i <= 1.0f);

		clustering[i] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
		//fflush(stdout);
	}

	average_clustering = c_avg / N_deg2;
	if (DEBUG) assert (average_clustering >= 0.0f && average_clustering <= 1.0f);

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
bool measureConnectedComponents(Node &nodes, const Edge &edges, const int &N_tar, const int &rank, int &N_cc, int &N_gcc, Stopwatch &sMeasureConnectedComponents, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (N_tar > 0);
		#ifdef MPI_ENABLED
		assert (rank >= 0);
		#endif
	}

	int elements;
	int i;

	stopwatchStart(&sMeasureConnectedComponents);

	try {
		nodes.cc_id = (int*)malloc(sizeof(int) * N_tar);
		if (nodes.cc_id == NULL)
			throw std::bad_alloc();
		memset(nodes.cc_id, 0, sizeof(int) * N_tar);
		hostMemUsed += sizeof(int) * N_tar;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Components", hostMemUsed, devMemUsed, rank);

	if (rank == 0) {
		for (i = 0; i < N_tar; i++) {
			elements = 0;
			if (!nodes.cc_id[i] && (nodes.k_in[i] + nodes.k_out[i]) > 0) {
				bfsearch(nodes, edges, i, ++N_cc, elements);
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

	if (DEBUG) {
		assert (N_cc > 0);
		assert (N_gcc > 1);
	}

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
bool measureSuccessRatio(const Node &nodes, const Edge &edges, bool * const core_edge_exists, float &success_ratio, const int &N_tar, const float &k_tar, const double &N_sr, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, const int &num_mpi_threads, const int &rank, Stopwatch &sMeasureSuccessRatio, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		assert (!nodes.crd->isNull());
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);

		if (manifold == HYPERBOLIC)
			assert (dim == 1);

		if (dim == 1) {
			assert (nodes.crd->getDim() == 2);
		} else if (dim == 3) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (manifold == DE_SITTER);
		}

		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		assert (N_tar > 0);
		assert (N_sr > 0 && N_sr <= ((uint64_t)N_tar * (N_tar - 1)) >> 1);
		assert (!(dim == 1 && manifold == DE_SITTER));
		if (manifold == DE_SITTER && universe)
			assert (alpha > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (edge_buffer >= 0);
		#ifdef MPI_ENABLED
		assert (num_mpi_threads > 0);
		assert (rank >= 0);
		#endif
	}

	uint64_t stride = (uint64_t)N_tar * (N_tar - 1) / (static_cast<uint64_t>(N_sr) << 1);
	uint64_t npairs = static_cast<uint64_t>(N_sr);
	uint64_t n_trav = 0;
	uint64_t n_succ = 0;

	double *table;
	bool *used;
	long size = 0L;
	size_t u_size;

	#ifdef MPI_ENABLED
	int edges_size = static_cast<int>(N_tar * k_tar / 2 + edge_buffer);
	int core_edges_size = static_cast<int>(POW2(core_edge_fraction * N_tar, EXACT));
	#endif

	//Check out-degrees of earliest nodes
	//for (int t = 0; t < 100; t++)
	//	printf("%d\n", nodes.k_out[t]);
	//exit(11);

	stopwatchStart(&sMeasureSuccessRatio);

	//TEST THESE PARAMETERS
	u_size = sizeof(bool) * N_tar;
	//printf_mpi(rank, "Original u_size: %zu\n", u_size);
	#ifdef _OPENMP
	u_size *= omp_get_max_threads();
	//printf_mpi(rank, "OpenMP Threads:  %d\n", omp_get_max_threads());
	//printf_mpi(rank, "Max u_size:      %zu\n", u_size);
	#endif
	#ifdef MPI_ENABLED
	//u_size /= (num_mpi_threads / omp_get_max_threads());
	//printf_mpi(rank, "MPI Threads:     %d\n", num_mpi_threads);
	//printf_mpi(rank, "Divided u_size:  %zu\n", u_size);
	#endif

	try {
		used = (bool*)malloc(u_size);
		if (used == NULL)
			throw std::bad_alloc();
		memset(used, 0, u_size);
		hostMemUsed += u_size;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	if (!getLookupTable("./etc/geodesic_table.cset.bin", &table, &size))
		return false;
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Success Ratio", hostMemUsed, devMemUsed, rank);

	#ifdef MPI_ENABLED
	//Broadcast:
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->x(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.crd->y(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (dim == 3) {
		MPI_Bcast(nodes.crd->w(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(nodes.crd->z(), N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	if (manifold == DE_SITTER)
		MPI_Bcast(nodes.id.tau, N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.k_in, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(nodes.k_out, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.past_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edges, edges_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.past_edge_row_start, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(edges.future_edge_row_start, N_tar, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(core_edge_exists, core_edges_size, MPI::BOOL, 0, MPI_COMM_WORLD);
	#endif

	uint64_t start = 0;
	uint64_t finish = npairs;

	#ifdef MPI_ENABLED
	uint64_t mpi_chunk = npairs / num_mpi_threads;
	start = rank * mpi_chunk;
	finish = start + mpi_chunk;
	#endif

	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 1) reduction (+ : n_trav, n_succ)
	#endif
	for (uint64_t k = start; k < finish; k++) {
		//Pick Unique Pair
		uint64_t vec_idx = k * stride + 1;
		int i = static_cast<int>(vec_idx / (N_tar - 1));
		int j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		int do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		//If either node is isolated, continue
		if (!(nodes.k_in[i] + nodes.k_out[i]) || !(nodes.k_in[j] + nodes.k_out[j]))
			continue;

		//If the nodes are in different components, continue
		if (nodes.cc_id[i] != nodes.cc_id[j])
			continue;

		//Set all nodes to "not yet used"
		memset(used + N_tar * omp_get_thread_num(), 0, sizeof(bool) * N_tar);

		//Begin Traversal from i to j
		bool success = traversePath(nodes, edges, core_edge_exists, &used[N_tar*omp_get_thread_num()], table, N_tar, dim, manifold, a, zeta, alpha, core_edge_fraction, size, universe, compact, i, j);

		n_trav++;
		if (success)
			n_succ++;
	}

	#ifdef MPI_ENABLED
	//Reduce (In-Place):
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
		MPI_Reduce(MPI_IN_PLACE, &n_succ, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&n_succ, NULL, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
		MPI_Reduce(MPI_IN_PLACE, &n_trav, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	else
		MPI_Reduce(&n_trav, NULL, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	if (rank == 0 && n_trav > 0)
		success_ratio = static_cast<float>(n_succ) / n_trav;

	free(used);
	used = NULL;
	hostMemUsed -= u_size;

	stopwatchStop(&sMeasureSuccessRatio);

	if (!bench) {
		printf_mpi(rank, "\tCalculated Success Ratio.\n");
		if (rank == 0) printf_cyan();
		printf_mpi(rank, "\t\tSuccess Ratio: %f\n", success_ratio);
		printf_mpi(rank, "\t\tTraversed Pairs: %" PRIu64 "\n", n_trav);
		if (rank == 0) printf_std();
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
//O(xxx) Efficiency (revise this)
bool traversePath(const Node &nodes, const Edge &edges, const bool * const core_edge_exists, bool * const &used, const double * const table, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const float &core_edge_fraction, const long &size, const bool &universe, const bool &compact, int source, int dest)
{
	if (DEBUG) {
		assert (!nodes.crd->isNull());
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == HYPERBOLIC);

		if (manifold == HYPERBOLIC)
			assert (dim == 1);

		if (dim == 1) {
			assert (nodes.crd->getDim() == 2);
		} else if (dim == 3) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (manifold == DE_SITTER);
		}

		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.k_in != NULL);
		assert (nodes.k_out != NULL);
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);
		assert (used != NULL);
		assert (table != NULL);
		
		assert (N_tar > 0);
		if (manifold == DE_SITTER) {
			assert (a > 0.0);
			assert (HALF_PI - zeta > 0.0);
			if (universe)
				assert (alpha > 0.0);
		}
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (size > 0);
		assert (source >= 0 && source < N_tar);
		assert (dest >= 0 && dest < N_tar);
	}

	float min_dist = 0.0f;
	int loc = source;
	int idx_a = source;
	int idx_b = dest;

	float dist;
	int next;

	bool success = false;

	//While the current location (loc) is not equal to the destination (dest)
	while (loc != dest) {
		next = loc;
		dist = INF;
		min_dist = INF;
		used[loc] = true;

		//These would indicate corrupted data
		if (DEBUG) {
			assert (!(edges.past_edge_row_start[loc] == -1 && nodes.k_in[loc] > 0));
			assert (!(edges.past_edge_row_start[loc] != -1 && nodes.k_in[loc] == 0));
			assert (!(edges.future_edge_row_start[loc] == -1 && nodes.k_out[loc] > 0));
			assert (!(edges.future_edge_row_start[loc] != -1 && nodes.k_out[loc] == 0));
		}

		//(1) Check past relations
		for (int m = 0; m < nodes.k_in[loc]; m++) {
			idx_a = edges.past_edges[edges.past_edge_row_start[loc]+m];

			//(A) If the current location's (loc's) past neighbor (idx_a) is the destination (idx_b) then return true
			if (idx_a == idx_b)
				return true;

			//(B) If the current location's past neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, idx_a, idx_b))
				return true;

			//(C) Otherwise find the past neighbor closest to the destination
			if (manifold == DE_SITTER) {
				if (compact)
					dist = distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], dim, manifold, a, alpha, universe, compact);
				else
					dist = distance(table, nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], dim, manifold, a, alpha, size, universe, compact);
			} else if (manifold == HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), dim, manifold, zeta);

			if (dist < 0)
				return false;

			//Save the minimum distance
			if (dist <= min_dist) {
				min_dist = dist;
				next = idx_a;
			}
		}

		//(2) Check future relations
		//OpenMP is implemented here for the early nodes which have lots of out-degrees
		//However, it does not appear to provide a speedup...
		#ifdef _OPENMP
		float priv_min_dist = min_dist;
		int priv_next = next;
		//bool make_parallel = nodes.k_out[loc] > 10000;
		//bool make_parallel = false;
		#pragma omp parallel shared (next, min_dist) \
				     firstprivate (idx_a, priv_min_dist, priv_next) \
				     if (false)
		{
		#pragma omp for schedule (dynamic, 1)
		#endif
		for (int m = 0; m < nodes.k_out[loc]; m++) {
			#ifdef _OPENMP
			if (priv_next == idx_b || priv_next == -1)
				continue;
			#endif

			idx_a = edges.future_edges[edges.future_edge_row_start[loc]+m];

			//(D) If the current location's future neighbor is the destination then return true
			if (idx_a == idx_b) {
				#ifdef _OPENMP
				priv_min_dist = 0.0f;
				priv_next = idx_b;
				continue;
				#else
				return true;
				#endif
			}

			//(E) If the current location's future neighbor is directly connected to the destination then return true
			if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, idx_a, idx_b)) {
				#ifdef _OPENMP
				priv_min_dist = 0.0f;
				priv_next = idx_b;
				continue;
				#else
				return true;
				#endif
			}

			//(F) Otherwise find the future neighbor closest to the destination
			if (manifold == DE_SITTER) {
				if (compact)
					dist = distanceEmb(nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], dim, manifold, a, alpha, universe, compact);
				else
					dist = distance(table, nodes.crd->getFloat4(idx_a), nodes.id.tau[idx_a], nodes.crd->getFloat4(idx_b), nodes.id.tau[idx_b], dim, manifold, a, alpha, size, universe, compact);
			} else if (manifold == HYPERBOLIC)
				dist = distanceH(nodes.crd->getFloat2(idx_a), nodes.crd->getFloat2(idx_b), dim, manifold, zeta);

			if (dist < 0)
				idx_a = -1;

			#ifdef _OPENMP
			if (dist <= priv_min_dist) {
				priv_min_dist = dist;
				priv_next = idx_a;
			}
			#else
			if (dist <= min_dist) {
				min_dist = dist;
				next = idx_a;
			}
			#endif
		}

		#ifdef _OPENMP
		if (next != idx_b) {
			#pragma omp flush (min_dist)
			if (priv_min_dist <= min_dist) {
				#pragma omp critical
				{
					if (priv_min_dist <= min_dist) {
						min_dist = priv_min_dist;
						next = priv_next;
					}
				}
			}
		}
		}
		#endif

		if (next == idx_b)
			return true;
		else if (next == -1)
			return false;

		if (!used[next])
			loc = next;
		else
			break;
	}

	return success;
}

//Takes N_df measurements of in-degree and out-degree fields at time tau_m
//O(xxx) Efficiency (revise this)
bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const int &N_tar, int &N_df, const double &tau_m, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const double &delta, long &seed, Stopwatch &sMeasureDegreeField, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No Null Pointers
		assert (c->getDim() == 4);
		assert (!c->isNull());
		assert (c->w() != NULL);
		assert (c->x() != NULL);
		assert (c->y() != NULL);
		assert (c->z() != NULL);

		//Parameters in Correct Ranges
		assert (N_tar > 0);
		assert (N_df > 0);
		assert (tau_m > 0.0);
		assert (dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
		assert (HALF_PI - zeta > 0.0);
		if (universe)
			assert (alpha > 0.0);
	}

	double *table;
	float4 test_node;
	double eta_m;
	double d_size/*, x, rval*/;
	float dt, dx;
	long size = 0L;
	int k_in, k_out;
	int i, j;

	//Numerical Integration Parameters
	double *params = NULL;
	double *params2 = (double*)malloc(sizeof(double));

	//Calculate theoretical values
	double k_in_theory = 0.0;
	double k_out_theory = 0.0;
	bool theoretical = universe && verbose;

	//Modify number of samples
	N_df = 1;

	IntData idata = IntData();
	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-5;
	if (universe && (USE_GSL || theoretical))
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	stopwatchStart(&sMeasureDegreeField);

	//Allocate memory for data
	try {
		in_degree_field = (int*)malloc(sizeof(int) * N_df);
		if (in_degree_field == NULL)
			throw std::bad_alloc();
		memset(in_degree_field, 0, N_df);
		hostMemUsed += sizeof(int) * N_df;

		out_degree_field = (int*)malloc(sizeof(int) * N_df);
		if (out_degree_field == NULL)
			throw std::bad_alloc();
		memset(out_degree_field, 0, N_df);
		hostMemUsed += sizeof(int) * N_df;

		if (theoretical) {
			if (!getLookupTable("./etc/ctuc_table.cset.bin", &table, &size))
				return false;

			params = (double*)malloc(size + sizeof(double) * 4);
			if (params == NULL)
				throw std::bad_alloc();
			hostMemUsed += size + sizeof(double) * 4;

			params2 = (double*)malloc(sizeof(double));
			if (params == NULL)
				throw std::bad_alloc();
		}
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Degree Fields", hostMemUsed, devMemUsed, 0);
	
	//Calculate eta_m
	if (universe) {
		if (USE_GSL) {
			//Numerical Integration
			idata.upper = tau_m * a;
			params2[0] = a;
			eta_m = integrate1D(&tToEtaUniverse, (void*)params2, &idata, QAGS) / alpha;
			free(params2);
		} else
			//Exact Solution
			eta_m = tauToEtaUniverseExact(tau_m, a, alpha);
	} else
		eta_m = tauToEta(tau_m);
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
		hostMemUsed -= size + sizeof(double) * 4;

		free(params2);
		params2 = NULL;

		free(table);
		table = NULL;
	}

	//Take N_df measurements of the fields
	for (i = 0; i < N_df; i++) {
		test_node.x = 1.0f;
		test_node.y = 1.0f;
		test_node.z = 1.0f;

		//Sample Theta from (0, 2pi)
		/*x = TWO_PI * ran2(&seed);
		test_node.x = static_cast<float>(x);
		if (DEBUG) assert (test_node.x > 0.0f && test_node.x < static_cast<float>(TWO_PI));

		//Sample Phi from (0, pi)
		x = HALF_PI;
		rval = ran2(&seed);
		if (!newton(&solvePhi, &x, 250, TOL, &rval, NULL, NULL, NULL, NULL, NULL)) 
			return false;
		test_node.y = static_cast<float>(x);
		if (DEBUG) assert (test_node.y > 0.0f && test_node.y < static_cast<float>(M_PI));

		//Sample Chi from (0, pi)
		test_node.z = static_cast<float>(ACOS(1.0 - 2.0 * ran2(&seed), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
		if (DEBUG) assert (test_node.z > 0.0f && test_node.z < static_cast<float>(M_PI));*/

		k_in = 0;
		k_out = 0;

		//Compare test node to N_tar other nodes
		float4 new_node;
		for (j = 0; j < N_tar; j++) {
			//Calculate sign of spacetime interval
			new_node = c->getFloat4(j);
			dt = static_cast<float>(ABS(static_cast<double>(c->w(j) - test_node.w), STL));

			if (compact) {
				if (DIST_V2)
					dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v2(new_node, test_node)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
				else
					dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v1(new_node, test_node)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			} else {
				if (DIST_V2)
					dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v2(new_node, test_node)), APPROX ? BITWISE : STL));
				else
					dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v1(new_node, test_node)), APPROX ? BITWISE : STL));
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

	if (universe && (USE_GSL || theoretical))
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
bool measureAction()
{
	return true;
}
