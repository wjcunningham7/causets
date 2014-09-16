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
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Parameters in correct ranges
		assert (N_tar > 0);
		assert (N_deg2 > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	}

	float c_i, c_k, c_max;
	float c_avg = 0.0;
	int i, j, k;

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
		printMemUsed("to Measure Clustering", hostMemUsed, devMemUsed);

	//i represents the node we are calculating the clustering coefficient for (node #1 in triplet)
	//j represents the second node in the triplet
	//k represents the third node in the triplet
	//j and k are not interchanging or else the number of triangles would be doubly counted

	for (i = 0; i < N_tar; i++) {
		//printf("\nNode %d:\n", i);
		//printf("\tDegrees: %d\n", (nodes.k_in[i] + nodes.k_out[i]));
		//printf("\t\tIn-Degrees: %d\n", nodes.k_in[i]);
		//printf("\t\tOut-Degrees: %d\n", nodes.k_out[i]);
		//fflush(stdout);

		//Ingore nodes of degree 0 and 1
		if (nodes.k_in[i] + nodes.k_out[i] < 2) {
			clustering[i] = 0;
			continue;
		}

		c_i = 0.0;
		c_k = static_cast<float>((nodes.k_in[i] + nodes.k_out[i]));
		c_max = c_k * (c_k - 1.0) / 2.0;

		//(1) Consider both neighbors in the past
		if (edges.past_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_in[i]; j++)
				//3 < 2 < 1
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.past_edges[edges.past_edge_row_start[i]+j]))
						c_i += 1.0;

		//(2) Consider both neighbors in the future
		if (edges.future_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_out[i]; j++)
				//1 < 3 < 2
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.future_edges[edges.future_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0;

		//(3) Consider one neighbor in the past and one in the future
		if (edges.past_edge_row_start[i] != -1 && edges.future_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_out[i]; j++)
				for (k = 0; k < nodes.k_in[i]; k++)
					//3 < 1 < 2
					if (nodesAreConnected(nodes, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, edges.past_edges[edges.past_edge_row_start[i]+k], edges.future_edges[edges.future_edge_row_start[i]+j]))
						c_i += 1.0;

		if (DEBUG) assert (c_max > 0.0);
		c_i = c_i / c_max;
		if (DEBUG) assert (c_i <= 1.0);

		clustering[i] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
		//fflush(stdout);
	}

	average_clustering = c_avg / N_deg2;
	if (DEBUG) assert (average_clustering >= 0.0 && average_clustering <= 1.0);

	stopwatchStop(&sMeasureClustering);

	if (!bench) {
		printf("\tCalculated Clustering Coefficients.\n");
		printf_cyan();
		printf("\t\tAverage Clustering: %f\n", average_clustering);
		printf_std();
		fflush(stdout);
		if (calc_autocorr) {
			autocorr2 acClust(5);
			for (i = 0; i < N_tar; i++)
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
bool measureConnectedComponents(Node &nodes, const Edge &edges, const int &N_tar, int &N_cc, int &N_gcc, Stopwatch &sMeasureConnectedComponents, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No Null Pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);

		//Parameters in Correct Ranges
		assert (N_tar > 0);
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
		printMemUsed("to Measure Components", hostMemUsed, devMemUsed);

	for (i = 0; i < N_tar; i++) {
		elements = 0;
		if (!nodes.cc_id[i] && (nodes.k_in[i] + nodes.k_out[i]) > 0) {
			//printf("BEFORE\n");
			bfsearch(nodes, edges, i, ++N_cc, elements);
			//printf("AFTER\n");
		}
		if (elements > N_gcc)
			N_gcc = elements;
	}

	stopwatchStop(&sMeasureConnectedComponents);

	if (DEBUG) {
		assert (N_cc > 0);
		assert (N_gcc > 1);
	}

	if (!bench) {
		printf("\tCalculated Number of Connected Components.\n");
		printf_cyan();
		printf("\t\tIdentified %d Components.\n", N_cc);
		printf("\t\tSize of Giant Component: %d\n", N_gcc);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureConnectedComponents.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Calculates the Success Ratio using N_sr Unique Pairs of Nodes
//O(xxx) Efficiency (revise this)
bool measureSuccessRatio(const Node &nodes, const Edge &edges, const bool * const core_edge_exists, float &success_ratio, const int &N_tar, const double &N_sr, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const float &core_edge_fraction, Stopwatch &sMeasureSuccessRatio, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No Null Pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Parameters in Correct Ranges
		assert (N_tar > 0);
		assert (N_sr > 0 && N_sr <= ((uint64_t)N_tar * (N_tar - 1)) >> 1);
		assert (!(dim == 1 && manifold == DE_SITTER));
		if (manifold == DE_SITTER && universe)
			assert (alpha > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	}

	float dist, min_dist;
	int loc, next;
	int idx_a, idx_b;
	int i, j, m;
	int do_map;
	int n_trav;

	uint64_t stride = (uint64_t)N_tar * (N_tar - 1) / (static_cast<uint64_t>(N_sr) << 1);
	uint64_t vec_idx;
	uint64_t k;

	bool *used;

	stopwatchStart(&sMeasureSuccessRatio);

	try {
		used = (bool*)malloc(sizeof(bool) * N_tar);
		if (used == NULL)
			throw std::bad_alloc();
		memset(used, 0, sizeof(bool) * N_tar);
		hostMemUsed += sizeof(bool) * N_tar;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Success Ratio", hostMemUsed, devMemUsed);

	n_trav = 0;
	for (k = 0; k < static_cast<uint64_t>(N_sr); k++) {
	//for (k = 0; k < 1; k++) {
		//Pick Unique Pair
		vec_idx = k * stride + 1;
		i = static_cast<int>(vec_idx / (N_tar - 1));
		j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		if (!(nodes.k_in[i] + nodes.k_out[i]) || !(nodes.k_in[j] + nodes.k_out[j]))
			continue;
		if (nodes.cc_id[i] != nodes.cc_id[j])
			continue;

		//printf("%d -> %d:\n", nodes.id.AS[i], nodes.id.AS[j]);
		//printf("---------\n");

		//printf("%d -> ", nodes.id.AS[i]);

		//Reset boolean array
		memset(used, 0, sizeof(bool) * N_tar);

		//Begin Traversal from i to j
		loc = i;
		idx_b = j;
		while (loc != j) {
			idx_a = loc;
			min_dist = INF;
			used[loc] = true;
			next = loc;

			//Check Past Connections
			if (edges.past_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes.k_in[loc]; m++) {
					idx_a = edges.past_edges[edges.past_edge_row_start[loc]+m];
					//printf("\tPast Candidate: %d\n", nodes.id.AS[idx_a]);
					if (idx_a == idx_b) {
						loc = idx_b;
						goto PathSuccess;
					}
					if (manifold == DE_SITTER)
						dist = distanceDS(NULL, nodes.c.sc[idx_a], nodes.id.tau[idx_a], nodes.c.sc[idx_b], nodes.id.tau[idx_b], dim, manifold, a, alpha, universe);
					else if (manifold == HYPERBOLIC)
						dist = distanceH(nodes.c.hc[idx_a], nodes.c.hc[idx_b], dim, manifold, zeta);
					//printf("\t\tDistance: %f\n", dist);
					if (dist <= min_dist) {
						min_dist = dist;
						next = edges.past_edges[edges.past_edge_row_start[loc]+m];
					}
				}
			}

			//Check Future Connections
			if (edges.future_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes.k_out[loc]; m++) {
					idx_a = edges.future_edges[edges.future_edge_row_start[loc]+m];
					//printf("\tFuture Candidate: %d\n", nodes.id.AS[idx_a]);
					if (idx_a == idx_b) {
						loc = idx_b;
						goto PathSuccess;
					}
					if (manifold == DE_SITTER)
						dist = distanceDS(NULL, nodes.c.sc[idx_a], nodes.id.tau[idx_a], nodes.c.sc[idx_b], nodes.id.tau[idx_b], dim, manifold, a, alpha, universe);
					else if (manifold == HYPERBOLIC)
						dist = distanceH(nodes.c.hc[idx_a], nodes.c.hc[idx_b], dim, manifold, zeta);
					//printf("\t\tDistance: %f\n", dist);
					if (dist <= min_dist) {
						min_dist = dist;
						next = edges.future_edges[edges.future_edge_row_start[loc]+m];
					}
				}
			}

			//printf("Moving To: %d\n", nodes.id.AS[next]);
		
			if (!used[next])
				loc = next;
			else
				break;

			//printf("%d -> ", nodes.id.AS[next]);
		}

		PathSuccess:
		if (loc == j) //{
			success_ratio += 1.0;
			//printf("%d (S)\n", nodes.id.AS[loc]);
		//} else
		//	printf("(F)\n");
		//printf("\n");
		n_trav++;
	}

	if (n_trav)
		success_ratio /= n_trav;

	free(used);
	used = NULL;
	hostMemUsed -= sizeof(bool) * N_tar;

	stopwatchStop(&sMeasureSuccessRatio);

	if (!bench) {
		printf("\tCalculated Success Ratio.\n");
		printf_cyan();
		printf("\t\tSuccess Ratio: %f\n", success_ratio);
		printf("\t\tTraversed Pairs: %d\n", n_trav);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureSuccessRatio.elapsedTime);
		fflush(stdout);
	}

	return true;
}
