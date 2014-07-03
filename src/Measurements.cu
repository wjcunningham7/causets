#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
bool measureClustering(float *& clustering, const Node &nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, Stopwatch &sMeasureClustering, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &calc_autocorr, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No null pointers
		assert (past_edges != NULL);
		assert (future_edges != NULL);
		assert (past_edge_row_start != NULL);
		assert (future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Parameters in correct ranges
		assert (N_tar > 0);
		assert (N_deg2 > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	}

	float c_i, c_k, c_max;
	float c_avg = 0.0;
	int cls_idx = 0;
	int i, j, k;

	stopwatchStart(&sMeasureClustering);

	try {
		clustering = (float*)malloc(sizeof(float) * N_deg2);
		if (clustering == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(float) * N_deg2;
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
		if (nodes.k_in[i] + nodes.k_out[i] < 2)
			continue;

		c_i = 0.0;
		c_k = static_cast<float>((nodes.k_in[i] + nodes.k_out[i]));
		c_max = c_k * (c_k - 1.0) / 2.0;

		//(1) Consider both neighbors in the past
		if (past_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_in[i]; j++)
				//3 < 2 < 1
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, future_edges, future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, past_edges[past_edge_row_start[i]+k], past_edges[past_edge_row_start[i]+j]))
						c_i += 1.0;

		//(2) Consider both neighbors in the future
		if (future_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_out[i]; j++)
				//1 < 3 < 2
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, future_edges, future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, future_edges[future_edge_row_start[i]+k], future_edges[future_edge_row_start[i]+j]))
						c_i += 1.0;

		//(3) Consider one neighbor in the past and one in the future
		if (past_edge_row_start[i] != -1 && future_edge_row_start[i] != -1)
			for (j = 0; j < nodes.k_out[i]; j++)
				for (k = 0; k < nodes.k_in[i]; k++)
					//3 < 1 < 2
					if (nodesAreConnected(nodes, future_edges, future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, past_edges[past_edge_row_start[i]+k], future_edges[future_edge_row_start[i]+j]))
						c_i += 1.0;

		if (DEBUG) assert (c_max > 0.0);
		c_i = c_i / c_max;
		if (DEBUG) assert (c_i <= 1.0);

		clustering[cls_idx] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
		//fflush(stdout);
	}

	average_clustering = c_avg / N_deg2;
	if (DEBUG) assert (average_clustering >= 0.0 && average_clustering <= 1.0);

	//Print average clustering to file
	/*std::ofstream cls;
	cls.open("clustering.txt", std::ios::app);
	cls << average_clustering << std::endl;
	cls.flush();
	cls.close();*/

	stopwatchStop(&sMeasureClustering);

	if (!bench) {
		printf("\tCalculated Clustering Coefficients.\n");
		printf("\t\tAverage Clustering: %f\n", average_clustering);
		fflush(stdout);
		if (calc_autocorr) {
			autocorr2 acClust(5);
			for (i = 0; i < N_deg2; i++)
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

//Calculates the Success Ratio using N_sr Unique Pairs of Nodes
//O(xxx) Efficiency (revise this)
bool measureSuccessRatio(const Node &nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &success_ratio, const int &N_tar, const int64_t &N_sr, const double &a, const double &alpha, const float &core_edge_fraction, Stopwatch &sMeasureSuccessRatio, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No Null Pointers
		assert (past_edges != NULL);
		assert (future_edges != NULL);
		assert (past_edge_row_start != NULL);
		assert (future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Parameters in Correct Ranges
		assert (N_tar > 0);
		assert (N_sr > 0 && N_sr <= (uint64_t)N_tar * (N_tar - 1) / 2);
		assert (alpha > 0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	}

	float dist, min_dist;
	int loc, next;
	int idx_a, idx_b;
	int i, j, k, m;
	int do_map;
	int n_trav;

	uint64_t stride = (uint64_t)N_tar * (N_tar - 1) / (2 * N_sr);
	uint64_t vec_idx;

	bool *used;

	stopwatchStart(&sMeasureSuccessRatio);

	try {
		used = (bool*)malloc(sizeof(bool) * N_tar);
		if (used == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * N_tar;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}
	
	memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
	if (verbose)
		printMemUsed("to Measure Success Ratio", hostMemUsed, devMemUsed);

	n_trav = 0;
	for (k = 0; k < N_sr; k++) {
		//Pick Unique Pair
		vec_idx = k * stride;
		i = (int)(vec_idx / N_tar);
		j = (int)(vec_idx % N_tar);
		do_map = i >= j;

		if (j == 0)
			continue;

		if (j < N_tar / 2) {
			i = i + do_map * (2 * (N_tar / 2 - i) - 1);
			j = j + do_map * (2 * (N_tar / 2 - j));
		} else
			j += N_tar / 2;

		if (nodes.k_in[i] + nodes.k_out[i] == 0 || nodes.k_in[j] + nodes.k_out[j] == 0)
			continue;

		//Reset boolean array
		memset(used, 0, sizeof(bool) * N_tar);

		//Begin Traversal from i to j
		loc = i;
		idx_b = j;
		while (loc != j) {
			idx_a = loc;
			min_dist = distance(nodes.sc[idx_a], nodes.tau[idx_a], nodes.sc[idx_b], nodes.tau[idx_b], a, alpha);
			used[loc] = true;
			next = loc;

			//Check Past Connections
			if (past_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes.k_in[loc]; m++) {
					idx_a = past_edges[past_edge_row_start[loc]+m];
					dist = distance(nodes.sc[idx_a], nodes.tau[idx_a], nodes.sc[idx_b], nodes.tau[idx_b], a, alpha);
					if (dist <= min_dist) {
						min_dist = dist;
						next = past_edges[past_edge_row_start[loc]+m];
					}
				}
			}

			//Check Future Connections
			if (future_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes.k_out[loc]; m++) {
					idx_a = future_edges[future_edge_row_start[loc]+m];
					dist = distance(nodes.sc[idx_a], nodes.tau[idx_a], nodes.sc[idx_b], nodes.tau[idx_b], a, alpha);
					if (dist <= min_dist) {
						min_dist = dist;
						next = future_edges[future_edge_row_start[loc]+m];
					}
				}
			}
			
			if (!used[next])
				loc = next;
			else
				break;
		}

		if (loc == j)
			success_ratio += 1.0;
		n_trav++;
	}

	success_ratio /= n_trav;

	free(used);
	used = NULL;
	hostMemUsed -= sizeof(bool) * N_tar;

	stopwatchStop(&sMeasureSuccessRatio);

	if (!bench) {
		printf("\tCalculated Success Ratio.\n");
		printf("\t\tSuccess Ratio: %f\n", success_ratio);
		printf("\t\tTraversed Pairs: %d\n", n_trav);
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sMeasureSuccessRatio.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Returns true if two nodes are causally connected
//Note: past_idx must be less than future_idx
//O(1) Efficiency for Adjacency Matrix
//O(k) Efficiency for Adjacency List
static bool nodesAreConnected(const Node &nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx)
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

//Returns the distance between two nodes
//O(xxx) Efficiency (revise this)
static float distance(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const double &a, const double &alpha)
{
	if (DEBUG) {
		//Parameters in Correct Ranges
		assert (tau_a != tau_b);
		assert (a > 0.0);
		assert (alpha > 0.0);
	}
	
	IntData idata = IntData();
	idata.tol = 1e-5;

	GSL_EmbeddedZ1_Parameters p;
	p.a = a;
	p.alpha = alpha;

	float z0_a, z0_b;
	float z1_a, z1_b;
	float dt2, dx2;
	float power = 2.0f / 3.0f;

	//Solve for z1 in Rotated Plane
	z1_a = alpha * POW(SINH(1.5f * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
	z1_b = alpha * POW(SINH(1.5f * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

	//Use Numerical Integration for z0
	idata.upper = z1_a;
	z0_a = integrate1D(&embeddedZ1, (void*)&p, &idata, QNG);
	idata.upper = z1_b;
	z0_b = integrate1D(&embeddedZ1, (void*)&p, &idata, QNG);
	
	//Solve for Temporal Portion of Invariant Interval
	dt2 = z0_a * z0_b;

	//Rotate Into z2, z3, z4 Planes
	dx2 = z1_a * z1_b * (X1(node_a.y) * X1(node_b.y) +
			     X2(node_a.y, node_a.z) * X2(node_b.y, node_b.y) +
			     X3(node_a.y, node_a.z, node_a.x) * X3(node_b.y, node_b.z, node_b.x) +
			     X4(node_a.y, node_a.z, node_a.x) * X4(node_b.y, node_b.z, node_b.x));

	//If dx2 - dt2 < 0, the interval is time-like
	//If dx2 - dt2 > 0, the interval is space-like (causal)

	if (DEBUG) {
		//This is not correct, revise this
		if (tau_a < tau_b)
			assert (dx2 - dt2 > 0);
		else
			assert (dx2 - dt2 < 0);
	}

	return SQRT(dx2 - dt2, STL);
}
