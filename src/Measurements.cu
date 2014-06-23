#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
bool measureClustering(float *& clustering, const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, Stopwatch &sMeasureClustering, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &calc_autocorr, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No null pointers
		assert (nodes != NULL);
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
		//printf("\tDegrees: %d\n", (nodes[i].k_in + nodes[i].k_out));
		//printf("\t\tIn-Degrees: %d\n", nodes[i].k_in);
		//printf("\t\tOut-Degrees: %d\n", nodes[i].k_out);
		//fflush(stdout);

		//Ingore nodes of degree 0 and 1
		if (nodes[i].k_in + nodes[i].k_out < 2)
			continue;

		c_i = 0.0;
		c_k = static_cast<float>((nodes[i].k_in + nodes[i].k_out));
		c_max = c_k * (c_k - 1.0) / 2.0;

		//(1) Consider both neighbors in the past
		if (past_edge_row_start[i] != -1)
			for (j = 0; j < nodes[i].k_in; j++)
				//3 < 2 < 1
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, future_edges, future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, past_edges[past_edge_row_start[i]+k], past_edges[past_edge_row_start[i]+j]))
						c_i += 1.0;

		//(2) Consider both neighbors in the future
		if (future_edge_row_start[i] != -1)
			for (j = 0; j < nodes[i].k_out; j++)
				//1 < 3 < 2
				for (k = 0; k < j; k++)
					if (nodesAreConnected(nodes, future_edges, future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction, future_edges[future_edge_row_start[i]+k], future_edges[future_edge_row_start[i]+j]))
						c_i += 1.0;

		//(3) Consider one neighbor in the past and one in the future
		if (past_edge_row_start[i] != -1 && future_edge_row_start[i] != -1)
			for (j = 0; j < nodes[i].k_out; j++)
				for (k = 0; k < nodes[i].k_in; k++)
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
bool measureSuccessRatio(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &success_ratio, const int &N_tar, const int64_t &N_sr, const float &core_edge_fraction, Stopwatch &sMeasureSuccessRatio, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench)
{
	//Assert Statements

	float dist, min_dist;
	int loc, next;
	int i, j, k, m;
	int n_trav;

	uint64_t stride = (uint64_t)N_tar * (N_tar - 1) / (2 * N_sr);
	uint64_t vec_idx, mat_idx;

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
		mat_idx = vec2MatIdx(N_tar, vec_idx);

		i = (int)(mat_idx / N_tar);
		j = (int)(mat_idx % N_tar);

		if (nodes[i].k_in + nodes[i].k_out == 0 || nodes[j].k_in + nodes[j].k_out == 0)
			continue;

		//Reset boolean array
		memset(used, 0, sizeof(bool) * N_tar);

		//Begin Traversal from i to j
		loc = i;
		while (loc != j) {
			min_dist = distance(nodes[loc], nodes[j]);
			used[loc] = true;
			next = loc;

			//Check Past Connections
			if (past_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes[loc].k_in; m++) {
					dist = distance(nodes[past_edges[past_edge_row_start[loc]+m]], nodes[j]);
					if (dist <= min_dist) {
						min_dist = dist;
						next = past_edges[past_edge_row_start[loc]+m];
					}
				}
			}

			//Check Future Connections
			if (future_edge_row_start[loc] != -1) {
				for (m = 0; m < nodes[loc].k_out; m++) {
					dist = distance(nodes[future_edges[future_edge_row_start[loc]+m]], nodes[j]);
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
bool nodesAreConnected(const Node * const nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx)
{
	if (DEBUG) {
		//No null pointers
		assert (nodes != NULL);
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
		for (i = 0; i < nodes[past_idx].k_out; i++)
			if (future_edges[future_edge_row_start[past_idx] + i] == future_idx)
				return true;

	return false;
}

//Returns the distance between two nodes
//O(xxx) Efficiency (revise this)
float distance(const Node &node0, const Node &node1)
{
	return 0.0;
}
