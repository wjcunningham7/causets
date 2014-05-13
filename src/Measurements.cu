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
		printMemUsed("Memory Necessary to Measure Clustering", hostMemUsed, devMemUsed);

	//i represents the node we are calculating the clustering coefficient for (node #1 in triplet)
	//j represents the second node in the triplet
	//k represents the third node in the triplet
	//j and k are not interchanging or else the number of triangles would be doubly counted

	for (i = 0; i < N_tar; i++) {
		//printf("\nNode %d:\n", i);
		//printf("\tDegrees: %d\n", (nodes[i].k_in + nodes[i].k_out));
		//printf("\t\tIn-Degrees: %d\n", nodes[i].k_in);
		//printf("\t\tOut-Degrees: %d\n", nodes[i].k_out);

		//Ingore nodes of degree 0 and 1
		if (nodes[i].k_in + nodes[i].k_out < 2)
			continue;

		c_i = 0.0;
		c_k = (float)(nodes[i].k_in + nodes[i].k_out);
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

		assert (c_max > 0.0);
		c_i = c_i / c_max;
		assert (c_i <= 1.0);

		clustering[cls_idx] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
	}

	average_clustering = c_avg / N_deg2;
	assert (average_clustering >= 0.0 && average_clustering <= 1.0);

	//Print average clustering to file
	/*std::ofstream cls;
	cls.open("clustering.txt", std::ios::app);
	cls << average_clustering << std::endl;
	cls.flush();
	cls.close();*/

	stopwatchStop(&sMeasureClustering);

	if (!bench) {
		printf("\tCalculated Clustering Coefficients.\n");
		printf("\t\tAverage Clustering:\t%f\n", average_clustering);
		if (calc_autocorr) {
			autocorr2 acClust(5);
			for (i = 0; i < N_deg2; i++)
				acClust.accum_data(clustering[i]);
			acClust.analysis();
			std::ofstream fout("clustAutoCorr.dat");
			acClust.fout_txt(fout);
			fout.close();
			printf("\t\tCalculated Autocorrelation.\n");
		}
	}
	if (verbose)
		printf("\t\tExecution Time: %5.9f sec\n", sMeasureClustering.elapsedTime);

	return true;
}

//Returns true if two nodes are causally connected
//Note: i past_idx must be less than future_idx
//O(1) Efficiency for Adjacency Matrix
//O(k) Efficiency for Adjacency List
bool nodesAreConnected(const Node * const nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx)
{
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

	int core_limit = (int)(core_edge_fraction * N_tar);
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
