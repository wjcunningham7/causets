#ifndef MEASUREMENTS_CU_
#define MEASUREMENTS_CU_

#include "autocorr2.cu"

void measureClustering(Network *network, CausetPerformance *cp);
bool nodesAreConnected(Network *network, int past_idx, int future_idx);

//Calculates clustering coefficient for each node in network
//O(N*k^3) Efficiency
void measureClustering(Network *network, CausetPerformance *cp)
{
	assert (network->network_properties.N_tar > 0);
	assert (network->network_properties.N_deg2 > 0);
	assert (network->nodes != NULL);
	assert (network->past_edges != NULL);
	assert (network->future_edges != NULL);
	assert (network->past_edge_row_start != NULL);
	assert (network->future_edge_row_start != NULL);

	float c_i, c_k, c_max;
	float c_avg = 0.0;
	int cls_idx = 0;
	int i, j, k;

	stopwatchStart(&cp->sMeasureClustering);

	try {
		network->network_observables.clustering = (float*)malloc(sizeof(float) * network->network_properties.N_deg2);
		if (network->network_observables.clustering == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(float) * network->network_properties.N_deg2;
	} catch (std::bad_alloc()) {
		fprintf(stderr, "Failed to allocate memory in %s on line %d!\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	memoryCheckpoint();
	if (CAUSET_DEBUG)
		printMemUsed("Memory Necessary to Measure Clustering", hostMemUsed, devMemUsed);

	//i represents the node we are calculating the clustering coefficient for (node #1 in triplet)
	//j represents the second node in the triplet
	//k represents the third node in the triplet
	//j and k are not interchanging or else the number of triangles would be doubly counted

	for (i = 0; i < network->network_properties.N_tar; i++) {
		//printf("\nNode %d:\n", i);
		//printf("\tDegrees: %d\n", (network->nodes[i].k_in + network->nodes[i].k_out));
		//printf("\t\tIn-Degrees: %d\n", network->nodes[i].k_in);
		//printf("\t\tOut-Degrees: %d\n", network->nodes[i].k_out);

		//Ingore nodes of degree 0 and 1
		if (network->nodes[i].k_in + network->nodes[i].k_out < 2)
			continue;

		c_i = 0.0;
		c_k = (float)(network->nodes[i].k_in + network->nodes[i].k_out);
		c_max = c_k * (c_k - 1.0) / 2.0;

		//(1) Consider both neighbors in the past
		if (network->past_edge_row_start[i] != -1)
			for (j = 0; j < network->nodes[i].k_in; j++)
				//3 < 2 < 1
				for (k = 0; k < j; k++)
					if (nodesAreConnected(network, network->past_edges[network->past_edge_row_start[i] + k], network->past_edges[network->past_edge_row_start[i] + j]))
						c_i += 1.0;

		//(2) Consider both neighbors in the future
		if (network->future_edge_row_start[i] != -1)
			for (j = 0; j < network->nodes[i].k_out; j++)
				//1 < 3 < 2
				for (k = 0; k < j; k++)
					if (nodesAreConnected(network, network->future_edges[network->future_edge_row_start[i] + k], network->future_edges[network->future_edge_row_start[i] + j]))
						c_i += 1.0;

		//(3) Consider one neighbor in the past and one in the future
		if (network->past_edge_row_start[i] != -1 && network->future_edge_row_start[i] != -1)
			for (j = 0; j < network->nodes[i].k_out; j++)
				for (k = 0; k < network->nodes[i].k_in; k++)
					//3 < 1 < 2
					if (nodesAreConnected(network, network->past_edges[network->past_edge_row_start[i] + k], network->future_edges[network->future_edge_row_start[i] + j]))
						c_i += 1.0;

		assert (c_max > 0.0);
		c_i = c_i / c_max;
		assert (c_i <= 1.0);

		network->network_observables.clustering[cls_idx] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n\n", c_i);
	}

	network->network_observables.average_clustering = c_avg / network->network_properties.N_deg2;
	assert (network->network_observables.average_clustering >= 0.0 && network->network_observables.average_clustering <= 1.0);

	//Print average clustering to file
	/*std::ofstream cls;
	cls.open("clustering.txt", std::ios::app);
	cls << network->network_observables.average_clustering << std::endl;
	cls.flush();
	cls.close();*/

	stopwatchStop(&cp->sMeasureClustering);

	if (!BENCH) {
		printf("\tCalculated Clustering Coefficients.\n");
		printf("\t\tAverage Clustering:\t%f\n", network->network_observables.average_clustering);
		if (network->network_properties.flags.calc_autocorr) {
			autocorr2 acClust(5);
			for (i = 0; i < network->network_properties.N_deg2; i++)
				acClust.accum_data(network->network_observables.clustering[i]);
			acClust.analysis();
			std::ofstream fout("clustAutoCorr.dat");
			acClust.fout_txt(fout);
			fout.close();
			printf("\t\tCalculated Autocorrelation.\n");
		}
	}
	if (CAUSET_DEBUG)
		printf("\t\tExecution Time: %5.9f sec\n", cp->sMeasureClustering.elapsedTime);
}

//Returns true if two nodes are causally connected
//Note: i past_idx must be less than future_idx
//O(1) Efficiency for Adjacency Matrix
//O(k) Efficiency for Adjacency List
bool nodesAreConnected(Network *network, int past_idx, int future_idx)
{
	assert (past_idx < future_idx);
	assert (past_idx >= 0 && past_idx < network->network_properties.N_tar);
	assert (future_idx >= 0 && future_idx < network->network_properties.N_tar);
	assert (network->network_properties.core_edge_fraction > 0.0 && network->network_properties.core_edge_fraction < 1.0);
	assert (network->nodes != NULL);
	assert (network->future_edges != NULL);
	assert (network->future_edge_row_start != NULL);
	assert (network->core_edge_exists != NULL);

	//Check if the adjacency matrix can be used
	int core_limit = (int)(network->network_properties.core_edge_fraction * network->network_properties.N_tar);
	int i;
	if (past_idx < core_limit && future_idx < core_limit)
		return (network->core_edge_exists[(past_idx * core_limit) + future_idx]);
	else if (network->future_edge_row_start[past_idx] == -1)
		return false;
	else
		for (i = 0; i < network->nodes[past_idx].k_out; i++)
			if (network->future_edges[network->future_edge_row_start[past_idx] + i] == future_idx)
				return true;

	return false;
}

#endif
