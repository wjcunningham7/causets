#ifndef MEASUREMENTS_CU_
#define MEASUREMENTS_CU_

void measureClustering(Network *network);
bool nodesAreConnected(Network *network, unsigned int past_idx, unsigned int future_idx);

//Calculates clustering coefficient for each node in network
void measureClustering(Network *network)
{
	float c_i, c_k, c_max;
	float c_avg = 0.0;
	unsigned int cls_idx = 0;

	network->network_observables.clustering = (float*)malloc(sizeof(float) * network->network_properties.N_deg2);
	if (network->network_observables.clustering == NULL) throw std::bad_alloc();
	hostMemUsed += sizeof(float) * network->network_properties.N_deg2;

	//i represents the node we are calculating the clustering coefficient for (node #1 in triplet)
	//j represents the second node in the triplet
	//k represents the third node in the triplet
	//j and k are not interchanging or else the number of triangles would be doubly counted

	for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
		//printf("\nNode %u:\n", i);
		//printf("\tDegrees: %u\n", (network->nodes[i].num_in + network->nodes[i].num_out));
		//printf("\t\tIn-Degrees: %u\n", network->nodes[i].num_in);
		//printf("\t\tOut-Degrees: %u\n", network->nodes[i].num_out);

		//Ingore nodes of degree 0 and 1
		if (network->nodes[i].num_in + network->nodes[i].num_out < 2)
			continue;

		c_i = 0.0;
		c_k = (float)(network->nodes[i].num_in + network->nodes[i].num_out);
		c_max = c_k * (c_k - 1.0) / 2.0;

		//(1) Consider both neighbors in the past
		if (network->past_edge_row_start[i] != -1)
			for (unsigned int j = 0; j < network->nodes[i].num_in; j++)
				//3 < 2 < 1
				for (unsigned int k = 0; k < j; k++)
					if (nodesAreConnected(network, network->past_edges[network->past_edge_row_start[i] + k], network->past_edges[network->past_edge_row_start[i] + j]))
						c_i += 1.0;

		//(2) Consider both neighbors in the future
		if (network->future_edge_row_start[i] != -1)
			for (unsigned int j = 0; j < network->nodes[i].num_out; j++)
				//1 < 3 < 2
				for (unsigned int k = 0; k < j; k++)
					if (nodesAreConnected(network, network->future_edges[network->future_edge_row_start[i] + k], network->future_edges[network->future_edge_row_start[i] + j]))
						c_i += 1.0;

		//(3) Consider one neighbor in the past and one in the future
		if (network->past_edge_row_start[i] != -1 && network->future_edge_row_start[i] != -1)
			for (unsigned int j = 0; j < network->nodes[i].num_out; j++)
				for (unsigned int k = 0; k < network->nodes[i].num_in; k++)
					//3 < 1 < 2
					if (nodesAreConnected(network, network->past_edges[network->past_edge_row_start[i] + k], network->future_edges[network->future_edge_row_start[i] + j]))
						c_i += 1.0;

		if (c_max != 0.0)
			c_i = c_i / c_max;
		else if (c_i != 0.0)
			throw CausetException("Clustering algorithm failed!\n");

		network->network_observables.clustering[cls_idx] = c_i;
		c_avg += c_i;

		//printf("\tConnected Triplets: %f\n", (c_i * c_max));
		//printf("\tMaximum Triplets: %f\n", c_max);
		//printf("\tClustering Coefficient: %f\n", c_i);
	}

	network->network_observables.average_clustering = c_avg / network->network_properties.N_deg2;

	/*std::ofstream cls;
	cls.open("clustering.txt", std::ios::app);
	cls << network->network_observables.average_clustering << std::endl;
	cls.flush();
	cls.close();*/

	printf("\tCalculated Clustering Coefficients.\n");
	printf("\t\tAverage Clustering: %f\n", network->network_observables.average_clustering);
}

//Returns true if two nodes are causally connected
//Note:  past_idx must be less than future_idx
bool nodesAreConnected(Network *network, unsigned int past_idx, unsigned int future_idx)
{
	if (past_idx >= future_idx || past_idx >= network->network_properties.N_tar || future_idx >= network->network_properties.N_tar)
		return false;

	//Check if the adjacency matrix can be used
	unsigned int core_limit = (unsigned int)(network->network_properties.core_edge_ratio * network->network_properties.N_tar);
	if (past_idx < core_limit && future_idx < core_limit)
		return (network->core_edge_exists[(past_idx * core_limit) + future_idx]);
	else if (network->future_edge_row_start[past_idx] == -1)
		return false;
	else
		for (unsigned int i = 0; i < network->nodes[past_idx].num_out; i++)
			if (network->future_edges[network->future_edge_row_start[past_idx] + i] == future_idx)
				return true;

	return false;
}

#endif
