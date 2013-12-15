#ifndef NETWORK_CREATOR_CU
#define NETWORK_CREATOR_CU

#include "Subroutines.cu"
#include "GPUSubroutines.cu"

//Primary Causet Subroutines
bool createNetwork(Network *network);
bool generateNodes(Network *network, bool &use_gpu);
bool linkNodes(Network *network, bool &use_gpu);

//Allocates memory for network
bool createNetwork(Network *network)
{
	//Implement subnet_size later
	//Chanage sizeof(Node) to reflect number of dimensions necessary
	//	so that for 1+1 phi and chi are not allocated
	try {
		network->nodes = (Node*)malloc(sizeof(Node) * network->network_properties.N_tar);
		if (network->nodes == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N_tar;
		//printMemUsed("Memory Allocated for Nodes", hostMemUsed, devMemUsed);

		network->past_edges = (unsigned int*)malloc(sizeof(unsigned int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer));
		if (network->past_edges == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(unsigned int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

		network->future_edges = (unsigned int*)malloc(sizeof(unsigned int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer));
		if (network->future_edges == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(unsigned int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

		network->past_edge_row_start = (int*)malloc(sizeof(int) * network->network_properties.N_tar);
		if (network->past_edge_row_start == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(int) * network->network_properties.N_tar;

		network->future_edge_row_start = (int*)malloc(sizeof(int) * network->network_properties.N_tar);
		if (network->future_edge_row_start == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(int) * network->network_properties.N_tar;

		network->core_edge_exists = (bool*)malloc(sizeof(bool) * powf(network->network_properties.core_edge_ratio * network->network_properties.N_tar, 2.0));
		if (network->core_edge_exists == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * powf(network->network_properties.core_edge_ratio * network->network_properties.N_tar, 2.0);

		//Allocate memory on GPU if necessary
		if (network->network_properties.flags.use_gpu) {
			checkCudaErrors(cuMemAlloc(&network->d_nodes, sizeof(Node) * network->network_properties.N_tar));
			devMemUsed += sizeof(Node) * network->network_properties.N_tar;

			checkCudaErrors(cuMemAlloc(&network->d_edges, sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2));
			devMemUsed += sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2;
		}

		memoryCheckpoint();
		//printMemUsed("Total Memory Allocated for Network", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure!\n");
		return false;
	}

	printf("\tMemory Successfully Allocated.\n");
	return true;
}

//Poisson Sprinkling
bool generateNodes(Network *network, bool &use_gpu)
{
	if (use_gpu) {
		if (!generateNodesGPU(network))
			return false;
	} else {
		//Initialize Newton-Raphson Parameters
		NewtonProperties np = NewtonProperties(network->network_properties.zeta, TOL, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim);

		//Generate coordinates for each of N nodes
		for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
			network->nodes[i] = Node();

			///////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~Theta~~~~~~~~~~~~~~~~~~~~~~~~~//
			//Sample Theta from (0, 2pi), as described on p. 2 of [1]//
			///////////////////////////////////////////////////////////

			network->nodes[i].theta = 2.0 * M_PI * ran2(&network->network_properties.seed);
			//printf("Theta: %5.5f\n", network->nodes[i].theta);

			if (network->network_properties.dim == 2)
				//CDF derived from PDF identified in (2) of [2]
				network->nodes[i].tau = etaToTau(atan(ran2(&network->network_properties.seed) / tan(network->network_properties.zeta)), network->network_properties.a);
			else if (network->network_properties.dim == 4) {
				/////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~~~~~~Tau~~~~~~~~~~~~~~~~~~~~//
				//CDF derived from PDF identified in (6) of [2]//
				/////////////////////////////////////////////////

				np.rval = ran2(&network->network_properties.seed);
				np.x = etaToTau((M_PI / 2.0) - np.zeta, network->network_properties.a);
				np.a = network->network_properties.a;	//Scaling from 'tau' to 't'
				np.max = 1000;	//Max number of Netwon-Raphson iterations

				newton(&solveTau, &np, &network->network_properties.seed);
				network->nodes[i].tau = np.x;
				
				////////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~Phi and Chi~~~~~~~~~~~~~~~~~~~~//	
				//CDFs derived from PDFs identified on p. 3 of [2]//
				////////////////////////////////////////////////////

				//Sample Phi from (0, pi)
				np.x = M_PI / 2.0;
				np.max = 100;

				newton(&solvePhi, &np, &network->network_properties.seed);
				network->nodes[i].phi = np.x;
				//printf("Phi: %5.5f\n", network->nodes[i].phi);

				//Sample Chi from (0, pi)
				network->nodes[i].chi = acosf(1.0 - (2.0 * ran2(&network->network_properties.seed)));
				//printf("Chi: %5.5f\n", network->nodes[i].chi);
			}
			//printf("Tau: %E\n", network->nodes[i].tau);
		}
	}

	printf("\tNodes Successfully Generated.\n");
	return true;
}

//Identify Causal Sets
bool linkNodes(Network *network, bool &use_gpu)
{
	float dt, dx;
	unsigned int core_limit = (unsigned int)(network->network_properties.core_edge_ratio * network->network_properties.N_tar);
	unsigned int future_idx = 0;
	unsigned int past_idx = 0;

	//Identify future connections
	for (unsigned int i = 0; i < network->network_properties.N_tar - 1; i++) {
		if (i < core_limit)
			network->core_edge_exists[(i*core_limit)+i] = false;
		network->future_edge_row_start[i] = future_idx;

		for (unsigned int j = i + 1; j < network->network_properties.N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			dt = tauToEta(network->nodes[j].tau, network->network_properties.a) - tauToEta(network->nodes[i].tau, network->network_properties.a);
			//printf("dt: %5.8f\n", dt);

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (network->network_properties.dim == 2)
				//Formula given on p. 2 of [2]
				dx = M_PI - fabs(M_PI - fabs(network->nodes[j].theta - network->nodes[i].theta));
			else if (network->network_properties.dim == 4)
				//Spherical Law of Cosines
				dx = acosf((X1(network->nodes[i].phi) * X1(network->nodes[j].phi)) + 
					    (X2(network->nodes[i].phi, network->nodes[i].chi) * X2(network->nodes[j].phi, network->nodes[j].chi)) + 
					    (X3(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X3(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta)) + 
					    (X4(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X4(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta)));
			//printf("dx: %5.8f\n", dx);

			//Core Edge Adjacency Matrix
			if (i < core_limit && j < core_limit) {
				if (dx > dt) {
					network->core_edge_exists[(i * core_limit) + j] = false;
					network->core_edge_exists[(j * core_limit) + i] = false;
				} else {
					network->core_edge_exists[(i * core_limit) + j] = true;
					network->core_edge_exists[(j * core_limit) + i] = true;
				}
			}
						
			//Ignore spacelike (non-causal) relations
			if (dx > dt) continue;
			
			network->future_edges[future_idx] = j;
			future_idx++;

			if (future_idx == (network->network_properties.N_tar * network->network_properties.k_tar / 2) + network->network_properties.edge_buffer)
				throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer.\n");

			//Record number of degrees for each node
			network->nodes[j].num_in++;
			network->nodes[i].num_out++;
		}

		//If there are no forward connections from node i, mark with -1
		if (network->future_edge_row_start[i] == future_idx)
			network->future_edge_row_start[i] = -1;
	}
	printf("\t\tForward Edges: %u\n", future_idx);

	/*std::ofstream deg;
	deg.open("degrees.txt", std::ios::app);
	deg << future_idx << std::endl;
	deg.flush();
	deg.close();*/

	//Identify past connections
	for (unsigned int i = 1; i < network->network_properties.N_tar; i++) {
		network->past_edge_row_start[i] = past_idx;
		for (unsigned int j = 0; j < i; j++) {
			if (network->future_edge_row_start[j] == -1)
				continue;

			for (unsigned int k = 0; k < network->nodes[j].num_out; k++) {
				if (i == network->future_edges[network->future_edge_row_start[j]+k]) {
					network->past_edges[past_idx] = j;
					past_idx++;
				}
			}
		}

		//If there are no backward connections from node i, mark with -1
		if (network->past_edge_row_start[i] == past_idx)
			network->past_edge_row_start[i] = -1;
	}
	//The quantities future_idx and past_idx should be equal
	//printf("\t\tBackward Edges: %u\n", past_idx);

	//Identify Resulting Network
	for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
		if (network->nodes[i].num_in + network->nodes[i].num_out > 0) {
			network->network_properties.N_res++;
			network->network_properties.k_res += network->nodes[i].num_in + network->nodes[i].num_out;

			if (network->nodes[i].num_in + network->nodes[i].num_out > 1)
				network->network_properties.N_deg2++;
		}
	}
	network->network_properties.k_res /= network->network_properties.N_res;

	//Debug:  Future vs Past Edges in Adjacency List
	/*for (unsigned int i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Forward Connections:\n");
		if (network->future_edge_row_start[i] == -1)
			printf("\tNo future connections.\n");
		else {
			for (unsigned int j = 0; j < network->nodes[i].num_out && j < 10; j++)
				printf("%d ", network->future_edges[network->future_edge_row_start[i]+j]);
			printf("\n");
		}

		printf("Backward Connections:\n");
		if (network->past_edge_row_start[i] == -1)
			printf("\tNo past connections.\n");
		else {
			for (unsigned int j = 0; j < network->nodes[i].num_in && j < 10; j++)
				printf("%d ", network->past_edges[network->past_edge_row_start[i]+j]);
			printf("\n");
		}
	}*/

	//Debug:  Future and Past Adjacency List Indices
	/*printf("\nFuture Edge Indices:\n");
	for (unsigned int i = 0; i < 20; i++)
		printf("%d\n", network->future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (unsigned int i = 0; i < 20; i++)
		printf("%d\n", network->past_edge_row_start[i]);

	unsigned int next_future_idx, next_past_idx;
	for (unsigned int i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %u\n", network->nodes[i].num_out);
		if (network->future_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (unsigned int j = 1; j < 100; j++) {
				if (network->future_edge_row_start[i+j] != -1) {
					next_future_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (network->future_edge_row_start[i+next_future_idx] - network->future_edge_row_start[i]));
		}

		printf("In-Degrees: %u\n", network->nodes[i].num_in);
		if (network->past_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (unsigned int j = 1; j < 100; j++) {
				if (network->past_edge_row_start[i+j] != -1) {
					next_past_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (network->past_edge_row_start[i+next_past_idx] - network->past_edge_row_start[i]));
		}
	}*/

	printf("\tCausets Successfully Connected.\n");
	printf("\t\tResulting Network Size: %u\n", network->network_properties.N_res);
	printf("\t\tResulting Average Degree: %f\n", network->network_properties.k_res);
	return true;
}

#endif