#ifndef NETWORK_CREATOR_CU_
#define NETWORK_CREATOR_CU_

#include "NetworkCreator.h"

//Allocates memory for network
//O(1) Efficiency
bool createNetwork(Network *network, CausetPerformance *cp, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	assert (network->network_properties.N_tar > 0);
	assert (network->network_properties.k_tar > 0.0);
	assert (network->network_properties.core_edge_fraction >= 0.0 && network->network_properties.core_edge_fraction <= 1.0);

	if (network->network_properties.flags.verbose) {
		//Estimate memory usage before allocating
		size_t mem = 0;
		mem += sizeof(Node) * network->network_properties.N_tar;
		mem += sizeof(int) * 2 * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);
		mem += sizeof(int) * 2 * network->network_properties.N_tar;
		mem += sizeof(bool) * powf(network->network_properties.core_edge_fraction * network->network_properties.N_tar, 2.0);

		size_t dmem = 0;
		if (network->network_properties.flags.use_gpu) {
			dmem += sizeof(Node) * network->network_properties.N_tar;
			dmem += sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2;
		}

		printMemUsed("for Network (Estimation)", mem, dmem);
		printf("\nContinue [y/N]?");
		char response = getchar();
		if (response != 'y')
			return false;
	}

	stopwatchStart(&cp->sCreateNetwork);

	try {
		network->nodes = (Node*)malloc(sizeof(Node) * network->network_properties.N_tar);
		if (network->nodes == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N_tar;
		if (network->network_properties.flags.verbose)
			printMemUsed("Memory Allocated for Nodes", hostMemUsed, devMemUsed);

		network->past_edges = (int*)malloc(sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer));
		if (network->past_edges == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

		network->future_edges = (int*)malloc(sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer));
		if (network->future_edges == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

		network->past_edge_row_start = (int*)malloc(sizeof(int) * network->network_properties.N_tar);
		if (network->past_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * network->network_properties.N_tar;

		network->future_edge_row_start = (int*)malloc(sizeof(int) * network->network_properties.N_tar);
		if (network->future_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * network->network_properties.N_tar;

		network->core_edge_exists = (bool*)malloc(sizeof(bool) * powf(network->network_properties.core_edge_fraction * network->network_properties.N_tar, 2.0));
		if (network->core_edge_exists == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * powf(network->network_properties.core_edge_fraction * network->network_properties.N_tar, 2.0);

		//Allocate memory on GPU if necessary
		if (network->network_properties.flags.use_gpu) {
			checkCudaErrors(cuMemAlloc(&network->d_nodes, sizeof(Node) * network->network_properties.N_tar));
			devMemUsed += sizeof(Node) * network->network_properties.N_tar;

			checkCudaErrors(cuMemAlloc(&network->d_edges, sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2));
			devMemUsed += sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2;
		}

		memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
		if (network->network_properties.flags.verbose)
			printMemUsed("Total Memory Allocated for Network", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	stopwatchStop(&cp->sCreateNetwork);

	if (!network->network_properties.flags.bench)
		printf("\tMemory Successfully Allocated.\n");
	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sCreateNetwork.elapsedTime);
	return true;
}

//Poisson Sprinkling
//O(N) Efficiency
bool generateNodes(Network *network, CausetPerformance *cp, bool &use_gpu)
{

	assert (network->network_properties.N_tar > 0);
	assert (network->network_properties.k_tar > 0.0);
	assert (network->network_properties.a > 0.0);
	assert (network->network_properties.dim == 1 || network->network_properties.dim == 3);
	if (network->network_properties.flags.universe)
		assert (network->network_properties.dim == 3);
	else
		assert (network->network_properties.zeta > 0.0 && network->network_properties.zeta < HALF_PI);	
	assert (network->network_properties.manifold == EUCLIDEAN || network->network_properties.manifold == DE_SITTER || network->network_properties.manifold == ANTI_DE_SITTER);
	assert (network->nodes != NULL);

	stopwatchStart(&cp->sGenerateNodes);

	if (use_gpu) {
		if (!generateNodesGPU(network))
			return false;
	} else {
		//Initialize Newton-Raphson Struct
		NewtonProperties np = NewtonProperties(network->network_properties.zeta, TOL, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim);

		//Generate coordinates for each of N nodes
		int i;
		for (i = 0; i < network->network_properties.N_tar; i++) {
			network->nodes[i] = Node();

			if (network->network_properties.manifold == EUCLIDEAN) {
				//
			} else if (network->network_properties.manifold == DE_SITTER) {
				///////////////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~~~~~~~~~Theta~~~~~~~~~~~~~~~~~~~~~~~~~//
				//Sample Theta from (0, 2pi), as described on p. 2 of [1]//
				///////////////////////////////////////////////////////////

				network->nodes[i].theta = TWO_PI * ran2(&network->network_properties.seed);
				assert (network->nodes[i].theta > 0.0 && network->nodes[i].theta < TWO_PI);
				//if (i % NPRINT == 0) printf("Theta: %5.5f\n", network->nodes[i].theta);
			} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
				//
			}

			if (network->network_properties.dim == 1) {
				if (network->network_properties.manifold == EUCLIDEAN) {
					//
				} else if (network->network_properties.manifold == DE_SITTER) {
					//CDF derived from PDF identified in (2) of [2]
					network->nodes[i].t = etaToT(atan(ran2(&network->network_properties.seed) / tan(network->network_properties.zeta)), network->network_properties.a);
					assert (network->nodes[i].t > 0.0);
				} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
					//
				}
			} else if (network->network_properties.dim == 3) {
				if (network->network_properties.manifold == EUCLIDEAN) {
					//
				} else if (network->network_properties.manifold == DE_SITTER) {
					/////////////////////////////////////////////////////////
					//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
					//CDF derived from PDF identified in (6) of [2] for 3+1//
					//and from PDF identified in (12) of [2] for universe  //
					/////////////////////////////////////////////////////////

					np.rval = ran2(&network->network_properties.seed);
					np.max = 1000;
					if (network->network_properties.flags.universe) {
						np.x = 0.5;
						np.tau0 = network->network_properties.tau0;
						if (!newton(&solveTau, &np, &network->network_properties.seed))
							return false;
					} else {
						np.x = etaToT(HALF_PI - np.zeta, network->network_properties.a);
						np.a = network->network_properties.a;
						if (!newton(&solveT, &np, &network->network_properties.seed))
							return false;
					}
					network->nodes[i].t = np.x;
					assert (network->nodes[i].t > 0.0);
				
					////////////////////////////////////////////////////
					//~~~~~~~~~~~~~~~~~Phi and Chi~~~~~~~~~~~~~~~~~~~~//	
					//CDFs derived from PDFs identified on p. 3 of [2]//
					//Phi given by [3]				  //
					////////////////////////////////////////////////////

					//Sample Phi from (0, pi)
					//For some reason the technique in [3] has not been producing the correct distribution...
					//network->nodes[i].phi = 0.5 * (M_PI * ran2(&network->network_properties.seed) + acos(ran2(&network->network_properties.seed)));
					np.rval = ran2(&network->network_properties.seed);
					np.x = HALF_PI;
					np.max = 250;

					if (!newton(&solvePhi, &np, &network->network_properties.seed))
						return false;
					network->nodes[i].phi = np.x;
					assert (network->nodes[i].phi > 0.0 && network->nodes[i].phi < M_PI);
					//if (i % NPRINT == 0) printf("Phi: %5.5f\n", network->nodes[i].phi);

					//Sample Chi from (0, pi)
					network->nodes[i].chi = acosf(1.0 - 2.0 * ran2(&network->network_properties.seed));
					assert (network->nodes[i].chi > 0.0 && network->nodes[i].chi < M_PI);
					//if (i % NPRINT == 0) printf("Chi: %5.5f\n", network->nodes[i].chi);
				} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
					//
				}
			}
			//if (i % NPRINT == 0) printf("T: %E\n", network->nodes[i].t);
		}

		//Debugging statements used to check coordinate distributions
		//printValues(network->nodes, network->network_properties.N_tar, "t_dist.cset.dbg.dat", "t");
		//printValues(network->nodes, network->network_properties.N_tar, "theta_dist.cset.dbg.dat", "theta");
		//printValues(network->nodes, network->network_properties.N_tar, "chi_dist.cset.dbg.dat", "chi");
		//printValues(network->nodes, network->network_properties.N_tar, "phi_dist.cset.dbg.dat", "phi");
	}

	stopwatchStop(&cp->sGenerateNodes);

	if (!network->network_properties.flags.bench)
		printf("\tNodes Successfully Generated.\n");
	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sGenerateNodes.elapsedTime);
	return true;
}

//Identify Causal Sets
//O(k*N^2) Efficiency
bool linkNodes(Network *network, CausetPerformance *cp, bool &use_gpu)
{
	assert (network->network_properties.N_tar > 0);
	assert (network->network_properties.a > 0.0);
	assert (network->network_properties.core_edge_fraction > 0.0 && network->network_properties.core_edge_fraction < 1.0);
	assert (network->network_properties.manifold == EUCLIDEAN || network->network_properties.manifold == DE_SITTER || network->network_properties.manifold == ANTI_DE_SITTER);
	assert (network->nodes != NULL);
	assert (network->past_edges != NULL);
	assert (network->future_edges != NULL);
	assert (network->past_edge_row_start != NULL);
	assert (network->future_edge_row_start != NULL);
	assert (network->core_edge_exists != NULL);
	
	float dt, dx;
	int core_limit = (int)(network->network_properties.core_edge_fraction * network->network_properties.N_tar);
	int future_idx = 0;
	int past_idx = 0;
	int i, j, k;

	stopwatchStart(&cp->sLinkNodes);

	for (i = 0; i < network->network_properties.N_tar; i++) {
		network->nodes[i].k_in = 0;
		network->nodes[i].k_out = 0;
	}

	//DEBUG
	if (network->network_properties.flags.universe)
		return false;

	//Identify future connections
	for (i = 0; i < network->network_properties.N_tar - 1; i++) {
		if (i < core_limit)
			network->core_edge_exists[(i*core_limit)+i] = false;
		network->future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < network->network_properties.N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			if (network->network_properties.manifold == EUCLIDEAN) {
				//
			} else if (network->network_properties.manifold == DE_SITTER) {
				dt = fabs(tToEta(network->nodes[j].t, network->network_properties.a) - tToEta(network->nodes[i].t, network->network_properties.a));
				assert (dt > 0.0 && dt < HALF_PI);
				//if (i % NPRINT == 0) printf("dt: %.5f\n", dt);
			} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
				//
			}

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (network->network_properties.dim == 1) {
				if (network->network_properties.manifold == EUCLIDEAN) {
					//
				} else if (network->network_properties.manifold == DE_SITTER) {
					//Formula given on p. 2 of [2]
					dx = M_PI - fabs(M_PI - fabs(network->nodes[j].theta - network->nodes[i].theta));
				} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
					//
				}
			} else if (network->network_properties.dim == 3) {
				if (network->network_properties.manifold == EUCLIDEAN) {
					//
				} else if (network->network_properties.manifold == DE_SITTER) {
					//Spherical Law of Cosines
					dx = acosf(X1(network->nodes[i].phi) * X1(network->nodes[j].phi) + 
						   X2(network->nodes[i].phi, network->nodes[i].chi) * X2(network->nodes[j].phi, network->nodes[j].chi) + 
						   X3(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X3(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta) + 
						   X4(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X4(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta));
				} else if (network->network_properties.manifold == ANTI_DE_SITTER) {
					//
				}
			}

			assert (dx > 0.0 && dx < HALF_PI);

			//if (i % NPRINT == 0) printf("dx: %.5f\n", dx);
			//if (i % NPRINT == 0) printf("cos(dx): %.5f\n", cosf(dx));

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
						
			//Link timelike relations
			try {
				if (dx < dt) {
					//if (i % NPRINT == 0) printf("\tConnected %d to %d\n", i, j);
					network->future_edges[future_idx] = j;
					future_idx++;
	
					if (future_idx == (network->network_properties.N_tar * network->network_properties.k_tar / 2) + network->network_properties.edge_buffer)
						throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
	
					//Record number of degrees for each node
					network->nodes[j].k_in++;
					network->nodes[i].k_out++;
				}
			} catch (CausetException c) {
				fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
				return false;
			} catch (std::exception e) {
				fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
				return false;
			}
		}

		//If there are no forward connections from node i, mark with -1
		if (network->future_edge_row_start[i] == future_idx)
			network->future_edge_row_start[i] = -1;
	}

	network->future_edge_row_start[network->network_properties.N_tar-1] = -1;
	//printf("\t\tEdges (forward): %d\n", future_idx);

	//printSpatialDistances(network->nodes, network->network_properties.manifold, network->network_properties.N_tar, network->network_properties.dim);

	//Write total degrees to file for this graph
	/*std::ofstream deg;
	deg.open("degrees.txt", std::ios::app);
	deg << future_idx << std::endl;
	deg.flush();
	deg.close();*/

	//Identify past connections
	network->past_edge_row_start[0] = -1;
	for (i = 1; i < network->network_properties.N_tar; i++) {
		network->past_edge_row_start[i] = past_idx;
		for (j = 0; j < i; j++) {
			if (network->future_edge_row_start[j] == -1)
				continue;

			for (k = 0; k < network->nodes[j].k_out; k++) {
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
	assert (future_idx == past_idx);
	//printf("\t\tEdges (backward): %d\n", past_idx);

	//Identify Resulting Network
	for (i = 0; i < network->network_properties.N_tar; i++) {
		if (network->nodes[i].k_in > 0 || network->nodes[i].k_out > 0) {
			network->network_properties.N_res++;
			network->network_properties.k_res += network->nodes[i].k_in + network->nodes[i].k_out;

			if (network->nodes[i].k_in + network->nodes[i].k_out > 1)
				network->network_properties.N_deg2++;
		} 
	}
	network->network_properties.k_res /= network->network_properties.N_res;

	assert (network->network_properties.N_res > 0);
	assert (network->network_properties.N_deg2 > 0);
	assert (network->network_properties.k_res > 0.0);

	//Debugging options used to visually inspect the adjacency lists and the adjacency pointer lists
	//compareAdjacencyLists(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);
	//compareAdjacencyListIndices(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);

	stopwatchStop(&cp->sLinkNodes);

	if (!network->network_properties.flags.bench) {
		printf("\tCausets Successfully Connected.\n");
		printf("\t\tResulting Network Size: %d\n", network->network_properties.N_res);
		printf("\t\tResulting Average Degree: %f\n", network->network_properties.k_res);
	}
	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sLinkNodes.elapsedTime);
	return true;
}

//Debug:  Future vs Past Edges in Adjacency List
//O(1) Efficiency
void compareAdjacencyLists(Node *nodes, int *future_edges, int *future_edge_row_start, int *past_edges, int *past_edge_row_start)
{
	assert (nodes != NULL);
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);
	assert (past_edges != NULL);
	assert (past_edge_row_start != NULL);

	int i, j;
	for (i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Forward Connections:\n");
		if (future_edge_row_start[i] == -1)
			printf("\tNo future connections.\n");
		else {
			for (j = 0; j < nodes[i].k_out && j < 10; j++)
				printf("%d ", future_edges[future_edge_row_start[i]+j]);
			printf("\n");
		}

		printf("Backward Connections:\n");
		if (past_edge_row_start[i] == -1)
			printf("\tNo past connections.\n");
		else {
			for (j = 0; j < nodes[i].k_in && j < 10; j++)
				printf("%d ", past_edges[past_edge_row_start[i]+j]);
			printf("\n");
		}
	}
}

//Debug:  Future and Past Adjacency List Indices
//O(1) Effiency
void compareAdjacencyListIndices(Node *nodes, int *future_edges, int *future_edge_row_start, int *past_edges, int *past_edge_row_start)
{
	assert (nodes != NULL);
	assert (future_edges != NULL);
	assert (future_edge_row_start != NULL);
	assert (past_edges != NULL);
	assert (past_edge_row_start != NULL);

	int max1 = 20;
	int max2 = 100;
	int i, j;

	printf("\nFuture Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", past_edge_row_start[i]);

	int next_future_idx, next_past_idx;
	for (i = 0; i < max1; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %d\n", nodes[i].k_out);
		if (future_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (j = 1; j < max2; j++) {
				if (future_edge_row_start[i+j] != -1) {
					next_future_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (future_edge_row_start[i+next_future_idx] - future_edge_row_start[i]));
		}

		printf("In-Degrees: %d\n", nodes[i].k_in);
		if (past_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (j = 1; j < max2; j++) {
				if (past_edge_row_start[i+j] != -1) {
					next_past_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (past_edge_row_start[i+next_past_idx] - past_edge_row_start[i]));
		}
	}
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
void printValues(Node *values, int num_vals, char *filename, char *coord)
{
	assert (values != NULL);
	assert (num_vals > 0);
	assert (filename != NULL);

	try {
		std::ofstream outputStream;
		outputStream.open(filename);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			if (strcmp(coord, "t") == 0)
				outputStream << values[i].t << std::endl;
			else if (strcmp(coord, "theta") == 0)
				outputStream << values[i].theta << std::endl;
			else if (strcmp(coord, "chi") == 0)
				outputStream << values[i].chi << std::endl;
			else if (strcmp(coord, "phi") == 0)
				outputStream << values[i].phi << std::endl;
			else
				throw CausetException("Unrecognized value in 'coord' parameter!\n");
		}
	
		outputStream.flush();
		outputStream.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		exit(EXIT_FAILURE);
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		exit(EXIT_FAILURE);
	}
}

//Write Spatial Distances to File
//O(N^2) Efficiency
void printSpatialDistances(Node *nodes, Manifold manifold, int N_tar, int dim)
{
	//Only de Sitter implemented here
	assert (nodes != NULL);
	assert (manifold == DE_SITTER);
	assert (dim == 1 || dim == 3);

	try {	
		std::ofstream dbgStream;
		float dx;
		dbgStream.open("distances.cset.dbg.dat");
		if (!dbgStream.is_open())
			throw CausetException("Failed to open 'distances.cset.dbg.dat' file!\n");
		int i, j;
		for (i = 0; i < N_tar - 1; i++) {
			for (j = i + 1; j < N_tar; j++) {
				if (dim == 1) dx = M_PI - fabs(M_PI - fabs(nodes[j].theta - nodes[i].theta));
				else if (dim == 3)
					dx = acosf((X1(nodes[i].phi) * X1(nodes[j].phi)) +
						   (X2(nodes[i].phi, nodes[i].chi) * X2(nodes[j].phi, nodes[j].chi)) +
						   (X3(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X3(nodes[j].phi, nodes[j].chi, nodes[j].theta)) +
						   (X4(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X4(nodes[j].phi, nodes[j].chi, nodes[j].theta)));
				dbgStream << dx << std::endl;
				if (i*N_tar+j > 500000) {
					i = N_tar;
					break;
				}
			}
		}
	
		dbgStream.flush();
		dbgStream.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		exit(EXIT_FAILURE);
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		exit(EXIT_FAILURE);
	}
}

#endif
