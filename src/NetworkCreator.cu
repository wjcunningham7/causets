#include "NetworkCreator.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Allocates memory for network
//O(1) Efficiency
bool createNetwork(Node *& nodes, int *& past_edges, int *& future_edges, int *& past_edge_row_start, int *& future_edge_row_start, bool *& core_edge_exists, CUdeviceptr &d_nodes, CUdeviceptr &d_edges, const int &N_tar, const float &k_tar, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &verbose, const bool &bench, const bool &yes)
{
	//Variables in correct ranges
	assert (N_tar > 0);
	assert (k_tar > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (edge_buffer >= 0);

	if (verbose && !yes) {
		//Estimate memory usage before allocating
		size_t mem = 0;
		mem += sizeof(Node) * N_tar;
		mem += sizeof(int) * 2 * (N_tar * k_tar / 2 + edge_buffer);
		mem += sizeof(int) * 2 * N_tar;
		mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);

		size_t dmem = 0;
		if (use_gpu) {
			dmem += sizeof(Node) * N_tar;
			dmem += sizeof(Node) * N_tar * k_tar / 2;
		}

		printMemUsed("for Network (Estimation)", mem, dmem);
		printf("\nContinue [y/N]?");
		fflush(stdout);
		char response = getchar();
		if (response != 'y')
			return false;
	}

	stopwatchStart(&sCreateNetwork);

	try {
		nodes = (Node*)malloc(sizeof(Node) * N_tar);
		if (nodes == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * N_tar;
		if (verbose)
			printMemUsed("Memory Allocated for Nodes", hostMemUsed, devMemUsed);

		past_edges = (int*)malloc(sizeof(int) * (N_tar * k_tar / 2 + edge_buffer));
		if (past_edges == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * (N_tar * k_tar / 2 + edge_buffer);

		future_edges = (int*)malloc(sizeof(int) * (N_tar * k_tar / 2 + edge_buffer));
		if (future_edges == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * (N_tar * k_tar / 2 + edge_buffer);

		past_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (past_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		future_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (future_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		core_edge_exists = (bool*)malloc(sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT));
		if (core_edge_exists == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);

		//Allocate memory on GPU if necessary
		if (use_gpu) {
			checkCudaErrors(cuMemAlloc(&d_nodes, sizeof(Node) * N_tar));
			devMemUsed += sizeof(Node) * N_tar;

			checkCudaErrors(cuMemAlloc(&d_edges, sizeof(Node) * N_tar * k_tar / 2));
			devMemUsed += sizeof(Node) * N_tar * k_tar / 2;
		}

		memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
		if (verbose)
			printMemUsed("Total Memory Allocated for Network", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	stopwatchStop(&sCreateNetwork);

	if (!bench) {
		printf("\tMemory Successfully Allocated.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sCreateNetwork.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Poisson Sprinkling
//O(N) Efficiency
bool generateNodes(Node * const &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &verbose, const bool &bench)
{
	//No null pointers
	assert (nodes != NULL);

	//Values are in correct ranges
	assert (N_tar > 0);
	assert (k_tar > 0.0);
	assert (dim == 1 || dim == 3);
	assert (manifold == EUCLIDEAN || manifold == DE_SITTER || manifold == ANTI_DE_SITTER);
	assert (a > 0.0);
	if (universe) {
		assert (dim == 3);
		assert (tau0 > 0.0);
	} else
		assert (zeta > 0.0 && zeta < HALF_PI);	

	IntData idata = IntData();
	if (USE_GSL && universe)
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	stopwatchStart(&sGenerateNodes);

	if (use_gpu) {
		//if (!generateNodesGPU(network))
			return false;
	} else {
		//Generate coordinates for each of N nodes
		double x, rval;
		int i;
		for (i = 0; i < N_tar; i++) {
			nodes[i] = Node();

			if (manifold == EUCLIDEAN) {
				//
			} else if (manifold == DE_SITTER) {
				///////////////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~~~~~~~~~Theta~~~~~~~~~~~~~~~~~~~~~~~~~//
				//Sample Theta from (0, 2pi), as described on p. 2 of [1]//
				///////////////////////////////////////////////////////////

				nodes[i].theta = TWO_PI * ran2(&seed);
				assert (nodes[i].theta > 0.0 && nodes[i].theta < TWO_PI);
				//if (i % NPRINT == 0) printf("Theta: %5.5f\n", nodes[i].theta); fflush(stdout);
			} else if (manifold == ANTI_DE_SITTER) {
				//
			}

			if (dim == 1) {
				if (manifold == EUCLIDEAN) {
					//
				} else if (manifold == DE_SITTER) {
					//CDF derived from PDF identified in (2) of [2]
					nodes[i].eta = ATAN(static_cast<float>(ran2(&seed)) / TAN(static_cast<float>(zeta), APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
					nodes[i].tau = etaToTau(nodes[i].eta);
					assert (nodes[i].eta > 0.0 && HALF_PI - nodes[i].eta > zeta);
				} else if (manifold == ANTI_DE_SITTER) {
					//
				}
			} else if (dim == 3) {
				if (manifold == EUCLIDEAN) {
					//
				} else if (manifold == DE_SITTER) {
					/////////////////////////////////////////////////////////
					//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
					//CDF derived from PDF identified in (6) of [2] for 3+1//
					//and from PDF identified in (12) of [2] for universe  //
					/////////////////////////////////////////////////////////

					rval = ran2(&seed);
					if (universe) {
						x = 0.5;
						if (!newton(&solveTau, &x, 1000, TOL, &tau0, &rval, NULL, NULL, NULL, NULL)) 
							return false;
					} else {
						//x = etaToT(HALF_PI - zeta, a);
						x = tau0 * a;
						if (!newton(&solveT, &x, 1000, TOL, &zeta, &a, &rval, NULL, NULL, NULL))
							return false;
						x /= a;
					}
					nodes[i].tau = x;
					assert (nodes[i].tau > 0.0);
					assert (nodes[i].tau < tau0);

					//Save eta values as well
					if (universe) {
						if (USE_GSL) {
							//Numerical Integration
							idata.upper = nodes[i].tau * a;
							nodes[i].eta = integrate1D(&tauToEtaUniverse, NULL, &idata, QAGS);
						} else
							//Exact Solution
							nodes[i].eta = tauToEtaUniverseExact(nodes[i].tau, a, alpha);
					} else
						nodes[i].eta = tauToEta(nodes[i].tau);
				
					////////////////////////////////////////////////////
					//~~~~~~~~~~~~~~~~~Phi and Chi~~~~~~~~~~~~~~~~~~~~//	
					//CDFs derived from PDFs identified on p. 3 of [2]//
					//Phi given by [3]				  //
					////////////////////////////////////////////////////

					//Sample Phi from (0, pi)
					//For some reason the technique in [3] has not been producing the correct distribution...
					//nodes[i].phi = 0.5 * (M_PI * ran2(&seed) + ACOS(static_cast<float>(ran2(&seed)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
					x = HALF_PI;
					rval = ran2(&seed);
					if (!newton(&solvePhi, &x, 250, TOL, &rval, NULL, NULL, NULL, NULL, NULL)) 
						return false;
					nodes[i].phi = x;
					assert (nodes[i].phi > 0.0 && nodes[i].phi < M_PI);
					//if (i % NPRINT == 0) printf("Phi: %5.5f\n", nodes[i].phi); fflush(stdout);

					//Sample Chi from (0, pi)
					nodes[i].chi = ACOS(1.0 - 2.0 * static_cast<float>(ran2(&seed)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
					assert (nodes[i].chi > 0.0 && nodes[i].chi < M_PI);
					//if (i % NPRINT == 0) printf("Chi: %5.5f\n", nodes[i].chi); fflush(stdout);
				} else if (manifold == ANTI_DE_SITTER) {
					//
				}
			}
			//if (i % NPRINT == 0) printf("T: %E\n", nodes[i].t);
		}

		//Debugging statements used to check coordinate distributions
		/*if (!printValues(nodes, N_tar, "tau_dist.cset.dbg.dat", "tau")) return false;
		if (!printValues(nodes, N_tar, "eta_dist.cset.dbg.dat", "eta")) return false;
		if (!printValues(nodes, N_tar, "theta_dist.cset.dbg.dat", "theta")) return false;
		if (!printValues(nodes, N_tar, "chi_dist.cset.dbg.dat", "chi")) return false;
		if (!printValues(nodes, N_tar, "phi_dist.cset.dbg.dat", "phi")) return false;
		printf("Check coordinate distributions now.\n");
		exit(EXIT_SUCCESS);*/
	}

	stopwatchStop(&sGenerateNodes);

	if (USE_GSL && universe)
		gsl_integration_workspace_free(idata.workspace);

	if (!bench) {
		printf("\tNodes Successfully Generated.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sGenerateNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Identify Causal Sets
//O(k*N^2) Efficiency
bool linkNodes(Node * const &nodes, int * const &past_edges, int * const &future_edges, int * const &past_edge_row_start, int * const &future_edge_row_start, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &verbose, const bool &bench)
{
	//No null pointers
	assert (nodes != NULL);
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (past_edge_row_start != NULL);
	assert (future_edge_row_start != NULL);
	assert (core_edge_exists != NULL);

	//Variables in correct ranges
	assert (N_tar > 0);
	assert (k_tar > 0.0);
	assert (dim == 1 || dim == 3);
	if (universe) {
		assert (dim == 3);
		assert (alpha > 0.0);
	}
	assert (manifold == EUCLIDEAN || manifold == DE_SITTER || manifold == ANTI_DE_SITTER);
	assert (a > 0.0);
	assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
	assert (edge_buffer >= 0);

	float dt, dx;
	int core_limit = static_cast<int>((core_edge_fraction * N_tar));
	int future_idx = 0;
	int past_idx = 0;
	int i, j, k;

	stopwatchStart(&sLinkNodes);

	for (i = 0; i < N_tar; i++) {
		nodes[i].k_in = 0;
		nodes[i].k_out = 0;
	}

	//Identify future connections
	for (i = 0; i < N_tar - 1; i++) {
		if (i < core_limit)
			core_edge_exists[(i*core_limit)+i] = false;
		future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			if (manifold == EUCLIDEAN) {
				//
			} else if (manifold == DE_SITTER) {
				dt = ABS(nodes[i].eta - nodes[j].eta, STL);
				//if (i % NPRINT == 0) printf("dt: %.9f\n", dt); fflush(stdout);
				assert (dt >= 0.0);
				assert (dt <= HALF_PI - zeta);
			} else if (manifold == ANTI_DE_SITTER) {
				//
			}

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (dim == 1) {
				if (manifold == EUCLIDEAN) {
					//
				} else if (manifold == DE_SITTER) {
					//Formula given on p. 2 of [2]
					dx = M_PI - ABS(static_cast<float>(M_PI) - ABS(nodes[j].theta - nodes[i].theta, STL), STL);
				} else if (manifold == ANTI_DE_SITTER) {
					//
				}
			} else if (dim == 3) {
				if (manifold == EUCLIDEAN) {
					//
				} else if (manifold == DE_SITTER) {
					//Spherical Law of Cosines
					dx = ACOS(X1(nodes[i].phi) * X1(nodes[j].phi) + 
						  X2(nodes[i].phi, nodes[i].chi) * X2(nodes[j].phi, nodes[j].chi) + 
						  X3(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X3(nodes[j].phi, nodes[j].chi, nodes[j].theta) + 
						  X4(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X4(nodes[j].phi, nodes[j].chi, nodes[j].theta), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
				} else if (manifold == ANTI_DE_SITTER) {
					//
				}
			}

			//if (i % NPRINT == 0) printf("dx: %.5f\n", dx); fflush(stdout);
			//if (i % NPRINT == 0) printf("cos(dx): %.5f\n", cosf(dx)); fflush(stdout);
			//assert (dx >= 0.0 && dx <= M_PI);

			//Core Edge Adjacency Matrix
			if (i < core_limit && j < core_limit) {
				if (dx > dt) {
					core_edge_exists[(i * core_limit) + j] = false;
					core_edge_exists[(j * core_limit) + i] = false;
				} else {
					core_edge_exists[(i * core_limit) + j] = true;
					core_edge_exists[(j * core_limit) + i] = true;
				}
			}
						
			//Link timelike relations
			try {
				if (dx < dt) {
					//if (i % NPRINT == 0) printf("\tConnected %d to %d\n", i, j); fflush(stdout);
					future_edges[future_idx] = j;
					future_idx++;
	
					if (future_idx == (N_tar * k_tar / 2) + edge_buffer)
						throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
	
					//Record number of degrees for each node
					nodes[j].k_in++;
					nodes[i].k_out++;
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
		if (future_edge_row_start[i] == future_idx)
			future_edge_row_start[i] = -1;
	}

	future_edge_row_start[N_tar-1] = -1;
	//printf("\t\tEdges (forward): %d\n", future_idx);
	//fflush(stdout);

	//if (!printSpatialDistances(nodes, manifold, N_tar, dim)) return false;

	//Write total degrees to file for this graph
	/*std::ofstream deg;
	deg.open("degrees.txt", std::ios::app);
	deg << future_idx << std::endl;
	deg.flush();
	deg.close();*/

	//Identify past connections
	past_edge_row_start[0] = -1;
	for (i = 1; i < N_tar; i++) {
		past_edge_row_start[i] = past_idx;
		for (j = 0; j < i; j++) {
			if (future_edge_row_start[j] == -1)
				continue;

			for (k = 0; k < nodes[j].k_out; k++) {
				if (i == future_edges[future_edge_row_start[j]+k]) {
					past_edges[past_idx] = j;
					past_idx++;
				}
			}
		}

		//If there are no backward connections from node i, mark with -1
		if (past_edge_row_start[i] == past_idx)
			past_edge_row_start[i] = -1;
	}

	//The quantities future_idx and past_idx should be equal
	assert (future_idx == past_idx);
	//printf("\t\tEdges (backward): %d\n", past_idx);
	//fflush(stdout);

	//Identify Resulting Network
	for (i = 0; i < N_tar; i++) {
		if (nodes[i].k_in > 0 || nodes[i].k_out > 0) {
			N_res++;
			k_res += nodes[i].k_in + nodes[i].k_out;

			if (nodes[i].k_in + nodes[i].k_out > 1)
				N_deg2++;
		} 
	}

	assert (N_res > 0);
	assert (N_deg2 > 0);
	assert (k_res > 0.0);

	k_res /= N_res;

	//Debugging options used to visually inspect the adjacency lists and the adjacency pointer lists
	//compareAdjacencyLists(nodes, past_edges, future_edges, past_edge_row_start, future_edge_row_start);
	//compareAdjacencyListIndices(nodes, past_edges, future_edges, past_edge_row_start, future_edge_row_start);

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf("\t\tResulting Network Size: %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Debug:  Future vs Past Edges in Adjacency List
//O(1) Efficiency
void compareAdjacencyLists(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start)
{
	//No null pointers
	assert (nodes != NULL);
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (past_edge_row_start != NULL);
	assert (future_edge_row_start != NULL);

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
	
		fflush(stdout);
	}
}

//Debug:  Future and Past Adjacency List Indices
//O(1) Effiency
void compareAdjacencyListIndices(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start)
{
	//No null pointers
	assert (nodes != NULL);
	assert (past_edges != NULL);
	assert (future_edges != NULL);
	assert (past_edge_row_start != NULL);
	assert (future_edge_row_start != NULL);

	int max1 = 20;
	int max2 = 100;
	int i, j;

	printf("\nFuture Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", past_edge_row_start[i]);
	fflush(stdout);

	int next_future_idx, next_past_idx;
	for (i = 0; i < max1; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %d\n", nodes[i].k_out);
		if (future_edge_row_start[i] == -1) {
			printf("Pointer: 0\n");
		} else {
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
		fflush(stdout);
	}
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(const Node * const nodes, const int num_vals, const char *filename, const char *coord)
{
	//No null pointers
	assert (nodes != NULL);
	assert (filename != NULL);
	assert (coord != NULL);

	//Variables in correct range
	assert (num_vals > 0);

	try {
		std::ofstream outputStream;
		outputStream.open(filename);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			if (strcmp(coord, "tau") == 0)
				outputStream << nodes[i].tau << std::endl;
			else if (strcmp(coord, "eta") == 0)
				outputStream << nodes[i].eta << std::endl;
			else if (strcmp(coord, "theta") == 0)
				outputStream << nodes[i].theta << std::endl;
			else if (strcmp(coord, "chi") == 0)
				outputStream << nodes[i].chi << std::endl;
			else if (strcmp(coord, "phi") == 0)
				outputStream << nodes[i].phi << std::endl;
			else
				throw CausetException("Unrecognized value in 'coord' parameter!\n");
		}
	
		outputStream.flush();
		outputStream.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}

//Write Spatial Distances to File
//O(N^2) Efficiency
bool printSpatialDistances(const Node * const nodes, const Manifold &manifold, const int &N_tar, const int &dim)
{
	//No null pointers
	assert (nodes != NULL);

	//Variables in correct ranges
	assert (manifold == DE_SITTER); //Only de Sitter implemented here
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
				if (dim == 1) dx = M_PI - ABS(static_cast<float>(M_PI) - ABS(nodes[j].theta - nodes[i].theta, STL), STL);
				else if (dim == 3)
					dx = ACOS((X1(nodes[i].phi) * X1(nodes[j].phi)) +
						  (X2(nodes[i].phi, nodes[i].chi) * X2(nodes[j].phi, nodes[j].chi)) +
						  (X3(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X3(nodes[j].phi, nodes[j].chi, nodes[j].theta)) +
						  (X4(nodes[i].phi, nodes[i].chi, nodes[i].theta) * X4(nodes[j].phi, nodes[j].chi, nodes[j].theta)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
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
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}
