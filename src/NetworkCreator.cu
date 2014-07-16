#include "NetworkCreator.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (DEBUG)
		assert (network_properties != NULL);

	//If no seed specified, choose random one
	if (network_properties->seed == -12345L) {
		srand(time(NULL));
		network_properties->seed = -1.0 * static_cast<long>(time(NULL));
	}
	
	//Benchmarking
	if (network_properties->flags.bench) {
		network_properties->flags.verbose = false;
		network_properties->graphID = 0;
		network_properties->flags.disp_network = false;
		network_properties->flags.print_network = false;
	}

	//If graph ID specified, prepare to read graph properties
	if (network_properties->graphID && network_properties->flags.verbose && !network_properties->flags.yes) {
		printf("You have chosen to load a graph from memory.  Some parameters may be ignored as a result.  Continue [y/N]? ");
		fflush(stdout);
		char response = getchar();
		getchar();
		if (response != 'y')
			return false;
	}

	try {
		if (network_properties->flags.universe) {
			int i;

			//Check for conflicting topologies
			if (network_properties->dim == 1 || network_properties->manifold != DE_SITTER)
				throw CausetException("Universe causet must be 3+1 DS topology!\n");
				
			//Check for too many parameters
			if (network_properties->flags.cc.conflicts[0] > 1 || network_properties->flags.cc.conflicts[1] > 1 || network_properties->flags.cc.conflicts[2] > 1 || network_properties->flags.cc.conflicts[3] > 1 || network_properties->flags.cc.conflicts[4] > 3 || network_properties->flags.cc.conflicts[5] > 3 || network_properties->flags.cc.conflicts[6] > 3)
				throw CausetException("Causet model has been over-constrained!  Use flag --conflicts to find your error.\n");
			//Check for too few parameters
			else if (network_properties->N_tar == 0 && network_properties->alpha == 0.0)
				throw CausetException("Causet model has been under-constrained!  Specify at least '-n', number of nodes, or '-A', alpha, to proceed.\n");
				
			//Solve for constrained parameters
			if (network_properties->flags.cc.conflicts[1] == 0 && network_properties->flags.cc.conflicts[2] == 0 && network_properties->flags.cc.conflicts[3] == 0) {
				//Solve for tau0, ratio, omegaM, and omegaL
				double x = 0.5;
				if (DEBUG) {
					assert (network_properties->N_tar > 0);
					assert (network_properties->alpha > 0.0);
					assert (network_properties->delta > 0.0);
				}

				if (!newton(&solveTau0, &x, 10000, TOL, &network_properties->alpha, &network_properties->delta, NULL, NULL, &network_properties->N_tar, NULL))
					return false;
				network_properties->tau0 = x;
				if (DEBUG) assert (network_properties->tau0 > 0.0);

				network_properties->ratio = POW2(SINH(1.5 * static_cast<float>(network_properties->tau0), STL), EXACT);
				if (DEBUG) assert(network_properties->ratio > 0.0);
				network_properties->omegaM = 1.0 / (network_properties->ratio + 1.0);
				network_properties->omegaL = 1.0 - network_properties->omegaM;
			} else if (network_properties->flags.cc.conflicts[1] == 0 || network_properties->flags.cc.conflicts[2] == 0 || network_properties->flags.cc.conflicts[3] == 0) {
				//If k_tar != 0 solve for tau0 here
				if (network_properties->k_tar != 0) {
					if (DEBUG)
						assert (network_properties->delta != 0);

					Stopwatch sSolveTau0 = Stopwatch();
					stopwatchStart(&sSolveTau0);

					//Solve for tau_0
					printf("\tEstimating Age of Universe.....\n");
					float kappa1 = network_properties->k_tar / network_properties->delta;
					float kappa2 = kappa1 / POW2(POW2(static_cast<float>(network_properties->a), EXACT), EXACT);

					//Read Lookup Table
					double *raduc_lookup;
					long size;
					std::ifstream lookup("./etc/raduc_table.cset.bin", std::ios::in | std::ios::binary | std::ios::ate);
					if (lookup.is_open()) {
						//Find size of file
						size = lookup.tellg();
						//Allocate Memory for Buffer
						char *memblock = (char*)malloc(size);
						if (memblock == NULL)
							throw std::bad_alloc();
						hostMemUsed += size;

						//Allocate Memory for Lookup Table
						raduc_lookup = (double*)malloc(size);
						if (raduc_lookup == NULL)
							throw std::bad_alloc();
						hostMemUsed += size;

						//Read File
						lookup.seekg(0, std::ios::beg);
						lookup.read(memblock, size);
						memcpy(raduc_lookup, memblock, size);

						//Free Memory
						free(memblock);
						memblock = NULL;
						hostMemUsed -= size;

						//Close Stream
						lookup.close();
					} else
						throw CausetException("Failed to open 'raduc_table.cset.bin' file!\n");

					//Identify tau0
					//Assumes values are written as (kappa, tau0)
					int t_idx = 0;
					for (i = 0; i < size / (2 * sizeof(double)); i += 2) {
						if (raduc_lookup[i] > kappa2) {
							t_idx = i;
							break;
						}
					}

					//Check if Table is Insufficient
					if (t_idx == 0)
						throw CausetException("Values from 'raduc_table.cset.bin' file do not include requested 'k_tar'.  Recreate table or change k_tar.\n");

					//Linear Interpolation
					network_properties->tau0 = raduc_lookup[t_idx-1] + (raduc_lookup[t_idx+1] - raduc_lookup[t_idx-1]) * (kappa2 - raduc_lookup[t_idx-2]) / (raduc_lookup[t_idx] - raduc_lookup[t_idx-2]);

					//Free Memory
					free(raduc_lookup);
					raduc_lookup = NULL;
					hostMemUsed -= size;

					//Solve for ratio, omegaM, and omegaL
					network_properties->ratio = POW2(SINH(1.5f * network_properties->tau0, STL), EXACT);
					network_properties->omegaM = 1.0 / (network_properties->ratio + 1.0);
					network_properties->omegaL = 1.0 - network_properties->omegaM;

					stopwatchStop(&sSolveTau0);
					printf("\t\tCompleted.\n");
					printf("\t\t\tExecution Time: %5.6f\n", sSolveTau0.elapsedTime);
				}
				
				if (network_properties->N_tar > 0 && network_properties->alpha > 0.0) {
					//Solve for delta
					network_properties->delta = 3.0 * network_properties->N_tar / (POW2(static_cast<float>(M_PI), EXACT) * POW3(static_cast<float>(network_properties->alpha), EXACT) * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0));
					if (DEBUG) assert (network_properties->delta > 0.0);
				} else if (network_properties->N_tar == 0) {
					//Solve for N_tar
					if (DEBUG) assert (network_properties->alpha > 0.0);
					network_properties->N_tar = static_cast<int>(POW2(static_cast<float>(M_PI), EXACT) * network_properties->delta * POW3(static_cast<float>(network_properties->alpha), EXACT) * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0) / 3.0);
					if (DEBUG) assert (network_properties->N_tar > 0);
				} else {
					//Solve for alpha
					if (DEBUG) assert (network_properties->N_tar > 0);
					network_properties->alpha = POW(3.0 * network_properties->N_tar / (POW2(static_cast<float>(M_PI), EXACT) * network_properties->delta * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0)), (1.0 / 3.0), STL);
					if (DEBUG) assert (network_properties->alpha > 0.0);
				}
			}
			
			//Finally, solve for R0
			if (DEBUG) {
				assert (network_properties->alpha > 0.0);
				assert (network_properties->ratio > 0.0);
			}
			network_properties->R0 = network_properties->alpha * POW(static_cast<float>(network_properties->ratio), 1.0f / 3.0f, STL);
			if (DEBUG) assert (network_properties->R0 > 0.0);
			
			if (network_properties->k_tar == 0.0) {
				//Use Monte Carlo integration to find k_tar
				printf("\tEstimating Expected Average Degrees.....\n");
				double r0 = POW(SINH(1.5f * network_properties->tau0, STL), 2.0f / 3.0f, STL);

				if (network_properties->flags.bench) {
					for (i = 0; i < NBENCH; i++) {
						stopwatchStart(&cp->sCalcDegrees);
						integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network_properties->seed, 0);
						stopwatchStop(&cp->sCalcDegrees);
						bm->bCalcDegrees += cp->sCalcDegrees.elapsedTime;
						stopwatchReset(&cp->sCalcDegrees);
					}
					bm->bCalcDegrees /= NBENCH;
				}

				stopwatchStart(&cp->sCalcDegrees);
				network_properties->k_tar = network_properties->delta * POW2(POW2(static_cast<float>(network_properties->a), EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network_properties->seed, 0) * 8.0 * M_PI / (SINH(3.0f * network_properties->tau0, STL) - 3.0 * network_properties->tau0);
				stopwatchStop(&cp->sCalcDegrees);
				printf("\t\tExecution Time: %5.6f sec\n", cp->sCalcDegrees.elapsedTime);
				printf("\t\tCompleted.\n");
			}
			
			//20% Buffer
			network_properties->edge_buffer = static_cast<int>(0.1 * network_properties->N_tar * network_properties->k_tar);

			//Check success ratio parameters if applicable
			if (network_properties->flags.calc_success_ratio)
				network_properties->N_sr *= static_cast<int64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;

			//Adjacency matrix not implemented in GPU algorithms
			if (network_properties->flags.use_gpu)
				network_properties->core_edge_fraction = 0.0;
				
			printf("\n");
			printf("\tParameters Constraining Universe Causal Set:\n");
			printf("\t--------------------------------------------\n");
			printf_cyan();
			printf("\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf("\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			printf("\t > Pseudoradius:\t\t%.6f\n", network_properties->a);
			printf("\t > Cosmological Constant:\t%.6f\n", network_properties->lambda);
			printf("\t > Rescaled Age:\t\t%.6f\n", network_properties->tau0);
			printf("\t > Dark Energy Density:\t\t%.6f\n", network_properties->omegaL);
			printf("\t > Matter Density:\t\t%.6f\n", network_properties->omegaM);
			printf("\t > Ratio:\t\t\t%.6f\n", network_properties->ratio);
			printf("\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf("\t > Alpha:\t\t\t%.6f\n", network_properties->alpha);
			printf("\t > Scaling Factor:\t\t%.6f\n", network_properties->R0);
			printf_std();
			fflush(stdout);
		} else {
			if (network_properties->N_tar == 0)
				throw CausetException("Flag '-n', number of nodes, must be specified!\n");
			else if (network_properties->k_tar == 0.0)
				throw CausetException("Flag '-k', expected average degrees, must be specified!\n");
				
			if (network_properties->dim == 1 && network_properties->manifold == DE_SITTER) {
				network_properties->zeta = HALF_PI - network_properties->tau0;
				network_properties->tau0 = etaToTau(HALF_PI - network_properties->zeta);
			}
			
			if (network_properties->flags.calc_success_ratio) 
				network_properties->N_sr *= static_cast<int64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__,  e.what(), __LINE__);
		return false;
	}

	return true;
}

//Allocates memory for network
//O(1) Efficiency
bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &verbose, const bool &bench, const bool &yes)
{
	if (DEBUG) {
		//Variables in correct ranges
		assert (N_tar > 0);
		assert (k_tar > 0.0);
		assert (dim == 1 || dim == 3);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (edge_buffer >= 0);
	}

	if (verbose && !yes) {
		//Estimate memory usage before allocating
		size_t mem = 0;
		if (manifold == DE_SITTER && dim == 3)
			mem += sizeof(float4) * N_tar;
		else if (manifold == HYPERBOLIC || dim == 1)
			mem += sizeof(float) * N_tar;
		mem += sizeof(float) * N_tar;
		mem += sizeof(int) * (N_tar << 1);
		mem += sizeof(int) * 2 * (N_tar * k_tar / 2 + edge_buffer);
		mem += sizeof(int) * (N_tar << 1);
		mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);

		size_t dmem = 0;
		if (use_gpu) {
			if (manifold == DE_SITTER && dim == 3)
				dmem += sizeof(float4) * N_tar;
			else if (manifold == HYPERBOLIC || dim == 1)
				dmem += sizeof(float) * N_tar;
			dmem += sizeof(int) * 2 * (N_tar * k_tar / 2 + edge_buffer);
			dmem += sizeof(int) * (N_tar << 2);
		}

		printMemUsed("for Network (Estimation)", mem, dmem);
		printf("\nContinue [y/N]?");
		fflush(stdout);
		char response = getchar();
		getchar();
		if (response != 'y')
			return false;
	}

	stopwatchStart(&sCreateNetwork);

	try {
		if (manifold == DE_SITTER && dim == 3) {
			nodes.sc = (float4*)malloc(sizeof(float4) * N_tar);
			if (nodes.sc == NULL)
				throw std::bad_alloc();
			hostMemUsed += sizeof(float4) * N_tar;
		} else if (manifold == HYPERBOLIC || dim == 1) {
			nodes.theta = (float*)malloc(sizeof(float) * N_tar);
			if (nodes.theta == NULL)
				throw std::bad_alloc();
			hostMemUsed += sizeof(float) * N_tar;
		}

		nodes.tau = (float*)malloc(sizeof(float) * N_tar);
		if (nodes.tau == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(float) * N_tar;

		if (use_gpu)
			checkCudaErrors(cuMemHostAlloc((void**)&nodes.k_in, sizeof(int) * N_tar, CU_MEMHOSTALLOC_DEVICEMAP));
		else {
			nodes.k_in = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.k_in == NULL)
				throw std::bad_alloc();
		}
		hostMemUsed += sizeof(int) * N_tar;

		if (use_gpu)
			checkCudaErrors(cuMemHostAlloc((void**)&nodes.k_out, sizeof(int) * N_tar, CU_MEMHOSTALLOC_DEVICEMAP));
		else {
			nodes.k_out = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.k_out == NULL)
				throw std::bad_alloc();
		}
		hostMemUsed += sizeof(int) * N_tar;

		if (verbose)
			printMemUsed("for Nodes", hostMemUsed, devMemUsed);

		if (use_gpu)
			checkCudaErrors(cuMemHostAlloc((void**)&edges.past_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer), CU_MEMHOSTALLOC_DEVICEMAP));
		else {
			edges.past_edges = (int*)malloc(sizeof(int) * (N_tar * k_tar / 2 + edge_buffer));
			if (edges.past_edges == NULL)
				throw std::bad_alloc();
		}
		hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer);

		if (use_gpu)
			checkCudaErrors(cuMemHostAlloc((void**)&edges.future_edges, sizeof(int) * (N_tar * k_tar / 2 + edge_buffer), CU_MEMHOSTALLOC_DEVICEMAP));
		else {
			edges.future_edges = (int*)malloc(sizeof(int) * (N_tar * k_tar / 2 + edge_buffer));
			if (edges.future_edges == NULL)
				throw std::bad_alloc();
		}
		hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer);

		edges.past_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (edges.past_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		edges.future_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (edges.future_edge_row_start == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(int) * N_tar;

		core_edge_exists = (bool*)malloc(sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT));
		if (core_edge_exists == NULL)
			throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * static_cast<unsigned int>(POW2(core_edge_fraction * N_tar, EXACT));

		memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
		if (verbose)
			printMemUsed("for Network", hostMemUsed, devMemUsed);
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

bool solveMaxTime(const int &N_tar, const float &k_tar, const int &dim, const double &a, double &zeta, double &tau0, const double &alpha, const bool &universe)
{
	if (universe) {
		if (DEBUG) {
			assert (a > 0.0);
			assert (tau0 > 0.0);
			assert (alpha > 0.0);
		}
	
		if (USE_GSL) {
			IntData idata = IntData();
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
			idata.upper = tau0 * a;
			zeta = HALF_PI - integrate1D(&tauToEtaUniverse, NULL, &idata, QAGS);
			gsl_integration_workspace_free(idata.workspace);
		} else
			//Exact Solution
			zeta = HALF_PI - tauToEtaUniverseExact(tau0, a, alpha);
			
		if (DEBUG) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		}
	} else {
		//Solve for eta0 using Newton-Raphson Method
		if (DEBUG) {
			assert (N_tar > 0);
			assert (k_tar > 0.0);
			assert (dim == 1 || dim == 3);
		}

		double x = 0.08;
		if (dim == 1)
			x = HALF_PI - 0.0001;

		if (!newton(&solveZeta, &x, 10000, TOL, NULL, NULL, NULL, &k_tar, &N_tar, &dim))
			return false;

		if (dim == 1)
			zeta = HALF_PI - x;
		else if (dim == 3)
			zeta = x;
		tau0 = etaToTau(HALF_PI - zeta);

		if (DEBUG) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
			assert (tau0 > 0.0);
		}

		printf("\tTranscendental Equation Solved:\n");
		//printf("\t\tZeta: %5.8f\n", zeta);
		printf("\t\tMaximum Conformal Time: %5.8f\n", HALF_PI - zeta);
		printf("\t\tMaximum Rescaled Time:  %5.8f\n", tau0);
		fflush(stdout);
	}

	return true;
}

//Poisson Sprinkling
//O(N) Efficiency
bool generateNodes(const Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//Values are in correct ranges
		assert (N_tar > 0);
		assert (k_tar > 0.0);
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
		if (universe) {
			assert (dim == 3);
			assert (tau0 > 0.0);
		} else
			assert (zeta > 0.0 && zeta < HALF_PI);
	}

	IntData idata = IntData();
	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-4;
	if (USE_GSL && universe)
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	stopwatchStart(&sGenerateNodes);

	//Generate coordinates for each of N nodes
	double x, rval;
	int i;
	for (i = 0; i < N_tar; i++) {
		///////////////////////////////////////////////////////////
		//~~~~~~~~~~~~~~~~~~~~~~~~~Theta~~~~~~~~~~~~~~~~~~~~~~~~~//
		//Sample Theta from (0, 2pi), as described on p. 2 of [1]//
		///////////////////////////////////////////////////////////

		//nodes.sc[i].x = TWO_PI * ran2(&seed);
		x = TWO_PI * ran2(&seed);
		if (DEBUG) assert (x > 0.0 && x < TWO_PI);
		//if (i % NPRINT == 0) printf("Theta: %5.5f\n", x); fflush(stdout);

		if (dim == 1) {
			nodes.theta[i] = x;

			//CDF derived from PDF identified in (2) of [2]
			nodes.tau[i] = ATAN(ran2(&seed) / TAN(static_cast<float>(zeta), APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
			if (DEBUG) {
				assert (nodes.tau[i] > 0.0);
				assert (nodes.tau[i] < HALF_PI - zeta + 0.0000001);
			}
		} else if (dim == 3) {
			nodes.sc[i].x = x;

			/////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (6) of [2] for 3+1//
			//and from PDF identified in (12) of [2] for universe  //
			/////////////////////////////////////////////////////////

			rval = ran2(&seed);
			if (universe) {
				x = 0.5;
				if (!newton(&solveTauUniverse, &x, 1000, TOL, &tau0, &rval, NULL, NULL, NULL, NULL)) 
					return false;
			} else {
				x = 3.5;
				if (!newton(&solveTau, &x, 1000, TOL, &zeta, NULL, &rval, NULL, NULL, NULL))
					return false;
			}

			nodes.tau[i] = x;

			if (DEBUG) {
				assert (nodes.tau[i] > 0.0);
				assert (nodes.tau[i] < tau0);
			}

			//Save eta values as well
			if (universe) {
				if (USE_GSL) {
					//Numerical Integration
					idata.upper = nodes.tau[i] * a;
					nodes.sc[i].w = integrate1D(&tauToEtaUniverse, NULL, &idata, QAGS) / alpha;
				} else
					//Exact Solution
					nodes.sc[i].w = tauToEtaUniverseExact(nodes.tau[i], a, alpha);
			} else
				nodes.sc[i].w = tauToEta(nodes.tau[i]);
				
			////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~Phi and Chi~~~~~~~~~~~~~~~~~~~~//	
			//CDFs derived from PDFs identified on p. 3 of [2]//
			//Phi given by [3]				  //
			////////////////////////////////////////////////////

			//Sample Phi from (0, pi)
			//For some reason the technique in [3] has not been producing the correct distribution...
			//nodes.sc[i].y = 0.5 * (M_PI * ran2(&seed) + ACOS(static_cast<float>(ran2(&seed)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			x = HALF_PI;
			rval = ran2(&seed);
			if (!newton(&solvePhi, &x, 250, TOL, &rval, NULL, NULL, NULL, NULL, NULL)) 
				return false;
			nodes.sc[i].y = x;
			if (DEBUG) assert (nodes.sc[i].y > 0.0 && nodes.sc[i].y < M_PI);
			//if (i % NPRINT == 0) printf("Phi: %5.5f\n", nodes.sc[i].y); fflush(stdout);

			//Sample Chi from (0, pi)
			nodes.sc[i].z = ACOS(1.0 - 2.0 * static_cast<float>(ran2(&seed)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
			if (DEBUG) assert (nodes.sc[i].z > 0.0 && nodes.sc[i].z < M_PI);
			//if (i % NPRINT == 0) printf("Chi: %5.5f\n", nodes.sc[i].z); fflush(stdout);
		}
		//if (i % NPRINT == 0) printf("eta: %E\n", nodes.sc[i].w);
		//if (i % NPRINT == 0) printf("tau: %E\n", nodes.tau[i]);
	}

	//Debugging statements used to check coordinate distributions
	/*if (!printValues(nodes, N_tar, "tau_dist.cset.dbg.dat", "tau")) return false;
	if (!printValues(nodes, N_tar, "eta_dist.cset.dbg.dat", "eta")) return false;
	if (!printValues(nodes, N_tar, "theta_dist.cset.dbg.dat", "theta")) return false;
	if (!printValues(nodes, N_tar, "chi_dist.cset.dbg.dat", "chi")) return false;
	if (!printValues(nodes, N_tar, "phi_dist.cset.dbg.dat", "phi")) return false;
	printf("Check coordinate distributions now.\n");
	exit(EXIT_SUCCESS);*/

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
bool linkNodes(const Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Variables in correct ranges
		assert (N_tar > 0);
		assert (k_tar > 0.0);
		assert (dim == 1 || dim == 3);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
		if (universe) {
			assert (dim == 3);
			assert (alpha > 0.0);
		}
		assert (a > 0.0);
		assert (core_edge_fraction >= 0.0 && core_edge_fraction <= 1.0);
		assert (edge_buffer >= 0);
	}

	float dt, dx;
	int core_limit = static_cast<int>((core_edge_fraction * N_tar));
	int future_idx = 0;
	int past_idx = 0;
	int i, j, k;

	stopwatchStart(&sLinkNodes);

	memset(nodes.k_in, 0, sizeof(int) * N_tar);
	memset(nodes.k_out, 0, sizeof(int) * N_tar);
	
	//Identify future connections
	for (i = 0; i < N_tar - 1; i++) {
		if (i < core_limit)
			core_edge_exists[(i*core_limit)+i] = false;
		edges.future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			if (manifold == DE_SITTER) {
				//Assume nodes are already temporally ordered
				if (dim == 1)
					dt = nodes.tau[j] - nodes.tau[i];
				else if (dim == 3)
					dt = nodes.sc[j].w - nodes.sc[i].w;
				//if (i % NPRINT == 0) printf("dt: %.9f\n", dt); fflush(stdout);
				if (DEBUG) {
					assert (dt >= 0.0);
					assert (dt <= HALF_PI - zeta);
				}
			} else if (manifold == HYPERBOLIC) {
				//
			}

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (dim == 1) {
				if (manifold == DE_SITTER) {
					//Formula given on p. 2 of [2]
					dx = M_PI - ABS(M_PI - ABS(nodes.theta[j] - nodes.theta[i], STL), STL);
				} else if (manifold == HYPERBOLIC) {
					//
				}
			} else if (dim == 3) {
				//Spherical Law of Cosines
				dx = ACOS(sphProduct(nodes.sc[i], nodes.sc[j]), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
			}

			//if (i % NPRINT == 0) printf("dx: %.5f\n", dx); fflush(stdout);
			//if (i % NPRINT == 0) printf("cos(dx): %.5f\n", cosf(dx)); fflush(stdout);
			if (DEBUG) assert (dx >= 0.0 && dx <= M_PI);

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
					//if (i % NPRINT == 0) printf("%d %d\n", i, j); fflush(stdout);
					edges.future_edges[future_idx] = j;
					future_idx++;
	
					if (future_idx == N_tar * k_tar / 2 + edge_buffer)
						throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
	
					//Record number of degrees for each node
					nodes.k_in[j]++;
					nodes.k_out[i]++;
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
		if (edges.future_edge_row_start[i] == future_idx)
			edges.future_edge_row_start[i] = -1;
	}

	edges.future_edge_row_start[N_tar-1] = -1;
	printf("\t\tEdges (forward): %d\n", future_idx);
	fflush(stdout);

	//if (!printSpatialDistances(nodes, manifold, N_tar, dim)) return false;

	//Write total degrees to file for this graph
	/*std::ofstream deg;
	deg.open("degrees.txt", std::ios::app);
	deg << future_idx << std::endl;
	deg.flush();
	deg.close();*/

	//Identify past connections
	edges.past_edge_row_start[0] = -1;
	for (i = 1; i < N_tar; i++) {
		edges.past_edge_row_start[i] = past_idx;
		for (j = 0; j < i; j++) {
			if (edges.future_edge_row_start[j] == -1)
				continue;

			for (k = 0; k < nodes.k_out[j]; k++) {
				if (i == edges.future_edges[edges.future_edge_row_start[j]+k]) {
					edges.past_edges[past_idx] = j;
					past_idx++;
				}
			}
		}

		//If there are no backward connections from node i, mark with -1
		if (edges.past_edge_row_start[i] == past_idx)
			edges.past_edge_row_start[i] = -1;
	}

	//The quantities future_idx and past_idx should be equal
	if (DEBUG) assert (future_idx == past_idx);
	//printf("\t\tEdges (backward): %d\n", past_idx);
	//fflush(stdout);

	//Identify Resulting Network
	for (i = 0; i < N_tar; i++) {
		if (nodes.k_in[i] + nodes.k_out[i] > 0) {
			N_res++;
			k_res += nodes.k_in[i] + nodes.k_out[i];

			if (nodes.k_in[i] + nodes.k_out[i] > 1)
				N_deg2++;
		} 
	}

	if (DEBUG) {
		assert (N_res > 0);
		assert (N_deg2 > 0);
		assert (k_res > 0.0);
	}

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
void compareAdjacencyLists(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int i, j;
	for (i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Forward Connections:\n");
		if (edges.future_edge_row_start[i] == -1)
			printf("\tNo future connections.\n");
		else {
			for (j = 0; j < nodes.k_out[i] && j < 10; j++)
				printf("%d ", edges.future_edges[edges.future_edge_row_start[i]+j]);
			printf("\n");
		}

		printf("Backward Connections:\n");
		if (edges.past_edge_row_start[i] == -1)
			printf("\tNo past connections.\n");
		else {
			for (j = 0; j < nodes.k_in[i] && j < 10; j++)
				printf("%d ", edges.past_edges[edges.past_edge_row_start[i]+j]);
			printf("\n");
		}
	
		fflush(stdout);
	}
}

//Debug:  Future and Past Adjacency List Indices
//O(1) Effiency
void compareAdjacencyListIndices(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int max1 = 20;
	int max2 = 100;
	int i, j;

	printf("\nFuture Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.past_edge_row_start[i]);
	fflush(stdout);

	int next_future_idx, next_past_idx;
	for (i = 0; i < max1; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %d\n", nodes.k_out[i]);
		if (edges.future_edge_row_start[i] == -1) {
			printf("Pointer: 0\n");
		} else {
			for (j = 1; j < max2; j++) {
				if (edges.future_edge_row_start[i+j] != -1) {
					next_future_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.future_edge_row_start[i+next_future_idx] - edges.future_edge_row_start[i]));
		}

		printf("In-Degrees: %d\n", nodes.k_in[i]);
		if (edges.past_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (j = 1; j < max2; j++) {
				if (edges.past_edge_row_start[i+j] != -1) {
					next_past_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.past_edge_row_start[i+next_past_idx] - edges.past_edge_row_start[i]));
		}
		fflush(stdout);
	}
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(const Node &nodes, const int num_vals, const char *filename, const char *coord)
{
	if (DEBUG) {
		//No null pointers
		assert (filename != NULL);
		assert (coord != NULL);

		//Variables in correct range
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream;
		outputStream.open(filename);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			if (strcmp(coord, "tau") == 0)
				outputStream << nodes.tau[i] << std::endl;
			else if (strcmp(coord, "eta") == 0)
				outputStream << nodes.sc[i].w << std::endl;
			else if (strcmp(coord, "theta") == 0)
				outputStream << nodes.sc[i].x << std::endl;
			else if (strcmp(coord, "phi") == 0)
				outputStream << nodes.sc[i].y << std::endl;
			else if (strcmp(coord, "chi") == 0)
				outputStream << nodes.sc[i].z << std::endl;
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
