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

	//Optimize Parameters for GPU
	if (network_properties->flags.use_gpu && network_properties->N_tar % (BLOCK_SIZE << 1)) {
		printf("If you are using the GPU, set the target number of nodes (--nodes) to be a multiple of double the thread block size (%d)!\n", BLOCK_SIZE << 1);
		fflush(stdout);
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

				double t = 6.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * network_properties->delta * network_properties->a * POW3(network_properties->alpha, EXACT));
				if (t > MTAU)
					x = LOG(t, STL) / 3.0;
				else if (!newton(&solveTau0, &x, 10000, TOL, &network_properties->alpha, &network_properties->delta, network_properties->a, NULL, &network_properties->N_tar, NULL))
					return false;
				network_properties->tau0 = x;
				if (DEBUG) assert (network_properties->tau0 > 0.0);

				if (t > MTAU)
					network_properties->ratio = exp(3.0 * network_properties->tau0) / 4.0;
				else
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
					//printf_red();
					//printf("kappa2: %f\n", kappa2);
					//printf_std();

					//Use Lookup Table
					network_properties->tau0 = lookup("./etc/raduc_table.cset.bin", NULL, &kappa2);

					//Solve for ratio, omegaM, and omegaL
					if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
						network_properties->ratio = exp(3.0 * network_properties->tau0) / 4.0;
					else
						network_properties->ratio = POW2(SINH(1.5f * network_properties->tau0, STL), EXACT);
					network_properties->omegaM = 1.0 / (network_properties->ratio + 1.0);
					network_properties->omegaL = 1.0 - network_properties->omegaM;

					stopwatchStop(&sSolveTau0);
					printf("\t\tExecution Time: %5.6f\n", sSolveTau0.elapsedTime);
					printf("\tCompleted.\n");
				}
				
				if (network_properties->N_tar > 0 && network_properties->alpha > 0.0) {
					//Solve for delta
					//Add MTAU approximation here
					network_properties->delta = 3.0 * network_properties->N_tar / (POW2(static_cast<float>(M_PI), EXACT) * POW3(static_cast<float>(network_properties->alpha), EXACT) * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0) * network_properties->a);
					if (DEBUG) assert (network_properties->delta > 0.0);
				} else if (network_properties->N_tar == 0) {
					//Solve for N_tar
					//Add MTAU approximation here
					if (DEBUG) assert (network_properties->alpha > 0.0);
					network_properties->N_tar = static_cast<int>(POW2(static_cast<float>(M_PI), EXACT) * network_properties->delta * POW3(static_cast<float>(network_properties->alpha), EXACT) * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0) * network_properties->a / 3.0);
					if (DEBUG) assert (network_properties->N_tar > 0);
				} else {
					//Solve for alpha
					if (DEBUG) assert (network_properties->N_tar > 0);
					if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
						network_properties->alpha = exp(LOG(6.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * network_properties->delta * network_properties->a), STL) / 3.0 - network_properties->tau0);
					else
						network_properties->alpha = static_cast<double>(POW(3.0 * network_properties->N_tar / (POW2(static_cast<float>(M_PI), EXACT) * network_properties->delta * (SINH(3.0 * static_cast<float>(network_properties->tau0), STL) - 3.0 * network_properties->tau0) * network_properties->a), (1.0 / 3.0), STL));
					if (DEBUG) assert (network_properties->alpha > 0.0);
				}
			}

			//Solve for Rescaled Densities
			if (DEBUG) {
				assert (network_properties->a > 0.0);
				assert (network_properties->tau0 > 0.0);
			}
			network_properties->rhoL = 1.0 / POW2(network_properties->a, EXACT);
			if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
				network_properties->rhoM = 4.0 * exp(-3.0 * network_properties->tau0) / POW2(network_properties->a, EXACT);
			else
				network_properties->rhoM = 1.0 / POW2(network_properties->a * SINH(1.5f * network_properties->tau0, STL), EXACT);
			if (DEBUG) {
				assert (network_properties->rhoL > 0.0);
				assert (network_properties->rhoM > 0.0);
			}

			//Rescale Alpha (factor of 'a' missing in (11) from [2])
			//network_properties->alpha /= POW(network_properties->a, 1.0f / 3.0f, STL);
			
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
				double r0;
				if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
					r0 = POW(0.5f, 2.0f / 3.0f, STL) * exp(network_properties->tau0);
				else
					r0 = POW(SINH(1.5f * network_properties->tau0, STL), 2.0f / 3.0f, STL);

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
				if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
					network_properties->k_tar = network_properties->delta * POW2(POW2(static_cast<float>(network_properties->a), EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network_properties->seed, 0) * 16.0 * M_PI * exp(-3.0 * network_properties->tau0);
				else
					network_properties->k_tar = network_properties->delta * POW2(POW2(static_cast<float>(network_properties->a), EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network_properties->seed, 0) * 8.0 * M_PI / (SINH(3.0f * network_properties->tau0, STL) - 3.0 * network_properties->tau0);
				stopwatchStop(&cp->sCalcDegrees);
				printf("\t\tExecution Time: %5.6f sec\n", cp->sCalcDegrees.elapsedTime);
				printf("\t\tCompleted.\n");
			}
			
			//20% Buffer
			network_properties->edge_buffer = static_cast<int>(0.1 * network_properties->N_tar * network_properties->k_tar);
			//printf("Edge Buffer: %d\n", network_properties->edge_buffer);

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
			printf("\t > Rescaled Energy Density:\t%.6f\n", network_properties->rhoL);
			printf("\t > Matter Density:\t\t%.6f\n", network_properties->omegaM);
			printf("\t > Rescaled Matter Density:\t%.6f\n", network_properties->rhoM);
			printf("\t > Ratio:\t\t\t%.6f\n", network_properties->ratio);
			printf("\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf("\t > Alpha:\t\t\t%.6f\n", network_properties->alpha);
			printf("\t > Scaling Factor:\t\t%.6f\n", network_properties->R0);
			printf("\n");
			printf_std();
			fflush(stdout);
		} else if (network_properties->manifold == DE_SITTER) {
			if (network_properties->N_tar == 0)
				throw CausetException("Flag '-n', number of nodes, must be specified!\n");
			else if (network_properties->k_tar == 0.0)
				throw CausetException("Flag '-k', expected average degrees, must be specified!\n");
				
			if (network_properties->dim == 1) {
				network_properties->zeta = HALF_PI - network_properties->tau0;
				network_properties->tau0 = etaToTau(HALF_PI - network_properties->zeta);
			}
		}
			
		//Check other parameters if applicable
		if (network_properties->flags.validate_embedding)
			network_properties->N_emb *= static_cast<uint64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;
		if (network_properties->flags.calc_success_ratio)
			network_properties->N_sr *= static_cast<uint64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;
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
		//Estimate memory usage before allocating (update this for new data structures)
		size_t mem = 0;
		if (dim == 3)
			mem += sizeof(float4) * N_tar;
		else if (dim == 1)
			mem += sizeof(float2) * N_tar;
		if (manifold == HYPERBOLIC)
			mem += sizeof(int) * N_tar;
		else if (manifold == DE_SITTER)
			mem += sizeof(float) * N_tar;
		mem += sizeof(int) * (N_tar << 1);
		mem += sizeof(int) * 2 * (N_tar * k_tar / 2 + edge_buffer);
		mem += sizeof(int) * (N_tar << 1);
		mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);

		size_t dmem = 0;
		if (use_gpu) {
			if (dim == 3)
				dmem += sizeof(float4) * N_tar;
			else if (dim == 1)
				dmem += sizeof(float2) * N_tar;
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
		if (manifold == DE_SITTER) {
			nodes.id.tau = (float*)malloc(sizeof(float) * N_tar);
			if (nodes.id.tau == NULL)
				throw std::bad_alloc();
			memset(nodes.id.tau, 0, sizeof(float) * N_tar);
			hostMemUsed += sizeof(float) * N_tar;
		} else if (manifold == HYPERBOLIC) {
			nodes.id.AS = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.id.AS == NULL)
				throw std::bad_alloc();
			memset(nodes.id.AS, 0, sizeof(int) * N_tar);
			hostMemUsed += sizeof(int) * N_tar;
		}

		if (dim == 3) {
			nodes.c.sc = (float4*)malloc(sizeof(float4) * N_tar);
			if (nodes.c.sc == NULL)
				throw std::bad_alloc();
			memset(nodes.c.sc, 0, sizeof(float4) * N_tar);
			hostMemUsed += sizeof(float4) * N_tar;
		} else if (dim == 1) {
			nodes.c.hc = (float2*)malloc(sizeof(float2) * N_tar);
			if (nodes.c.hc == NULL)
				throw std::bad_alloc();
			memset(nodes.c.hc, 0, sizeof(float2) * N_tar);
			hostMemUsed += sizeof(float2) * N_tar;
		}

		nodes.k_in = (int*)malloc(sizeof(int) * N_tar);
		if (nodes.k_in == NULL)
			throw std::bad_alloc();
		memset(nodes.k_in, 0, sizeof(int) * N_tar);
		hostMemUsed += sizeof(int) * N_tar;

		nodes.k_out = (int*)malloc(sizeof(int) * N_tar);
		if (nodes.k_out == NULL)
			throw std::bad_alloc();
		memset(nodes.k_out, 0, sizeof(int) * N_tar);
		hostMemUsed += sizeof(int) * N_tar;

		if (verbose)
			printMemUsed("for Nodes", hostMemUsed, devMemUsed);

		edges.past_edges = (int*)malloc(sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer));
		if (edges.past_edges == NULL)
			throw std::bad_alloc();
		memset(edges.past_edges, 0, sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer));
		hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer);

		edges.future_edges = (int*)malloc(sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer));
		if (edges.future_edges == NULL)
			throw std::bad_alloc();
		memset(edges.future_edges, 0, sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer));
		hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar / 2 + edge_buffer);

		edges.past_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (edges.past_edge_row_start == NULL)
			throw std::bad_alloc();
		memset(edges.past_edge_row_start, 0, sizeof(int) * N_tar);
		hostMemUsed += sizeof(int) * N_tar;

		edges.future_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
		if (edges.future_edge_row_start == NULL)
			throw std::bad_alloc();
		memset(edges.future_edge_row_start, 0, sizeof(int) * N_tar);
		hostMemUsed += sizeof(int) * N_tar;

		core_edge_exists = (bool*)malloc(sizeof(bool) * static_cast<unsigned int>(POW2(core_edge_fraction * N_tar, EXACT)));
		if (core_edge_exists == NULL)
			throw std::bad_alloc();
		memset(core_edge_exists, 0, sizeof(bool) * static_cast<unsigned int>(POW2(core_edge_fraction * N_tar, EXACT)));
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
			idata.limit = 50;
			idata.tol = 1e-4;
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
			idata.upper = tau0 * a;
			zeta = HALF_PI - integrate1D(&tauToEtaUniverse, NULL, &idata, QAGS);
			gsl_integration_workspace_free(idata.workspace);
		} else
			//Exact Solution
			zeta = HALF_PI - tauToEtaUniverseExact(tau0, a, alpha);
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
		printf_cyan();
		//printf("\t\tZeta: %5.8f\n", zeta);
		printf("\t\tMaximum Conformal Time: %5.8f\n", HALF_PI - zeta);
		printf("\t\tMaximum Rescaled Time:  %5.8f\n", tau0);
		printf_std();
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

		x = TWO_PI * ran2(&seed);
		if (DEBUG) assert (x > 0.0 && x < TWO_PI);
		//if (i % NPRINT == 0) printf("Theta: %5.5f\n", x); fflush(stdout);

		if (dim == 1) {
			nodes.c.hc[i].y = x;

			//CDF derived from PDF identified in (2) of [2]
			nodes.c.hc[i].x = ATAN(ran2(&seed) / TAN(static_cast<float>(zeta), APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
			if (DEBUG) {
				assert (nodes.c.hc[i].x > 0.0);
				assert (nodes.c.hc[i].x < HALF_PI - zeta + 0.0000001);
			}

			nodes.id.tau[i] = etaToTau(nodes.c.hc[i].x);
		} else if (dim == 3) {
			nodes.c.sc[i].x = x;

			/////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (6) of [2] for 3+1//
			//and from PDF identified in (12) of [2] for universe  //
			/////////////////////////////////////////////////////////

			rval = ran2(&seed);
			if (universe) {
				x = 0.5;
				if (tau0 > 1.8) {	//Determined by trial and error
					if (!bisection(&solveTauUnivBisec, &x, 2000, 0.0, tau0, TOL, true, &tau0, &rval, NULL, NULL, NULL, NULL))
						return false;
				} else {
					if (!newton(&solveTauUniverse, &x, 1000, TOL, &tau0, &rval, NULL, NULL, NULL, NULL)) 
						return false;
				}
			} else {
				x = 3.5;
				if (!newton(&solveTau, &x, 1000, TOL, &zeta, NULL, &rval, NULL, NULL, NULL))
					return false;
			}

			nodes.id.tau[i] = x;

			if (DEBUG) {
				assert (nodes.id.tau[i] > 0.0);
				assert (nodes.id.tau[i] < tau0);
			}

			//Save eta values as well
			if (universe) {
				if (USE_GSL) {
					//Numerical Integration
					idata.upper = nodes.id.tau[i] * a;
				} else
					//Exact Solution
					nodes.c.sc[i].w = tauToEtaUniverseExact(nodes.id.tau[i], a, alpha);
			} else
				nodes.c.sc[i].w = tauToEta(nodes.id.tau[i]);
				
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
			nodes.c.sc[i].y = x;
			if (DEBUG) assert (nodes.c.sc[i].y > 0.0 && nodes.c.sc[i].y < M_PI);
			//if (i % NPRINT == 0) printf("Phi: %5.5f\n", nodes.c.sc[i].y); fflush(stdout);

			//Sample Chi from (0, pi)
			nodes.c.sc[i].z = ACOS(1.0 - 2.0 * static_cast<float>(ran2(&seed)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
			if (DEBUG) assert (nodes.c.sc[i].z > 0.0 && nodes.c.sc[i].z < M_PI);
			//if (i % NPRINT == 0) printf("Chi: %5.5f\n", nodes.c.sc[i].z); fflush(stdout);
		}
		//if (i % NPRINT == 0) printf("eta: %E\n", nodes.c.sc[i].w);
		//if (i % NPRINT == 0) printf("tau: %E\n", nodes.id.tau[i]);
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
		assert (manifold == DE_SITTER);
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

	//Identify future connections
	for (i = 0; i < N_tar - 1; i++) {
		if (i < core_limit)
			core_edge_exists[(i*core_limit)+i] = false;
		edges.future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			//Assume nodes are already temporally ordered
			if (dim == 1)
				dt = nodes.c.hc[j].x - nodes.c.hc[i].x;
			else if (dim == 3)
				dt = nodes.c.sc[j].w - nodes.c.sc[i].w;
			//if (i % NPRINT == 0) printf("dt: %.9f\n", dt); fflush(stdout);
			if (DEBUG) {
				assert (dt >= 0.0);
				assert (dt <= HALF_PI - zeta);
			}

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (dim == 1) {
				//Formula given on p. 2 of [2]
				dx = M_PI - ABS(M_PI - ABS(nodes.c.hc[j].y - nodes.c.hc[i].y, STL), STL);
			} else if (dim == 3) {
				//Spherical Law of Cosines
				dx = ACOS(sphProduct(nodes.c.sc[i], nodes.c.sc[j]), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
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
					edges.future_edges[++future_idx] = j;
					//future_idx++;
	
					if (future_idx >= N_tar * k_tar / 2 + edge_buffer)
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
	printf("\t\tUndirected Links: %d\n", future_idx);
	fflush(stdout);

	//if (!printSpatialDistances(nodes, manifold, N_tar, dim)) return false;

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
	//compareAdjacencyLists(nodes, edges);
	//compareAdjacencyListIndices(nodes, edges);

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", (k_res * N_res) / N_tar);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}
