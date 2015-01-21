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

	#ifdef CUDA_ENABLED
	//Optimize Parameters for GPU
	if (network_properties->flags.use_gpu && network_properties->N_tar % (BLOCK_SIZE << 1)) {
		printf("If you are using the GPU, set the target number of nodes (--nodes) to be a multiple of double the thread block size (%d)!\n", BLOCK_SIZE << 1);
		fflush(stdout);
		return false;
	}
	#endif

	try {
		if (network_properties->flags.universe) {
			//Check for conflicting topological parameters
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
				if (DEBUG) {
					assert (network_properties->N_tar > 0);
					assert (network_properties->alpha > 0.0);
					assert (network_properties->delta > 0.0);
				}

				//BEGIN COMPACT EQUATIONS (Completed)

				double p1[4];
				double t;
				double x = 0.5;

				p1[0] = network_properties->alpha;
				p1[1] = network_properties->delta;
				p1[2] = network_properties->a;

				if (network_properties->flags.compact) {
					t = 6.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * network_properties->delta * network_properties->a * POW3(network_properties->alpha, EXACT));
					if (t > MTAU)
						x = LOG(t, STL) / 3.0;
					else if (!newton(&solveTau0Compact, &x, 10000, TOL, p1, NULL, &network_properties->N_tar))
						return false;
				} else {
					p1[3] = network_properties->chi_max;
					t = 9.0 * network_properties->N_tar / (M_PI * network_properties->delta * network_properties->a * POW3(network_properties->alpha * network_properties->chi_max, EXACT));
					if (t > MTAU)
						x = LOG(t, STL) / 3.0;
					else if (!newton(&solveTau0Flat, &x, 10000, TOL, p1, NULL, &network_properties->N_tar))
						return false;
				}

				//END COMPACT EQUATIONS

				network_properties->tau0 = x;
				if (DEBUG)
					assert (network_properties->tau0 > 0.0);

				if (t > MTAU)
					network_properties->ratio = exp(3.0 * network_properties->tau0) / 4.0;
				else
					network_properties->ratio = POW2(SINH(1.5 * network_properties->tau0, STL), EXACT);

				if (DEBUG)
					assert(network_properties->ratio > 0.0);

				network_properties->omegaM = 1.0 / (network_properties->ratio + 1.0);
				network_properties->omegaL = 1.0 - network_properties->omegaM;
			} else if (network_properties->flags.cc.conflicts[1] == 0 || network_properties->flags.cc.conflicts[2] == 0 || network_properties->flags.cc.conflicts[3] == 0) {
				//If k_tar != 0 solve for tau0 here
				if (network_properties->k_tar != 0.0) {
					if (DEBUG)
						assert (network_properties->delta != 0.0);

					Stopwatch sSolveTau0 = Stopwatch();
					stopwatchStart(&sSolveTau0);

					//Solve for tau_0
					printf("Estimating Age of Universe.....\n");
					fflush(stdout);
					double kappa1 = network_properties->k_tar / network_properties->delta;
					double kappa2 = kappa1 / POW2(POW2(network_properties->a, EXACT), EXACT);

					//Use Lookup Table
					double *table;
					long size = 0L;
					if (!getLookupTable("./etc/raduc_table.cset.bin", &table, &size))
						return false;
					network_properties->tau0 = lookupValue(table, size, NULL, &kappa2, true);
					//Check for NaN
					if (network_properties->tau0 != network_properties->tau0)
						return false;

					//Solve for ratio, omegaM, and omegaL
					if (network_properties->tau0 > LOG(MTAU, STL) / 3.0)
						network_properties->ratio = exp(3.0 * network_properties->tau0) / 4.0;
					else
						network_properties->ratio = POW2(SINH(1.5 * network_properties->tau0, STL), EXACT);

					network_properties->omegaM = 1.0 / (network_properties->ratio + 1.0);
					network_properties->omegaL = 1.0 - network_properties->omegaM;

					stopwatchStop(&sSolveTau0);
					if (network_properties->flags.verbose) {
						printf("\tExecution Time: %5.6f sec\n", sSolveTau0.elapsedTime);
					}
					printf("Task Completed.\n");
					fflush(stdout);
				}
				
				//BEGIN COMPACT EQUATIONS (Completed)

				if (network_properties->N_tar > 0 && network_properties->alpha > 0.0) {
					//Solve for delta
					if (network_properties->flags.compact)
						network_properties->delta = solveDeltaCompact(network_properties->N_tar, network_properties->a, network_properties->tau0, network_properties->alpha);
					else
						network_properties->delta = solveDeltaFlat(network_properties->N_tar, network_properties->a, network_properties->chi_max, network_properties->tau0, network_properties->alpha);
				} else if (network_properties->N_tar == 0) {
					//Solve for N_tar
					if (network_properties->flags.compact)
						network_properties->N_tar = solveNtarCompact(network_properties->a, network_properties->tau0, network_properties->alpha, network_properties->delta);
					else
						network_properties->N_tar = solveNtarFlat(network_properties->a, network_properties->chi_max, network_properties->tau0, network_properties->alpha, network_properties->delta);
				} else {
					//Solve for alpha
					if (network_properties->flags.compact)
						network_properties->alpha = solveAlphaCompact(network_properties->N_tar, network_properties->a, network_properties->tau0, network_properties->delta);
					else
						network_properties->alpha = solveAlphaFlat(network_properties->N_tar, network_properties->a, network_properties->chi_max, network_properties->tau0, network_properties->delta);
				}

				//END COMPACT EQUATIONS
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
				network_properties->rhoM = 1.0 / POW2(network_properties->a * SINH(1.5 * network_properties->tau0, STL), EXACT);

			if (DEBUG) {
				assert (network_properties->rhoL > 0.0);
				assert (network_properties->rhoM > 0.0);
			}

			//Make sure tau_m < tau_0 (if applicable)
			if (network_properties->flags.calc_deg_field && network_properties->tau_m >= network_properties->tau0)
				throw CausetException("You have chosen to measure the degree fields at a time greater than the maximum time!\n");

			//Finally, solve for R0
			if (DEBUG) {
				assert (network_properties->alpha > 0.0);
				assert (network_properties->ratio > 0.0);
			}

			network_properties->R0 = network_properties->alpha * POW(network_properties->ratio, 1.0 / 3.0, STL);

			if (DEBUG) assert (network_properties->R0 > 0.0);
			
			if (network_properties->k_tar == 0.0) {
				int method = 1;	//Use lookup table
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->a, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->seed, cp->sCalcDegrees, bm->bCalcDegrees, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network_properties->flags.verbose, network_properties->flags.bench, method))
					return false;
			}
			
			//20% Buffer
			network_properties->edge_buffer = static_cast<int>(0.1 * network_properties->N_tar * network_properties->k_tar);

			//Adjacency matrix not implemented in GPU algorithms
			#ifdef CUDA_ENABLED
			if (network_properties->flags.use_gpu && !LINK_NODES_GPU_V2)
				network_properties->core_edge_fraction = 0.0;
			#endif
				
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

//Calculate Expected Average Degree
//See Causal Set Notes for detailed explanation of methods
bool solveExpAvgDegree(float &k_tar, double &a, double &tau0, const double &alpha, const double &delta, long &seed, Stopwatch &sCalcDegrees, double &bCalcDegrees, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench, const int method)
{
	if (DEBUG) {
		//Variables in correct ranges
		assert (a > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
		assert (delta > 0.0);
		assert (method == 0 || method == 1 || method == 2);
	}

	printf("Estimating Expected Average Degree...\n");
	fflush(stdout);

	double *table;
	long size = 0L;

	int nb = static_cast<int>(bench) * NBENCH;
	int i;

	if (method == 0) {
		//Method 1 of 3: Use Monte Carlo integration to evaluate Kostia's formula
		double r0;
		if (tau0 > LOG(MTAU, STL) / 3.0)
			r0 = POW(0.5, 2.0 / 3.0, STL) * exp(tau0);
		else
			r0 = POW(SINH(1.5 * tau0, STL), 2.0 / 3.0, STL);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (tau0 > LOG(MTAU, STL) / 3.0)
				k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, NULL, seed, 0) * 16.0 * M_PI * exp(-3.0 * tau0);
			else
				k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, NULL, seed, 0) * 8.0 * M_PI / (SINH(3.0 * tau0, STL) - 3.0 * tau0);
			stopwatchStop(&sCalcDegrees);
		}	
	} else if (method == 1) {
		//Method 2 of 3: Lookup table to approximate method 1
		if (!getLookupTable("./etc/raduc_table.cset.bin", &table, &size))
			return false;

		k_tar = lookupValue(table, size, &tau0, NULL, true) * delta * POW2(POW2(a, EXACT), EXACT);
		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			lookupValue(table, size, &tau0, NULL, true);
			stopwatchStop(&sCalcDegrees);
		}	

		//Check for NaN
		if (k_tar != k_tar)
			return false;

		free(table);
		table = NULL;
	} else if (method == 2) {
		//Method 3 of 3: Will's formulation
		if (!getLookupTable("./etc/ctuc_table.cset.bin", &table, &size))
			return false;

		double *params = (double*)malloc(size + sizeof(double) * 3);
		if (params == NULL)
			throw std::bad_alloc();
		hostMemUsed += size + sizeof(double) * 3;

		double d_size = static_cast<double>(size);
		memcpy(params, &a, sizeof(double));
		memcpy(params + 1, &alpha, sizeof(double));
		memcpy(params + 2, &d_size, sizeof(double));
		memcpy(params + 3, table, size);

		IntData idata = IntData();
		idata.limit = 50;
		idata.tol = 1e-5;
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
		idata.upper = tau0 * a;

		double *params2 = &a;
		double max_time = integrate1D(&tToEtaUniverse, (void*)params2, &idata, QAGS) / alpha;

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate1D(&tToEtaUniverse, (void*)params2, &idata, QAGS);
			stopwatchStop(&sCalcDegrees);
		}

		gsl_integration_workspace_free(idata.workspace);

		k_tar = integrate2D(&averageDegreeUniverse, 0.0, 0.0, max_time, max_time, params, seed, 0);
		k_tar *= 4.0 * M_PI * delta * POW2(POW2(alpha, EXACT), EXACT);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate2D(&averageDegreeUniverse, 0.0, 0.0, max_time, max_time, params, seed, 0);
			stopwatchStop(&sCalcDegrees);
		}
		
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
		idata.upper = max_time;
		k_tar /= (3.0 * integrate1D(&psi, params, &idata, QAGS));

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate1D(&psi, params, &idata, QAGS);
			stopwatchStop(&sCalcDegrees);
		}

		gsl_integration_workspace_free(idata.workspace);

		free(params);
		params = NULL;
		hostMemUsed -= size + sizeof(double) * 3;

		free(table);
		table = NULL;
	}

	if (nb)
		bCalcDegrees = sCalcDegrees.elapsedTime / NBENCH;

	if (!bench) {
		printf("\tExpected Average Degree Successfully Calculated.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sCalcDegrees.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Allocates memory for network
//O(1) Efficiency
bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &link, const bool &relink, const bool &verbose, const bool &bench, const bool &yes)
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

	bool links_exist = link || relink;

	if (verbose && !yes) {
		//Estimate memory usage before allocating
		size_t mem = 0;
		if (dim == 3)
			mem += sizeof(float) * N_tar << 2;	//For Coordinate4D
		else if (dim == 1)
			mem += sizeof(float) * N_tar << 1;	//For Coordinate2D
		if (manifold == HYPERBOLIC)
			mem += sizeof(int) * N_tar;		//For AS
		else if (manifold == DE_SITTER)
			mem += sizeof(float) * N_tar;		//For tau
		if (links_exist) {
			mem += sizeof(int) * (N_tar << 1);	//For k_in and k_out
			mem += sizeof(int) * (N_tar * k_tar / 2 + edge_buffer) * 2;	//For edge lists
			mem += sizeof(int) * (N_tar << 1);	//For edge list pointers
			mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);	//For adjacency list
		}

		size_t dmem = 0;
		#ifdef CUDA_ENABLED
		if (use_gpu) {
			size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar / 2 + edge_buffer)));
			mem += sizeof(uint64_t) * d_edges_size;	//For encoded edge list
			mem += sizeof(int);			//For g_idx

			size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (2 * BLOCK_SIZE * GROUP_SIZE)));
			size_t mthread_size = mblock_size * BLOCK_SIZE;
			size_t m_edges_size = mthread_size * mthread_size;
			mem += sizeof(int) * mthread_size * NBUFFERS << 1;		//For k_in and k_out buffers (host)
			mem += sizeof(bool) * m_edges_size * NBUFFERS;			//For adjacency matrix buffers (host)
			dmem += sizeof(float) * mthread_size * 4 * NBUFFERS << 1;	//For 4-D coordinate buffers
			dmem += sizeof(int) * mthread_size * NBUFFERS << 1;		//For k_in and k_out buffers (device)
			dmem += sizeof(bool) * m_edges_size * NBUFFERS;			//For adjacency matrix buffers (device)
		}
		#endif

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
			nodes.crd = new Coordinates4D();

			nodes.crd->w() = (float*)malloc(sizeof(float) * N_tar);
			nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
			nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);
			nodes.crd->z() = (float*)malloc(sizeof(float) * N_tar);

			if (nodes.crd->w() == NULL || nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
				throw std::bad_alloc();

			memset(nodes.crd->w(), 0, sizeof(float) * N_tar);
			memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
			memset(nodes.crd->y(), 0, sizeof(float) * N_tar);
			memset(nodes.crd->z(), 0, sizeof(float) * N_tar);

			hostMemUsed += sizeof(float) * N_tar * 4;
		} else if (dim == 1) {
			nodes.crd = new Coordinates2D();

			nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
			nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);

			if (nodes.crd->x() == NULL || nodes.crd->y() == NULL)
				throw std::bad_alloc();

			memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
			memset(nodes.crd->y(), 0, sizeof(float) * N_tar);

			hostMemUsed += sizeof(float) * N_tar * 2;
		}

		if (links_exist) {
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
		}

		if (verbose)
			printMemUsed("for Nodes", hostMemUsed, devMemUsed);

		if (links_exist) {
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
		}

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

			double *param = (double*)malloc(sizeof(double));
			param[0] = a;

			zeta = HALF_PI - integrate1D(&tToEtaUniverse, (void*)param, &idata, QAGS) / alpha;

			gsl_integration_workspace_free(idata.workspace);
			free(param);
		} else
			//Exact Solution
			zeta = HALF_PI - tauToEtaUniverseExact(tau0, a, alpha);

		printf_cyan();
		printf("\t\tMaximum Conformal Time: %5.8f\n", HALF_PI - zeta);
		printf("\t\tMaximum Rescaled Time:  %5.8f\n", tau0);
		printf_std();
		fflush(stdout);
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

		int p3[2];
		p3[0] = N_tar;
		p3[1] = dim;

		if (!newton(&solveZeta, &x, 10000, TOL, NULL, &k_tar, p3))
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
bool generateNodes(Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//Values are in correct ranges
		assert (!nodes.crd->isNull());
		assert (N_tar > 0);
		assert (k_tar > 0.0);
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
		if (universe) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->x() != NULL);
			assert (nodes.crd->y() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (dim == 3);
			assert (chi_max > 0.0);
			assert (tau0 > 0.0);
		} else
			assert (zeta > 0.0 && zeta < HALF_PI);
	}

	IntData idata = IntData();
	double *param = NULL;

	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-4;

	if (USE_GSL && universe) {
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
		param = (double*)malloc(sizeof(double));
	}

	stopwatchStart(&sGenerateNodes);

	//Generate coordinates for each of N nodes
	double x, rval;
	int i;
	for (i = 0; i < N_tar; i++) {
		////////////////////////////////////////////////////////////
		//~~~~~~~~~~~~~~~~~~~~~~~~~Theta3~~~~~~~~~~~~~~~~~~~~~~~~~//
		//Sample Theta3 from (0, 2pi), as described on p. 2 of [1]//
		////////////////////////////////////////////////////////////

		x = TWO_PI * ran2(&seed);
		if (DEBUG) assert (x > 0.0 && x < TWO_PI);
		//if (i % NPRINT == 0) printf("Theta3: %5.5f\n", x); fflush(stdout);

		if (dim == 1) {
			nodes.crd->y(i) = static_cast<float>(x);

			/////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~Eta~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (2) of [2]//
			/////////////////////////////////////////////////

			nodes.crd->x(i) = static_cast<float>(ATAN(ran2(&seed) / TAN(zeta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			if (DEBUG) {
				assert (nodes.crd->x(i) > 0.0f);
				assert (nodes.crd->x(i) < static_cast<float>(HALF_PI - zeta));
			}

			nodes.id.tau[i] = static_cast<float>(etaToTau(static_cast<double>(nodes.crd->x(i))));
		} else if (dim == 3) {
			nodes.crd->z(i) = static_cast<float>(x);

			/////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (6) of [2] for 3+1//
			//and from PDF identified in (12) of [2] for universe  //
			/////////////////////////////////////////////////////////

			nodes.id.tau[i] = static_cast<float>(tau0) + 1.0f;
			rval = ran2(&seed);

			double p1[2];
			p1[1] = rval;

			if (universe) {
				x = 0.5;
				p1[0] = tau0;
				if (tau0 > 1.8) {	//Cutoff of 1.8 determined by trial and error
					if (!bisection(&solveTauUnivBisec, &x, 2000, 0.0, tau0, TOL, true, p1, NULL, NULL))
						return false;
				} else {
					if (!newton(&solveTauUniverse, &x, 1000, TOL, p1, NULL, NULL))
						return false;
				}
			} else {
				x = 3.5;
				p1[0] = zeta;
				if (!newton(&solveTau, &x, 1000, TOL, p1, NULL, NULL))
					return false;
			}

			nodes.id.tau[i] = static_cast<float>(x);

			if (DEBUG) {
				assert (nodes.id.tau[i] > 0.0f);
				assert (nodes.id.tau[i] < static_cast<float>(tau0));
			}

			//Save eta values as well
			if (universe) {
				if (USE_GSL) {
					//Numerical Integration
					idata.upper = static_cast<double>(nodes.id.tau[i]) * a;
					param[0] = a;
					nodes.crd->w(i) = static_cast<float>(integrate1D(&tToEtaUniverse, (void*)param, &idata, QAGS) / alpha);
				} else
					//Exact Solution
					nodes.crd->w(i) = static_cast<float>(tauToEtaUniverseExact(nodes.id.tau[i], a, alpha));
			} else
				nodes.crd->w(i) = static_cast<float>(tauToEta(static_cast<double>(nodes.id.tau[i])));

			if (DEBUG) {
				assert (nodes.crd->w(i) > 0.0);
				assert (nodes.crd->w(i) < tauToEtaUniverseExact(tau0, a, alpha));
			}
				
			///////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~Theta1 and Theta2~~~~~~~~~~~~~~~~~~//	
			//CDFs derived from PDFs identified on p. 3 of [2]   //
			//Phi given by [3]				     //
			///////////////////////////////////////////////////////

			if (compact) {
				//Sample Theta1 from (0, pi)
				x = HALF_PI;
				rval = ran2(&seed);
				if (!newton(&solveTheta1, &x, 250, TOL, &rval, NULL, NULL))
					return false;
				nodes.crd->x(i) = static_cast<float>(x);
				if (DEBUG) assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < static_cast<float>(M_PI));
			} else {
				nodes.crd->x(i) = static_cast<float>(POW(ran2(&seed), 1.0 / 3.0, APPROX ? FAST : STL) * chi_max);
				if (DEBUG) assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < static_cast<float>(chi_max));
			}
			//if (i % NPRINT == 0) printf("Theta1: %5.5f\n", nodes.crd->x(i)); fflush(stdout);

			//Sample Theta2 from (0, pi)
			nodes.crd->y(i) = static_cast<float>(ACOS(1.0 - 2.0 * ran2(&seed), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			if (DEBUG) assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < static_cast<float>(M_PI));
			//if (i % NPRINT == 0) printf("Theta2: %5.5f\n", nodes.crd->y(i)); fflush(stdout);
		}
		//if (i % NPRINT == 0) printf("eta: %E\n", nodes.crd->w(i));
		//if (i % NPRINT == 0) printf("tau: %E\n", nodes.id.tau[i]);
	}

	//Debugging statements used to check coordinate distributions
	/*if (!printValues(nodes, N_tar, "tau_dist.cset.dbg.dat", "tau")) return false;
	if (!printValues(nodes, N_tar, "eta_dist.cset.dbg.dat", "eta")) return false;
	if (!printValues(nodes, N_tar, "theta1_dist.cset.dbg.dat", "theta1")) return false;
	if (!printValues(nodes, N_tar, "theta2_dist.cset.dbg.dat", "theta2")) return false;
	if (!printValues(nodes, N_tar, "theta3_dist.cset.dbg.dat", "theta3")) return false;
	printf_red();
	printf("Check coordinate distributions now.\n");
	printf_std();
	fflush(stdout);
	exit(EXIT_SUCCESS);*/

	stopwatchStop(&sGenerateNodes);

	if (USE_GSL && universe) {
		gsl_integration_workspace_free(idata.workspace);
		free(param);
	}

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
bool linkNodes(Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//No null pointers
		assert (!nodes.crd->isNull());
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
		assert (core_edge_exists != NULL);

		//Variables in correct ranges
		assert (N_tar > 0);
		assert (k_tar > 0.0f);
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER);
		if (universe) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->x() != NULL);
			assert (nodes.crd->y() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (dim == 3);
			assert (alpha > 0.0);
		}
		assert (a > 0.0);
		assert (zeta > 0.0 && zeta < HALF_PI);
		assert (chi_max > 0.0);
		assert (tau0 > 0.0);
		assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
		assert (edge_buffer >= 0);
	}

	float dt = 0.0f, dx = 0.0f;
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
				dt = nodes.crd->x(j) - nodes.crd->x(i);
			else if (dim == 3)
				dt = nodes.crd->w(j) - nodes.crd->w(i);
			//if (i % NPRINT == 0) printf("dt: %.9f\n", dt); fflush(stdout);
			if (DEBUG) {
				assert (dt >= 0.0f);
				assert (dt <= static_cast<float>(HALF_PI - zeta));
			}

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (dim == 1) {
				//Formula given on p. 2 of [2]
				dx = static_cast<float>(M_PI - ABS(M_PI - ABS(static_cast<double>(nodes.crd->y(j) - nodes.crd->y(i)), STL), STL));
			} else if (dim == 3) {
				//BEGIN COMPACT EQUATIONS (Completed)

				if (compact) {
					//Spherical Law of Cosines
					dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v2(nodes.crd->getFloat4(i), nodes.crd->getFloat4(j))), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
				} else {
					//Distance on Flat Spacetime
					dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v2(nodes.crd->getFloat4(i), nodes.crd->getFloat4(j))), APPROX ? BITWISE : STL));
				}

				//END COMPACT EQUATIONS
			}

			//if (i % NPRINT == 0) printf("dx: %.5f\n", dx); fflush(stdout);
			if (compact) {
				if (DEBUG) assert (dx >= 0.0f && dx <= static_cast<float>(M_PI));
			} else {
				if (DEBUG) assert (dx >= 0.0f && dx <= 2.0f * static_cast<float>(chi_max));
			}

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
					edges.future_edges[future_idx++] = j;
	
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
					edges.past_edges[past_idx++] = j;
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
		assert (N_res >= 0);
		assert (N_deg2 >= 0);
		assert (k_res >= 0.0);
	}

	k_res /= N_res;

	//Debugging options used to visually inspect the adjacency lists and the adjacency pointer lists
	//compareAdjacencyLists(nodes, edges);
	//compareAdjacencyListIndices(nodes, edges);

	//Print Results
	/*if (!printDegrees(nodes, N_tar, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, past_idx, "past-edges_CPU.cset.dbg.dat", "future-edges_CPU.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_CPU.cset.dbg.dat", "future-edge-pointers_CPU.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	exit(EXIT_SUCCESS);*/

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tUndirected Links:         %d\n", future_idx);
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
