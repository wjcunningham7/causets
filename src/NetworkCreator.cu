#include "NetworkCreator.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm)
{
	if (DEBUG) {
		assert (network_properties != NULL);
		assert (ca != NULL);
		assert (cp != NULL);
		assert (bm != NULL);
	}

	//Initialize RNG
	if (network_properties->seed == -12345L) {
		srand(time(NULL));
		network_properties->seed = -1.0 * static_cast<long>(time(NULL));
	}

	//Benchmarking
	if (network_properties->flags.bench) {
		network_properties->graphID = 0;
		network_properties->flags.verbose = false;
		network_properties->flags.print_network = false;
	}

	int rank = network_properties->cmpi.rank;

	//Suppress queries if MPI is enabled
	#ifdef MPI_ENABLED
	if (network_properties->flags.verbose)
		network_properties->flags.yes = true;
	#endif

	//If a graph ID has been provided, warn user
	if (network_properties->graphID && network_properties->flags.verbose && !network_properties->flags.yes) {
		printf("You have chosen to load a graph from memory. Some parameters may be ignored as a result. Continue [y/N]? ");
		fflush(stdout);
		char response = getchar();
		getchar();
		if (response != 'y')
			return false;
	}

	//If the GPU is requested, optimize parameters
	#ifdef CUDA_ENABLED
	if (network_properties->flags.use_gpu && network_properties->N_tar % (BLOCK_SIZE << 1)) {
		printf_mpi(rank, "If you are using the GPU, set the target number of nodes (--nodes) to be a multiple of %d!\n", BLOCK_SIZE << 1);
		fflush(stdout);
		network_properties->cmpi.fail = 1;
	}

	if (checkMpiErrors(network_properties->cmpi))
		return false;

	//Adjacency matrix not implemented in certain GPU algorithms
	if (network_properties->flags.use_gpu && !LINK_NODES_GPU_V2)
		network_properties->core_edge_fraction = 0.0;
	#endif

	try {
		if (network_properties->manifold == DE_SITTER || network_properties->manifold == FLRW) {
			//Check for under-constrained system
			if (network_properties->N_tar == 0)
				throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
			if (network_properties->tau0 == 0.0)
				throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");

			//Initialize certain variables
			if (network_properties->delta == 0.0)
				network_properties->delta = 1000;
			
		}

		if (network_properties->manifold == DE_SITTER) {
			//Constrain the de Sitter system
			network_properties->zeta = HALF_PI - network_properties->tau0;
			network_properties->tau0 = etaToTau(HALF_PI - network_properties->zeta);

			if (DEBUG) {
				assert (network_properties->zeta > 0.0 && network_properties->zeta < HALF_PI);
				assert (network_properties->tau0 > 0.0);
			}

			double eta0 = HALF_PI - network_properties->zeta;
			if (network_properties->dim == 1) {
				network_properties->k_tar = network_properties->N_tar * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL) - 1.0) / (TAN(eta0, STL) * HALF_PI);
				network_properties->a = SQRT(network_properties->N_tar / (TWO_PI * network_properties->delta * TAN(eta0, STL)), STL);
			} else if (network_properties->dim == 3) {
				network_properties->k_tar = network_properties->N_tar * (12.0 * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL)) - (6.0 * LOG(COS(eta0, STL), STL) + 5.0) / POW2(COS(eta0, STL), EXACT) - 7.0) / (POW2(2.0 + 1.0 / POW2(COS(eta0, STL), EXACT), EXACT) * TAN(eta0, STL) * 3.0 * HALF_PI);
				network_properties->a = POW(network_properties->N_tar * 3.0 / (2.0 * POW2(M_PI, EXACT) * network_properties->delta * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL)), 1.0 / 4.0, STL);
				//printf("N_tar: %d\n", network_properties->N_tar);
				//printf("k_tar: %.6f\n", network_properties->k_tar);
				//printf("delta: %f\n", network_properties->delta);
				//printf("a: %.6f\n", network_properties->a);
			}

			if (DEBUG) {
				assert (network_properties->k_tar > 0.0);
				assert (network_properties->a > 0.0);
			}

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining %d+1 de Sitter Causal Set:\n", network_properties->dim);
			printf_mpi(rank, "\t--------------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
			printf_mpi(rank, "\t > Node Density: \t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Pseudoradius:\t\t%.6f\n", network_properties->a);
			if (!rank) printf_std();
			fflush(stdout);

			//Miscellaneous Tasks
			network_properties->flags.compact = true;

			if (!network_properties->cmpi.rank && network_properties->flags.gen_ds_table && !generateGeodesicLookupTable("geodesics_ds_table.cset.bin", 5.0, -5.0, 5.0, 0.01, 0.01, network_properties->manifold, network_properties->flags.verbose))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
		} else if (network_properties->manifold == FLRW) {
			//Check for under-constrained system (specific to FLRW)
			if (network_properties->alpha == 0.0)
				throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
			if (network_properties->dim == 1)
				throw CausetException("Flag '--dim', spatial dimension, must be (3) in FLRW spacetime!\n");

			//Constrain the FLRW system
			if (network_properties->flags.compact) {
				double q = 3.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * POW3(network_properties->alpha, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
				printf("N: %d\n", network_properties->N_tar);
				printf("alpha: %f\n", network_properties->alpha);
				printf("tau0: %f\n", network_properties->tau0);
				network_properties->a = POW(q / network_properties->delta, 1.0 / 4.0, STL);
				//\tilde{\alpha} -> \alpha
				network_properties->alpha *= network_properties->a;
				//Use lookup table to solve for k_tar
				int method = 1;
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->dim, network_properties->manifold, network_properties->a, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->seed, network_properties->cmpi.rank, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.verbose, network_properties->flags.bench, method))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;

			} else {
				//Non-Compact FLRW Constraints
			}

			if (DEBUG) {
				assert (network_properties->a > 0.0);
				assert (network_properties->k_tar > 0.0);
			}

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining the FLRW Causal Set:\n");
			printf_mpi(rank, "\t--------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			if (!rank) printf_red();
			printf_mpi(rank, "\t > Dark Energy Density:\t\t%.6f\n", network_properties->omegaL);
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Max. Rescaled Time:\t\t%.6f\n", network_properties->tau0);
			printf_mpi(rank, "\t > Spatial Scaling:\t\t%.6f\n", network_properties->alpha);
			printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
			if (!rank) printf_std();
			fflush(stdout);

			//Miscellaneous Tasks
			network_properties->zeta = HALF_PI - tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);

			if (DEBUG)
				assert (HALF_PI - network_properties->zeta > 0.0);

			if (!network_properties->cmpi.rank && network_properties->flags.gen_flrw_table && !generateGeodesicLookupTable("geodesics_flrw_table.cset.bin", 2.0, -5.0, 5.0, 0.01, 0.01, network_properties->manifold, network_properties->flags.verbose))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
		} else if (network_properties->manifold == HYPERBOLIC) {
			if (network_properties->dim != 1)
				throw CausetException("You must use --dim 1 for a hyperbolic manifold!\n");
			if (network_properties->zeta == 0.0)
				network_properties->zeta = 1.0;
		}

		//Miscellaneous Tasks
		if (network_properties->edge_buffer == 0.0)
			network_properties->edge_buffer = 0.2;

		if (network_properties->flags.calc_deg_field && network_properties->tau_m >= network_properties->tau0)
			throw CausetException("You have chosen to measure the degree field at a time greater than the maximum time!\n");
		if (network_properties->flags.calc_action && network_properties->max_cardinality >= network_properties->N_tar)
			throw CausetException("Maximum cardinality (specified by --action) must be less than N.\n");
		
		uint64_t pair_multiplier = static_cast<uint64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;
		if (network_properties->flags.calc_success_ratio && network_properties->N_sr <= 1.0)
			network_properties->N_sr *= pair_multiplier;
		if (network_properties->flags.validate_embedding && network_properties->N_emb <= 1.0)
			network_properties->N_emb *= pair_multiplier;
		if (network_properties->flags.validate_distances && network_properties->N_dst <= 1.0)
			network_properties->N_dst *= pair_multiplier;
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		network_properties->cmpi.fail = 1;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		network_properties->cmpi.fail = 1;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		network_properties->cmpi.fail = 1;
	}

	if (checkMpiErrors(network_properties->cmpi))
		return false;

	return true;
}

//Calculate Expected Average Degree in Compact FLRW Spacetime
//See Causal Set Notes for detailed explanation of methods
bool solveExpAvgDegree(float &k_tar, const int &dim, const Manifold &manifold, double &a, double &tau0, const double &alpha, const double &delta, long &seed, const int &rank, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &verbose, const bool &bench, const int method)
{
	if (DEBUG) {
		assert (ca != NULL);
		assert (dim == 3);
		assert (manifold == FLRW);
		assert (a > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
		assert (delta > 0.0);
		assert (method == 0 || method == 1 || method == 2);
	}

	printf_mpi(rank, "\tEstimating Expected Average Degree...\n");
	fflush(stdout);

	int nb = static_cast<int>(bench) * NBENCH;
	int i;

	double *table;
	long size = 0L;

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
				k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeFLRW, 0.0, 0.0, r0, r0, NULL, seed, 0) * 16.0 * M_PI * exp(-3.0 * tau0);
			else
				k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeFLRW, 0.0, 0.0, r0, r0, NULL, seed, 0) * 8.0 * M_PI / (SINH(3.0 * tau0, STL) - 3.0 * tau0);
			stopwatchStop(&sCalcDegrees);
		}	
	} else if (method == 1) {
		//Method 2 of 3: Lookup table to approximate method 1
		if (!getLookupTable("./etc/raduc_table.cset.bin", &table, &size))
			return false;
		ca->hostMemUsed += size;

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
		ca->hostMemUsed -= size;
	} else if (method == 2) {
		//Method 3 of 3: Will's formulation
		if (!getLookupTable("./etc/ctuc_table.cset.bin", &table, &size))
			return false;
		ca->hostMemUsed += size;

		double *params = (double*)malloc(size + sizeof(double) * 3);
		if (params == NULL)
			throw std::bad_alloc();
		ca->hostMemUsed += size + sizeof(double) * 3;

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
		double max_time = integrate1D(&tToEtaFLRW, (void*)params2, &idata, QAGS) / alpha;

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate1D(&tToEtaFLRW, (void*)params2, &idata, QAGS);
			stopwatchStop(&sCalcDegrees);
		}

		gsl_integration_workspace_free(idata.workspace);

		k_tar = integrate2D(&averageDegreeFLRW, 0.0, 0.0, max_time, max_time, params, seed, 0);
		k_tar *= 4.0 * M_PI * delta * POW2(POW2(alpha, EXACT), EXACT);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate2D(&averageDegreeFLRW, 0.0, 0.0, max_time, max_time, params, seed, 0);
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
		ca->hostMemUsed -= size + sizeof(double) * 3;

		free(table);
		table = NULL;
		ca->hostMemUsed -= size;
	}

	if (nb)
		bCalcDegrees = sCalcDegrees.elapsedTime / NBENCH;

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sCalcDegrees.elapsedTime);
		fflush(stdout);
	}

	if (!bench) {
		printf_mpi(rank, "\tExpected Average Degree Successfully Calculated.\n");
		printf_mpi(rank, "\t\t<k> = %f\n", k_tar);
		fflush(stdout);
	}

	return true;
}

//Allocates memory for network
//O(1) Efficiency
bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &link, const bool &relink, const bool &verbose, const bool &bench, const bool &yes)
{
	if (DEBUG) {
		assert (ca != NULL);
		assert (N_tar > 0);
		assert (k_tar > 0.0f);
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == FLRW || manifold == HYPERBOLIC);
		if (manifold == HYPERBOLIC)
			assert (dim == 1);
		assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
		assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	}

	int rank = cmpi.rank;
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
		else if (manifold == DE_SITTER || manifold == FLRW)
			mem += sizeof(float) * N_tar;		//For tau
		if (links_exist) {
			mem += sizeof(int) * (N_tar << 1);	//For k_in and k_out
			mem += sizeof(int) * static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer));	//For edge lists
			mem += sizeof(int) * (N_tar << 1);	//For edge list pointers
			mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);	//For adjacency list
		}

		size_t dmem = 0;
		#ifdef CUDA_ENABLED
		if (use_gpu) {
			size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar * (1.0 + edge_buffer) / 2)));
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

		printMemUsed("for Network (Estimation)", mem, dmem, 0);
		printf("\nContinue [y/N]?");
		fflush(stdout);
		char response = getchar();
		getchar();
		if (response != 'y')
			return false;
	}

	stopwatchStart(&sCreateNetwork);

	try {
		if (manifold == DE_SITTER || manifold == FLRW) {
			nodes.id.tau = (float*)malloc(sizeof(float) * N_tar);
			if (nodes.id.tau == NULL)
				throw std::bad_alloc();
			memset(nodes.id.tau, 0, sizeof(float) * N_tar);
			ca->hostMemUsed += sizeof(float) * N_tar;
		} else if (manifold == HYPERBOLIC) {
			nodes.id.AS = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.id.AS == NULL)
				throw std::bad_alloc();
			memset(nodes.id.AS, 0, sizeof(int) * N_tar);
			ca->hostMemUsed += sizeof(int) * N_tar;
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

			ca->hostMemUsed += sizeof(float) * N_tar * 4;
		} else if (dim == 1) {
			nodes.crd = new Coordinates2D();

			nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
			nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);

			if (nodes.crd->x() == NULL || nodes.crd->y() == NULL)
				throw std::bad_alloc();

			memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
			memset(nodes.crd->y(), 0, sizeof(float) * N_tar);

			ca->hostMemUsed += sizeof(float) * N_tar * 2;
		}

		if (links_exist) {
			nodes.k_in = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.k_in == NULL)
				throw std::bad_alloc();
			memset(nodes.k_in, 0, sizeof(int) * N_tar);
			ca->hostMemUsed += sizeof(int) * N_tar;

			nodes.k_out = (int*)malloc(sizeof(int) * N_tar);
			if (nodes.k_out == NULL)
				throw std::bad_alloc();
			memset(nodes.k_out, 0, sizeof(int) * N_tar);
			ca->hostMemUsed += sizeof(int) * N_tar;
		}

		if (verbose)
			printMemUsed("for Nodes", ca->hostMemUsed, ca->devMemUsed, rank);

		if (links_exist) {
			edges.past_edges = (int*)malloc(sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
			if (edges.past_edges == NULL)
				throw std::bad_alloc();
			memset(edges.past_edges, 0, sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
			ca->hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2);

			edges.future_edges = (int*)malloc(sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
			if (edges.future_edges == NULL)
				throw std::bad_alloc();
			memset(edges.future_edges, 0, sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
			ca->hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2);

			edges.past_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
			if (edges.past_edge_row_start == NULL)
				throw std::bad_alloc();
			memset(edges.past_edge_row_start, 0, sizeof(int) * N_tar);
			ca->hostMemUsed += sizeof(int) * N_tar;
	
			edges.future_edge_row_start = (int*)malloc(sizeof(int) * N_tar);
			if (edges.future_edge_row_start == NULL)
				throw std::bad_alloc();
			memset(edges.future_edge_row_start, 0, sizeof(int) * N_tar);
			ca->hostMemUsed += sizeof(int) * N_tar;

			core_edge_exists = (bool*)malloc(sizeof(bool) * static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT)));
			if (core_edge_exists == NULL)
				throw std::bad_alloc();
			memset(core_edge_exists, 0, sizeof(bool) * static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT)));
			ca->hostMemUsed += sizeof(bool) * static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT));
		}

		memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
		if (verbose)
			printMemUsed("for Network", ca->hostMemUsed, ca->devMemUsed, rank);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		cmpi.fail = 1;
	}

	if (checkMpiErrors(cmpi))
		return false;

	stopwatchStop(&sCreateNetwork);

	if (!bench) {
		printf_mpi(rank, "\tMemory Successfully Allocated.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sCreateNetwork.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Poisson Sprinkling
//O(N) Efficiency
bool generateNodes(Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &compact, const bool &verbose, const bool &bench)
{
	if (DEBUG) {
		//Values are in correct ranges
		assert (!nodes.crd->isNull());
		assert (N_tar > 0);
		assert (k_tar > 0.0f);
		assert (dim == 1 || dim == 3);
		assert (manifold == DE_SITTER || manifold == FLRW);
		assert (a >= 0.0);
		assert (tau0 > 0.0);
		if (manifold == FLRW) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->x() != NULL);
			assert (nodes.crd->y() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (dim == 3);
			assert (chi_max > 0.0);
		} else if (manifold == DE_SITTER)
			assert (zeta > 0.0 && zeta < HALF_PI);
	}

	IntData idata = IntData();
	double *param = NULL;

	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-4;

	if (USE_GSL && manifold == FLRW) {
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

			do nodes.crd->x(i) = static_cast<float>(ATAN(ran2(&seed) / TAN(zeta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			while (nodes.crd->x(i) >= static_cast<float>(HALF_PI - zeta));

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
			//and from PDF identified in (12) of [2] for FLRW      //
			/////////////////////////////////////////////////////////

			do {
				rval = ran2(&seed);

				double p1[2];
				p1[1] = rval;

				if (manifold == FLRW) {
					x = 0.5;
					p1[0] = tau0;
					if (tau0 > 1.8) {	//Cutoff of 1.8 determined by trial and error
						if (!bisection(&solveTauUnivBisec, &x, 2000, 0.0, tau0, TOL, true, p1, NULL, NULL))
							return false;
					} else {
						if (!newton(&solveTauUniverse, &x, 1000, TOL, p1, NULL, NULL))
							return false;
					}
				} else if (manifold == DE_SITTER) {
					x = 3.5;
					p1[0] = zeta;
					if (!newton(&solveTau, &x, 1000, TOL, p1, NULL, NULL))
						return false;
				}

				nodes.id.tau[i] = static_cast<float>(x);
			} while (nodes.id.tau[i] >= static_cast<float>(tau0));

			if (DEBUG) {
				assert (nodes.id.tau[i] > 0.0f);
				assert (nodes.id.tau[i] < static_cast<float>(tau0));
			}

			//Save eta values as well
			if (manifold == FLRW) {
				if (USE_GSL) {
					//Numerical Integration
					idata.upper = static_cast<double>(nodes.id.tau[i]) * a;
					param[0] = a;
					nodes.crd->w(i) = static_cast<float>(integrate1D(&tToEtaFLRW, (void*)param, &idata, QAGS) / alpha);
				} else
					//Exact Solution
					nodes.crd->w(i) = static_cast<float>(tauToEtaFLRWExact(nodes.id.tau[i], a, alpha));

				if (DEBUG) assert (nodes.crd->w(i) < tauToEtaFLRWExact(tau0, a, alpha));
			} else if (manifold == DE_SITTER) {
				nodes.crd->w(i) = static_cast<float>(tauToEta(static_cast<double>(nodes.id.tau[i])));
				if (DEBUG) assert (nodes.crd->w(i) < tauToEta(tau0));
			}
			if (DEBUG) assert (nodes.crd->w(i) > 0.0);
				
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
	exit(0);*/

	stopwatchStop(&sGenerateNodes);

	if (USE_GSL && manifold == FLRW) {
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
bool linkNodes(Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &compact, const bool &verbose, const bool &bench)
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
		assert (manifold == DE_SITTER || manifold == FLRW);
		if (manifold == FLRW) {
			assert (nodes.crd->getDim() == 4);
			assert (nodes.crd->w() != NULL);
			assert (nodes.crd->x() != NULL);
			assert (nodes.crd->y() != NULL);
			assert (nodes.crd->z() != NULL);
			assert (dim == 3);
			assert (alpha > 0.0);
			if (!compact)
				assert (chi_max > 0.0);
		}
		assert (zeta > 0.0 && zeta < HALF_PI);
		assert (tau0 > 0.0);
		assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
		assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	}

	int core_limit = static_cast<int>((core_edge_fraction * N_tar));
	int future_idx = 0;
	int past_idx = 0;
	int i, j, k;

	bool related;

	stopwatchStart(&sLinkNodes);

	//Identify future connections
	for (i = 0; i < N_tar - 1; i++) {
		if (i < core_limit)
			core_edge_exists[(i*core_limit)+i] = false;
		edges.future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			//Assume nodes are already temporally ordered
			related = nodesAreRelated(nodes.crd, N_tar, dim, manifold, zeta, chi_max, compact, i, j);

			//Core Edge Adjacency Matrix
			if (i < core_limit && j < core_limit) {
				uint64_t idx1 = static_cast<uint64_t>(i) * core_limit + j;
				uint64_t idx2 = static_cast<uint64_t>(j) * core_limit + i;

				if (related) {
					core_edge_exists[idx1] = true;
					core_edge_exists[idx2] = true;
				} else {
					core_edge_exists[idx1] = false;
					core_edge_exists[idx2] = false;
				}
			}
						
			//Link timelike relations
			try {
				if (related) {
					//if (i % NPRINT == 0) printf("%d %d\n", i, j); fflush(stdout);
					edges.future_edges[future_idx++] = j;
	
					if (future_idx >= static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer) / 2))
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
	if (DEBUG && !compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction))
		return false;

	//Print Results
	/*if (!printDegrees(nodes, N_tar, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, past_idx, "past-edges_CPU.cset.dbg.dat", "future-edges_CPU.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_CPU.cset.dbg.dat", "future-edge-pointers_CPU.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	exit(0);*/

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
