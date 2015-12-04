#include "NetworkCreator.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm)
{
	#if DEBUG
	assert (network_properties != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	#endif

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

	//Disable the default GSL Error Handler
	disableGSLErrHandler();

	try {
		if (network_properties->manifold == DE_SITTER || network_properties->manifold == DUST || network_properties->manifold == FLRW) {
			//Check for under-constrained system
			if (network_properties->N_tar == 0)
				throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
			if (network_properties->tau0 == 0.0)
				throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");
			if (network_properties->manifold == DE_SITTER && !network_properties->flags.compact && !network_properties->r_max)
				throw CausetException("Flag '--slice', spatial scaling, must be specified!\n");
		}

		if (network_properties->manifold == DE_SITTER) {
			//Constrain the de Sitter system
			if (!network_properties->delta)
				network_properties->a = 1.0;

			if (network_properties->flags.compact) {
				network_properties->zeta = HALF_PI - network_properties->tau0;
				network_properties->tau0 = etaToTauCompact(HALF_PI - network_properties->zeta);
			} else {
				network_properties->zeta = HALF_PI + 1.0;
				network_properties->zeta1 = HALF_PI - tauToEtaFlat(network_properties->tau0);
			}

			#if DEBUG
			if (network_properties->flags.compact)
				assert (network_properties->zeta > 0.0 && network_properties->zeta < HALF_PI);
			else
				assert (network_properties->zeta > HALF_PI);
			assert (network_properties->tau0 > 0.0);
			#endif

			double eta0 = HALF_PI - network_properties->zeta;
			double eta1 = HALF_PI - network_properties->zeta1;
			if (network_properties->dim == 1) {
				#if DEBUG
				assert (network_properties->flags.compact);
				#endif

				if (network_properties->flags.symmetric) {
					network_properties->k_tar = network_properties->N_tar * (eta0 / (sin(eta0) * cos(eta0)) - 1.0) / (HALF_PI * tan(eta0));
					if (!!network_properties->delta)
						network_properties->a = SQRT(network_properties->N_tar / (4.0 * M_PI * network_properties->delta * TAN(eta0, STL)), STL);
					else
						network_properties->delta = network_properties->N_tar / (4.0 * M_PI * POW2(network_properties->a, EXACT) * TAN(eta0, STL));
				} else {
					network_properties->k_tar = network_properties->N_tar * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL) - 1.0) / (TAN(eta0, STL) * HALF_PI);
					if (!!network_properties->delta)
						network_properties->a = SQRT(network_properties->N_tar / (TWO_PI * network_properties->delta * TAN(eta0, STL)), STL);
					else
						network_properties->delta = network_properties->N_tar / (TWO_PI * POW2(network_properties->a, EXACT) * TAN(eta0, STL));
				}
			} else if (network_properties->dim == 3) {
				if (network_properties->flags.compact) {
					if (network_properties->flags.symmetric) {
						IntData idata;
						idata.limit = 50;
						idata.tol = 1e-5;
						idata.lower = -eta0;
						idata.upper = eta0;
						double t1 = (sin(eta0) + sin(5.0 * eta0)) / (3.0 * POW3(cos(eta0), EXACT));
						double t2 = 2.0 * integrate1D(&averageDegreeSym, NULL, &idata, QNG);
						double t3 = (2.0 * eta0 * eta0 - 1.0) * (3.0 * sin(eta0) + sin(3.0 * eta0)) / (3.0 * POW3(cos(eta0), EXACT));

						//DEBUG
						//double kappa = TWO_PI * (t1 + t2 + t3) / (tan(eta0) * (2.0 + 1.0 / POW2(cos(eta0), EXACT)));
						//printf("kappa: %f\n", kappa);

						network_properties->k_tar = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) * (t1 + t2 + t3) / (M_PI * tan(eta0) * (2.0 + 1.0 / POW2(cos(eta0), EXACT)) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
						if (!!network_properties->delta)
							network_properties->a = POW(3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * network_properties->delta * (3.0 * sin(eta0) + sin(3.0 * eta0))), 0.25, STL);
						else
							network_properties->delta = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
					} else {
						network_properties->k_tar = network_properties->N_tar * (12.0 * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL)) - (6.0 * LOG(COS(eta0, STL), STL) + 5.0) / POW2(COS(eta0, STL), EXACT) - 7.0) / (POW2(2.0 + 1.0 / POW2(COS(eta0, STL), EXACT), EXACT) * TAN(eta0, STL) * 3.0 * HALF_PI);
						if (!!network_properties->delta)
							network_properties->a = POW(network_properties->N_tar * 3.0 / (2.0 * POW2(M_PI, EXACT) * network_properties->delta * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL)), 0.25, STL);
						else
							network_properties->delta = network_properties->N_tar * 3.0 / (2.0 * POW2(M_PI * POW2(network_properties->a, EXACT), EXACT) * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL));
					}
				} else {
					int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());

					network_properties->k_tar = 9.0 * network_properties->N_tar * POW2(POW3(eta0 * eta1, EXACT), EXACT) * integrate2D(&rescaledDegreeDeSitterFlat, eta0, eta0, eta1, eta1, NULL, seed, 0) / (POW3(network_properties->r_max, EXACT) * POW2(POW3(eta1, EXACT) - POW3(eta0, EXACT), EXACT));
					if (!!network_properties->delta)
						network_properties->a = POW(9.0 * network_properties->N_tar * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * network_properties->delta * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT))), 0.25, STL);
					else
						network_properties->delta = 9.0 * network_properties->N_tar * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT)));
				}
			}

			#if DEBUG
			assert (network_properties->k_tar > 0.0);
			assert (network_properties->a > 0.0);
			assert (network_properties->delta > 0.0);
			#endif

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining %d+1 de Sitter Causal Set:\n", network_properties->dim);
			printf_mpi(rank, "\t--------------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			if (network_properties->flags.compact) {
				if (network_properties->flags.symmetric)
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", -eta0);
				else
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t0.0\n");
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
			} else {
				printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", eta0);
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta1);
			}
			printf_mpi(rank, "\t > Max. Rescaled  Time:\t\t%.6f\n", network_properties->tau0);
			if (!network_properties->flags.compact)
				printf_mpi(rank, "\t > Spatial Cutoff:\t\t%.6f\n", network_properties->r_max);
			printf_mpi(rank, "\t > Node Density: \t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Pseudoradius:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Random Seed:\t\t\t%Ld\n", network_properties->seed);
			if (!rank) printf_std();
			fflush(stdout);

			//Miscellaneous Tasks
			if (!network_properties->cmpi.rank && network_properties->flags.gen_ds_table && !generateGeodesicLookupTable("geodesics_ds_table.cset.bin", 5.0, -5.0, 5.0, 0.01, 0.01, network_properties->manifold, network_properties->flags.verbose))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
		} else if (network_properties->manifold == DUST) {
			//Check for under-constrained system
			if (!network_properties->alpha)
				throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");

			if (network_properties->dim == 1)
				throw CausetException("Flag '--dim', spatial dimension, must be (3) in Dust spacetime!\n");

			//Constrain the dust system
			if (!network_properties->delta)
				network_properties->delta = 1000;

			//Let alpha characterize the boundary effects
			if (!network_properties->r_max)
				network_properties->r_max = 1.0;

			//Dust Constraints
			int method = 0;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->dim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;

			double q = network_properties->N_tar / (M_PI * POW3(network_properties->alpha * network_properties->r_max * network_properties->tau0, EXACT));
			network_properties->a = POW(q / network_properties->delta, 0.25, STL);
			network_properties->alpha *= network_properties->a;

			network_properties->zeta = HALF_PI - tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);

			#if DEBUG
			assert (network_properties->a > 0.0);
			assert (network_properties->k_tar > 0.0);
			assert (network_properties->zeta < HALF_PI);
			#endif

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining the Dusty Causal Set:\n");
			printf_mpi(rank, "\t---------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Number of Expected Nodes:\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			printf_mpi(rank, "\t > Max. Rescaled Time:\t\t%.6f\n", network_properties->tau0);
			printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", HALF_PI - network_properties->zeta);
			printf_mpi(rank, "\t > Spatial Scaling:\t\t%.6f\n", network_properties->alpha);
			printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Random Seed:\t\t\t%Ld\n", network_properties->seed);
			if (!rank) printf_std();
			fflush(stdout);
		} else if (network_properties->manifold == FLRW) {
			//Check for under-constrained system
			if (!network_properties->alpha)
				throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
			if (network_properties->dim == 1)
				throw CausetException("Flag '--dim', spatial dimension, must be (3) in FLRW spacetime!\n");

			//Constrain the FLRW system
			if (!network_properties->delta)
				network_properties->delta = 1000;

			if (network_properties->flags.compact) {
				double q = 3.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * POW3(network_properties->alpha, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
				network_properties->a = POW(q / network_properties->delta, 1.0 / 4.0, STL);
				//\tilde{\alpha} -> \alpha
				network_properties->alpha *= network_properties->a;
				//Use lookup table to solve for k_tar
				int method = 1;
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->dim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			} else {
				//This makes alpha characterize the boundary effects
				if (!network_properties->r_max)
					network_properties->r_max = 1.0;

				//Non-Compact FLRW Constraints
				int method = 0;
				//int method = 1;
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->dim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
				
				double q = 9.0 * network_properties->N_tar / (TWO_PI * POW3(network_properties->alpha * network_properties->r_max, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
				network_properties->a = POW(q / network_properties->delta, 0.25, STL);
				network_properties->alpha *= network_properties->a;
			}

			network_properties->zeta = HALF_PI - tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);

			#if DEBUG
			assert (network_properties->a > 0.0);
			assert (network_properties->k_tar > 0.0);
			assert (network_properties->zeta < HALF_PI);
			#endif

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
			printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", HALF_PI - network_properties->zeta);
			printf_mpi(rank, "\t > Spatial Scaling:\t\t%.6f\n", network_properties->alpha);
			printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Random Seed:\t\t\t%Ld\n", network_properties->seed);
			if (!rank) printf_std();
			fflush(stdout);

			//Miscellaneous Tasks
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

		#ifdef CUDA_ENABLED
		//Determine group size and decoding method
		if (network_properties->flags.use_gpu) {
			long glob_mem = 5000000000L;
			long mem = glob_mem + 1L;
			long d_edges_size = static_cast<long>(exp2(ceil(log2(network_properties->N_tar * network_properties->k_tar * (1.0 + network_properties->edge_buffer) / 2.0))));
			float gsize = 0.5f;
			bool dcpu = false;

			while (mem > glob_mem) {
				gsize *= 2.0f;
				//long mbsize = static_cast<long>(ceil(static_cast<float>(network_properties->N_tar) / (BLOCK_SIZE * gsize * 2)));
				long mbsize = static_cast<long>(ceil(static_cast<float>(network_properties->N_tar) / (BLOCK_SIZE * gsize)));
				long mtsize = mbsize * BLOCK_SIZE;
				long mesize = mtsize * mtsize;
				long gmbsize = static_cast<long>(network_properties->N_tar * network_properties->k_tar * (1.0 + network_properties->edge_buffer) / (BLOCK_SIZE * gsize * 2));
				long gmtsize = gmbsize * BLOCK_SIZE;

				long mem1 = (40L * mtsize + mesize) * NBUFFERS;
				long mem2 = 4L * (2L * d_edges_size + gmtsize);
				long mem3 = 8L * (network_properties->N_tar + 2L * BLOCK_SIZE);

				if (mem2 > glob_mem / 4L) {
					mem2 = 0L;
					dcpu = true;
				}

				long max = mem1;
				if (mem2 > max) max = mem2;
				if (mem3 > max) max = mem3;
				mem = max;
			}

			network_properties->group_size = gsize < NBUFFERS ? NBUFFERS : gsize;
			network_properties->flags.decode_cpu = dcpu;
		}
		#endif

		if (network_properties->flags.calc_deg_field && network_properties->tau_m >= network_properties->tau0)
			throw CausetException("You have chosen to measure the degree field at a time greater than the maximum time!\n");
		
		uint64_t pair_multiplier = static_cast<uint64_t>(network_properties->N_tar) * (network_properties->N_tar - 1) / 2;
		if (network_properties->flags.calc_success_ratio && network_properties->N_sr <= 1.0)
			network_properties->N_sr *= pair_multiplier;
		if (network_properties->flags.validate_embedding && network_properties->N_emb <= 1.0)
			network_properties->N_emb *= pair_multiplier;
		if (network_properties->flags.validate_distances && network_properties->N_dst <= 1.0)
			network_properties->N_dst *= pair_multiplier;

		if (network_properties->flags.calc_action) {
			#if DEBUG
			assert (network_properties->max_cardinality == -1 || network_properties->max_cardinality == 1);
			#endif
			if (network_properties->max_cardinality == -1)
				network_properties->max_cardinality = 5;
			else
				network_properties->max_cardinality = network_properties->N_tar - 1;
		}
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

//Calculate Expected Average Degree in the Dust or FLRW Spacetime
//See Causal Set Notes for detailed explanation of methods
bool solveExpAvgDegree(float &k_tar, const int &N_tar, const int &dim, const Manifold &manifold, double &a, const double &r_max, double &tau0, const double &alpha, const double &delta, const int &rank, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &compact, const bool &verbose, const bool &bench, const int method)
{
	#if DEBUG
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (dim == 3);
	assert (manifold == DUST || manifold == FLRW);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	assert (delta > 0.0);
	assert (method == 0 || method == 1 || method == 2);
	if (manifold == DUST) {
		assert (!compact);
		assert (method == 0);
	}
	if (!compact) {
		assert (method == 0 || method == 1);
		assert (r_max > 0.0);
	} else
		assert (a > 0.0);
	#endif

	printf_mpi(rank, "\tEstimating Expected Average Degree...\n");
	fflush(stdout);

	int nb = static_cast<int>(bench) * NBENCH;
	int i;

	double *table;
	long size = 0L;
	int seed = static_cast<int>(4000000000 * mrng.rng());

	if (method == 0) {
		//Method 1 of 3: Use Monte Carlo integration
		double r0;
		if (tau0 > LOG(MTAU, STL) / 3.0)
			r0 = POW(0.5, 2.0 / 3.0, STL) * exp(tau0);
		else
			r0 = POW(SINH(1.5 * tau0, STL), 2.0 / 3.0, STL);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (compact) {
				if (tau0 > LOG(MTAU, STL) / 3.0)
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeFLRW, 0.0, 0.0, r0, r0, NULL, seed, 0) * 16.0 * M_PI * exp(-3.0 * tau0);
				else
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&rescaledDegreeFLRW, 0.0, 0.0, r0, r0, NULL, seed, 0) * 8.0 * M_PI / (SINH(3.0 * tau0, STL) - 3.0 * tau0);
			} else {
				if (manifold == DUST) {
					double kappa = integrate2D(&rescaledDegreeDust, 0.0, 0.0, tau0, tau0, NULL, seed, 0);
					kappa *= 108 * M_PI / POW3(tau0, EXACT);
					printf("kappa: %.8e\n", kappa);
					k_tar = (N_tar * kappa) / (M_PI * POW3(alpha * r_max * tau0, EXACT));
				} else if (manifold == FLRW) {
					double kappa = integrate2D(&rescaledDegreeFLRW_NC, 0.0, 0.0, tau0, tau0, NULL, seed, 0);
					kappa *= 8.0 * M_PI;
					kappa /= SINH(3.0 * tau0, STL) - 3.0 * tau0;
					//printf("kappa: %.8e\n", kappa);
					k_tar = (9.0 * kappa * N_tar) / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
				}
			}
			stopwatchStop(&sCalcDegrees);
		}	
	} else if (method == 1) {
		//Method 2 of 3: Lookup table to approximate method 1
		if (compact) {
			if (!getLookupTable("./etc/raduc_table.cset.bin", &table, &size))
				return false;
		} else {
			if (!getLookupTable("./etc/raducNC_table.cset.bin", &table, &size))
				return false;
		}
		ca->hostMemUsed += size;

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (compact)
				k_tar = lookupValue(table, size, &tau0, NULL, true) * delta * POW2(POW2(a, EXACT), EXACT);
			else
				k_tar = lookupValue(table, size, &tau0, NULL, true) * 9.0 * N_tar / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
			stopwatchStop(&sCalcDegrees);
		}	

		//Check for NaN
		if (k_tar != k_tar)
			return false;

		free(table);
		table = NULL;
		ca->hostMemUsed -= size;
	} else if (method == 2) {
		//Method 3 of 3: Explicit Solution
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
		idata.upper = tau0;

		double max_time;
		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			max_time = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * a / alpha;
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
bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, const int &group_size, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &decode_cpu, const bool &link, const bool &relink, const bool &verbose, const bool &bench, const bool &yes)
{
	#if DEBUG
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (dim == 1 || dim == 3);
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW || manifold == HYPERBOLIC);
	if (manifold == HYPERBOLIC)
		assert (dim == 1);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

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
		else if (manifold == DE_SITTER || manifold == DUST || manifold == FLRW)
			mem += sizeof(float) * N_tar;		//For tau
		if (links_exist) {
			mem += sizeof(int) * (N_tar << 1);	//For k_in and k_out
			mem += sizeof(int) * static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer));	//For edge lists
			mem += sizeof(int) * (N_tar << 1);	//For edge list pointers
			mem += sizeof(bool) * POW2(core_edge_fraction * N_tar, EXACT);	//For adjacency list
		}

		size_t dmem = 0;
		#ifdef CUDA_ENABLED
		size_t dmem1 = 0, dmem2 = 0, dmem3 = 0;
		if (use_gpu) {
			size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar * (1.0 + edge_buffer) / 2)));
			mem += sizeof(uint64_t) * d_edges_size;	//For encoded edge list
			mem += sizeof(int);			//For g_idx

			size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE * group_size << 1)));
			size_t mthread_size = mblock_size * BLOCK_SIZE;
			size_t m_edges_size = mthread_size * mthread_size;
			size_t nbuf = GEN_ADJ_LISTS_GPU_V2 ? NBUFFERS : 1;
			mem += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (host)
			mem += sizeof(bool) * m_edges_size * nbuf;			//For adjacency matrix buffers (host)
			dmem1 += sizeof(float) * mthread_size * 4 * nbuf << 1;	//For 4-D coordinate buffers
			dmem1 += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (device)
			dmem1 += sizeof(bool) * m_edges_size * nbuf;			//For adjacency matrix buffers (device)

			size_t g_mblock_size = static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / (BLOCK_SIZE * group_size << 1));
			size_t g_mthread_size = g_mblock_size * BLOCK_SIZE;
			dmem2 += sizeof(uint64_t) * d_edges_size;	//Encoded edge list used during parallel sorting
			dmem2 += sizeof(int) * (DECODE_LISTS_GPU_V2 ? g_mthread_size : d_edges_size);	//For edge lists
			if (decode_cpu)
				dmem2 = 0;

			dmem3 += sizeof(int) * N_tar << 1;	//Edge list pointers
			dmem3 += sizeof(int) * BLOCK_SIZE << 2;	//Buffers used for scanning

			dmem = dmem1 > dmem2 ? dmem1 : dmem2;
			dmem = dmem > dmem3 ? dmem : dmem3;
		}
		#endif

		printMemUsed("for Network (Estimation)", mem, dmem, rank);
		printf("\nContinue [y/N]?");
		fflush(stdout);
		char response = getchar();
		getchar();
		if (response != 'y')
			return false;
	}

	stopwatchStart(&sCreateNetwork);

	try {
		if (manifold == DE_SITTER || manifold == DUST || manifold == FLRW) {
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

	if (nodes.crd->isNull()) {
		printf("Null in thread %d\n", rank);
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
bool generateNodes(Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, MersenneRNG &mrng, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &symmetric, const bool &compact, const bool &verbose, const bool &bench)
{
	#if DEBUG
	//Values are in correct ranges
	assert (!nodes.crd->isNull());
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (dim == 1 || dim == 3);
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW);
	assert (a >= 0.0);
	assert (tau0 > 0.0);
	if (manifold == DUST || manifold == FLRW) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (dim == 3);
		assert (zeta < HALF_PI);
	} else if (manifold == DE_SITTER) {
		if (compact) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	}
	if (!compact)
		assert (r_max > 0.0);
	#endif

	IntData idata = IntData();
	//Modify these two parameters to trade off between speed and accuracy
	idata.limit = 50;
	idata.tol = 1e-4;

	if (USE_GSL && manifold == FLRW)
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	stopwatchStart(&sGenerateNodes);

	//Generate coordinates for each of N nodes
	double x, rval;
	int i;
	for (i = 0; i < N_tar; i++) {
		////////////////////////////////////////////////////////////
		//~~~~~~~~~~~~~~~~~~~~~~~~~Theta3~~~~~~~~~~~~~~~~~~~~~~~~~//
		//Sample Theta3 from (0, 2pi), as described on p. 2 of [1]//
		////////////////////////////////////////////////////////////

		x = TWO_PI * mrng.rng();
		#if DEBUG
		assert (x > 0.0 && x < TWO_PI);
		#endif
		//if (i % NPRINT == 0) printf("Theta3: %5.5f\n", x); fflush(stdout);

		if (dim == 1) {
			nodes.crd->y(i) = static_cast<float>(x);

			/////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~Eta~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (2) of [2]//
			/////////////////////////////////////////////////

			do nodes.crd->x(i) = static_cast<float>(ATAN(mrng.rng() / TAN(zeta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			while (nodes.crd->x(i) >= static_cast<float>(HALF_PI - zeta));

			#if DEBUG
			assert (nodes.crd->x(i) > 0.0f);
			assert (nodes.crd->x(i) < static_cast<float>(HALF_PI - zeta));
			#endif

			// Use the symmetric interval between (-eta0, eta0)
			if (symmetric) {
				if (!!(i % 2))
					nodes.crd->x(i) *= -1.0;
			}

			nodes.id.tau[i] = static_cast<float>(etaToTauCompact(static_cast<double>(nodes.crd->x(i))));
		} else if (dim == 3) {
			nodes.crd->z(i) = static_cast<float>(x);

			/////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~T~~~~~~~~~~~~~~~~~~~~~~~~//
			//CDF derived from PDF identified in (6) of [2] for 3+1//
			//and from PDF identified in (12) of [2] for FLRW      //
			/////////////////////////////////////////////////////////

			do {
				rval = mrng.rng();

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
				} else if (manifold == DUST) {
					x = tau0 * POW(rval, 1.0 / 3.0, STL);
					nodes.crd->w(i) = tauToEtaDust(x, a, alpha);
				} else if (manifold == DE_SITTER) {
					if (compact) {
						x = 3.5;
						p1[0] = zeta;
						if (!newton(&solveTau, &x, 1000, TOL, p1, NULL, NULL))
							return false;

						if (symmetric) {
							if (!!(i % 2))
								x *= -1.0;
						}
					} else {
						//In this case the eta distribution is used
						double eta0 = HALF_PI - zeta;
						double eta1 = HALF_PI - zeta1;
						nodes.crd->w(i) = eta0 * POW(1.0 - rval * (1.0 - POW3(eta0 / eta1, EXACT)), -1.0 / 3.0, STL);
						x = etaToTauFlat(nodes.crd->w(i));
					}
				}

				nodes.id.tau[i] = static_cast<float>(x);
			} while (nodes.id.tau[i] >= static_cast<float>(tau0));

			#if DEBUG
			//assert (nodes.id.tau[i] >= 0.0f);
			assert (nodes.id.tau[i] < static_cast<float>(tau0));
			#endif

			//Save eta values as well
			if (manifold == FLRW) {
				if (USE_GSL) {
					//Numerical Integration
					idata.upper = static_cast<double>(nodes.id.tau[i]);
					nodes.crd->w(i) = static_cast<float>(integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * a / alpha);
				} else
					//Exact Solution
					nodes.crd->w(i) = static_cast<float>(tauToEtaFLRWExact(nodes.id.tau[i], a, alpha));

				//#if DEBUG 
				//assert (nodes.crd->w(i) < tauToEtaFLRWExact(tau0, a, alpha));
				//#endif
			} else if (manifold == DE_SITTER && compact) {
				nodes.crd->w(i) = static_cast<float>(tauToEtaCompact(static_cast<double>(nodes.id.tau[i])));
				#if DEBUG
				assert (fabs(nodes.crd->w(i)) < tauToEtaCompact(tau0));
				#endif
			}
			#if DEBUG
			if (manifold == DE_SITTER && !compact)
				assert (nodes.crd->w(i) < 0.0);
			//else
			//	assert (nodes.crd->w(i) > 0.0);
			#endif
				
			///////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~Theta1 and Theta2~~~~~~~~~~~~~~~~~~//	
			//CDFs derived from PDFs identified on p. 3 of [2]   //
			//Phi given by [3]				     //
			///////////////////////////////////////////////////////

			if (compact) {
				//Sample Theta1 from (0, pi)
				x = HALF_PI;
				rval = mrng.rng();
				if (!newton(&solveTheta1, &x, 250, TOL, &rval, NULL, NULL))
					return false;
				nodes.crd->x(i) = static_cast<float>(x);
				#if DEBUG
				assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < static_cast<float>(M_PI));
				#endif
			} else {
				nodes.crd->x(i) = static_cast<float>(POW(mrng.rng(), 1.0 / 3.0, APPROX ? FAST : STL) * r_max);
				#if DEBUG
				assert (nodes.crd->x(i) >= 0.0f && nodes.crd->x(i) <= static_cast<float>(r_max));
				#endif
			}
			//if (i % NPRINT == 0) printf("Theta1: %5.5f\n", nodes.crd->x(i)); fflush(stdout);

			//Sample Theta2 from (0, pi)
			nodes.crd->y(i) = static_cast<float>(ACOS(1.0 - 2.0 * mrng.rng(), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			#if DEBUG
			assert (nodes.crd->y(i) >= 0.0f && nodes.crd->y(i) < static_cast<float>(M_PI));
			#endif
			//if (i % NPRINT == 0) printf("Theta2: %5.5f\n", nodes.crd->y(i)); fflush(stdout);
		}
		//if (i % NPRINT == 0) printf("eta: %5.5f\n", nodes.crd->w(i));
		//if (i % NPRINT == 0) printf("tau: %E\n", nodes.id.tau[i]);
	}

	//Manually alter points
	/*nodes.id.tau[0] = tau0 * 0.95;
	nodes.crd->w(0) = tauToEtaFlat(nodes.id.tau[0]);
	nodes.crd->x(0) = r_max * 0.95;
	nodes.crd->y(0) = 0.01;
	nodes.crd->z(0) = 0.01;

	nodes.id.tau[1] = tau0 * 0.96;
	nodes.crd->w(1) = tauToEtaFlat(nodes.id.tau[1]);
	nodes.crd->x(1) = r_max * 0.95;
	nodes.crd->y(1) = HALF_PI / 12.0;
	nodes.crd->z(1) = 0.01;

	double omega12;
	nodesAreRelated(nodes.crd, N_tar, dim, manifold, a, zeta, zeta1, r_max, alpha, compact, 0, 1, &omega12);
	printf("omega12: %f\n", omega12);*/

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
	printChk();*/

	stopwatchStop(&sGenerateNodes);

	if (USE_GSL && manifold == FLRW)
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
bool linkNodes(Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &symmetric, const bool &compact, const bool &verbose, const bool &bench)
{
	#if DEBUG
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
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW);
	assert (a > 0.0);
	assert (tau0 > 0.0);
	if (manifold == DE_SITTER) {
		if (compact) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (manifold == DUST || manifold == FLRW) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (dim == 3);
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	if (!compact)
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

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
			related = nodesAreRelated(nodes.crd, N_tar, dim, manifold, a, zeta, zeta1, r_max, alpha, symmetric, compact, i, j, NULL);

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
	#if DEBUG
	assert (future_idx == past_idx);
	#endif
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

	#if DEBUG
	assert (N_res >= 0);
	assert (N_deg2 >= 0);
	assert (k_res >= 0.0);
	#endif

	if (N_res > 0)
		k_res /= N_res;

	//Debugging options used to visually inspect the adjacency lists and the adjacency pointer lists
	//compareAdjacencyLists(nodes, edges);
	//compareAdjacencyListIndices(nodes, edges);
	//if(!compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, core_edge_exists, N_tar, core_edge_fraction))
	//	return false;

	//Print Results
	if (!printDegrees(nodes, N_tar, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;
	/*if (!printEdgeLists(edges, past_idx, "past-edges_CPU.cset.dbg.dat", "future-edges_CPU.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_CPU.cset.dbg.dat", "future-edge-pointers_CPU.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);*/
	printChk();

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		printf_cyan();
		printf("\t\tUndirected Links:         %d\n", future_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", (k_res * N_res) / N_tar);
		printf_red();
		printf("\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}
