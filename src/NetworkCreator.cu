/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "NetworkCreator.h"

const constexpr char *Spacetime::stdims[];
const constexpr char *Spacetime::manifolds[];
const constexpr char *Spacetime::regions[];
const constexpr char *Spacetime::curvatures[];
const constexpr char *Spacetime::symmetries[];

bool initVars(NetworkProperties * const network_properties, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm)
{
	#if DEBUG
	assert (network_properties != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	#endif

	Spacetime spacetime = network_properties->spacetime;
	int rank = network_properties->cmpi.rank;

	//Benchmarking
	if (network_properties->flags.bench) {
		network_properties->graphID = 0;
		network_properties->flags.verbose = false;
		network_properties->flags.print_network = false;
	}

	//Suppress queries if MPI is enabled
	#ifdef MPI_ENABLED
	if (network_properties->cmpi.num_mpi_threads > 1) {
		if (network_properties->flags.verbose)
			network_properties->flags.yes = true;
		network_properties->flags.use_bit = true;
		network_properties->core_edge_fraction = 1.0;
	}
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

	//If no node positions used, require an edge list is being read
	if (!network_properties->graphID && network_properties->flags.no_pos) {
		printf_mpi(rank, "Flag 'nopos' cannot be used if a graph is not being read.\n");
		fflush(stdout);
		network_properties->cmpi.fail = 1;
	}

	if (network_properties->flags.relink && network_properties->flags.no_pos) {
		printf_mpi(rank, "Flag 'nopos' cannot be used together with 'relink'.\n");
		fflush(stdout);
		network_properties->cmpi.fail = 1;
	}

	if (!network_properties->flags.binomial) {
		//Sample N from Poisson statistic
		Engine eng((long)time(NULL));
		PDistribution pdist(network_properties->N_tar);
		PGenerator prng(eng, pdist);
		for (int i = 0; i < 1000; i++)
			prng();
		network_properties->N = prng();
	} else
		network_properties->N = network_properties->N_tar;

	if (network_properties->gt != RGG) {
		network_properties->flags.use_bit = true;
		network_properties->core_edge_fraction = 1.0;
		network_properties->flags.no_pos = true;
		if (network_properties->flags.relink) {
			network_properties->flags.relink = false;
			network_properties->flags.link = true;
		}
	}

	#ifdef CUDA_ENABLED
	//If the GPU is requested, optimize parameters
	if (!LINK_NODES_GPU_V2 && network_properties->flags.use_gpu && network_properties->N % (BLOCK_SIZE << 1)) {
		printf_mpi(rank, "If you are using the GPU, set the target number of nodes (--nodes) to be a multiple of %d!\n", BLOCK_SIZE << 1);
		printf_mpi(rank, "Alternatively, set LINK_NODES_GPU_V2=true in inc/Constants.h and recompile.\n");
		fflush(stdout);
		network_properties->cmpi.fail = 1;
	}

	if (network_properties->flags.use_gpu && network_properties->flags.no_pos) {
		printf_mpi(rank, "Conflicting parameters: no_pos and use_gpu.  GPU linking requires the use of node positions.\n");
		fflush(stdout);
		network_properties->cmpi.fail = 1;
	}
	#endif

	if (checkMpiErrors(network_properties->cmpi))
		return false;

	//Disable the default GSL Error Handler
	disableGSLErrHandler();

	try {
		double *table;
		double eta0 = 0.0, eta1 = 0.0;
		double q;
		long size = 0L;
		int method;

		if (network_properties->gt == RGG) {
			//Check for an under-constrained system
			if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
				if (!network_properties->N)
					throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
				if (!network_properties->tau0 && !((spacetime.regionIs("Saucer_S") || spacetime.regionIs("Saucer_T")) && SPECIAL_SAUCER) && !spacetime.regionIs("Slab_N3"))
					throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");
				if ((spacetime.curvatureIs("Flat") || spacetime.curvatureIs("Negative")) && (spacetime.regionIs("Slab") || spacetime.regionIs("Slab_S1") || spacetime.regionIs("Slab_TS") || spacetime.regionIs("Triangle_T") || spacetime.regionIs("Triangle_S") || spacetime.regionIs("Cube")) && !((spacetime.regionIs("Saucer_S") || spacetime.regionIs("Saucer_T")) && SPECIAL_SAUCER)) {
					if ((spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter")) && !network_properties->r_max)
						throw CausetException("Flag '--radius', spatial scaling, must be specified!\n");
					else if ((spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) && !network_properties->alpha)
						throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
				}
				if (spacetime.curvatureIs("Positive") && spacetime.manifoldIs("FLRW") && !network_properties->alpha)
					throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
			} else if (spacetime.manifoldIs("Hyperbolic")) {
				if (!network_properties->N)
					throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
				if (!network_properties->r_max)
					throw CausetException("Flag '--radius', maximum radius, must be specified!\n");
			} else if (spacetime.manifoldIs("Polycone")) {
				if (!network_properties->N)
					throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
				if (!network_properties->tau0)
					throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");
				if (!network_properties->gamma)
					throw CausetException("Flag '--gamma', degree exponent, must be specified!\n");
			}

			//Default constraints
			if (spacetime.manifoldIs("Minkowski")) {
				#if SPECIAL_SAUCER
				if (spacetime.regionIs("Saucer_S")) {
					network_properties->tau0 = network_properties->eta0 = 1.0;
					network_properties->r_max = 1.5;
				} else if (spacetime.regionIs("Saucer_T")) {
					network_properties->tau0 = network_properties->eta0 = 1.5;
					network_properties->r_max = 1.0;
				} else
				#endif
				if (spacetime.regionIs("Slab_N3")) {
					network_properties->tau0 = network_properties->eta0 = 1.5;
					network_properties->r_max = 0.5;
				} else
					network_properties->eta0 = network_properties->tau0;
				eta0 = network_properties->eta0;
				network_properties->zeta = HALF_PI - eta0;
				network_properties->a = 1.0;

				#if DEBUG
				assert (network_properties->eta0 > 0.0);
				#endif
			} else if (spacetime.manifoldIs("De_Sitter")) {
				if (!network_properties->delta)
					network_properties->a = 1.0;

				if (spacetime.curvatureIs("Flat")) {
					//We take eta_min = -1 so that rescaled time
					//will begin at tau = 0
					//In this case, the '--age' flag reads tau0
					network_properties->zeta = HALF_PI + 1.0;
					network_properties->zeta1 = HALF_PI - tauToEtaFlat(network_properties->tau0);

					#if DEBUG
					assert (network_properties->zeta > HALF_PI);
					#endif
				} else if (spacetime.curvatureIs("Positive")) {
					//Re-write variables to their correct locations
					//This is because the '--age' flag has read eta0
					//into the tau0 variable
					network_properties->eta0 = network_properties->tau0;
					network_properties->zeta = HALF_PI - network_properties->tau0;
					network_properties->tau0 = etaToTauSph(HALF_PI - network_properties->zeta);

					#if DEBUG
					assert (network_properties->zeta > 0.0 && network_properties->zeta < HALF_PI);
					#endif
				} else if (spacetime.curvatureIs("Negative"))
					network_properties->zeta = HALF_PI - tauToEtaHyp(network_properties->tau0);

				eta0 = HALF_PI - network_properties->zeta;
				eta1 = HALF_PI - network_properties->zeta1;
				network_properties->eta0 = eta0;
			} else if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
				//The pseudoradius takes a default value of 1
				if (!network_properties->delta)
					network_properties->a = 1.0;

				//The maximum radius takes a default value of 1
				//This allows alpha to characterize the spatial cutoff
				if (spacetime.regionIs("Slab") && !network_properties->r_max)
					network_properties->r_max = 1.0;
			} else if (spacetime.manifoldIs("Hyperbolic")) {
				//The hyperbolic curvature takes a default value of 1
				if (!network_properties->zeta)
					network_properties->zeta = 1.0;
			} else if (spacetime.manifoldIs("Polycone")) {
				if (!network_properties->delta)
					network_properties->a = 1.0;
				network_properties->eta0 = tauToEtaPolycone(network_properties->tau0, network_properties->a, network_properties->gamma);
				eta0 = network_properties->eta0;
			}

			/*printf("stdim: %s\n", Spacetime::stdims[spacetime.get_stdim()]);
			printf("manifold: %s\n", Spacetime::manifolds[spacetime.get_manifold()]);
			printf("region: %s\n", Spacetime::regions[spacetime.get_region()]);
			printf("curvature: %s\n", Spacetime::curvatures[spacetime.get_curvature()]);
			printf("symmetry: %s\n", Spacetime::symmetries[spacetime.get_symmetry()]);*/
			//printChk(2);

			//Solve for the remaining constraints
			if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Flat", "Temporal")) {
				//Assumes r_max >> tau0
				network_properties->k_tar = 2.0 * network_properties->N * network_properties->eta0 / (3.0 * network_properties->r_max);
				network_properties->delta = network_properties->N / (4.0 * network_properties->eta0 * network_properties->r_max);
				network_properties->flags.has_exact_k = false;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Positive", "Temporal")) {
				//Assumes a light cones never 'wrap around' the cylinder
				network_properties->k_tar = 2.0 * network_properties->N * network_properties->tau0 / (3.0 * M_PI);
				network_properties->delta = network_properties->N / (4.0 * M_PI * network_properties->tau0);
				network_properties->flags.has_exact_k = true;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_T1", "Flat", "Temporal")) {
				//A guess - will be accurate when eta0 ~ r_max
				network_properties->k_tar = 2.0 * network_properties->N * network_properties->eta0 / (3.0 * network_properties->r_max);
				network_properties->flags.has_exact_k = false;
				double volume = 2.0 * volume_75499530_2(network_properties->r_max, eta0);
				network_properties->delta = network_properties->N / volume;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_S1", "Flat", "Temporal")) {
				//A guess - will be accurate when eta0 ~ r_max
				network_properties->k_tar = 2.0 * network_properties->N * network_properties->eta0 / (3.0 * network_properties->r_max);
				network_properties->flags.has_exact_k = false;
				double volume = 2.0 * volume_75499530_2(eta0, network_properties->r_max);
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_TS", "Flat", "Temporal")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				network_properties->delta = network_properties->N;	//Volume is 1
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_N3", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				network_properties->delta = network_properties->N / 1.5;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Diamond", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->delta = 2.0 * static_cast<double>(network_properties->N) / POW2(network_properties->eta0, EXACT);
				network_properties->flags.has_exact_k = true;
				network_properties->r_max = eta0 / 2.0;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_S", "Flat", "Temporal")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				#if SPECIAL_SAUCER
				double volume = volume_77834_1(network_properties->r_max) - volume_77834_1(-network_properties->r_max);
				#else
				double beta = 1.0 - eta0;
				double volume = 2.0 * (sqrt(1.0 - POW2(beta, EXACT)) - POW2(beta, EXACT) * log((1.0 + sqrt(1.0 - POW2(beta, EXACT))) / beta));
				network_properties->r_max = sqrt(1.0 - POW2(beta, EXACT));
				#endif
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_T", "Flat", "Temporal")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				#if SPECIAL_SAUCER
				double volume = volume_77834_1(network_properties->tau0) - volume_77834_1(-network_properties->tau0);
				#else
				double beta = sqrt(1.0 - POW2(eta0));
				double volume = 2.0 * (sqrt(1.0 - POW2(beta, EXACT)) - POW2(beta, EXACT) * log((1.0 + sqrt(1.0 - POW2(beta, EXACT))) / beta));
				network_properties->r_max = 1.0 - beta;
				#endif
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_T", "Flat", "Temporal")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				double volume = network_properties->eta0 * network_properties->r_max;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_S", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				double volume = network_properties->eta0 * network_properties->r_max;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "None")) {
				network_properties->k_tar = network_properties->N * (network_properties->eta0 / TAN(network_properties->eta0, STL) - LOG(COS(network_properties->eta0, STL), STL) - 1.0) / (TAN(network_properties->eta0, STL) * HALF_PI);
				network_properties->flags.has_exact_k = true;
				if (!!network_properties->delta)
					network_properties->a = SQRT(network_properties->N / (TWO_PI * network_properties->delta * TAN(network_properties->eta0, STL)), STL);
				else
					network_properties->delta = network_properties->N / (TWO_PI * POW2(network_properties->a, EXACT) * TAN(network_properties->eta0, STL));
			} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "Temporal")) {
				network_properties->k_tar = (network_properties->N / M_PI) * ((network_properties->eta0 / TAN(network_properties->eta0, STL) - 1.0) / TAN(network_properties->eta0, STL) + network_properties->eta0);
				network_properties->flags.has_exact_k = true;
				if (!!network_properties->delta)
					network_properties->a = SQRT(network_properties->N / (4.0 * M_PI * network_properties->delta * TAN(network_properties->eta0, STL)), STL);
				else
					network_properties->delta = network_properties->N / (4.0 * M_PI * POW2(network_properties->a, EXACT) * TAN(network_properties->eta0, STL));
			} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Negative", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				network_properties->delta = network_properties->N / (TWO_PI * cosh(network_properties->tau0) - 1.0);
			} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;
				double u0 = (HALF_PI - network_properties->zeta) / sqrt(2.0);
				double w = (network_properties->zeta - network_properties->zeta1) / sqrt(2.0);
				double volume = 2.0 * POW2(network_properties->a) * log(POW2(2.0 * u0 + w) / (4.0 * u0 * (u0 + w)));
				network_properties->delta = network_properties->N / volume;
				network_properties->r_max = ((HALF_PI - network_properties->zeta1) + 1.0) / 2.0;
			} else if (spacetime.spacetimeIs("2", "Dust", "Diamond", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;

				//double volume = 0.0;
				network_properties->delta = 1.0;
				network_properties->alpha *= network_properties->a;
				eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
				network_properties->eta0 = eta0;
				network_properties->r_max = eta0 / 2.0;
				network_properties->zeta = HALF_PI - eta0;
			} else if (spacetime.spacetimeIs("2", "Hyperbolic", "Slab", "Flat", "None")) {
				//We have not yet calculated the actual values
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;
				network_properties->delta = 1.0;
			} else if (spacetime.spacetimeIs("2", "Hyperbolic", "Slab", "Positive", "None")) {
				//We have not yet calculated the actual value
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				double volume;
				if (network_properties->flags.growing)
					volume = 16.0 * M_PI * POW2(sinh(network_properties->r_max / (4.0 * network_properties->zeta)));
				else
					volume = 2.0 * M_PI * (cosh(network_properties->r_max / network_properties->zeta) - 1.0);
				if (!!network_properties->delta)
					network_properties->zeta = sqrt(network_properties->N / (volume * network_properties->delta));
				else
					network_properties->delta = network_properties->N / (volume * POW2(network_properties->zeta));
				network_properties->tau0 = network_properties->r_max / network_properties->zeta;
			} else if (spacetime.spacetimeIs("2", "Polycone", "Slab", "Positive", "None")) {
				double g = network_properties->gamma / (network_properties->gamma - 2.0);
				network_properties->k_tar = network_properties->N * (g + 1.0) * pow(network_properties->tau0, 1.0 - g) / ((3.0 + g) * M_PI);
				network_properties->flags.has_exact_k = true;

				double volume = TWO_PI * pow(network_properties->a, g - 1.0) * pow(network_properties->tau0, g + 1.0) / (g + 1.0);
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("3", "Minkowski", "Slab", "Flat", "Temporal")) {
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				double volume = TWO_PI * POW2(network_properties->r_max) * network_properties->eta0;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("3", "Minkowski", "Diamond", "Flat", "None")) {
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				network_properties->r_max = network_properties->eta0 / 2.0;
				double volume = M_PI * POW3(eta0) / 12;
				network_properties->delta = static_cast<float>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("3", "Minkowski", "Cube", "Flat", "None")) {
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				double volume = POW2(network_properties->r_max) * network_properties->tau0;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("3", "De_Sitter", "Diamond", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;

				network_properties->r_max = (eta1 + 1.0) / 2.0;
				//double volume = 0.0;
				network_properties->delta = 1.0;
			} else if (spacetime.spacetimeIs("3", "Dust", "Diamond", "Flat", "None")) {
				network_properties->k_tar = network_properties->N / 2.0;
				network_properties->flags.has_exact_k = false;

				//double volume = 0.0;
				network_properties->delta = 1.0;
				network_properties->alpha *= network_properties->a;
				network_properties->eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
				eta0 = network_properties->eta0;
				network_properties->zeta = HALF_PI - eta0;
				network_properties->r_max = eta0 / 2.0;
			} else if (spacetime.spacetimeIs("4", "Minkowski", "Diamond", "Flat", "None")) {
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				network_properties->r_max = network_properties->eta0 / 2.0;
				double volume = POW2(POW2(eta0)) * M_PI / 24.0;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Flat", "None")) {
				int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());
				network_properties->k_tar = 9.0 * network_properties->N * POW2(POW3(eta0 * eta1, EXACT), EXACT) * integrate2D(&averageDegree_10788_0, eta0, eta0, eta1, eta1, NULL, seed, 0) / (POW3(network_properties->r_max, EXACT) * POW2(POW3(eta1, EXACT) - POW3(eta0, EXACT), EXACT));
				network_properties->flags.has_exact_k = true;
				if (!!network_properties->delta)
					network_properties->a = POW(9.0 * network_properties->N * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * network_properties->delta * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT))), 0.25, STL);
				else
					network_properties->delta = 9.0 * network_properties->N * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT)));
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "None")) {
				network_properties->k_tar = network_properties->N * (12.0 * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL)) - (6.0 * LOG(COS(eta0, STL), STL) + 5.0) / POW2(COS(eta0, STL), EXACT) - 7.0) / (POW2(2.0 + 1.0 / POW2(COS(eta0, STL), EXACT), EXACT) * TAN(eta0, STL) * 3.0 * HALF_PI);
				network_properties->flags.has_exact_k = true;
				if (!!network_properties->delta)
					network_properties->a = POW(network_properties->N * 3.0 / (2.0 * POW2(M_PI, EXACT) * network_properties->delta * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL)), 0.25, STL);
				else
					network_properties->delta = network_properties->N * 3.0 / (2.0 * POW2(M_PI * POW2(network_properties->a, EXACT), EXACT) * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL));
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "Temporal")) {
				network_properties->k_tar = 2.0 * network_properties->N * POW3(cos(eta0), EXACT) * (-51.0 * sin(eta0) + 7.0 * sin(3.0 * eta0) + 6.0 * (eta0 * (3.0 + 1.0 / POW2(cos(eta0), EXACT)) + tan(eta0)) / cos(eta0)) / (3.0 * M_PI * POW2(3.0 * sin(eta0) + sin(3.0 * eta0), EXACT));
				network_properties->flags.has_exact_k = true;
				if (!!network_properties->delta)
					network_properties->a = POW(3.0 * network_properties->N * POW3(cos(eta0), EXACT) / (2.0 * POW2(M_PI, EXACT) * network_properties->delta * (3.0 * sin(eta0) + sin(3.0 * eta0))), 0.25, STL);
				else
					network_properties->delta = 3.0 * network_properties->N * POW3(cos(eta0), EXACT) / (2.0 * POW2(M_PI, EXACT) * POW2(POW2(network_properties->a, EXACT), EXACT) * (3.0 * sin(eta0) + sin(3.0 * eta0)));

			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Flat", "None")) {
				double xi = eta0 / sqrt(2.0);
				double w = (eta1 - eta0) / sqrt(2.0);
				double mu = LOG(POW2(w + 2.0 * xi, EXACT) / (4.0 * xi * (w + xi)), STL) - POW2(w / (w + 2.0 * xi), EXACT);
				if (!!network_properties->delta)
					network_properties->a = POW(3.0 * network_properties->N / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
				else
					network_properties->delta = 3.0 * network_properties->N / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
				if (!getLookupTable("./etc/tables/average_degree_11300_0_table.cset.bin", &table, &size))
					throw CausetException("Average degree table not found!\n");
				network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
				if (network_properties->k_tar != network_properties->k_tar)
					throw CausetException("Value not found in average degree table!\n");
				network_properties->flags.has_exact_k = true;
				network_properties->r_max = w / sqrt(2.0);
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Positive", "None")) {
				double xi = eta0 / sqrt(2.0);
				double mu = log(0.5 * (1.0 / cos(sqrt(2.0) * xi) + 1.0)) - 1.0 / POW2(cos(xi / sqrt(2.0)), EXACT) + 1.0;
				if (!!network_properties->delta)
					network_properties->a = POW(3.0 * network_properties->N / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
				else
					network_properties->delta = 3.0 * network_properties->N / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
				if (!getLookupTable("./etc/tables/average_degree_13348_0_table.cset.bin", &table, &size))
					throw CausetException("Average degree table not found!\n");
				network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &eta0, NULL, true);
				if (network_properties->k_tar != network_properties->k_tar)
					throw CausetException("Value not found in average degree table!\n");
				network_properties->flags.has_exact_k = true;
			} else if (spacetime.spacetimeIs("4", "Dust", "Slab", "Flat", "None")) {
				if (!!network_properties->delta)
					network_properties->a = POW(network_properties->N / (M_PI * network_properties->delta * POW3(network_properties->alpha * network_properties->tau0, EXACT)), 0.25, STL);
				else
					network_properties->delta = network_properties->N / (M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * POW3(network_properties->alpha * network_properties->tau0, EXACT));
				
				int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());
				network_properties->k_tar = (108.0 * M_PI / POW3(network_properties->tau0, EXACT)) * network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * integrate2D(&averageDegree_10820_0, 0.0, 0.0, network_properties->tau0, network_properties->tau0, NULL, seed, 0);
				network_properties->flags.has_exact_k = true;
				network_properties->alpha *= network_properties->a;
				eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
				network_properties->zeta = HALF_PI - eta0;
			} else if (spacetime.spacetimeIs("4", "Dust", "Diamond", "Flat", "None")) {
				double t = POW2(POW2(1.5 * network_properties->tau0, EXACT), EXACT);
				if (!!network_properties->delta)
					network_properties->a = POW(2970.0 * 64.0 * network_properties->N / (1981.0 * M_PI * network_properties->delta * t), 0.25, STL);
				else
					network_properties->delta = 2970.0 * 64.0 * network_properties->N / (1981.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * t);
				network_properties->alpha = 2.0 * network_properties->a; //This property should not affect results in the diamond
				if (!getLookupTable("./etc/tables/average_degree_11332_0_table.cset.bin", &table, &size))
					throw CausetException("Average degree table not found!\n");
				network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
				if (network_properties->k_tar != network_properties->k_tar)
					throw CausetException("Value not found in average degree table!\n");
				network_properties->flags.has_exact_k = true;
				eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
				network_properties->eta0 = eta0;
				network_properties->zeta = HALF_PI - eta0;
				network_properties->r_max = eta0 / 2.0;
			} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Flat", "None")) {
				method = 0;
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->spacetime, network_properties->N, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.verbose, network_properties->flags.bench, method))
					network_properties->cmpi.fail = 1;
				network_properties->flags.has_exact_k = true;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
					
				q = 9.0 * network_properties->N / (TWO_PI * POW3(network_properties->alpha * network_properties->r_max, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
				if (!!network_properties->delta)
					network_properties->a = POW(q / network_properties->delta, 0.25, STL);
				else
					network_properties->delta = q / POW2(POW2(network_properties->a, EXACT), EXACT);
				network_properties->alpha *= network_properties->a;
				eta0 = tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);
				network_properties->zeta = HALF_PI - eta0;
			} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Positive", "None")) {
				q = 3.0 * network_properties->N / (POW2(M_PI, EXACT) * POW3(network_properties->alpha, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
				if (!!network_properties->delta)
					network_properties->a = POW(q / network_properties->delta, 0.25, STL);
				else
					network_properties->delta = q / POW2(POW2(network_properties->a, EXACT), EXACT);
				network_properties->alpha *= network_properties->a;
				eta0 = tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);
				network_properties->zeta = HALF_PI - eta0;

				method = 1;
				if (!solveExpAvgDegree(network_properties->k_tar, network_properties->spacetime, network_properties->N, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.verbose, network_properties->flags.bench, method))
					network_properties->cmpi.fail = 1;
				network_properties->flags.has_exact_k = true;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			} else if (spacetime.spacetimeIs("4", "FLRW", "Diamond", "Flat", "None")) {
				//We REQUIRE a = alpha for this spacetime
				eta0 = tauToEtaFLRWExact(network_properties->tau0, 1.0, 1.0);
				network_properties->eta0 = eta0;
				network_properties->zeta = HALF_PI - eta0;
				network_properties->r_max = eta0 / 2.0;

				//Bisection Method
				double res = 1.0, tol = 1.0e-10;
				double lower = 0.0, upper = network_properties->tau0;
				int iter = 0, max_iter = 10000;

				double x0 = 0.0;
				while (upper - lower > tol && iter < max_iter) {
					x0 = (lower + upper) / 2.0;
					res = tauToEtaFLRWExact(x0, 1.0, 1.0);
					res -= eta0 / 2.0;
					if (res < 0.0)
						lower = x0;
					else
						upper = x0;
					iter++;
				}
				//Store the result in zeta1 variable
				network_properties->zeta1 = x0;

				IntData idata;
				idata.limit = 100;
				idata.tol = 1e-8;
				idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
				idata.upper = network_properties->zeta1;
				double params[3];
				params[0] = network_properties->tau0;
				params[1] = eta0;
				params[2] = network_properties->zeta1;
				double vol_lower = integrate1D(&volume_11396_0_lower, &params, &idata, QAGS);
				assert (vol_lower == vol_lower);
				idata.lower = idata.upper;
				idata.upper = network_properties->tau0;
				double vol_upper = integrate1D(&volume_11396_0_upper, &params, &idata, QAGS);
				assert (vol_upper == vol_upper);
				double mu = vol_lower + vol_upper;
				gsl_integration_workspace_free(idata.workspace);

				if (!!network_properties->delta)
					network_properties->a = POW(3.0 * network_properties->N / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
				else
					network_properties->delta = 3.0 * network_properties->N / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
				network_properties->alpha = network_properties->a;

				if (!getLookupTable("./etc/tables/average_degree_11396_0_table.cset.bin", &table, &size))
					throw CausetException("Average degree table not found!\n");
				network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
				if (network_properties->k_tar != network_properties->k_tar)
					throw CausetException("Value not found in average degree table!\n");
				network_properties->flags.has_exact_k = true;
			} else if (spacetime.spacetimeIs("5", "Minkowski", "Diamond", "Flat", "None")) {
				network_properties->k_tar = 10.0;
				network_properties->flags.has_exact_k = false;

				network_properties->r_max = network_properties->eta0 / 2.0;
				double volume = eta0 * POW2(POW2(eta0) * M_PI) / 160.0;
				network_properties->delta = static_cast<double>(network_properties->N) / volume;
			} else {
				#if DEBUG
				printf("Spacetime ID: [%s]\n", spacetime.toHexString());
				#endif
				throw CausetException("Spacetime parameters not supported!\n");
			}

			if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone")) {
				#if DEBUG
				assert (network_properties->k_tar > 0.0);
				if (spacetime.manifoldIs("Hyperbolic"))
					assert (network_properties->zeta > 0.0);
				else
					assert (network_properties->a > 0.0);
				assert (network_properties->delta > 0.0);
				if (!((spacetime.manifoldIs("De_Sitter") && spacetime.curvatureIs("Flat")) || spacetime.manifoldIs("Hyperbolic")))
					assert (network_properties->zeta < HALF_PI);
				#endif

				//Display Constraints
				printf_mpi(rank, "\n");
				printf_mpi(rank, "\tParameters Constraining the %d+1 %s Causal Set:\n", atoi(Spacetime::stdims[spacetime.get_stdim()]) - 1, Spacetime::manifolds[spacetime.get_manifold()]);
				printf_mpi(rank, "\t--------------------------------------------\n");
				if (!rank) printf_cyan();
				printf_mpi(rank, "\t > Manifold:\t\t\t%s", Spacetime::manifolds[spacetime.get_manifold()]);
				if (spacetime.manifoldIs("Hyperbolic") && network_properties->flags.growing) {
					printf_mpi(rank, " (Growing Model");
					if (network_properties->flags.link_epso)
						printf_mpi(rank, ", EPSO)\n");
					else
						printf_mpi(rank, ")\n");
				} else
					printf_mpi(rank, "\n");
				printf_mpi(rank, "\t > Spacetime Dimension:\t\t%d+1\n", atoi(Spacetime::stdims[spacetime.get_stdim()]) - 1);
				printf_mpi(rank, "\t > Region:\t\t\t%s", Spacetime::regions[spacetime.get_region()]);
				#if SPECIAL_SAUCER
				if (spacetime.manifoldIs("Minkowski") && (spacetime.regionIs("Saucer_S") || spacetime.regionIs("Saucer_T")))
					printf_mpi(rank, " (Special)\n");
				else
				#endif
					printf_mpi(rank, "\n");
				printf_mpi(rank, "\t > Curvature:\t\t\t%s\n", Spacetime::curvatures[spacetime.get_curvature()]);
				printf_mpi(rank, "\t > Symmetry:\t\t\t%s\n", Spacetime::symmetries[spacetime.get_symmetry()]);
				printf_mpi(rank, "\t > Spacetime ID:\t\t%s\n", spacetime.toHexString());
				if (!rank) printf_std();
				printf_mpi(rank, "\t--------------------------------------------\n");
				if (!rank) printf_cyan();
				printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N);
				printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
				if (network_properties->flags.has_exact_k)
					printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
				if (spacetime.symmetryIs("Temporal")) {
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", -eta0);
					printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
				} else if (spacetime.manifoldIs("De_Sitter") && spacetime.curvatureIs("Flat")) {
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", eta0);
					printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta1);
				} else if ((spacetime.manifoldIs("De_Sitter") && spacetime.curvatureIs("Negative")) || spacetime.manifoldIs("Polycone")) {
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t-\u221E\n");
					printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
				} else if (!spacetime.manifoldIs("Hyperbolic")) {
					printf_mpi(rank, "\t > Min. Conformal Time:\t\t0.0\n");
					printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
				} else {
					printf_mpi(rank, "\t > Max. Radius:\t\t\t%.6f\n", network_properties->r_max);
					printf_mpi(rank, "\t > Hyperbolic Curvature:\t%.6f\n", network_properties->zeta);
				}
				if (!(spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("Hyperbolic")))
					printf_mpi(rank, "\t > Max. Rescaled Time:\t\t%.6f\n", network_properties->tau0);
				if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("FLRW"))
					printf_mpi(rank, "\t > Dark Energy Density:\t\t%.6f\n", network_properties->omegaL);
				if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
					printf_mpi(rank, "\t > Spatial Scaling:\t\t%.6f\n", network_properties->alpha);
				if (spacetime.curvatureIs("Flat") && (spacetime.regionIs("Slab") || spacetime.regionIs("Slab_T1") || spacetime.regionIs("Slab_S1") || spacetime.regionIs("Slab_TS") || spacetime.regionIs("Slab_N3") || spacetime.regionIs("Saucer_S") || spacetime.regionIs("Saucer_T") || spacetime.regionIs("Triangle_T") || spacetime.regionIs("Triangle_S")))
					printf_mpi(rank, "\t > Spatial Cutoff:\t\t%.6f\n", network_properties->r_max);
				if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Polycone"))
					printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
				if (spacetime.manifoldIs("Polycone"))
					printf_mpi(rank, "\t > Degree Exponent:\t\t%.6f\n", network_properties->gamma);
				printf_mpi(rank, "\t > Random Seed:\t\t\t%Ld\n", network_properties->seed);
				if (!rank) { printf_std(); printf("\n"); }
				fflush(stdout);
			}
		} else {
			if (!network_properties->N)
				throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining the Causal Set:\n");
			printf_mpi(rank, "\t--------------------------------------------\n");
			printf_mpi(rank, "\t > Graph Type:\t\t");
			if (network_properties->gt == KR_ORDER)
				printf_mpi(rank, "Kleitman-Rothschild partial order\n");
			else if (network_properties->gt == RANDOM)
				printf_mpi(rank, "Random partial order\n");
			else if (network_properties->gt == _2D_ORDER)
				printf_mpi(rank, "2D order\n");
			printf_mpi(rank, "\t > Number of Nodes:\t%d\n", network_properties->N);
			printf_mpi(rank, "\t > Random Seed:\t\t%Ld\n", network_properties->seed);
			if (!rank) { printf_std(); printf("\n"); }
			fflush(stdout);
		}

		//Miscellaneous Tasks
		if (!network_properties->edge_buffer)
			network_properties->edge_buffer = 0.2;

		//if (network_properties->k_tar >= network_properties->N / 32 - 1) {
			//This is when a bit array is smaller than the adjacency lists
			//network_properties->flags.use_bit = true;
			//network_properties->core_edge_fraction = 1.0;
			//printf_dbg("USE_BIT = true\n");
		//}

		#ifdef CUDA_ENABLED
		//Adjacency matrix not implemented in certain GPU algorithms
		if (network_properties->flags.use_gpu && !LINK_NODES_GPU_V2) {
			network_properties->flags.use_bit = false;
			network_properties->core_edge_fraction = 0.0;
		}

		//Determine group size and decoding method
		if (network_properties->flags.use_gpu) {
			long mem = GLOB_MEM + 1L;
			long d_edges_size = (!network_properties->gt == RGG || network_properties->flags.use_bit) ? 1L : static_cast<long>(exp2(ceil(log2(network_properties->N * network_properties->k_tar * (1.0 + network_properties->edge_buffer) / 2.0))));
			float gsize = 0.5f;
			bool dcpu = false;

			while (mem > GLOB_MEM) {
				//Used in generateLists_v2
				//The group size - the number of groups, along one index, the full matrix is broken up into
				gsize *= 2.0f;
				//The 'mega-block' size - the number of thread blocks along index 'i' within a group	
				long mbsize = static_cast<long>(ceil(static_cast<float>(network_properties->N) / (BLOCK_SIZE * gsize)));
				//The 'mega-thread' size - the number of threads along a dimension of a group
				long mtsize = mbsize * BLOCK_SIZE;
				//The 'mega-edges' size - the number of edges represented by the sub-matrix passed to the GPU
				long mesize = mtsize * mtsize;

				//Used in decodeLists_v2
				long gmbsize = static_cast<long>(network_properties->N * network_properties->k_tar * (1.0 + network_properties->edge_buffer) / (BLOCK_SIZE * gsize * 2));
				long gmtsize = gmbsize * BLOCK_SIZE;

				long mem1 = (40L * mtsize + mesize) * NBUFFERS;							//For generating
				long mem2 = network_properties->flags.use_bit ? 0L : 4L * (2L * d_edges_size + gmtsize);	//For decoding

				if (mem2 > GLOB_MEM / 4L) {
					mem2 = 0L;
					dcpu = true;
				}

				mem = mem1 > mem2 ? mem1 : mem2;
			}

			network_properties->group_size = gsize < NBUFFERS ? NBUFFERS : gsize;
			network_properties->flags.decode_cpu = dcpu;
		}
		#endif

		uint64_t pair_multiplier = static_cast<uint64_t>(network_properties->N) * (network_properties->N - 1) / 2;
		if (network_properties->flags.calc_success_ratio && network_properties->N_sr <= 1.0)
			network_properties->N_sr *= pair_multiplier;
		if (network_properties->flags.calc_geo_dis && network_properties->N_gd <= 1.0)
			network_properties->N_gd *= pair_multiplier;

		if (network_properties->flags.calc_action) {
			#if DEBUG
			assert (network_properties->max_cardinality == -1 || network_properties->max_cardinality == 1);
			#endif
			if (network_properties->max_cardinality == -1)
				network_properties->max_cardinality = 5;
			else
				network_properties->max_cardinality = network_properties->N;
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

//Calculate Expected Average Degree in the FLRW Spacetime
//See Causal Set Notes for detailed explanation of methods
//NOTE: This method is largely historical - only a small portion is used
//  in practice, but it offers several methods to achieve the same outcome
bool solveExpAvgDegree(float &k_tar, const Spacetime &spacetime, const int &N, double &a, const double &r_max, double &tau0, const double &alpha, const double &delta, const int &rank, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &verbose, const bool &bench, const int method)
{
	#if DEBUG
	assert (ca != NULL);
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("FLRW"));
	assert (N > 0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	assert (method == 0 || method == 1 || method == 2);
	if (spacetime.curvatureIs("Flat")) {
		assert (method == 0 || method == 1);
		assert (r_max > 0.0);
	} else {
		assert (delta > 0.0);
		assert (a > 0.0);
	}
	#endif

	printf_mpi(rank, "\tEstimating Expected Average Degree...\n");
	fflush(stdout);

	int nb = static_cast<int>(bench) * NBENCH;
	int i;

	double *table;
	double kappa;
	long size = 0L;
	int seed = static_cast<int>(4000000000 * mrng.rng());

	switch (method) {
	case 0:
	{
		//Method 1 of 3: Use Monte Carlo integration
		double r0;
		if (tau0 > LOG(MTAU, STL) / 3.0)
			r0 = POW(0.5, 2.0 / 3.0, STL) * exp(tau0);
		else
			r0 = POW(SINH(1.5 * tau0, STL), 2.0 / 3.0, STL);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Flat", "None")) {
				kappa = integrate2D(&averageDegree_10884_0, 0.0, 0.0, tau0, tau0, NULL, seed, 0);
				kappa *= 8.0 * M_PI;
				kappa /= SINH(3.0 * tau0, STL) - 3.0 * tau0;
				k_tar = (9.0 * kappa * N) / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
			} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Positive", "None")) {
				if (tau0 > LOG(MTAU, STL) / 3.0)
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&averageDegree_12932_0, 0.0, 0.0, r0, r0, NULL, seed, 0) * 16.0 * M_PI * exp(-3.0 * tau0);
				else
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&averageDegree_12932_0, 0.0, 0.0, r0, r0, NULL, seed, 0) * 8.0 * M_PI / (SINH(3.0 * tau0, STL) - 3.0 * tau0);
			} else {
				fprintf(stderr, "Spacetime parameters not supported!\n");
				return false;
			}
			stopwatchStop(&sCalcDegrees);
		}
		break;
	}
	case 1:
	{
		//Method 2 of 3: Lookup table to approximate method 1
		if (spacetime.curvatureIs("Positive")) {
			if (!getLookupTable("./etc/tables/raduc_table.cset.bin", &table, &size))
				return false;
		} else if (spacetime.curvatureIs("Flat")) {
			if (!getLookupTable("./etc/tables/raducNC_table.cset.bin", &table, &size))
				return false;
		} else
			return false;
		
		ca->hostMemUsed += size;

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (spacetime.curvatureIs("Positive"))
				k_tar = lookupValue(table, size, &tau0, NULL, true) * delta * POW2(POW2(a, EXACT), EXACT);
			else if (spacetime.curvatureIs("Flat"))
				k_tar = lookupValue(table, size, &tau0, NULL, true) * 9.0 * N / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
			else
				return false;
			stopwatchStop(&sCalcDegrees);
		}	

		//Check for NaN
		if (k_tar != k_tar)
			return false;

		free(table);
		table = NULL;
		ca->hostMemUsed -= size;
		break;
	}
	case 2:
	{
		//Method 3 of 3: Explicit Solution
		if (!getLookupTable("./etc/tables/ctuc_table.cset.bin", &table, &size))
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

		k_tar = integrate2D(&averageDegree_12932_0_a, 0.0, 0.0, max_time, max_time, params, seed, 0);
		k_tar *= 4.0 * M_PI * delta * POW2(POW2(alpha, EXACT), EXACT);

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate2D(&averageDegree_12932_0_a, 0.0, 0.0, max_time, max_time, params, seed, 0);
			stopwatchStop(&sCalcDegrees);
		}
	
		idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
		idata.upper = max_time;
		k_tar /= (3.0 * integrate1D(&averageDegree_12932_0_b, params, &idata, QAGS));

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			integrate1D(&averageDegree_12932_0_b, params, &idata, QAGS);
			stopwatchStop(&sCalcDegrees);
		}

		gsl_integration_workspace_free(idata.workspace);

		free(params);
		params = NULL;
		ca->hostMemUsed -= size + sizeof(double) * 3;

		free(table);
		table = NULL;
		ca->hostMemUsed -= size;
		break;
	}
	default:
		return false;
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
bool createNetwork(Node &nodes, Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, const float &core_edge_fraction, const float &edge_buffer, const GraphType &gt, CausetMPI &cmpi, const int &group_size, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &decode_cpu, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &mpi_split, const bool &verbose, const bool &bench, const bool &yes)
{
	#if DEBUG
	assert (ca != NULL);
	assert (N > 0);
	if (gt == RGG) {
		assert (k_tar > 0.0f);
		assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") || spacetime.stdimIs("5"));
		assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"));
		if (spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"))
			assert (spacetime.stdimIs("2"));
	}
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	int rank = cmpi.rank;
	bool links_exist = link || relink;

	if (verbose && !yes) {
		//Estimate memory usage before allocating
		size_t mem = 0;
		if (!no_pos) {
			#if EMBED_NODES
			mem += sizeof(float) * N * (atoi(Spacetime::stdims[spacetime.get_stdim()]) + 1);
			#else
			mem += sizeof(float) * N * atoi(Spacetime::stdims[spacetime.get_stdim()]);
			#endif
			if (spacetime.manifoldIs("Hyperbolic"))
				mem += sizeof(int) * N;		//For AS
			else if (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Polycone"))
				mem += sizeof(float) * N;		//For tau
		}
		if (links_exist) {
			mem += sizeof(int) * (N << 1);	//For k_in and k_out
			if (!use_bit) {
				mem += sizeof(int) * static_cast<int64_t>(N) * k_tar * (1.0 + edge_buffer);	//For edge lists
				mem += sizeof(int64_t) * (N << 1);	//For edge list pointers
			}
			#ifdef MPI_ENABLED
			if (mpi_split) {
				mem += static_cast<uint64_t>(POW2(core_edge_fraction * N)) / (8 * cmpi.num_mpi_threads);	//Adjacency matrix
				mem += static_cast<uint64_t>(core_edge_fraction * N) * ceil(static_cast<int>(N * core_edge_fraction) / (2.0 * POW2(cmpi.num_mpi_threads, EXACT))) / 8;
			} else
			#endif
				mem += static_cast<uint64_t>(POW2(core_edge_fraction * N)) / 8;
		}

		size_t dmem = 0;
		#ifdef CUDA_ENABLED
		size_t dmem1 = 0, dmem2 = 0;
		if (use_gpu && gt == RGG) {
			size_t d_edges_size = pow(2.0, ceil(log2(N * k_tar * (1.0 + edge_buffer) / 2)));
			if (!use_bit)
				mem += sizeof(uint64_t) * d_edges_size;	//For encoded edge list
			mem += sizeof(int64_t);				//For g_idx

			size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N) / (BLOCK_SIZE * group_size)));
			size_t mthread_size = mblock_size * BLOCK_SIZE;
			size_t m_edges_size = mthread_size * mthread_size;
			size_t nbuf = GEN_ADJ_LISTS_GPU_V2 ? NBUFFERS : 1;
			mem += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (host)
			mem += sizeof(bool) * m_edges_size * nbuf;		//For adjacency matrix buffers (host)
			#if EMBED_NODES
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			#else
			dmem1 += sizeof(float) * mthread_size * atoi(Spacetime::stdims[spacetime.get_stdim()]) * nbuf << 1;	//For coordinate buffers
			#endif
			dmem1 += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (device)
			dmem1 += sizeof(bool) * m_edges_size * nbuf;			//For adjacency matrix buffers (device)

			if (!use_bit) {
				size_t g_mblock_size = static_cast<uint64_t>(N) * k_tar * (1.0 + edge_buffer) / (BLOCK_SIZE * group_size << 1);
				size_t g_mthread_size = g_mblock_size * BLOCK_SIZE;
				dmem2 += sizeof(uint64_t) * d_edges_size;	//Encoded edge list used during parallel sorting
				dmem2 += sizeof(int) * (DECODE_LISTS_GPU_V2 ? g_mthread_size : d_edges_size);	//For edge lists
				if (decode_cpu)
					dmem2 = 0;
			}

			dmem = dmem1 > dmem2 ? dmem1 : dmem2;
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
		if (!no_pos) {
			if ((spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) || (spacetime.manifoldIs("Hyperbolic") && spacetime.curvatureIs("Positive")) || spacetime.manifoldIs("Polycone")) {
				nodes.id.tau = (float*)malloc(sizeof(float) * N);
				if (nodes.id.tau == NULL)
					throw std::bad_alloc();
				memset(nodes.id.tau, 0, sizeof(float) * N);
				ca->hostMemUsed += sizeof(float) * N;
			} else if (spacetime.manifoldIs("Hyperbolic") && spacetime.curvatureIs("Flat")) {
				nodes.id.AS = (int*)malloc(sizeof(int) * N);
				if (nodes.id.AS == NULL)
					throw std::bad_alloc();
				memset(nodes.id.AS, 0, sizeof(int) * N);
				ca->hostMemUsed += sizeof(int) * N;
			}

			#if EMBED_NODES
			if (spacetime.stdimIs("4")) {
				nodes.crd = new Coordinates5D();

				nodes.crd->v() = (float*)malloc(sizeof(float) * N);
				nodes.crd->w() = (float*)malloc(sizeof(float) * N);
				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->v() == NULL || nodes.crd->w() == NULL || nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->v(), 0, sizeof(float) * N);
				memset(nodes.crd->w(), 0, sizeof(float) * N);
				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);
				memset(nodes.crd->z(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 5;
			} else if (spacetime.stdimIs("2")) {
				nodes.crd = new Coordinates3D();

				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);
				memset(nodes.crd->z(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 3;
			}
			#else
			if (spacetime.stdimIs("5")) {
				nodes.crd = new Coordinates5D();

				nodes.crd->v() = (float*)malloc(sizeof(float) * N);
				nodes.crd->w() = (float*)malloc(sizeof(float) * N);
				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->v() == NULL || nodes.crd->w() == NULL || nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->v(), 0, sizeof(float) * N);
				memset(nodes.crd->w(), 0, sizeof(float) * N);
				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);
				memset(nodes.crd->z(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 5;
			} else if (spacetime.stdimIs("4")) {
				nodes.crd = new Coordinates4D();
	
				nodes.crd->w() = (float*)malloc(sizeof(float) * N);
				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->w() == NULL || nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->w(), 0, sizeof(float) * N);
				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);
				memset(nodes.crd->z(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 4;
			} else if (spacetime.stdimIs("3")) {
				nodes.crd = new Coordinates3D();

				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);
				memset(nodes.crd->z(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 3;
			} else if (spacetime.stdimIs("2")) {
				nodes.crd = new Coordinates2D();

				nodes.crd->x() = (float*)malloc(sizeof(float) * N);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N);

				if (nodes.crd->x() == NULL || nodes.crd->y() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->x(), 0, sizeof(float) * N);
				memset(nodes.crd->y(), 0, sizeof(float) * N);

				ca->hostMemUsed += sizeof(float) * N * 2;
			}
			#endif
		}

		if (links_exist) {
			nodes.k_in = (int*)malloc(sizeof(int) * N);
			if (nodes.k_in == NULL)
				throw std::bad_alloc();
			memset(nodes.k_in, 0, sizeof(int) * N);
			ca->hostMemUsed += sizeof(int) * N;

			nodes.k_out = (int*)malloc(sizeof(int) * N);
			if (nodes.k_out == NULL)
				throw std::bad_alloc();
			memset(nodes.k_out, 0, sizeof(int) * N);
			ca->hostMemUsed += sizeof(int) * N;
		}

		if (verbose)
			printMemUsed("for Nodes", ca->hostMemUsed, ca->devMemUsed, rank);

		if (links_exist) {
			if (!use_bit) {
				edges.past_edges = (int*)malloc(sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2));
				if (edges.past_edges == NULL)
					throw std::bad_alloc();
				memset(edges.past_edges, 0, sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2));
				ca->hostMemUsed += sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2);

				edges.future_edges = (int*)malloc(sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2));
				if (edges.future_edges == NULL)
					throw std::bad_alloc();
				memset(edges.future_edges, 0, sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2));
				ca->hostMemUsed += sizeof(int) * static_cast<uint64_t>(N * k_tar * (1.0 + edge_buffer) / 2);

				edges.past_edge_row_start = (int64_t*)malloc(sizeof(int64_t) * N);
				if (edges.past_edge_row_start == NULL)
					throw std::bad_alloc();
				memset(edges.past_edge_row_start, 0, sizeof(int64_t) * N);
				ca->hostMemUsed += sizeof(int64_t) * N;
	
				edges.future_edge_row_start = (int64_t*)malloc(sizeof(int64_t) * N);
				if (edges.future_edge_row_start == NULL)
					throw std::bad_alloc();
				memset(edges.future_edge_row_start, 0, sizeof(int64_t) * N);
				ca->hostMemUsed += sizeof(int64_t) * N;
			}

			int length = 0;
			if (mpi_split) {
				length = static_cast<int>(ceil(static_cast<float>(static_cast<int>(N * core_edge_fraction)) / cmpi.num_mpi_threads));
				int n = static_cast<unsigned int>(POW2(cmpi.num_mpi_threads, EXACT)) << 1;
				if (length % n)
					length += n - (length % n);
			} else
				length = static_cast<int>(ceil(N * core_edge_fraction));
			adj.reserve(length);
			for (int i = 0; i < length; i++) {
				FastBitset fb(static_cast<uint64_t>(core_edge_fraction * N));
				adj.push_back(fb);
				ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
			}

			#ifdef MPI_ENABLED
			if (mpi_split && cmpi.num_mpi_threads > 1) {
				int buflen = length / (cmpi.num_mpi_threads << 1);
				cmpi.adj_buf.reserve(buflen);
				for (int i = 0; i < buflen; i++) {
					FastBitset fb(static_cast<uint64_t>(core_edge_fraction * N));
					cmpi.adj_buf.push_back(fb);
					ca->hostMemUsed += sizeof(BlockType) * fb.getNumBlocks();
				}
			}
			#endif
		}

		memoryCheckpoint(ca->hostMemUsed, ca->maxHostMemUsed, ca->devMemUsed, ca->maxDevMemUsed);
		if (verbose)
			printMemUsed("for Network", ca->hostMemUsed, ca->devMemUsed, rank);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		cmpi.fail = 1;
	}

	if (!no_pos && nodes.crd->isNull()) {
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
bool generateNodes(Node &nodes, const Spacetime &spacetime, const int &N, const float &k_tar, const double &a, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const double gamma, CausetMPI &cmpi, MersenneRNG &mrng, Stopwatch &sGenerateNodes, const bool &growing, const bool &verbose, const bool &bench)
{
	#if DEBUG
	//Values are in correct ranges
	assert (!nodes.crd->isNull());
	assert (N > 0);
	assert (k_tar > 0.0f);
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") || spacetime.stdimIs("5"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"));
	if (!spacetime.manifoldIs("Hyperbolic")) {
		assert (a >= 0.0);
		assert (tau0 > 0.0);
	}
	if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		//assert (nodes.crd->getDim() == 4);
		//assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		//assert (nodes.crd->z() != NULL);
		//assert (spacetime.stdimIs("4"));
		assert (zeta < HALF_PI);
	} else if (spacetime.manifoldIs("De_Sitter")) {
		if (spacetime.curvatureIs("Positive")) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (spacetime.curvatureIs("Flat")) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (spacetime.manifoldIs("Hyperbolic")) {
		assert (r_max > 0.0);
		assert (zeta > 0.0);
	} else if (spacetime.manifoldIs("Polycone"))
		assert (gamma > 0.0);
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	#endif

	bool DEBUG_COORDS = false;
	//Enable this to validate the nodes are being generated with the correct
	//distributions - it will use rejection sampling from the slab's distributions
	bool DEBUG_DIAMOND = true;

	stopwatchStart(&sGenerateNodes);

	IntData *idata = NULL;
	double params[3];
	double xi = eta0 / sqrt(2.0);
	double w = (zeta - zeta1) / sqrt(2.0);
	double mu = 0.0, mu1 = 0.0, mu2 = 0.0;
	double p1 = 0.0;

	//Rejection sampling vs exact CDF inversion
	bool use_rejection = false;
	if (DEBUG_DIAMOND && spacetime.regionIs("Diamond"))
		use_rejection = true;
	if (spacetime.regionIs("Slab_T1") || spacetime.regionIs("Slab_S1") || spacetime.regionIs("Slab_TS") || spacetime.regionIs("Slab_N3") || spacetime.regionIs("Saucer_S") || spacetime.regionIs("Saucer_T") || spacetime.regionIs("Triangle_T") || spacetime.regionIs("Triangle_S"))
		use_rejection = true;

	//Initialize GSL integration structure
	//There is one 'workspace' per OpenMP thread to avoid
	//write conflicts in the for loop
	size_t i_size = (use_rejection ? 1 : omp_get_max_threads()) * sizeof(IntData);
	if ((USE_GSL || spacetime.regionIs("Diamond")) && spacetime.manifoldIs("FLRW")) {
		try {
			idata = (IntData*)malloc(i_size);
			if (idata == NULL)
				throw std::bad_alloc();
			memset(idata, 0, i_size);
		} catch (std::bad_alloc) {
			fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
			return false;
		}

		for (int i = 0; i < (int)(i_size / sizeof(IntData)); i++) {
			idata[i] = IntData();
			//Modify these two parameters to trade off between speed and accuracy
			idata[i].limit = 50;
			idata[i].tol = 1e-4;
			idata[i].workspace = gsl_integration_workspace_alloc(idata[i].nintervals);
		}
	}

	//Initialize constants
	if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_S", "Flat", "Temporal")) {
		mu1 = volume_77834_1(1.5);
		mu2 = volume_77834_1(-1.5);
		mu = mu1 - mu2;
	} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Flat", "None")) {
		mu = POW2(2.0 * xi + w) / (4.0 * xi * (xi + w));
		mu1 = (2.0 * xi) / (2.0 * xi + w);
	} else if (spacetime.spacetimeIs("2", "Polycone", "Slab", "Positive", "None"))
		mu = gamma / (gamma - 2.0);
	else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Flat", "None"))
		mu = LOG(POW2(w + 2.0 * xi, EXACT) / (4.0 * xi * (w + xi)), STL) - POW2(w / (w + 2.0 * xi), EXACT);
	else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Positive", "None"))
		mu = LOG(0.5 * (1.0 / COS(sqrt(2.0) * xi, APPROX ? FAST : STL) + 1.0), STL) - 1.0 / POW2(COS(xi / sqrt(2.0), APPROX ? FAST : STL), EXACT) + 1.0;
	else if (spacetime.spacetimeIs("4", "FLRW", "Diamond", "Flat", "None")) {
		params[0] = tau0;
		params[1] = HALF_PI - zeta;
		params[2] = zeta1;
		(*idata).limit = 100;
		(*idata).tol = 1e-8;
		(*idata).upper = zeta1;
		mu1 = integrate1D(&volume_11396_0_lower, params, idata, QAGS);
		(*idata).lower = (*idata).upper;
		(*idata).upper = tau0;
		mu2 = integrate1D(&volume_11396_0_upper, params, idata, QAGS);
		mu = mu1 + mu2;
		p1 = mu1 / mu;
		(*idata).limit = 50;
		(*idata).tol = 1e-4;
	} else
		mu = 1.0;

	#ifndef _OPENMP
	UGenerator &urng = mrng.rng;
	NDistribution ndist(0.0, 1.0);
	NGenerator nrng(mrng.eng, ndist);
	#endif

	//Generate coordinates for each of N nodes
	int mpi_chunk = N / cmpi.num_mpi_threads;
	int mpi_offset = mpi_chunk * cmpi.rank;
	if (use_rejection) {
		#ifdef _OPENMP
		UGenerator &urng = mrng.rng;
		NDistribution ndist(0.0, 1.0);
		NGenerator nrng(mrng.eng, ndist);
		#endif
		#if EMBED_NODES
		float2 emb2;
		float3 emb3;
		float4 emb4;
		#endif
		double eta;

		//Use the rejection method
		int i = 0;
		while (i < N) {
			if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_S", "Flat", "Temporal")) {
				#if SPECIAL_SAUCER
				nodes.crd->x(i) = 2.0 * urng() - 1.0;
				nodes.crd->y(i) = 3.0 * urng() - 1.5;
				if (fabs(nodes.crd->x(i)) > eta_77834_1(nodes.crd->y(i), eta0))
					continue;
				#else
				nodes.crd->x(i) = (2.0 * urng() - 1.0) * eta0;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if (fabs(nodes.crd->x(i)) > eta_77834_1(nodes.crd->y(i), eta0))
					continue;
				#endif
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_T1", "Flat", "Temporal")) {
				nodes.crd->x(i) = (2.0 * urng() - 1.0) * eta0;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * eta0;
				if (fabs(nodes.crd->y(i)) > eta_75499530_2(nodes.crd->x(i), r_max, eta0))
					continue;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_S1", "Flat", "Temporal")) {
				nodes.crd->x(i) = (2.0 * urng() - 1.0) * r_max;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if (fabs(nodes.crd->x(i)) > eta_75499530_2(nodes.crd->y(i), eta0, r_max))
					continue;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_TS", "Flat", "Temporal")) {
				nodes.crd->x(i) = urng() - 0.5;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if ((fabs(nodes.crd->y(i)) <= 0.5 && fabs(nodes.crd->x(i)) > eta_75499530_2(nodes.crd->y(i), eta0, 0.5)) ||
				    (fabs(nodes.crd->y(i)) > 0.5 && fabs(nodes.crd->y(i)) > eta_75499530_2(nodes.crd->x(i), r_max, 0.5)))
					continue;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_N3", "Flat", "None")) {
				nodes.crd->x(i) = 1.5 * urng();
				nodes.crd->y(i) = urng() - 0.5;
				if (nodes.crd->x(i) <= 0.5 && (fabs(nodes.crd->y(i)) > nodes.crd->x(i)))
					continue;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Saucer_T", "Flat", "Temporal")) {
				#if SPECIAL_SAUCER
				nodes.crd->x(i) = 3.0 * urng() - 1.5;
				nodes.crd->y(i) = 2.0 * urng() - 1.0;
				if (fabs(nodes.crd->y(i)) > eta_77834_1(nodes.crd->x(i), r_max))
					continue;
				#else
				nodes.crd->x(i) = (2.0 * urng() - 1.0) * eta0;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if (fabs(nodes.crd->y(i)) > eta_77834_1(nodes.crd->x(i), r_max))
					continue;
				#endif
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_T", "Flat", "Temporal")) {
				nodes.crd->x(i) = (2.0 * urng() - 1.0) * eta0;
				nodes.crd->y(i) = urng() * r_max;
				if (fabs(nodes.crd->x(i)) > eta_76546058_2(nodes.crd->y(i), eta0, r_max))
					continue;
			} else if (spacetime.spacetimeIs("2", "Minkowski", "Triangle_S", "Flat", "None")) {
				nodes.crd->x(i) = urng() * eta0;
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if (fabs(nodes.crd->y(i)) > eta_76546058_2(nodes.crd->x(i), r_max, eta0))
					continue;
			} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Flat", "None")) {
				nodes.crd->x(i) = (HALF_PI - zeta1) / (urng() * (1.0 + HALF_PI - zeta1) - (HALF_PI - zeta1));
				nodes.crd->y(i) = ((HALF_PI - zeta1 + 1.0) / 2.0) * (2.0 * urng() - 1.0);
				if (!iad(nodes.crd->x(i), nodes.crd->y(i), HALF_PI - zeta, HALF_PI - zeta1))
					continue;
				nodes.id.tau[i] = etaToTauFlat(nodes.crd->x(i));
			} else if (spacetime.spacetimeIs("2", "Dust", "Diamond", "Flat", "None")) {
				//nodes.crd->x(i) = get_radius(urng, r_max, 6.0);
				nodes.id.tau[i] = tau0 * pow(urng(), 0.6);
				nodes.crd->x(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
				nodes.crd->y(i) = (2.0 * urng() - 1.0) * r_max;
				if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0))
					continue;
				//nodes.id.tau[i] = etaToTauDust(nodes.crd->x(i), a, alpha);
			} else if (spacetime.spacetimeIs("3", "Minkowski", "Diamond", "Flat", "None")) {
				nodes.crd->x(i) = urng() * eta0;
				nodes.crd->y(i) = get_radius(urng, eta0 * 0.5, 3);
				if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0))
					continue;
				nodes.crd->z(i) = urng() * TWO_PI;
			} else if (spacetime.spacetimeIs("3", "De_Sitter", "Diamond", "Flat", "None")) {
				nodes.crd->x(i) = -sqrtf(-POW2(HALF_PI - zeta1) / (urng() * (POW2(HALF_PI - zeta1) - 1.0) - POW2(HALF_PI - zeta1)));
				nodes.crd->y(i) = get_radius(urng, r_max, 3);
				if (!iad(nodes.crd->x(i), nodes.crd->y(i), -1.0, HALF_PI - zeta1))
					continue;
				nodes.crd->z(i) = urng() * TWO_PI;
				nodes.id.tau[i] = etaToTauFlat(nodes.crd->x(i));
			} else if (spacetime.spacetimeIs("3", "Dust", "Diamond", "Flat", "None")) {
				nodes.id.tau[i] = tau0 * pow(urng(), 3.0 / 7.0);
				nodes.crd->x(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
				nodes.crd->y(i) = get_radius(urng, r_max, 3);
				if (!iad(nodes.crd->x(i), nodes.crd->y(i), 0.0, eta0))
					continue;
				nodes.crd->z(i) = get_azimuthal_angle(urng);
			} else if (spacetime.spacetimeIs("4", "Minkowski", "Diamond", "Flat", "None")) {
				nodes.crd->w(i) = urng() * eta0;
				nodes.crd->x(i) = get_radius(urng, eta0 * 0.5, 4);
				if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, eta0))
					continue;
				nodes.crd->y(i) = get_zenith_angle(urng);
				nodes.crd->z(i) = get_azimuthal_angle(urng);
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Flat", "None")) {
				#if EMBED_NODES
				nodes.crd->v(i) = get_4d_asym_flat_deSitter_slab_eta(urng, HALF_PI - zeta, HALF_PI - zeta1);
				emb3 = get_4d_asym_flat_deSitter_slab_cartesian(urng, nrng, r_max);
				r = sqrt(POW2(emb3.x, EXACT) + POW2(emb3.y, EXACT) + POW2(emb3.z, EXACT));
				if (!iad(nodes.crd->v(i), r, eta0, HALF_PI  - zeta, HALF_PI - zeta1))
					continue;
				nodes.id.tau[i] = etaToTauFlat(nodes.crd->v(i));
				nodes.crd->x(i) = emb3.x;
				nodes.crd->y(i) = emb3.y;
				nodes.crd->z(i) = emb3.z;
				#else
				nodes.crd->w(i) = get_4d_asym_flat_deSitter_slab_eta(urng, HALF_PI - zeta, HALF_PI - zeta1);
				nodes.crd->x(i) = get_4d_asym_flat_deSitter_slab_radius(urng, r_max);
				if (!iad(nodes.crd->w(i), nodes.crd->x(i), HALF_PI - zeta, HALF_PI - zeta1))
					continue;
				nodes.id.tau[i] = etaToTauFlat(nodes.crd->w(i));
				nodes.crd->y(i) = get_4d_asym_flat_deSitter_diamond_theta2(urng);
				nodes.crd->z(i) = get_4d_asym_flat_deSitter_diamond_theta3(urng);
				#endif
			} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Positive", "None")) {
				#if EMBED_NODES
				nodes.crd->v(i) = get_4d_asym_sph_deSitter_slab_eta(urng, zeta);
				emb4 = get_4d_asym_sph_deSitter_slab_emb(nrng);
				r = acosf(emb4.w);
				if (!iad(nodes.crd->v(i), r, 0.0, eta0))
					continue;
				nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
				nodes.crd->w(i) = emb4.w;
				nodes.crd->x(i) = emb4.x;
				nodes.crd->y(i) = emb4.y;
				nodes.crd->z(i) = emb4.z;
				#else
				nodes.crd->w(i) = get_4d_asym_sph_deSitter_slab_eta(urng, zeta);
				nodes.crd->x(i) = get_4d_asym_sph_deSitter_slab_theta1(urng);
				if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta))
					continue;
				nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
				nodes.crd->y(i) = get_4d_asym_sph_deSitter_diamond_theta2(urng);
				nodes.crd->z(i) = get_4d_asym_sph_deSitter_diamond_theta3(urng);
				#endif
			} else if (spacetime.spacetimeIs("4", "Dust", "Diamond", "Flat", "None")) {
				nodes.id.tau[i] = get_4d_asym_flat_dust_slab_tau(urng, tau0);
				#if EMBED_NODES
				nodes.crd->v(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
				emb3 = get_4d_asym_flat_dust_slab_cartesian(urng, nrng, r_max);
				r = sqrt(POW2(emb3.x, EXACT) + POW2(emb3.y, EXACT) + POW2(emb3.z, EXACT));
				if (!iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta))
					continue;
				nodes.crd->x(i) = emb3.x;
				nodes.crd->y(i) = emb3.y;
				nodes.crd->z(i) = emb3.z;
				#else
				nodes.crd->w(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
				nodes.crd->x(i) = get_4d_asym_flat_dust_slab_radius(urng, r_max);
				if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta))
					continue;
				nodes.crd->y(i) = get_4d_asym_flat_dust_diamond_theta2(urng);
				nodes.crd->z(i) = get_4d_asym_flat_dust_diamond_theta3(urng);
				#endif
			} else if (spacetime.spacetimeIs("4", "FLRW", "Diamond", "Flat", "None")) {
				nodes.id.tau[i] = get_4d_asym_flat_flrw_slab_tau(urng, tau0);
				if (USE_GSL) {
					(*idata).lower = 0.0;
					(*idata).upper = nodes.id.tau[i];
					eta = integrate1D(&tauToEtaFLRW, NULL, idata, QAGS) * a / alpha;
				} else
					eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
				#if EMBED_NODES
				nodes.crd->v(i) = eta;
				emb3 = get_4d_asym_flat_flrw_slab_cartesian(urng, nrng, r_max);
				r = sqrt(POW2(emb3.x, EXACT) + POW2(emb3.y, EXACT) + POW2(emb3.z, EXACT));
				if (!iad(nodes.crd->v(i), r, 0.0, HALF_PI - zeta))
					continue;
				nodes.crd->x(i) = emb3.x;
				nodes.crd->y(i) = emb3.y;
				nodes.crd->z(i) = emb3.z;
				#else
				nodes.crd->w(i) = eta;
				nodes.crd->x(i) = get_4d_asym_flat_flrw_slab_radius(urng, r_max);
				if (!iad(nodes.crd->w(i), nodes.crd->x(i), 0.0, HALF_PI - zeta))
					continue;
				nodes.crd->y(i) = get_4d_asym_flat_flrw_slab_theta2(urng);
				nodes.crd->z(i) = get_4d_asym_flat_flrw_slab_theta3(urng);
				#endif
			} else if (spacetime.spacetimeIs("5", "Minkowski", "Diamond", "Flat", "None")) {
				nodes.crd->v(i) = urng() * eta0;
				nodes.crd->w(i) = get_radius(urng, eta0 * 0.5, 5);
				if (!iad(nodes.crd->v(i), nodes.crd->w(i), 0.0, eta0))
					continue;
				nodes.crd->x(i) = get_5d_asym_flat_minkowski_diamond_theta1(urng);
				nodes.crd->y(i) = get_5d_asym_flat_minkowski_diamond_theta2(urng);
				nodes.crd->z(i) = get_5d_asym_flat_minkowski_diamond_theta3(urng);
			} else {
				fprintf(stderr, "Spacetime parameters not supported!\n");
				assert (false);
			}

			#if DEBUG
			if (!validateCoordinates(nodes, spacetime, eta0, zeta, zeta1, r_max, tau0, i))
				i--;
			#endif

			i++;
		}
	} else {
		//Use exact CDF inversion formulae (see notes)
		int start = mpi_offset;
		int finish = start + mpi_chunk;

		#ifdef _OPENMP
		unsigned int seed = static_cast<unsigned int>(mrng.rng() * 400000000);
		#pragma omp parallel if (N < 1000)
		{
		//Initialize one RNG per thread
		Engine eng(seed ^ omp_get_thread_num());
		UDistribution udist(0.0, 1.0);
		UGenerator urng(eng, udist);
		NDistribution ndist(0.0, 1.0);
		NGenerator nrng(eng, ndist);
		#pragma omp for schedule(static, 128)
		#endif
		for (int i = start; i < finish; i++) {
			#if EMBED_NODES
			float2 emb2;
			float3 emb3;
			float4 emb4;
			#endif
			double eta, r;
			double u, v;
			int tid = omp_get_thread_num();

			do {
				if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Flat", "Temporal")) {
					nodes.crd->x(i) = get_2d_sym_flat_minkowski_slab_eta(urng, eta0);
					nodes.crd->y(i) = get_2d_sym_flat_minkowski_slab_radius(urng, r_max);
				} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab", "Positive", "Temporal")) {
					nodes.crd->x(i) = get_2d_sym_sph_minkowski_slab_eta(urng, eta0);
					nodes.crd->y(i) = get_2d_sym_sph_minkowski_slab_theta(urng);
				} else if (spacetime.spacetimeIs("2", "Minkowski", "Diamond", "Flat", "None")) {
					u = get_2d_asym_flat_minkowski_diamond_u(urng, xi);
					v = get_2d_asym_flat_minkowski_diamond_v(urng, xi);
					nodes.crd->x(i) = (u + v) / sqrt(2.0);
					nodes.crd->y(i) = (u - v) / sqrt(2.0);
				} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "None")) {
					nodes.crd->x(i) = get_2d_asym_sph_deSitter_slab_eta(urng, eta0);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
					#if EMBED_NODES
					emb2 = get_2d_asym_sph_deSitter_slab_emb(urng);
					nodes.crd->y(i) = emb2.x;
					nodes.crd->z(i) = emb2.y;
					#else
					nodes.crd->y(i) = get_2d_asym_sph_deSitter_slab_theta(urng);
					#endif
				} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Positive", "Temporal")) {
					nodes.crd->x(i) = get_2d_sym_sph_deSitter_slab_eta(urng, eta0);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
					#if EMBED_NODES
					emb2 = get_2d_sym_sph_deSitter_slab_emb(urng);
					nodes.crd->y(i) = emb2.x;
					nodes.crd->z(i) = emb2.y;
					#else
					nodes.crd->y(i) = get_2d_sym_sph_deSitter_slab_theta(urng);
					#endif
				} else if (spacetime.spacetimeIs("2", "De_Sitter", "Slab", "Negative", "None")) {
					nodes.id.tau[i] = get_2d_asym_hyp_deSitter_slab_tau(urng, tau0);
					nodes.crd->x(i) = tauToEtaHyp(nodes.id.tau[i]);
					nodes.crd->y(i) = get_2d_asym_hyp_deSitter_slab_theta(urng);
				} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Flat", "None")) {
					u = get_2d_asym_flat_deSitter_diamond_u(urng, mu, mu1, xi, w);
					v = get_2d_asym_flat_deSitter_diamond_v(urng, u, xi, w);
					nodes.crd->x(i) = (u + v) / sqrt(2.0);
					nodes.crd->y(i) = (u - v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauFlat(nodes.crd->x(i));
				} else if (spacetime.spacetimeIs("2", "De_Sitter", "Diamond", "Positive", "None")) {
					nodes.crd->x(i) = get_2d_asym_sph_deSitter_diamond_eta(urng);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
					#if EMBED_NODES
					emb2 = get_2d_asym_sph_deSitter_diamond_emb(mrng.rng, nodes.crd->x(i));
					nodes.crd->y(i) = emb2.x;
					nodes.crd->z(i) = emb2.y;
					#else
					nodes.crd->y(i) = get_2d_asym_sph_deSitter_diamond_theta(urng, nodes.crd->x(i));
					#endif
				} else if (spacetime.spacetimeIs("2", "Hyperbolic", "Slab", "Positive", "None")) {
					assert (!EMBED_NODES);
					if (growing)
						nodes.crd->x(i) = get_2d_asym_sph_hyperbolic_slab_nonuniform_radius(urng, r_max, zeta);
					else
						nodes.crd->x(i) = get_2d_asym_sph_hyperbolic_slab_radius(urng, r_max, zeta);
					nodes.id.tau[i] = nodes.crd->x(i) / zeta;
					nodes.crd->y(i) = get_2d_asym_sph_hyperbolic_slab_theta(urng);
				} else if (spacetime.spacetimeIs("2", "Polycone", "Slab", "Positive", "None")) {
					nodes.id.tau[i] = get_2d_asym_sph_polycone_slab_tau(urng, tau0, mu);
					nodes.crd->x(i) = tauToEtaPolycone(nodes.id.tau[i], a, gamma);
					nodes.crd->y(i) = get_2d_asym_sph_polycone_slab_theta(urng);
				} else if (spacetime.spacetimeIs("3", "Minkowski", "Slab", "Flat", "Temporal")) {
					assert (!EMBED_NODES);
					nodes.crd->x(i) = get_3d_sym_flat_minkowski_slab_eta(urng, eta0);
					nodes.crd->y(i) = get_3d_sym_flat_minkowski_slab_radius(urng, r_max);
					nodes.crd->z(i) = get_3d_sym_flat_minkowski_slab_theta(urng);
				} else if (spacetime.spacetimeIs("3", "Minkowski", "Diamond", "Flat", "None")) {
					u = get_3d_asym_flat_minkowski_diamond_u(urng, eta0);
					v = get_3d_asym_flat_minkowski_diamond_v(urng, u);
					nodes.crd->x(i) = (u + v) / sqrt(2.0);
					nodes.crd->y(i) = (u - v) / sqrt(2.0);
					nodes.crd->z(i) = get_3d_asym_flat_minkowski_diamond_theta(urng);
				} else if (spacetime.spacetimeIs("3", "Minkowski", "Cube", "Flat", "None")) {
					assert (!EMBED_NODES);
					nodes.crd->x(i) = get_3d_asym_flat_minkowski_cube_eta(urng, eta0);
					nodes.crd->y(i) = get_3d_asym_flat_minkowski_cube_x(urng, r_max);
					nodes.crd->z(i) = get_3d_asym_flat_minkowski_cube_y(urng, r_max);
				} else if (spacetime.spacetimeIs("4", "Minkowski", "Diamond", "Flat", "None")) {
					u = get_4d_asym_flat_minkowski_diamond_u(urng, eta0);
					v = get_4d_asym_flat_minkowski_diamond_v(urng, eta0, u);
					nodes.crd->w(i) = (u + v) / sqrt(2.0);
					nodes.crd->x(i) = (u - v) / sqrt(2.0);
					nodes.crd->y(i) = get_4d_asym_flat_minkowski_diamond_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_minkowski_diamond_theta3(urng);
				} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Flat", "None")) {
					#if EMBED_NODES
					nodes.crd->v(i) = get_4d_asym_flat_deSitter_slab_eta(urng, HALF_PI - zeta, HALF_PI - zeta1);
					nodes.id.tau[i] = etaToTauFlat(nodes.crd->v(i));
					emb3 = get_4d_asym_flat_deSitter_slab_cartesian(urng, nrng, r_max);
					nodes.crd->x(i) = emb3.x;
					nodes.crd->y(i) = emb3.y;
					nodes.crd->z(i) = emb3.z;
					#else
					nodes.crd->w(i) = get_4d_asym_flat_deSitter_slab_eta(urng, HALF_PI - zeta, HALF_PI - zeta1);
					nodes.id.tau[i] = etaToTauFlat(nodes.crd->w(i));
					nodes.crd->x(i) = get_4d_asym_flat_deSitter_slab_radius(mrng.rng, r_max);
					nodes.crd->y(i) = get_4d_asym_flat_deSitter_slab_theta2(mrng.rng);
					nodes.crd->z(i) = get_4d_asym_flat_deSitter_slab_theta3(mrng.rng);
					#endif
				} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "None")) {
					#if EMBED_NODES
					nodes.crd->v(i) = get_4d_asym_sph_deSitter_slab_eta(urng, zeta);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
					emb4 = get_4d_asym_sph_deSitter_slab_emb(nrng);
					nodes.crd->w(i) = emb4.w;
					nodes.crd->x(i) = emb4.x;
					nodes.crd->y(i) = emb4.y;
					nodes.crd->z(i) = emb4.z;
					#else
					nodes.crd->w(i) = get_4d_asym_sph_deSitter_slab_eta(urng, zeta);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
					nodes.crd->x(i) = get_4d_asym_sph_deSitter_slab_theta1(urng);
					nodes.crd->y(i) = get_4d_asym_sph_deSitter_slab_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_sph_deSitter_slab_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "Temporal")) {
					#if EMBED_NODES
					nodes.crd->v(i) = get_4d_sym_sph_deSitter_slab_eta(urng, zeta);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
					emb4 = get_4d_sym_sph_deSitter_slab_emb(nrng);
					nodes.crd->w(i) = emb4.w;
					nodes.crd->x(i) = emb4.x;
					nodes.crd->y(i) = emb4.y;
					nodes.crd->z(i) = emb4.z;
					#else
					nodes.crd->w(i) = get_4d_sym_sph_deSitter_slab_eta(urng, zeta);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
					nodes.crd->x(i) = get_4d_sym_sph_deSitter_slab_theta1(urng);
					nodes.crd->y(i) = get_4d_sym_sph_deSitter_slab_theta2(urng);
					nodes.crd->z(i) = get_4d_sym_sph_deSitter_slab_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Flat", "None")) {
					u = get_4d_asym_flat_deSitter_diamond_u(urng, xi, mu);
					v = get_4d_asym_flat_deSitter_diamond_v(urng, u, xi);
					#if EMBED_NODES
					nodes.crd->v(i) = (u + v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauFlat(nodes.crd->v(i));
					emb3 = get_4d_asym_flat_deSitter_diamond_cartesian(urng, nrng);
					nodes.crd->x(i) = emb3.x;
					nodes.crd->y(i) = emb3.y;
					nodes.crd->z(i) = emb3.z;
					#else
					nodes.crd->w(i) = (u + v) / sqrt(2.0);
					nodes.crd->x(i) = (u - v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauFlat(nodes.crd->w(i));
					nodes.crd->y(i) = get_4d_asym_flat_deSitter_diamond_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_deSitter_diamond_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "De_Sitter", "Diamond", "Positive", "None")) {
					u = get_4d_asym_sph_deSitter_diamond_u(urng, xi, mu);
					v = get_4d_asym_sph_deSitter_diamond_v(urng, u);
					#if EMBED_NODES
					nodes.crd->v(i) = (u + v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
					emb4 = get_4d_asym_sph_deSitter_diamond_emb(urng, nrng, u, v);
					nodes.crd->w(i) = emb4.w;
					nodes.crd->x(i) = emb4.x;
					nodes.crd->y(i) = emb4.y;
					nodes.crd->z(i) = emb4.z;
					#else
					nodes.crd->w(i) = (u + v) / sqrt(2.0);
					nodes.crd->x(i) = (u - v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
					nodes.crd->y(i) = get_4d_asym_sph_deSitter_diamond_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_sph_deSitter_diamond_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "Dust", "Slab", "Flat", "None")) {
					nodes.id.tau[i] = get_4d_asym_flat_dust_slab_tau(urng, tau0);
					#if EMBED_NODES
					nodes.crd->v(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
					emb3 = get_4d_asym_flat_dust_slab_cartesian(urng, nrng, r_max);
					nodes.crd->x(i) = emb3.x;
					nodes.crd->y(i) = emb3.y;
					nodes.crd->z(i) = emb3.z;
					#else
					nodes.crd->w(i) = tauToEtaDust(nodes.id.tau[i], a, alpha);
					nodes.crd->x(i) = get_4d_asym_flat_dust_slab_radius(urng, r_max);
					nodes.crd->y(i) = get_4d_asym_flat_dust_slab_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_dust_slab_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "Dust", "Diamond", "Flat", "None")) {
					u = get_4d_asym_flat_dust_diamond_u(urng, xi);
					v = get_4d_asym_flat_dust_diamond_v(urng, u);
					#if EMBED_NODES
					nodes.crd->v(i) = (u + v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauDust(nodes.crd->v(i), a, alpha);
					emb3 = get_4d_asym_flat_dust_diamond_cartesian(urng, nrng, u, v);
					nodes.crd->x(i) = emb3.x;
					nodes.crd->y(i) = emb3.y;
					nodes.crd->z(i) = emb3.z;
					#else
					nodes.crd->w(i) = (u + v) / sqrt(2.0);
					nodes.id.tau[i] = etaToTauDust(nodes.crd->w(i), a, alpha);
					nodes.crd->x(i) = (u - v) / sqrt(2.0);
					nodes.crd->y(i) = get_4d_asym_flat_dust_diamond_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_dust_diamond_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Flat", "None")) {
					nodes.id.tau[i] = get_4d_asym_flat_flrw_slab_tau(urng, tau0);
					if (USE_GSL) {
						idata[tid].lower = 0.0;
						idata[tid].upper = nodes.id.tau[i];
						eta = integrate1D(&tauToEtaFLRW, NULL, &idata[tid], QAGS) * a / alpha;
					} else
						eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
					#if EMBED_NODES
					nodes.crd->v(i) = eta;
					emb3 = get_4d_asym_flat_flrw_slab_cartesian(urng, nrng, r_max);
					nodes.crd->x(i) = emb3.x;
					nodes.crd->y(i) = emb3.y;
					nodes.crd->z(i) = emb3.z;
					#else
					nodes.crd->w(i) = eta;
					nodes.crd->x(i) = get_4d_asym_flat_flrw_slab_radius(urng, r_max);
					nodes.crd->y(i) = get_4d_asym_flat_flrw_slab_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_flrw_slab_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "FLRW", "Slab", "Positive", "None")) {
					nodes.id.tau[i] = get_4d_asym_sph_flrw_slab_tau(urng, tau0);
					if (USE_GSL) {
						idata[tid].lower = 0.0;
						idata[tid].upper = nodes.id.tau[i];
						eta = integrate1D(&tauToEtaFLRW, NULL, &idata[tid], QAGS) * a / alpha;
					} else
						eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
					#if EMBED_NODES
					nodes.crd->v(i) = eta;
					emb4 = get_4d_asym_sph_flrw_slab_cartesian(nrng);
					nodes.crd->w(i) = emb4.w;
					nodes.crd->x(i) = emb4.x;
					nodes.crd->y(i) = emb4.y;
					nodes.crd->z(i) = emb4.z;
					#else
					nodes.crd->w(i) = eta;
					nodes.crd->x(i) = get_4d_asym_sph_flrw_slab_theta1(urng);
					nodes.crd->y(i) = get_4d_asym_sph_flrw_slab_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_sph_flrw_slab_theta3(urng);
					#endif
				} else if (spacetime.spacetimeIs("4", "FLRW", "Diamond", "Flat", "None")) {
					nodes.id.tau[i] = get_4d_asym_flat_flrw_diamond_tau(urng, &idata[tid], params, tau0, zeta1, p1, mu, mu1);
					if (USE_GSL) {
						idata[tid].lower = 0.0;
						idata[tid].upper = nodes.id.tau[i];
						eta = integrate1D(&tauToEtaFLRW, NULL, &idata[tid], QAGS) * a / alpha;
					} else
						eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
					r = get_4d_asym_flat_flrw_diamond_radius(urng, eta, zeta);
					#if EMBED_NODES
					nodes.crd->v(i) = eta;
					emb3 = get_sph_d3(nrng);
					nodes.crd->x(i) = r * emb3.x;
					nodes.crd->y(i) = r * emb3.y;
					nodes.crd->z(i) = r * emb3.z;
					#else
					nodes.crd->w(i) = eta;
					nodes.crd->x(i) = r;
					nodes.crd->y(i) = get_4d_asym_flat_flrw_diamond_theta2(urng);
					nodes.crd->z(i) = get_4d_asym_flat_flrw_diamond_theta3(urng);
					#endif
					break;
				} else if (spacetime.spacetimeIs("5", "Minkowski", "Diamond", "Flat", "None")) {
					u = get_5d_asym_flat_minkowski_diamond_u(urng, eta0);
					v = get_5d_asym_flat_minkowski_diamond_v(urng, u);
					nodes.crd->v(i) = (u + v) / sqrt(2.0);
					nodes.crd->w(i) = (u - v) / sqrt(2.0);
					nodes.crd->x(i) = get_5d_asym_flat_minkowski_diamond_theta1(urng);
					nodes.crd->y(i) = get_5d_asym_flat_minkowski_diamond_theta2(urng);
					nodes.crd->z(i) = get_5d_asym_flat_minkowski_diamond_theta3(urng);
				} else {
					fprintf(stderr, "Spacetime parameters not supported!\n");
					assert (false);
				}
			} while (!validateCoordinates(nodes, spacetime, eta0, zeta, zeta1, r_max, tau0, i));
		}
		#ifdef _OPENMP
		}
		#endif
	}

	//Free GSL workspace memory
	if ((USE_GSL || spacetime.regionIs("Diamond")) && spacetime.manifoldIs("FLRW")) {
		for (int i = 0; i < (int)(i_size / sizeof(IntData)); i++)
			gsl_integration_workspace_free(idata[i].workspace);
		free(idata);
		idata = NULL;
	}

	#ifdef MPI_ENABLED
	if (nodes.id.tau != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.id.tau, mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	if (nodes.crd->v() != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.crd->v(), mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	if (nodes.crd->w() != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.crd->w(), mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	if (nodes.crd->x() != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.crd->x(), mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	if (nodes.crd->y() != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.crd->y(), mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	if (nodes.crd->z() != NULL)
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nodes.crd->z(), mpi_chunk, MPI_FLOAT, MPI_COMM_WORLD);
	#endif

	//Debugging statements used to check coordinate distributions
	//if (cmpi.rank == 0 && !printValues(nodes, spacetime, N, "tau_dist_rank0.cset.dbg.dat", "tau")) return false;
	//if (cmpi.rank == 1 && !printValues(nodes, spacetime, N, "tau_dist_rank1.cset.dbg.dat", "tau")) return false;
	//if (!printValues(nodes, spacetime, N, "tau_dist.cset.dbg.dat", "tau")) return false;
	//if (!printValues(nodes, spacetime, N, "eta_dist.cset.dbg.dat", "eta")) return false;
	//if (!printValues(nodes, spacetime, N, "radial_dist.cset.dbg.dat", "x")) return false;
	//if (!printValues(nodes, spacetime, N, "u_dist_rej.cset.dbg.dat", "u")) return false;
	//if (!printValues(nodes, spacetime, N, "v_dist_rej.cset.dbg.dat", "v")) return false;
	//if (!printValues(nodes, spacetime, N, "x_dist.cset.dbg.dat", "y")) return false;
	//if (!printValues(nodes, spacetime, N, "theta1_dist.cset.dbg.dat", "theta1")) return false;
	//if (!printValues(nodes, spacetime, N, "theta2_dist.cset.dbg.dat", "theta2")) return false;
	//if (!printValues(nodes, spacetime, N, "theta3_dist.cset.dbg.dat", "theta3")) return false;
	/*printf_red();
	printf("Check coordinate distributions now.\n");
	printf_std();
	fflush(stdout);
	printChk();*/

	if (DEBUG_COORDS) {
		if (nodes.id.tau != NULL)
			printValues(nodes, spacetime, N, "tau_dist.cset.dbg.dat", "tau");
		if (spacetime.stdimIs("2")) {
			printValues(nodes, spacetime, N, "eta_dist.cset.dbg.dat", "x");
			if (spacetime.curvatureIs("Flat"))
				printValues(nodes, spacetime, N, "x_dist.cset.dbg.dat", "y");
			else
				printValues(nodes, spacetime, N, "theta_dist.cset.dbg.dat", "y");
		} else if (spacetime.stdimIs("4")) {
			printValues(nodes, spacetime, N, "eta_dist.cset.dbg.dat", "w");
			if (spacetime.curvatureIs("Flat"))
				printValues(nodes, spacetime, N, "radial_dist.cset.dbg.dat", "x");
			else if (spacetime.curvatureIs("Positive"))
				printValues(nodes, spacetime, N, "theta1_dist.cset.dbg.dat", "x");
			printValues(nodes, spacetime, N, "theta2_dist.cset.dbg.dat", "y");
			printValues(nodes, spacetime, N, "theta3_dist.cset.dbg.dat", "z");
		}
		printf_red();
		printf("\tCheck coordinate distributions.\n");
		printf_std();
		fflush(stdout);
	}

	stopwatchStop(&sGenerateNodes);

	if (!bench) {
		printf_mpi(cmpi.rank, "\tNodes Successfully Generated.\n");
		fflush(stdout);
	}

	if (verbose) {
		printf_mpi(cmpi.rank, "\t\tExecution Time: %5.6f sec\n", sGenerateNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool generateKROrder(Node &nodes, Bitvector &adj, const int N_tar, const int N, int &N_res, float &k_res, int &N_deg2, Stopwatch &sGenKR, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (nodes.k_in != NULL);
	assert (nodes.k_out != NULL);
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	#endif

	int rank = 0;

	stopwatchStart(&sGenKR);

	Engine eng((long)time(NULL));
	PDistribution pdist(N_tar >> 1);
	PGenerator prng(eng, pdist);
	for (int i = 0; i < 1000; i++) prng();

	int mid = prng();
	//printf("size of middle layer: %d\n", mid);

	PDistribution pdist2(N_tar >> 2);
	prng.distribution().param(pdist2.param());
	prng.distribution().reset();

	int low = prng();
	//printf("size of first layer: %d\n", low);

	int high = N - low - mid;
	//printf("size of final layer: %d\n", high);

	for (int i = 0; i < low; i++) {
		for (int j = low; j < N; j++) {
			adj[i].set(j);
			adj[j].set(i);
		}
		nodes.k_out[i] = N - low;
	}
	for (int i = low; i < low + mid; i++) {
		for (int j = low + mid; j < N; j++) {
			adj[i].set(j);
			adj[j].set(i);
		}
		nodes.k_in[i] = low;
		nodes.k_out[i] = high;
	}
	for (int i = low + mid; i < N; i++)
		nodes.k_in[i] = low + mid;

	N_res = N;
	N_deg2 = N;
	uint64_t nlinks = (uint64_t)mid * (low + high) + (uint64_t)low * high;
	k_res = (long double)nlinks / N_res;

	stopwatchStop(&sGenKR);

	if (!bench) {
		printf_mpi(rank, "\tCauset Successfully Connected.\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tFirst Layer:\t%d elements\n", low);
		printf_mpi(rank, "\t\tSecond Layer:\t%d elements\n", mid);
		printf_mpi(rank, "\t\tThird Layer:\t%d elements\n", high);
		printf_mpi(rank, "\t\tUndirected Relations:     %" PRIu64 "\n", nlinks);
		printf_mpi(rank, "\t\tResulting Average Degree: %f\n", k_res);
		if (!rank) printf_std();
		if (!rank) fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sGenKR.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool generateRandomOrder(Node &nodes, Bitvector &adj, const int N, int &N_res, float &k_res, int &N_deg2, MersenneRNG &mrng, Stopwatch &sGenRandom, const bool verbose, const bool bench)
{
	#if DEBUG
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	#endif

	int rank = 0;

	stopwatchStart(&sGenRandom);

	for (int i = 0; i < N - 1; i++) {
		for (int j = i + 1; j < N; j++) {
			if (mrng.rng() < 0.1) {
				adj[i].set(j);
				adj[j].set(i);
			}
		}
	}

	transitiveClosure(adj, N);

	uint64_t nlinks = 0;
	for (int i = 0; i < N; i++) {
		if (i > 0)
			nodes.k_in[i] = (int)adj[i].partial_count(0, i);
		if (i < N - 1)
			nodes.k_out[i] = (int)adj[i].partial_count(i, N - i + 1);
		nlinks += nodes.k_out[i];
	}

	N_res = N;
	N_deg2 = N;
	k_res = (long double)nlinks / N_res;

	//if (!printDegrees(nodes, N, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;

	stopwatchStop(&sGenRandom);

	if (!bench) {
		printf_mpi(rank, "\tCauset Successfully Connected.\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tUndirected Relations:     %" PRIu64 "\n", nlinks);
		printf_mpi(rank, "\t\tResulting Average Degree: %f\n", k_res);
		if (!rank) printf_std();
		if (!rank) fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sGenRandom.elapsedTime);
		fflush(stdout);
	}

	return true;
}

bool linkNodes_v2(Node &nodes, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const double gamma, CausetMPI &cmpi, Stopwatch &sLinkNodes, const bool &link_epso, const bool &has_exact_k, const bool &use_bit, const bool &mpi_split, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (adj.size() >= (size_t)N);
	assert (N > 0);
	assert (k_tar > 0.0f);
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") || spacetime.stdimIs("5"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic") || spacetime.manifoldIs("Polycone"));
	if (!spacetime.manifoldIs("Hyperbolic"))
		assert (a > 0.0);
	assert (tau0 > 0.0);
	if (spacetime.manifoldIs("De_Sitter")) {
		if (spacetime.curvatureIs("Positive")) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (spacetime.curvatureIs("Flat")) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		//assert (nodes.crd->getDim() == 4);
		//assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		//assert (nodes.crd->z() != NULL);
		//assert (spacetime.stdimIs("4"));
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	} else if (spacetime.manifoldIs("Hyperbolic")) {
		assert (zeta > 0.0);
		assert (r_max > 0.0);
	} else if (spacetime.manifoldIs("Polycone"))
		assert (gamma > 2.0);
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	assert (use_bit);
	#endif

	#if EMBED_NODES
	if (spacetime.manifoldIs("Hyperbolic")) {
		fprintf(stderr, "linkNodes_v2 not implemented for EMBED_NODES=true and MANIFOLD=HYPERBOLIC.  Find me on line %d in %s.\n", __LINE__, __FILE__);
		if (!!N)
			return false;
	}
	#endif

	if (verbose || bench) {
		if (!cmpi.rank) printf_mag();
		printf_mpi(cmpi.rank, "Using Version 2 (linkNodes).\n");
		if (!cmpi.rank) printf_std();
	}

	int64_t idx = 0;
	int rank = cmpi.rank;
	int mpi_chunk = N / cmpi.num_mpi_threads;
	int mpi_offset = rank * mpi_chunk;

	#ifdef MPI_ENABLED
	uint64_t npairs = static_cast<uint64_t>(N) * mpi_chunk;
	uint64_t start = rank * npairs;
	uint64_t finish = start + npairs;
	if (!mpi_split)
		mpi_offset = 0;
	#else
	uint64_t n = N + N % 2;
	uint64_t npairs = n * (n - 1) / 2;
	uint64_t start = 0ULL;
	uint64_t finish = npairs;
	mpi_offset = 0ULL;
	#endif
	stopwatchStart(&sLinkNodes);

	#ifdef _OPENMP
	#pragma omp parallel for schedule (dynamic, 1) reduction (+ : idx) if (finish - start >= 1024)
	#endif
	for (uint64_t k = start; k < finish; k++) {
		#ifdef MPI_ENABLED
		int i = static_cast<int>(k / N);
		int j = static_cast<int>(k % N);
		if (i == j) continue;
		#else
		int i = static_cast<int>(k / (n - 1));
		int j = static_cast<int>(k % (n - 1) + 1);
		int do_map = i >= j;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);
		if (j == N) continue;
		#endif

		bool related;
		if (spacetime.manifoldIs("Hyperbolic"))
			related = nodesAreRelatedHyperbolic(nodes, spacetime, N, zeta, r_max, link_epso, i, j, NULL);
		else
			related = nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j, NULL);

		if (related) {
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			adj[i-mpi_offset].set(j);

			#ifndef MPI_ENABLED
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			adj[j].set(i);
			#endif

			if (i < j) {
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				nodes.k_in[j]++;
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				nodes.k_out[i]++;

				idx++;
			}
		}
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, nodes.k_in, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, nodes.k_out, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &idx, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
	if (!mpi_split)
		for (int i = 0; i < N; i++)
			MPI_Bcast(adj[i].getAddress(), adj[i].getNumBlocks(), BlockTypeMPI, rank, MPI_COMM_WORLD);
	#endif

	uint64_t kr = 0;
	for (int i = 0; i < N; i++) {
		if (nodes.k_in[i] + nodes.k_out[i] > 0) {
			N_res++;
			kr += nodes.k_in[i] + nodes.k_out[i];

			if (nodes.k_in[i] + nodes.k_out[i] > 1)
				N_deg2++;
		} 
	}

	#if DEBUG
	assert (N_res >= 0);
	assert (N_deg2 >= 0);
	#endif

	if (N_res > 0)
		k_res = static_cast<long double>(kr) / N_res;

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf_mpi(rank, "\tCausets Successfully Connected.\n");
		if (spacetime.manifoldIs("Hyperbolic") && link_epso)
			printf_mpi(rank, "\tEPSO Linking Rule Used.\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tUndirected Relations:     %" PRIu64 "\n", idx);
		printf_mpi(rank, "\t\tResulting Network Size:   %d\n", N_res);
		printf_mpi(rank, "\t\tResulting Average Degree: %f\n", k_res);
		printf_mpi(rank, "\t\t    Incl. Isolated Nodes: %f\n", k_res * ((float)N_res / N));
		if (has_exact_k) {
			if (!rank) printf_red();
			printf_mpi(rank, "\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		}
		if (!rank) printf_std();
		if (!rank) fflush(stdout);
	}

	if (verbose) {
		printf_mpi(rank, "\t\tExecution Time: %5.6f sec\n", sLinkNodes.elapsedTime);
		fflush(stdout);
	}
	
	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	return true;
}

//Identify Causal Sets
//O(k*N^2) Efficiency
bool linkNodes_v1(Node &nodes, Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &link_epso, const bool &has_exact_k, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	} else {
		assert (adj.size() >= (size_t)N);
		assert (core_edge_fraction == 1.0f);
	}
	assert (N > 0);
	assert (k_tar > 0.0f);
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Hyperbolic"));
	if (!spacetime.manifoldIs("Hyperbolic"))
		assert (a > 0.0);
	assert (tau0 > 0.0);
	if (spacetime.manifoldIs("De_Sitter")) {
		if (spacetime.curvatureIs("Positive")) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (spacetime.curvatureIs("Flat")) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		assert (nodes.crd->getDim() == 4);
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (spacetime.stdimIs("4"));
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	} else if (spacetime.manifoldIs("Hyperbolic")) {
		assert (zeta > 0.0);
		assert (r_max > 0.0);
	}
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
	assert (core_edge_fraction >= 0.0f && core_edge_fraction <= 1.0f);
	assert (edge_buffer >= 0.0f && edge_buffer <= 1.0f);
	#endif

	#if EMBED_NODES
	if (spacetime.manifoldIs("Hyperbolic")) {
		fprintf(stderr, "linkNodes_v1 not implemented for EMBED_NODES=true and MANIFOLD=HYPERBOLIC.  Find me on line %d in %s.\n", __LINE__, __FILE__);
		if (!!N)
			return false;
	}
	#endif

	if (verbose || bench)
		printf_dbg("Using Version 1 (linkNodes).\n");
	if (bench) {
		memset(edges.future_edges, 0, sizeof(int) * (uint64_t)N * k_tar * (1.0 + edge_buffer) / 2);
		memset(edges.past_edges, 0, sizeof(int) * (uint64_t)N * k_tar * (1.0 + edge_buffer) / 2);
		memset(edges.future_edge_row_start, 0, sizeof(int) * N);
		memset(edges.past_edge_row_start, 0, sizeof(int) * N);
		memset(nodes.k_in, 0, sizeof(int) * N);
		memset(nodes.k_out, 0, sizeof(int) * N);
	}

	uint64_t future_idx = 0;
	uint64_t past_idx = 0;
	int core_limit = static_cast<int>(core_edge_fraction * N);
	int i, j, k;

	bool related;

	stopwatchStart(&sLinkNodes);

	//Identify future connections
	for (i = 0; i < N - 1; i++) {
		if (!use_bit)
			edges.future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N; j++) {
			//Apply Causal Condition (Light Cone)
			//Assume nodes are already temporally ordered
			if (spacetime.manifoldIs("Hyperbolic"))
				related = nodesAreRelatedHyperbolic(nodes, spacetime, N, zeta, r_max, link_epso, i, j, NULL);
			else
				related = nodesAreRelated(nodes.crd, spacetime, N, a, zeta, zeta1, r_max, alpha, i, j, NULL);

			//Core Edge Adjacency Matrix
			if (i < core_limit && j < core_limit) {
				if (related) {
					adj[i].set(j);
					adj[j].set(i);
				}
			}
						
			//Link timelike relations
			try {
				if (related) {
					if (!use_bit) {
						edges.future_edges[future_idx++] = j;
	
						if (future_idx >= static_cast<int64_t>(N) * k_tar * (1.0 + edge_buffer) / 2)
							throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
					} else
						future_idx++;
	
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

		if (!use_bit) {
			//If there are no forward connections from node i, mark with -1
			if (static_cast<uint64_t>(edges.future_edge_row_start[i]) == future_idx)
				edges.future_edge_row_start[i] = -1;
		}
	}

	if (!use_bit) {
		edges.future_edge_row_start[N-1] = -1;

		//Identify past connections
		edges.past_edge_row_start[0] = -1;
		for (i = 1; i < N; i++) {
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
			if (static_cast<uint64_t>(edges.past_edge_row_start[i]) == past_idx)
				edges.past_edge_row_start[i] = -1;
		}

		//The quantities future_idx and past_idx should be equal
		#if DEBUG
		assert (future_idx == past_idx);
		#endif
	}

	//Identify Resulting Network
	uint64_t kr = 0;
	for (i = 0; i < N; i++) {
		if (nodes.k_in[i] + nodes.k_out[i] > 0) {
			N_res++;
			kr += nodes.k_in[i] + nodes.k_out[i];

			if (nodes.k_in[i] + nodes.k_out[i] > 1)
				N_deg2++;
		} 
	}

	#if DEBUG
	assert (N_res >= 0);
	assert (N_deg2 >= 0);
	#endif

	if (N_res > 0)
		k_res = static_cast<long double>(kr) / N_res;

	//Debugging options used to visually inspect the adjacency lists and the adjacency pointer lists
	//compareAdjacencyLists(nodes, edges);
	//compareAdjacencyListIndices(nodes, edges);
	//if(!compareCoreEdgeExists(nodes.k_out, edges.future_edges, edges.future_edge_row_start, adj, N, core_edge_fraction))
	//	return false;

	//Print Results
	/*if (!printDegrees(nodes, N, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;
	if (!printAdjMatrix(adj, N, "adj_matrix_CPU.cset.dbg.dat", 1, 0)) return false;
	if (!printEdgeLists(edges, past_idx, "past-edges_CPU.cset.dbg.dat", "future-edges_CPU.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N, "past-edge-pointers_CPU.cset.dbg.dat", "future-edge-pointers_CPU.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	printChk();*/

	stopwatchStop(&sLinkNodes);

	if (!bench) {
		printf("\tCausets Successfully Connected.\n");
		if (spacetime.manifoldIs("Hyperbolic") && link_epso)
			printf("\tEPSO Linking Rule Used.\n");
		printf_cyan();
		printf("\t\tUndirected Relations:     %" PRIu64 "\n", future_idx);
		printf("\t\tResulting Network Size:   %d\n", N_res);
		printf("\t\tResulting Average Degree: %f\n", k_res);
		printf("\t\t    Incl. Isolated Nodes: %f\n", k_res * ((float)N_res / N));
		if (has_exact_k) {
			printf_red();
			printf("\t\tResulting Error in <k>:   %f\n", fabs(k_tar - k_res) / k_tar);
		}
		printf_std();
		fflush(stdout);
	}

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sLinkNodes.elapsedTime);
		fflush(stdout);
	}

	return true;
}
