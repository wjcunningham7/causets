#include "NetworkCreator.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
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

	unsigned int spacetime = network_properties->spacetime;
	int rank = network_properties->cmpi.rank;

	//Make sure the spacetime is fully defined
	if (!rank) printf_red();
	if (!get_stdim(spacetime)) {
		printf_mpi(rank, "The spacetime dimension has not been defined!  Use flag '--stdim' to continue.\n");
		network_properties->cmpi.fail = 1;
	}
	if (!get_manifold(spacetime)) {
		printf_mpi(rank, "The manifold has not been defined!  Use flag '--manifold' to continue.\n");
		network_properties->cmpi.fail = 1;
	}
	if (!get_region(spacetime)) {
		printf_mpi(rank, "The region has not been defined!  Use flag '--region' to continue.\n");
		network_properties->cmpi.fail = 1;
	}
	if (!get_curvature(spacetime)) {
		printf_mpi(rank, "The curvature has not been defined!  Use flag '--curvature' to continue.\n");
		network_properties->cmpi.fail = 1;
	}
	if (!get_symmetry(spacetime)) {
		printf_mpi(rank, "The symmetry has not been defined!  Use flag '--symmetry' to continue.\n");
		network_properties->cmpi.fail = 1;
	}
	if (!rank) printf_std();
	fflush(stdout);

	//Benchmarking
	if (network_properties->flags.bench) {
		network_properties->graphID = 0;
		network_properties->flags.verbose = false;
		network_properties->flags.print_network = false;
	}

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

	if (network_properties->k_tar >= network_properties->N_tar / 32 - 1) {
		//This is when a bit array is smaller than the adjacency lists
		network_properties->flags.use_bit = true;
		network_properties->core_edge_fraction = 1.0;
	}

	#ifdef CUDA_ENABLED
	//If the GPU is requested, optimize parameters
	if (!LINK_NODES_GPU_V2 && network_properties->flags.use_gpu && network_properties->N_tar % (BLOCK_SIZE << 1)) {
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

	//Adjacency matrix not implemented in certain GPU algorithms
	if (network_properties->flags.use_gpu && !LINK_NODES_GPU_V2) {
		network_properties->flags.use_bit = false;
		network_properties->core_edge_fraction = 0.0;
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

		//Check for an under-constrained system
		if (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW)) {
			if (!network_properties->N_tar)
				throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
			if (!network_properties->tau0)
				throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");
			if (get_curvature(spacetime) & FLAT && get_region(spacetime) & SLAB) {
				if (get_manifold(spacetime) & DE_SITTER && !network_properties->r_max)
					throw CausetException("Flag '--slice', spatial scaling, must be specified!\n");
				else if (get_manifold(spacetime) & (DUST | FLRW) && !network_properties->alpha)
					throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
			}
		}

		//Default constraints
		if (get_manifold(spacetime) & MINKOWSKI) {
			network_properties->eta0 = network_properties->tau0;
			eta0 = network_properties->eta0;
			network_properties->zeta = HALF_PI - eta0;
			network_properties->a = 1.0;

			#if DEBUG
			assert (network_properties->eta0 > 0.0);
			#endif
		} else if (get_manifold(spacetime) & DE_SITTER) {
			//The pseudoradius takes a default value of 1
			if (!network_properties->delta)
				network_properties->a = 1.0;

			if (get_curvature(spacetime) & FLAT) {
				//We take eta_min = -1 so that rescaled time
				//will begin at tau = 0
				//In this case, the '--age' flag reads tau0
				network_properties->zeta = HALF_PI + 1.0;
				network_properties->zeta1 = HALF_PI - tauToEtaFlat(network_properties->tau0);

				#if DEBUG
				assert (network_properties->zeta > HALF_PI);
				#endif
			} else if (get_curvature(spacetime) & POSITIVE) {
				//Re-write variables to their correct locations
				//This is because the '--age' flag has read eta0
				//into the tau0 variable
				network_properties->eta0 = network_properties->tau0;
				network_properties->zeta = HALF_PI - network_properties->tau0;
				network_properties->tau0 = etaToTauSph(HALF_PI - network_properties->zeta);

				#if DEBUG
				assert (network_properties->zeta > 0.0 && network_properties->zeta < HALF_PI);
				#endif
			}

			eta0 = HALF_PI - network_properties->zeta;
			eta1 = HALF_PI - network_properties->zeta1;
			network_properties->eta0 = eta0;
		} else if (get_manifold(spacetime) & (DUST | FLRW)) {
			//The pseudoradius takes a default value of 1
			if (!network_properties->delta)
				network_properties->a = 1.0;

			//The maximum radius takes a default value of 1
			//This allows alpha to characterize the spatial cutoff
			if (get_region(spacetime) & SLAB && !network_properties->r_max)
				network_properties->r_max = 1.0;
		} else if (get_manifold(spacetime) & HYPERBOLIC) {
			//The hyperbolic curvature takes a default value of 1
			if (!network_properties->zeta)
				network_properties->zeta = 1.0;
		}

		//Solve for the remaining constraints
		switch (spacetime) {
		case (2 | MINKOWSKI | DIAMOND | FLAT | ASYMMETRIC):
			network_properties->k_tar = network_properties->N_tar / 2.0;
			network_properties->delta = 2.0 * network_properties->N_tar / POW2(network_properties->eta0, EXACT);
			network_properties->r_max = network_properties->eta0 / 2.0;
			break;
		case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			network_properties->k_tar = network_properties->N_tar * (network_properties->eta0 / TAN(network_properties->eta0, STL) - LOG(COS(network_properties->eta0, STL), STL) - 1.0) / (TAN(network_properties->eta0, STL) * HALF_PI);
			if (!!network_properties->delta)
				network_properties->a = SQRT(network_properties->N_tar / (TWO_PI * network_properties->delta * TAN(network_properties->eta0, STL)), STL);
			else
				network_properties->delta = network_properties->N_tar / (TWO_PI * POW2(network_properties->a, EXACT) * TAN(network_properties->eta0, STL));
			break;
		case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			network_properties->k_tar = (network_properties->N_tar / M_PI) * ((network_properties->eta0 / TAN(network_properties->eta0, STL) - 1.0) / TAN(network_properties->eta0, STL) + network_properties->eta0);
			if (!!network_properties->delta)
				network_properties->a = SQRT(network_properties->N_tar / (4.0 * M_PI * network_properties->delta * TAN(network_properties->eta0, STL)), STL);
			else
				network_properties->delta = network_properties->N_tar / (4.0 * M_PI * POW2(network_properties->a, EXACT) * TAN(network_properties->eta0, STL));
			break;
		case (2 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			break;
		case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):	
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			break;
		case (2 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			break;
		case (2 | HYPERBOLIC | SLAB | FLAT | ASYMMETRIC):
			//Nothing else needs to be done
			//but we don't want to trigger 'default'
			break;
		case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
		{
			int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());
			network_properties->k_tar = 9.0 * network_properties->N_tar * POW2(POW3(eta0 * eta1, EXACT), EXACT) * integrate2D(&averageDegree_10788_0, eta0, eta0, eta1, eta1, NULL, seed, 0) / (POW3(network_properties->r_max, EXACT) * POW2(POW3(eta1, EXACT) - POW3(eta0, EXACT), EXACT));
			if (!!network_properties->delta)
				network_properties->a = POW(9.0 * network_properties->N_tar * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * network_properties->delta * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT))), 0.25, STL);
			else
				network_properties->delta = 9.0 * network_properties->N_tar * POW3(eta0 * eta1, EXACT) / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * POW3(network_properties->r_max, EXACT) * (POW3(eta1, EXACT) - POW3(eta0, EXACT)));
			break;
		}
		case (4 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			network_properties->k_tar = network_properties->N_tar * (12.0 * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL)) - (6.0 * LOG(COS(eta0, STL), STL) + 5.0) / POW2(COS(eta0, STL), EXACT) - 7.0) / (POW2(2.0 + 1.0 / POW2(COS(eta0, STL), EXACT), EXACT) * TAN(eta0, STL) * 3.0 * HALF_PI);
			if (!!network_properties->delta)
				network_properties->a = POW(network_properties->N_tar * 3.0 / (2.0 * POW2(M_PI, EXACT) * network_properties->delta * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL)), 0.25, STL);
			else
				network_properties->delta = network_properties->N_tar * 3.0 / (2.0 * POW2(M_PI * POW2(network_properties->a, EXACT), EXACT) * (2.0 + 1.0 / POW2(COS(eta0, STL), EXACT)) * TAN(eta0, STL));
			break;
		case (4 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
		{
			IntData idata;
			idata.limit = 50;
			idata.tol = 1e-5;
			idata.lower = -eta0;
			idata.upper = eta0;
			double t1 = (sin(eta0) + sin(5.0 * eta0)) / (3.0 * POW3(cos(eta0), EXACT));
			double t2 = 2.0 * integrate1D(&averageDegree_21028_0, NULL, &idata, QNG);
			double t3 = (2.0 * eta0 * eta0 - 1.0) * (3.0 * sin(eta0) + sin(3.0 * eta0)) / (3.0 * POW3(cos(eta0), EXACT));

			network_properties->k_tar = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) * (t1 + t2 + t3) / (M_PI * tan(eta0) * (2.0 + 1.0 / POW2(cos(eta0), EXACT)) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
			if (!!network_properties->delta)
				network_properties->a = POW(3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * network_properties->delta * (3.0 * sin(eta0) + sin(3.0 * eta0))), 0.25, STL);
			else
				network_properties->delta = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
			break;
		}
		case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
		{
			double xi = eta0 / sqrt(2.0);
			double w = (eta1 - eta0) / sqrt(2.0);
			double mu = LOG(POW2(w + 2.0 * xi, EXACT) / (4.0 * xi * (w + xi)), STL) - POW2(w / (w + 2.0 * xi), EXACT);
			if (!!network_properties->delta)
				network_properties->a = POW(3.0 * network_properties->N_tar / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
			else
				network_properties->delta = 3.0 * network_properties->N_tar / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
			if (!getLookupTable("./etc/tables/average_degree_11300_0_table.cset.bin", &table, &size))
				throw CausetException("Average degree table not found!\n");
			network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
			network_properties->r_max = w / sqrt(2.0);
			break;
		}
		case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
		{
			double xi = eta0 / sqrt(2.0);
			double mu = log(0.5 * (1.0 / cos(sqrt(2.0) * xi) + 1.0)) - 1.0 / POW2(cos(xi / sqrt(2.0)), EXACT) + 1.0;
			if (!!network_properties->delta)
				network_properties->a = POW(3.0 * network_properties->N_tar / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
			else
				network_properties->delta = 3.0 * network_properties->N_tar / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
			if (!getLookupTable("./etc/tables/average_degree_13348_0_table.cset.bin", &table, &size))
				throw CausetException("Average degree table not found!\n");
			network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &eta0, NULL, true);
			break;
		}
		case (4 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			break;
		case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
		{
			if (!!network_properties->delta)
				network_properties->a = POW(network_properties->N_tar / (M_PI * network_properties->delta * POW3(network_properties->alpha * network_properties->tau0, EXACT)), 0.25, STL);
			else
				network_properties->delta = network_properties->N_tar / (M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * POW3(network_properties->alpha * network_properties->tau0, EXACT));
			
			int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());
			network_properties->k_tar = (108.0 * M_PI / POW3(network_properties->tau0, EXACT)) * network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * integrate2D(&averageDegree_10820_0, 0.0, 0.0, network_properties->tau0, network_properties->tau0, NULL, seed, 0);
			network_properties->alpha *= network_properties->a;
			eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
			network_properties->zeta = HALF_PI - eta0;
			break;
		}
		case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
		{
			double t = POW2(POW2(1.5 * network_properties->tau0, EXACT), EXACT);
			if (!!network_properties->delta)
				network_properties->a = POW(2970.0 * 64.0 * network_properties->N_tar / (1981.0 * M_PI * network_properties->delta * t), 0.25, STL);
			else
				network_properties->delta = 2970.0 * 64.0 * network_properties->N_tar / (1981.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * t);
			network_properties->alpha = 2.0 * network_properties->a; //This property should not affect results in the diamond
			if (!getLookupTable("./etc/tables/average_degree_11332_0_table.cset.bin", &table, &size))
				throw CausetException("Average degree table not found!\n");
			network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
			eta0 = tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
			network_properties->eta0 = eta0;
			network_properties->zeta = HALF_PI - eta0;
			network_properties->r_max = eta0 / 2.0;
			break;
		}
		case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
			method = 1;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->spacetime, network_properties->N_tar, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
				
			q = 9.0 * network_properties->N_tar / (TWO_PI * POW3(network_properties->alpha * network_properties->r_max, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
			if (!!network_properties->delta)
				network_properties->a = POW(q / network_properties->delta, 0.25, STL);
			else
				network_properties->delta = q / POW2(POW2(network_properties->a, EXACT), EXACT);
			network_properties->alpha *= network_properties->a;
			eta0 = tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);
			network_properties->zeta = HALF_PI - eta0;
			break;
		case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
			q = 3.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * POW3(network_properties->alpha, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
			if (!!network_properties->delta)
				network_properties->a = POW(q / network_properties->delta, 0.25, STL);
			else
				network_properties->delta = q / POW2(POW2(network_properties->a, EXACT), EXACT);
			network_properties->alpha *= network_properties->a;
			eta0 = tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);
			network_properties->zeta = HALF_PI - eta0;

			method = 1;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->spacetime, network_properties->N_tar, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
			break;
		case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
		{
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
			idata.key = GSL_INTEG_GAUSS61;
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
			idata.upper = network_properties->zeta1;
			double params[3];
			params[0] = network_properties->tau0;
			params[1] = eta0;
			params[2] = network_properties->zeta1;
			double vol_lower = integrate1D(&volume_11396_0_lower, &params, &idata, QAG);
			idata.lower = idata.upper;
			idata.upper = network_properties->tau0;
			double vol_upper = integrate1D(&volume_11396_0_upper, &params, &idata, QAG);
			double mu = vol_lower + vol_upper;
			gsl_integration_workspace_free(idata.workspace);

			if (!!network_properties->delta)
				network_properties->a = POW(3.0 * network_properties->N_tar / (4.0 * M_PI * network_properties->delta * mu), 0.25, STL);
			else
				network_properties->delta = 3.0 * network_properties->N_tar / (4.0 * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * mu);
			network_properties->alpha = network_properties->a;

			if (!getLookupTable("./etc/tables/average_degree_11396_0_table.cset.bin", &table, &size))
				throw CausetException("Average degree table not found!\n");
			network_properties->k_tar = network_properties->delta * POW2(POW2(network_properties->a, EXACT), EXACT) * lookupValue(table, size, &network_properties->tau0, NULL, true);
			break;
		}
		default:
			throw CausetException("Spacetime parameters not supported!\n");
		}

		if (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW)) {
			#if DEBUG
			assert (network_properties->k_tar > 0.0);
			assert (network_properties->a > 0.0);
			assert (network_properties->delta > 0.0);
			if (!((get_manifold(spacetime) & DE_SITTER) && (get_curvature(spacetime) & FLAT)))
				assert (network_properties->zeta < HALF_PI);
			#endif

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining the %d+1 %s Causal Set:\n", get_stdim(spacetime) - 1, manifoldNames[(unsigned int)(log2((float)get_manifold(spacetime) / ManifoldFirst))].c_str());
			printf_mpi(rank, "\t--------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Manifold:\t\t\t%s\n", manifoldNames[(unsigned int)(log2((float)get_manifold(spacetime) / ManifoldFirst))].c_str());
			printf_mpi(rank, "\t > Spacetime Dimension:\t\t%d+1\n", get_stdim(spacetime) - 1);
			printf_mpi(rank, "\t > Region:\t\t\t%s\n", regionNames[(unsigned int)(log2((float)get_region(spacetime) / RegionFirst))].c_str());
			printf_mpi(rank, "\t > Curvature:\t\t\t%s\n", curvatureNames[(unsigned int)(log2((float)get_curvature(spacetime) / CurvatureFirst))].c_str());
			printf_mpi(rank, "\t > Temporal Symmetry:\t\t%s\n", symmetryNames[(unsigned int)(log2((float)get_symmetry(spacetime) / SymmetryFirst))].c_str());
			printf_mpi(rank, "\t > Spacetime ID:\t\t%u\n", network_properties->spacetime);
			if (!rank) printf_std();
			printf_mpi(rank, "\t--------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			if (get_symmetry(spacetime) & SYMMETRIC) {
				printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", -eta0);
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
			} else if ((get_manifold(spacetime) & DE_SITTER) && (get_curvature(spacetime) & FLAT)) {
				printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", eta0);
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta1);
			} else {
				printf_mpi(rank, "\t > Min. Conformal Time:\t\t0.0\n");
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
			}
			printf_mpi(rank, "\t > Max. Rescaled Time:\t\t%.6f\n", network_properties->tau0);
			if (get_manifold(spacetime) & (DE_SITTER | FLRW))
				printf_mpi(rank, "\t > Dark Energy Density:\t\t%.6f\n", network_properties->omegaL);
			if (get_manifold(spacetime) & (DUST | FLRW))
				printf_mpi(rank, "\t > Spatial Scaling:\t\t%.6f\n", network_properties->alpha);
			if (get_curvature(spacetime) & FLAT && get_region(spacetime) & SLAB)
				printf_mpi(rank, "\t > Spatial Cutoff:\t\t%.6f\n", network_properties->r_max);
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW))
				printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Random Seed:\t\t\t%Ld\n", network_properties->seed);
			if (!rank) { printf_std(); printf("\n"); }
			fflush(stdout);

			//Miscellaneous Tasks
			if (get_manifold(spacetime) & DE_SITTER) {
				if (!network_properties->cmpi.rank && network_properties->flags.gen_ds_table && !generateGeodesicLookupTable("etc/tables/geodesics_ds_table.cset.bin", 5.0, -5.0, 5.0, 0.01, 0.01, network_properties->spacetime, network_properties->flags.verbose))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			} else if (get_manifold(spacetime) & FLRW) {
				if (!network_properties->cmpi.rank && network_properties->flags.gen_flrw_table && !generateGeodesicLookupTable("etc/tables/geodesics_flrw_table.cset.bin", 2.0, -5.0, 5.0, 0.01, 0.01, network_properties->spacetime, network_properties->flags.verbose))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			}
			
		}

		//Miscellaneous Tasks
		if (!network_properties->edge_buffer)
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

//Calculate Expected Average Degree in the FLRW Spacetime
//See Causal Set Notes for detailed explanation of methods
//NOTE: This method is largely historical - only a small portion is used
//  in practice, but it offers several methods to achieve the same outcome
bool solveExpAvgDegree(float &k_tar, const unsigned int &spacetime, const int &N_tar, double &a, const double &r_max, double &tau0, const double &alpha, const double &delta, const int &rank, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &verbose, const bool &bench, const int method)
{
	#if DEBUG
	assert (ca != NULL);
	assert (get_stdim(spacetime) & 4);
	assert (get_manifold(spacetime) & FLRW);
	assert (N_tar > 0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	assert (method == 0 || method == 1 || method == 2);
	if (get_curvature(spacetime) & FLAT) {
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
			switch (spacetime) {
			case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
				kappa = integrate2D(&averageDegree_10884_0, 0.0, 0.0, tau0, tau0, NULL, seed, 0);
				kappa *= 8.0 * M_PI;
				kappa /= SINH(3.0 * tau0, STL) - 3.0 * tau0;
				k_tar = (9.0 * kappa * N_tar) / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
				break;
			case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
				if (tau0 > LOG(MTAU, STL) / 3.0)
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&averageDegree_12932_0, 0.0, 0.0, r0, r0, NULL, seed, 0) * 16.0 * M_PI * exp(-3.0 * tau0);
				else
					k_tar = delta * POW2(POW2(a, EXACT), EXACT) * integrate2D(&averageDegree_12932_0, 0.0, 0.0, r0, r0, NULL, seed, 0) * 8.0 * M_PI / (SINH(3.0 * tau0, STL) - 3.0 * tau0);
				break;
			default:
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
		if (get_curvature(spacetime) & POSITIVE) {
			if (!getLookupTable("./etc/tables/raduc_table.cset.bin", &table, &size))
				return false;
		} else if (get_curvature(spacetime) & FLAT) {
			if (!getLookupTable("./etc/tables/raducNC_table.cset.bin", &table, &size))
				return false;
		} else
			return false;
		
		ca->hostMemUsed += size;

		for (i = 0; i <= nb; i++) {
			stopwatchStart(&sCalcDegrees);
			if (get_curvature(spacetime) & POSITIVE)
				k_tar = lookupValue(table, size, &tau0, NULL, true) * delta * POW2(POW2(a, EXACT), EXACT);
			else if (get_curvature(spacetime) & FLAT)
				k_tar = lookupValue(table, size, &tau0, NULL, true) * 9.0 * N_tar / (TWO_PI * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));
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
bool createNetwork(Node &nodes, Edge &edges, std::vector<bool> &core_edge_exists, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, const int &group_size, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &decode_cpu, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench, const bool &yes)
{
	#if DEBUG
	assert (ca != NULL);
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW | HYPERBOLIC));
	if (get_manifold(spacetime) & HYPERBOLIC)
		assert (get_stdim(spacetime) == 2);
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
			if (get_stdim(spacetime) == 4)
				mem += sizeof(float) * N_tar * 5;	//For Coordinate5D
			else if (get_stdim(spacetime) == 2)
				mem += sizeof(float) * N_tar * 3;	//For Coordinate3D
			#else
			if (get_stdim(spacetime) == 4)
				mem += sizeof(float) * N_tar << 2;	//For Coordinate4D
			else if (get_stdim(spacetime) == 2)
				mem += sizeof(float) * N_tar << 1;	//For Coordinate2D
			#endif
			if (get_manifold(spacetime) & HYPERBOLIC)
				mem += sizeof(int) * N_tar;		//For AS
			else if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW))
				mem += sizeof(float) * N_tar;		//For tau
		}
		if (links_exist) {
			mem += sizeof(int) * (N_tar << 1);	//For k_in and k_out
			if (!use_bit) {
				mem += sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer));	//For edge lists
				mem += sizeof(int) * (N_tar << 1);	//For edge list pointers
			}
			mem += sizeof(bool) * static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT)) / 8;	//Adjacency matrix
		}

		size_t dmem = 0;
		#ifdef CUDA_ENABLED
		size_t dmem1 = 0, dmem2 = 0, dmem3 = 0;
		if (use_gpu) {
			size_t d_edges_size = pow(2.0, ceil(log2(N_tar * k_tar * (1.0 + edge_buffer) / 2)));
			if (!use_bit)
				mem += sizeof(uint64_t) * d_edges_size;	//For encoded edge list
			mem += sizeof(int);			//For g_idx

			size_t mblock_size = static_cast<unsigned int>(ceil(static_cast<float>(N_tar) / (BLOCK_SIZE * group_size << 1)));
			size_t mthread_size = mblock_size * BLOCK_SIZE;
			size_t m_edges_size = mthread_size * mthread_size;
			size_t nbuf = GEN_ADJ_LISTS_GPU_V2 ? NBUFFERS : 1;
			mem += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (host)
			mem += sizeof(bool) * m_edges_size * nbuf;		//For adjacency matrix buffers (host)
			#if EMBED_NODES
			fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
			assert (false);
			#else
			dmem1 += sizeof(float) * mthread_size * 4 * nbuf << 1;	//For 4-D coordinate buffers
			#endif
			dmem1 += sizeof(int) * mthread_size * nbuf << 1;		//For k_in and k_out buffers (device)
			dmem1 += sizeof(bool) * m_edges_size * nbuf;			//For adjacency matrix buffers (device)

			if (!use_bit) {
				size_t g_mblock_size = static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / (BLOCK_SIZE * group_size << 1));
				size_t g_mthread_size = g_mblock_size * BLOCK_SIZE;
				dmem2 += sizeof(uint64_t) * d_edges_size;	//Encoded edge list used during parallel sorting
				dmem2 += sizeof(int) * (DECODE_LISTS_GPU_V2 ? g_mthread_size : d_edges_size);	//For edge lists
				if (decode_cpu)
					dmem2 = 0;

				dmem3 += sizeof(int) * N_tar << 1;	//Edge list pointers
				dmem3 += sizeof(int) * BLOCK_SIZE << 2;	//Buffers used for scanning
			}

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
		if (!no_pos) {
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
				nodes.id.tau = (float*)malloc(sizeof(float) * N_tar);
				if (nodes.id.tau == NULL)
					throw std::bad_alloc();
				memset(nodes.id.tau, 0, sizeof(float) * N_tar);
				ca->hostMemUsed += sizeof(float) * N_tar;
			} else if (get_manifold(spacetime) & HYPERBOLIC) {
				nodes.id.AS = (int*)malloc(sizeof(int) * N_tar);
				if (nodes.id.AS == NULL)
					throw std::bad_alloc();
				memset(nodes.id.AS, 0, sizeof(int) * N_tar);
				ca->hostMemUsed += sizeof(int) * N_tar;
			}

			#if EMBED_NODES
			if (get_stdim(spacetime) == 4) {
				nodes.crd = new Coordinates5D();

				nodes.crd->v() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->w() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N_tar);

				if (nodes.crd->v() == NULL || nodes.crd->w() == NULL || nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->v(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->w(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->y(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->z(), 0, sizeof(float) * N_tar);

				ca->hostMemUsed += sizeof(float) * N_tar * 5;
			} else if (get_stdim(spacetime) == 2) {
				nodes.crd = new Coordinates3D();

				nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->z() = (float*)malloc(sizeof(float) * N_tar);

				if (nodes.crd->x() == NULL || nodes.crd->y() == NULL || nodes.crd->z() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->y(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->z(), 0, sizeof(float) * N_tar);

				ca->hostMemUsed += sizeof(float) * N_tar * 3;
			}
			#else
			if (get_stdim(spacetime) == 4) {
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
			} else if (get_stdim(spacetime) == 2) {
				nodes.crd = new Coordinates2D();

				nodes.crd->x() = (float*)malloc(sizeof(float) * N_tar);
				nodes.crd->y() = (float*)malloc(sizeof(float) * N_tar);

				if (nodes.crd->x() == NULL || nodes.crd->y() == NULL)
					throw std::bad_alloc();

				memset(nodes.crd->x(), 0, sizeof(float) * N_tar);
				memset(nodes.crd->y(), 0, sizeof(float) * N_tar);

				ca->hostMemUsed += sizeof(float) * N_tar * 2;
			}
			#endif
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
			if (!use_bit) {
				edges.past_edges = (int*)malloc(sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
				if (edges.past_edges == NULL)
					throw std::bad_alloc();
				memset(edges.past_edges, 0, sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
				ca->hostMemUsed += sizeof(int) * static_cast<unsigned int>(N_tar * k_tar * (1.0 + edge_buffer) / 2);

				edges.future_edges = (int*)malloc(sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
				if (edges.future_edges == NULL)
					throw std::bad_alloc();
				memset(edges.future_edges, 0, sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / 2));
				ca->hostMemUsed += sizeof(int) * static_cast<uint64_t>(N_tar * k_tar * (1.0 + edge_buffer) / 2);

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
			}

			core_edge_exists.resize(POW2(core_edge_fraction * N_tar, EXACT));
			ca->hostMemUsed += sizeof(bool) * static_cast<uint64_t>(POW2(core_edge_fraction * N_tar, EXACT)) / 8;
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
bool generateNodes(Node &nodes, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const double &a, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, MersenneRNG &mrng, Stopwatch &sGenerateNodes, const bool &verbose, const bool &bench)
{
	#if DEBUG
	//Values are in correct ranges
	assert (!nodes.crd->isNull());
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW));
	assert (a >= 0.0);
	assert (tau0 > 0.0);
	if (get_manifold(spacetime) & (DUST | FLRW)) {
		#if EMBED_NODES
		assert (nodes.crd->getDim() == 5);
		assert (nodes.crd->v() != NULL);
		#else
		assert (nodes.crd->getDim() == 4);
		#endif
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_stdim(spacetime) == 4);
		assert (zeta < HALF_PI);
	} else if (get_manifold(spacetime) & DE_SITTER) {
		if (get_curvature(spacetime) & POSITIVE) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (get_curvature(spacetime) & FLAT) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	}
	if (get_curvature(spacetime) & FLAT)
		assert (r_max > 0.0);
	#endif

	//Enable this to validate the nodes are being generated with the correct
	//distributions - it will use rejection sampling from the slab's distributions
	bool DEBUG_DIAMOND = false;

	stopwatchStart(&sGenerateNodes);

	IntData *idata = NULL;
	double params[3];
	double xi = eta0 / sqrt(2.0);
	double w = (zeta - zeta1) / sqrt(2.0);
	double mu, mu1, mu2;
	double p1;

	//Rejection sampling vs exact CDF inversion
	bool use_rejection = false;
	if (DEBUG_DIAMOND && get_region(spacetime) & DIAMOND)
		use_rejection = true;

	//Initialize GSL integration structure
	//There is one 'workspace' per OpenMP thread to avoid
	//write conflicts in the for loop
	size_t i_size = (use_rejection ? 1 : omp_get_max_threads()) * sizeof(IntData);
	if ((USE_GSL || get_region(spacetime) & DIAMOND) && get_manifold(spacetime) & FLRW) {
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
	switch (spacetime) {
	case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
		mu = LOG(POW2(w + 2.0 * xi, EXACT) / (4.0 * xi * (w + xi)), STL) - POW2(w / (w + 2.0 * xi), EXACT);
		break;
	case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
		mu = LOG(0.5 * (1.0 / COS(sqrt(2.0) * xi, APPROX ? FAST : STL) + 1.0), STL) - 1.0 / POW2(COS(xi / sqrt(2.0), APPROX ? FAST : STL), EXACT) + 1.0;
		break;
	case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
		params[0] = tau0;
		params[1] = HALF_PI - zeta;
		params[2] = zeta1;
		(*idata).limit = 100;
		(*idata).tol = 1e-8;
		(*idata).key = GSL_INTEG_GAUSS61;
		(*idata).upper = zeta1;
		mu1 = integrate1D(&volume_11396_0_lower, params, idata, QAG);
		(*idata).lower = (*idata).upper;
		(*idata).upper = tau0;
		mu2 = integrate1D(&volume_11396_0_upper, params, idata, QAG);
		mu = mu1 + mu2;
		p1 = mu1 / mu;
		(*idata).limit = 50;
		(*idata).tol = 1e-4;
		break;
	default:
		mu = 1.0;
	}

	#ifndef _OPENMP
	UGenerator &urng = mrng.rng;
	NDistribution ndist(0.0, 1.0);
	NGenerator nrng(mrng.eng, ndist);
	#endif

	//Generate coordinates for each of N nodes
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
		while (i < N_tar) {
			switch (spacetime) {
			case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
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
				break;
			case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
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
				break;
			case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
			{
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
				break;
			}
			case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
			{
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
				break;
			}
			default:
				fprintf(stderr, "Spacetime parameters not supported!\n");
				assert (false);
			}

			#if DEBUG
			validateCoordinates(nodes, spacetime, eta0, zeta, zeta1, r_max, tau0, i);
			#endif

			i++;
		}
	} else {
		//Use exact CDF inversion formulae (see notes)
		#ifdef _OPENMP
		unsigned int seed = static_cast<unsigned int>(mrng.rng() * 400000000);
		#pragma omp parallel
		{
		//Initialize one RNG per thread
		Engine eng(seed ^ omp_get_thread_num());
		UDistribution udist(0.0, 1.0);
		UGenerator urng(eng, udist);
		NDistribution ndist(0.0, 1.0);
		NGenerator nrng(eng, ndist);
		#pragma omp for schedule (dynamic, 1)
		#endif
		for (int i = 0; i < N_tar; i++) {
			#if EMBED_NODES
			float2 emb2;
			float3 emb3;
			float4 emb4;
			#endif
			double eta, r;
			double u, v;
			int tid = omp_get_thread_num();
			switch (spacetime) {
			case (2 | MINKOWSKI | DIAMOND | FLAT | ASYMMETRIC):
				u = get_2d_asym_flat_minkowski_diamond_u(urng, xi);
				v = get_2d_asym_flat_minkowski_diamond_v(urng, xi);
				nodes.crd->x(i) = (u + v) / sqrt(2.0);
				nodes.crd->y(i) = (u - v) / sqrt(2.0);
				break;
			case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
				nodes.crd->x(i) = get_2d_asym_sph_deSitter_slab_eta(urng, eta0);
				nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
				#if EMBED_NODES
				emb2 = get_2d_asym_sph_deSitter_slab_emb(urng);
				nodes.crd->y(i) = emb2.x;
				nodes.crd->z(i) = emb2.y;
				#else
				nodes.crd->y(i) = get_2d_asym_sph_deSitter_slab_theta(urng);
				#endif
				break;
			case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
				nodes.crd->x(i) = get_2d_sym_sph_deSitter_slab_eta(urng, eta0);
				nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
				#if EMBED_NODES
				emb2 = get_2d_sym_sph_deSitter_slab_emb(urng);
				nodes.crd->y(i) = emb2.x;
				nodes.crd->z(i) = emb2.y;
				#else
				nodes.crd->y(i) = get_2d_sym_sph_deSitter_slab_theta(urng);
				#endif
				break;
			case (2 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
				fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
				assert (false);
				break;
			case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
				nodes.crd->x(i) = get_2d_asym_sph_deSitter_diamond_eta(urng);
				nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
				#if EMBED_NODES
				emb2 = get_2d_asym_sph_deSitter_diamond_emb(mrng.rng, nodes.crd->x(i));
				nodes.crd->y(i) = emb2.x;
				nodes.crd->z(i) = emb2.y;
				#else
				nodes.crd->y(i) = get_2d_asym_sph_deSitter_diamond_theta(urng, nodes.crd->x(i));
				#endif
				break;
			case (2 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
				fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
				assert (false);
				break;
			case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
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
				break;
			case (4 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
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
				break;
			case (4 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
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
				break;
			case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
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
				break;
			case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
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
				break;
			case (4 | DE_SITTER | DIAMOND | POSITIVE | SYMMETRIC):
				fprintf(stderr, "Not yet implemented on line %d in file %s\n", __LINE__, __FILE__);
				assert (false);
				break;
			case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
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
				break;
			case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
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
				break;
			case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
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
				break;
			case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
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
				break;
			case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
			{
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
			}
			default:
				fprintf(stderr, "Spacetime parameters not supported!\n");
				assert (false);
			}

			#if DEBUG
			validateCoordinates(nodes, spacetime, eta0, zeta, zeta1, r_max, tau0, i);
			#endif
		}
		#ifdef _OPENMP
		}
		#endif
	}


	//Free GSL workspace memory
	if ((USE_GSL || get_region(spacetime) & DIAMOND) && get_manifold(spacetime) & FLRW) {
		for (int i = 0; i < (int)(i_size / sizeof(IntData)); i++)
			gsl_integration_workspace_free(idata[i].workspace);
		free(idata);
		idata = NULL;
	}

	//Debugging statements used to check coordinate distributions
	//if (!printValues(nodes, spacetime, N_tar, "tau_dist.cset.dbg.dat", "tau")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "eta_dist.cset.dbg.dat", "eta")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "u_dist.cset.dbg.dat", "u")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "v_dist.cset.dbg.dat", "v")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "theta1_dist.cset.dbg.dat", "theta1")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "theta2_dist.cset.dbg.dat", "theta2")) return false;
	//if (!printValues(nodes, spacetime, N_tar, "theta3_dist.cset.dbg.dat", "theta3")) return false;
	/*printf_red();
	printf("Check coordinate distributions now.\n");
	printf_std();
	fflush(stdout);*/
	//printChk();

	stopwatchStop(&sGenerateNodes);

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
bool linkNodes(Node &nodes, Edge &edges, std::vector<bool> &core_edge_exists, const unsigned int &spacetime, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &use_bit, const bool &verbose, const bool &bench)
{
	#if DEBUG
	//No null pointers
	assert (!nodes.crd->isNull());
	if (!use_bit) {
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	} else
		assert (core_edge_fraction == 1.0f);

	//Variables in correct ranges
	assert (N_tar > 0);
	assert (k_tar > 0.0f);
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (MINKOWSKI | DE_SITTER | DUST | FLRW));
	assert (a > 0.0);
	assert (tau0 > 0.0);
	if (get_manifold(spacetime) & DE_SITTER) {
		if (get_curvature(spacetime) & POSITIVE) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (get_curvature(spacetime) & FLAT) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (get_manifold(spacetime) & (DUST | FLRW)) {
		#if EMBED_NODES
		assert (nodes.crd->getDim() == 5);
		assert (nodes.crd->v() != NULL);
		#else
		assert (nodes.crd->getDim() == 4);
		#endif
		assert (nodes.crd->w() != NULL);
		assert (nodes.crd->x() != NULL);
		assert (nodes.crd->y() != NULL);
		assert (nodes.crd->z() != NULL);
		assert (get_stdim(spacetime) == 4);
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	if (get_curvature(spacetime) & FLAT)
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
		if (!use_bit)
			edges.future_edge_row_start[i] = future_idx;

		for (j = i + 1; j < N_tar; j++) {
			//Apply Causal Condition (Light Cone)
			//Assume nodes are already temporally ordered
			related = nodesAreRelated(nodes.crd, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, i, j, NULL);

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
					if (!use_bit) {
						//if (i % NPRINT == 0) printf("%d %d\n", i, j); fflush(stdout);
						edges.future_edges[future_idx++] = j;
	
						if (future_idx >= static_cast<int>(N_tar * k_tar * (1.0 + edge_buffer) / 2))
							throw CausetException("Not enough memory in edge adjacency list.  Increase edge buffer or decrease network size.\n");
					}
	
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
			if (edges.future_edge_row_start[i] == future_idx)
				edges.future_edge_row_start[i] = -1;
		}
	}

	if (!use_bit) {
		edges.future_edge_row_start[N_tar-1] = -1;

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
	}

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
	/*if (!printDegrees(nodes, N_tar, "in-degrees_CPU.cset.dbg.dat", "out-degrees_CPU.cset.dbg.dat")) return false;
	if (!printEdgeLists(edges, past_idx, "past-edges_CPU.cset.dbg.dat", "future-edges_CPU.cset.dbg.dat")) return false;
	if (!printEdgeListPointers(edges, N_tar, "past-edge-pointers_CPU.cset.dbg.dat", "future-edge-pointers_CPU.cset.dbg.dat")) return false;
	printf_red();
	printf("Check files now.\n");
	printf_std();
	fflush(stdout);
	printChk();*/

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
