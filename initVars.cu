	try {
		double eta0 = 0.0, eta1 = 0.0;
		double q;
		int method;

		//Check for an under-constrained system
		if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
			if (!network_properties->N_tar)
				throw CausetException("Flag '--nodes', number of nodes, must be specified!\n");
			if (!network_properties->tau0)
				throw CausetException("Flag '--age', temporal cutoff, must be specified!\n");
			if ((get_manifold(spacetime) | get_symmetry(spacetime)) & (DE_SITTER | ASYMMETRIC) && !network_properties->r_max)
				throw CausetException("Flag '--slice', spatial scaling, must be specified!\n");
			else if (get_manifold(spacetime) & (DUST | FLRW)) {
				if (!network_properties->alpha)
					throw CausetException("Flag '--alpha', spatial scale, must be specified!\n");
			}
		}

		//Default constraints
		if (get_manifold(spacetime) & DE_SITTER) {
			//The pseudoradius takes a default value of 1
			if (!network_properties->delta)
				network_properties->a = 1.0;

			if (get_curvature(spacetime) & FLAT) {
				//We take eta_min = -1 so that rescaled time
				//will begin at tau = 0
				//In this case, the '--age' flag reads tau0
				network_properties->zeta = HALF_PI + 1.0;
				network_properties->zeta1 = HALF_PI - tauToEtaFlat(network_properties->tau0;

				#if DEBUG
				assert (network_properties->zeta > HALF_PI);
				#endif
			} else if (get_curvature(spacetime) & POSITIVE) {
				//Re-write variables to their correct locations
				//This is because the '--age' flag has read eta0
				//into the tau0 variable
				network_properties->zeta = HALF_PI - network_properties->tau0;
				network_properties->tau0 = etaToTauCompact(HALF_PI - network_properties->zeta);

				#if DEBUG
				assert (network_properties->zeta > 0.0 && network_properties->zeta < HALF_PI);
				#endif
			}

			eta0 = HALF_PI - network_properties->zeta;
			eta1 = HALF_PI - network_properties->zeta1;
		} else if (get_manifold(spacetime) & (DUST | FLRW)) {
			//The density takes a default value of 1000
			if (!network_properties->delta)
				network_properties->delta = 1000;

			//The maximum radius takes a default value of 1
			//This allows alpha to characterize the spatial cutoff
			if (!network_properties->r_max)
				network_properties->r_max = 1.0;
		} else if (get_manifold(spacetime) & HYPERBOLIC) {
			//The hyperbolic curvature takes a default value of 1
			if (!network_properties->zeta)
				network_properties->zeta = 1.0;
		}

		//Solve for the remaining constraints
		switch (spacetime) {
		case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			network_properties->k_tar = network_properties->N_tar * (eta0 / TAN(eta0, STL) - LOG(COS(eta0, STL), STL) - 1.0) / (TAN(eta0, STL) * HALF_PI);
			if (!!network_properties->delta)
				network_properties->a = SQRT(network_properties->N_tar / (TWO_PI * network_properties->delta * TAN(eta0, STL)), STL);
			else
				network_properties->delta = network_properties->N_tar / (TWO_PI * POW2(network_properties->a, EXACT) * TAN(eta0, STL));
			break;
		case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			network_properties->k_tar = (network_properties->N_tar / M_PI) * ((eta0 / TAN(eta0, STL) - 1.0) / TAN(eta0, STL) + eta0);
			if (!!network_properties->delta)
				network_properties->a = SQRT(network_properties->N_tar / (4.0 * M_PI * network_properties->delta * TAN(eta0, STL)), STL);
			else
				network_properties->delta = network_properties->N_tar / (4.0 * M_PI * POW2(network_properties->a, EXACT) * TAN(eta0, STL));
			break;
		case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			//Add this
			break;
		case (2 | HYPERBOLIC | SLAB | FLAT | ASYMMETRIC):
			//Nothing else needs to be done
			//but we don't want to trigger 'default'
			break;
		case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
		{
			int seed = static_cast<int>(4000000000 * network_properties->mrng.rng());

			network_properties->k_tar = 9.0 * network_properties->N_tar * POW2(POW3(eta0 * eta1, EXACT), EXACT) * integrate2D(&rescaledDegreeDeSitterFlat, eta0, eta0, eta1, eta1, NULL, seed, 0) / (POW3(network_properties->r_max, EXACT) * POW2(POW3(eta1, EXACT) - POW3(eta0, EXACT), EXACT));
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
			double t2 = 2.0 * integrate1D(&averageDegreeSym, NULL, &idata, QNG);
			double t3 = (2.0 * eta0 * eta0 - 1.0) * (3.0 * sin(eta0) + sin(3.0 * eta0)) / (3.0 * POW3(cos(eta0), EXACT));

			network_properties->k_tar = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) * (t1 + t2 + t3) / (M_PI * tan(eta0) * (2.0 + 1.0 / POW2(cos(eta0), EXACT)) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
			if (!!network_properties->delta)
				network_properties->a = POW(3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * network_properties->delta * (3.0 * sin(eta0) + sin(3.0 * eta0))), 0.25, STL);
			else
				network_properties->delta = 3.0 * network_properties->N_tar * POW3(cos(eta0), EXACT) / (TWO_PI * M_PI * POW2(POW2(network_properties->a, EXACT), EXACT) * (3.0 * sin(eta0) + sin(3.0 * eta0)));
			break;
		}
		case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
			//Add this
			break;
		case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			//Add this
			break;
		case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
			method = 0;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->stdim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;

			q = network_properties->N_tar / (M_PI * POW3(network_properties->alpha * network_properties->r_max * network_properties->tau0, EXACT));
			network_properties->a = POW(q / network_properties->delta, 0.25, STL);
			network_properties->alpha *= network_properties->a;

			network_properties->zeta = HALF_PI - tauToEtaDust(network_properties->tau0, network_properties->a, network_properties->alpha);
			break;
		case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
			//Add this
			break;
		case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
			method = 1;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->stdim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
				
			q = 9.0 * network_properties->N_tar / (TWO_PI * POW3(network_properties->alpha * network_properties->r_max, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
			network_properties->a = POW(q / network_properties->delta, 0.25, STL);
			network_properties->alpha *= network_properties->a;
			network_properties->zeta = HALF_PI - tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);
			break;
		case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
			q = 3.0 * network_properties->N_tar / (POW2(M_PI, EXACT) * POW3(network_properties->alpha, EXACT) * (SINH(3.0 * network_properties->tau0, STL) - 3.0 * network_properties->tau0));
			network_properties->a = POW(q / network_properties->delta, 1.0 / 4.0, STL);
			network_properties->alpha *= network_properties->a;
			network_properties->zeta = HALF_PI - tauToEtaFLRWExact(network_properties->tau0, network_properties->a, network_properties->alpha);

			method = 1;
			if (!solveExpAvgDegree(network_properties->k_tar, network_properties->N_tar, network_properties->stdim, network_properties->manifold, network_properties->a, network_properties->r_max, network_properties->tau0, network_properties->alpha, network_properties->delta, network_properties->cmpi.rank, network_properties->mrng, ca, cp->sCalcDegrees, bm->bCalcDegrees, network_properties->flags.compact, network_properties->flags.verbose, network_properties->flags.bench, method))
				network_properties->cmpi.fail = 1;

			if (checkMpiErrors(network_properties->cmpi))
				return false;
			break;
		case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
			//Add this
			break;
		default:
			throw CausetException("Spacetime parameters not supported!\n");
		}

		if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
			#if DEBUG
			assert (network_properties->k_tar > 0.0);
			assert (network_properties->a > 0.0);
			assert (network_properties->delta > 0.0);
			if (!((get_manifold(spacetime) | get_curvature(spacetime)) & (DE_SITTER | FLAT)))
				assert (network_properties->zeta < HALF_PI);
			#endif

			//Display Constraints
			printf_mpi(rank, "\n");
			printf_mpi(rank, "\tParameters Constraining the %d+1 %s Causal Set:\n", get_stdim(spacetime) - 1, manifoldNames[log2(get_manifold(spacetime) / ManifoldFirst)]);
			printf_mpi(rank, "\t--------------------------------------------\n");
			if (!rank) printf_cyan();
			printf_mpi(rank, "\t > Manifold: %s\n", manifoldNames[log2(get_manifold(spacetime) / ManifoldFirst)]);
			printf_mpi(rank, "\t > Spacetime Dimension: %d+1\n", get_stdim(spacetime) - 1);
			printf_mpi(rank, "\t > Region: %s\n", regionNames[log2(get_region(spacetime) / RegionFirst)]);
			printf_mpi(rank, "\t > Curvature: %s\n", curvatureNames[log2(get_curvature(spacetime) / CurvatureFirst)]);
			printf_mpi(rank, "\t > Temporal Symmetry: %s\n", symmetryNames[log2(get_symmetry(spacetime) / SymmetryFirst)]);
			printf_mpi("\n");
			printf_mpi(rank, "\t > Number of Nodes:\t\t%d\n", network_properties->N_tar);
			printf_mpi(rank, "\t > Node Density:\t\t%.6f\n", network_properties->delta);
			printf_mpi(rank, "\t > Expected Degrees:\t\t%.6f\n", network_properties->k_tar);
			if (get_symmetry(spacetime) & SYMMETRIC) {
				printf_mpi(rank, "\t > Min. Conformal Time:\t\t%.6f\n", -eta0);
				printf_mpi(rank, "\t > Max. Conformal Time:\t\t%.6f\n", eta0);
			} else if ((get_manifold(spacetime) | get_curvature(spacetime)) & (DE_SITTER | FLAT)) {
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
			else if (get_curvature(spacetime) & FLAT)
				printf_mpi(rank, "\t > Spatial Cutoff:\t\t%.6f\n", network_properties->r_max);
			printf_mpi(rank, "\t > Temporal Scaling:\t\t%.6f\n", network_properties->a);
			printf_mpi(rank, "\t > Ransom Seed:\t\t\t%Ld\n", network_properties->seed);
			if (!rank) printf_std();
			fflush(stdout);

			//Miscellaneous Tasks
			if (get_manifold(spacetime) & DE_SITTER) {
				if (!network_properties->cmpi.rank && network_properties->flags.gen_ds_table && !generateGeodesicLookupTable("geodesics_ds_table.cset.bin", 5.0, -5.0, 5.0, 0.01, 0.01, network_properties->manifold, network_properties->flags.verbose))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			} else if (get_manifold(spacetime) & FLRW) {
				if (!network_properties->cmpi.rank && network_properties->flags.gen_flrw_table && !generateGeodesicLookupTable("geodesics_flrw_table.cset.bin", 2.0, -5.0, 5.0, 0.01, 0.01, network_properties->manifold, network_properties->flags.verbose))
					network_properties->cmpi.fail = 1;

				if (checkMpiErrors(network_properties->cmpi))
					return false;
			}
			
		}
	}
