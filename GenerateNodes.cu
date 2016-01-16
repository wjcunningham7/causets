	NDistribution ndist(0.0, 1.0);
	NGenerator nrng(mrng.eng, ndist);
	float tol = 1.0e-8;
	for (int i = 0; i < N_tar; i++) {
		double eta;
		float4 emb4;
		float2 emb2;
		switch (spacetime) {
		case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			nodes.crd->x(i) = get_2d_asym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = tauToEtaSph(nodes.crd-x(i));
			#if EMBED_NODES
			emb2 = get_2d_asym_sph_deSitter_slab_emb(mrng.rng);
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_asym_sph_deSitter_slab_theta(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < HALF_PI - zeta);
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
			#else
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
			#endif
			#endif
			break;
		case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			nodes.crd->x(i) = get_2d_sym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = tauToEtaSph(nodes.crd->x(i));
			#if EMBED_NODES
			emb2 = get_2d_sym_sph_deSitter_slab_emb(mrng.rng);
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_sym_sph_deSitter_slab_theta(mrng.rng);
			#endif

			#if DEBUG
			assert (fabs(nodes.crd->x(i)) < HALF_PI - zeta);
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
			#else
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
			#endif
			#endif
			break;
		case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			nodes.crd->x(i) = get_2d_asym_sph_deSitter_diamond_eta(mrng.rng);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->x(i));
			#if EMBED_NODES
			emb2 = get_2d_asym_sph_deSitter_diamond_emb(mrng.rng, nodes.crd->x(i));
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_asym_sph_deSitter_diamond_theta(mrng.rng, nodes.crd->x(i));
			#endif

			#if DEBUG
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < HALF_PI - zeta);
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			//y, z
			#else
			//y
			#endif
			#endif
			break;
		case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
			#if EMBED_NODES
			nodes.crd->v(i) = get_4d_asym_flat_deSitter_slab_eta(mrng.rng, zeta, zeta1);
			nodes.id.tau[i] = etaToTauFlat(nodes.crd->v(i));
			emb4 = get_4d_asym_flat_deSitter_slab_cartesian(mrng.rng, nrng, r_max);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_flat_deSitter_slab_eta(mrng.rng, zeta, zeta1);
			nodes.id.tau[i] = etaToTauFlat(nodes.crd->w(i));
			nodes.crd->x(i) = get_4d_asym_flat_deSitter_slab_radius(mrng.rng, r_max);
			nodes.crd->y(i) = get_4d_asym_flat_deSitter_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_flat_deSitter_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (HALF_PI - nodes.crd->v(i) > zeta && HALF_PI - nodes.crd->v(i) < zeta1);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - r_max) < tol);
			#else
			assert (HALF_PI - nodes.crd->w(i) > zeta && HALF_PI - nodes.crd->w(i) < zeta1);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			#if EMBED_NODES
			nodes.crd->v(i) = get_4d_asym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
			emb4 = get_4d_asym_sph_deSitter_slab_emb(nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
			nodes.crd->x(i) = get_4d_asym_sph_deSitter_slab_theta1(mrng.rng);
			nodes.crd->y(i) = get_4d_asym_sph_deSitter_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_sph_deSitter_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			#if EMBED_NODES
			nodes.crd->v(i) = get_4d_sym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
			emb4 = get_4d_sym_sph_deSitter_slab_emb(nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_sym_sph_deSitter_slab_eta(mrng.rng, zeta);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
			nodes.crd->x(i) = get_4d_sym_sph_deSitter_slab_theta1(mrng.rng);
			nodes.crd->y(i) = get_4d_sym_sph_deSitter_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_sym_sph_deSitter_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (fabs(nodes.crd->v(i)) < HALF_PI - zeta);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
			#else
			assert (fabs(nodes.crd->v(i)) < HALF_PI - zeta);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
			#if EMBED_NODES
			nodes.crd->v(i) = get_4d_asym_flat_deSitter_diamond_eta(mrng.rng);
			nodes.id.tau[i] = etaToTauFlat(nodes.crd->v(i));
			emb4 = get_4d_asym_flat_deSitter_diamond_cartesian(mrng.rng, nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_flat_deSitter_diamond_eta(mrng.rng);
			nodes.id.tau[i] = etaToTauFlat(nodes.crd->w(i));
			nodes.crd->x(i) = get_4d_asym_flat_deSitter_diamond_radius(mrng.rng);
			nodes.crd->y(i) = get_4d_asym_flat_deSitter_diamond_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_flat_deSitter_diamond_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			//emb
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			//x
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			#if EMBED_NODES
			nodes.crd-v(i) = get_4d_asym_sph_deSitter_diamond_eta(mrng.rng);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->v(i));
			emb4 = get_4d_asym_sph_deSitter_diamond_emb(mrng.rng, nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_sph_deSitter_diamond_eta(mrng.rng);
			nodes.id.tau[i] = etaToTauSph(nodes.crd->w(i));
			nodes.crd->x(i) = get_4d_asym_sph_deSitter_diamond_theta1(mrng.rng);
			nodes.crd->y(i) = get_4d_asym_sph_deSitter_diamond_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_sph_deSitter_diamond_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			//emb
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			//x
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
			#if EMBED_NODES
			nodes.id.tau[i] = get_4d_asym_flat_dust_slab_tau(mrng.rng, tau0);
			nodes.crd->v(i) = tauToEtaDust(nodes.id.tau[i]);
			emb4 = get_4d_asym_flat_dust_slab_cartesian(mrng.rng, nrng, r_max);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.id.tau[i] = get_4d_asym_flat_dust_slab_tau(mrng.rng, tau0);
			nodes.crd->w(i) = tauToEtaDust(nodes.id.tau[i]);
			nodes.crd->x(i) = get_4d_asym_flat_dust_slab_radius(mrng.rng, r_max);
			nodes.crd->y(i) = get_4d_asym_flat_dust_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_flat_dust_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - r_max) < tol);
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
			//Add this
			goto default;
			break;
		case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
			nodes.id.tau[i] = get_4d_asym_flat_flrw_slab_tau(mrng.rng, tau0);
			if (USE_GSL) {
				idata.upper = nodes.id.tau[i];
				eta = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * a / alpha;
			} else
				eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
			#if EMBED_NODES
			nodes.crd->v(i) = eta;
			emb4 = get_4d_asym_flat_flrw_slab_cartesian(mrng.rng, nrng, r_max);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = eta;
			nodes.crd->x(i) = get_4d_asym_flat_flrw_slab_radius(mrng.rng, r_max);
			nodes.crd->y(i) = get_4d_asym_flat_flrw_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_flat_flrw_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - r_max) < tol);
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < r_max);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
			nodes.id.tau[i] = get_4d_asym_sph_flrw_slab_tau(mrng.rng, tau0);
			if (USE_GSL) {
				idata.upper = nodes.id.tau[i];
				eta = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * a / alpha;
			} else
				eta = tauToEtaFLRWExact(nodes.id.tau[i], a, alpha);
			#if EMBED_NODES
			nodes.crd->v(i) = eta;
			emb4 = get_4d_asym_sph_flrw_slab_cartesian(nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = eta;
			nodes.crd->x(i) = get_4d_asym_sph_flrw_slab_theta1(mrng.rng);
			nodes.crd->y(i) = get_4d_asym_sph_flrw_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_sph_flrw_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.id.tau[i] > 0.0f && nodes.id.tau[i] < tau0);
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < HALF_PI - zeta);
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < HALF_PI - zeta);
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
			//Add this
			goto default;
			break;
		default:
			throw CausetException("Spacetime parameters not supported!\n");
		}
	}
