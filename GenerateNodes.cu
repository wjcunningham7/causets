	NDistribution ndist(0.0, 1.0);
	NGenerator nrng(mrng.eng, ndist);
	float tol = 1.0e-8;
	for (int i = 0; i < N_tar; i++) {
		switch (spacetime) {
		case (2 | DE_SITTER | SLAB | POSITIVE | ASYMMETRIC):
			nodes.crd->x(i) = get_2d_asym_sph_deSitter_slab_eta(mrng.rng, zeta);
			#if EMBED_NODES
			float2 emb = get_2d_asym_sph_deSitter_slab_emb(mrng.rng);
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_asym_sph_deSitter_slab_theta(mrng.rng);
			#endif

			#if DEBUG
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < (float)(HALF_PI - zeta));
			#if EMBED_NODES
			assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
			#else
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
			#endif
			#endif
			break;
		case (2 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			nodes.crd->x(i) = get_2d_sym_sph_deSitter_slab_eta(mrng.rng, zeta);
			#if EMBED_NODES
			float2 emb = get_2d_sym_sph_deSitter_slab_emb(mrng.rng);
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_sym_sph_deSitter_slab_theta(mrng.rng);
			#endif

			#if DEBUG
			assert (fabs(nodes.crd->x(i)) < (float)(HALF_PI - zeta));
			#if EMBED_NODES
			assert (fabs(POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0) < tol);
			#else
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < TWO_PI);
			#endif
			#endif
			break;
		case (2 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			nodes.crd->x(i) = get_2d_asym_sph_deSitter_diamond_eta(mrng.rng);
			#if EMBED_NODES
			float2 emb = get_2d_asym_sph_deSitter_diamond_emb(mrng.rng, nodes.crd->x(i));
			nodes.crd->y(i) = emb.x;
			nodes.crd->z(i) = emb.y;
			#else
			nodes.crd->y(i) = get_2d_asym_sph_deSitter_diamond_theta(mrng.rng, nodes.crd->x(i));
			#endif

			#if DEBUG
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < (float)(HALF_PI - zeta));
			#if EMBED_NODES
			//
			#else
			//
			#endif
			#endif
			break;
		case (4 | DE_SITTER | SLAB | FLAT | ASYMMETRIC):
			#if EMBED_NODES
			nodes.crd->v(i) = get_4d_asym_flat_deSitter_slab_eta(mrng.rng, zeta, zeta1);
			float4 emb = get_4d_asym_flat_deSitter_slab_cartesian(mrng.rng, nrng, r_max);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_flat_deSitter_slab_eta(mrng.rng, zeta, zeta1);
			nodes.crd->x(i) = get_4d_asym_flat_deSitter_slab_radius(mrng.rng, r_max);
			nodes.crd->y(i) = get_4d_asym_flat_deSitter_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_flat_deSitter_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
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
			nodes.crd->v(i) = get_4d_asym_sph_deSitter_slab_eta(mrng.rng);
			float4 emb = get_4d_asym_sph_deSitter_slab_emb(nrng);
			nodes.crd->w(i) = emb.w;
			nodes.crd->x(i) = emb.x;
			nodes.crd->y(i) = emb.y;
			nodes.crd->z(i) = emb.z;
			#else
			nodes.crd->w(i) = get_4d_asym_sph_deSitter_slab_eta(mrng.rng);
			nodes.crd->x(i) = get_4d_asym_sph_deSitter_slab_theta1(mrng.rng);
			nodes.crd->y(i) = get_4d_asym_sph_deSitter_slab_theta2(mrng.rng);
			nodes.crd->z(i) = get_4d_asym_sph_deSitter_slab_theta3(mrng.rng);
			#endif

			#if DEBUG
			#if EMBED_NODES
			assert (nodes.crd->v(i) > 0.0f && nodes.crd->v(i) < (float)(HALF_PI - zeta));
			assert (fabs(POW2(nodes.crd->w(i), EXACT) + POW2(nodes.crd->x(i), EXACT) + POW2(nodes.crd->y(i), EXACT) + POW2(nodes.crd->z(i), EXACT) - 1.0f) < tol);
			#else
			assert (nodes.crd->w(i) > 0.0f && nodes.crd->w(i) < (float)(HALF_PI - zeta));
			assert (nodes.crd->x(i) > 0.0f && nodes.crd->x(i) < M_PI);
			assert (nodes.crd->y(i) > 0.0f && nodes.crd->y(i) < M_PI);
			assert (nodes.crd->z(i) > 0.0f && nodes.crd->z(i) < TWO_PI);
			#endif
			#endif
			break;
		case (4 | DE_SITTER | SLAB | POSITIVE | SYMMETRIC):
			//
			break;
		case (4 | DE_SITTER | DIAMOND | FLAT | ASYMMETRIC):
			//
			break;
		case (4 | DE_SITTER | DIAMOND | POSITIVE | ASYMMETRIC):
			//
			break;
		case (4 | DUST | SLAB | FLAT | ASYMMETRIC):
			//
			break;
		case (4 | DUST | DIAMOND | FLAT | ASYMMETRIC):
			//
			break;
		case (4 | FLRW | SLAB | FLAT | ASYMMETRIC):
			//
			break;
		case (4 | FLRW | SLAB | POSITIVE | ASYMMETRIC):
			//
			break;
		case (4 | FLRW | DIAMOND | FLAT | ASYMMETRIC):
			//
			break;
		default:
			throw CausetException("Spacetime parameters not supported!\n");
		}
	}
