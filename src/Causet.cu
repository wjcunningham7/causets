#include "NetworkCreator.h"
#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

int main(int argc, char **argv)
{
	//Initialize Data Structures
	CausetMPI cmpi = CausetMPI();
	CaResources ca = CaResources();
	CuResources cu = CuResources();
	Benchmark bm = Benchmark();

	//MPI Initialization
	#ifdef MPI_ENABLED
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &cmpi.num_mpi_threads);
	MPI_Comm_rank(MPI_COMM_WORLD, &cmpi.rank);
	#endif

	CausetPerformance cp = CausetPerformance();
	stopwatchStart(&cp.sCauset);

	//Initialize 'Network' structure
	Network network = Network(parseArgs(argc, argv, &cmpi));

	int e_start = printStart((const char**)argv, cmpi.rank);
	bool success = false;

	#ifdef CUDA_ENABLED
	//Identify and Connect to GPU
	if (network.network_properties.flags.use_gpu)
		connectToGPU(&cu, argc, argv, cmpi.rank);
	#endif

	//Create Causal Set Graph	
	if (network.network_properties.graphID == 0 && !initializeNetwork(&network, &ca, &cp, &bm, cu.cuContext)) goto CausetExit;
	else if (network.network_properties.graphID != 0 && !loadNetwork(&network, &ca, &cp, &bm, cu.cuContext)) goto CausetExit;

	//Measure Graph Properties
	if (network.network_properties.flags.test) goto CausetPoint2;
	if (!measureNetworkObservables(&network, &ca, &cp, &bm)) goto CausetExit;

	//Print Results
	if (!network.network_properties.flags.bench) printMemUsed(NULL, ca.maxHostMemUsed, ca.maxDevMemUsed, cmpi.rank);
	#ifdef MPI_ENABLED
	if (!cmpi.rank) {
	#endif
	if (network.network_properties.flags.bench && !printBenchmark(bm, network.network_properties.flags, network.network_properties.flags.link, network.network_properties.flags.relink)) {
		cmpi.fail = 1;
		goto CausetPoint1;
	}
	if (network.network_properties.flags.print_network && !printNetwork(network, cp, cu.gpuID)) {
		cmpi.fail = 1;
		goto CausetPoint1;
	}
	#ifdef MPI_ENABLED
	}
	#endif

	CausetPoint1:
	if (checkMpiErrors(cmpi)) goto CausetExit;

	//Free Resources
	destroyNetwork(&network, ca.hostMemUsed, ca.devMemUsed);

	CausetPoint2:
	#ifdef CUDA_ENABLED
	//Release GPU
	if (network.network_properties.flags.use_gpu) cuCtxDetach(cu.cuContext);
	#endif

	//Identify Potential Memory Leaks
	success = !ca.hostMemUsed && !ca.devMemUsed;
	if (!success) {
		printf_mpi(cmpi.rank, "WARNING: Memory leak detected!\n");
		#if DEBUG
		printf_mpi(cmpi.rank, "Epsilon = %zd bytes.\n", ca.hostMemUsed);
		#endif
	}

	CausetExit:
	//Exit Program
	stopwatchStop(&cp.sCauset);

	printFinish((const char**)argv, e_start, cmpi.rank, success ? PASSED : FAILED);

	printf_mpi(cmpi.rank, "Time: %5.6f sec\n", cp.sCauset.elapsedTime);
	printf_mpi(cmpi.rank, "PROGRAM COMPLETED\n\n");
	fflush(stdout);
	
	#ifdef MPI_ENABLED
	MPI_Finalize();
	#endif

	return 0;
}

//Parse Command Line Arguments
NetworkProperties parseArgs(int argc, char **argv, CausetMPI *cmpi)
{
	NetworkProperties network_properties = NetworkProperties();
	if (cmpi != NULL)
		network_properties.cmpi = *cmpi;
	int rank = cmpi->rank;

	int c, longIndex;
	//Single-character options
	static const char *optString = ":A:a:b:Cc:k:d:e:F:Gg:hm:n:r:s:S:vyz:";
	//Multi-character options
	static const struct option longOpts[] = {
		{ "action",	required_argument,	NULL, 'A' },
		{ "age",	required_argument,	NULL, 'a' },
		{ "alpha",	required_argument,	NULL,  0  },
		{ "autocorr",	no_argument,		NULL,  0  },
		{ "benchmark",	no_argument,		NULL,  0  },
		{ "buffer",	required_argument,	NULL, 'b' },
		{ "clustering",	no_argument,		NULL, 'C' },
		{ "components", no_argument,		NULL,  0  },
		{ "core",	required_argument,	NULL, 'c' },
		{ "curvature",	required_argument,	NULL,  0  },
		{ "delta",	required_argument,	NULL, 'd' },
		{ "distances",	required_argument,	NULL,  0  },
		{ "embedding",  required_argument,	NULL,  0  },
		{ "energy",	required_argument,	NULL, 'e' },
		{ "fields",	required_argument,	NULL, 'F' },
		{ "gen-ds-table", no_argument,		NULL,  0  },
		{ "gen-flrw-table", no_argument,	NULL,  0  },
		//{ "geodesics",	no_argument,		NULL, 'G' },
		{ "gpu", 	no_argument, 		NULL,  0  },
		{ "graph",	required_argument,	NULL, 'g' },
		{ "help", 	no_argument,		NULL, 'h' },
		{ "quiet-read",	no_argument,		NULL,  0  },
		{ "link",	no_argument,		NULL,  0  },
		{ "manifold",	required_argument,	NULL, 'm' },
		{ "print", 	no_argument, 		NULL,  0  },
		{ "nodes", 	required_argument,	NULL, 'n' },
		{ "nopos",	no_argument,		NULL,  0  },
		{ "read-old-format", no_argument,	NULL,  0  },
		{ "region",	required_argument,	NULL, 'r' },
		{ "relink",	no_argument,		NULL,  0  },
		{ "seed",	required_argument,	NULL, 's' },
		{ "slice",	required_argument,	NULL,  0  },
		{ "spacetime",	required_argument,	NULL,  0  },
		{ "stdim",	required_argument,	NULL,  0  },
		{ "success",	required_argument,	NULL, 'S' },
		{ "symmetry",	required_argument,	NULL,  0  },
		{ "test",	no_argument,		NULL,  0  },
		{ "verbose", 	no_argument, 		NULL,  0  },
		{ "version",	no_argument,		NULL, 'v' },
		{ "zeta",	required_argument,	NULL, 'z' },
		{ NULL,		0,			0,     0  }
	};

	try {
		while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
			switch (c) {
			case 'A':	//Flag for calculating action
				network_properties.flags.calc_action = true;
				if (!strcmp(optarg, "local"))
					network_properties.max_cardinality = -1;
				else if (!strcmp(optarg, "smeared"))
					network_properties.max_cardinality = 1;
				else
					throw CausetException("Invalid argument for 'action' parameter!\n");
				break;
			case 'a':
				//Temporal cutuff (age)
				network_properties.tau0 = atof(optarg);
				if (network_properties.tau0 <= 0.0)
					throw CausetException("Invalid argument for 'age' parameter!\n");
				network_properties.omegaL = POW2(tanh(1.5 * network_properties.tau0), EXACT);
				network_properties.omegaM = 1.0 - network_properties.omegaL;
				break;
			case 'b':
				//Edge buffer in adjacency lists
				network_properties.edge_buffer = atof(optarg);
				if (network_properties.edge_buffer < 0.0 || network_properties.edge_buffer > 1.0)
					throw CausetException("Invalid argument for 'buffer' parameter!\n");
				break;
			case 'C':	//Flag for calculating clustering
				network_properties.flags.calc_clustering = true;
				break;
			case 'c':	//Core edge fraction (used for adjacency matrix)
				network_properties.core_edge_fraction = atof(optarg);
				if (network_properties.core_edge_fraction == 1.0)
					network_properties.flags.use_bit = true;
				if (network_properties.core_edge_fraction < 0.0 || network_properties.core_edge_fraction > 1.0)
					throw CausetException("Invalid argument for 'c' parameter!\n");
				break;
			case 'd':	//Density of nodes
				network_properties.delta = atof(optarg);
				if (network_properties.delta <= 0.0)
					throw CausetException("Invalid argument for 'delta' parameter!\n");
				break;
			case 'e':	//Density of dark energy
				network_properties.omegaL = atof(optarg);
				if (network_properties.omegaL <= 0.0 || network_properties.omegaL >= 1.0)
					throw CausetException("Invalid input for 'energy' parameter!\n");
				network_properties.omegaM = 1.0 - network_properties.omegaL;
				network_properties.tau0 = atanh(SQRT(network_properties.omegaL, STL)) / 1.5;
				break;
			case 'F':	//Measure Degree Field with Test Nodes
				network_properties.flags.calc_deg_field = true;
				network_properties.tau_m = atof(optarg);
				break;
			/*case 'G':	//Flag for estimating geodesics
				network_properties.flags.calc_geodesics = true;
				break;*/
			case 'g':	//Graph ID
				network_properties.graphID = atoi(optarg);
				if (network_properties.graphID < 0)
					throw CausetException("Invalid argument for 'Graph ID' parameter!\n");
				break;
			//case 'h' is located at the end
			case 'm':	//Manifold
				if (!strcmp(optarg, "minkowski"))
					network_properties.spacetime |= MINKOWSKI;
				else if (!strcmp(optarg, "milne"))
					network_properties.spacetime |= MILNE;
				else if (!strcmp(optarg, "desitter"))
					network_properties.spacetime |= DE_SITTER;
				else if (!strcmp(optarg, "dust"))
					network_properties.spacetime |= DUST;
				else if (!strcmp(optarg, "flrw"))
					network_properties.spacetime |= FLRW;
				else if (!strcmp(optarg, "hyperbolic"))
					network_properties.spacetime |= HYPERBOLIC;
				else
					throw CausetException("Invalid argument for 'manifold' parameter!\n");
				break;
			case 'n':	//Number of nodes
				network_properties.N_tar = atoi(optarg);
				if (network_properties.N_tar <= 0)
					throw CausetException("Invalid argument for 'nodes' parameter!\n");
				break;
			case 'r':	//Region
				if (!strcmp(optarg, "slab"))
					network_properties.spacetime |= SLAB;
				else if (!strcmp(optarg, "diamond"))
					network_properties.spacetime |= DIAMOND;
				else
					throw CausetException("Invalid argument for 'region' parameter!\n");
				break;
			case 'S':	//Flag for calculating success ratio
				network_properties.flags.calc_success_ratio = true;
				network_properties.flags.calc_components = true;
				network_properties.N_sr = atof(optarg);
				if (network_properties.N_sr <= 0.0)
					throw CausetException("Invalid argument for 'success' parameter!\n");
				break;
			case 's':	//Random seed
				#ifdef _OPENMP
				printf_dbg("NOTE: OpenMP produces non-deterministic behavior.\nCompile without OpenMP to set the seed.\nProgram will abort.\n\n");
				network_properties.flags.test = true;
				#endif
				network_properties.seed = atol(optarg);
				if (network_properties.seed <= 0L)
					throw CausetException("Invalid argument for 'seed' parameter!\n");
				break;
			//case 'v' is located at the end
			case 'y':	//Suppress user input
				network_properties.flags.yes = true;
				break;
			case 'z':	//Zeta
				network_properties.zeta = atof(optarg);
				if (network_properties.zeta == 0.0)
					throw CausetException("Invalid argument for 'zeta' parameter!\n");
			case 0:
				if (!strcmp("alpha", longOpts[longIndex].name)) {
					//Taken to be \tilde{\alpha} when input
					network_properties.alpha = atof(optarg);
					if (network_properties.alpha <= 0.0)
						throw CausetException("Invalid argument for 'alpha' parameter!\n");
				} else if (!strcmp("autocorr", longOpts[longIndex].name))
					//Flag to calculate autocorrelation of selected variables
					network_properties.flags.calc_autocorr = true;
				else if (!strcmp("benchmark", longOpts[longIndex].name))
					//Flag to benchmark selected routines
					network_properties.flags.bench = true;
				else if (!strcmp("components", longOpts[longIndex].name))
					//Flag for Finding Connected Components
					network_properties.flags.calc_components = true;
				else if (!strcmp("curvature", longOpts[longIndex].name)) {
					//Flag for setting spatial curvature
					if (!strcmp(optarg, "flat"))
						network_properties.spacetime |= FLAT;
					else if (!strcmp(optarg, "positive"))
						network_properties.spacetime |= POSITIVE;
					else
						throw CausetException("Invalid argument for 'curvature' parameter!\n");
				} else if (!strcmp("distances", longOpts[longIndex].name)) {
					//Flag for comparing distance methods
					network_properties.flags.validate_distances = true;
					network_properties.N_dst = atof(optarg);
					if (network_properties.N_dst <= 0.0)
						throw CausetException("Invalid argument for 'distances' parameter!\n");
				} else if (!strcmp("embedding", longOpts[longIndex].name)) {
					//Flag to validate embedding of FLRW into de Sitter
					network_properties.flags.validate_embedding = true;
					network_properties.N_emb = atof(optarg);
					if (network_properties.N_emb <= 0.0)
						throw CausetException("Invalid argument for 'embedding' parameter!\n");
				} else if (!strcmp("gen-ds-table", longOpts[longIndex].name))
					network_properties.flags.gen_ds_table = true;
				else if (!strcmp("gen-flrw-table", longOpts[longIndex].name))
					network_properties.flags.gen_flrw_table = true;
				else if (!strcmp("gpu", longOpts[longIndex].name)) {
					//Flag to use GPU accelerated routines
					network_properties.flags.use_gpu = true;
					#ifndef CUDA_ENABLED
					throw CausetException("Recompile with 'make gpu' to use the --gpu flag!\n");
					#endif
				} else if (!strcmp("link", longOpts[longIndex].name))
					//Flag for Reading Nodes (and not links) and Re-Linking
					network_properties.flags.link = true;
				else if (!strcmp("nopos", longOpts[longIndex].name))
					//Flag to Skip Node Generation/Reading
					network_properties.flags.no_pos = true;
				else if (!strcmp("print", longOpts[longIndex].name))
					//Flag to print results to file in 'dat' folder
					network_properties.flags.print_network = true;
				else if (!strcmp("quiet-read", longOpts[longIndex].name))
					//Flag to ignore warnings when reading graph
					network_properties.flags.quiet_read = true;
				else if (!strcmp("read-old-format", longOpts[longIndex].name))
					network_properties.flags.read_old_format = true;
				else if (!strcmp("relink", longOpts[longIndex].name))
					//Flag for Reading Nodes (and not links) and Re-Linking
					network_properties.flags.relink = true;
				else if (!strcmp("slice", longOpts[longIndex].name))
					//Size of spatial slice
					network_properties.r_max = atof(optarg);
				else if (!strcmp("spacetime", longOpts[longIndex].name)) {
					//Spacetime ID
					network_properties.spacetime = atoi(optarg);
				} else if (!strcmp("stdim", longOpts[longIndex].name)) {
					//Spacetime dimensions (2 or 4 right now)
					if (atoi(optarg) == 2 || atoi(optarg) == 4)
						network_properties.spacetime |= atoi(optarg);
					else
						throw CausetException("Invalid argument for 'stdim' parameter!\n");
				} else if (!strcmp("symmetry", longOpts[longIndex].name)) {
					//Symmetric about temporal axis
					if (!strcmp(optarg, "symmetric"))
						network_properties.spacetime |= SYMMETRIC;
					else if (!strcmp(optarg, "asymmetric"))
						network_properties.spacetime |= ASYMMETRIC;
					else
						throw CausetException("Invalid argument for 'symmetry' parameter!\n");
				} else if (!strcmp("test", longOpts[longIndex].name))
					//Test parameters
					network_properties.flags.test = true;
				else if (!strcmp("verbose", longOpts[longIndex].name))
					//Verbose output
					network_properties.flags.verbose = true;
				else {
					//Unrecognized options
					fprintf(stderr, "Option --%s is not recognized.\n", longOpts[longIndex].name);
					#ifdef MPI_ENABLED
					MPI_Abort(MPI_COMM_WORLD, 5);
					#else
					exit(5);
					#endif
				}
				break;
			case 'v':
				//Print the version information
				printf_mpi(rank, "CausalSet (Northeastern Causal Set Simulator)\n");
				printf_mpi(rank, "Copyright (C) 2013-2016 Will Cunningham\n");
				printf_mpi(rank, "Platform: Redhat Linux x86 64-bit Kernel\n");
				printf_mpi(rank, "Developed for use with NVIDIA K20m GPU\n");
				printf_mpi(rank, "Version 0.3\n");
				printf_mpi(rank, "See doc/VERSION for supported spacetimes\n");
				#ifdef MPI_ENABLED
				MPI_Abort(MPI_COMM_WORLD, 0);
				#else
				exit(0);
				#endif
			case 'h':
				//Print help menu
				printf_mpi(rank, "\nUsage  :  CausalSet [options]\n\n");
				printf_mpi(rank, "CausalSet Options...................\n");
				printf_mpi(rank, "====================================\n");
				printf_mpi(rank, "Flag:\t\t\tMeaning:\t\t\tSuggested Values:\n");
				printf_mpi(rank, "  -A, --action\t\tMeasure Action\t\t\t\"local\", \"smeared\"\n");
				printf_mpi(rank, "  -a, --age\t\tRescaled (Conformal) Age of\n");
				printf_mpi(rank, "\t\t\tFLRW (de Sitter) Spacetime\t0.85\n");
				printf_mpi(rank, "      --alpha\t\tSpatial Scaling/Cutoff\t\t2.0\n");
				//printf_mpi(rank, "      --autocorr\tCalculate Autocorrelations\n");
				printf_mpi(rank, "      --benchmark\tBenchmark Algorithms\n");
				printf_mpi(rank, "  -b, --buffer\t\tEdge Buffer\t\t\t0.3\n");
				printf_mpi(rank, "  -C, --clustering\tMeasure Clustering\n");
				printf_mpi(rank, "\t\t\tor Spherical Foliation\n");
				printf_mpi(rank, "      --components\tMeasure Graph Components\n");
				printf_mpi(rank, "  -c, --core\t\tCore Edge Fraction\t\t0.01\n");
				printf_mpi(rank, "      --curvature\tSpatial Curvature\t\t\"flat\", \"positive\"\n");
				printf_mpi(rank, "  -d, --delta\t\tNode Density\t\t\t10000\n");
				printf_mpi(rank, "      --distances\tValidate Distance Methods\t0.01, 10000\n");
				printf_mpi(rank, "      --embedding\tValidate Embedding\t\t0.01, 10000\n");
				printf_mpi(rank, "  -e, --energy\t\tDark Energy Density\t\t0.73\n");
				printf_mpi(rank, "  -F, --fields\t\tMeasure Degree Fields\n");
				printf_mpi(rank, "      --gen-ds-table\tGenerate de Sitter\n");
				printf_mpi(rank, "\t\t\tGeodesic Table\n");
				printf_mpi(rank, "      --gen-flrw-table\tGenerate FLRW Geodesic Table\n");
				//printf_mpi(rank, "  -G, --geodesics\tGeodesic Estimator\n");
				#ifdef CUDA_ENABLED
				printf_mpi(rank, "      --gpu\t\tUse GPU Acceleration\n");
				#endif
				printf_mpi(rank, "  -g, --graph\t\tGraph ID\t\t\tCheck dat/*.cset.out files\n");
				printf_mpi(rank, "  -h, --help\t\tDisplay This Menu\n");
				printf_mpi(rank, "      --link\t\tLink Nodes to Create Graph\n");
				printf_mpi(rank, "  -m, --manifold\tManifold\t\t\t\"desitter\", \"dust\",\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"flrw\", \"hyperbolic\"\n");
				printf_mpi(rank, "  -n, --nodes\t\tNumber of Nodes\t\t\t1000, 10000, 100000\n");
				printf_mpi(rank, "      --nopos\t\tNo Node Positions\n");
				printf_mpi(rank, "      --print\t\tPrint Results\n");
				printf_mpi(rank, "      --quiet-read\tIgnore any warnings when\n");
				printf_mpi(rank, "\t\t\treading graph\n");
				printf_mpi(rank, "      --read-old-format\tRead Positions in Old Format\n");
				printf_mpi(rank, "  -r, --region\t\tSpacetime Region\t\t\"slab\", \"diamond\"\n");
				printf_mpi(rank, "      --relink\t\tIgnore Pre-Existing Links\n");
				printf_mpi(rank, "  -S, --success\t\tCalculate Success Ratio\t\t0.01, 10000\n");
				printf_mpi(rank, "  -s, --seed\t\tRandom Seed\t\t\t18100\n");
				printf_mpi(rank, "      --slice\t\tSize of Spatial Slice\t\t1.0\n");
				printf_mpi(rank, "      --stdim\t\tSpacetime Dimensions\t\t2 or 4\n");
				printf_mpi(rank, "      --symmetry\tTemporal Symmetry\t\t\"symmetric\", \"asymmetric\"\n");
				printf_mpi(rank, "      --test\t\tTest Parameters Only\n");
				printf_mpi(rank, "      --verbose\t\tVerbose Output\n");
				printf_mpi(rank, "  -y\t\t\tSuppress User Queries\n");
				printf_mpi(rank, "  -z, --zeta\t\tHyperbolic Curvature\t\t1.0\n");
				printf_mpi(rank, "\n");

				printf_mpi(rank, "Report bugs to w.cunningham@neu.edu\n");
				printf_mpi(rank, "GitHub repository home page: <https://bitbucket.org/dk-lab/2015_code_causets>\n");
				#ifdef MPI_ENABLED
				MPI_Abort(MPI_COMM_WORLD, 0);
				#else
				exit(0);
				#endif
			case ':':
				//Single-character flag needs an argument
				if (!!optopt)
					fprintf(stderr, "%s : option '-%c' requires an argument\n", argv[0], optopt);
				else
					fprintf(stderr, "%s : option '%s' requires an argument\n", argv[0], argv[optind-1]);
				#ifdef MPI_ENABLED
				MPI_Abort(MPI_COMM_WORLD, 6);
				#else
				exit(6);
				#endif
			case '?':	//Unrecognized flag
			default:	//Default case
				if (!!optopt)
					fprintf(stderr, "%s : option '-%c' is not recognized.\n", argv[0], optopt);
				else
					fprintf(stderr, "%s : option '%s' is not recognized.\n", argv[0], argv[optind-1]);
				#ifdef MPI_ENABLED
				MPI_Abort(MPI_COMM_WORLD, 7);
				#else
				exit(7);
				#endif
			}
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 8);
		#else
		exit(8);
		#endif
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		#ifdef MPI_ENABLED
		MPI_Abort(MPI_COMM_WORLD, 9);
		#else
		exit(9);
		#endif
	}
	
	//Initialize RNG
	if (network_properties.seed == 12345L) {
		srand(time(NULL));
		network_properties.seed = static_cast<long>(time(NULL));
		network_properties.mrng.rng.engine().seed(network_properties.seed);
		network_properties.mrng.rng.distribution().reset();
	}

	return network_properties;
}

//Handles all network generation and initialization procedures
bool initializeNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	#endif

	int rank = network->network_properties.cmpi.rank;
	int nb = static_cast<int>(network->network_properties.flags.bench) * NBENCH;
	int i;

	#ifdef MPI_ENABLED
	printf_mpi(rank, "\n\t[ ***   MPI MODULE ACTIVE  *** ]\n");
	#endif

	#ifdef _OPENMP
	printf_mpi(rank, "\n\t[ *** OPENMP MODULE ACTIVE *** ]\n");
	#endif

	//Initialize variables using constraints
	if (!initVars(&network->network_properties, ca, cp, bm))
		return false;

	//If 'test' flag specified, exit here
	if (network->network_properties.flags.test)
		return true;

	printf_mpi(rank, "\nInitializing Network...\n");
	fflush(stdout);

	//Allocate memory needed by pointers
	for (i = 0; i <= nb; i++) {
		if (!createNetwork(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.group_size, ca, cp->sCreateNetwork, network->network_properties.flags.use_gpu, network->network_properties.flags.decode_cpu, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;
		if (!!(i-nb))
			destroyNetwork(network, ca->hostMemUsed, ca->devMemUsed);
	}

	if (!!nb)
		bm->bCreateNetwork = cp->sCreateNetwork.elapsedTime / NBENCH;

	#ifdef MPI_ENABLED
	if (!rank) {
	#endif
	//Generate coordinates of spacetime nodes and then order nodes temporally using quicksort
	int low = 0;
	int high = network->network_properties.N_tar - 1;

	for (i = 0; i <= nb; i++) {
		if (!generateNodes(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.a, network->network_properties.eta0, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.mrng, cp->sGenerateNodes, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
			network->network_properties.cmpi.fail = 1;
			goto InitExit;
		}

		//Quicksort
		stopwatchStart(&cp->sQuicksort);
		quicksort(network->nodes, network->network_properties.spacetime, low, high);
		stopwatchStop(&cp->sQuicksort);
	}

	if (!nb) {
		printf("\tQuick Sort Successfully Performed.\n");
		if (get_symmetry(network->network_properties.spacetime) & ASYMMETRIC && !(get_manifold(network->network_properties.spacetime) & HYPERBOLIC)) {
			printf_cyan();
			float min_time = 0.0;
			if (get_manifold(network->network_properties.spacetime) & MINKOWSKI) {
				if (get_stdim(network->network_properties.spacetime) == 2)
					min_time = network->nodes.crd->x(0);
			} else
				min_time = network->nodes.id.tau[0];
			printf("\t\tMinimum Rescaled Time:  %f\n", min_time);
			printf_std();
		}
	} else {
		bm->bGenerateNodes = cp->sGenerateNodes.elapsedTime / NBENCH;
		bm->bQuicksort = cp->sQuicksort.elapsedTime / NBENCH;
	}

	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sQuicksort.elapsedTime);
	fflush(stdout);

	//Identify edges as points connected by timelike intervals
	if (network->network_properties.flags.link) {
		for (i = 0; i <= nb; i++) {
			#ifdef CUDA_ENABLED
			if (network->network_properties.flags.use_gpu) {
				#if LINK_NODES_GPU_V2
				if (!linkNodesGPU_v2(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.group_size, ca, cp->sLinkNodes, ctx, network->network_properties.flags.decode_cpu, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto InitExit;
				}
				#else
				if (!linkNodesGPU_v1(network->nodes, network->edges, network->adj, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, ca, cp->sLinkNodesGPU, network->network_properties.flags.compact, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto InitExit;
				}
				#endif
			} else {
			#endif
				if (!linkNodes(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto InitExit;
				}
			#ifdef CUDA_ENABLED
			}
			#endif
		}

		if (nb) {
			if (network->network_properties.flags.use_gpu)
				bm->bLinkNodesGPU = cp->sLinkNodesGPU.elapsedTime / NBENCH;
			else
				bm->bLinkNodes = cp->sLinkNodes.elapsedTime / NBENCH;
		}
	}
	#ifdef MPI_ENABLED
	}
	#endif

	//Use this statement if you're creating a degree lookup table
	/*if (network->network_properties.flags.link)
		printf("rad: %.16f\n", network->network_observables.k_res / (network->network_properties.delta * pow(network->network_properties.a, (double)get_stdim(network->network_properties.spacetime))));*/

	InitExit:
	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	printf_mpi(rank, "Task Completed.\n");
	fflush(stdout);
	return true;
}

bool measureNetworkObservables(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	#endif

	//if (network->nodes.crd->isNull())
	//	printf("rank: %d\n", network->network_properties.cmpi.rank);
	//MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);
	
	if (!network->network_properties.flags.calc_clustering && !network->network_properties.flags.calc_components && !network->network_properties.flags.validate_embedding && !network->network_properties.flags.calc_success_ratio && !network->network_properties.flags.calc_deg_field && !network->network_properties.flags.validate_distances && !network->network_properties.flags.calc_action /*&& !network->network_properties.flags.calc_geodesics*/)
		return true;

	int rank = network->network_properties.cmpi.rank;
	bool links_exist = network->network_properties.flags.link || network->network_properties.flags.relink;
	int nb = static_cast<int>(network->network_properties.flags.bench) * NBENCH;
	int i;
		
	printf_mpi(rank, "\nCalculating Network Observables...\n");
	fflush(stdout);

	#ifdef MPI_ENABLED
	if (!rank) {
	#endif
	//Measure Clustering
	if (network->network_properties.flags.calc_clustering) {
		for (i = 0; i <= nb; i++) {
			if (!links_exist) {
				printf_red();
				printf("\tCannot calculate clustering if links do not exist!\n");
				printf("\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_clustering = false;
				break;
			} else if (network->network_properties.flags.use_bit) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate clustering with only the bit array.\n");
				printf_mpi(rank, "\tImplement new algorithm to proceed.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_clustering = false;
				break;
			}

			if (!measureClustering(network->network_observables.clustering, network->nodes, network->edges, network->adj, network->network_observables.average_clustering, network->network_properties.N_tar, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, ca, cp->sMeasureClustering, network->network_properties.flags.calc_autocorr, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
		}

		if (nb)
			bm->bMeasureClustering = cp->sMeasureClustering.elapsedTime / NBENCH;
	}
	#ifdef MPI_ENABLED
	}
	#endif

	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	//Measure Connectedness
	if (network->network_properties.flags.calc_components) {
		for (i = 0; i <= nb; i++) {
			if (!links_exist) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate connected components if links do not exist!\n");
				printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_components = false;
				break;
			}

			if (!measureConnectedComponents(network->nodes, network->edges, network->adj, network->network_properties.N_tar, network->network_properties.cmpi, network->network_observables.N_cc, network->network_observables.N_gcc, ca, cp->sMeasureConnectedComponents, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}

		if (nb)
			bm->bMeasureConnectedComponents = cp->sMeasureConnectedComponents.elapsedTime / NBENCH;
	}

	//Validate Embedding
	if (network->network_properties.flags.validate_embedding) {
		if (network->network_properties.flags.no_pos) {
			if (!rank) printf_red();
			printf_mpi(rank, "\tCannot validate embedding if positions do not exist!\n");
			printf_mpi(rank, "\tRemove flag '--nopos' to read node positions.\n\tSkipping this subroutine.\n\n");
			if (!rank) printf_std();
			fflush(stdout);
			network->network_properties.flags.validate_embedding = false;
		} else if (network->network_properties.flags.use_bit) {
			if (!rank) printf_red();
			printf_mpi(rank, "\tCannot validate embedding with only the bit array.\n");
			printf_mpi(rank, "\tImplement new algorithm to proceed.\n\tSkipping this subroutine.\n\n");
			if (!rank) printf_std();
			fflush(stdout);
			network->network_properties.flags.validate_embedding = false;
		}

		if (!validateEmbedding(network->network_observables.evd, network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_emb, network->network_observables.N_res, network->network_observables.k_res, network->network_properties.a, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.mrng, ca, cp->sValidateEmbedding, network->network_properties.flags.verbose))
			return false;
	}

	//Measure Success Ratio
	if (network->network_properties.flags.calc_success_ratio) {
		for (i = 0; i <= nb; i++) {
			if (!links_exist) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate success ratio if links do not exist!\n");
				printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_success_ratio = false;
				break;
			} else if (network->network_properties.flags.no_pos) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate success ratio if positions do not exist!\n");
				printf_mpi(rank, "\tRemove flag '--nopos' to read node positions.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_success_ratio = false;
				break;
			}

			if (!measureSuccessRatio(network->nodes, network->edges, network->adj, network->network_observables.success_ratio, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_sr, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.mrng, ca, cp->sMeasureSuccessRatio, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}

		if (nb)
			bm->bMeasureSuccessRatio = cp->sMeasureSuccessRatio.elapsedTime / NBENCH;
	}

	#ifdef MPI_ENABLED
	if (!rank) {
	#endif
	//Measure Degree Fields
	if (network->network_properties.flags.calc_deg_field) {
		for (i = 0; i <= nb; i++) {
			if (network->network_properties.flags.no_pos) {
				printf_red();
				printf("\tCannot calculate degree fields if positions do not exist!\n");
				printf("\tRemove flag '--nopos' to read node positions.\n\tSkipping this subroutine.\n\n");
				printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_deg_field = false;
				break;
			}

			if (!measureDegreeField(network->network_observables.in_degree_field, network->network_observables.out_degree_field, network->network_observables.avg_idf, network->network_observables.avg_odf, network->nodes.crd, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.N_df, network->network_properties.tau_m, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.alpha, network->network_properties.delta, ca, cp->sMeasureDegreeField, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}
		}

		if (nb)
			bm->bMeasureDegreeField = cp->sMeasureDegreeField.elapsedTime / NBENCH;
	}

	//Validate Distance Methods
	if (network->network_properties.flags.validate_distances) {
		if (network->network_properties.flags.no_pos) {
			printf_red();
			printf("\tCannot validate distances if positions do not exist!\n");
			printf("\tRemove flag '--nopos' to read node positions.\n\tSkipping this subroutine.\n\n");
			printf_std();
			fflush(stdout);
			network->network_properties.flags.validate_distances = false;
		}

		if (!validateDistances(network->network_observables.dvd, network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.N_dst, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.mrng, ca, cp->sValidateDistances, network->network_properties.flags.verbose)) {
			network->network_properties.cmpi.fail = 1;
			goto MeasureExit;
		}
	}
	#ifdef MPI_ENABLED
	}
	#endif

	//Measure Action
	if (network->network_properties.flags.calc_action) {
		for (i = 0; i <= nb; i++) {
			#if ACTION_GPU && !MPI_ENABLED
			network->network_properties.max_cardinality = network->network_properties.N_tar - 1;
			if (!measureActionGPU(network->network_observables.cardinalities, network->network_observables.action, network->nodes, network->edges, network->adj, network->network_properties.N_tar, network->network_properties.stdim, network->network_properties.manifold, network->network_properties.core_edge_fraction, ca, cp->sMeasureAction, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.compact, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			#elif ACTION_V2
			if (!measureAction_v2(network->network_observables.cardinalities, network->network_observables.action, network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.max_cardinality, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, ca, cp->sMeasureAction, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}
			#else
			if (!measureAction_v1(network->network_observables.cardinalities, network->network_observables.action, network->nodes, network->edges, network->adj, network->network_properties.N_tar, network->network_properties.max_cardinality, network->network_properties.stdim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, ca, cp->sMeasureAction, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.compact, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}
			#endif
		}

		if (nb)
			bm->bMeasureAction = cp->sMeasureAction.elapsedTime / NBENCH;
	}
	
	//Measure Geodesics w/ Geodesic Estimator

	MeasureExit:
	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	printf_mpi(rank, "Task Completed.\n");
	fflush(stdout);

	return true;
}

//Load Network Data from Existing File
//O(xxx) Efficiency (revise this)
//Reads the following files:
//	-Node position data		(./dat/pos/*.cset.pos.dat)
//	-Edge data			(./dat/edg/*.cset.edg.dat)
bool loadNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	assert (cp != NULL);

	//Values in Correct Ranges (add this)

	assert (network->network_properties.graphID);
	assert (!network->network_properties.flags.bench);
	#endif

	unsigned int spacetime = network->network_properties.spacetime;
	int rank = network->network_properties.cmpi.rank;

	#ifdef MPI_ENABLED
	printf_mpi(rank, "\n\t[ ***   MPI MODULE ACTIVE  *** ]\n");
	#endif

	#ifdef _OPENMP
	printf_mpi(rank, "\n\t[ *** OPENMP MODULE ACTIVE *** ]\n");
	#endif

	printf_mpi(rank, "\nLoading Graph from File.....\n");
	fflush(stdout);

	try {
		std::ifstream dataStream;
		std::stringstream dataname;
		std::string line;

		IntData idata = IntData();
		idata.limit = 50;
		idata.tol = 1e-4;
		if (USE_GSL && get_manifold(spacetime) & FLRW && !network->network_properties.flags.no_pos)
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

		uint64_t *edges;
		char *delimeters;

		uint64_t key;
		int N_edg;
		int node_idx;
		int i, j;
		unsigned int e0, e1;
		unsigned int idx0 = 0, idx1 = 0;

		std::string message;
		char d[] = " \t";
		delimeters = &d[0];
		N_edg = 0;

		#ifdef MPI_ENABLED
		if (!rank) {
		#endif

		//Identify Basic Network Properties
		if (!network->network_properties.flags.no_pos) {
			dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				while(getline(dataStream, line))
					network->network_properties.N_tar++;
				dataStream.close();
			} else {
				message = "Failed to open node position file!\n";
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint1;
			}
		}
	
		if (network->network_properties.flags.link) {	
			dataname.str("");
			dataname.clear();
			dataname << "./dat/edg/" << network->network_properties.graphID << ".cset.edg.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				while(getline(dataStream, line))
					N_edg++;
				dataStream.close();
			} else {
				message = "Failed to open edge list file!\n";
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint1;
			}
		}

		#ifdef MPI_ENABLED
		}
		#endif

		//printf("Number of Nodes: %d\n", network->network_properties.N_tar);
		//printf("Number of Edges: %d\n", N_edg);

		LoadPoint1:
		if (checkMpiErrors(network->network_properties.cmpi)) {
			if (!rank)
				throw CausetException(message.c_str());
			else
				return false;
		}

		#ifdef MPI_ENABLED
		//Broadcast:
		// > N_tar
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&network->network_properties.N_tar, 1, MPI_INT, 0, MPI_COMM_WORLD);
		#endif

		if (!initVars(&network->network_properties, ca, cp, bm))
			return false;
		if (!!N_edg)
			network->network_properties.k_tar = 2.0 * N_edg / network->network_properties.N_tar;

		printf_mpi(rank, "\nFinished Gathering Peripheral Network Data.\n");
		fflush(stdout);

		if (network->network_properties.flags.test)
			return true;

		unsigned int core_limit = static_cast<unsigned int>(network->network_properties.core_edge_fraction * network->network_properties.N_tar);

		//Allocate Memory	
		if (!createNetwork(network->nodes, network->edges, network->adj, spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.group_size, ca, cp->sCreateNetwork, network->network_properties.flags.use_gpu, network->network_properties.flags.decode_cpu, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;

		#ifdef MPI_ENABLED
		if (rank == 0) {
		#endif
		//Read Node Positions
		if (!network->network_properties.flags.no_pos) {
			printf("\tReading Node Position Data.....\n");
			fflush(stdout);
			dataname.str("");
			dataname.clear();
			dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				for (i = 0; i < network->network_properties.N_tar; i++) {
					getline(dataStream, line);

					if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
						if (get_stdim(spacetime) == 2) {
							network->nodes.crd->x(i) = atof(strtok((char*)line.c_str(), delimeters));
							network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
						} else if (get_stdim(spacetime) == 4) {
							if (get_manifold(spacetime) & FLRW) {
								network->nodes.id.tau[i] = atof(strtok((char*)line.c_str(), delimeters));
								if (USE_GSL) {
									idata.upper = network->nodes.id.tau[i];
									network->nodes.crd->w(i) = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * network->network_properties.a / network->network_properties.alpha;
								} else
									network->nodes.crd->w(i) = tauToEtaFLRWExact(network->nodes.id.tau[i], network->network_properties.a, network->network_properties.alpha);
							} else if (get_manifold(spacetime) & DUST) {
								network->nodes.id.tau[i] = atof(strtok((char*)line.c_str(), delimeters));
								network->nodes.crd->w(i) = tauToEtaDust(network->nodes.id.tau[i], network->network_properties.a, network->network_properties.alpha);
							} else if (get_manifold(spacetime) & DE_SITTER) {
								network->nodes.crd->w(i) = atof(strtok((char*)line.c_str(), delimeters));
								network->nodes.id.tau[i] = etaToTauSph(network->nodes.crd->w(i));
							}

							if (network->network_properties.flags.read_old_format) {
								network->nodes.crd->z(i) = atof(strtok(NULL, delimeters));
								network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
								network->nodes.crd->x(i) = atof(strtok(NULL, delimeters));
							} else {
								network->nodes.crd->x(i) = atof(strtok(NULL, delimeters));
								network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
								network->nodes.crd->z(i) = atof(strtok(NULL, delimeters));
							}
						}
					} else if (get_manifold(spacetime) & HYPERBOLIC) {
						network->nodes.id.AS[i] = atoi(strtok((char*)line.c_str(), delimeters));
						network->nodes.crd->x(i) = atof(strtok(NULL, delimeters));
						network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
					}

					if (!network->network_properties.flags.quiet_read) {
						if (get_stdim(spacetime) == 2) {
							if (get_symmetry(spacetime) & ASYMMETRIC && network->nodes.crd->x(i) < 0.0) {
								message = "Invalid value parsed for 'eta/r' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (network->nodes.crd->y(i) < 0.0 || network->nodes.crd->y(i) > TWO_PI) {
								message = "Invalid value for 'theta' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
						} else if (get_stdim(spacetime) == 4 && get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
							if ((get_symmetry(spacetime) & SYMMETRIC && fabs(network->nodes.crd->w(i)) > static_cast<float>(HALF_PI - network->network_properties.zeta)) || (get_symmetry(spacetime) & ASYMMETRIC && (network->nodes.crd->w(i) < 0.0 || network->nodes.crd->w(i) > static_cast<float>(HALF_PI - network->network_properties.zeta)))) {
								message = "Invalid value for 'eta' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (network->nodes.crd->x(i) < 0.0 || (get_curvature(spacetime) & POSITIVE && network->nodes.crd->x(i) > M_PI) || (get_curvature(spacetime) & FLAT && network->nodes.crd->x(i) > network->network_properties.r_max)) {
								message = "Invalid value for 'theta_1' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (network->nodes.crd->y(i) < 0.0 || network->nodes.crd->y(i) > M_PI) {
								message = "Invalid value for 'theta_2' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (network->nodes.crd->z(i) < 0.0 || network->nodes.crd->z(i) > TWO_PI) {
								message = "Invalid value for 'theta_3' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
						}
					}
				}
				dataStream.close();
			} else {
				message = "Failed to open node position file!\n";
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint2;
			}
			printf("\t\tCompleted.\n");

			if (USE_GSL && get_manifold(spacetime) & FLRW)
				gsl_integration_workspace_free(idata.workspace);

			//Quicksort
			quicksort(network->nodes, network->network_properties.spacetime, 0, network->network_properties.N_tar - 1);
		}

		//Re-Link Using linkNodes Subroutine
		if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW) && network->network_properties.flags.relink && !network->network_properties.flags.no_pos) {
			#ifdef CUDA_ENABLED
			if (network->network_properties.flags.use_gpu) {
				#if LINK_NODES_GPU_V2
				if (!linkNodesGPU_v2(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.group_size, ca, cp->sLinkNodes, ctx, network->network_properties.flags.decode_cpu, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto LoadPoint2;
				}
				#else
				if (!linkNodesGPU_v1(network->nodes, network->edges, network->adj, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, ca, cp->sLinkNodesGPU, network->network_properties.flags.compact, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto LoadPoint2;
				}
				#endif
			} else {
			#endif
				if (!linkNodes(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					goto LoadPoint2;
				}
			#ifdef CUDA_ENABLED
			}
			#endif

			goto LoadPoint2;
		} else if (!network->network_properties.flags.link)
			goto LoadPoint2;

		//Populate Hashmap
		if (get_manifold(spacetime) & HYPERBOLIC)
			for (i = 0; i < network->network_properties.N_tar; i++)
				network->nodes.AS_idx.insert(std::make_pair(network->nodes.id.AS[i], i));

		//Read Edges
		printf("\tReading Edge List Data.....\n");
		fflush(stdout);
		dataname.str("");
		dataname.clear();
		dataname << "./dat/edg/" << network->network_properties.graphID << ".cset.edg.dat";
		dataStream.open(dataname.str().c_str());
		if (dataStream.is_open()) {
			if (!network->network_properties.flags.use_bit) {
				edges = (uint64_t*)malloc(sizeof(uint64_t) * N_edg);
				if (edges == NULL) {
					message = "bad alloc";
					network->network_properties.cmpi.fail = 1;
					goto LoadPoint2;
				}
				memset(edges, 0, sizeof(uint64_t) * N_edg);
				ca->hostMemUsed += sizeof(uint64_t) * N_edg;
			} else
				edges = (uint64_t*)malloc(sizeof(uint64_t));

			for (i = 0; i < N_edg; i++) {
				getline(dataStream, line);
				e0 = atoi(strtok((char*)line.c_str(), delimeters));
				e1 = atoi(strtok(NULL, delimeters));
				//printf("%d %d\n", e0, e1);

				if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
					idx0 = e0;
					idx1 = e1;
				} else if (get_manifold(spacetime) & HYPERBOLIC) {
					idx0 = network->nodes.AS_idx.at(e0);
					idx1 = network->nodes.AS_idx.at(e1);
				}

				if (idx1 < idx0) {
					idx0 ^= idx1;
					idx1 ^= idx0;
					idx0 ^= idx1;
				}

				assert (idx0 < (unsigned int)network->network_properties.N_tar);
				assert (idx1 < (unsigned int)network->network_properties.N_tar);

				network->nodes.k_in[idx1]++;
				network->nodes.k_out[idx0]++;

				if (!network->network_properties.flags.use_bit)
					edges[i] = ((uint64_t)idx0) << 32 | ((uint64_t)idx1);

				if (idx0 < core_limit && idx1 < core_limit) {
					network->adj[(idx0*core_limit)+idx1] = true;
					network->adj[(idx1*core_limit)+idx0] = true;
				}
				//printf("%d %d\n", idx0, idx1);
			}
			printf("\t\tRead Raw Data.\n");
			fflush(stdout);

			if (!network->network_properties.flags.use_bit) {
				//Sort Edge List
				quicksort(edges, 0, N_edg - 1);
				printf("\t\tFirst Quicksort Performed.\n");

				//Future Edge Data
				node_idx = -1;
				for (i = 0; i < N_edg; i++) {
					key = edges[i];
					idx0 = key >> 32;
					idx1 = key & 0x00000000FFFFFFFF;
					network->edges.future_edges[i] = idx1;
					edges[i] = ((uint64_t)idx1) << 32 | ((uint64_t)idx0);
					//printf("%d %d\n", idx0, idx1);
			
					if ((int)idx0 != node_idx) {
						if (idx0 - node_idx > 1)
							for(j = 0; j < (int)idx0 - node_idx - 1; j++)
								network->edges.future_edge_row_start[idx0-j-1] = -1;
						network->edges.future_edge_row_start[idx0] = i;
						node_idx = idx0;
					}
				}
				for (i = idx0 + 1; i < network->network_properties.N_tar; i++)
					network->edges.future_edge_row_start[i] = -1;
				printf("\t\tFuture Edges Parsed.\n");
				fflush(stdout);

				//Resort Edge List
				quicksort(edges, 0, N_edg - 1);
				printf("\t\tSecond Quicksort Performed.\n");
				fflush(stdout);

				//Populate Past Edge List
				node_idx = -1;
				for (i = 0; i < N_edg; i++) {
					key = edges[i];
					idx0 = key >> 32;
					idx1 = key & 0x00000000FFFFFFFF;
					network->edges.past_edges[i] = idx1;

					if ((int)idx0 != node_idx) {
						if (idx0 - node_idx > 1)
							for (j = 0; j < (int)idx0 - node_idx - 1; j++)
								network->edges.past_edge_row_start[idx0-j-1] = -1;
						network->edges.past_edge_row_start[idx0] = i;
						node_idx = idx0;
					}
				}

				for (i = idx0 + 1; i < network->network_properties.N_tar; i++)
					network->edges.past_edge_row_start[i] = -1;
				printf("\t\tPast Edges Parsed.\n");
				fflush(stdout);

				free(edges);
				edges = NULL;
				ca->hostMemUsed -= sizeof(uint64_t) * N_edg;
			}

			/*if (!printDegrees(network->nodes, network->network_properties.N_tar, "in-degrees_FILE.cset.dbg.dat", "out-degrees_FILE.cset.dbg.dat")) return false;
			if (!printEdgeLists(network->edges, N_edg, "past-edges_FILE.cset.dbg.dat", "future-edges_FILE.cset.dbg.dat")) return false;
			if (!printEdgeListPointers(network->edges, network->network_properties.N_tar, "past-edge-pointers_FILE.cset.dbg.dat", "future-edge-pointers_FILE.cset.dbg.dat")) return false;
			printf_red();
			printf("Check files now.\n");
			printf_std();
			fflush(stdout);
			exit(0);*/

			//Identify Resulting Properties
			for (i = 0; i < network->network_properties.N_tar; i++) {
				if (network->nodes.k_in[i] + network->nodes.k_out[i] > 0) {
					network->network_observables.N_res++;
					if (network->nodes.k_in[i] + network->nodes.k_out[i] > 1)
						network->network_observables.N_deg2++;
				}
			}
			network->network_observables.k_res = static_cast<float>(N_edg << 1) / network->network_observables.N_res;
			printf("\t\tProperties Identified.\n");
			printf_cyan();
			printf("\t\t\tResulting Network Size:   %d\n", network->network_observables.N_res);
			printf("\t\t\tResulting Average Degree: %f\n", network->network_observables.k_res);
			printf("\t\t\t    Incl. Isolated Nodes: %f\n", (network->network_observables.k_res * network->network_observables.N_res) / network->network_properties.N_tar);
			printf_std();
			fflush(stdout);

			dataStream.close();
		} else
			throw CausetException("Failed to open edge list file!\n");
		#ifdef MPI_ENABLED
		}
		#endif

		LoadPoint2:
		if (checkMpiErrors(network->network_properties.cmpi)) {
			if (!rank) {
				if (!strcmp(message.c_str(), "bad alloc"))
					throw std::bad_alloc();
				else
					throw CausetException(message.c_str());
			} else
				return false;
		}
		
		if ((get_manifold(spacetime) & (DE_SITTER | DUST | FLRW) && network->network_properties.flags.relink) || !network->network_properties.flags.link)
			return true;

		printf_mpi(rank, "\t\tCompleted.\n");
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	} catch (std::out_of_range) {
		fprintf(stderr, "Error using unordered map when reading edge list!\n");
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	if (network->network_properties.flags.verbose) {
		printf_mpi(rank, "\tGraph Properties:\n");
		if (!rank) printf_cyan();
		printf_mpi(rank, "\t\tResulting Network Size: %d\n", network->network_observables.N_res);
		printf_mpi(rank, "\t\tExpected Average Degrees: %f\n", network->network_observables.k_res);
		if (!rank) printf_std();
	}

	printf_mpi(rank, "Task Completed.\n");
	fflush(stdout);

	return true;
}

//Print to File
bool printNetwork(Network &network, CausetPerformance &cp, const int &gpuID)
{
	if (!network.network_properties.flags.print_network)
		return false;

	printf("Printing Results to File...\n");
	fflush(stdout);

	int i, j, k;

	try {
		std::ofstream outputStream;
		std::stringstream sstm;

		//Generate Filename
		if (network.network_properties.flags.use_gpu)
			sstm << "Dev" << gpuID << "_";
		else
			sstm << "CPU_";
		sstm << "ST-" << network.network_properties.spacetime << "_";
		sstm << "VER-" << VERSION << "_";
		sstm << network.network_properties.seed;
		std::string filename = sstm.str();

		//Write Simulation Parameters and Main Results to File
		sstm.str("");
		sstm.clear();
		sstm << "./dat/" << filename << ".cset.out";
		outputStream.open(sstm.str().c_str());
		if (!outputStream.is_open())
			throw CausetException("Failed to open graph file!\n");
		outputStream << "Causet Simulation\n";
		if (network.network_properties.graphID == 0)
			network.network_properties.graphID = static_cast<int>(time(NULL));
		outputStream << "Graph ID: " << network.network_properties.graphID << std::endl;

		time_t rawtime;
		struct tm * timeinfo;
		static char buffer[80];
		time(&rawtime);
		if (rawtime == (time_t)-1)
			throw CausetException("Function 'time' failed to execute!\n");
		timeinfo = localtime(&rawtime);
		size_t s = strftime(buffer, 80, "%X %x", timeinfo);
		if (s == 0)
			throw CausetException("Function 'strftime' failed to execute!\n");
		outputStream << buffer << std::endl;

		unsigned int spacetime = network.network_properties.spacetime;
		bool links_exist = network.network_properties.flags.link || network.network_properties.flags.relink;

		if (get_manifold(spacetime) & (DUST | FLRW)) {
			outputStream << "\nCauset Initial Parameters:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			outputStream << "Number of Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Expected Degrees (k_tar)\t\t" << network.network_properties.k_tar << std::endl;
			if (get_manifold(spacetime) & FLRW)
				outputStream << "Dark Energy Density (omegaL)\t\t" << network.network_properties.omegaL << std::endl;
			outputStream << "Rescaled Age (tau0)\t\t\t" << network.network_properties.tau0 << std::endl;
			outputStream << "Spatial Scaling (alpha)\t\t\t" << network.network_properties.alpha << std::endl;
			outputStream << "Temporal Scaling (a)\t\t\t" << network.network_properties.a << std::endl;
			outputStream << "Node Density (delta)\t\t\t" << network.network_properties.delta << std::endl;

			outputStream << "\nCauset Resulting Parameters:" << std::endl;
			outputStream << "----------------------------" << std::endl;
			if (links_exist) {
				outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_observables.N_res << std::endl;
				outputStream << "Resulting Average Degrees (k_res)\t" << network.network_observables.k_res << std::endl;
				outputStream << "Resulting Error in <k>\t\t\t" << (fabs(network.network_properties.k_tar - network.network_observables.k_res) / network.network_properties.k_tar) << std::endl;
			}
			outputStream << "Minimum Rescaled Time\t\t\t" << network.nodes.id.tau[0] << std::endl;
		} else if (get_manifold(spacetime) & DE_SITTER) {
			outputStream << "\nCauset Input Parameters:" << std::endl;
			outputStream << "------------------------" << std::endl;
			outputStream << "Number of Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Expected Degrees (k_tar)\t\t" << network.network_properties.k_tar << std::endl;
			outputStream << "Max. Conformal Time (eta_0)\t\t" << HALF_PI - network.network_properties.zeta << std::endl;
			outputStream << "Pseudoradius (a)\t\t\t" << network.network_properties.a << std::endl;

			outputStream << "\nCauset Calculated Values:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			if (links_exist) {
				outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_observables.N_res << std::endl;
				outputStream << "Resulting Average Degrees (k_res)\t" << network.network_observables.k_res << std::endl;
				outputStream << "    Incl. Isolated Nodes\t\t" << (network.network_observables.k_res * network.network_observables.N_res) / network.network_properties.N_tar << std::endl;
				outputStream << "Resulting Error in <k>\t\t\t" << (fabs(network.network_properties.k_tar - network.network_observables.k_res) / network.network_properties.k_tar) << std::endl;
			}
			outputStream << "Minimum Rescaled Time\t\t\t" << network.nodes.id.tau[0] << std::endl;
		}

		if (network.network_properties.flags.calc_clustering)
			outputStream << "Average Clustering\t\t\t" << network.network_observables.average_clustering << std::endl;

		if (network.network_properties.flags.calc_components) {
			outputStream << "Number of Connected Components\t\t" << network.network_observables.N_cc << std::endl;
			outputStream << "Size of Giant Connected Component\t" << network.network_observables.N_gcc << std::endl;
		}

		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "Success Ratio\t\t\t\t" << network.network_observables.success_ratio << std::endl;

		if (network.network_properties.flags.calc_deg_field) {
			outputStream << "Degree Field Measurement Time\t\t" << network.network_properties.tau_m << std::endl;
			outputStream << "Average In-Degree Field Value\t\t" << network.network_observables.avg_idf << std::endl;
			outputStream << "Average Out-Degree Field Value\t\t" << network.network_observables.avg_odf << std::endl;
		}

		if (network.network_properties.flags.calc_action)
			outputStream << "Action\t\t\t\t\t" << network.network_observables.action << std::endl;


		outputStream << "\nNetwork Analysis Results:" << std::endl;
		outputStream << "-------------------------" << std::endl;
		outputStream << "Node Position Data:\t\t\t" << "pos/" << network.network_properties.graphID << ".cset.pos.dat" << std::endl;
		if (links_exist) {
			outputStream << "Node Edge Data:\t\t\t\t" << "edg/" << network.network_properties.graphID << ".cset.edg.dat" << std::endl;
			outputStream << "Degree Distribution Data:\t\t" << "dst/" << network.network_properties.graphID << ".cset.dst.dat" << std::endl;
			outputStream << "In-Degree Distribution Data:\t\t" << "idd/" << network.network_properties.graphID << ".cset.idd.dat" << std::endl;
			outputStream << "Out-Degree Distribution Data:\t\t" << "odd/" << network.network_properties.graphID << ".cset.odd.dat" << std::endl;
		}

		if (network.network_properties.flags.calc_clustering) {
			outputStream << "Clustering Coefficient Data:\t\t" << "cls/" << network.network_properties.graphID << ".cset.cls.dat" << std::endl;
			outputStream << "Clustering by Degree Data:\t\t" << "cdk/" << network.network_properties.graphID << ".cset.cdk.dat" << std::endl;
		}

		if (network.network_properties.flags.validate_embedding)
			outputStream << "Embedding Confusion Matrix Data:\t" << "emb/" << network.network_properties.graphID << ".cset.emb.dat" << std::endl;

		if (network.network_properties.flags.calc_deg_field) {
			outputStream << "In-Degree Field Data:\t\t" << "idf/" << network.network_properties.graphID << ".cset.idf.dat" << std::endl;
			outputStream << "Out-Degree Field Data: \t\t" << "odf/" << network.network_properties.graphID << ".cset.odf.dat" << std::endl;
		}

		if (network.network_properties.flags.calc_action)
			outputStream << "Action/Cardinality Data:\t\t" << "act/" << network.network_properties.graphID << ".cset.act.dat" << std::endl;

		outputStream << "\nAlgorithmic Performance:" << std::endl;
		outputStream << "--------------------------" << std::endl;
		if (get_manifold(spacetime) & (DUST | FLRW))
			outputStream << "calcDegrees:\t\t\t\t" << cp.sCalcDegrees.elapsedTime << " sec" << std::endl;
		outputStream << "createNetwork:\t\t\t\t" << cp.sCreateNetwork.elapsedTime << " sec" << std::endl;
		outputStream << "generateNodes:\t\t\t\t" << cp.sGenerateNodes.elapsedTime << " sec" << std::endl;
		outputStream << "quicksort:\t\t\t\t" << cp.sQuicksort.elapsedTime << " sec" << std::endl;
		if (links_exist) {
			if (network.network_properties.flags.use_gpu)
				outputStream << "linkNodesGPU:\t\t\t\t" << cp.sLinkNodesGPU.elapsedTime << " sec" << std::endl;
			else
				outputStream << "linkNodes:\t\t\t\t" << cp.sLinkNodes.elapsedTime << " sec" << std::endl;
		}

		if (network.network_properties.flags.calc_clustering)
			outputStream << "measureClustering:\t\t\t" << cp.sMeasureClustering.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_components)
			outputStream << "measureComponents:\t\t\t" << cp.sMeasureConnectedComponents.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.validate_embedding)
			outputStream << "validateEmbedding:\t\t\t" << cp.sValidateEmbedding.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "measureSuccessRatio:\t\t\t" << cp.sMeasureSuccessRatio.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_deg_field)
			outputStream << "measureDegreeField:\t\t" << cp.sMeasureDegreeField.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_action)
			outputStream << "measureAction:\t\t\t\t" << cp.sMeasureAction.elapsedTime << " sec" << std::endl;

		outputStream.flush();
		outputStream.close();

		//Add Data Key
		std::ofstream mapStream;
		mapStream.open("./etc/data_keys.cset.key", std::ios::app);
		if (!mapStream.is_open())
			throw CausetException("Failed to open 'etc/data_keys.cset.key' file!\n");
		mapStream << network.network_properties.graphID << "\t" << filename << std::endl;
		mapStream.close();

		std::ofstream dataStream;

		//Write Data to File
		sstm.str("");
		sstm.clear();
		sstm << "./dat/pos/" << network.network_properties.graphID << ".cset.pos.dat";
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open node position file!\n");
		dataStream << std::fixed << std::setprecision(9);
		for (i = 0; i < network.network_properties.N_tar; i++) {
			if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
				if (get_stdim(spacetime) == 4) {
					if (get_manifold(spacetime) & (DUST | FLRW))
						dataStream << network.nodes.id.tau[i];
					else if (get_manifold(spacetime) & DE_SITTER)
						dataStream << network.nodes.crd->w(i);
					dataStream << " " << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i) << " " << network.nodes.crd->z(i);
				} else if (get_stdim(spacetime) == 2)
					dataStream << network.nodes.crd->x(i);
			} else if (get_manifold(spacetime) & HYPERBOLIC)
				dataStream << network.nodes.id.AS[i] << " " << network.nodes.crd->y(i) << " " << network.nodes.crd->x(i);
			dataStream << "\n";
		}
		dataStream.flush();
		dataStream.close();

		int idx = 0;
		if (links_exist) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/edg/" << network.network_properties.graphID << ".cset.edg.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open edge list file!\n");
			for (i = 0; i < network.network_properties.N_tar; i++) {
				for (j = 0; j < network.nodes.k_out[i]; j++)
					dataStream << i << " " << network.edges.future_edges[idx + j] << "\n";
				idx += network.nodes.k_out[i];
			}
			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << "./dat/dst/" << network.network_properties.graphID << ".cset.dst.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open degree distribution file!\n");

			int k_max = network.network_observables.N_res - 1;
			int *deg_dist = (int*)malloc(sizeof(int) * (k_max+1));
			assert (deg_dist != NULL);
			memset(deg_dist, 0, sizeof(int) * (k_max+1));

			for (i = 0; i < network.network_properties.N_tar; i++)
				deg_dist[network.nodes.k_in[i] + network.nodes.k_out[i]]++;

			for (k = 1; k <= k_max; k++)
				if (!!deg_dist[k])
					dataStream << k << " " << deg_dist[k] << "\n";

			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << "./dat/idd/" << network.network_properties.graphID << ".cset.idd.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open in-degree distribution file!\n");

			memset(deg_dist, 0, sizeof(int) * (k_max+1));
			for (i = 0; i < network.network_properties.N_tar; i++)
				deg_dist[network.nodes.k_in[i]]++;

			for (k = 1; k <= k_max; k++)
				if (!!deg_dist[k])
					dataStream << k << " " << deg_dist[k] << "\n";

			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << "./dat/odd/" << network.network_properties.graphID << ".cset.odd.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open out-degree distribution file!\n");

			memset(deg_dist, 0, sizeof(int) * (k_max+1));
			for (i = 0; i < network.network_properties.N_tar; i++)
				deg_dist[network.nodes.k_out[i]]++;

			for (k = 1; k <= k_max; k++)
				if (!!deg_dist[k])
					dataStream << k << " " << deg_dist[k] << "\n";

			dataStream.flush();
			dataStream.close();

			free(deg_dist);
			deg_dist = NULL;
		}

		if (network.network_properties.flags.calc_clustering) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/cls/" << network.network_properties.graphID << ".cset.cls.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open clustering coefficient file!\n");
			for (i = 0; i < network.network_properties.N_tar; i++)
				dataStream << network.network_observables.clustering[i] << "\n";
			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << "./dat/cdk/" << network.network_properties.graphID << ".cset.cdk.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open clustering distribution file!\n");
			double cdk;
			int ndk;
			for (i = 0; i < network.network_properties.N_tar; i++) {
				cdk = 0.0;
				ndk = 0;
				for (j = 0; j < network.network_properties.N_tar; j++) {
					if (i == (network.nodes.k_in[j] + network.nodes.k_out[j])) {
						cdk += network.network_observables.clustering[j];
						ndk++;
					}
				}
				if (ndk == 0)
					ndk++;
				dataStream << i << " " << (cdk / ndk) << "\n";
			}
			dataStream.flush();
			dataStream.close();
		}

		if (network.network_properties.flags.validate_embedding) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/emb/" << network.network_properties.graphID << ".cset.emb.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open embedding confusion matrix file!\n");
			dataStream << "True Positives:  " << static_cast<double>(network.network_observables.evd.confusion[0]) / network.network_observables.evd.A1S << std::endl;
			dataStream << "True Negatives:  " << static_cast<double>(network.network_observables.evd.confusion[1]) / network.network_observables.evd.A1T << std::endl;
			dataStream << "False Positives: " << static_cast<double>(network.network_observables.evd.confusion[2]) / network.network_observables.evd.A1T << std::endl;
			dataStream << "False Negatives: " << static_cast<double>(network.network_observables.evd.confusion[3]) / network.network_observables.evd.A1S << std::endl;

			dataStream.flush();
			dataStream.close();
		}

		if (network.network_properties.flags.calc_deg_field) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/idf/" << network.network_properties.graphID << ".cset.idf.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open in-degree field file!\n");
			for (i = 0; i < network.network_properties.N_df; i++)
				dataStream << network.network_observables.in_degree_field[i] << "\n";

			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << "./dat/odf/" << network.network_properties.graphID << ".cset.odf.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open out-degree field file!\n");
			for (i = 0; i < network.network_properties.N_df; i++)
				dataStream << network.network_observables.out_degree_field[i] << "\n";

			dataStream.flush();
			dataStream.close();
		}

		if (network.network_properties.flags.calc_action) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/act/" << network.network_properties.graphID << ".cset.act.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open action file!\n");
			for (i = 0; i < network.network_properties.max_cardinality; i++)
				dataStream << network.network_observables.cardinalities[i] << "\n";

			dataStream.flush();
			dataStream.close();
		}

		printf("\tFilename: %s.cset.out\n", filename.c_str());
		fflush(stdout);
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}
	
	printf("Task Completed.\n\n");
	fflush(stdout);

	return true;
}

//Print Benchmarking Data
bool printBenchmark(const Benchmark &bm, const CausetFlags &cf, const bool &link, const bool &relink)
{
	//Print to File
	FILE *f;
	bool links_exist = link || relink;

	try {
		f = fopen("bench.log", "w");
		if (f == NULL)
			throw CausetException("Failed to open file 'bench.log'\n");
		fprintf(f, "Causet Simulation Benchmark Results\n");
		fprintf(f, "-----------------------------------\n");
		fprintf(f, "Times Averaged over %d Runs:\n", NBENCH);
		fprintf(f, "\tcalcDegrees:\t\t%5.6f sec\n", bm.bCalcDegrees);
		fprintf(f, "\tcreateNetwork:\t\t%5.6f sec\n", bm.bCreateNetwork);
		fprintf(f, "\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
		fprintf(f, "\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
		if (links_exist) {
			if (cf.use_gpu)
				fprintf(f, "\tlinkNodesGPU:\t\t%5.6f sec\n", bm.bLinkNodesGPU);
			else
				fprintf(f, "\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
		}
		if (cf.calc_clustering)
			fprintf(f, "\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);
		if (cf.calc_components)
			fprintf(f, "\tmeasureComponents:\t%5.6f sec\n", bm.bMeasureConnectedComponents);
		if (cf.calc_success_ratio)
			fprintf(f, "\tmeasureSuccessRatio:\t%5.6f sec\n", bm.bMeasureSuccessRatio);
		if (cf.calc_deg_field)
			fprintf(f, "\tmeasureDegreeField:\t%5.6f sec\n", bm.bMeasureDegreeField);
		if (cf.calc_action)
			fprintf(f, "\tmeasureAction:\t%5.6f sec\n", bm.bMeasureAction);

		fclose(f);
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	//Print to Terminal
	printf("\nCauset Simulation Benchmark Results\n");
	printf("-----------------------------------\n");
	printf("Time Averaged over %d Runs:\n", NBENCH);
	printf("\tcalcDegrees:\t\t%5.6f sec\n", bm.bCalcDegrees);
	printf("\tcreateNetwork:\t\t%5.6f sec\n", bm.bCreateNetwork);
	printf("\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
	printf("\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
	if (links_exist) {
		if (cf.use_gpu)
			printf("\tlinkNodesGPU:\t\t%5.6f sec\n", bm.bLinkNodesGPU);
		else
			printf("\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
	}
	if (cf.calc_clustering)
		printf("\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);
	if (cf.calc_components)
		printf("\tmeasureConnectedComponents:\t%5.6f sec\n", bm.bMeasureConnectedComponents);
	if (cf.calc_success_ratio)
		printf("\tmeasureSuccessRatio:\t%5.6f sec\n", bm.bMeasureSuccessRatio);
	if (cf.calc_deg_field)
		printf("\tmeasureDegreeField:\t%5.6f sec\n", bm.bMeasureDegreeField);
	if (cf.calc_action)
		printf("\tmeasureAction:\t\t%5.6f sec\n", bm.bMeasureAction);
	printf("\n");
	fflush(stdout);

	return true;
}

//Free Memory
void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed)
{
	unsigned int spacetime = network->network_properties.spacetime;
	int rank = network->network_properties.cmpi.rank;
	bool links_exist = network->network_properties.flags.link || network->network_properties.flags.relink;

	if (!network->network_properties.flags.no_pos) {
		if (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW)) {
			free(network->nodes.id.tau);
			network->nodes.id.tau = NULL;
			hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
		} else if (get_manifold(spacetime) & HYPERBOLIC) {
			free(network->nodes.id.AS);
			network->nodes.id.AS = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_tar;
		}

		if (get_stdim(spacetime) == 4) {
			#if EMBED_NODES
			free(network->nodes.crd->v());
			network->nodes.crd->v() = NULL;
			hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
			#endif

			free(network->nodes.crd->w());
			network->nodes.crd->w() = NULL;

			free(network->nodes.crd->x());
			network->nodes.crd->x() = NULL;

			free(network->nodes.crd->y());
			network->nodes.crd->y() = NULL;

			free(network->nodes.crd->z());
			network->nodes.crd->z() = NULL;

			hostMemUsed -= sizeof(float) * network->network_properties.N_tar * 4;
		} else if (get_stdim(spacetime) == 2) {
			free(network->nodes.crd->x());
			network->nodes.crd->x() = NULL;

			free(network->nodes.crd->y());
			network->nodes.crd->y() = NULL;

			#if EMBED_NODES
			free(network->nodes.crd->z());
			network->nodes.crd->z() = NULL;
			hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
			#endif

			hostMemUsed -= sizeof(float) * network->network_properties.N_tar * 2;
		}
	}

	if (links_exist) {
		free(network->nodes.k_in);
		network->nodes.k_in = NULL;
		hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

		free(network->nodes.k_out);
		network->nodes.k_out = NULL;
		hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

		if (!network->network_properties.flags.use_bit) {
			free(network->edges.past_edges);
			network->edges.past_edges = NULL;
			hostMemUsed -= sizeof(int) * static_cast<uint64_t>(network->network_properties.N_tar * network->network_properties.k_tar * (1.0 + network->network_properties.edge_buffer) / 2);

			free(network->edges.future_edges);
			network->edges.future_edges = NULL;
			hostMemUsed -= sizeof(int) * static_cast<uint64_t>(network->network_properties.N_tar * network->network_properties.k_tar * (1.0 + network->network_properties.edge_buffer) / 2);

			free(network->edges.past_edge_row_start);
			network->edges.past_edge_row_start = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

			free(network->edges.future_edge_row_start);
			network->edges.future_edge_row_start = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_tar;
		}

		network->adj.clear();
		network->adj.swap(network->adj);
		hostMemUsed -= sizeof(bool) * static_cast<uint64_t>(POW2(network->network_properties.core_edge_fraction * network->network_properties.N_tar, EXACT)) / 8;
	}

	if (network->network_properties.flags.bench)
		return;

	if (!rank) {
		if (network->network_properties.flags.calc_clustering) {
			free(network->network_observables.clustering);
			network->network_observables.clustering = NULL;
			hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
		}
	}

	if (network->network_properties.flags.calc_components) {
		free(network->nodes.cc_id);
		network->nodes.cc_id = NULL;
		hostMemUsed -= sizeof(int) * network->network_properties.N_tar;
	}

	if (network->network_properties.flags.validate_embedding) {
		free(network->network_observables.evd.confusion);
		network->network_observables.evd.confusion = NULL;
		hostMemUsed -= sizeof(uint64_t) * 4;
	}

	if (!rank) {
		if (network->network_properties.flags.calc_deg_field) {
			free(network->network_observables.in_degree_field);
			network->network_observables.in_degree_field = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_df;

			free(network->network_observables.out_degree_field);
			network->network_observables.out_degree_field = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_df;
		}

		if (network->network_properties.flags.validate_distances) {
			free(network->network_observables.dvd.confusion);
			network->network_observables.dvd.confusion = NULL;
			hostMemUsed -= sizeof(uint64_t) * 2;
		}

		if (network->network_properties.flags.calc_action) {
			free(network->network_observables.cardinalities);
			network->network_observables.cardinalities = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.max_cardinality * omp_get_max_threads();
		}
	}
}
