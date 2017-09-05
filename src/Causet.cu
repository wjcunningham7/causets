/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#include "NetworkCreator.h"
#include "Measurements.h"

int main(int argc, char **argv)
{
	//Initialize Data Structures
	CausetMPI cmpi = CausetMPI();
	CaResources ca = CaResources();
	CuResources cu = CuResources();
	Benchmark bm = Benchmark();

	//MPI Initialization
	#ifdef MPI_ENABLED
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &cmpi.num_mpi_threads);
	MPI_Comm_rank(MPI_COMM_WORLD, &cmpi.rank);
	#endif

	CausetPerformance cp = CausetPerformance();
	stopwatchStart(&cp.sCauset);

	//Initialize 'Network' structure
	Network network = Network(parseArgs(argc, argv, &cmpi));

	bool _bench = network.network_properties.flags.bench;
	if (_bench)
		network.network_properties.flags.bench = false;

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

	if (_bench)
		network.network_properties.flags.bench = true;

	//Measure Graph Properties
	if (network.network_properties.flags.test) goto CausetPoint2;
	if (!measureNetworkObservables(&network, &ca, &cp, &bm, cu.cuContext)) goto CausetExit;

	//Print Results
	if (!network.network_properties.flags.bench) printMemUsed(NULL, ca.maxHostMemUsed, ca.maxDevMemUsed, cmpi.rank);
	#ifdef MPI_ENABLED
	if (!cmpi.rank)
	#endif
	{
	if (network.network_properties.flags.bench && !printBenchmark(bm, network.network_properties.flags, network.network_properties.flags.link, network.network_properties.flags.relink)) {
		cmpi.fail = 1;
		goto CausetPoint1;
	}
	if (network.network_properties.flags.print_network && !printNetwork(network, cp, cu.gpuID)) {
		cmpi.fail = 1;
		goto CausetPoint1;
	}
	}

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

	std::string stdim;
	std::string manifold;
	std::string region;
	std::string curvature;
	std::string symmetry;

	int c, longIndex;
	//Single-character options
	static const char *optString = ":A:a:b:Cc:k:d:e:g:H:hm:n:r:s:S:vyz:";
	//Multi-character options
	static const struct option longOpts[] = {
		{ "action",		required_argument,	NULL, 'A' },
		{ "action-theory",	required_argument,	NULL,  0  },
		{ "age",		required_argument,	NULL, 'a' },
		{ "alpha",		required_argument,	NULL,  0  },
		{ "benchmark",		no_argument,		NULL,  0  },
		{ "beta",		required_argument,	NULL,  0  },
		{ "buffer",		required_argument,	NULL, 'b' },
		{ "chain",		no_argument,		NULL,  0  },
		{ "clustering",		no_argument,		NULL, 'C' },
		{ "components", 	no_argument,		NULL,  0  },
		{ "core",		required_argument,	NULL, 'c' },
		{ "curvature",		required_argument,	NULL,  0  },
		{ "datdir",		required_argument,	NULL,  0  },
		{ "delta",		required_argument,	NULL, 'd' },
		{ "energy",		required_argument,	NULL, 'e' },
		{ "geo-discon",		required_argument,	NULL,  0  },
		{ "gpu", 		no_argument, 		NULL,  0  },
		{ "graph",		required_argument,	NULL, 'g' },
		{ "growing",		no_argument,		NULL,  0  },
		{ "help", 		no_argument,		NULL, 'h' },
		{ "hubs",		required_argument,	NULL, 'H' },
		{ "quiet-read",		no_argument,		NULL,  0  },
		{ "link",		no_argument,		NULL,  0  },
		{ "link-epso",		no_argument,		NULL,  0  },
		{ "manifold",		required_argument,	NULL, 'm' },
		{ "mpi-split",		no_argument,		NULL,  0  },
		{ "mu",			required_argument,	NULL,  0  },
		{ "print", 		no_argument, 		NULL,  0  },
		{ "print-edges",	no_argument,		NULL,  0  },
		{ "print-dot",		no_argument,		NULL,  0  },
		{ "nodes", 		required_argument,	NULL, 'n' },
		{ "nopos",		no_argument,		NULL,  0  },
		{ "radius",		required_argument,	NULL,  0  },
		{ "read-old-format",	no_argument,		NULL,  0  },
		{ "region",		required_argument,	NULL, 'r' },
		{ "relink",		no_argument,		NULL,  0  },
		{ "seed",		required_argument,	NULL, 's' },
		{ "spacetime",		required_argument,	NULL,  0  },
		{ "stdim",		required_argument,	NULL,  0  },
		{ "stretch",		no_argument,		NULL,  0  },
		{ "strict-routing",	no_argument,		NULL,  0  },
		{ "success",		required_argument,	NULL, 'S' },
		{ "symmetry",		required_argument,	NULL,  0  },
		{ "test",		no_argument,		NULL,  0  },
		{ "verbose", 		no_argument, 		NULL,  0  },
		{ "version",		no_argument,		NULL, 'v' },
		{ "zeta",		required_argument,	NULL, 'z' },
		{ NULL,			0,			0,     0  }
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
			case 'a':	//Temporal cutuff (age)
				network_properties.tau0 = atof(optarg);
				if (network_properties.tau0 <= 0.0)
					throw CausetException("Invalid argument for 'age' parameter!\n");
				network_properties.omegaL = POW2(tanh(1.5 * network_properties.tau0), EXACT);
				network_properties.omegaM = 1.0 - network_properties.omegaL;
				break;
			case 'b':	//Edge buffer in adjacency lists
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
			case 'g':	//Graph ID
				network_properties.graphID = atoi(optarg);
				if (network_properties.graphID < 0)
					throw CausetException("Invalid argument for 'Graph ID' parameter!\n");
				break;
			case 'H':	//Calculate hub connectivity
				network_properties.flags.calc_hubs = true;
				network_properties.N_hubs = atoi(optarg);
				if (network_properties.N_hubs <= 0)
					throw CausetException("Invalid argument for 'hubs' parameter!\n");
				break;
			//case 'h' is located at the end
			case 'm':	//Manifold
				if (std::find(Spacetime::manifolds, Spacetime::manifolds + Spacetime::nmanifolds, std::string(optarg)) == Spacetime::manifolds + Spacetime::nmanifolds)
					throw CausetException("Invalid argument for 'manifold' parameter!\n");
				manifold.assign(optarg);
				break;
			case 'n':	//Number of nodes
				network_properties.N_tar = atoi(optarg);
				if (network_properties.N_tar < 0)
					throw CausetException("Invalid argument for 'nodes' parameter!\n");
				break;
			case 'r':	//Region
				if (std::find(Spacetime::regions, Spacetime::regions + Spacetime::nregions, std::string(optarg)) == Spacetime::regions + Spacetime::nregions)
					throw CausetException("Invalid argument for 'region' parameter!\n");
				region.assign(optarg);
				break;
			case 'S':	//Flag for calculating success ratio
				network_properties.flags.calc_success_ratio = true;
				network_properties.flags.calc_components = true;
				network_properties.N_sr = atof(optarg);
				if (network_properties.N_sr <= 0.0)
					throw CausetException("Invalid argument for 'success' parameter!\n");
				break;
			case 's':	//Random seed
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
				break;
			case 0:
				if (!strcmp("action-theory", longOpts[longIndex].name)) {
					//Calculate data for plot of <S>
					network_properties.flags.calc_action_theory = true;
					network_properties.N_actth = atoi(optarg);
					if (network_properties.N_actth <= 0.0 || network_properties.N_actth > 30)
						throw CausetException("Invalid argument for 'action-theory' parameter!\n");
				} else if (!strcmp("alpha", longOpts[longIndex].name)) {
					//Taken to be \tilde{\alpha} when input
					network_properties.alpha = atof(optarg);
					if (network_properties.alpha <= 0.0)
						throw CausetException("Invalid argument for 'alpha' parameter!\n");
				} else if (!strcmp("benchmark", longOpts[longIndex].name))
					//Flag to benchmark selected routines
					network_properties.flags.bench = true;
				else if (!strcmp("beta", longOpts[longIndex].name)) {
					//Inverse temperature
					if (!strcmp(optarg, "infinity"))
						network_properties.beta = -1;
					else
						network_properties.beta = atof(optarg);
					if (network_properties.beta <= 0.0 && network_properties.beta != -1)
						throw CausetException("Invalid argument for 'beta' parameter!\n");
				} else if (!strcmp("chain", longOpts[longIndex].name))
					//Calculate longest chain
					network_properties.flags.calc_chain = true;
				else if (!strcmp("components", longOpts[longIndex].name))
					//Flag for Finding Connected Components
					network_properties.flags.calc_components = true;
				else if (!strcmp("curvature", longOpts[longIndex].name)) {
					//Flag for setting spatial curvature
					if (std::find(Spacetime::curvatures, Spacetime::curvatures + Spacetime::ncurvatures, std::string(optarg)) == Spacetime::curvatures + Spacetime::ncurvatures)
						throw CausetException("Invalid argument for 'curvature' parameter!\n");
					curvature.assign(optarg);
				} else if (!strcmp("datdir", longOpts[longIndex].name))
					//Set I/O directory for data
					network_properties.datdir.assign(optarg);
				else if (!strcmp("geo-discon", longOpts[longIndex].name)) {
					network_properties.flags.calc_geo_dis = true;
					network_properties.N_gd = atof(optarg);
					if (network_properties.N_gd <= 0.0)
						throw CausetException("Invalid argument for 'geo-discon' parameter!\n");
				} else if (!strcmp("gpu", longOpts[longIndex].name)) {
					//Flag to use GPU accelerated routines
					network_properties.flags.use_gpu = true;
					#ifndef CUDA_ENABLED
					throw CausetException("Recompile with 'make gpu' to use the --gpu flag!\n");
					#endif
				} else if (!strcmp("growing", longOpts[longIndex].name))
					//Use growing H2 model
					network_properties.flags.growing = true;
				else if (!strcmp("link", longOpts[longIndex].name))
					//Flag for Reading Nodes (and not links) and Re-Linking
					network_properties.flags.link = true;
				else if (!strcmp("link-epso", longOpts[longIndex].name))
					//Use EPSO for linking threshold in H2 model
					network_properties.flags.link_epso = true;
				else if (!strcmp("mpi-split", longOpts[longIndex].name))
					//Split adjacency matrix
					network_properties.flags.mpi_split = true;
				else if (!strcmp("mu", longOpts[longIndex].name)) {
					//Chemical potential
					network_properties.mu = atof(optarg);
					if (network_properties.mu <= 0.0)
						throw CausetException("Invalid argument for 'mu' parameter!\n");
				} else if (!strcmp("nopos", longOpts[longIndex].name))
					//Flag to Skip Node Generation/Reading
					network_properties.flags.no_pos = true;
				else if (!strcmp("print", longOpts[longIndex].name))
					//Flag to print results to file in 'dat' folder
					network_properties.flags.print_network = true;
				else if (!strcmp("print-edges", longOpts[longIndex].name))
					//Flag to print links to file in 'dat/edg' folder
					network_properties.flags.print_edges = true;
				else if (!strcmp ("print-dot", longOpts[longIndex].name))
					//Flag to print links to dot file in 'dat/dot' folder
					network_properties.flags.print_dot = true;
				else if (!strcmp("quiet-read", longOpts[longIndex].name))
					//Flag to ignore warnings when reading graph
					network_properties.flags.quiet_read = true;
				else if (!strcmp("radius", longOpts[longIndex].name))
					//Radius of spatial dimensions
					network_properties.r_max = atof(optarg);
				else if (!strcmp("read-old-format", longOpts[longIndex].name))
					network_properties.flags.read_old_format = true;
				else if (!strcmp("relink", longOpts[longIndex].name))
					//Flag for Reading Nodes (and not links) and Re-Linking
					network_properties.flags.relink = true;
				else if (!strcmp("spacetime", longOpts[longIndex].name)) {
					//Spacetime ID
					printf("NOT SUPPORTED!\n");
					throw CausetException("Invalid parameter: spacetime.\n");
					//std::stringstream ss;
					//ss << std::hex << optarg;
					//ss >> network_properties.spacetime;
					//printf("spacetime: %d\n", network_properties.spacetime);
				} else if (!strcmp("stdim", longOpts[longIndex].name)) {
					//Spacetime dimensions
					if (std::find(Spacetime::stdims, Spacetime::stdims + Spacetime::nstdims, std::string(optarg)) == Spacetime::stdims + Spacetime::nstdims)
						throw CausetException("Invalid argument for 'stdim' parameter!\n");
					stdim.assign(optarg);
				} else if (!strcmp("stretch", longOpts[longIndex].name)) {
					//Stretch across greedy paths
					network_properties.flags.calc_stretch = true;
					if (!network_properties.flags.calc_success_ratio) {
						network_properties.flags.calc_components = true;
						network_properties.flags.calc_success_ratio = true;
						network_properties.N_sr = 0.01;
					}
				} else if (!strcmp("strict-routing", longOpts[longIndex].name))
					//Use strict routing protocol
					network_properties.flags.strict_routing = true;
				else if (!strcmp("symmetry", longOpts[longIndex].name)) {
					//Symmetric about temporal axis
					if (std::find(Spacetime::symmetries, Spacetime::symmetries + Spacetime::nsymmetries, std::string(optarg)) == Spacetime::symmetries + Spacetime::nsymmetries)
						throw CausetException("Invalid argument for 'symmetry' parameter!\n");
					symmetry.assign(optarg);
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
				printf_mpi(rank, "Copyright (C) 2013-2017 Will Cunningham\n");
				printf_mpi(rank, "Platform: Redhat Linux x86 64-bit Kernel\n");
				printf_mpi(rank, "Developed for use with NVIDIA K20m GPU\n");
				printf_mpi(rank, "Version %s\n", VERSION);
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
				printf_mpi(rank, "      --action-theory\tMeasure Th. Action to 2^x\t20\n");
				printf_mpi(rank, "  -a, --age\t\tRescaled (Conformal) Age of\n");
				printf_mpi(rank, "\t\t\tFLRW (de Sitter) Spacetime\t0.85\n");
				printf_mpi(rank, "      --alpha\t\tSpatial Scaling/Cutoff\t\t2.0\n");
				printf_mpi(rank, "      --benchmark\tBenchmark Algorithms\n");
				printf_mpi(rank, "      --beta\t\tInverse Temperature\t\t0.5, \"infinity\"\n");
				printf_mpi(rank, "  -b, --buffer\t\tEdge Buffer\t\t\t0.3\n");
				printf_mpi(rank, "      --chain\t\tMeasure Maximal Chain\n");
				printf_mpi(rank, "  -C, --clustering\tMeasure Clustering\n");
				printf_mpi(rank, "      --components\tMeasure Graph Components\n");
				printf_mpi(rank, "  -c, --core\t\tCore Edge Fraction\t\t0.01\n");
				printf_mpi(rank, "      --curvature\tSpatial Curvature\t\t\"Flat\", \"Positive\", \"Negative\"\n");
				printf_mpi(rank, "      --datdir\t\tData Directory\t\t\t\"./dat/\"\n");
				printf_mpi(rank, "  -d, --delta\t\tNode Density\t\t\t10000\n");
				printf_mpi(rank, "  -e, --energy\t\tDark Energy Density\t\t0.73\n");
				printf_mpi(rank, "      --geo-discon\tFrac. Geodesic Discon. Pairs\t10000\n");
				#ifdef CUDA_ENABLED
				printf_mpi(rank, "      --gpu\t\tUse GPU Acceleration\n");
				#endif
				printf_mpi(rank, "  -g, --graph\t\tGraph ID\t\t\tCheck dat/*.cset.out files\n");
				printf_mpi(rank, "      --growing\t\tUse Growing H2 Model\n");
				printf_mpi(rank, "  -h, --help\t\tDisplay This Menu\n");
				printf_mpi(rank, "  -H, --hubs\t\tCalculate Hub Density\t\t20\n");
				printf_mpi(rank, "      --link\t\tLink Nodes to Create Graph\n");
				printf_mpi(rank, "      --link-epso\tLink Nodes using EPSO Rule\n");
				printf_mpi(rank, "  -m, --manifold\tSpacetime Manifold\t\t\"Minkowski\", \"De_Sitter\",\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Anti_de_Sitter\", \"Dust\", \"FLRW\",\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Hyperbolic\"\n");
				printf_mpi(rank, "      --mu\t\tChemical Potential\t\t10.0\n");
				printf_mpi(rank, "  -n, --nodes\t\tNumber of Nodes\t\t\t1000, 10000, 100000\n");
				printf_mpi(rank, "      --nopos\t\tNo Node Positions\n");
				printf_mpi(rank, "      --print\t\tPrint Results\n");
				printf_mpi(rank, "      --print-edges\tPrint Edge List\n");
				printf_mpi(rank, "      --print-dot\tPrint Edges to Dot File\n");
				printf_mpi(rank, "      --quiet-read\tIgnore any warnings when\n");
				printf_mpi(rank, "\t\t\treading graph\n");
				printf_mpi(rank, "      --radius\t\tRadius of Spatial Dimension\t1.0\n");
				printf_mpi(rank, "      --read-old-format\tRead Positions in Old Format\n");
				printf_mpi(rank, "  -r, --region\t\tSpacetime Region\t\t\"Slab\", \"Slab_T1\", \"Slab_T2\", \"Slab_S1\",\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Slab_S2\", \"Slab_N1\", \"Slab_N2\", \"Slab_N3\"\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Half_Diamond\", \"Diamond\", \"Saucer_T\"\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Saucer_S\", \"Triangle_T\", \"Triangle_S\"\n");
				printf_mpi(rank, "\t\t\t\t\t\t\t\"Triangle_N\", \"Cube\"\n");
				printf_mpi(rank, "      --relink\t\tIgnore Pre-Existing Links\n");
				printf_mpi(rank, "  -s, --seed\t\tRandom Seed\t\t\t18100\n");
				printf_mpi(rank, "      --spacetime\tSpacetime ID\t\t\tSee VERSION\n");
				printf_mpi(rank, "      --stdim\t\tSpacetime Dimensions\t\t2, 3, 4\n");
				printf_mpi(rank, "      --stretch\t\tMeasure Stretch of Greedy Paths\n");
				printf_mpi(rank, "      --strict-routing\tUse Strict Routing Protocol\n");
				printf_mpi(rank, "  -S, --success\t\tCalculate Success Ratio\t\t0.01, 10000\n");
				printf_mpi(rank, "      --symmetry\tTemporal Symmetry\t\t\"None\", \"Temporal\"\n");
				printf_mpi(rank, "      --test\t\tTest Parameters Only\n");
				printf_mpi(rank, "      --verbose\t\tVerbose Output\n");
				printf_mpi(rank, "  -y\t\t\tSuppress User Queries\n");
				printf_mpi(rank, "  -z, --zeta\t\tHyperbolic Curvature\t\t1.0\n");
				printf_mpi(rank, "\n");

				printf_mpi(rank, "Report bugs to w.cunningham@neu.edu\n");
				printf_mpi(rank, "Bitbucket repository home page: <https://bitbucket.org/dk-lab/2015_code_causets>\n");
				#ifdef MPI_ENABLED
				MPI_Abort(MPI_COMM_WORLD, 0);
				#else
				exit(0);
				#endif
			case ':':
				//Single-character flag needs an argument
				if (!!optopt)
					fprintf(stderr, "%s : option '-%c' requires an argument.\n", argv[0], optopt);
				else
					fprintf(stderr, "%s : option '%s' requires an argument.\n", argv[0], argv[optind-1]);
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

		if (stdim.empty())
			throw CausetException("The spacetime dimension has not been defined!  Use flag '--stdim' to continue.\n");
		if (manifold.empty())
			throw CausetException("The manifold has not been defined!  Use flag '--manifold' to continue.\n");
		if (region.empty())
			throw CausetException("The region has not been defined!  Use flag '--region' to continue.\n");
		if (curvature.empty())
			throw CausetException("The curvature has not been defined!  Use flag '--curvature' to continue.\n");
		if (symmetry.empty())
			throw CausetException("The symmetry has not been defined!  Use flag '--symmetry' to continue.\n");
		network_properties.spacetime.set_spacetime(stdim.c_str(), manifold.c_str(), region.c_str(), curvature.c_str(), symmetry.c_str());
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
	}
	#ifdef MPI_ENABLED
	network_properties.seed ^= rank;
	#endif
	network_properties.mrng.rng.engine().seed(network_properties.seed);
	network_properties.mrng.rng.distribution().reset();

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
	int nb = static_cast<int>(network->network_properties.flags.bench) * (NBENCH - 1);
	int i;

	#ifdef _OPENMP
	printf_mpi(rank, "\n\t[ *** OPENMP MODULE ACTIVE *** ]\n");
	#endif

	#ifdef AVX2_ENABLED
	printf_mpi(rank, "\n\t[ ***  AVX2 MODULE ACTIVE  *** ]\n");
	#endif

	#ifdef MPI_ENABLED
	printf_mpi(rank, "\n\t[ ***   MPI MODULE ACTIVE  *** ]\n");
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
		if (!createNetwork(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.group_size, ca, cp->sCreateNetwork, network->network_properties.flags.use_gpu, network->network_properties.flags.decode_cpu, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.mpi_split, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;
		if (!!(i-nb))
			destroyNetwork(network, ca->hostMemUsed, ca->devMemUsed);
	}

	if (!!nb)
		bm->bCreateNetwork = cp->sCreateNetwork.elapsedTime / NBENCH;

	//Generate coordinates of spacetime nodes and then order nodes temporally using quicksort
	int low = 0;
	int high = network->network_properties.N_tar - 1;

	for (i = 0; i <= nb; i++) {
		if (!generateNodes(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.a, network->network_properties.eta0, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.cmpi, network->network_properties.mrng, cp->sGenerateNodes, network->network_properties.flags.growing, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
			network->network_properties.cmpi.fail = 1;
			goto InitExit;
		}

		//Quicksort
		stopwatchStart(&cp->sQuicksort);
		quicksort(network->nodes, network->network_properties.spacetime, low, high);
		stopwatchStop(&cp->sQuicksort);
	}

	#ifdef MPI_ENABLED
	if (!rank)
	#endif
	{
	if (!nb) {
		printf("\tQuick Sort Successfully Performed.\n");
		Spacetime st = network->network_properties.spacetime;
		if (st.symmetryIs("None") && !(st.manifoldIs("Hyperbolic") && st.curvatureIs("Flat"))) {
			printf_cyan();
			float min_time = 0.0;
			if (st.manifoldIs("Minkowski")) {
				if (st.stdimIs("2"))
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
	}

	//Identify edges as points connected by timelike intervals
	if (network->network_properties.flags.link) {
		for (i = 0; i <= nb; i++) {
			if (!linkNodes(network, ca, cp, ctx)) {
				network->network_properties.cmpi.fail = 1;
				goto InitExit;
			}
		}

		if (nb) {
			if (network->network_properties.flags.use_gpu)
				bm->bLinkNodesGPU = cp->sLinkNodesGPU.elapsedTime / NBENCH;
			else
				bm->bLinkNodes = cp->sLinkNodes.elapsedTime / NBENCH;
		}
	}

	InitExit:
	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	printf_mpi(rank, "Task Completed.\n");
	fflush(stdout);
	return true;
}

bool measureNetworkObservables(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	#endif

	if (!network->network_properties.flags.calc_clustering && !network->network_properties.flags.calc_components && !network->network_properties.flags.calc_success_ratio && !network->network_properties.flags.calc_action && !network->network_properties.flags.calc_action_theory && !network->network_properties.flags.calc_chain && !network->network_properties.flags.calc_hubs && !network->network_properties.flags.calc_geo_dis)
		return true;

	int rank = network->network_properties.cmpi.rank;
	bool links_exist = network->network_properties.flags.link || network->network_properties.flags.relink;
	int nb = static_cast<int>(network->network_properties.flags.bench) * (NBENCH - 1);
	int i;
		
	printf_mpi(rank, "\nCalculating Network Observables...\n");
	fflush(stdout);

	#ifdef MPI_ENABLED
	if (!rank)
	#endif
	{

	//------------//
	// CLUSTERING //
	//------------//

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

			if (!measureClustering(network->network_observables.clustering, network->nodes, network->edges, network->adj, network->network_observables.average_clustering, network->network_properties.N_tar, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, ca, cp->sMeasureClustering, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
		}

		if (nb)
			bm->bMeasureClustering = cp->sMeasureClustering.elapsedTime / NBENCH;
	}
	}

	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	//----------------------//
	// CONNECTED COMPONENTS //
	//----------------------//

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

	#ifdef MPI_ENABLED
	if (!rank)
	#endif
	{

	//--------------//
	// NAVIGABILITY //
	//--------------//

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

			if (!measureSuccessRatio(network->nodes, network->edges, network->adj, network->network_observables.success_ratio, network->network_observables.success_ratio2, network->network_observables.stretch, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_sr, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.mrng, ca, cp->sMeasureSuccessRatio, network->network_properties.flags.link_epso, network->network_properties.flags.use_bit, network->network_properties.flags.calc_stretch, network->network_properties.flags.strict_routing, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
		}

		if (nb)
			bm->bMeasureSuccessRatio = cp->sMeasureSuccessRatio.elapsedTime / NBENCH;
	}
	}

	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	//--------//
	// ACTION //
	//--------//

	if (network->network_properties.flags.calc_action) {
		for (i = 0; i <= nb; i++) {
			if (!links_exist) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate action if links do not exist!\n");
				printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_action = false;
				break;
			}

			//-------------------------//
			// BENINCASA-DOWKER ACTION //
			//-------------------------//

			#ifdef MPI_ENABLED
			if (network->network_properties.cmpi.num_mpi_threads > 1) {
				if (network->network_properties.N_tar >= ACTION_MPI_V5) {
					if (!measureAction_v5(network->network_observables.cardinalities, network->network_observables.action, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.cmpi, ca, cp->sMeasureAction, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
						network->network_properties.cmpi.fail = 1;
						break;
					}
				} else if (network->network_properties.flags.mpi_split || network->network_properties.N_tar >= ACTION_MPI_V4) {
					if (!measureAction_v4(network->network_observables.cardinalities, network->network_observables.action, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.cmpi, ca, cp->sMeasureAction, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
						network->network_properties.cmpi.fail = 1;
						break;
					}
				} else {
					if (!measureAction_v6(network->network_observables.cardinalities, network->network_observables.action, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.cmpi, network->network_properties.mrng, ca, cp->sMeasureAction, network->network_properties.flags.use_bit, network->network_properties.flags.mpi_split, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
						network->network_properties.cmpi.fail = 1;
						break;
					}
				}
			} else
			#endif
			{
			#if ACTION_V2
			if (network->network_properties.flags.use_bit && (network->network_properties.flags.link || network->network_properties.flags.relink)) {
				if (!measureAction_v3(network->network_observables.cardinalities, network->network_observables.action, network->adj, network->nodes.k_in, network->nodes.k_out, network->network_properties.spacetime, network->network_properties.N_tar, ca, cp->sMeasureAction, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
					network->network_properties.cmpi.fail = 1;
					break;
				}
			} else if (!measureAction_v2(network->network_observables.cardinalities, network->network_observables.action, network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.max_cardinality, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, ca, cp->sMeasureAction, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
			#else
			if (!measureAction_v1(network->network_observables.cardinalities, network->network_observables.action, network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.max_cardinality, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, ca, cp->sMeasureAction, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
			#endif
			}

			/*#ifdef MPI_ENABLED
			printf_mpi(rank, "This section has not been written to be MPI compatible.\n");
			printf_mpi(rank, "Find this warning on line %d in file %s.\n", __LINE__, __FILE__);
			return false;
			#endif*/
			
			//-----------------//
			// TIMELIKE ACTION //
			//-----------------//

			//DEBUG
			/*printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, "eta_dist.cset.dbg.dat", "x");
			printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, "x_dist.cset.dbg.dat", "y");

			if (!timelikeActionCandidates(network->network_observables.timelike_candidates, network->network_observables.chaintime, network->nodes, network->adj, network->nodes.k_in, network->nodes.k_out, network->network_properties.spacetime, network->network_properties.N_tar, ca, cp->sMeasureActionTimelike, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}

			//Create subnetwork based on timelike candidates
			Network subgraph = Network(*network);
			if (!configureSubgraph(&subgraph, network->nodes, network->network_observables.timelike_candidates, ca, ctx)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}

			if (!measureTimelikeAction(network, &subgraph, network->network_observables.timelike_candidates, ca)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}

			destroyNetwork(&subgraph, ca->hostMemUsed, ca->devMemUsed);*/

			if (network->network_properties.flags.bench) {
				free(network->network_observables.cardinalities);
				network->network_observables.cardinalities = NULL;
				ca->hostMemUsed -= sizeof(uint64_t) * network->network_properties.N_tar * omp_get_max_threads();
			}	
		}

		if (nb)
			bm->bMeasureAction = cp->sMeasureAction.elapsedTime / NBENCH;
	}

	if (checkMpiErrors(network->network_properties.cmpi))
		return false;

	#ifdef MPI_ENABLED
	if (!rank)
	#endif
	{

	/////////////////////
	// ACTION (THEORY) //
	/////////////////////

	if (network->network_properties.flags.calc_action_theory) {
		if (!links_exist) {
			if (!rank) printf_red();
			printf_mpi(rank, "\tCannot calculate theoretical action if links do not exist!\n");
			printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
			if (!rank) printf_std();
			fflush(stdout);

			network->network_properties.flags.calc_action_theory = false;
			goto MeasureExit;
		}

		if (!measureTheoreticalAction(network->network_observables.actth, network->network_properties.N_actth, network->nodes, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.eta0, network->network_properties.delta, ca, cp->sMeasureThAction, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
			network->network_properties.cmpi.fail = 1;
			goto MeasureExit;
		}
	}

	//---------------//
	// MAXIMAL CHAIN //
	//---------------//

	if (network->network_properties.flags.calc_chain) {
		for (i = 0; i <= nb; i++) {
			printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, "eta_dist.cset.dbg.dat", "x");
			printValues(network->nodes, network->network_properties.spacetime, network->network_properties.N_tar, "x_dist.cset.dbg.dat", "y");

			network->nodes.crd->y(0) = network->network_properties.r_max;
			network->nodes.crd->y(network->network_properties.N_tar - 1) = network->network_properties.r_max;

			//Create subgraph based on nodes with x > 0
			Network subgraph = Network(*network);
			std::vector<unsigned int> candidates;
			candidates.reserve(network->network_properties.N_tar >> 1);
			for (int j = 0; j < network->network_properties.N_tar; j++)
				if (network->nodes.crd->y(j) >= 0.0)
					candidates.push_back(j);
			if (!configureSubgraph(&subgraph, network->nodes, candidates, ca, ctx)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}

			printValues(subgraph.nodes, subgraph.network_properties.spacetime, subgraph.network_properties.N_tar, "eta_dist_sub.cset.dbg.dat", "x");
			printValues(subgraph.nodes, subgraph.network_properties.spacetime, subgraph.network_properties.N_tar, "x_dist_sub.cset.dbg.dat", "y");

			if (!measureChain(network->network_observables.chain_sym, network->network_observables.chain_asym, network->adj, subgraph.adj, network->network_properties.spacetime, network->network_properties.N_tar, subgraph.network_properties.N_tar, ca, cp->sMeasureChain, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}

			destroyNetwork(&subgraph, ca->hostMemUsed, ca->devMemUsed);
		}

		if (nb)
			bm->bMeasureChain = cp->sMeasureChain.elapsedTime / NBENCH;
	}

	//-------------//
	// HUB DENSITY //
	//-------------//

	if (network->network_properties.flags.calc_hubs) {
		for (i = 0; i <= nb; i++) {
			if (!links_exist) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate hub density if links do not exist!\n");
				printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_hubs = false;
				break;
			}
			
			if (!measureHubDensity(network->network_observables.hub_density, network->network_observables.hub_densities, network->adj, network->nodes.k_in, network->nodes.k_out, network->network_properties.N_tar, network->network_properties.N_hubs, ca, cp->sMeasureHubs, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				break;
			}
		}

		if (nb)
			bm->bMeasureHubs = cp->sMeasureHubs.elapsedTime / NBENCH;
	}

	//---------------------------------------------//
	// FRACTION OF GEODESICALLY DISCONNECTED PAIRS //
	//---------------------------------------------//

	if (network->network_properties.flags.calc_geo_dis) {
		for (i = 0; i <= nb; i++) {
			if (network->network_properties.flags.no_pos) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate geodesically disconnected pairs if positions do not exist!\n");
				printf_mpi(rank, "\tRemove flag '--nopos' to read node positions.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_geo_dis = false;
				goto MeasureExit;
			} else if (!links_exist) {
				if (!rank) printf_red();
				printf_mpi(rank, "\tCannot calculate geodesically disconnectd pairs if links do not exist!\n");
				printf_mpi(rank, "\tUse flag '--link' or '--relink' to generate links.\n\tSkipping this subroutine.\n\n");
				if (!rank) printf_std();
				fflush(stdout);

				network->network_properties.flags.calc_geo_dis = false;
				goto MeasureExit;
			}

			if (!measureGeoDis(network->network_observables.geo_discon, network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.N_gd, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.mrng, cp->sMeasureGeoDis, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench)) {
				network->network_properties.cmpi.fail = 1;
				goto MeasureExit;
			}
		}
		
		if (nb)
			bm->bMeasureGeoDis = cp->sMeasureGeoDis.elapsedTime / NBENCH;
	}
	}
	
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
	assert (bm != NULL);
	assert (network->network_properties.graphID);
	assert (!network->network_properties.flags.bench);
	#endif

	#if EMBED_NODES
	fprintf(stderr, "linkNodesGPU_v2 not implemented for EMBED_NODES=true.  Find me on line %d in %s.\n", __LINE__, __FILE__);
	return false;
	#endif

	Spacetime spacetime = network->network_properties.spacetime;
	int rank = network->network_properties.cmpi.rank;

	#ifdef MPI_ENABLED
	printf_mpi(rank, "\n\t[ ***   MPI MODULE ACTIVE  *** ]\n");
	#endif

	#ifdef _OPENMP
	printf_mpi(rank, "\n\t[ *** OPENMP MODULE ACTIVE *** ]\n");
	#endif

	#ifdef AVX2_ENABLED
	printf_mpi(rank, "\n\t[ ***  AVX2 MODULE ACTIVE  *** ]\n");
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
		if (USE_GSL && spacetime.manifoldIs("FLRW") && !network->network_properties.flags.no_pos)
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

		char *delimeters;
		int64_t N_edg;
		int64_t i;

		std::string prefix = network->network_properties.datdir;

		std::string message;
		char d[] = " \t";
		delimeters = &d[0];
		N_edg = 0;

		#ifndef MPI_ENABLED
		uint64_t *edges;
		uint64_t key;
		int node_idx;
		int j;
		unsigned int e0, e1;
		unsigned int idx0 = 0, idx1 = 0;
		unsigned int core_limit = static_cast<unsigned int>(network->network_properties.core_edge_fraction * network->network_properties.N_tar);
		#endif

		#ifdef MPI_ENABLED
		if (!rank)
		#endif
		{

		//Identify Basic Network Properties
		if (!network->network_properties.flags.no_pos) {
			network->network_properties.N_tar = 0;
			dataname << prefix << "pos/" << network->network_properties.graphID << ".cset.pos.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				while (getline(dataStream, line))
					network->network_properties.N_tar++;
				dataStream.close();
			} else {
				message = "Failed to open node position file!\n";
				printf_dbg("Filename: %s\n", dataname.str().c_str());
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint1;
			}
		}
	
		if (network->network_properties.flags.use_bit)
			network->network_properties.k_tar = 1.0;
		else if (network->network_properties.flags.link) {	
			dataname.str("");
			dataname.clear();
			dataname << prefix << "edg/" << network->network_properties.graphID << ".cset.edg.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				while (getline(dataStream, line))
					N_edg++;
				dataStream.close();
			} else {
				message = "Failed to open edge list file!\n";
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint1;
			}
		}
		}

		LoadPoint1:
		if (checkMpiErrors(network->network_properties.cmpi)) {
			if (!rank)
				throw CausetException(message.c_str());
			else
				return false;
		}

		#ifdef MPI_ENABLED
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&network->network_properties.N_tar, 1, MPI_INT, 0, MPI_COMM_WORLD);
		#endif

		if (!initVars(&network->network_properties, ca, cp, bm))
			return false;
		if (!!N_edg)
			network->network_properties.k_tar = static_cast<float>(static_cast<long double>(N_edg) * 2.0 / network->network_properties.N_tar);

		printf_mpi(rank, "\nFinished Gathering Peripheral Network Data.\n");
		fflush(stdout);

		if (network->network_properties.flags.test)
			return true;

		//Allocate Memory	
		if (!createNetwork(network->nodes, network->edges, network->adj, spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.group_size, ca, cp->sCreateNetwork, network->network_properties.flags.use_gpu, network->network_properties.flags.decode_cpu, network->network_properties.flags.link, network->network_properties.flags.relink, network->network_properties.flags.no_pos, network->network_properties.flags.use_bit, network->network_properties.flags.mpi_split, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;

		#ifdef MPI_ENABLED
		if (!rank)
		#endif
		{
		//Read Node Positions
		if (!network->network_properties.flags.no_pos) {
			printf("\tReading Node Position Data.....\n");
			fflush(stdout);
			dataname.str("");
			dataname.clear();
			dataname << prefix << "pos/" << network->network_properties.graphID << ".cset.pos.dat";
			dataStream.open(dataname.str().c_str());
			if (dataStream.is_open()) {
				for (i = 0; i < network->network_properties.N_tar; i++) {
					getline(dataStream, line);

					if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
						if (spacetime.stdimIs("2")) {
							network->nodes.crd->x(i) = atof(strtok((char*)line.c_str(), delimeters));
							network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
						} else if (spacetime.stdimIs("4")) {
							if (spacetime.manifoldIs("FLRW")) {
								network->nodes.id.tau[i] = atof(strtok((char*)line.c_str(), delimeters));
								if (USE_GSL) {
									idata.upper = network->nodes.id.tau[i];
									network->nodes.crd->w(i) = integrate1D(&tauToEtaFLRW, NULL, &idata, QAGS) * network->network_properties.a / network->network_properties.alpha;
								} else
									network->nodes.crd->w(i) = tauToEtaFLRWExact(network->nodes.id.tau[i], network->network_properties.a, network->network_properties.alpha);
							} else if (spacetime.manifoldIs("Dust")) {
								network->nodes.id.tau[i] = atof(strtok((char*)line.c_str(), delimeters));
								network->nodes.crd->w(i) = tauToEtaDust(network->nodes.id.tau[i], network->network_properties.a, network->network_properties.alpha);
							} else if (spacetime.manifoldIs("De_Sitter")) {
								network->nodes.crd->w(i) = atof(strtok((char*)line.c_str(), delimeters));
								if (spacetime.curvatureIs("Flat"))
									network->nodes.id.tau[i] = etaToTauFlat(network->nodes.crd->w(i));
								else if (spacetime.curvatureIs("Positive"))
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
					} else if (spacetime.manifoldIs("Hyperbolic")) {
						network->nodes.id.AS[i] = atoi(strtok((char*)line.c_str(), delimeters));
						network->nodes.crd->x(i) = atof(strtok(NULL, delimeters));
						network->nodes.crd->y(i) = atof(strtok(NULL, delimeters));
					}

					if (!network->network_properties.flags.quiet_read) {
						if (spacetime.stdimIs("2")) {
							if (spacetime.symmetryIs("None") && network->nodes.crd->x(i) < 0.0) {
								message = "Invalid value parsed for 'eta/r' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (!spacetime.manifoldIs("Minkowski") && (network->nodes.crd->y(i) < 0.0 || network->nodes.crd->y(i) > TWO_PI)) {
								message = "Invalid value for 'theta' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
						} else if (spacetime.stdimIs("4") && (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))) {
							if ((spacetime.curvatureIs("Positive") && ((spacetime.symmetryIs("Temporal") && fabs(network->nodes.crd->w(i)) > static_cast<float>(HALF_PI - network->network_properties.zeta)) || (spacetime.symmetryIs("None") && (network->nodes.crd->w(i) < 0.0 || network->nodes.crd->w(i) > static_cast<float>(HALF_PI - network->network_properties.zeta))))) || (spacetime.curvatureIs("Flat") && ((spacetime.manifoldIs("De_Sitter") && (network->nodes.crd->w(i) < HALF_PI - network->network_properties.zeta || network->nodes.crd->w(i) > HALF_PI - network->network_properties.zeta1)) || ((spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) && (network->nodes.crd->w(i) < 0.0 || network->nodes.crd->w(i) > HALF_PI - network->network_properties.zeta))))) {
								message = "Invalid value for 'eta' in node position file!\n";
								network->network_properties.cmpi.fail = 1;
								goto LoadPoint2;
							}
							if (network->nodes.crd->x(i) < 0.0 || (spacetime.curvatureIs("Positive") && network->nodes.crd->x(i) > M_PI) || (spacetime.curvatureIs("Flat") && network->nodes.crd->x(i) > network->network_properties.r_max)) {
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
				fprintf(stderr, "Failed to open node position file!\n");
				message = "Failed to open node position file!\n";
				network->network_properties.cmpi.fail = 1;
				goto LoadPoint2;
			}
			printf("\t\tCompleted.\n");
			fflush(stdout);

			if (USE_GSL && spacetime.manifoldIs("FLRW"))
				gsl_integration_workspace_free(idata.workspace);

			//Quicksort
			quicksort(network->nodes, network->network_properties.spacetime, 0, network->network_properties.N_tar - 1);
		}
		}

		//Broadcast
		#ifdef MPI_ENABLED
		MPI_Barrier(MPI_COMM_WORLD);
		if (network->nodes.id.tau != NULL)
			MPI_Bcast(network->nodes.id.tau, network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (network->nodes.crd->v() != NULL)
			MPI_Bcast(network->nodes.crd->v(), network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (network->nodes.crd->w() != NULL)
			MPI_Bcast(network->nodes.crd->w(), network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (network->nodes.crd->x() != NULL)
			MPI_Bcast(network->nodes.crd->x(), network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (network->nodes.crd->y() != NULL)
			MPI_Bcast(network->nodes.crd->y(), network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (network->nodes.crd->z() != NULL)
			MPI_Bcast(network->nodes.crd->z(), network->network_properties.N_tar, MPI_FLOAT, 0, MPI_COMM_WORLD);
		#endif

		//Re-Link Using linkNodes Subroutine
		if ((spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) && network->network_properties.flags.relink && !network->network_properties.flags.no_pos) {
			if (!linkNodes(network, ca, cp, ctx))
				network->network_properties.cmpi.fail = 1;
			goto LoadPoint2;
		} else if (!network->network_properties.flags.link)
			goto LoadPoint2;

		//Populate Hashmap
		if (spacetime.manifoldIs("Hyperbolic"))
			for (i = 0; i < network->network_properties.N_tar; i++)
				network->nodes.AS_idx.insert(std::make_pair(network->nodes.id.AS[i], i));

		#ifndef MPI_ENABLED
		//Read Edges
		printf("\tReading Edge List Data.....\n");
		fflush(stdout);
		dataname.str("");
		dataname.clear();
		dataname << prefix << "edg/" << network->network_properties.graphID << ".cset.edg.dat";
		if (network->network_properties.flags.use_bit)
			dataStream.open(dataname.str().c_str(), std::ios::binary);
		else
			dataStream.open(dataname.str().c_str());
		if (dataStream.is_open()) {
			if (network->network_properties.flags.use_bit) {
				for (i = 0; i < network->network_properties.N_tar; i++) {
					dataStream.read(reinterpret_cast<char*>(network->adj[i].getAddress()), sizeof(BlockType) * network->adj[i].getNumBlocks());
					if (!!i)
						network->nodes.k_in[i] = network->adj[i].partial_count(0, i);
					if (!!(network->network_properties.N_tar - i - 1))
						network->nodes.k_out[i] = network->adj[i].partial_count(i + 1, network->network_properties.N_tar - i - 1);
					int k = network->nodes.k_in[i] + network->nodes.k_out[i];
					N_edg += k;
					if (!!k) {
						network->network_observables.N_res++;
						if (!!(k - 1))
							network->network_observables.N_deg2++;
					}
				}
				network->network_properties.k_tar = static_cast<float>(static_cast<long double>(N_edg) / network->network_properties.N_tar);
				network->network_observables.k_res = static_cast<float>(static_cast<long double>(N_edg) / network->network_observables.N_res);
			} else {
				edges = (uint64_t*)malloc(sizeof(uint64_t) * N_edg);
				if (edges == NULL) {
					message = "bad alloc";
					network->network_properties.cmpi.fail = 1;
					goto LoadPoint2;
				}
				memset(edges, 0, sizeof(uint64_t) * N_edg);
				ca->hostMemUsed += sizeof(uint64_t) * N_edg;
			
				//Read Raw Data	
				for (i = 0; i < N_edg; i++) {
					getline(dataStream, line);
					e0 = atoi(strtok((char*)line.c_str(), delimeters));
					e1 = atoi(strtok(NULL, delimeters));

					if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
						idx0 = e0;
						idx1 = e1;
					} else if (spacetime.manifoldIs("Hyperbolic")) {
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

					if (!network->network_properties.flags.use_bit)
						edges[i] = ((uint64_t)idx0) << 32 | ((uint64_t)idx1);

					if (idx0 < core_limit && idx1 < core_limit) {
						network->adj[idx0].set(idx1);
						network->adj[idx1].set(idx0);
					}

					network->nodes.k_in[idx1]++;
					network->nodes.k_out[idx0]++;
				}

				//Sort Edge List
				quicksort(edges, 0, N_edg - 1);

				//Future Edge Data
				node_idx = -1;
				for (i = 0; i < N_edg; i++) {
					key = edges[i];
					idx0 = key >> 32;
					idx1 = key & 0x00000000FFFFFFFF;
					network->edges.future_edges[i] = idx1;
					edges[i] = ((uint64_t)idx1) << 32 | ((uint64_t)idx0);
			
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

				//Resort Edge List
				quicksort(edges, 0, N_edg - 1);

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

				free(edges);
				edges = NULL;
				ca->hostMemUsed -= sizeof(uint64_t) * N_edg;

				for (i = 0; i < network->network_properties.N_tar; i++) {
					if (network->nodes.k_in[i] + network->nodes.k_out[i] > 0) {
						network->network_observables.N_res++;
						if (network->nodes.k_in[i] + network->nodes.k_out[i] > 1)
							network->network_observables.N_deg2++;
					}
				}
				network->network_observables.k_res = static_cast<float>(static_cast<long double>(N_edg) * 2 / network->network_observables.N_res);
			}

			/*if (!printDegrees(network->nodes, network->network_properties.N_tar, "in-degrees_FILE.cset.dbg.dat", "out-degrees_FILE.cset.dbg.dat")) return false;
			if (!printEdgeLists(network->edges, N_edg, "past-edges_FILE.cset.dbg.dat", "future-edges_FILE.cset.dbg.dat")) return false;
			if (!printEdgeListPointers(network->edges, network->network_properties.N_tar, "past-edge-pointers_FILE.cset.dbg.dat", "future-edge-pointers_FILE.cset.dbg.dat")) return false;
			printf_red();
			printf("Check files now.\n");
			printf_std();
			fflush(stdout);
			exit(0);*/

			//if (!compareCoreEdgeExists(network->nodes.k_out, network->edges.future_edges, network->edges.future_edge_row_start, network->adj, network->network_properties.N_tar, network->network_properties.core_edge_fraction))
			//	return false;

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
		
		if (((spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) && network->network_properties.flags.relink) || !network->network_properties.flags.link)
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

	#ifdef MPI_ENABLED
	if (network.network_properties.cmpi.rank)
		return true;
	#endif

	printf("Printing Results to File...\n");
	fflush(stdout);

	int i, j, k;
	int idx = 0;

	try {
		std::ofstream outputStream;
		std::ofstream mapStream;
		std::ofstream dataStream;
		std::stringstream sstm;

		//Generate Filename
		if (network.network_properties.flags.use_gpu)
			sstm << "Dev" << gpuID << "_";
		else
			sstm << "CPU_";
		sstm << "ST-" << network.network_properties.spacetime.toHexString() << "_";
		sstm << "VER-" << VERSION << "_";
		sstm << network.network_properties.seed;
		std::string filename = sstm.str();
		std::string prefix = network.network_properties.datdir;

		//Write Simulation Parameters and Main Results to File
		sstm.str("");
		sstm.clear();
		sstm << prefix << filename << ".cset.out";
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

		Spacetime spacetime = network.network_properties.spacetime;
		bool links_exist = network.network_properties.flags.link || network.network_properties.flags.relink;

		if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
			outputStream << "\nCauset Initial Parameters:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			outputStream << "Number of Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Expected Degrees (k_tar)\t\t" << network.network_properties.k_tar << std::endl;
			if (spacetime.manifoldIs("FLRW"))
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
			if (spacetime.symmetryIs("None") && !spacetime.manifoldIs("Hyperbolic"))
				outputStream << "Minimum Rescaled Time\t\t\t" << network.nodes.id.tau[0] << std::endl;
		} else if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Hyperbolic")) {
			outputStream << "\nCauset Input Parameters:" << std::endl;
			outputStream << "------------------------" << std::endl;
			outputStream << "Number of Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Expected Degrees (k_tar)\t\t" << network.network_properties.k_tar << std::endl;
			if (spacetime.manifoldIs("Hyperbolic")) {
				outputStream << "Max. Radius (r_max)\t\t\t" << network.network_properties.r_max << std::endl;
				outputStream << "Hyperbolic Curvature (zeta)\t\t" << network.network_properties.zeta << std::endl;
			} else {
				outputStream << "Max. Conformal Time (eta_0)\t\t";
				if (spacetime.manifoldIs("De Sitter") && spacetime.curvatureIs("Flat"))
					outputStream << HALF_PI - network.network_properties.zeta1 << std::endl;
				else
					outputStream << HALF_PI - network.network_properties.zeta << std::endl;
			}
			if (spacetime.manifoldIs("De_Sitter"))
				outputStream << "Pseudoradius (a)\t\t\t" << network.network_properties.a << std::endl;

			outputStream << "\nCauset Calculated Values:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			if (links_exist) {
				outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_observables.N_res << std::endl;
				outputStream << "Resulting Average Degrees (k_res)\t" << network.network_observables.k_res << std::endl;
				outputStream << "    Incl. Isolated Nodes\t\t" << (network.network_observables.k_res * network.network_observables.N_res) / network.network_properties.N_tar << std::endl;
				outputStream << "Resulting Error in <k>\t\t\t" << (fabs(network.network_properties.k_tar - network.network_observables.k_res) / network.network_properties.k_tar) << std::endl;
			}
			if (spacetime.manifoldIs("De_Sitter"))
				outputStream << "Minimum Rescaled Time\t\t\t" << network.nodes.id.tau[0] << std::endl;
			else if (spacetime.manifoldIs("Hyperbolic"))
				outputStream << "Minimum Radial Coordinate\t\t" << network.nodes.crd->x(0) << std::endl;
		}

		if (network.network_properties.flags.calc_clustering)
			outputStream << "Average Clustering\t\t\t" << network.network_observables.average_clustering << std::endl;

		if (network.network_properties.flags.calc_components) {
			outputStream << "Number of Connected Components\t\t" << network.network_observables.N_cc << std::endl;
			outputStream << "Size of Giant Connected Component\t" << network.network_observables.N_gcc << std::endl;
		}

		if (network.network_properties.flags.calc_success_ratio) {
			outputStream << "Success Ratio Type 1\t\t\t" << network.network_observables.success_ratio << std::endl;
			#if !TRAVERSE_V2
			outputStream << "Success Ratio Type 2\t\t\t" << network.network_observables.success_ratio2 << std::endl;
			#endif
			if (network.network_properties.flags.calc_stretch)
				outputStream << "Stretch\t\t\t\t\t" << network.network_observables.stretch << std::endl;
		}

		if (network.network_properties.flags.calc_action)
			outputStream << "Action\t\t\t\t\t" << network.network_observables.action << std::endl;

		if (network.network_properties.flags.calc_hubs)
			outputStream << "Hub Density\t\t\t\t" << network.network_observables.hub_density << std::endl;

		outputStream << "\nNetwork Analysis Results:" << std::endl;
		outputStream << "-------------------------" << std::endl;
		outputStream << "Node Position Data:\t\t\t" << "pos/" << network.network_properties.graphID << ".cset.pos.dat" << std::endl;
		if (links_exist) {
			if (network.network_properties.flags.print_edges)
				outputStream << "Node Edge Data:\t\t\t\t" << "edg/" << network.network_properties.graphID << ".cset.edg.dat" << std::endl;
			outputStream << "Degree Distribution Data:\t\t" << "dst/" << network.network_properties.graphID << ".cset.dst.dat" << std::endl;
			outputStream << "In-Degree Distribution Data:\t\t" << "idd/" << network.network_properties.graphID << ".cset.idd.dat" << std::endl;
			outputStream << "Out-Degree Distribution Data:\t\t" << "odd/" << network.network_properties.graphID << ".cset.odd.dat" << std::endl;
		}

		if (network.network_properties.flags.calc_clustering) {
			outputStream << "Clustering Coefficient Data:\t\t" << "cls/" << network.network_properties.graphID << ".cset.cls.dat" << std::endl;
			outputStream << "Clustering by Degree Data:\t\t" << "cdk/" << network.network_properties.graphID << ".cset.cdk.dat" << std::endl;
		}

		if (network.network_properties.flags.calc_action)
			outputStream << "Action/Cardinality Data:\t\t" << "act/" << network.network_properties.graphID << ".cset.act.dat" << std::endl;

		outputStream << "\nAlgorithmic Performance:" << std::endl;
		outputStream << "--------------------------" << std::endl;
		if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
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
		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "measureSuccessRatio:\t\t\t" << cp.sMeasureSuccessRatio.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_action)
			outputStream << "measureAction:\t\t\t\t" << cp.sMeasureAction.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_hubs)
			outputStream << "measureHubDensity:\t\t\t\t" << cp.sMeasureHubs.elapsedTime << " sec" << std::endl;

		outputStream.flush();
		outputStream.close();

		//Add Data Key
		sstm.str("");
		sstm.clear();
		sstm << prefix << "../etc/data_keys.cset.key";
		mapStream.open(sstm.str().c_str(), std::ios::app);
		if (!mapStream.is_open())
			throw CausetException("Failed to open 'data_keys.cset.key' file!\n");
		mapStream << network.network_properties.graphID << "\t" << filename << std::endl;
		mapStream.close();

		if (network.network_properties.flags.bench)
			goto PrintExit;

		//Write Data to File
		sstm.str("");
		sstm.clear();
		sstm << prefix << "pos/" << network.network_properties.graphID << ".cset.pos.dat";
		printf("Node position filename: %s\n", sstm.str().c_str());
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open node position file!\n");
		dataStream << std::fixed << std::setprecision(9);
		for (i = 0; i < network.network_properties.N_tar; i++) {
			if (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
				if (spacetime.stdimIs("4")) {
					if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW"))
						dataStream << network.nodes.id.tau[i];
					else if (spacetime.manifoldIs("De_Sitter"))
						dataStream << network.nodes.crd->w(i);
					dataStream << " " << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i) << " " << network.nodes.crd->z(i);
				} else if (spacetime.stdimIs("3"))
					dataStream << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i) << " " << network.nodes.crd->z(i);
				else if (spacetime.stdimIs("2"))
					dataStream << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i);
			} else if (spacetime.manifoldIs("Hyperbolic")) {
				if (spacetime.curvatureIs("Flat"))
					dataStream << network.nodes.id.AS[i] << " " << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i);
				else if (spacetime.curvatureIs("Positive"))
					dataStream << network.nodes.id.tau[i] << " " << network.nodes.crd->x(i) << " " << network.nodes.crd->y(i);
			}
			dataStream << "\n";
		}
		dataStream.flush();
		dataStream.close();

		if (links_exist) {
			if (network.network_properties.flags.print_edges) {
				sstm.str("");
				sstm.clear();
				sstm << prefix << "edg/" << network.network_properties.graphID << ".cset.edg.dat";

				if (network.network_properties.flags.use_bit) {
					dataStream.open(sstm.str().c_str(), std::ios::binary);
					if (!dataStream.is_open())
						throw CausetException("Failed to open edge list file!\n");
					for (i = 0; i < network.network_properties.N_tar; i++)
						dataStream.write(reinterpret_cast<const char*>(network.adj[i].getAddress()), sizeof(BlockType) * network.adj[i].getNumBlocks());
					dataStream.flush();
					dataStream.close();

					if (network.network_properties.flags.print_dot) {
						sstm.str("");
						sstm.clear();
						sstm << prefix << "dot/" << network.network_properties.graphID << ".cset.dot";
						dataStream.open(sstm.str().c_str());
						if (!dataStream.is_open())
							throw CausetException("Failed to open dot file!\n");
						dataStream << "digraph \"causet\" {\n";
						dataStream << "rankdir=BT; concentrate=true;\n";
						for (int i = 0; i < network.network_properties.N_tar; i++) {
							dataStream << i << " [shape=plaintext];\n";
							for (int j = i + 1; j < network.network_properties.N_tar; j++)
								if (network.adj[i].read(j))
									dataStream << i << "->" << j << "; ";
							if (!!network.nodes.k_out[i])
								dataStream << "\n";
						}
						dataStream << "}\n";
						dataStream.flush();
						dataStream.close();
					}
				} else {
					dataStream.open(sstm.str().c_str());
					if (!dataStream.is_open())
						throw CausetException("Failed to open edge list file!\n");
					for (i = 0; i < network.network_properties.N_tar; i++) {
						for (j = 0; j < network.nodes.k_out[i]; j++)
							dataStream << i << " " << network.edges.future_edges[idx+j] << "\n";
						idx += network.nodes.k_out[i];
					}
					dataStream.flush();
					dataStream.close();
				}
			}

			sstm.str("");
			sstm.clear();
			sstm << prefix << "dst/" << network.network_properties.graphID << ".cset.dst.dat";
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
			sstm << prefix << "idd/" << network.network_properties.graphID << ".cset.idd.dat";
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
			sstm << prefix << "odd/" << network.network_properties.graphID << ".cset.odd.dat";
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
			sstm << prefix << "cls/" << network.network_properties.graphID << ".cset.cls.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open clustering coefficient file!\n");
			for (i = 0; i < network.network_properties.N_tar; i++)
				if (network.nodes.k_in[i] + network.nodes.k_out[i] > 1)
					dataStream << network.nodes.k_in[i] + network.nodes.k_out[i] << " " << network.network_observables.clustering[i] << "\n";
			dataStream.flush();
			dataStream.close();

			sstm.str("");
			sstm.clear();
			sstm << prefix << "cdk/" << network.network_properties.graphID << ".cset.cdk.dat";
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
				if (cdk > 0.0)
					dataStream << i << " " << (cdk / ndk) << "\n";
			}
			dataStream.flush();
			dataStream.close();
		}

		if (network.network_properties.flags.calc_action) {
			sstm.str("");
			sstm.clear();
			sstm << prefix << "act/" << network.network_properties.graphID << ".cset.act.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open action file!\n");
			for (i = 0; i < network.network_properties.max_cardinality; i++)
				dataStream << network.network_observables.cardinalities[i] << "\n";

			dataStream.flush();
			dataStream.close();
		}

		if (network.network_properties.flags.calc_hubs) {
			sstm.str("");
			sstm.clear();
			sstm << prefix << "hub/" << network.network_properties.graphID << ".cset.hub.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open hub file!\n");
			for (int m = 0; m < network.network_properties.N_hubs; m++)
				dataStream << network.network_observables.hub_densities[m] << "\n";
			dataStream.flush();
			dataStream.close();
		}

		PrintExit:

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
	if (cf.calc_action)
		printf("\tmeasureAction:\t\t%5.6f sec\n", bm.bMeasureAction);
	printf("\n");
	fflush(stdout);

	return true;
}

//Free Memory
void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed)
{
	Spacetime spacetime = network->network_properties.spacetime;
	int rank = network->network_properties.cmpi.rank;
	bool links_exist = network->network_properties.flags.link || network->network_properties.flags.relink;

	if (!network->network_properties.flags.no_pos) {
		if ((spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) || (spacetime.manifoldIs("Hyperbolic") && spacetime.curvatureIs("Positive"))) {
			free(network->nodes.id.tau);
			network->nodes.id.tau = NULL;
			hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
		} else if (spacetime.manifoldIs("Hyperbolic") && spacetime.curvatureIs("Flat")) {
			free(network->nodes.id.AS);
			network->nodes.id.AS = NULL;
			hostMemUsed -= sizeof(int) * network->network_properties.N_tar;
		}

		if (spacetime.stdimIs("4")) {
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
		} else if (spacetime.stdimIs("3")) {
			free(network->nodes.crd->x());
			network->nodes.crd->x() = NULL;

			free(network->nodes.crd->y());
			network->nodes.crd->y() = NULL;

			free(network->nodes.crd->z());
			network->nodes.crd->z() = NULL;

			hostMemUsed -= sizeof(float) * network->network_properties.N_tar * 3;
		} else if (spacetime.stdimIs("2")) {
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
			hostMemUsed -= sizeof(int64_t) * network->network_properties.N_tar;

			free(network->edges.future_edge_row_start);
			network->edges.future_edge_row_start = NULL;
			hostMemUsed -= sizeof(int64_t) * network->network_properties.N_tar;
		}

		int length = 0;
		if (network->network_properties.flags.mpi_split) {
			length = static_cast<int>(ceil(static_cast<float>(static_cast<int>(network->network_properties.N_tar * network->network_properties.core_edge_fraction)) / network->network_properties.cmpi.num_mpi_threads));
			int n = static_cast<unsigned int>(POW2(network->network_properties.cmpi.num_mpi_threads, EXACT)) << 1;
			if (length % n)
				length += n - (length % n);
		} else
			length = static_cast<int>(ceil(network->network_properties.N_tar * network->network_properties.core_edge_fraction));
		for (int i = 0; i < length; i++)
			hostMemUsed -= sizeof(BlockType) * network->adj[i].getNumBlocks();
		network->adj.clear();
		network->adj.swap(network->adj);

		#ifdef MPI_ENABLED
		if (network->network_properties.flags.mpi_split && network->network_properties.cmpi.num_mpi_threads > 1) {
			int buflen = length / (network->network_properties.cmpi.num_mpi_threads << 1);
			for (int i = 0; i < buflen; i++)
				hostMemUsed -= sizeof(BlockType) * network->network_properties.cmpi.adj_buf[i].getNumBlocks();
			network->network_properties.cmpi.adj_buf.clear();
			network->network_properties.cmpi.adj_buf.swap(network->network_properties.cmpi.adj_buf);
		}
		#endif
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

	if (network->network_properties.flags.calc_action) {
		free(network->network_observables.cardinalities);
		network->network_observables.cardinalities = NULL;
		hostMemUsed -= sizeof(uint64_t) * network->network_properties.max_cardinality * omp_get_max_threads();

		/*hostMemUsed -= sizeof(unsigned int) * network->network_observables.timelike_candidates.size();
		network->network_observables.timelike_candidates.clear();
		network->network_observables.timelike_candidates.swap(network->network_observables.timelike_candidates);
	
		free(network->network_observables.chaintime);
		network->network_observables.chaintime = NULL;
		hostMemUsed -= sizeof(int) * network->network_properties.N_tar;*/
	}

	if (network->network_properties.flags.calc_action_theory) {
		free(network->network_observables.actth);
		network->network_observables.actth = NULL;
		int min = 10;
		int nsamples = 3 * (network->network_properties.N_actth - min) + 1;
		hostMemUsed -= sizeof(double) * nsamples * omp_get_max_threads();
	}

	if (network->network_properties.flags.calc_hubs) {
		free(network->network_observables.hub_densities);
		network->network_observables.hub_densities = NULL;
		hostMemUsed -= sizeof(float) * (network->network_properties.N_hubs + 1);
	}
}

//--------------------//
// Additional Methods //
//--------------------//

bool linkNodes(Network * const network, CaResources * const ca, CausetPerformance * const cp, const CUcontext &ctx)
{
	#ifdef CUDA_ENABLED
	if (network->network_properties.flags.use_gpu && network->network_properties.N_tar >= GPU_MIN) {
		#if LINK_NODES_GPU_V2
		if (!linkNodesGPU_v2(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.r_max, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, network->network_properties.cmpi, network->network_properties.group_size, ca, cp->sLinkNodesGPU, ctx, network->network_properties.flags.decode_cpu, network->network_properties.flags.link_epso, network->network_properties.flags.has_exact_k, network->network_properties.flags.use_bit, network->network_properties.flags.mpi_split, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
		#else
		if (!linkNodesGPU_v1(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, ca, cp->sLinkNodesGPU, network->network_properties.flags.link_epso, network->network_properties.flags.has_exact_k, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
		#endif
	} else
	#endif
	{
		//#ifdef MPI_ENABLED
		//if (network->network_properties.cmpi.num_mpi_threads > 1) {
		if (network->network_properties.flags.use_bit) {
			if (!linkNodes_v2(network->nodes, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.cmpi, cp->sLinkNodes, network->network_properties.flags.link_epso, network->network_properties.flags.has_exact_k, network->network_properties.flags.use_bit, network->network_properties.flags.mpi_split, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		} else
		//#endif
		{
			if (!linkNodes_v1(network->nodes, network->edges, network->adj, network->network_properties.spacetime, network->network_properties.N_tar, network->network_properties.k_tar, network->network_observables.N_res, network->network_observables.k_res, network->network_observables.N_deg2, network->network_properties.a, network->network_properties.zeta, network->network_properties.zeta1, network->network_properties.r_max, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.link_epso, network->network_properties.flags.has_exact_k, network->network_properties.flags.use_bit, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}
	}

	return true;
}

bool configureSubgraph(Network *subgraph, const Node &nodes, std::vector<unsigned int> candidates, CaResources * const ca, const CUcontext &ctx)
{
	#ifdef DEBUG
	assert (candidates.size() > 0);
	#endif

	printf("\n\tPreparing to Generate Subgraph...\n");
	fflush(stdout);

	subgraph->network_properties.N_tar = candidates.size();

	subgraph->network_properties.flags.use_gpu = false;	
	subgraph->network_properties.flags.has_exact_k = false;

	subgraph->network_properties.flags.use_bit = true;

	subgraph->network_properties.flags.calc_clustering = false;
	subgraph->network_properties.flags.calc_components = false;
	subgraph->network_properties.flags.calc_success_ratio = false;
	subgraph->network_properties.flags.calc_action = false;

	subgraph->network_properties.flags.verbose = false;
	subgraph->network_properties.flags.bench = false;

	subgraph->network_observables.clustering = NULL;
	subgraph->network_observables.cardinalities = NULL;

	CausetPerformance cp = CausetPerformance();
	if (!createNetwork(subgraph->nodes, subgraph->edges, subgraph->adj, subgraph->network_properties.spacetime, subgraph->network_properties.N_tar, subgraph->network_properties.k_tar, subgraph->network_properties.core_edge_fraction, subgraph->network_properties.edge_buffer, subgraph->network_properties.cmpi, subgraph->network_properties.group_size, ca, cp.sCreateNetwork, subgraph->network_properties.flags.use_gpu, subgraph->network_properties.flags.decode_cpu, subgraph->network_properties.flags.link, subgraph->network_properties.flags.relink, subgraph->network_properties.flags.no_pos, subgraph->network_properties.flags.use_bit, subgraph->network_properties.flags.mpi_split, subgraph->network_properties.flags.verbose, subgraph->network_properties.flags.bench, subgraph->network_properties.flags.yes))
		return false;

	for (int i = 0; i < subgraph->network_properties.N_tar; i++) {
		if (subgraph->nodes.id.tau != NULL)
			subgraph->nodes.id.tau[i] = nodes.id.tau[candidates[i]];
		if (subgraph->nodes.crd->v() != NULL)
			subgraph->nodes.crd->v(i) = nodes.crd->v(candidates[i]);
		if (subgraph->nodes.crd->w() != NULL)
			subgraph->nodes.crd->w(i) = nodes.crd->w(candidates[i]);
		if (subgraph->nodes.crd->x() != NULL)
			subgraph->nodes.crd->x(i) = nodes.crd->x(candidates[i]);
		if (subgraph->nodes.crd->y() != NULL)
			subgraph->nodes.crd->y(i) = nodes.crd->y(candidates[i]);
		if (subgraph->nodes.crd->z() != NULL)
			subgraph->nodes.crd->z(i) = nodes.crd->z(candidates[i]);
	}

	//int low = 0;
	//int high = subgraph->network_properties.N_tar - 1;
	//quicksort(subgraph->nodes, subgraph->network_properties.spacetime, low, high);

	if (!linkNodes(subgraph, ca, &cp, ctx))
		return false;

	printf("\tGenerated Subgraph from Candidates.\n");
	fflush(stdout);

	return true;
}
