#include "NetworkCreator.h"
#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

int main(int argc, char **argv)
{
	CausetPerformance cp = CausetPerformance();
	stopwatchStart(&cp.sCauset);

	Network network = Network(parseArgs(argc, argv));
	Benchmark bm = Benchmark();
	Resources resources = Resources();
	
	long init_seed = network.network_properties.seed;
	bool success = false;

	shrQAStart(argc, argv);

	if (network.network_properties.flags.use_gpu)
		connectToGPU(&resources, argc, argv);
	
	if (network.network_properties.graphID == 0 && !initializeNetwork(&network, &cp, &bm, resources.hostMemUsed, resources.maxHostMemUsed, resources.devMemUsed, resources.maxDevMemUsed)) goto CausetExit;
	else if (network.network_properties.graphID != 0 && !loadNetwork(&network, &cp, resources.hostMemUsed, resources.maxHostMemUsed, resources.devMemUsed, resources.maxDevMemUsed)) goto CausetExit;

	if (!measureNetworkObservables(&network, &cp, &bm, resources.hostMemUsed, resources.maxHostMemUsed, resources.devMemUsed, resources.maxDevMemUsed)) goto CausetExit;

	if (network.network_properties.flags.bench && !printBenchmark(bm, network.network_properties.flags)) goto CausetExit;
	if (network.network_properties.flags.disp_network && !displayNetwork(network.nodes, network.future_edges, argc, argv)) goto CausetExit;
	if (!network.network_properties.flags.bench) printMemUsed(NULL, resources.maxHostMemUsed, resources.maxDevMemUsed);
	if (network.network_properties.flags.print_network && !printNetwork(network, cp, init_seed, resources.gpuID)) goto CausetExit;
	
	destroyNetwork(&network, resources.hostMemUsed, resources.devMemUsed);
	if (network.network_properties.flags.use_gpu) cuCtxDetach(resources.cuContext);

	success = true;
	stopwatchStop(&cp.sCauset);

	CausetExit:
	if (cp.sCauset.stopTime.tv_sec == 0 && cp.sCauset.stopTime.tv_usec == 0)
		stopwatchStop(&cp.sCauset);
	shrQAFinish(argc, (const char**)argv, success ? QA_PASSED : QA_FAILED);
	printf("Time: %5.6f sec\n", cp.sCauset.elapsedTime);
	printf("PROGRAM COMPLETED\n");
	fflush(stdout);
}

//Parse Command Line Arguments
NetworkProperties parseArgs(int argc, char **argv)
{
	NetworkProperties network_properties = NetworkProperties();

	//Initialize conflict array to zeros (no conflicts)
	for (int i = 0; i < 7; i++)
		network_properties.flags.cc.conflicts[i] = 0;

	int c, longIndex;
	//Single-character options
	static const char *optString = ":m:n:k:d:s:a:c:g:t:A:D:o:r:l:uvyCSh";
	//Multi-character options
	static const struct option longOpts[] = {
		{ "manifold",	required_argument,	NULL, 'm' },
		{ "nodes", 	required_argument,	NULL, 'n' },
		{ "degrees",	required_argument,	NULL, 'k' },
		{ "dim",	required_argument,	NULL, 'd' },
		{ "seed",	required_argument,	NULL, 's' },
		{ "radius",	required_argument,	NULL, 'a' },
		{ "core",	required_argument,	NULL, 'c' },
		{ "clustering",	no_argument,		NULL, 'C' },
		{ "success",	required_argument,	NULL, 'S' },
		{ "graph",	required_argument,	NULL, 'g' },
		{ "universe",	no_argument,		NULL, 'u' },
		{ "age",	required_argument,	NULL, 't' },
		{ "alpha",	required_argument,	NULL, 'A' },
		{ "delta",	required_argument,	NULL, 'D' },
		{ "energy",	required_argument,	NULL, 'o' },
		{ "ratio",	required_argument,	NULL, 'r' },
		{ "lambda",	required_argument,	NULL, 'l' },

		{ "help", 	no_argument,		NULL, 'h' },
		{ "gpu", 	no_argument, 		NULL,  0  },
		{ "display", 	no_argument, 		NULL,  0  },
		{ "print", 	no_argument, 		NULL,  0  },
		{ "verbose", 	no_argument, 		NULL, 'v' },
		{ "benchmark",	no_argument,		NULL,  0  },
		{ "autocorr",	no_argument,		NULL,  0  },
		{ "conflicts",  no_argument,		NULL,  0  },
		{ NULL,		0,			0,     0  }
	};

	try {
		while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
			switch (c) {
			case 'm':	//Manifold
				if (strcmp(optarg, "e"))
					network_properties.manifold = EUCLIDEAN;
				else if (strcmp(optarg, "d"))
					network_properties.manifold = DE_SITTER;
				else if (strcmp(optarg, "a"))
					network_properties.manifold = ANTI_DE_SITTER;
				else
					throw CausetException("Invalid argument for 'manifold' parameter!\n");

				if (network_properties.manifold != DE_SITTER) {
					printf("Only de Sitter manifold currently supported!  Reverting to default value.\n");
					fflush(stdout);
					network_properties.manifold = DE_SITTER;
				}

				break;
			case 'n':	//Number of nodes
				network_properties.N_tar = atoi(optarg);
				if (network_properties.N_tar <= 0)
					throw CausetException("Invalid argument for 'nodes' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'k':	//Average expected degrees
				network_properties.k_tar = atof(optarg);
				if (network_properties.k_tar <= 0.0)
					throw CausetException("Invalid argument for 'degrees' parameter!\n");
				break;
			case 'd':	//Spatial dimensions
				network_properties.dim = atoi(optarg);
				if (!(atoi(optarg) == 1 || atoi(optarg) == 3))
					throw CausetException("Invalid argument for 'dimension' parameter!\n");
				assert (network_properties.dim != 1);	//Fix the eta distribution for 1D
				break;
			case 's':	//Random seed
				network_properties.seed = -1.0 * atol(optarg);
				if (network_properties.seed >= 0.0L)
					throw CausetException("Invalid argument for 'seed' parameter!\n");
				break;
			case 'a':	//Pseudoradius
				network_properties.a = atof(optarg);
				
				if (network_properties.a <= 0.0)
					throw CausetException("Invalid argument for 'a' parameter!\n");

				network_properties.lambda = 3.0 / POW2(network_properties.a, EXACT);

				network_properties.flags.cc.conflicts[0]++;

				break;
			case 'c':	//Core edge fraction (used for adjacency matrix)
				network_properties.core_edge_fraction = atof(optarg);
				if (network_properties.core_edge_fraction <= 0.0 || network_properties.core_edge_fraction >= 1.0)
					throw CausetException("Invalid argument for 'c' parameter!\n");
				break;
			case 'C':	//Flag for calculating clustering
				network_properties.flags.calc_clustering = true;
				break;
			case 'S':	//Flag for calculating success ratio
				network_properties.flags.calc_success_ratio = true;

				if (strcmp(optarg, "all") == 0)
					network_properties.N_sr = -1;
				else {
					network_properties.N_sr = static_cast<int64_t>(atol(optarg));
					if (network_properties.N_sr <= 0)
						throw CausetException("Invalid argument for 'success' parameter!\n");
				}
				break;
			case 'g':	//Graph ID
				network_properties.graphID = atoi(optarg);
				if (network_properties.graphID <= 0)
					throw CausetException("Invalid argument for 'Graph ID' parameter!\n");
				break;
			case 'u':	//Flag for creating universe causet
				network_properties.flags.universe = true;
				break;
			case 't':	//Age of universe
				network_properties.tau0 = atof(optarg);

				if (network_properties.tau0 <= 0.0)
					throw CausetException("Invalid argument for 'age' parameter!\n");

				network_properties.ratio = POW2(SINH(1.5 * static_cast<float>(network_properties.tau0), STL), EXACT);
				network_properties.omegaM = 1.0 / (network_properties.ratio + 1.0);
				network_properties.omegaL = 1.0 - network_properties.omegaM;

				network_properties.flags.cc.conflicts[2]++;
				network_properties.flags.cc.conflicts[3]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'A':	//Rescaled ratio of dark energy density to matter density
				network_properties.alpha = atof(optarg);

				if (network_properties.alpha <= 0.0)
					throw CausetException("Invalid argument for 'alpha' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'D':	//Density of nodes
				network_properties.delta = atof(optarg);

				if (network_properties.delta <= 0.0)
					throw CausetException("Invalid argument for 'delta' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'o':	//Density of dark energy
				network_properties.omegaL = atof(optarg);

				if (network_properties.omegaL <= 0.0 || network_properties.omegaL >= 1.0)
					throw CausetException("Invalid input for 'energy' parameter!\n");

				network_properties.omegaM = 1.0 - network_properties.omegaL;
				network_properties.ratio = network_properties.omegaL / network_properties.omegaM;
				network_properties.tau0 = (2.0 / 3.0) * ASINH(SQRT(static_cast<float>(network_properties.ratio), STL), STL, DEFAULT);
					
				network_properties.flags.cc.conflicts[1]++;
				network_properties.flags.cc.conflicts[2]++;
				network_properties.flags.cc.conflicts[5]++;

				break;
			case 'r':	//Ratio of dark energy density to matter density
				network_properties.ratio = atof(optarg);

				if (network_properties.ratio <= 0.0)
					throw CausetException("Invalid argument for 'ratio' parameter!\n");

				network_properties.tau0 = (2.0 / 3.0) * ASINH(SQRT(static_cast<float>(network_properties.ratio), STL), STL, DEFAULT);
				network_properties.omegaM = 1.0 / (network_properties.ratio + 1.0);
				network_properties.omegaL = 1.0 - network_properties.omegaM;
				
				network_properties.flags.cc.conflicts[1]++;
				network_properties.flags.cc.conflicts[3]++;
				network_properties.flags.cc.conflicts[4]++;

				break;
			case 'l':	//Cosmological constant
				network_properties.lambda = atof(optarg);

				if (network_properties.lambda <= 0.0)
					throw CausetException("Invalid argument for 'lambda' parameter!\n");

				network_properties.a = SQRT(3.0 / static_cast<float>(network_properties.lambda), STL);

				network_properties.flags.cc.conflicts[0]++;

				break;
			case 'v':	//Verbose output
				network_properties.flags.verbose = true;
				break;
			case 'y':	//Suppress user input
				network_properties.flags.yes = true;
				break;
			case 0:
				if (strcmp("gpu", longOpts[longIndex].name) == 0)
					//Flag to use GPU accelerated routines
					network_properties.flags.use_gpu = true;
				else if (strcmp("display", longOpts[longIndex].name) == 0) {
					//Flag to use OpenGL to display network
					//network_properties.flags.disp_network = true;
					printf("Display not supported:  Ignoring Flag.\n");
					fflush(stdout);
				} else if (strcmp("print", longOpts[longIndex].name) == 0)
					//Flag to print results to file in 'dat' folder
					network_properties.flags.print_network = true;
				else if (strcmp("benchmark", longOpts[longIndex].name) == 0)
					//Flag to benchmark selected routines
					network_properties.flags.bench = true;
				else if (strcmp("autocorr", longOpts[longIndex].name) == 0)
					//Flag to calculate autocorrelation of selected variables
					network_properties.flags.calc_autocorr = true;
				else if (strcmp("conflicts", longOpts[longIndex].name) == 0) {
					//Print conflicting parameters
					printf("\nParameter Conflicts:\n");
					printf("--------------------\n");
					printf(" > a, lambda\n");
					printf(" > energy, ratio\n");
					printf(" > energy, age\n");
					printf(" > age, ratio\n");
					printf(" > n, delta, alpha, ratio\n");
					printf(" > n, delta, alpha, energy\n");
					printf(" > n, delta, alpha, age\n\n");
					printf("Specifying any of these combinations will over-constrain the system!\n\n");
					exit(EXIT_SUCCESS);
				} else {
					//Unrecognized options
					fprintf(stderr, "Option --%s is not recognized.\n", longOpts[longIndex].name);
					exit(EXIT_FAILURE);
				}
				break;
			case 'h':
				//Print help menu
				printf("\nUsage  :  CausalSet [options]\n\n");
				printf("CausalSet Options...................\n");
				printf("====================================\n");
				printf("Flag:\t\t\tVariable:\t\t\tSuggested Values:\n");
				printf("  -A, --alpha\t\tUnphysical Parameter\t\t2.0\n");
				printf("  -a, --radius\t\tPseudoradius\t\t\t1.0\n");
				printf("  -C, --clustering\tCalculate Clustering\n");
				printf("  -c, --core\t\tCore Edge Ratio\t\t\t0.01\n");
				printf("  -D, --delta\t\tNode Density\t\t\t10000\n");
				printf("  -d, --dim\t\tSpatial Dimensions\t\t1 or 3\n");
				printf("  -g, --graph\t\tGraph ID\t\t\tCheck dat/*.cset.out files\n");
				printf("  -h, --help\t\tDisplay this menu\n");
				printf("  -k, --degrees\t\tExpected Average Degrees\t10-100\n");
				printf("  -l, --lambda\t\tCosmological Constant\t\t3.0\n");
				printf("  -m, --manifold\tManifold\t\t\tEUCLIDEAN, DE_SITTER, ANTI_DE_SITTER\n");
				printf("  -n, --nodes\t\tNumber of Nodes\t\t\t100-100000\n");
				printf("  -o, --energy\t\tDark Energy Density\t\t0.73\n");
				printf("  -r, --ratio\t\tEnergy to Matter Ratio\t\t2.7\n");
				printf("  -S, --success\t\tCalculate Success Ratio\t 10000, all\n");
				printf("  -s, --seed\t\tRNG Seed\t\t\t18100L\n");
				printf("  -t, --age\t\tRescaled Age of Universe\t0.85\n");
				printf("  -u, --universe\tUniverse Causet\n");
				printf("  -v, --verbose\t\tVerbose Output\n");
				printf("  -y\t\t\tSuppress user queries\n");
				printf("\n");

				printf("Flag:\t\t\tPurpose:\n");
				printf("  --autocorr\t\tCalculate Autocorrelations\n");
				printf("  --benchmark\t\tBenchmark Algorithms\n");
				printf("  --conflicts\t\tShow Parameter Conflicts\n");
				printf("  --display\t\tDisplay Graph\n");
				printf("  --gpu\t\t\tUse GPU\n");
				printf("  --print\t\tPrint Results\n");
				printf("\n");
				exit(EXIT_SUCCESS);
			case ':':
				//Single-character flag needs an argument
				fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], optopt);
				exit(EXIT_FAILURE);
			case '?':	//Unrecognized flag
			default:	//Default case
				fprintf(stderr, "%s:option -%c' is not recognized.\n", argv[0], optopt);
				exit(EXIT_FAILURE);
			}
		}

		//Make sure necessary parameters have been specified
		if (!network_properties.flags.universe) {
			if (network_properties.N_tar == 0)
				throw CausetException("Flag '-n', number of nodes, must be specified!\n");
			else if (network_properties.k_tar == 0.0)
				throw CausetException("Flag '-k', expected average degrees, must be specified!\n");

			if (network_properties.flags.calc_success_ratio) {
				if (network_properties.N_sr == -1)
					network_properties.N_sr = static_cast<int64_t>(network_properties.N_tar) * (network_properties.N_tar - 1) / 2;
				else if (network_properties.N_sr > static_cast<int64_t>(network_properties.N_tar) * (network_properties.N_tar - 1) / 2) {
					if (!network_properties.flags.yes) {
						printf("\nYou have requested too many comparisons in success ratio algorithm.  Set to max permitted [y/N]? ");
						fflush(stdout);
						char response = getchar();
						getchar();
						if (response != 'y')
							exit(EXIT_FAILURE);
					}
					network_properties.N_sr = static_cast<int64_t>(network_properties.N_tar) * (network_properties.N_tar - 1) / 2;
				}
			}
		} else if (network_properties.dim == 1)
			throw CausetException("1+1 not supported for universe causet!\n");

		//Prepare to benchmark algorithms
		if (network_properties.flags.bench) {
			network_properties.flags.verbose = false;
			network_properties.graphID = 0;
			network_properties.flags.disp_network = false;
			network_properties.flags.print_network = false;
		}

		//If no seed specified, choose random one
		if (network_properties.seed == -12345L) {
			srand(time(NULL));
			network_properties.seed = -1.0 * static_cast<long>(time(NULL));
		}

		//If graph ID specified, prepare to read graph properties
		if (network_properties.graphID != 0) {
			if (network_properties.flags.verbose && !network_properties.flags.yes) {
				printf("You have chosen to load a graph from memory.  Some parameters may be ignored as a result.  Continue [y/N]? ");
				fflush(stdout);
				char response = getchar();
				getchar();
				if (response != 'y')
					exit(EXIT_FAILURE);
			}

			//Not currently supported for 1+1 causet
			network_properties.dim = 3;
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		exit(EXIT_FAILURE);
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		exit(EXIT_FAILURE);
	}

	return network_properties;
}

//Handles all network generation and initialization procedures
bool initializeNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (DEBUG) {
		//No null pointers
		assert (network != NULL);
		assert (cp != NULL);
		assert (bm != NULL);
	}

	#ifdef _OPENMP
		printf("\t[ *** OPENMP MODULE ACTIVE *** ]\n");
	#endif
	
	bool tmp = false;
	int i;

	//Causet of our universe
	if (network->network_properties.flags.universe) {
		try {
			//First check for too many parameters
			if (network->network_properties.flags.cc.conflicts[0] > 1 || network->network_properties.flags.cc.conflicts[1] > 1 || network->network_properties.flags.cc.conflicts[2] > 1 || network->network_properties.flags.cc.conflicts[3] > 1 || network->network_properties.flags.cc.conflicts[4] > 3 || network->network_properties.flags.cc.conflicts[5] > 3 || network->network_properties.flags.cc.conflicts[6] > 3)
				throw CausetException("Causet model has been over-constrained!  Use flag --conflicts to find your error.\n");
			//Second check for too few parameters
			else if (network->network_properties.N_tar == 0 && network->network_properties.alpha == 0.0)
				throw CausetException("Causet model has been under-constrained!  Specify at least '-n', number of nodes, or '-A', alpha, to proceed.\n");
		} catch (CausetException c) {
			fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
			return false;
		} catch (std::exception e) {
			fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__,  e.what(), __LINE__);
			return false;
		}

		//Solve for Constrained Parameters
		if (network->network_properties.flags.cc.conflicts[1] == 0 && network->network_properties.flags.cc.conflicts[2] == 0 && network->network_properties.flags.cc.conflicts[3] == 0) {
			//Solve for tau0, ratio, omegaM, and omegaL
			double x = 0.5;
			if (DEBUG) {
				assert (network->network_properties.N_tar > 0);
				assert (network->network_properties.alpha > 0.0);
				assert (network->network_properties.delta > 0.0);
			}

			if (!newton(&solveTau0, &x, 10000, TOL, &network->network_properties.alpha, &network->network_properties.delta, NULL, NULL, &network->network_properties.N_tar, NULL))
				return false;
			network->network_properties.tau0 = x;
			if (DEBUG) assert (network->network_properties.tau0 > 0.0);
			
			network->network_properties.ratio = POW2(SINH(1.5 * static_cast<float>(network->network_properties.tau0), STL), EXACT);
			if (DEBUG) assert(network->network_properties.ratio > 0.0);
			network->network_properties.omegaM = 1.0 / (network->network_properties.ratio + 1.0);
			network->network_properties.omegaL = 1.0 - network->network_properties.omegaM;
		} else if (network->network_properties.flags.cc.conflicts[1] == 0 || network->network_properties.flags.cc.conflicts[2] == 0 || network->network_properties.flags.cc.conflicts[3] == 0) {
			if (network->network_properties.N_tar > 0 && network->network_properties.alpha > 0.0) {
				//Solve for delta
				network->network_properties.delta = 3.0 * network->network_properties.N_tar / (POW2(static_cast<float>(M_PI), EXACT) * POW3(static_cast<float>(network->network_properties.alpha), EXACT) * (SINH(3.0 * static_cast<float>(network->network_properties.tau0), STL) - 3.0 * network->network_properties.tau0));
				if (DEBUG) assert (network->network_properties.delta > 0.0);
			} else if (network->network_properties.N_tar == 0) {
				//Solve for N_tar
				if (DEBUG) assert (network->network_properties.alpha > 0.0);
				network->network_properties.N_tar = static_cast<int>(POW2(static_cast<float>(M_PI), EXACT) * network->network_properties.delta * POW3(static_cast<float>(network->network_properties.alpha), EXACT) * (SINH(3.0 * static_cast<float>(network->network_properties.tau0), STL) - 3.0 * network->network_properties.tau0) / 3.0);
				if (DEBUG) assert (network->network_properties.N_tar > 0);
			} else {
				//Solve for alpha
				if (DEBUG) assert (network->network_properties.N_tar > 0);
				network->network_properties.alpha = POW(3.0 * network->network_properties.N_tar / (POW2(static_cast<float>(M_PI), EXACT) * network->network_properties.delta * (SINH(3.0 * static_cast<float>(network->network_properties.tau0), STL) - 3.0 * network->network_properties.tau0)), (1.0 / 3.0), STL);
				if (DEBUG) assert (network->network_properties.alpha > 0.0);
			}
		}
		//Finally, solve for R0
		if (DEBUG) {
			assert (network->network_properties.alpha > 0.0);
			assert (network->network_properties.ratio > 0.0);
		}
		network->network_properties.R0 = network->network_properties.alpha * POW(static_cast<float>(network->network_properties.ratio), 1.0f / 3.0f, STL);
		if (DEBUG) assert (network->network_properties.R0 > 0.0);

		//Use Monte Carlo integration to find k_tar
		if (network->network_properties.k_tar == 0.0) {
			printf("\tEstimating Expected Average Degrees.....\n");
			double r0 = POW(SINH(1.5f * network->network_properties.tau0, STL), 2.0f / 3.0f, STL);

			if (network->network_properties.flags.bench) {
				for (i = 0; i < NBENCH; i++) {
					stopwatchStart(&cp->sCalcDegrees);
					integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network->network_properties.seed, 0);
					stopwatchStop(&cp->sCalcDegrees);
					bm->bCalcDegrees += cp->sCalcDegrees.elapsedTime;
					stopwatchReset(&cp->sCalcDegrees);
				}
				bm->bCalcDegrees /= NBENCH;
			}

			stopwatchStart(&cp->sCalcDegrees);
			network->network_properties.k_tar = network->network_properties.delta * POW2(POW2(static_cast<float>(network->network_properties.a), EXACT), EXACT) * integrate2D(&rescaledDegreeUniverse, 0.0, 0.0, r0, r0, network->network_properties.seed, 0) * 8.0 * M_PI / (SINH(3.0f * network->network_properties.tau0, STL) - 3.0 * network->network_properties.tau0);
			stopwatchStop(&cp->sCalcDegrees);
			printf("\t\tCompleted.\n");
		}

		//20% Buffer
		network->network_properties.edge_buffer = static_cast<int>(0.1 * network->network_properties.N_tar * network->network_properties.k_tar);

		//Check success ratio parameters if applicable
		if (network->network_properties.flags.calc_success_ratio) {
			if (network->network_properties.N_sr == -1)
				network->network_properties.N_sr = static_cast<int64_t>(network->network_properties.N_tar) * (network->network_properties.N_tar - 1) / 2;
			else if (network->network_properties.N_sr > static_cast<int64_t>(network->network_properties.N_tar) * (network->network_properties.N_tar - 1) / 2) {
				if (!network->network_properties.flags.yes) {
					printf("\nYou have requested too many comparisons in success ratio algorithm.  Set to max permitted [y/N]? ");
					fflush(stdout);
					char response = getchar();
					getchar();
					if (response != 'y')
						return false;
				}
				network->network_properties.N_sr = static_cast<int64_t>(network->network_properties.N_tar) * (network->network_properties.N_tar - 1) / 2;
			}
		}

		printf("\n");
		printf("\tParameters Constraining Universe Causal Set:\n");
		printf("\t--------------------------------------------\n");
		printf("\t > Number of Nodes:\t\t%d\n", network->network_properties.N_tar);
		printf("\t > Expected Degrees:\t\t%.6f\n", network->network_properties.k_tar);
		printf("\t > Pseudoradius:\t\t%.6f\n", network->network_properties.a);
		printf("\t > Cosmological Constant:\t%.6f\n", network->network_properties.lambda);
		printf("\t > Rescaled Age:\t\t%.6f\n", network->network_properties.tau0);
		printf("\t > Dark Energy Density:\t\t%.6f\n", network->network_properties.omegaL);
		printf("\t > Matter Density:\t\t%.6f\n", network->network_properties.omegaM);
		printf("\t > Ratio:\t\t\t%.6f\n", network->network_properties.ratio);
		printf("\t > Node Density:\t\t%.6f\n", network->network_properties.delta);
		printf("\t > Alpha:\t\t\t%.6f\n", network->network_properties.alpha);
		printf("\t > Scaling Factor:\t\t%.6f\n", network->network_properties.R0);
		fflush(stdout);
	}

	printf("\nInitializing Network...\n");
	fflush(stdout);

	//Allocate memory needed by pointers
	if (network->network_properties.flags.bench) {
		tmp = network->network_properties.flags.calc_clustering;
		network->network_properties.flags.calc_clustering = false;

		for (i = 0; i < NBENCH; i++) {
			if (!createNetwork(network->nodes, network->d_nodes, network->past_edges, network->d_past_edges, network->future_edges, network->d_future_edges, network->past_edge_row_start, network->d_past_edge_row_start, network->future_edge_row_start, network->d_future_edge_row_start, network->core_edge_exists, network->d_k_in, network->d_k_out, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sCreateNetwork, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.use_gpu, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
				return false;
				
			bm->bCreateNetwork += cp->sCreateNetwork.elapsedTime;
			destroyNetwork(network, hostMemUsed, devMemUsed);
			stopwatchReset(&cp->sCreateNetwork);
		}
		bm->bCreateNetwork /= NBENCH;
	
		if (tmp)
			network->network_properties.flags.calc_clustering = true;
	}

	tmp = network->network_properties.flags.bench;
	if (tmp)
		network->network_properties.flags.bench = false;
	if (!createNetwork(network->nodes, network->d_nodes, network->past_edges, network->d_past_edges, network->future_edges, network->d_future_edges, network->past_edge_row_start, network->d_past_edge_row_start, network->future_edge_row_start, network->d_future_edge_row_start, network->core_edge_exists, network->d_k_in, network->d_k_out, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sCreateNetwork, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.use_gpu, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
		return false;
	if (tmp)
		network->network_properties.flags.bench = true;

	if (!network->network_properties.flags.universe) {
		//Solve for eta0 using Newton-Raphson Method
		if (DEBUG) {
			assert (network->network_properties.N_tar > 0);
			assert (network->network_properties.k_tar > 0.0);
			assert (network->network_properties.dim == 1 || network->network_properties.dim == 3);
		}

		double x = 0.08;
		if (network->network_properties.dim == 1)
			x = HALF_PI - 0.0001;

		if (!newton(&solveZeta, &x, 10000, TOL, NULL, NULL, NULL, &network->network_properties.k_tar, &network->network_properties.N_tar, &network->network_properties.dim))
			return false;

		if (network->network_properties.dim == 1)
			network->network_properties.zeta = HALF_PI - x;
		else
			network->network_properties.zeta = x;
		network->network_properties.tau0 = etaToTau(HALF_PI - network->network_properties.zeta);

		if (DEBUG) {
			assert (network->network_properties.zeta > 0);
			assert (network->network_properties.zeta < HALF_PI);
		}

		printf("\tTranscendental Equation Solved:\n");
		//printf("\t\tZeta: %5.8f\n", network->network_properties.zeta);
		printf("\t\tMaximum Conformal Time: %5.8f\n", HALF_PI - network->network_properties.zeta);
		printf("\t\tMaximum Rescaled Time:  %5.8f\n", network->network_properties.tau0);
		fflush(stdout);
	} else {
		if (USE_GSL) {
			IntData idata = IntData();
			idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
			idata.upper = network->network_properties.tau0 * network->network_properties.a;
			network->network_properties.zeta = HALF_PI - integrate1D(&tauToEtaUniverse, NULL, &idata, QAGS);
			gsl_integration_workspace_free(idata.workspace);
		} else
			//Exact Solution
			network->network_properties.zeta = HALF_PI - tauToEtaUniverseExact(network->network_properties.tau0, network->network_properties.a, network->network_properties.alpha);
	}

	//Generate coordinates of spacetime nodes and then order nodes temporally using quicksort
	int low = 0;
	int high = network->network_properties.N_tar - 1;
	if (network->network_properties.flags.bench) {
		for (i = 0; i < NBENCH; i++) {
			if (!generateNodes(network->nodes, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.seed, cp->sGenerateNodes, network->network_properties.flags.use_gpu, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
				
			//Quicksort
			stopwatchStart(&cp->sQuicksort);
			quicksort(network->nodes, low, high);
			stopwatchStop(&cp->sQuicksort);

			bm->bGenerateNodes += cp->sGenerateNodes.elapsedTime;
			bm->bQuicksort += cp->sQuicksort.elapsedTime;
		
			stopwatchReset(&cp->sGenerateNodes);
			stopwatchReset(&cp->sQuicksort);
		}
		
		bm->bGenerateNodes /= NBENCH;
		bm->bQuicksort /= NBENCH;
	}

	if (tmp)
		network->network_properties.flags.bench = false;
	if (!generateNodes(network->nodes, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.seed, cp->sGenerateNodes, network->network_properties.flags.use_gpu, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
		return false;
	if (tmp)
		network->network_properties.flags.bench = true;

	//Quicksort
	stopwatchStart(&cp->sQuicksort);
	quicksort(network->nodes, low, high);
	stopwatchStop(&cp->sQuicksort);
			
	printf("\tQuick Sort Successfully Performed.\n");
	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sQuicksort.elapsedTime);
	fflush(stdout);

	//Identify edges as points connected by timelike intervals
	if (network->network_properties.flags.bench) {
		for (i = 0; i < NBENCH; i++) {
			if (network->network_properties.flags.use_gpu) {
				if (!linkNodesGPU(network->nodes, network->d_nodes, network->past_edges, network->d_past_edges, network->future_edges, network->d_future_edges, network->past_edge_row_start, network->d_past_edge_row_start, network->future_edge_row_start, network->d_future_edge_row_start, network->core_edge_exists, network->d_k_in, network->d_k_out, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
					return false;

				bm->bLinkNodesGPU += cp->sLinkNodesGPU.elapsedTime;
				stopwatchReset(&cp->sLinkNodes);
			} else {
				if (!linkNodes(network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
					return false;
				
				bm->bLinkNodes += cp->sLinkNodes.elapsedTime;
				stopwatchReset(&cp->sLinkNodes);
			}
		}

		if (network->network_properties.flags.use_gpu)
			bm->bLinkNodesGPU /= NBENCH;
		else
			bm->bLinkNodes /= NBENCH;
	}

	if (tmp)
		network->network_properties.flags.bench = false;
	if (network->network_properties.flags.use_gpu) {
		if (!linkNodesGPU(network->nodes, network->d_nodes, network->past_edges, network->d_past_edges, network->future_edges, network->d_future_edges, network->past_edge_row_start, network->d_past_edge_row_start, network->future_edge_row_start, network->d_future_edge_row_start, network->core_edge_exists, network->d_k_in, network->d_k_out, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
	} else if (!linkNodes(network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
	if (tmp)
		network->network_properties.flags.bench = true;

	printf("Task Completed.\n");
	fflush(stdout);
	return true;
}

bool measureNetworkObservables(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (DEBUG) {
		//No null pointers
		assert (network != NULL);
		assert (cp != NULL);
		assert (bm != NULL);
	}

	if (!network->network_properties.flags.calc_clustering && !network->network_properties.flags.calc_success_ratio)
		return true;

	printf("\nCalculating Network Observables...\n");
	fflush(stdout);

	int i;
	bool tmp;

	if (network->network_properties.flags.calc_clustering) {
		if (network->network_properties.flags.bench) {
			for (i = 0; i < NBENCH; i++) {
				if (!measureClustering(network->network_observables.clustering, network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_observables.average_clustering, network->network_properties.N_tar, network->network_properties.N_deg2, network->network_properties.core_edge_fraction, cp->sMeasureClustering, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.calc_autocorr, network->network_properties.flags.verbose, network->network_properties.flags.bench))
					return false;
				bm->bMeasureClustering += cp->sMeasureClustering.elapsedTime;
				stopwatchReset(&cp->sMeasureClustering);
			}
			bm->bMeasureClustering /= NBENCH;
		}
		
		tmp = network->network_properties.flags.bench;
		if (tmp)
			network->network_properties.flags.bench = false;
		if (!measureClustering(network->network_observables.clustering, network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_observables.average_clustering, network->network_properties.N_tar, network->network_properties.N_deg2, network->network_properties.core_edge_fraction, cp->sMeasureClustering, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.calc_autocorr, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
		if (tmp)
			network->network_properties.flags.bench = true;
	}

	if (network->network_properties.flags.calc_success_ratio) {
		if (network->network_properties.flags.bench) {
			for (i = 0; i < NBENCH; i++) {
				if (!measureSuccessRatio(network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_observables.success_ratio, network->network_properties.N_tar, network->network_properties.N_sr, network->network_properties.core_edge_fraction, cp->sMeasureSuccessRatio, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.verbose, network->network_properties.flags.bench))
					return false;
				bm->bMeasureSuccessRatio += cp->sMeasureSuccessRatio.elapsedTime;
				stopwatchReset(&cp->sMeasureSuccessRatio);
			}
			bm->bMeasureSuccessRatio /= NBENCH;
		}

		tmp = network->network_properties.flags.bench;
		if (tmp)
			network->network_properties.flags.bench = false;
		if (!measureSuccessRatio(network->nodes, network->past_edges, network->future_edges, network->past_edge_row_start, network->future_edge_row_start, network->core_edge_exists, network->network_observables.success_ratio, network->network_properties.N_tar, network->network_properties.N_sr, network->network_properties.core_edge_fraction, cp->sMeasureSuccessRatio, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;
		if (tmp)
			network->network_properties.flags.bench = true;
	}

	printf("Task Completed.\n");
	fflush(stdout);

	return true;
}

//Plot using OpenGL
bool displayNetwork(const Node &nodes, const int * const future_edges, int argc, char **argv)
{
	if (DEBUG)
		assert (future_edges != NULL);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(900, 900);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Causets");
	glutDisplayFunc(display);
	glOrtho(0.0f, 0.01f, 0.0f, 6.3f, -1.0f, 1.0f);
	glutMainLoop();

	return true;
}

//Display Function for OpenGL Instructions
void display()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_LINES);
		glColor3f(0.0f, 0.0f, 0.0f);
		//Draw Lines
	glEnd();

	glLoadIdentity();
	glutSwapBuffers();
}

//Load Network Data from Existing File
//O(k*N^2)
//Reads the following files:
//	-Primary simulation output file (./dat/*.cset.out)
//	-Node position data		(./dat/pos/*.cset.pos.dat)
//	-Edge data			(./dat/edg/*.cset.edg.dat)
bool loadNetwork(Network * const network, CausetPerformance * const cp, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	/*if (DEBUG) {
		assert (network != NULL);
		assert (cp != NULL);
		assert (network->network_properties.graphID != 0);
		assert (!network->network_properties.flags.bench);
	}

	//Not currently supported (will be implemented later)
	assert (!network->network_properties.flags.universe);
	//Prevents this function from executing until it is updated
	assert (network->network_properties.N_tar < 0);

	printf("Loading Graph from File.....\n");
	fflush(stdout);

	int idx1 = 0, idx2 = 0, idx3 = 0;
	int i, j, k;
	try {
		//Read Data Keys
		printf("\tReading Data Keys.\n");
		fflush(stdout);
		std::ifstream dataStream;
		std::string line;
		char *pch, *filename;
		filename = NULL;
	
		dataStream.open("./dat/data_keys.key");
		if (dataStream.is_open()) {
			while (getline(dataStream, line)) {
				pch = strtok((char*)line.c_str(), "\t");
				if (atoi(pch) == network->network_properties.graphID) {
					filename = strtok(NULL, "\t");
					break;
				}
			}
			dataStream.close();
			if (filename == NULL)
				throw CausetException("Failed to locate graph file!\n");
		} else
			throw CausetException("Failed to open 'data_keys.key' file!\n");
			
		//Read Main Data File
		printf("\tReading Simulation Parameters.\n");
		fflush(stdout);
		std::stringstream fullname;
		fullname << "./dat/" << filename << ".cset.out";
		dataStream.open(fullname.str().c_str());
		if (dataStream.is_open()) {
			//Read N_tar
			for (i = 0; i < 7; i++)
				getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 3; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.N_tar = atoi(pch);
			if (network->network_properties.N_tar <= 0)
				throw CausetException("Invalid value for number of nodes!\n");
			printf("\t\tN_tar:\t%d\n", network->network_properties.N_tar);
			fflush(stdout);

			//Read k_tar
			getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 5; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.k_tar = atof(pch);
			if (network->network_properties.k_tar <= 0.0)
				throw CausetException("Invalid value for expected average degreed!\n");
			printf("\t\tk_tar:\t%f\n", network->network_properties.k_tar);
			fflush(stdout);

			//Read a
			getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 2; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.a = atof(pch);
			if (network->network_properties.a <= 0.0)
				throw CausetException("Invalid value for pseudoradius!\n");
			printf("\t\ta:\t%f\n", network->network_properties.a);
			fflush(stdout);

			//Read eta_0 (save as zeta)
			for (i = 0; i < 6; i++)
				getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 4; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.zeta = HALF_PI - atof(pch);
			if (network->network_properties.zeta <= 0.0 || network->network_properties.zeta >= HALF_PI)
				throw CausetException("Invalid value for eta0!\n");
			printf("\t\teta0:\t%f\n", HALF_PI - network->network_properties.zeta);
			fflush(stdout);

			dataStream.close();
		} else
			throw CausetException("Failed to open simulation parameters file!\n");

		//if (!createNetwork(network, cp, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed))
		//	return false;

		//Read node positions
		printf("\tReading Node Position Data.\n");
		fflush(stdout);
		std::stringstream dataname;
		dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
		dataStream.open(dataname.str().c_str());
		if (dataStream.is_open()) {
			for (i = 0; i < network->network_properties.N_tar; i++) {
				getline(dataStream, line);
				network->nodes[i] = Node();
				network->nodes[i].tau = 0.0;
				network->nodes[i].eta = 0.0;
				network->nodes[i].theta = atof(strtok(NULL, " "));
				network->nodes[i].phi = atof(strtok(NULL, " "));
				network->nodes[i].chi = atof(strtok(NULL, " "));
				
				if (network->nodes[i].tau <= 0.0)
					throw CausetException("Invalid value parsed for t in node position file!\n");
				if (network->nodes[i].theta <= 0.0 || network->nodes[i].theta >= TWO_PI)
					throw CausetException("Invalid value parsed for theta in node position file!\n");
				if (network->nodes[i].phi <= 0.0 || network->nodes[i].phi >= M_PI)
					throw CausetException("Invalid value parsed for phi in node position file!\n");
				if (network->nodes[i].chi <= 0.0 || network->nodes[i].chi >= M_PI)
					throw CausetException("Invalid value parsed for chi in node position file!\n");
			}
			dataStream.close();
		} else
			throw CausetException("Failed to open node position file!\n");

		//for (int i = 0; i < network->network_properties.N_tar; i++) {
		//	printf("%f\t%f\t%f\t%f\n", network->nodes[i].t, network->nodes[i].theta, network->nodes[i].phi, network->nodes[i].chi);
		//	fflush(stdout);

		printf("\tReading Edge Data.\n");
		fflush(stdout);	
		dataname.str("");
		dataname.clear();
		dataname << "./dat/edg/" << network->network_properties.graphID << ".cset.edg.dat";
		dataStream.open(dataname.str().c_str());
		int diff;
		if (dataStream.is_open()) {
			int n1, n2;
			network->future_edge_row_start[0] = 0;
			while (getline(dataStream, line)) {
				//Read pairs of connected nodes (past, future)
				n1 = atoi(strtok((char*)line.c_str(), " "));
				n2 = atoi(strtok(NULL, " "));
				
				if (n1 < 0 || n2 < 0 || n1 >= network->network_properties.N_tar || n2 >= network->network_properties.N_tar || n2 <= n1)
					throw CausetException("Corrupt edge list file!\n");

				//Check if a node is skipped (k_i = 0)
				diff = n1 - idx1;

				//This should be a CausetException
				assert (diff >= 0);

				//Multiple nodes skipped
				if (diff > 1)
					for (i = 0; i < diff - 1; i++)
						network->future_edge_row_start[++idx1] = -1;

				//At least one node skipped
				if (diff > 0)
					network->future_edge_row_start[++idx1] = idx2;

				network->nodes[idx1].k_out++;
				network->future_edges[idx2++] = n2;
			}

			//Assign pointer values for all latest disconnected nodes
			for (i = idx1 + 1; i < network->network_properties.N_tar; i++)
				network->future_edge_row_start[i] = -1;
			dataStream.close();
		} else
			throw CausetException("Failed to open edge list file!\n");
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	//Assign past node list and pointer values
	network->past_edge_row_start[0] = -1;
	for (i = 0; i < network->network_properties.N_tar; i++) {
		network->past_edge_row_start[i] = idx3;
		for (j = 0; j < i; j++) {
			if (network->future_edge_row_start[j] == -1) 
				continue;

			for (k = 0; k < network->nodes[j].k_out; k++)
				if (i == network->future_edges[network->future_edge_row_start[j]+k])
					network->past_edges[idx3++] = j;
		}

		network->nodes[i].k_in = idx3 - network->past_edge_row_start[i];
		if (DEBUG) assert(network->nodes[i].k_in >= 0);

		if (network->past_edge_row_start[i] == idx3)
			network->past_edge_row_start[i] = -1;
	}

	//compareAdjacencyLists(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);
	//compareAdjacencyListIndices(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);

	//Adjacency Matrix
	printf("\tPopulating Adjacency Matrix.\n");
	fflush(stdout);
	int core_limit = (int)(network->network_properties.core_edge_fraction * network->network_properties.N_tar);
	for (i = 0; i < core_limit; i++)
		for (j = 0; j < core_limit; j++)
			network->core_edge_exists[(i*core_limit)+j] = false;

	idx1 = 0, idx2 = 0;
	while (idx1 < core_limit) {
		if (network->future_edge_row_start[idx1] != -1) {
			for (i = 0; i < network->nodes[idx1].k_out; i++) {
				idx2 = network->future_edges[network->future_edge_row_start[idx1]+i];
				if (idx2 < core_limit) {
					network->core_edge_exists[(core_limit*idx1)+idx2] = true;
					network->core_edge_exists[(core_limit*idx2)+idx1] = true;
				}
			}
		}
		idx1++;
	}

	//Properties of Giant Connected Component (GCC)
	printf("\tResulting Network:\n");
	fflush(stdout);
	for (i = 0; i < network->network_properties.N_tar; i++) {
		if (network->nodes[i].k_in > 0 || network->nodes[i].k_out > 0) {
			network->network_properties.N_res++;
			network->network_properties.k_res+= network->nodes[i].k_in + network->nodes[i].k_out;

			if (network->nodes[i].k_in + network->nodes[i].k_out > 1)
				network->network_properties.N_deg2++;
		}
	}
	network->network_properties.k_res /= network->network_properties.N_res;

	printf("\t\tN_res:  %d\n", network->network_properties.N_res);
	printf("\t\tk_res:  %f\n", network->network_properties.k_res);
	printf("\t\tN_deg2: %d\n", network->network_properties.N_deg2);

	printf("Task Completed.\n");
	fflush(stdout);

	return true;*/

	return false;
}

//Print to File
bool printNetwork(Network &network, const CausetPerformance &cp, const long &init_seed, const int &gpuID)
{
	if (!network.network_properties.flags.print_network)
		return false;

	printf("Printing Results to File...\n");
	fflush(stdout);

	int i, j, k;
	try {
		//Confirm directory structure exists
		mkdir("./dat", 777);
		mkdir("./dat/pos", 777);
		mkdir("./dat/edg", 777);
		mkdir("./dat/dst", 777);
		mkdir("./dat/idd", 777);
		mkdir("./dat/odd", 777);
		mkdir("./dat/cls", 777);
		mkdir("./dat/cdk", 777);

		std::ofstream outputStream;
		std::stringstream sstm;

		if (network.network_properties.flags.use_gpu)
			sstm << "Dev" << gpuID << "_";
		else
			sstm << "CPU_";
		if (network.network_properties.flags.universe) {
			sstm << "U_";
			sstm << network.network_properties.tau0 << "_";
			sstm << network.network_properties.alpha << "_";
			sstm << network.network_properties.a << "_";
			sstm << network.network_properties.delta << "_";
		} else {
			sstm << network.network_properties.N_tar << "_";
			sstm << network.network_properties.k_tar << "_";
			sstm << network.network_properties.a << "_";
			sstm << network.network_properties.dim << "_";
		}
		sstm << init_seed;
		std::string filename = sstm.str();

		sstm.str("");
		sstm.clear();
		sstm << "./dat/" << filename << ".cset.out";
		outputStream.open(sstm.str().c_str());
		if (!outputStream.is_open())
			throw CausetException("Failed to open graph file!\n");
		outputStream << "Causet Simulation\n";
		if (network.network_properties.graphID == 0)
			network.network_properties.graphID = (int)time(NULL);
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

		if (network.network_properties.flags.universe) {
			outputStream << "\nCauset Initial Parameters:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			outputStream << "Target Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Target Degrees (k_tar)\t\t\t" << network.network_properties.k_tar << std::endl;
			outputStream << "Pseudoradius (a)\t\t\t" << network.network_properties.a << std::endl;
			outputStream << "Cosmological Constant (lambda)\t\t" << network.network_properties.lambda << std::endl;
			outputStream << "Rescaled Age (tau0)\t\t\t" << network.network_properties.tau0 << std::endl;
			outputStream << "Dark Energy Density (omegaL)\t\t" << network.network_properties.omegaL << std::endl;
			outputStream << "Matter Density (omegaM)\t\t\t" << network.network_properties.omegaM << std::endl;
			outputStream << "Ratio (ratio)\t\t\t\t" << network.network_properties.ratio << std::endl;
			outputStream << "Node Density (delta)\t\t\t" << network.network_properties.delta << std::endl;
			outputStream << "Alpha (alpha)\t\t\t\t" << network.network_properties.alpha << std::endl;
			outputStream << "Scaling Factor (R0)\t\t\t" << network.network_properties.R0 << std::endl;

			outputStream << "\nCauset Resulting Parameters:" << std::endl;
			outputStream << "----------------------------" << std::endl;
			outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_properties.N_res << std::endl;
			outputStream << "Resulting Average Degrees (k_res)\t" << network.network_properties.k_res << std::endl;
		} else {
			outputStream << "\nCauset Input Parameters:" << std::endl;
			outputStream << "------------------------" << std::endl;
			outputStream << "Target Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
			outputStream << "Target Expected Average Degrees (k_tar)\t" << network.network_properties.k_tar << std::endl;
			outputStream << "Pseudoradius (a)\t\t\t" << network.network_properties.a << std::endl;

			outputStream << "\nCauset Calculated Values:" << std::endl;
			outputStream << "--------------------------" << std::endl;
			outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_properties.N_res << std::endl;
			outputStream << "Resulting Average Degrees (k_res)\t" << network.network_properties.k_res << std::endl;
			outputStream << "Maximum Conformal Time (eta_0)\t\t" << (HALF_PI - network.network_properties.zeta) << std::endl;
			outputStream << "Maximum Rescaled Time (tau_0)  \t\t" << network.network_properties.tau0 << std::endl;
		}

		if (network.network_properties.flags.calc_clustering)
			outputStream << "Average Clustering\t\t\t" << network.network_observables.average_clustering << std::endl;
		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "Success Ratio\t\t\t\t" << network.network_observables.success_ratio << std::endl;

		outputStream << "\nNetwork Analysis Results:" << std::endl;
		outputStream << "-------------------------" << std::endl;
		outputStream << "Node Position Data:\t\t\t" << "pos/" << network.network_properties.graphID << ".cset.pos.dat" << std::endl;
		outputStream << "Node Edge Data:\t\t\t\t" << "edg/" << network.network_properties.graphID << ".cset.edg.dat" << std::endl;
		outputStream << "Degree Distribution Data:\t\t" << "dst/" << network.network_properties.graphID << ".cset.dst.dat" << std::endl;
		outputStream << "In-Degree Distribution Data:\t\t" << "idd/" << network.network_properties.graphID << ".cset.idd.dat" << std::endl;
		outputStream << "Out-Degree Distribution Data:\t\t" << "odd/" << network.network_properties.graphID << ".cset.odd.dat" << std::endl;

		if (network.network_properties.flags.calc_clustering) {
			outputStream << "Clustering Coefficient Data:\t" << "cls/" << network.network_properties.graphID << ".cset.cls.dat" << std::endl;
			outputStream << "Clustering by Degree Data:\t" << "cdk/" << network.network_properties.graphID << ".cset.cdk.dat" << std::endl;
		}

		outputStream << "\nAlgorithmic Performance:" << std::endl;
		outputStream << "--------------------------" << std::endl;
		outputStream << "calcDegrees:         " << cp.sCalcDegrees.elapsedTime << " sec" << std::endl;
		outputStream << "createNetwork:       " << cp.sCreateNetwork.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.use_gpu)
			outputStream << "generateNodesGPU:    " << cp.sGenerateNodesGPU.elapsedTime << " sec" << std::endl;
		else
			outputStream << "generateNodes:       " << cp.sGenerateNodes.elapsedTime << " sec" << std::endl;
		outputStream << "quicksort:           " << cp.sQuicksort.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.use_gpu)
			outputStream << "linkNodesGPU:        " << cp.sLinkNodesGPU.elapsedTime << " sec" << std::endl;
		else
			outputStream << "linkNodes:           " << cp.sLinkNodes.elapsedTime << " sec" << std::endl;

		if (network.network_properties.flags.calc_clustering)
			outputStream << "measureClustering:   " << cp.sMeasureClustering.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "measureSuccessRatio: " << cp.sMeasureSuccessRatio.elapsedTime << " sec" << std::endl;

		outputStream << "Total Time:        " << cp.sCauset.elapsedTime << " sec" << std::endl;

		outputStream.flush();
		outputStream.close();

		std::ofstream mapStream;
		mapStream.open("./dat/data_keys.key", std::ios::app);
		if (!mapStream.is_open())
			throw CausetException("Failed to open 'dat/data_keys.key' file!\n");
		mapStream << network.network_properties.graphID << "\t" << filename << std::endl;
		mapStream.close();

		std::ofstream dataStream;

		sstm.str("");
		sstm.clear();
		sstm << "./dat/pos/" << network.network_properties.graphID << ".cset.pos.dat";
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open node position file!\n");
		for (i = 0; i < network.network_properties.N_tar; i++) {
			if (network.network_properties.flags.universe)
				dataStream << network.nodes.tau[i];
			else
				dataStream << network.nodes.sc[i].w;
			dataStream << " " << network.nodes.sc[i].x;
			if (network.network_properties.dim == 3)
				dataStream << " " << network.nodes.sc[i].y << " " << network.nodes.sc[i].z;
			dataStream << std::endl;
		}
		dataStream.flush();
		dataStream.close();

		int idx = 0;
		sstm.str("");
		sstm.clear();
		sstm << "./dat/edg/" << network.network_properties.graphID << ".cset.edg.dat";
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open edge list file!\n");
		for (i = 0; i < network.network_properties.N_tar; i++) {
			for (j = 0; j < network.nodes.k_out[i]; j++)
				dataStream << i << " " << network.future_edges[idx + j] << std::endl;
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
		int k_max = network.network_properties.N_res - 1;
		for (k = 1; k <= k_max; k++) {
			idx = 0;
			for (i = 0; i < network.network_properties.N_tar; i++)
				if (network.nodes.k_in[i] + network.nodes.k_out[i] == k)
					idx++;
			if (idx > 0)
				dataStream << k << " " << idx << std::endl;
		}
		dataStream.flush();
		dataStream.close();

		sstm.str("");
		sstm.clear();
		sstm << "./dat/idd/" << network.network_properties.graphID << ".cset.idd.dat";
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open in-degree distribution file!\n");
		for (k = 1; k <= k_max; k++) {
			idx = 0;
			for (i = 0; i < network.network_properties.N_tar; i++)
				if (network.nodes.k_in[i] == k)
					idx++;
			if (idx > 0)
				dataStream << k << " " << idx << std::endl;
		}
		dataStream.flush();
		dataStream.close();

		sstm.str("");
		sstm.clear();
		sstm << "./dat/odd/" << network.network_properties.graphID << ".cset.odd.dat";
		dataStream.open(sstm.str().c_str());
		if (!dataStream.is_open())
			throw CausetException("Failed to open out-degree distribution file!\n");
		for (k = 1; k <= k_max; k++) {
			idx = 0;
			for (i = 0; i < network.network_properties.N_tar; i++)
				if (network.nodes.k_out[i] == k)
					idx++;
			if (idx > 0)
				dataStream << k << " " << idx << std::endl;
		}
		dataStream.flush();
		dataStream.close();

		if (network.network_properties.flags.calc_clustering) {
			sstm.str("");
			sstm.clear();
			sstm << "./dat/cls/" << network.network_properties.graphID << ".cset.cls.dat";
			dataStream.open(sstm.str().c_str());
			if (!dataStream.is_open())
				throw CausetException("Failed to open clustering coefficient file!\n");
			for (i = 0; i < network.network_properties.N_tar; i++)
				dataStream << network.network_observables.clustering[i] << std::endl;
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
				dataStream << i << " " << (cdk / ndk) << std::endl;
			}
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

bool printBenchmark(const Benchmark &bm, const CausetFlags &cf)
{
	//Print to File
	FILE *f;

	try {
		f = fopen("bench.log", "w");
		if (f == NULL)
			throw CausetException("Failed to open file 'bench.log'\n");
		fprintf(f, "Causet Simulation Benchmark Results\n");
		fprintf(f, "-----------------------------------\n");
		fprintf(f, "Times Averaged over %d Runs:\n", NBENCH);
		fprintf(f, "\tcalcDegrees:\t\t%5.6f sec\n", bm.bCalcDegrees);
		fprintf(f, "\tcreateNetwork:\t\t%5.6f sec\n", bm.bCreateNetwork);
		if (cf.use_gpu)
			fprintf(f, "\tgenerateNodesGPU:\t%5.6f sec\n", bm.bGenerateNodesGPU);
		else
			fprintf(f, "\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
		fprintf(f, "\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
		if (cf.use_gpu)
			fprintf(f, "\tlinkNodesGPU:\t\t%5.6f sec\n", bm.bLinkNodesGPU);
		else
			fprintf(f, "\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
		if (cf.calc_clustering)
			fprintf(f, "\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);
		if (cf.calc_success_ratio)
			fprintf(f, "\tmeasureSuccessRatio:\t%5.6f sec\n", bm.bMeasureSuccessRatio);

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
	if (cf.use_gpu)
		printf("\tgenerateNodesGPU:\t%5.6f sec\n", bm.bGenerateNodesGPU);
	else
		printf("\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
	printf("\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
	if (cf.use_gpu)
		printf("\tlinkNodesGPU:\t\t%5.6f sec\n", bm.bLinkNodesGPU);
	else
		printf("\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
	if (cf.calc_clustering)
		printf("\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);
	if (cf.calc_success_ratio)
		printf("\tmeasureSuccessRatio:\t%5.6f sec\n", bm.bMeasureSuccessRatio);
	printf("\n");
	fflush(stdout);

	return true;
}

//Free Memory
void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed)
{
	free(network->nodes.sc);
	network->nodes.sc = NULL;
	hostMemUsed -= sizeof(float4) * network->network_properties.N_tar;

	free(network->nodes.tau);
	network->nodes.tau = NULL;
	hostMemUsed -= sizeof(float) * network->network_properties.N_tar;

	free(network->nodes.k_in);
	network->nodes.k_in = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->nodes.k_out);
	network->nodes.k_out = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->past_edges);
	network->past_edges = NULL;
	hostMemUsed -= sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

	free(network->future_edges);
	network->future_edges = NULL;
	hostMemUsed -= sizeof(int) * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

	free(network->past_edge_row_start);
	network->past_edge_row_start = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->future_edge_row_start);
	network->future_edge_row_start = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->core_edge_exists);
	network->core_edge_exists = NULL;
	hostMemUsed -= sizeof(bool) * POW2(network->network_properties.core_edge_fraction * network->network_properties.N_tar, EXACT);

	if (network->network_properties.flags.calc_clustering) {
		free(network->network_observables.clustering);
		network->network_observables.clustering = NULL;
		hostMemUsed -= sizeof(float) * network->network_properties.N_deg2;
	}
}
