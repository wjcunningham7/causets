#include "NetworkCreator.h"
#include "Measurements.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
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
	else if (network.network_properties.graphID != 0 && !loadNetwork(&network, &cp, &bm, resources.hostMemUsed, resources.maxHostMemUsed, resources.devMemUsed, resources.maxDevMemUsed)) goto CausetExit;

	if (!measureNetworkObservables(&network, &cp, &bm, resources.hostMemUsed, resources.maxHostMemUsed, resources.devMemUsed, resources.maxDevMemUsed)) goto CausetExit;

	if (network.network_properties.flags.bench && !printBenchmark(bm, network.network_properties.flags)) goto CausetExit;
	if (network.network_properties.flags.disp_network && !displayNetwork(network.nodes, network.edges.future_edges, argc, argv)) goto CausetExit;
	if (!network.network_properties.flags.bench) printMemUsed(NULL, resources.maxHostMemUsed, resources.maxDevMemUsed);
	if (network.network_properties.flags.print_network && !printNetwork(network, cp, init_seed, resources.gpuID)) goto CausetExit;

	destroyNetwork(&network, resources.hostMemUsed, resources.devMemUsed);
	if (network.network_properties.flags.use_gpu) cuCtxDetach(resources.cuContext);

	success = !resources.hostMemUsed && !resources.devMemUsed;
	if (!success)
		printf("WARNING: Memory leak detected!\n");

	CausetExit:
	stopwatchStop(&cp.sCauset);
	shrQAFinish(argc, (const char**)argv, success ? QA_PASSED : QA_FAILED);
	printf("Time: %5.6f sec\n", cp.sCauset.elapsedTime);
	printf("PROGRAM COMPLETED\n");
	fflush(stdout);
}

//Parse Command Line Arguments
static NetworkProperties parseArgs(int argc, char **argv)
{
	NetworkProperties network_properties = NetworkProperties();

	//Initialize conflict array to zeros (no conflicts)
	for (int i = 0; i < 7; i++)
		network_properties.flags.cc.conflicts[i] = 0;

	int c, longIndex;
	//Single-character options
	static const char *optString = ":m:n:k:d:s:a:c:g:t:A:D:o:r:l:uvyCSGh";
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
		{ "components", no_argument,		NULL, 'G' },
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
				if (strcmp(optarg, "d"))
					network_properties.manifold = DE_SITTER;
				else if (strcmp(optarg, "h"))
					network_properties.manifold = HYPERBOLIC;
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

				network_properties.flags.cc.conflicts[2]++;
				network_properties.flags.cc.conflicts[3]++;
				network_properties.flags.cc.conflicts[6]++;
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
				network_properties.flags.calc_components = true;
				network_properties.N_sr = atof(optarg);
				if (network_properties.N_sr <= 0.0 && network_properties.N_sr > 1.0)
					throw CausetException("Invalid argument for 'success' parameter!\n");
				break;
			case 'G':	//Flag for Finding (Giant) Connected Component
				network_properties.flags.calc_components = true;
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

				network_properties.ratio = POW2(SINH(1.5f * network_properties.tau0, STL), EXACT);
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

				network_properties.a = SQRT(3.0f / network_properties.lambda, STL);

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
				printf("  -G, --components\tIdentify Giant Component\n");
				printf("  -g, --graph\t\tGraph ID\t\t\tCheck dat/*.cset.out files\n");
				printf("  -h, --help\t\tDisplay this menu\n");
				printf("  -k, --degrees\t\tExpected Average Degrees\t10-100\n");
				printf("  -l, --lambda\t\tCosmological Constant\t\t3.0\n");
				printf("  -m, --manifold\tManifold\t\t\t[d]e sitter, [h]yperbolic\n");
				printf("  -n, --nodes\t\tNumber of Nodes\t\t\t100-100000\n");
				printf("  -o, --energy\t\tDark Energy Density\t\t0.73\n");
				printf("  -r, --ratio\t\tEnergy to Matter Ratio\t\t2.7\n");
				printf("  -S, --success\t\tCalculate Success Ratio\t\t0.5\n");
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
static bool initializeNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
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
	
	int nb = static_cast<int>(network->network_properties.flags.bench) * NBENCH;
	int i;

	if (!initVars(&network->network_properties, cp, bm, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed))
		return false;

	printf("\nInitializing Network...\n");
	fflush(stdout);

	//Allocate memory needed by pointers
	for (i = 0; i <= nb; i++) {
		if (!createNetwork(network->nodes, network->edges, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.manifold, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sCreateNetwork, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.use_gpu, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;
		if (!nb)
			destroyNetwork(network, hostMemUsed, devMemUsed);
	}

	if (!nb)
		bm->bCreateNetwork = cp->sCreateNetwork.elapsedTime / NBENCH;

	if (!solveMaxTime(network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.flags.universe))
		return false;

	//Generate coordinates of spacetime nodes and then order nodes temporally using quicksort
	int low = 0;
	int high = network->network_properties.N_tar - 1;

	for (i = 0; i <= nb; i++) {
		if (!generateNodes(network->nodes, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.seed, cp->sGenerateNodes, network->network_properties.flags.use_gpu, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
			return false;

		//Quicksort
		stopwatchStart(&cp->sQuicksort);
		quicksort(network->nodes, low, high);
		stopwatchStop(&cp->sQuicksort);
	}

	if (nb)
		printf("\tQuick Sort Successfully Performed.\n");
	else {
		bm->bGenerateNodes = cp->sGenerateNodes.elapsedTime / NBENCH;
		bm->bQuicksort = cp->sQuicksort.elapsedTime / NBENCH;
	}

	if (network->network_properties.flags.verbose)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sQuicksort.elapsedTime);
	fflush(stdout);

	//Identify edges as points connected by timelike intervals
	for (i = 0; i <= nb; i++) {
		if (network->network_properties.flags.use_gpu) {
			if (!linkNodesGPU(network->nodes, network->edges, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.a, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		} else {
			if (!linkNodes(network->nodes, network->edges, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.N_res, network->network_properties.k_res, network->network_properties.N_deg2, network->network_properties.dim, network->network_properties.manifold, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sLinkNodes, network->network_properties.flags.universe, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}
	}

	if (!nb) {
		if (network->network_properties.flags.use_gpu)
			bm->bLinkNodesGPU = cp->sLinkNodesGPU.elapsedTime / NBENCH;
		else
			bm->bLinkNodes = cp->sLinkNodes.elapsedTime / NBENCH;
	}

	printf("Task Completed.\n");
	fflush(stdout);
	return true;
}

static bool measureNetworkObservables(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (DEBUG) {
		//No null pointers
		assert (network != NULL);
		assert (cp != NULL);
		assert (bm != NULL);
	}

	if (!network->network_properties.flags.calc_clustering && !network->network_properties.flags.calc_components && !network->network_properties.flags.calc_success_ratio)
		return true;

	printf("\nCalculating Network Observables...\n");
	fflush(stdout);

	int nb = static_cast<int>(network->network_properties.flags.bench) * NBENCH;
	int i;

	if (network->network_properties.flags.calc_clustering) {
		for (i = 0; i <= nb; i++) {
			if (!measureClustering(network->network_observables.clustering, network->nodes, network->edges, network->core_edge_exists, network->network_observables.average_clustering, network->network_properties.N_tar, network->network_properties.N_deg2, network->network_properties.core_edge_fraction, cp->sMeasureClustering, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.calc_autocorr, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}

		if (!nb)
			bm->bMeasureClustering = cp->sMeasureClustering.elapsedTime / NBENCH;
	}

	if (network->network_properties.flags.calc_components) {
		for (i = 0; i <= nb; i++) {
			if (!measureConnectedComponents(network->nodes, network->edges, network->network_properties.N_tar, network->network_properties.N_cc, network->network_properties.N_gcc, cp->sMeasureConnectedComponents, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}

		if (!nb)
			bm->bMeasureConnectedComponents = cp->sMeasureConnectedComponents.elapsedTime / NBENCH;
	}


	if (network->network_properties.flags.calc_success_ratio) {
		for (i = 0; i <= nb; i++) {
			if (!measureSuccessRatio(network->nodes, network->edges, network->core_edge_exists, network->network_observables.success_ratio, network->network_properties.N_tar, network->network_properties.N_sr, network->network_properties.a, network->network_properties.alpha, network->network_properties.core_edge_fraction, cp->sMeasureSuccessRatio, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.verbose, network->network_properties.flags.bench))
				return false;
		}

		if (!nb)
			bm->bMeasureSuccessRatio = cp->sMeasureSuccessRatio.elapsedTime / NBENCH;
	}

	printf("Task Completed.\n");
	fflush(stdout);

	return true;
}

//Plot using OpenGL
static bool displayNetwork(const Node &nodes, const int * const future_edges, int argc, char **argv)
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
//O(xxx)
//Reads the following files:
//	-Node position data		(./dat/pos/*.cset.pos.dat)
//	-Edge data			(./dat/edg/*.cset.edg.dat)
static bool loadNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed)
{
	if (DEBUG) {
		//No Null Pointers
		assert (network != NULL);
		assert (cp != NULL);

		//Values in Correct Ranges (add this)

		//Logical Conditions
		assert (network->network_properties.graphID != 0);
		assert (!network->network_properties.flags.bench);
	}

	if (!initVars(&network->network_properties, cp, bm, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed))
		return false;

	printf("Loading Graph from File.....\n");
	fflush(stdout);

	//int idx1 = 0, idx2 = 0, idx3 = 0;
	int i;
	try {
		//Read Data Keys
		printf("\tReading Data Keys.\n");
		fflush(stdout);
		std::ifstream dataStream;
		std::string line;
		char *pch, *filename;
		filename = NULL;
	
		dataStream.open("./etc/data_keys.cset.key");
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
			throw CausetException("Failed to open 'data_keys.cset.key' file!\n");
		
		//Allocate Memory	
		if (!createNetwork(network->nodes, network->edges, network->core_edge_exists, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.manifold, network->network_properties.core_edge_fraction, network->network_properties.edge_buffer, cp->sCreateNetwork, hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed, network->network_properties.flags.use_gpu, network->network_properties.flags.verbose, network->network_properties.flags.bench, network->network_properties.flags.yes))
			return false;

		//Solve for eta0 and tau0
		if (!solveMaxTime(network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim, network->network_properties.a, network->network_properties.zeta, network->network_properties.tau0, network->network_properties.alpha, network->network_properties.flags.universe))
			return false;

		//Read node positions
		printf("\tReading Node Position Data.\n");
		fflush(stdout);
		std::stringstream dataname;
		dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
		dataStream.open(dataname.str().c_str());
		if (dataStream.is_open()) {
			while (getline(dataStream, line))
				network->network_properties.N_tar++;
			dataStream.seekg(0, dataStream.beg);

			for (i = 0; i < network->network_properties.N_tar; i++) {
				getline(dataStream, line);

				if (network->network_properties.manifold == DE_SITTER) {
					if (network->network_properties.dim == 1) {
						network->nodes.tau[i] = atof(strtok(NULL, " "));
						network->nodes.theta[i] = atof(strtok(NULL, " "));
					} else if (network->network_properties.dim == 3) {
						if (network->network_properties.flags.universe) {
							network->nodes.tau[i] = atof(strtok(NULL, " "));
							//Solve for eta using numerical integration
						} else {
							network->nodes.sc[i].w = atof(strtok(NULL, " "));
							network->nodes.tau[i] = etaToTau(network->nodes.sc[i].w);
						}
						network->nodes.sc[i].x = atof(strtok(NULL, " "));
						network->nodes.sc[i].y = atof(strtok(NULL, " "));
						network->nodes.sc[i].z = atof(strtok(NULL, " "));
					}
				} else if (network->network_properties.manifold == HYPERBOLIC) {
					strtok(NULL, " ");
					network->nodes.theta[i] = atof(strtok(NULL, " "));
					network->nodes.tau[i] = atof(strtok(NULL, " "));
				}

				if (network->network_properties.dim == 1 || network->network_properties.manifold == HYPERBOLIC) {
					if (network->nodes.tau[i] < 0.0)
						throw CausetException("Invalid value parsed for 'tau' in node position file!\n");
					if (network->nodes.theta[i] <= 0.0 || network->nodes.theta[i] >= TWO_PI)
						throw CausetException("Invalid value for 'theta' in node position file!\n");
				} else if (network->network_properties.dim == 3 && network->network_properties.manifold == DE_SITTER) {
					if (network->nodes.sc[i].w <= 0.0 || network->nodes.sc[i].w >= HALF_PI)
						throw CausetException("Invalid value for 'eta' in node position file!\n");
					if (network->nodes.sc[i].x <= 0.0 || network->nodes.sc[i].x >= TWO_PI)
						throw CausetException("Invalid value for 'theta' in node position file!\n");
					if (network->nodes.sc[i].y <= 0.0 || network->nodes.sc[i].y >= M_PI)
						throw CausetException("Invalid value for 'phi' in node position file!\n");
					if (network->nodes.sc[i].z <= 0.0 || network->nodes.sc[i].z >= M_PI)
						throw CausetException("Invalid value for 'chi' in node position file!\n");
				}
			}
			dataStream.close();
		} else
			throw CausetException("Failed to open node position file!\n");

		//Read Adjacency List
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	printf("Task Completed.\n");
	fflush(stdout);

	return true;
}

//Print to File
static bool printNetwork(Network &network, const CausetPerformance &cp, const long &init_seed, const int &gpuID)
{
	if (!network.network_properties.flags.print_network)
		return false;

	printf("Printing Results to File...\n");
	fflush(stdout);

	int i, j, k;
	try {
		//Confirm directory structure exists
		boost::filesystem::create_directory("./dat");
		boost::filesystem::create_directory("./dat/pos");
		boost::filesystem::create_directory("./dat/edg");
		boost::filesystem::create_directory("./dat/dst");
		boost::filesystem::create_directory("./dat/idd");
		boost::filesystem::create_directory("./dat/odd");
		boost::filesystem::create_directory("./dat/cls");
		boost::filesystem::create_directory("./dat/cdk");

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
		if (network.network_properties.flags.calc_components) {
			outputStream << "Number of Connected Components\t" << network.network_properties.N_cc << std::endl;
			outputStream << "Size of Giant Connected Component\t" << network.network_properties.N_gcc << std::endl;
		}
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
		if (network.network_properties.flags.calc_components)
			outputStream << "measureComponents:   " << cp.sMeasureConnectedComponents.elapsedTime << " sec" << std::endl;
		if (network.network_properties.flags.calc_success_ratio)
			outputStream << "measureSuccessRatio: " << cp.sMeasureSuccessRatio.elapsedTime << " sec" << std::endl;

		outputStream << "Total Time:        " << cp.sCauset.elapsedTime << " sec" << std::endl;

		outputStream.flush();
		outputStream.close();

		std::ofstream mapStream;
		mapStream.open("./etc/data_keys.cset.key", std::ios::app);
		if (!mapStream.is_open())
			throw CausetException("Failed to open 'etc/data_keys.cset.key' file!\n");
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
				dataStream << i << " " << network.edges.future_edges[idx + j] << std::endl;
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

static bool printBenchmark(const Benchmark &bm, const CausetFlags &cf)
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
		if (cf.calc_components)
			fprintf(f, "\tmeasureComponents:\t%5.6f sec\n", bm.bMeasureConnectedComponents);
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
	if (cf.calc_components)
		printf("\tmeasureConnectedComponents:\t%5.6f sec\n", bm.bMeasureConnectedComponents);
	if (cf.calc_success_ratio)
		printf("\tmeasureSuccessRatio:\t%5.6f sec\n", bm.bMeasureSuccessRatio);
	printf("\n");
	fflush(stdout);

	return true;
}

//Free Memory
static void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed)
{
	if (network->network_properties.manifold == DE_SITTER && network->network_properties.dim == 3) {
		free(network->nodes.sc);
		network->nodes.sc = NULL;
		hostMemUsed -= sizeof(float4) * network->network_properties.N_tar;
	} else if (network->network_properties.manifold == HYPERBOLIC || network->network_properties.dim == 1) {
		free(network->nodes.theta);
		network->nodes.theta = NULL;
		hostMemUsed -= sizeof(float) * network->network_properties.N_tar;
	}

	free(network->nodes.tau);
	network->nodes.tau = NULL;
	hostMemUsed -= sizeof(float) * network->network_properties.N_tar;

	if (network->network_properties.flags.use_gpu)
		cuMemFreeHost(network->nodes.k_in);
	else
		free(network->nodes.k_in);
	network->nodes.k_in = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	if (network->network_properties.flags.use_gpu)
		cuMemFreeHost(network->nodes.k_out);
	else
		free(network->nodes.k_out);
	network->nodes.k_out = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	if (network->network_properties.flags.use_gpu)
		cuMemFreeHost(network->edges.past_edges);
	else
		free(network->edges.past_edges);
	network->edges.past_edges = NULL;
	hostMemUsed -= sizeof(int) * static_cast<unsigned int>(network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

	if (network->network_properties.flags.use_gpu)
		cuMemFreeHost(network->edges.future_edges);
	else
		free(network->edges.future_edges);
	network->edges.future_edges = NULL;
	hostMemUsed -= sizeof(int) * static_cast<unsigned int>(network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);

	free(network->edges.past_edge_row_start);
	network->edges.past_edge_row_start = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->edges.future_edge_row_start);
	network->edges.future_edge_row_start = NULL;
	hostMemUsed -= sizeof(int) * network->network_properties.N_tar;

	free(network->core_edge_exists);
	network->core_edge_exists = NULL;
	hostMemUsed -= sizeof(bool) * static_cast<unsigned int>(POW2(network->network_properties.core_edge_fraction * network->network_properties.N_tar, EXACT));

	if (network->network_properties.flags.bench)
		return;

	if (network->network_properties.flags.calc_clustering) {
		free(network->network_observables.clustering);
		network->network_observables.clustering = NULL;
		hostMemUsed -= sizeof(float) * network->network_properties.N_deg2;
	}

	if (network->network_properties.flags.calc_components) {
		free(network->nodes.cc);
		network->nodes.cc = NULL;
		hostMemUsed -= sizeof(int) * network->network_properties.N_tar;
	}
}
