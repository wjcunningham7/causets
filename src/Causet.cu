#include "NetworkCreator.cu"
#include "Measurements.cu"

int main(int argc, char **argv)
{
	Network network = Network(parseArgs(argc, argv));
	CausetPerformance cp = CausetPerformance();
	Benchmark bm = Benchmark();
	
	stopwatchStart(&cp.sCauset);
	long init_seed = network.network_properties.seed;
	bool success = false;

	shrQAStart(argc, argv);

	if (network.network_properties.flags.use_gpu)
		connectToGPU(argc, argv);
	
	if (network.network_properties.graphID == 0 && !initializeNetwork(&network, &cp, &bm)) goto CausetExit;
	else if (network.network_properties.graphID != 0 && !loadNetwork(&network, &cp)) goto CausetExit;

	measureNetworkObservables(&network, &cp, &bm);

	stopwatchStop(&cp.sCauset);

	if (BENCH && !printBenchmark(bm, network.network_properties.flags)) goto CausetExit;
	if (network.network_properties.flags.disp_network && !displayNetwork(network.nodes, network.future_edges, argc, argv)) goto CausetExit;
	if (!BENCH) printMemUsed(NULL, maxHostMemUsed, maxDevMemUsed);
	if (network.network_properties.flags.print_network && !printNetwork(network, cp, init_seed)) goto CausetExit;
	
	destroyNetwork(&network);
	if (network.network_properties.flags.use_gpu) cuCtxDetach(cuContext);

	success = true;

	CausetExit:
	shrQAFinish(argc, (const char**)argv, success ? QA_PASSED : QA_FAILED);
	printf("Time:  %5.6f s\n", cp.sCauset.elapsedTime);
	printf("PROGRAM COMPLETED\n");
}

//Handles all network generation and initialization procedures
bool initializeNetwork(Network *network, CausetPerformance *cp, Benchmark *bm)
{
	assert (network != NULL);
	assert (cp != NULL);
	assert (bm != NULL);

	printf("Initializing Causet Network...\n");
	
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
			double guess = 0.5;
			assert (network->network_properties.N_tar > 0 && network->network_properties.alpha > 0.0 && network->network_properties.delta > 0.0);
			NewtonProperties np = NewtonProperties(guess, TOL, 10000, network->network_properties.N_tar, network->network_properties.alpha, network->network_properties.delta);
			newton(&solveTau0, &np, &network->network_properties.seed);
			network->network_properties.tau0 = np.x;
			assert (network->network_properties.tau0 > 0.0);
			
			network->network_properties.ratio = powf(sinh(1.5 * network->network_properties.tau0), 2.0);
			assert(network->network_properties.ratio > 0.0 && network->network_properties.ratio < 1.0);
			network->network_properties.omegaM = 1.0 / (network->network_properties.ratio + 1.0);
			network->network_properties.omegaL = 1.0 - network->network_properties.omegaM;
		} else if (network->network_properties.flags.cc.conflicts[1] == 0 || network->network_properties.flags.cc.conflicts[2] == 0 || network->network_properties.flags.cc.conflicts[3] == 0) {
			if (network->network_properties.N_tar > 0 && network->network_properties.alpha > 0.0) {
				//Solve for delta
				network->network_properties.delta = 3.0 * network->network_properties.N_tar / (M_PI * M_PI * powf(network->network_properties.alpha, 3.0) * (sinh(3.0 * network->network_properties.tau0) - 3.0 * network->network_properties.tau0));
				assert (network->network_properties.delta > 0.0);
			} else if (network->network_properties.N_tar == 0) {
				//Solve for N_tar
				assert (network->network_properties.alpha > 0.0);
				network->network_properties.N_tar = M_PI * M_PI * network->network_properties.delta * powf(network->network_properties.alpha, 3.0) * (sinh(3.0 * network->network_properties.tau0) - 3.0 * network->network_properties.tau0) / 3.0;
				assert (network->network_properties.N_tar > 0);
			} else {
				//Solve for alpha
				assert (network->network_properties.N_tar > 0);
				network->network_properties.alpha = powf(3.0 * network->network_properties.N_tar / (M_PI * M_PI * network->network_properties.delta * (sinh(3.0 * network->network_properties.tau0) - 3.0 * network->network_properties.tau0)), (1.0 / 3.0));
				assert (network->network_properties.alpha > 0.0);
			}
		}
		//Finally, solve for R0
		assert (network->network_properties.alpha > 0.0 && network->network_properties.ratio > 0.0 && network->network_properties.ratio < 1.0);
		network->network_properties.R0 = network->network_properties.alpha * powf(network->network_properties.ratio, (1.0 / 3.0));
		assert (network->network_properties.R0 > 0.0);
	}

	bool tmp = false;
	int i;

	//Allocate memory needed by pointers
	if (BENCH) {
		tmp = network->network_properties.flags.calc_clustering;
		network->network_properties.flags.calc_clustering = false;

		for (i = 0; i < NBENCH; i++) {
			if (!createNetwork(network, cp))
				return false;
				
			bm->bCreateNetwork += cp->sCreateNetwork.elapsedTime;
			destroyNetwork(network);
			stopwatchReset(&cp->sCreateNetwork);
		}
		bm->bCreateNetwork /= NBENCH;
	
		if (tmp)
			network->network_properties.flags.calc_clustering = true;
	}

	tmp = BENCH;
	if (tmp)
		BENCH = false;
	if (!createNetwork(network, cp))
		return false;
	if (tmp)
		BENCH = true;

	//Solve for eta0 using Newton-Raphson Method
	double guess = 0.08;
	assert (network->network_properties.N_tar > 0 && network->network_properties.k_tar > 0.0 && (network->network_properties.dim == 1 || network->network_properties.dim == 3));
	NewtonProperties np = NewtonProperties(guess, TOL, 10000, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim);
	if (network->network_properties.dim == 1)
		np.x = (M_PI / 2.0) - 0.0001;

	newton(&solveZeta, &np, &network->network_properties.seed);
	network->network_properties.zeta = np.x;
	if (network->network_properties.dim == 1)
		network->network_properties.zeta = (M_PI / 2.0) - np.x;
	assert (network->network_properties.zeta > 0 && network->network_properties.zeta < M_PI / 2.0);

	printf("\tTranscendental Equation Solved:\n");
	//printf("\t\tZeta: %5.8f\n", network->network_properties.zeta);
	printf("\t\tMaximum Conformal Time: %5.8f\n", (M_PI / 2.0) - network->network_properties.zeta);
	printf("\t\tMaximum Rescaled Time:  %5.8f\n", etaToT((M_PI / 2.0) - network->network_properties.zeta, network->network_properties.a));

	//Generate coordinates of nodes in 1+1 or 3+1 de Sitter spacetime
	//and then order nodes temporally using quicksort
	int low = 0;
	int high = network->network_properties.N_tar - 1;
	if (BENCH) {
		for (i = 0; i < NBENCH; i++) {
			if (!generateNodes(network, cp, network->network_properties.flags.use_gpu))
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
		BENCH = false;
	if (!generateNodes(network, cp, network->network_properties.flags.use_gpu))
		return false;
	if (tmp)
		BENCH = true;

	//Quicksort
	stopwatchStart(&cp->sQuicksort);
	quicksort(network->nodes, low, high);
	stopwatchStop(&cp->sQuicksort);
			
	printf("\tQuick Sort Successfully Performed.\n");
	if (CAUSET_DEBUG)
		printf("\t\tExecution Time: %5.6f sec\n", cp->sQuicksort.elapsedTime);

	//Identify edges as points connected by timelike intervals
	if (BENCH) {
		for (i = 0; i < NBENCH; i++) {
			if (!linkNodes(network, cp, network->network_properties.flags.use_gpu))
				return false;
				
			bm->bLinkNodes += cp->sLinkNodes.elapsedTime;
			stopwatchReset(&cp->sLinkNodes);
		}
		bm->bLinkNodes /= NBENCH;
	}

	if (tmp)
		BENCH = false;
	if (!linkNodes(network, cp, network->network_properties.flags.use_gpu))
		return false;
	if (tmp)
		BENCH = true;

	printf("Task Completed.\n");
	return true;
}

void measureNetworkObservables(Network *network, CausetPerformance *cp, Benchmark *bm)
{
	assert (network != NULL);
	assert (cp != NULL);
	assert (bm != NULL);

	printf("\nCalculating Network Observables...\n");

	if (network->network_properties.flags.calc_clustering) {
		if (BENCH) {
			int i;
			for (i = 0; i < NBENCH; i++) {
				measureClustering(network, cp);
				bm->bMeasureClustering += cp->sMeasureClustering.elapsedTime;
				stopwatchReset(&cp->sMeasureClustering);
			}
			bm->bMeasureClustering /= NBENCH;
		}
		
		bool tmp = BENCH;
		if (tmp)
			BENCH = false;
		measureClustering(network, cp);
		if (tmp)
			BENCH = true;
	}

	printf("Task Completed.\n");
}

//Plot using OpenGL
bool displayNetwork(Node *nodes, int *future_edges, int argc, char **argv)
{
	assert (nodes != NULL);
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
bool loadNetwork(Network *network, CausetPerformance *cp)
{
	assert (network != NULL);
	assert (cp != NULL);
	assert (network->network_properties.graphID != 0);
	assert (!BENCH);

	printf("Loading Graph from File.....\n");

	int idx1 = 0, idx2 = 0, idx3 = 0;
	int i, j, k;
	try {
		//Read Data Keys
		printf("\tReading Data Keys.\n");
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

			//Read k_tar
			getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 5; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.k_tar = atof(pch);
			if (network->network_properties.k_tar <= 0.0)
				throw CausetException("Invalid value for expected average degreed!\n");
			printf("\t\tk_tar:\t%f\n", network->network_properties.k_tar);

			//Read a
			getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 2; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.a = atof(pch);
			if (network->network_properties.a <= 0.0)
				throw CausetException("Invalid value for pseudoradius!\n");
			printf("\t\ta:\t%f\n", network->network_properties.a);

			//Read eta_0 (save as zeta)
			for (i = 0; i < 6; i++)
				getline(dataStream, line);
			pch = strtok((char*)line.c_str(), " \t");
			for (i = 0; i < 4; i++)
				pch = strtok(NULL, " \t");
			network->network_properties.zeta = (M_PI / 2.0) - atof(pch);
			if (network->network_properties.zeta <= 0.0 || network->network_properties.zeta >= M_PI / 2.0)
				throw CausetException("Invalid value for eta0!\n");
			printf("\t\teta0:\t%f\n", (M_PI / 2.0) - network->network_properties.zeta);

			dataStream.close();
		} else
			throw CausetException("Failed to open simulation parameters file!\n");

		if (!createNetwork(network, cp))
			return false;

		//Read node positions
		printf("\tReading Node Position Data.\n");
		std::stringstream dataname;
		dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
		dataStream.open(dataname.str().c_str());
		if (dataStream.is_open()) {
			for (i = 0; i < network->network_properties.N_tar; i++) {
				getline(dataStream, line);
				network->nodes[i] = Node();
				network->nodes[i].t = etaToT(atof(strtok((char*)line.c_str(), " ")), network->network_properties.a);
				network->nodes[i].theta = atof(strtok(NULL, " "));
				network->nodes[i].phi = atof(strtok(NULL, " "));
				network->nodes[i].chi = atof(strtok(NULL, " "));
				
				if (network->nodes[i].t <= 0.0)
					throw CausetException("Invalid value parsed for t in node position file!\n");
				if (network->nodes[i].theta <= 0.0 || network->nodes[i].theta >= 2 * M_PI)
					throw CausetException("Invalid value parsed for theta in node position file!\n");
				if (network->nodes[i].phi <= 0.0 || network->nodes[i].phi >= M_PI)
					throw CausetException("Invalid value parsed for phi in node position file!\n");
				if (network->nodes[i].chi <= 0.0 || network->nodes[i].chi >= M_PI)
					throw CausetException("Invalid value parsed for chi in node position file!\n");
			}
			dataStream.close();
		} else
			throw CausetException("Failed to open node position file!\n");

		//for (int i = 0; i < network->network_properties.N_tar; i++)
		//	printf("%f\t%f\t%f\t%f\n", network->nodes[i].t, network->nodes[i].theta, network->nodes[i].phi, network->nodes[i].chi);

		printf("\tReading Edge Data.\n");
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
		assert(network->nodes[i].k_in >= 0);

		if (network->past_edge_row_start[i] == idx3)
			network->past_edge_row_start[i] = -1;
	}

	//compareAdjacencyLists(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);
	//compareAdjacencyListIndices(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);

	//Adjacency Matrix
	printf("\tPopulating Adjacency Matrix.\n");
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

	return true;
}

//Print to File
bool printNetwork(Network network, CausetPerformance cp, long init_seed)
{
	if (!network.network_properties.flags.print_network)
		return false;

	printf("Printing Results to File...\n");

	int i, j, k;
	try {
		std::ofstream outputStream;
		std::stringstream sstm;

		if (network.network_properties.flags.use_gpu)
			sstm << "Dev" << gpuID << "_";
		else
			sstm << "CPU_";
		sstm << network.network_properties.N_tar << "_";
		sstm << network.network_properties.k_tar << "_";
		sstm << network.network_properties.a << "_";
		sstm << network.network_properties.dim << "_";
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

		outputStream << "\nCauset Input Parameters:" << std::endl;
		outputStream << "------------------------" << std::endl;
		outputStream << "Target Nodes (N_tar)\t\t\t" << network.network_properties.N_tar << std::endl;
		outputStream << "Target Expected Average Degrees (k_tar)\t" << network.network_properties.k_tar << std::endl;
		outputStream << "Pseudoradius (a)\t\t\t" << network.network_properties.a << std::endl;

		outputStream << "\nCauset Calculated Values:" << std::endl;
		outputStream << "--------------------------" << std::endl;
		outputStream << "Resulting Nodes (N_res)\t\t\t" << network.network_properties.N_res << std::endl;
		outputStream << "Resulting Average Degrees (k_res)\t" << network.network_properties.k_res << std::endl;
		outputStream << "Maximum Conformal Time (eta_0)\t\t" << ((M_PI / 2.0) - network.network_properties.zeta) << std::endl;
		outputStream << "Maximum Rescaled Time (t_0)  \t\t" << etaToT((M_PI / 2.0) - network.network_properties.zeta, network.network_properties.a) << std::endl;

		if (network.network_properties.flags.calc_clustering)
			outputStream << "Average Clustering\t\t\t" << network.network_observables.average_clustering << std::endl;

		outputStream << "\nNetwork Analysis Results:" << std::endl;
		outputStream << "-------------------------" << std::endl;
		outputStream << "Node Position Data:\t\t" << "pos/" << network.network_properties.graphID << ".cset.pos.dat" << std::endl;
		outputStream << "Node Edge Data:\t\t\t" << "edg/" << network.network_properties.graphID << ".cset.edg.dat" << std::endl;
		outputStream << "Degree Distribution Data:\t" << "dst/" << network.network_properties.graphID << ".cset.dst.dat" << std::endl;
		outputStream << "In-Degree Distribution Data:\t" << "idd/" << network.network_properties.graphID << ".cset.idd.dat" << std::endl;
		outputStream << "Out-Degree Distribution Data:\t" << "odd/" << network.network_properties.graphID << ".cset.odd.dat" << std::endl;

		if (network.network_properties.flags.calc_clustering) {
			outputStream << "Clustering Coefficient Data:\t" << "cls/" << network.network_properties.graphID << ".cset.cls.dat" << std::endl;
			outputStream << "Clustering by Degree Data:\t" << "cdk/" << network.network_properties.graphID << ".cset.cdk.dat" << std::endl;
		}

		outputStream << "\nAlgorithmic Performance:" << std::endl;
		outputStream << "--------------------------" << std::endl;
		outputStream << "createNetwork:     " << cp.sCreateNetwork.elapsedTime << " sec" << std::endl;
		outputStream << "generateNodes:     " << cp.sGenerateNodes.elapsedTime << " sec" << std::endl;
		outputStream << "quicksort:         " << cp.sQuicksort.elapsedTime << " sec" << std::endl;
		outputStream << "linkNodes:         " << cp.sLinkNodes.elapsedTime << " sec" << std::endl;

		if (network.network_properties.flags.calc_clustering)
			outputStream << "measureClustering: " << cp.sMeasureClustering.elapsedTime << " sec" << std::endl;

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
			dataStream << tToEta(network.nodes[i].t, network.network_properties.a) << " " << network.nodes[i].theta;
			if (network.network_properties.dim == 3)
				dataStream << " " << network.nodes[i].phi << " " << network.nodes[i].chi;
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
			for (j = 0; j < network.nodes[i].k_out; j++)
				dataStream << i << " " << network.future_edges[idx + j] << std::endl;
			idx += network.nodes[i].k_out;
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
				if (network.nodes[i].k_in + network.nodes[i].k_out == k)
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
				if (network.nodes[i].k_in == k)
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
				if (network.nodes[i].k_out == k)
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
					if (i == (network.nodes[j].k_in + network.nodes[j].k_out)) {
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
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}
	
	printf("Task Completed.\n\n");

	return true;
}

bool printBenchmark(Benchmark bm, CausetFlags cf)
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
		fprintf(f, "\tcreateNetwork:\t\t%5.6f sec\n", bm.bCreateNetwork);
		fprintf(f, "\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
		fprintf(f, "\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
		fprintf(f, "\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
		if (cf.calc_clustering)
			fprintf(f, "\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);

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
	printf("\tcreateNetwork:\t\t%5.6f sec\n", bm.bCreateNetwork);
	printf("\tgenerateNodes:\t\t%5.6f sec\n", bm.bGenerateNodes);
	printf("\tquicksort:\t\t%5.6f sec\n", bm.bQuicksort);
	printf("\tlinkNodes:\t\t%5.6f sec\n", bm.bLinkNodes);
	if (cf.calc_clustering)
		printf("\tmeasureClustering:\t%5.6f sec\n", bm.bMeasureClustering);
	printf("\n");

	return true;
}

//Free Memory
void destroyNetwork(Network *network)
{
	free(network->nodes);
	network->nodes = NULL;
	hostMemUsed -= sizeof(Node) * network->network_properties.N_tar;

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
	hostMemUsed -= sizeof(bool) * powf(network->network_properties.core_edge_fraction * network->network_properties.N_tar, 2.0);

	if (network->network_properties.flags.calc_clustering) {
		free(network->network_observables.clustering);
		network->network_observables.clustering = NULL;
		hostMemUsed -= sizeof(float) * network->network_properties.N_deg2;
	}

	if (network->network_properties.flags.use_gpu) {
		cuMemFree(network->d_nodes);
		network->d_nodes = NULL;
		devMemUsed -= sizeof(Node) * network->network_properties.N_tar;

		cuMemFree(network->d_edges);
		network->d_edges = NULL;
		devMemUsed -= sizeof(Node) * network->network_properties.N_tar * network->network_properties.k_tar / 2;
	}
}
