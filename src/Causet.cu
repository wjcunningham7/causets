#include "NetworkCreator.cu"
#include "Measurements.cu"

int main(int argc, char **argv)
{
	stopwatchStart();
	Network network = Network(parseArgs(argc, argv));
	long init_seed = network.network_properties.seed;
	bool success = false;

	shrQAStart(argc, argv);

	if (network.network_properties.flags.use_gpu)
		connectToGPU(argc, argv);
	
	if (network.network_properties.graphID != 0) {
		if (!loadNetwork(&network)) goto CausetExit;
	}
	else if (!initializeNetwork(&network)) goto CausetExit;

	measureNetworkObservables(&network);

	if (network.network_properties.flags.disp_network && !displayNetwork(network.nodes, network.future_edges, argc, argv)) goto CausetExit;
	printMemUsed(NULL, maxHostMemUsed, maxDevMemUsed);
	if (network.network_properties.flags.print_network && !printNetwork(network, init_seed)) goto CausetExit;
	if (!destroyNetwork(&network)) goto CausetExit;

	if (network.network_properties.flags.use_gpu)
		cuCtxDetach(cuContext);

	success = true;

	CausetExit:
	shrQAFinish(argc, (const char**)argv, success ? QA_PASSED : QA_FAILED);
	printf("Time:  %5.9f s\n", stopwatchReadSeconds());
	printf("PROGRAM COMPLETED\n");
}

//Handles all network generation and initialization procedures
bool initializeNetwork(Network *network)
{
	printf("Initializing Causet Network...\n");

	//Allocate memory needed by pointers
	if (!createNetwork(network))
		return false;

	//Solve for eta0 using Newton-Raphson Method
	double guess = 0.08;
	NewtonProperties np = NewtonProperties(guess, TOL, 10000, network->network_properties.N_tar, network->network_properties.k_tar, network->network_properties.dim);
	if (network->network_properties.dim == 1)
		np.x = (M_PI / 2.0) - 0.0001;

	newton(&solveZeta, &np, &network->network_properties.seed);
	network->network_properties.zeta = np.x;
	if (network->network_properties.dim == 1)
		network->network_properties.zeta = (M_PI / 2.0) - np.x;

	printf("\tTranscendental Equation Solved:\n");
	//printf("\t\tZeta: %5.8f\n", network->network_properties.zeta);
	printf("\t\tMaximum Conformal Time: %5.8f\n", (M_PI / 2.0) - network->network_properties.zeta);
	printf("\t\tMaximum Rescaled Time:  %5.8f\n", etaToTau((M_PI / 2.0) - network->network_properties.zeta, network->network_properties.a));

	//Generate coordinates of nodes in 1+1 or 3+1 de Sitter spacetime
	if (!generateNodes(network, network->network_properties.flags.use_gpu))
		return false;

	//Order nodes temporally
	int low  = 0;
	int high = network->network_properties.N_tar - 1;

	quicksort(network->nodes, low, high);
	printf("\tQuick Sort Successfully Performed.\n");

	//Identify edges as points connected by timelike intervals
	if (!linkNodes(network, network->network_properties.flags.use_gpu))
		return false;

	printf("Task Completed.\n");
	return true;
}

void measureNetworkObservables(Network *network)
{
	printf("\nCalculating Network Observables...\n");

	if (network->network_properties.flags.calc_clustering)
		measureClustering(network);

	printf("Task Completed.\n");
}

//Plot using OpenGL
bool displayNetwork(Node *nodes, unsigned int *future_edges, int argc, char **argv)
{
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
bool loadNetwork(Network *network)
{
	if (network->network_properties.graphID == 0)
		return false;
	printf("Loading Graph from File.....\n");

	//Read Data Keys
	printf("\tReading Data Keys.\n");
	std::ifstream dataStream;
	std::string line;
	char *pch, *filename;
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
	}

	//Read Main Data File
	printf("\tReading Simulation Parameters.\n");
	std::stringstream fullname;
	fullname << "./dat/" << filename << ".cset.out";
	dataStream.open(fullname.str().c_str());
	if (dataStream.is_open()) {
		//Read N_tar
		for (unsigned int i = 0; i < 7; i++)
			getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 3; i++)
			pch = strtok(NULL, " \t");
		network->network_properties.N_tar = atoi(pch);
		printf("\t\tN_tar:\t%u\n", network->network_properties.N_tar);
		
		//Read k_tar
		getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 5; i++)
			pch = strtok(NULL, " \t");
		network->network_properties.k_tar = atof(pch);
		printf("\t\tk_tar:\t%f\n", network->network_properties.k_tar);

		//Read a
		getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 2; i++)
			pch = strtok(NULL, " \t");
		network->network_properties.a = atof(pch);
		printf("\t\ta:\t%f\n", network->network_properties.a);

		//Read N_res
		for (unsigned int i = 0; i < 4; i++)
			getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 3; i++)
			pch = strtok(NULL, " \t");
		printf("\t\tN_res:\t%u\n", atoi(pch));

		//Read k_res
		getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 4; i++)
			pch = strtok(NULL, " \t");
		printf("\t\tk_res:\t%f\n", atof(pch));

		//Read eta_0 (save as zeta)
		getline(dataStream, line);
		pch = strtok((char*)line.c_str(), " \t");
		for (unsigned int i = 0; i < 4; i++)
			pch = strtok(NULL, " \t");
		network->network_properties.zeta = (M_PI / 2.0) - atof(pch);
		printf("\t\teta0:\t%f\n", (M_PI / 2.0) - network->network_properties.zeta);
	
		dataStream.close();
	}

	if (!createNetwork(network)) return false;

	//Read node positions
	printf("\tReading Node Position Data.\n");
	std::stringstream dataname;
	dataname << "./dat/pos/" << network->network_properties.graphID << ".cset.pos.dat";
	dataStream.open(dataname.str().c_str());
	if (dataStream.is_open()) {
		for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
			try {
				getline(dataStream, line);
				network->nodes[i] = Node();
				network->nodes[i].tau = etaToTau(atof(strtok((char*)line.c_str(), " ")), network->network_properties.a);
				network->nodes[i].theta = atof(strtok(NULL, " "));
				network->nodes[i].phi = atof(strtok(NULL, " "));
				network->nodes[i].chi = atof(strtok(NULL, " "));
			} catch (std::exception e) {
				printf("Corrupt data file:  %s\n", dataname.str().c_str());
				dataStream.close();
				return false;
			}
		}
		dataStream.close();
	}

	//for (unsigned int i = 0; i < network->network_properties.N_tar; i++)
	//	printf("%f\t%f\t%f\t%f\n", network->nodes[i].tau, network->nodes[i].theta, network->nodes[i].phi, network->nodes[i].chi);

	printf("\tReading Edge Data.\n");
	dataname.str("");
	dataname.clear();
	dataname << "./dat/edg/" << network->network_properties.graphID << ".cset.edg.dat";
	dataStream.open(dataname.str().c_str());
	unsigned int idx1 = 0, idx2 = 0, idx3 = 0;
	unsigned int diff;
	if (dataStream.is_open()) {
		unsigned int n1, n2;
		network->future_edge_row_start[0] = 0;
		while (getline(dataStream, line)) {
			try {
				//Read pairs of connected nodes (past, future)
				n1 = atoi(strtok((char*)line.c_str(), " "));
				n2 = atoi(strtok(NULL, " "));

				//Check if a node is skipped (k_i = 0)
				diff = n1 - idx1;
				
				//Multiple nodes skipped
				if (diff > 1) {
					for (unsigned int i = 0; i < diff - 1; i++)
						network->future_edge_row_start[++idx1] = -1;
				}

				//At least one node skipped
				if (diff > 0)
					network->future_edge_row_start[++idx1] = idx2;
				
				network->nodes[idx1].k_out++;
				network->future_edges[idx2++] = n2;
			} catch (std::exception e) {
				printf("Corrupt data file:  %s\n", dataname.str().c_str());
				dataStream.close();
				return false;
			}
		}

		//Assign pointer values for all latest disconnected nodes
		for (unsigned int i = idx1 + 1; i < network->network_properties.N_tar; i++)
			network->future_edge_row_start[i] = -1;
		dataStream.close();
	}

	//Assign past node list and pointer values
	network->past_edge_row_start[0] = -1;
	for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
		network->past_edge_row_start[i] = idx3;
		for (unsigned int j = 0; j < i; j++) {
			if (network->future_edge_row_start[j] == -1) continue;

			for (unsigned int k = 0; k < network->nodes[j].k_out; k++)
				if (i == network->future_edges[network->future_edge_row_start[j]+k])
					network->past_edges[idx3++] = j;
		}

		network->nodes[i].k_in = idx3 - network->past_edge_row_start[i];

		if (network->past_edge_row_start[i] == idx3)
			network->past_edge_row_start[i] = -1;
	}

	//compareAdjacencyLists(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);
	//compareAdjacencyListIndices(network->nodes, network->future_edges, network->future_edge_row_start, network->past_edges, network->past_edge_row_start);

	//Adjacency Matrix
	printf("\tPopulating Adjacency Matrix.\n");
	unsigned int core_limit = (unsigned int)(network->network_properties.core_edge_fraction * network->network_properties.N_tar);
	for (unsigned int i = 0; i < core_limit; i++)
		for (unsigned int j = 0; j < core_limit; j++)
			network->core_edge_exists[(i*core_limit)+j] = false;

	idx1 = 0, idx2 = 0;
	while (idx1 < core_limit) {
		if (network->future_edge_row_start[idx1] != -1) {
			for (unsigned int i = 0; i < network->nodes[idx1].k_out; i++) {
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
	for (unsigned int i = 0; i < network->network_properties.N_tar; i++) {
		if (network->nodes[i].k_in > 0 || network->nodes[i].k_out > 0) {
			network->network_properties.N_res++;
			network->network_properties.k_res+= network->nodes[i].k_in + network->nodes[i].k_out;

			if (network->nodes[i].k_in + network->nodes[i].k_out > 1)
				network->network_properties.N_deg2++;
		}
	}
	network->network_properties.k_res /= network->network_properties.N_res;

	printf("\t\tN_res:  %u\n", network->network_properties.N_res);
	printf("\t\tk_res:  %f\n", network->network_properties.k_res);
	printf("\t\tN_deg2: %u\n", network->network_properties.N_deg2);

	printf("Task Completed.\n");

	return true;
}

//Print to File
bool printNetwork(Network network, long init_seed)
{
	printf("Printing Results to File...\n");

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
	outputStream << "Causet Simulation\n";
	if (network.network_properties.graphID == 0)
		network.network_properties.graphID = (int)time(NULL);
	outputStream << "Graph ID: " << network.network_properties.graphID << std::endl;

	time_t rawtime;
	struct tm * timeinfo;
	static char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%X %x", timeinfo);
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
	outputStream << "Maximum Rescaled Time (tau_0)\t\t" << etaToTau((M_PI / 2.0) - network.network_properties.zeta, network.network_properties.a) << std::endl;

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

	outputStream.flush();
	outputStream.close();

	std::ofstream mapStream;
	mapStream.open("./dat/data_keys.key", std::ios::app);
	mapStream << network.network_properties.graphID << "\t" << filename << std::endl;
	mapStream.close();

	std::ofstream dataStream;

	sstm.str("");
	sstm.clear();
	sstm << "./dat/pos/" << network.network_properties.graphID << ".cset.pos.dat";
	dataStream.open(sstm.str().c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++) {
		dataStream << tauToEta(network.nodes[i].tau, network.network_properties.a) << " " << network.nodes[i].theta;
		if (network.network_properties.dim == 3)
			dataStream << " " << network.nodes[i].phi << " " << network.nodes[i].chi;
		dataStream << std::endl;
	}
	dataStream.flush();
	dataStream.close();

	unsigned int idx = 0;
	sstm.str("");
	sstm.clear();
	sstm << "./dat/edg/" << network.network_properties.graphID << ".cset.edg.dat";
	dataStream.open(sstm.str().c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++) {
		for (unsigned int j = 0; j < network.nodes[i].k_out; j++)
			dataStream << i << " " << network.future_edges[idx + j] << std::endl;
		idx += network.nodes[i].k_out;
	}
	dataStream.flush();
	dataStream.close();

	sstm.str("");
	sstm.clear();
	sstm << "./dat/dst/" << network.network_properties.graphID << ".cset.dst.dat";
	dataStream.open(sstm.str().c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
		if (network.nodes[i].k_in + network.nodes[i].k_out > 0)
			dataStream << (network.nodes[i].k_in + network.nodes[i].k_out) << std::endl;
	dataStream.flush();
	dataStream.close();

	sstm.str("");
	sstm.clear();
	sstm << "./dat/idd/" << network.network_properties.graphID << ".cset.idd.dat";
	dataStream.open(sstm.str().c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
		if (network.nodes[i].k_in > 0)
			dataStream << network.nodes[i].k_in << std::endl;
	dataStream.flush();
	dataStream.close();

	sstm.str("");
	sstm.clear();
	sstm << "./dat/odd/" << network.network_properties.graphID << ".cset.odd.dat";
	dataStream.open(sstm.str().c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
		if (network.nodes[i].k_out > 0)
			dataStream << network.nodes[i].k_out << std::endl;
	dataStream.flush();
	dataStream.close();

	if (network.network_properties.flags.calc_clustering) {
		sstm.str("");
		sstm.clear();
		sstm << "./dat/cls/" << network.network_properties.graphID << ".cset.cls.dat";
		dataStream.open(sstm.str().c_str());
		for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
			dataStream << network.network_observables.clustering[i] << std::endl;
		dataStream.flush();
		dataStream.close();

		sstm.str("");
		sstm.clear();
		sstm << "./dat/cdk/" << network.network_properties.graphID << ".cset.cdk.dat";
		dataStream.open(sstm.str().c_str());
		double cdk;
		unsigned int ndk;
		for (unsigned int i = 0; i < network.network_properties.N_tar; i++) {
			cdk = 0.0;
			ndk = 0;
			for (unsigned int j = 0; j < network.network_properties.N_tar; j++) {
				if (i == (network.nodes[j].k_in + network.nodes[j].k_out)) {
					cdk += network.network_observables.clustering[j];
					ndk++;
				}
			}
			if (ndk == 0) ndk++;
			dataStream << i << " " << (cdk / ndk) << std::endl;
		}
		dataStream.flush();
		dataStream.close();
	}
	
	printf("\tFilename: %s.cset.out\n", filename.c_str());
	printf("Task Completed.\n\n");

	return true;
}

//Free Memory
bool destroyNetwork(Network *network)
{
	free(network->nodes);			network->nodes = NULL;			hostMemUsed -= sizeof(Node)  * network->network_properties.N_tar;
	free(network->past_edges);			network->past_edges = NULL;			hostMemUsed -= sizeof(unsigned int)  * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);
	free(network->future_edges);		network->future_edges = NULL;		hostMemUsed -= sizeof(unsigned int)  * (network->network_properties.N_tar * network->network_properties.k_tar / 2 + network->network_properties.edge_buffer);
	free(network->past_edge_row_start);	network->past_edge_row_start = NULL;	hostMemUsed -= sizeof(int)   * network->network_properties.N_tar;
	free(network->future_edge_row_start);	network->future_edge_row_start = NULL;	hostMemUsed -= sizeof(int)   * network->network_properties.N_tar;
	free(network->core_edge_exists);		network->core_edge_exists = NULL;		hostMemUsed -= sizeof(bool)  * powf(network->network_properties.core_edge_fraction * network->network_properties.N_tar, 2.0);

	if (network->network_properties.flags.calc_clustering) {
		free(network->network_observables.clustering);
		network->network_observables.clustering = NULL;
		hostMemUsed -= sizeof(float) * network->network_properties.N_deg2;
	}

	if (network->network_properties.flags.use_gpu) {
		cuMemFree(network->d_nodes);
		network->d_nodes = NULL;
		devMemUsed  -= sizeof(Node)  * network->network_properties.N_tar;

		cuMemFree(network->d_edges);
		network->d_edges = NULL;
		devMemUsed  -= sizeof(Node)  * network->network_properties.N_tar * network->network_properties.k_tar / 2;
	}

	return true;
}
