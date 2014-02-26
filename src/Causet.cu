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
	
	if (!initializeNetwork(&network)) goto CausetExit;
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
	sstm << network.network_properties.dim;
	sstm << init_seed;
	std::string filename = sstm.str();

	outputStream.open(("./dat/" + filename + ".cset.out").c_str());
	outputStream << "Causet Simulation\n";
	outputStream << "Graph ID: " << (int)time(NULL) << std::endl;

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
	outputStream << "Node Position Data:\t\t" << "pos/" << filename << ".cset.pos.dat" << std::endl;
	outputStream << "Node Edge Data:\t\t\t" << "edg/" << filename << ".cset.edg.dat" << std::endl;
	outputStream << "Degree Distribution Data:\t" << "dst/" << filename << ".cset.dst.dat" << std::endl;

	if (network.network_properties.flags.calc_clustering) {
		outputStream << "Clustering Coefficient Data:\t" << "cls/" << filename << ".cset.cls.dat" << std::endl;
		outputStream << "Clustering by Degree Data:\t" << "cdk/" << filename << ".cset.cdk.dat" << std::endl;
	}

	outputStream << "\nAlgorithmic Performance:" << std::endl;
	outputStream << "--------------------------" << std::endl;

	outputStream.flush();
	outputStream.close();

	std::ofstream dataStream;

	dataStream.open(("./dat/pos/" + filename + ".cset.pos.dat").c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++) {
		dataStream << tauToEta(network.nodes[i].tau, network.network_properties.a) << " " << network.nodes[i].theta;
		if (network.network_properties.dim == 3)
			dataStream << " " << network.nodes[i].phi << " " << network.nodes[i].chi;
		dataStream << std::endl;
	}
	dataStream.flush();
	dataStream.close();

	unsigned int idx = 0;
	dataStream.open(("./dat/edg/" + filename + ".cset.edg.dat").c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++) {
		for (unsigned int j = 0; j < network.nodes[i].k_out; j++)
			dataStream << i << " " << network.future_edges[idx + j] << std::endl;
		idx += network.nodes[i].k_out;
	}
	dataStream.flush();
	dataStream.close();

	dataStream.open(("./dat/dst/" + filename + ".cset.dst.dat").c_str());
	for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
		if (network.nodes[i].k_in + network.nodes[i].k_out > 0)
			dataStream << (network.nodes[i].k_in + network.nodes[i].k_out) << std::endl;
	dataStream.flush();
	dataStream.close();

	if (network.network_properties.flags.calc_clustering) {
		dataStream.open(("./dat/cls/" + filename + ".cset.cls.dat").c_str());
		for (unsigned int i = 0; i < network.network_properties.N_tar; i++)
			dataStream << network.network_observables.clustering[i] << std::endl;
		dataStream.flush();
		dataStream.close();

		dataStream.open(("./dat/cdk/" + filename + ".cset.cdk.dat").c_str());
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
