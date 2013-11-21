#ifndef CAUSET_SUBROUTINES_CUH_
#define CAUSET_SUBROUTINES_CUH_

#include "CausetSubroutines.hpp"
#include "GPUSubroutines.cuh"
#include "CausetOperations.cuh"

//Handles all network generation and initialization procedures
bool initializeNetwork(Network *network)
{
	printf("Initializing Causet Network...\n");

	//Allocate memory needed by pointers
	if (!createNetwork(network))
		return false;

	//Solve for eta0 using Newton-Raphson Method
	double guess = 0.08;
	NewtonProperties np = NewtonProperties(guess, TOL, 10000, network->network_properties.N, network->network_properties.k, network->network_properties.dim);
	if (network->network_properties.dim == 2)
		np.x = (M_PI / 2.0) - 0.0001;

	newton(&solveZeta, &np, &network->network_properties.seed);
	network->network_properties.zeta = np.x;
	if (network->network_properties.dim == 2)
		network->network_properties.zeta = (M_PI / 2.0) - np.x;

	printf("\tTranscendental Equation Solved:\n");
	//printf("\t\tZeta: %5.8f\n", network->network_properties.zeta);
	printf("\t\tMaximum Conformal Time: %5.8f\n", (M_PI / 2.0) - network->network_properties.zeta);
	printf("\t\tMaximum Rescaled Time: %5.8f\n", etaToTau((M_PI / 2.0) - network->network_properties.zeta, network->network_properties.a));

	//Generate coordinates of nodes in 1+1 or 3+1 de Sitter spacetime
	if (!generateNodes(network, network->network_properties.flags.use_gpu))
		return false;

	//Order nodes temporally
	int low  = 0;
	int high = network->network_properties.N - 1;

	quicksort(network->nodes, network->network_properties.a, low, high);
	printf("\tQuick sort successfully performed.\n");

	//Identify edges as points connected by timelike intervals
	if (!linkNodes(network, network->network_properties.flags.use_gpu))
		return false;

	printf("\tCompleted.\n");
	return true;
}

//Allocates memory for network
bool createNetwork(Network *network)
{
	//Implement subnet_size later
	//Chanage sizeof(Node) to reflect number of dimensions necessary
	//	so that for 1+1 phi and chi are not allocated
	try {
		network->nodes = (Node*)malloc(sizeof(Node) * network->network_properties.N);
		if (network->nodes == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N;
		//printMemUsed("Memory Allocated for Nodes", hostMemUsed, devMemUsed);

		network->edges = (unsigned int*)malloc(sizeof(Node) * network->network_properties.N * network->network_properties.k / 2);
		if (network->edges == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N * network->network_properties.k / 2;

		network->edge_row_start = (unsigned int*)malloc(sizeof(unsigned int) * (network->network_properties.N - 1));
		if (network->edge_row_start == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(unsigned int) * (network->network_properties.N - 1);

		network->core_edge_exists = (bool*)malloc(sizeof(bool) * powf(network->network_properties.core_edge_ratio * network->network_properties.N, 2.0));
		if (network->core_edge_exists == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(bool) * powf(network->network_properties.core_edge_ratio * network->network_properties.N, 2.0);

		//Allocate memory on GPU if necessary
		if (network->network_properties.flags.use_gpu) {
			checkCudaErrors(cuMemAlloc(&network->d_nodes, sizeof(Node) * network->network_properties.N));
			devMemUsed += sizeof(Node) * network->network_properties.N;

			checkCudaErrors(cuMemAlloc(&network->d_edges, sizeof(Node) * network->network_properties.N * network->network_properties.k / 2));
			devMemUsed += sizeof(Node) * network->network_properties.N * network->network_properties.k / 2;
		}

		memoryCheckpoint();
		//printMemUsed("Total Memory Allocated for Network", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure!\n");
		return false;
	}

	printf("\tMemory Successfully Allocated.\n");
	return true;
}

//Poisson Sprinkling
bool generateNodes(Network *network, bool &use_gpu)
{
	if (use_gpu) {
		if (!generateNodesGPU(network))
			return false;
	} else {
		//Initialize Newton-Raphson Parameters
		NewtonProperties np = NewtonProperties(network->network_properties.zeta, TOL, network->network_properties.N, network->network_properties.k, network->network_properties.dim);

		//Generate coordinates for each of N nodes
		for (unsigned int i = 0; i < network->network_properties.N; i++) {
			network->nodes[i] = Node();

			///////////////////////////////////////////////////////////
			//~~~~~~~~~~~~~~~~~~~~~~~~~Theta~~~~~~~~~~~~~~~~~~~~~~~~~//
			//Sample Theta from (0, 2pi), as described on p. 2 of [1]//
			///////////////////////////////////////////////////////////

			network->nodes[i].theta = 2.0 * M_PI * ran2(&network->network_properties.seed);
			//printf("Theta: %5.5f\n", network->nodes[i].theta);

			if (network->network_properties.dim == 2)
				//CDF derived from PDF identified in (2) of [2]
				network->nodes[i].tau = etaToTau(atan(ran2(&network->network_properties.seed) / tan(network->network_properties.zeta)), network->network_properties.a);
			else if (network->network_properties.dim == 4) {
				/////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~~~~~~Tau~~~~~~~~~~~~~~~~~~~~//
				//CDF derived from PDF identified in (6) of [2]//
				/////////////////////////////////////////////////

				np.rval = ran2(&network->network_properties.seed);
				np.x = etaToTau((M_PI / 2.0) - np.zeta, network->network_properties.a);	//Pick appropriate starting value for tau distribution
				np.a = network->network_properties.a;	//Scaling from 'tau' to 't'
				np.max = 1000;	//Max number of Netwon-Raphson iterations

				newton(&solveTau, &np, &network->network_properties.seed);
				network->nodes[i].tau = np.x;
				
				////////////////////////////////////////////////////
				//~~~~~~~~~~~~~~~~~Phi and Chi~~~~~~~~~~~~~~~~~~~~//	
				//CDFs derived from PDFs identified on p. 3 of [2]//
				////////////////////////////////////////////////////

				//Sample Phi from (0, pi)
				np.x = M_PI / 2.0;
				np.max = 100;

				newton(&solvePhi, &np, &network->network_properties.seed);
				network->nodes[i].phi = np.x;
				//printf("Phi: %5.5f\n", network->nodes[i].phi);

				//Sample Chi from (0, pi)
				network->nodes[i].chi = acosf(1.0 - (2.0 * ran2(&network->network_properties.seed)));
				//printf("Chi: %5.5f\n", network->nodes[i].chi);
			}
			//printf("Tau: %E\n", network->nodes[i].tau);
		}
	}

	printf("\tNodes Successfully Generated.\n");
	return true;
}

//Identify Causal Sets
bool linkNodes(Network *network, bool &use_gpu)
{
	float dt, dx;
	unsigned int idx  = 0;

	for (unsigned int i = 0; i < network->network_properties.N - 1; i++) {
		network->edge_row_start[i] = idx;
		for (unsigned int j = i + 1; j < network->network_properties.N; j++) {
			//Causal Condition (Light Cone)
			dt = tauToEta(network->nodes[j].tau, network->network_properties.a) - tauToEta(network->nodes[i].tau, network->network_properties.a);
			//printf("dt: %5.8f\n", dt);

			//////////////////////////////////////////
			//~~~~~~~~~~~Spatial Distances~~~~~~~~~~//
			//////////////////////////////////////////

			if (network->network_properties.dim == 2)
				//Formula given on p. 2 of [2]
				dx = M_PI - fabs(M_PI - fabs(network->nodes[j].theta - network->nodes[i].theta));
			else if (network->network_properties.dim == 4)
				//Spherical Law of Cosines
				dx = acosf((X1(network->nodes[i].phi) * X1(network->nodes[j].phi)) + 
					    (X2(network->nodes[i].phi, network->nodes[i].chi) * X2(network->nodes[j].phi, network->nodes[j].chi)) + 
					    (X3(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X3(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta)) + 
					    (X4(network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta) * X4(network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta)));
			//printf("dx: %5.8f\n", dx);

			//Core Edge Adjacency Matrix
			if (i < network->network_properties.core_edge_ratio * network->network_properties.N && j < network->network_properties.core_edge_ratio * network->network_properties.N) {
				if (dx > dt)
					network->core_edge_exists[(unsigned int)(i * network->network_properties.core_edge_ratio * network->network_properties.N) + j] = false;
				else
					network->core_edge_exists[(unsigned int)(i * network->network_properties.core_edge_ratio * network->network_properties.N) + j] = true;
			}
						
			//Ignore spacelike (non-causal) relations
			if (dx > dt) continue;
						
			network->edges[idx] = j;
			idx++;

			//Record number of degrees for each node
			network->nodes[j].num_in++;
			network->nodes[i].num_out++;
		}
	}
	//printf("Forward Edges: %u\n", idx);

	printf("\tCausets Successfully Connected.\n");
	return true;
}

//Plot using OpenGL
bool displayNetwork(Node *nodes, unsigned int *edges, int argc, char **argv)
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

//Print to File
bool printNetwork(Network network)
{
	printf("Printing Results to File...\n");

	std::ofstream outputStream;
	std::stringstream sstm;
	
	if (network.network_properties.flags.use_gpu)
		sstm << "Dev" << gpuID << "_";
	else
		sstm << "CPU_";
	sstm << network.network_properties.N << "_";
	sstm << network.network_properties.k << "_";
	sstm << network.network_properties.a << "_";
	sstm << network.network_properties.dim;
	sstm << "-" << (int)time(NULL);
	std::string filename = sstm.str();

	outputStream.open(("./dat/" + filename + ".cset").c_str());
	outputStream << "Causet Simulation\n";

	time_t rawtime;
	struct tm * timeinfo;
	static char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%X %x", timeinfo);
	outputStream << buffer << std::endl;

	outputStream << "\nCauset Input Parameters:" << std::endl;
	outputStream << "------------------------" << std::endl;
	outputStream << "Nodes (N)\t\t\t" << network.network_properties.N << std::endl;
	outputStream << "Expected Average Degrees (k)\t" << network.network_properties.k << std::endl;
	outputStream << "Pseudoradius (a)\t\t" << network.network_properties.a << std::endl;

	outputStream << "\nCauset Calculated Values:" << std::endl;
	outputStream << "--------------------------" << std::endl;
	outputStream << "Maximum Conformal Time (eta_0)\t" << ((M_PI / 2.0) - network.network_properties.zeta) << std::endl;
	outputStream << "Maximum Rescaled Time (tau_0)\t" << etaToTau((M_PI / 2.0) - network.network_properties.zeta, network.network_properties.a) << std::endl;

	outputStream << "\nNetwork Analysis Results:" << std::endl;
	outputStream << "-------------------------" << std::endl;
	outputStream << "Node Position Data:\t\t" << "pos/" << filename << "_POS.cset" << std::endl;
	outputStream << "Node Edge Data:\t\t\t" << "edg/" << filename << "_EDG.cset" << std::endl;
	outputStream << "Degree Distribution Data:\t" << "dst/" << filename << "_DST.cset" << std::endl;

	outputStream << "\nAlgorithmic Performance:" << std::endl;
	outputStream << "--------------------------" << std::endl;

	outputStream.flush();
	outputStream.close();

	std::ofstream dataStream;

	dataStream.open(("./dat/pos/" + filename + "_POS.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N; i++) {
		dataStream << tauToEta(network.nodes[i].tau, network.network_properties.a) << " " << network.nodes[i].theta;
		if (network.network_properties.dim == 4)
			dataStream << " " << network.nodes[i].phi << " " << network.nodes[i].chi;
		dataStream << std::endl;
	}
	dataStream.flush();
	dataStream.close();

	unsigned int idx = 0;
	dataStream.open(("./dat/edg/" + filename + "_EDG.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N - 1; i++) {
		for (unsigned int j = 0; j < network.nodes[i].num_out; j++)
			dataStream << i << " " << network.edges[idx + j] << std::endl;
		idx += network.nodes[i].num_out;
	}
	dataStream.flush();
	dataStream.close();

	dataStream.open(("./dat/dst/" + filename + "_DST.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N; i++)
		dataStream << (network.nodes[i].num_in + network.nodes[i].num_out) << std::endl;
	dataStream.flush();
	dataStream.close();
	
	printf("\tFilename: %s.cset\n", filename.c_str());
	printf("\tCompleted.\n\n");

	return true;
}

//Free Memory
bool destroyNetwork(Network *network)
{
	free(network->nodes);		network->nodes = NULL;		hostMemUsed -= sizeof(Node)  * network->network_properties.N;
	free(network->edges);		network->edges = NULL;		hostMemUsed -= sizeof(Node)  * network->network_properties.N * network->network_properties.k / 2;
	free(network->edge_row_start);	network->edge_row_start = NULL;	hostMemUsed -= sizeof(unsigned int) * (network->network_properties.N - 1);
	free(network->core_edge_exists);	network->core_edge_exists = NULL;	hostMemUsed -= sizeof(bool) * powf(network->network_properties.core_edge_ratio * network->network_properties.N, 2.0);

	cuMemFree(network->d_nodes);	network->d_nodes = NULL;		devMemUsed  -= sizeof(Node)  * network->network_properties.N;
	cuMemFree(network->d_edges);	network->d_edges = NULL;		devMemUsed  -= sizeof(Node)  * network->network_properties.N * network->network_properties.k / 2;

	return true;
}

//Sort nodes temporally by eta coordinate
void quicksort(Node *nodes, double a, int low, int high)
{
	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) / 2;
		swap(&nodes[low], &nodes[k]);
		key = tauToEta(nodes[low].tau, a);
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (tauToEta(nodes[i].tau, a) <= key))
				i++;
			while ((j >= low) && (tauToEta(nodes[j].tau, a) > key))
				j--;
			if (i < j)
				swap(&nodes[i], &nodes[j]);
		}

		swap(&nodes[low], &nodes[j]);
		quicksort(nodes, a, low, j - 1);
		quicksort(nodes, a, j + 1, high);
	}
}

//Exchange two nodes
void swap(Node *n, Node *m)
{
	Node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}

//Newton-Raphson Method
//Solves Transcendental Equations
void newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed)
{
	double x1;
	double res = 1.0;

	int iter = 0;
	while (abs(res) > np->tol && iter < np->max) {
		res = (*solve)(np);
		//printf("res: %E\n", res);
		if (res != res) {
			printf("NaN Error in Newton-Raphson\n");
			exit(0);
		}

		x1 = np->x + res;
		//printf("x1: %E\n", x1);

		np->x = x1;
		iter++;
	}

	//printf("Newton-Raphson Results:\n");
	//printf("Tolerance: %E\n", np->tol);
	//printf("%d of %d iterations performed.\n", iter, np->max);
	//printf("Residual: %E\n", res);
	//printf("Solution: %E\n", np->x);
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

#endif
