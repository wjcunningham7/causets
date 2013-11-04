#ifndef CAUSET_SUBROUTINES_CUH_
#define CAUSET_SUBROUTINES_CUH_

#include "CausetSubroutines.hpp"
#include "GPUSubroutines.cuh"

//Handles all network generation routines
bool initializeNetwork(Network *network)
{
	printf("Initializing Causet Network...\n");
	if (!createNetwork(network))
		return false;

	float guess = (M_PI / 2.0) - 0.0000001;
	unsigned int max_iter = 10000;

	float eta0 = newton(guess, max_iter, network->network_properties.N, network->network_properties.k, network->network_properties.dim);
	//printf("\tEta0: %5.5f\n", eta0);

	if (!generateNodes(network, eta0, network->network_properties.flags.use_gpu))
		return false;

	int low  = 0;
	int high = network->network_properties.N - 1;

	quicksort(network->nodes, low, high);
	printf("\tQuick sort successfully performed.\n");

	if (!linkNodes(network, network->network_properties.flags.use_gpu))
		return false;

	printf("\tCompleted.\n");
	return true;
}

//Allocates memory for network
bool createNetwork(Network *network)
{
	//Implement subnet_size later
	try {
		network->nodes = (Node*)malloc(sizeof(Node) * network->network_properties.N);
		if (network->nodes == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N;
		//printMemUsed("Memory Allocated for Nodes", hostMemUsed, devMemUsed);

		network->links = (unsigned int*)malloc(sizeof(Node) * network->network_properties.N * network->network_properties.k / 2);
		if (network->links == NULL) throw std::bad_alloc();
		hostMemUsed += sizeof(Node) * network->network_properties.N * network->network_properties.k / 2;

		if (network->network_properties.flags.use_gpu) {
			checkCudaErrors(cuMemAlloc(&network->d_nodes, sizeof(Node) * network->network_properties.N));
			devMemUsed += sizeof(Node) * network->network_properties.N;

			checkCudaErrors(cuMemAlloc(&network->d_links, sizeof(Node) * network->network_properties.N * network->network_properties.k / 2));
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
bool generateNodes(Network *network, float &eta0, bool &use_gpu)
{
	if (use_gpu) {
		if (!generateNodesGPU(network, eta0))
			return false;
	} else {
		for (unsigned int i = 0; i < network->network_properties.N; i++) {
			network->nodes[i].eta = atanf(ran2(&network->network_properties.seed) * tan(eta0));
			//printf("Eta: %5.5f\n", network.nodes[i].eta);

			//Sample Theta from (0, 2pi)
			network->nodes[i].theta = 2.0 * M_PI * ran2(&network->network_properties.seed);
			//printf("Theta: %5.5f\n", network.nodes[i].theta);

			if (network->network_properties.dim == 4) {
				//Sample Phi from (0, pi)
				network->nodes[i].phi = powf((3.0 * M_PI / 2.0) * ran2(&network->network_properties.seed), 1.0 / 3.0);	//Gives phi between [0, pi/2] with O(phi^5) error
				if (ran2(&network->network_properties.seed) > 0.5)	//Reflects half the time to double range to (0, pi)
					network->nodes[i].phi = M_PI - network->nodes[i].phi;
				//printf("Phi: %5.5f\n", network.nodes[i].phi);

				//Sample Chi from (0, pi)
				network->nodes[i].chi = acosf(1.0 - (2.0 * ran2(&network->network_properties.seed)));
				//printf("Chi: %5.5f\n", network.nodes[i].chi);
			}

			network->nodes[i].num_in = 0;
			network->nodes[i].num_out = 0;
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
		for (unsigned int j = i + 1; j < network->network_properties.N; j++) {
			//Causal Condition (Light Cone)
			if (network->network_properties.dim == 2) {

				dt = network->nodes[j].eta - network->nodes[i].eta;
				//printf("dt: %5.5f\n", dt);

				//Periodic Boundary Condition
				dx = M_PI - fabs(M_PI - fabs(network->nodes[j].theta - network->nodes[i].theta));
				//printf("dx: %5.5f\n", dx);
			} else if (network->network_properties.dim == 4) {
				//Transform coordinates so dS is embedded in 5D Euclidean manifold
				//Distance measured there is equal due to  invariance
				dx = sqrt(powf(Z1(network->network_properties.a, network->nodes[j].eta, network->nodes[j].phi) - Z1(network->network_properties.a, network->nodes[i].eta, network->nodes[i].phi), 2) +
					powf(Z2(network->network_properties.a, network->nodes[j].eta, network->nodes[j].phi, network->nodes[j].chi) - Z2(network->network_properties.a, network->nodes[i].eta, network->nodes[i].phi, network->nodes[i].chi), 2) +
					powf(Z3(network->network_properties.a, network->nodes[j].eta, network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta) - Z3(network->network_properties.a, network->nodes[i].eta, network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta), 2) +
					powf(Z4(network->network_properties.a, network->nodes[j].eta, network->nodes[j].phi, network->nodes[j].chi, network->nodes[j].theta) - Z4(network->network_properties.a, network->nodes[i].eta, network->nodes[i].phi, network->nodes[i].chi, network->nodes[i].theta), 2) -
					powf(Z0(network->network_properties.a, network->nodes[j].eta) - Z0(network->network_properties.a, network->nodes[i].eta), 2) +
					powf(network->network_properties.a * (network->nodes[j].eta - network->nodes[i].eta) / sinf(network->nodes[i].eta), 2));
			}
			printf("dx: %5.5f\n", dx);
			
			if (dx > dt) continue;

			network->links[idx] = j;
			idx++;

			network->nodes[j].num_in++;
			network->nodes[i].num_out++;
		}
	}
	//printf("Idx: %5.5f\n", idx);

	printf("\tCausets Successfully Connected.\n");
	return true;
}

//Plot using OpenGL
bool displayNetwork(Node *nodes, unsigned int *links, int argc, char **argv)
{
	/*glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(900, 900);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Causets");
	glutDisplayFunc(display);
	glOrtho(0.0f, 0.01f, 0.0f, 6.3f, -1.0f, 1.0f);
	glutMainLoop();*/

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
	sstm << network.network_properties.N << "_";
	sstm << network.network_properties.k << "_";
	sstm << network.network_properties.dim << "_";
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
	outputStream << "Nodes (N)\t" << network.network_properties.N << std::endl;
	outputStream << "Average Degrees (k)\t" << network.network_properties.k << std::endl;

	outputStream << "\nNetwork Analysis Results:" << std::endl;
	outputStream << "-------------------------" << std::endl;
	outputStream << "Node Position Data:\t" << "pos/" << filename << "_POS.cset" << std::endl;
	outputStream << "Node Link Data:\t" << "lnk/" << filename << "_LNK.cset" << std::endl;
	outputStream << "Degree Distribution Data:\t" << "dst/" << filename << "_DST.cset" << std::endl;

	outputStream << "\nAlgorithmic Performance:" << std::endl;
	outputStream << "--------------------------" << std::endl;

	outputStream.flush();
	outputStream.close();

	std::ofstream dataStream;

	dataStream.open(("./dat/pos/" + filename + "_POS.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N; i++) {
		dataStream << network.nodes[i].eta << " " << network.nodes[i].theta;
		if (network.network_properties.dim == 4)
			dataStream << " " << network.nodes[i].phi << " " << network.nodes[i].chi;
		dataStream << std::endl;
	}
	dataStream.flush();
	dataStream.close();

	unsigned int idx = 0;
	dataStream.open(("./dat/lnk/" + filename + "_LNK.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N - 1; i++) {
		for (unsigned int j = 0; j < network.nodes[i].num_out; j++)
			dataStream << i << " " << network.links[idx + j] << std::endl;
		idx += network.nodes[i].num_out;
	}
	dataStream.flush();
	dataStream.close();

	dataStream.open(("./dat/dst/" + filename + "_DST.cset").c_str());
	for (unsigned int i = 0; i < network.network_properties.N; i++)
		dataStream << (network.nodes[i].num_in + network.nodes[i].num_out) << std::endl;
	dataStream.flush();
	dataStream.close();

	printf("Completed.\n");
	printf("Filename: %s.cset\n\n", filename.c_str());

	return true;
}

bool destroyNetwork(Network *network)
{
	free(network->nodes);	network->nodes = NULL;	hostMemUsed -= sizeof(Node) * network->network_properties.N;
	free(network->links);	network->links = NULL;	hostMemUsed -= sizeof(Node) * network->network_properties.N * network->network_properties.k / 2;

	cuMemFree(network->d_nodes);	network->d_nodes = NULL;	devMemUsed -= sizeof(Node) * network->network_properties.N;
	cuMemFree(network->d_links);	network->d_links = NULL;	devMemUsed -= sizeof(Node) * network->network_properties.N * network->network_properties.k / 2;

	return true;
}

//Newton-Raphson Method
//Solves Transcendental Equation for eta0
//Eta0 in (-pi/2, pi/2)
float newton(float guess, unsigned int &max_iter, unsigned int &N, unsigned int &k, unsigned int &dim)
{
	float res, x0, x1;
	res = 1.0;
	x0 = guess;

	int iter = 0;
	while (fabs(res) > TOL && iter < max_iter) {
		if (dim == 2)
			x1 = x0 - (f2D(x0, N, k) / fprime2D(x0));
		else if (dim == 4)
			x1 = x0 - (f4D(x0) / fprime4D(x0));
		res = x1 - x0;
		x0 = x1;
		iter++;
	}

	return x0;
}

void quicksort(Node *nodes, int low, int high)
{
	int i, j, k;
	float key;

	if (low < high) {
		k = (low + high) / 2;
		swap(&nodes[low], &nodes[k]);
		key = nodes[low].eta;
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (nodes[i].eta <= key))
				i++;
			while ((j >= low) && (nodes[j].eta > key))
				j--;
			if (i < j)
				swap(&nodes[i], &nodes[j]);
		}

		swap(&nodes[low], &nodes[j]);
		quicksort(nodes, low, j - 1);
		quicksort(nodes, j + 1, high);
	}
}

void swap(Node *n, Node *m)
{
	Node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}

inline float f2D(float &x, unsigned int &N, unsigned int &k)
{
	return ((2.0 / M_PI) * (((x / tanf(x)) + logf(1.0 / cosf(x)) - 1.0) / tanf(x))) - ((float)k / N);
}

inline float fprime2D(float &x)
{
	return (2.0 / M_PI) * (((1.0 / tanf(x)) * ((1.0 / tanf(x)) - (x / (sinf(x) * sinf(x))) + tanf(x))) - ((1.0 / (sinf(x) * sinf(x))) * (logf(1.0 / cosf(x)) + (x / tanf(x)) - 1.0)));
}

inline float f4D(float &x)
{
	return (2.0 / (3.0 * M_PI)) * ((12 * ((x / tanf(x)) + logf(1.0 / cosf(x)))) + (((6 * logf(1.0 / cosf(x))) - 5.0) * (1.0 / (cosf(x) * cosf(x)))) - 7.0) / (powf((2.0 + (1.0 / (cosf(x) * cosf(x)))), 2) * tanf(x));
}

inline float fprime4D(float &x)
{
	return (2.0 / (3.0 * M_PI)) * ((-4 * (1.0 / cosf(x) * cosf(x)) * (-7.0 + 12.0 * ((x / tanf(x)) + logf(1.0 / cosf(x))) + (-5.0 + 6.0 * logf(1.0 / cosf(x))) * (1.0 / (cosf(x) * cosf(x))))) / (powf(2.0 + (1.0 / (cosf(x) * cosf(x))), 3))
		- (1.0 / (sinf(x) * sinf(x)) * (-7.0 + 12.0 * ((x / tanf(x)) + logf(1.0 / cosf(x))) + (-5.0 + 6.0 * logf(1.0 / cosf(x))) * (1.0 / (cosf(x) * cosf(x))))) / (powf(2.0 + (1.0 / (cosf(x) * cosf(x))), 2))
		+ (1.0 / (powf(2.0 + (1.0 / (cosf(x) * cosf(x))), 2))) * (1.0 / tanf(x)) * ((6.0 * tanf(x) / (cosf(x) * cosf(x))) + 2.0 * (-5.0 + 6.0 * logf(1.0 / cosf(x))) * tanf(x) / (cosf(x) * cosf(x))) + 12.0 * ((1.0 / tanf(x)) - x * (1.0 / (sinf(x) * sinf(x))) + tanf(x)));
}

inline float Z0(float &a, float &eta)
{
	return a * tanf(eta);
}

inline float Z1(float &a, float &eta, float &phi)
{
	return a * (1.0 / cosf(eta)) * cosf(phi);
}

inline float Z2(float &a, float &eta, float &phi, float &chi)
{
	return a * (1.0 / cosf(eta)) * sinf(phi) * cosf(chi);
}

inline float Z3(float &a, float &eta, float &phi, float &chi, float &theta)
{
	return a * (1.0 / cosf(eta)) * sinf(phi) * sinf(chi) * cosf(theta);
}

inline float Z4(float &a, float &eta, float &phi, float &chi, float &theta)
{
	return a * (1.0 / cosf(eta)) * sinf(phi) * sinf(chi) * sinf(theta);
}

void display()
{
	/*glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_LINES);
		glColor3f(0.0f, 0.0f, 0.0f);
		//Draw Lines
	glEnd();

	glLoadIdentity();
	glutSwapBuffers();*/
}

#endif