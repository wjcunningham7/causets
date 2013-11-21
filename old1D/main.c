#include "main.h"

int main(int argc, char **argv)
{
	printf("Initialization\n");
	//Initialize RNG
	ran2(&idum);

	//Initialize Variables
	parseArgs(argc, argv);

	//Create Nodes
	struct node *nodes = NULL;
	nodes = (struct node*)malloc(sizeof(struct node) * N);
	assert(nodes != NULL);

	printf("Newton Method\n");
	//Determine Eta0
	float eta0 = newton((M_PI / 2.0) - 0.0000001, 10000);
	//printf("Eta0:    %E\n", eta0);
	//printf("f(eta0): %E\n", f(eta0));

	//Poisson Sprinkling
	printf("Poisson Sprinkling\n");
	unsigned int i, j;
	for (i = 0; i < N; i++) {
		nodes[i].eta   = atan(ran2(&idum) * tan(eta0));
		nodes[i].t     = acosh(1.0 / cos(nodes[i].eta));
		nodes[i].theta = 2.0 * M_PI * ran2(&idum);

		//printf("i %u\teta %f\ttheta %f\n", i, nodes[i].eta, nodes[i].theta);

		nodes[i].numin = 0;
		nodes[i].numout = 0;
	}

	//Order Nodes Temporally
	printf("Quicksort\n");
	quicksort(nodes, 0, N - 1);

	//Allocate Link Structures
	printf("Causets\n");
	unsigned int M = N * D / 2;
	unsigned int delta = 1000;

	unsigned int *indeg = NULL;
	indeg = (unsigned int*)malloc(sizeof(unsigned int) * (M + delta));
	assert(indeg != NULL);

	unsigned int *outdeg = NULL;
	outdeg = (unsigned int*)malloc(sizeof(unsigned int) * (M + delta));
	assert(outdeg != NULL);

	//Identify Causets
	float dx, dt;
	unsigned int inIdx  = 0;
	unsigned int outIdx = 0;
	for (i = 0; i < N - 1; i++) {
		//Look forward in time from node i to node j
		for (j = i + 1; j < N; j++) {
			//Do they lie within each other's light cones?
			dx = M_PI - fabs(M_PI - fabs(nodes[j].theta - nodes[i].theta));
			dt = nodes[j].eta - nodes[i].eta;
			if (dx > dt)
				continue;

			//printf("dx: %f\tdt: %f\n", dx, dt);

			//Check for free memory
			if (inIdx == M + delta) {
				printf("Not enough memory allocated!  Increase 'delta' and try again.\n");
				exit(-1);
			}

			//In-Degrees
			indeg[inIdx] = i;
			inIdx++;
			nodes[j].numin++;

			//Out-Degrees
			outdeg[outIdx] = j;
			outIdx++;
			nodes[i].numout++;
		}
	}
	//printf("In: %u\tOut: %u\n", inIdx, outIdx);

	//Print Results to File
	printf("Printing Results\n");
	FILE *outfile;
	outfile = fopen("locations.txt", "w");
	if (outfile == NULL) {
		printf("Error opening file: locations.txt\n");
		exit(1);
	}

	for (i = 0; i < N; i++)
		fprintf(outfile, "%5.9f %5.9f\n", nodes[i].eta, nodes[i].theta);
	fclose(outfile);

	outfile = fopen("connections.txt", "w");
	if (outfile == NULL) {
		printf("Error opening file: connections.txt\n");
		exit(1);
	}

	outIdx = 0;
	for (i = 0; i < N - 1; i++) {
		for (j = 0; j < nodes[i].numout; j++)
			fprintf(outfile, "%u %u\n", i, outdeg[outIdx+j]);
		outIdx += nodes[i].numout;
	}

	fclose(outfile);

	outfile = fopen("limits.txt", "w");
	if (outfile == NULL) {
		printf("Error opening file: limits.txt\n");
		exit(1);
	}

	fprintf(outfile, "Spacetime Patch Limits:\n");
	fprintf(outfile, "---------------------\n");
	fprintf(outfile, "Eta:   (0, %5.9f)\n", eta0);
	fprintf(outfile, "Theta: (0, %5.9f)\n", (2.0 * M_PI));
	fprintf(outfile, "Connections: %u\n", outIdx);
	fclose(outfile);

	outfile = fopen("distribution.txt", "w");
	if (outfile == NULL) {
		printf("Error opening file: distribution.txt\n");
		exit(1);
	}
	
	for (i = 0; i < N; i++)
		fprintf(outfile, "%u\n", (nodes[i].numin + nodes[i].numout));
	fclose(outfile);

	//Free Memory
	free(nodes);	nodes  = NULL;
	free(indeg);	indeg  = NULL;
	free(outdeg);	outdeg = NULL;

	printf("Successful Exit.\n");
}

//Newton-Raphson Numerical Algorithm
//Solves a transcendental equation for eta0
//(D/N) = (2 / PI) * ((eta0 / tan(eta0)) + ln(sec(eta0)) - 1) / tan(eta0)
//Note eta0 must lie in the region (-PI/2, PI/2)
float newton(float guess, int max_iter)
{
	float res, x0, x1;
	res = 1.0;
	x0 = guess;

	//x_1 = x_0 - (f(x_0) / f'(x_0))
	int iter = 0;
	while (abs(res) > TOL && iter < max_iter) {
		x1 = x0 - (f(x0) / f_prime(x0));
		res = x1 - x0;
		x0 = x1;
		iter++;
	}

	if (x0 < M_PI / 4.0) {
		printf("Newton method found incorrect solution!  Try different guess.\n");
		exit(0);
	}

	return x0;
}

float f(float x)
{
	return ((2.0 / M_PI) * (((x / tan(x)) + log(1.0 / cos(x)) - 1.0) / tan(x))) - ((float)D / N);
}

//Found using Wolfram Mathematica 8
float f_prime(float x)
{
	return (2.0 / M_PI) * (((1.0 / tan(x)) * ((1.0 / tan(x)) - (x / (sin(x) * sin(x))) + tan(x))) - ((1.0 / (sin(x) * sin(x))) * (log(1.0 / cos(x)) + (x / tan(x)) - 1.0)));
}

void quicksort(struct node *array, int low, int high)
{
	int i, j, k;
	float key;
	if (low < high) {
		k = (low + high) / 2;
		swap(&array[low], &array[k]);
		key = array[low].eta;
		i = low + 1;
		j = high;

		while (i <= j) {
			while ((i <= high) && (array[i].eta <= key))
				i++;
			while ((j >= low) && (array[j].eta > key))
				j--;
			if (i < j)
				swap(&array[i], &array[j]);
		}

		swap(&array[low], &array[j]);
		quicksort(array, low, j - 1);
		quicksort(array, j + 1, high);
	}
}

void swap(struct node *n, struct node *m) {
	struct node temp;
	temp = *n;
	*n = *m;
	*m = temp;
}