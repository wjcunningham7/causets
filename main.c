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
	float eta0 = newton(M_PI / 4.0, 10000);

	//Poisson Sprinkling
	printf("Poisson Sprinkling\n");
	float timescale = M_PI;
	float anglescale = 2.0 * M_PI;
	unsigned int i, j;
	
	for (i = 0; i < N; i++) {
		nodes[i].eta   = atan(ran2(&idum) * tan(eta0)) * timescale;
		nodes[i].t     = acosh(1.0 / cos(nodes[i].eta));
		nodes[i].theta = ran2(&idum) * anglescale;

		nodes[i].numin = 0;
		nodes[i].numout = 0;
	}

	//Order Nodes Temporally
	printf("Quicksort\n");
	quicksort(nodes, 0, N - 1);
	
	//Allocate Link Structures
	printf("Causets\n");
	unsigned int M = N * D / 2;

	unsigned int *indeg = NULL;
	indeg = (unsigned int*)malloc(sizeof(unsigned int) * M);
	assert(indeg != NULL);

	unsigned int *outdeg = NULL;
	outdeg = (unsigned int*)malloc(sizeof(unsigned int) * M);
	assert(outdeg != NULL);

	//Identify Causets
	unsigned int dx, dt;
	unsigned int inIdx  = 0;
	unsigned int outIdx = 0;
	printf("M: %d\n", M);
	for (i = 0; i < N - 1; i++) {
		//Look forward in time from node i to node j
		for (j = i + 1; j < N; j++) {
			//Do they lie within each other's light cones?
			dx = M_PI - abs(M_PI - abs(nodes[j].theta - nodes[i].theta));
			dt = nodes[j].eta - nodes[i].eta;
			if (dx > dt)
				continue;

			//In-Degrees
			printf("inIdx: %d\n", inIdx);
			indeg[inIdx] = i;
			inIdx++;
			nodes[j].numin++;

			//Out-Degrees
			/*outdeg[outIdx] = j;
			outIdx++;
			nodes[i].numout++;*/
		}
	}
	printf("Got Here!\n");
	exit(0);

	//Print Results to File
	printf("Printing Results\n");
	FILE *outfile;
	outfile = fopen("connections.txt", "w");
	if (outfile == NULL) {
		printf("Error opening file: connections.txt\n");
		exit(1);
	}

	outIdx = 0;
	for (i = 0; i < N - 1; i++) {
		for (j = 0; j < nodes[i].numout; j++)
			fprintf(outfile, "%d %d\n", i, outdeg[outIdx+j]);
		outIdx += nodes[i].numout - 1;
	}

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
