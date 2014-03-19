#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ran2.h"

#define TOL (10^-6)

//Global Variables
unsigned int N = 100;
unsigned int D = 5;
long idum = -12345L;

//Data Structures
struct node {
	float eta;
	float t;
	float theta;

	unsigned int numin;
	unsigned int numout;
};	

//Function Prototypes
void parseArgs(int argc, char** argv);
void printNodes(struct node *nodes);
float newton(float guess, int max_iter);
float f(float x);
float f_prime(float x);
void quicksort(struct node *array, int low, int high);
void swap(struct node *n, struct node *m);

//Parse Command Line Arguments
void parseArgs(int argc, char** argv)
{
	int c, longIndex;
	static const char *optString = ":N:D:S:h";
	static const struct option longOpts[] = {{ "help", no_argument, NULL, 'h' }};

	while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
		switch (c) {
			case 'N':
				N = atoi(optarg);
				break;
			case 'D':
				D = atoi(optarg);
				break;
			case 'S':
				idum = atoi(optarg);
				ran2(&idum);
				break;
			case 'h':
				printf("\nUsage  : CausalSets [options]\n\n");
				printf("CausalSets Options..................\n");
				printf("====================================\n");
				printf("Flag:\tParam.:\t\tVariable:\t\t\tSuggested Values:\n");
				printf(" -N\t<int>\t\tNumber of Nodes\t\t\t100-1000\n");
				printf(" -D\t<int>\t\tAverage Degrees of Freedom\t5-20\n");
				printf(" -S\t<int>\t\tRandom Seed\t\t\tN/A\n\n");
				exit(0);
			case ':':
				fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], optopt);
				break;
			case '?':
				fprintf(stderr, "%s: option '-%c' is not recognized: ignored\n", argv[0], optopt);
				break;
		}
	}
}

void printNodes(struct node *nodes)
{
	unsigned int i;
	for (i = 0; i < N; i++)
		printf("%3.3f, %3.3f\n", nodes[i].eta, nodes[i].theta);
}
