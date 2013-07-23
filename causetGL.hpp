#ifndef CAUSETGL_HPP
#define CAUSETGL_HPP

#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <string>

#include <GL/glut.h>

//Number of Nodes
unsigned int N;

//Number of Links
unsigned int K;

//Used to rescale data to viewport
unsigned int sizeFactor = 100;

//Locations of Nodes
float** loc;

//Links between Nodes
unsigned int** con;

void parseArgs(int argc, char** argv);
void initData();
void initGL();
void display();
void resize(int w, int h);

void parseArgs(int argc, char** argv)
{
	int c, longIndex;
	static const char *optString = ":N:K:F:h";
	static const struct option longOpts[] = {{ "help", no_argument, NULL, 'h' }};

	while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
		switch (c) {
			case 'N':
				N = atoi(optarg);
				break;
			case 'K':
				K = atoi(optarg);
				break;
			case 'F':
				sizeFactor = atoi(optarg);
				break;
			case 'h':
				printf("Add help menu!\n");
				break;
			case ':':
				fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], optopt);
				break;
			case '?':
				fprintf(stderr, "%s: option '-%c' is not recognized: ignored\n", argv[0], optopt);
				break;
		}
	}
}

#endif
