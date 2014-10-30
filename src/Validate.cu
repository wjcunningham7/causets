#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Debug:  Future vs Past Edges in Adjacency List
//O(1) Efficiency
void compareAdjacencyLists(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int i, j;
	for (i = 0; i < 20; i++) {
		printf("\nNode i: %d\n", i);

		printf("Forward Connections:\n");
		if (edges.future_edge_row_start[i] == -1)
			printf("\tNo future connections.\n");
		else {
			for (j = 0; j < nodes.k_out[i] && j < 10; j++)
				printf("%d ", edges.future_edges[edges.future_edge_row_start[i]+j]);
			printf("\n");
		}

		printf("Backward Connections:\n");
		if (edges.past_edge_row_start[i] == -1)
			printf("\tNo past connections.\n");
		else {
			for (j = 0; j < nodes.k_in[i] && j < 10; j++)
				printf("%d ", edges.past_edges[edges.past_edge_row_start[i]+j]);
			printf("\n");
		}
	
		fflush(stdout);
	}
}

//Debug:  Future and Past Adjacency List Indices
//O(1) Effiency
void compareAdjacencyListIndices(const Node &nodes, const Edge &edges)
{
	if (DEBUG) {
		//No null pointers
		assert (edges.past_edges != NULL);
		assert (edges.future_edges != NULL);
		assert (edges.past_edge_row_start != NULL);
		assert (edges.future_edge_row_start != NULL);
	}

	int max1 = 20;
	int max2 = 100;
	int i, j;

	printf("\nFuture Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.future_edge_row_start[i]);
	printf("\nPast Edge Indices:\n");
	for (i = 0; i < max1; i++)
		printf("%d\n", edges.past_edge_row_start[i]);
	fflush(stdout);

	int next_future_idx = -1;
	int next_past_idx = -1;

	for (i = 0; i < max1; i++) {
		printf("\nNode i: %d\n", i);

		printf("Out-Degrees: %d\n", nodes.k_out[i]);
		if (edges.future_edge_row_start[i] == -1) {
			printf("Pointer: 0\n");
		} else {
			for (j = 1; j < max2; j++) {
				if (edges.future_edge_row_start[i+j] != -1) {
					next_future_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.future_edge_row_start[i+next_future_idx] - edges.future_edge_row_start[i]));
		}

		printf("In-Degrees: %d\n", nodes.k_in[i]);
		if (edges.past_edge_row_start[i] == -1)
			printf("Pointer: 0\n");
		else {
			for (j = 1; j < max2; j++) {
				if (edges.past_edge_row_start[i+j] != -1) {
					next_past_idx = j;
					break;
				}
			}
			printf("Pointer: %d\n", (edges.past_edge_row_start[i+next_past_idx] - edges.past_edge_row_start[i]));
		}
		fflush(stdout);
	}
}

//Generate confusion matrix for geodesic distances in universe with matter
//Save matrix values as well as d_theta and d_eta to file
bool validateEmbedding(EVData &evd, const Node &nodes, const Edge &edges, const int &N_tar, const double &N_emb, const int &N_res, const float &k_res, const int &dim, const Manifold &manifold, const double &a, const double &alpha, long &seed, Stopwatch &sValidateEmbedding, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose)
{
	if (DEBUG) {
		assert (N_tar > 0);
		assert (dim == 3);
		assert (manifold == DE_SITTER);
		assert (universe);	//Just for now
	}

	uint64_t stride = (uint64_t)N_tar * (N_tar - 1) / (static_cast<uint64_t>(N_emb) << 1);
	uint64_t vec_idx;
	uint64_t k;
	int do_map;
	int i, j;

	stopwatchStart(&sValidateEmbedding);

	try {
		evd.confusion = (double*)malloc(sizeof(double) * 4);
		if (evd.confusion == NULL)
			throw std::bad_alloc();
		memset(evd.confusion, 0, sizeof(double) * 4);
		hostMemUsed += sizeof(double) * 4;

		evd.tn = (float*)malloc(sizeof(float) * (static_cast<uint64_t>(N_emb) << 1));
		if (evd.tn == NULL)
			throw std::bad_alloc();
		memset(evd.tn, 0, sizeof(float) * (static_cast<uint64_t>(N_emb) << 1));
		hostMemUsed += sizeof(float) * (static_cast<uint64_t>(N_emb) << 1);

		evd.fp = (float*)malloc(sizeof(float) * (static_cast<uint64_t>(N_emb) << 1));
		if (evd.fp == NULL)
			throw std::bad_alloc();
		memset(evd.fp, 0, sizeof(float) * (static_cast<uint64_t>(N_emb) << 1));
		hostMemUsed += sizeof(float) * (static_cast<uint64_t>(N_emb) << 1);

		memoryCheckpoint(hostMemUsed, maxHostMemUsed, devMemUsed, maxDevMemUsed);
		if (verbose)
			printMemUsed("for Embedding Validation", hostMemUsed, devMemUsed);
	} catch (std::bad_alloc) {
		fprintf(stderr, "Memory allocation failure in %s on line %d!\n", __FILE__, __LINE__);
		return false;
	}

	for (k = 0; k < static_cast<uint64_t>(N_emb); k++) {
		vec_idx = k * stride + 1;
		i = static_cast<int>(vec_idx / (N_tar - 1));
		j = static_cast<int>(vec_idx % (N_tar - 1) + 1);
		do_map = i >= j;

		if (j < N_tar >> 1) {
			i = i + do_map * ((((N_tar >> 1) - i) << 1) - 1);
			j = j + do_map * (((N_tar >> 1) - j) << 1);
		}

		distanceDS(&evd, nodes.c.sc[i], nodes.id.tau[i], nodes.c.sc[j], nodes.id.tau[j], dim, manifold, a, alpha, universe);
	}

	//Normalize
	float A1T = N_res * k_res / 2;
	double A1S = N_tar * (N_tar - 1) / 2 - A1T;
	evd.confusion[0] /= A1S;
	evd.confusion[1] /= A1T;
	evd.confusion[2] /= A1T;
	evd.confusion[3] /= A1S;

	stopwatchStop(&sValidateEmbedding);

	printf("\tCalcuated Confusion Matrix.\n");
	printf_cyan();
	printf("\t\tTrue  Positives: %f\n", evd.confusion[0]);
	printf("\t\tFalse Negatives: %f\n", evd.confusion[1]);
	printf_red();
	printf("\t\tTrue  Negatives: %f\n", evd.confusion[2]);
	printf("\t\tFalse Positives: %f\n", evd.confusion[3]);
	printf_std();
	fflush(stdout);

	if (verbose) {
		printf("\t\tExecution Time: %5.6f sec\n", sValidateEmbedding.elapsedTime);
		fflush(stdout);
	}

	return true;
}

//Write Node Coordinates to File
//O(num_vals) Efficiency
bool printValues(const Node &nodes, const int num_vals, const char *filename, const char *coord)
{
	if (DEBUG) {
		//No null pointers
		assert (filename != NULL);
		assert (coord != NULL);

		//Variables in correct range
		assert (num_vals > 0);
	}

	try {
		std::ofstream outputStream;
		outputStream.open(filename);
		if (!outputStream.is_open())
			throw CausetException("Failed to open file in 'printValues' function!\n");

		int i;
		for (i = 0; i < num_vals; i++) {
			if (strcmp(coord, "tau") == 0)
				outputStream << nodes.id.tau[i] << std::endl;
			else if (strcmp(coord, "eta") == 0)
				outputStream << nodes.c.sc[i].w << std::endl;
				//outputStream << nodes.c.hc[i].x << std::endl;
			else if (strcmp(coord, "theta") == 0)
				outputStream << nodes.c.sc[i].x << std::endl;
			else if (strcmp(coord, "phi") == 0)
				outputStream << nodes.c.sc[i].y << std::endl;
			else if (strcmp(coord, "chi") == 0)
				outputStream << nodes.c.sc[i].z << std::endl;
			else
				throw CausetException("Unrecognized value in 'coord' parameter!\n");
		}
	
		outputStream.flush();
		outputStream.close();
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		return false;
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		return false;
	}

	return true;
}
