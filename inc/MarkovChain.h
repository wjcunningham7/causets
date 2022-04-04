/////////////////////////////
//(C) Will Cunningham 2020 //
//    Perimeter Institute  //
/////////////////////////////

#ifndef MARKOV_CHAIN_H
#define MARKOV_CHAIN_H

#include "Causet.h"
#include "Operations.h"

struct MarkovData {
	MarkovData(std::string _name) : name(_name), dset(0), tau_int(0.0), tau_exp(0.0), tau_err(0.0), mean(0.0), stddev(0.0) {}

	std::vector<double> data;
	std::string name;
	hid_t dset;
	double tau_int;
	double tau_exp;
	double tau_err;
	double mean;
	double stddev;
};

inline float linkAction1(const uint64_t nlink, const uint64_t nrel, const float A, const float B)
{
	return A * nlink + B * (nrel - nlink);
}

inline double relation_pair_action(Bitvector &adj, const int N, const float A, const float B)
{
	uint64_t cnt = 0;
	for (int i = 0; i < N; i++) {	//First element (row)
		for (int j = i + 1; j < N; j++) {	//Second element (col)
			uint64_t r0 = adj[i].read(j);	//First relation
			for (int m = i; m < N; m++) {	//Third element (row)
				for (int n = m + 1; n < N; n++) { //Fourth element (col)
					if (i == m && n <= j) continue;
					uint64_t r1 = adj[m].read(n);	//Second relation
					cnt += r0 ^ r1;
				}
			}
		}
	}

	uint64_t num_relation_pairs = ((uint64_t)N * (N - 1)) >> 1;
	double action = (long double)cnt / num_relation_pairs;
	action *= A;

	return action;
}

inline double bd_action(uint64_t * const cardinalities, Bitvector &adj, const int stdim, const int N, const double epsilon)
{
	memset(cardinalities, 0, sizeof(uint64_t) * N * omp_get_max_threads());

	//Symmetrize 'adj' - copy upper triangle to lower triangle
	for (int i = 1; i < N; i++) { //Row
		for (int j = 0; j < i; j++) { //Column
			if (adj[j].read(i))	//The transpose element
				adj[i].set(j);
			else
				adj[i].unset(j);
		}
	}

	unsigned n = N + (N & 1);
	uint64_t npairs = (uint64_t)n * (n - 1) / 2;
	#ifdef _OPENMP
	#pragma omp parallel for schedule(static, 32) if (npairs > 1024)
	#endif
	for (uint64_t k = 0; k < npairs; k++) {
		unsigned tid = omp_get_thread_num();
		uint64_t i = k / (n - 1);
		uint64_t j = k % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);
		if ((int)j == N) continue;
		//Ignore pairs which are not related
		if (!adj[i].read(j)) continue;

		cardinalities[tid*N+adj[i].partial_vecprod(adj[j], i, j - i + 1)+1]++;
	}

	for (int i = 1; i < omp_get_max_threads(); i++)
		for (int j = 0; j < N; j++)
			cardinalities[j] += cardinalities[i*N+j];
	cardinalities[0] = N;
	double action = calcAction(cardinalities, stdim, epsilon);

	return action;
}

inline double diamond_action(Bitvector &adj, Bitvector &awork, const int N, const float A, const float B)
{
	//Symmetrize 'adj' - copy upper triangle to lower triangle
	for (int i = 1; i < N; i++) { //Row
		for (int j = 0; j < i; j++) { //Column
			if (adj[j].read(i))	//The transpose element
				adj[i].set(j);
			else
				adj[i].unset(j);
		}
	}

	unsigned n = N + (N & 1);
	uint64_t npairs = (uint64_t)n * (n - 1) / 2;
	uint64_t cnt1 = 0, cnt2 = 0;
	#ifdef _OPENMP
	#pragma omp parallel for schedule (static, 32) if (N > omp_get_max_threads() && N > 32) reduction(+ : cnt1, cnt2)
	#endif
	for (uint64_t m = 0; m < npairs; m++) {
		unsigned tid = omp_get_thread_num();
		//Pairs (i,j) span the set of intervals
		uint64_t i = m / (n - 1);
		uint64_t j = m % (n - 1) + 1;
		uint64_t do_map = i >= j ? 1ULL : 0ULL;
		i += do_map * ((((n >> 1) - i) << 1) - 1);
		j += do_map * (((n >> 1) - j) << 1);
		if ((int)j == N) continue;
		//Ignore pairs which are not related
		if (!adj[i].read(j)) continue;

		//Count the cardinality of the interval
		adj[i].clone(awork[tid]);
		awork[tid].setIntersection(adj[j]);
		uint64_t card = awork[tid].partial_count(i, j - i + 1);
		//Ignore intervals which do not contain just two elements
		if (card != 2) continue;

		//Check for other relations in the interval
		//
		//The version of the action we consider uses weight 'A' for diamonds
		//and weight 'B' for 4-element chains
		uint64_t firstbit = awork[tid].next_bit();
		uint64_t internal = adj[firstbit].partial_vecprod(adj[j], firstbit, j - firstbit + 1);
		if (!internal) cnt1++;	//It's a diamond
		else cnt2++;	//It's a 4-element chain
	}

	return A * (double)cnt1 + B * (double)cnt2;
}

inline bool metropolis(Bitvector &adj, Bitvector &links, Bitvector &awork, uint64_t * const cardinalities, double &old_action, const int stdim, const double beta, const double epsilon, const WeightFunction wf, const unsigned N, const double A, const double B, const uint64_t nrel, const uint64_t nlink, uint64_t &num_accept, MersenneRNG &mrng)
{
	double new_action;
	switch (wf) {
	case WEIGHTLESS:
		new_action = 0.0;
		break;
	case BD_ACTION_2D:
	case BD_ACTION_3D:
	case BD_ACTION_4D:
		new_action = bd_action(cardinalities, adj, stdim, N, epsilon);
		break;
	case RELATION:
		new_action = linkAction1(nlink, nrel, A, B);
		break;
	case RELATION_PAIR:
		new_action = relation_pair_action(adj, N, A, B) + B * nrel;
		break;
	case LINK_PAIR:
		new_action = relation_pair_action(links, N, A, B) + B * nlink;
		break;
	case DIAMOND:
		new_action = diamond_action(adj, awork, N, A, B);
		break;
	case ANTIPERCOLATION:
		new_action = (double)(nrel - nlink) * log(2.0);
		break;
	default:
		assert (false);
	}

	double dS = beta * (new_action - old_action);
	bool accept = dS <= 0 || exp(-dS) > mrng.rng();
	if (accept) {
		old_action = new_action;
		num_accept++;
	}
	return accept;
}

bool generateMarkovChain(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx);

void markovSweep(Network * const network, Bitvector &workspace, Bitvector &awork, FastBitset &workspace2, FastBitset &cluster, int * const lengths, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, uint64_t steps_per_sweep, uint64_t &nlink, uint64_t &nrel, uint64_t &num_accept, uint64_t &num_trials, const int stdim, double A, double B);

#ifdef MPI_ENABLED
void replicaExchange(Network * const network, float * const betas, uint64_t * const num_swaps, uint64_t * const num_swap_attempts, uint64_t &nlink, uint64_t &nrel, int rank, int num_active_procs);
#endif

void recordMarkovObservables(Network * const network, Bitvector &workspace, FastBitset &workspace2, FastBitset &cluster, int * const lengths, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, std::vector<double> &acorr, std::vector<double> &lags, std::vector<MarkovData> &observables, uint64_t nlink, uint64_t nrel, uint64_t npairs, bool activeproc, bool update_autocorr, bool &acorr_converged, hid_t &file, hid_t &dspace, hid_t &mspace, hid_t &plist, hsize_t * const rowdim, hsize_t * rowoffset);

//Identify random graph elements
std::pair<unsigned,unsigned> randomLink(Bitvector &links, FastBitset &workspace, MersenneRNG &mrng, const uint64_t nlinks);

std::pair<unsigned, unsigned> randomAntiRelation(Bitvector &adj, Bitvector &links, Bitvector &workspace, const int N, MersenneRNG &mrng, uint64_t &nrel);

std::pair<unsigned, unsigned> randomAntiLink(Bitvector &adj, Bitvector &links, Bitvector &workspace, const int N, MersenneRNG &mrng, uint64_t &nrel);

//Candidates for updates
void linkMove1(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B);

void linkMove2(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B);

void linkMove3(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B);

void matrixMove1(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B);

void matrixMove2(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B);

void orderMove(Bitvector &adj, Bitvector &links, Bitvector &awork, std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> Uwork, std::vector<unsigned> Vwork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B);

void clusterMove1(Bitvector &adj, Bitvector &links, FastBitset &cluster, Bitvector &workspace, Bitvector &awork, FastBitset &workspace2, int * const &lengths, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B);

void clusterMove2(Bitvector &adj, Bitvector &links, Bitvector &awork, std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B);

#endif
