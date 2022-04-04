/////////////////////////////
//(C) Will Cunningham 2020 //
//    Perimeter Institute  //
/////////////////////////////

#include "MarkovChain.h"
#include "Subroutines.h"
#include "Measurements.h"

bool generateMarkovChain(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx)
{
	#if DEBUG
	assert (network != NULL);
	assert (ca != NULL);
	assert (cp != NULL);
	assert (bm != NULL);
	assert (network->network_properties.gt == RGG || network->network_properties.gt == KR_ORDER || network->network_properties.gt == RANDOM || network->network_properties.gt == RANDOM_DAG || network->network_properties.gt == ANTICHAIN || network->network_properties.gt == _2D_ORDER);
	#endif

	/* Initialization */

	if (!network->network_properties.flags.mcmc)
		return true;

	int rank = network->network_properties.cmpi.rank;
	int nprocs = network->network_properties.cmpi.num_mpi_threads;
	bool activeproc = true;
	int num_active_procs = 1;

	printf_mpi(rank, "\nGenerating Markov Chain...\n");
	fflush(stdout);

	#ifdef MPI_ENABLED
	bool select_optimal_temps = false;

	if (nprocs > 1 && !network->network_properties.flags.exchange_replicas)
		assert (false);
	#endif

	/* Replica exchange initialization */
	float *betas = NULL;
	uint64_t *num_swaps = NULL, *num_swap_attempts = NULL;
	if (network->network_properties.flags.exchange_replicas) {
		FILE *f = fopen("temperatures.txt", "r");
		unsigned nlines = 0;
		char *line = NULL; size_t len = 0;
		while (getline(&line, &len, f) != -1)
			nlines++;
		if (!nlines) {
			fprintf(stderr, "Input file 'temperatures.txt' is empty!\n");
			return false;
		}
		fseek(f, 0, SEEK_SET);

		betas = MALLOC<float>(nlines);
		unsigned i = 0;
		while (getline(&line, &len, f) != -1)
			betas[i++] = atof(line);
		fclose(f);

		num_active_procs = std::min((int)nlines, nprocs);
		if (rank < num_active_procs)
			network->network_properties.beta = betas[rank];
		else
			activeproc = false;

		num_swaps = MALLOC<uint64_t, true>(num_active_procs * num_active_procs);
		num_swap_attempts = MALLOC<uint64_t, true>(num_active_procs * num_active_procs);
	}

	/* Workspace initialization */

	Bitvector workspace;
	workspace.reserve(network->network_properties.N);
	for (int i = 0; i < network->network_properties.N; i++) {
		FastBitset fb(network->network_properties.N);
		workspace.push_back(fb);
	}

	FastBitset workspace2(network->network_properties.N);
	FastBitset cluster(network->network_properties.N);
	int *lengths = MALLOC<int>(network->network_properties.N);
	memset(lengths, -1, sizeof(int) * network->network_properties.N);

	Bitvector awork;
	awork.reserve(omp_get_max_threads());
	for (int i = 0; i < omp_get_max_threads(); i++) {
		FastBitset fb(network->network_properties.N);
		awork.push_back(fb);
	}

	network->network_observables.cardinalities = MALLOC<uint64_t, true>(network->network_properties.N * omp_get_max_threads());

	std::vector<unsigned> Uwork(network->network_properties.N, 0), Vwork(network->network_properties.N, 0);

	std::vector<double> acorr(network->network_properties.sweeps, 0), lags;
	for (unsigned i = 1; i <= network->network_properties.sweeps; i++)
		lags.push_back(log((double)i));

	/* File I/O initialization */

	hid_t file = 0, group = 0, dspace = 0, mspace = 0, plist;
	hsize_t rowdim[2];
	hsize_t rowoffset[2] = { 0, activeproc ? (hsize_t)rank : 0 };
	hsize_t nrow = network->network_properties.sweeps;
	hsize_t ncol = num_active_procs;
	std::string filename = network->network_properties.datdir;
	std::string groupname = "markov";
	filename.append("observables.h5");
	#ifdef MPI_ENABLED
	if (!rank)
		save_h5_matrix_init_mpi(filename.c_str(), groupname.c_str(), file, group);
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	//Markov chain data
	std::vector<MarkovData> observables;
	observables.push_back(MarkovData("action"));
	observables.push_back(MarkovData("ordering_fraction"));
	observables.push_back(MarkovData("link_fraction"));
	observables.push_back(MarkovData("height"));
	observables.push_back(MarkovData("aspect_ratio"));
	observables.push_back(MarkovData("minimal_elements"));
	observables.push_back(MarkovData("maximal_elements"));

	//These are the datasets ; one for each observable
	std::vector<std::pair<hid_t,std::string>> dsets;
	for (size_t i = 0; i < observables.size(); i++)
		dsets.push_back(std::make_pair(observables[i].dset, observables[i].name));
	save_h5_matrix_init<double>(filename.c_str(), groupname.c_str(), dsets, file, group, dspace, mspace, rowdim, nrow, ncol);
	for (size_t i = 0; i < observables.size(); i++)
		observables[i].dset = dsets[i].first;
	bool acorr_converged = false;
	rowdim[1] = 1;	//Each process only writes to one column

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	plist = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
	#else
	plist = H5P_DEFAULT;
	#endif

	/* Monte Carlo variable initialization */

	uint64_t npairs = (uint64_t)network->network_properties.N * (network->network_properties.N - 1) >> 1;
	uint64_t steps_per_sweep = MARKOV_SAMPLE_ALL ? 1 : npairs / 4;

	//Prepare the adjacency and link matrices with
	//transitive closure/reduction operations
	//and count the number of links and relations
	int stdim = 0;	//Used for the BD action
	uint64_t nrel = 0, nlink = 0;	//nrel = links + transitive relations
	if (network->network_properties.gt == RANDOM_DAG) {
		for (int i = 0; i < network->network_properties.N; i++)
			network->adj[i].clone(network->links[i]);
		transitiveReduction(network->adj, network->links, network->network_properties.N);
		for (int i = 0; i < network->network_properties.N - 1; i++)
			nlink += network->adj[i].partial_count(i + 1, network->network_properties.N - i - 1);
		for (int i = 0; i < network->network_properties.N; i++)
			network->adj[i].clone(workspace[i]);
		transitiveClosure(network->adj, workspace, network->network_properties.N);
		for (int i = 0; i < network->network_properties.N - 1; i++)
			nrel += network->adj[i].partial_count(i + 1, network->network_properties.N - i - 1);
	} else {
		for (int i = 0; i < network->network_properties.N - 1; i++)
			nrel += network->adj[i].partial_count(i + 1, network->network_properties.N - i - 1);
		transitiveReduction(network->links, network->adj, network->network_properties.N);
		for (int i = 0; i < network->network_properties.N - 1; i++)
			nlink += network->links[i].partial_count(i + 1, network->network_properties.N - i - 1);
	}

	if (!rank) printf_cyan();
	printf_mpi(rank, "\tWeight Function: %s\n", wf_strings[network->network_properties.wf]);
	printf_mpi(rank, "\tSweeps: %u\n", network->network_properties.sweeps);
	if (!network->network_properties.flags.exchange_replicas) {
		printf_mpi(rank, "\tBeta: %f\n", network->network_properties.beta);
		printf_mpi(rank, "\tInitial Links: %" PRIu64 "\n", nlink);
		printf_mpi(rank, "\tInitial Transitive Relations: %" PRIu64 "\n", nrel - nlink);
	}

	//Details of the action

	//The coefficients for the non-BD actions
	const double A = network->network_properties.couplings[0];
	const double B = network->network_properties.couplings[1];
	switch (network->network_properties.wf) {
	case WEIGHTLESS:
		//The action is S=0; entropy determines measure
		network->network_observables.action = 0.0;
		break;
	case BD_ACTION_2D:
	case BD_ACTION_3D:
	case BD_ACTION_4D:
		//Use the smeared Benincasa-Dowker action
		stdim = network->network_properties.wf + 1;
		network->network_observables.action = bd_action(network->network_observables.cardinalities, network->adj, stdim, network->network_properties.N, network->network_properties.epsilon);
		printf_mpi(rank, "\tEpsilon: %f\n", network->network_properties.epsilon);
		break;
	case RELATION:
		//The relation action is S = A * n_1 + B * sum(n_i, 2, N-2)
		//This makes 'A' weight the links and 'B' weight the transitive relations
		//When A=B, it is purely a relation action; when B=0 it is a link action

		network->network_observables.action = A * nlink + B * (nrel - nlink);
		printf_mpi(rank, "\tAction Coefficients: (A,B)=(%.8f,%.8f)\n", A, B);
		break;
	case RELATION_PAIR:
		//The relation pair action is S = A*r*r' + B*R
		//A controls the interacting term (+1 for a pair both existing
		//or not existing and -1 for one existing and one not existing)
		//B controls the non-interacting term (number of relations)

		network->network_observables.action = relation_pair_action(network->adj, network->network_properties.N, A, B) + B * nrel;
		printf_mpi(rank, "\tAction Coefficients: (A,B)=(%.8f,%.8f)\n", A, B);
		break;
	case LINK_PAIR:
		//The link pair action is S = A*l*l' + B*L
		//This works similar to the relation pair action

		network->network_observables.action = relation_pair_action(network->links, network->network_properties.N, A, B) + B * nlink;
		printf_mpi(rank, "\tAction Coefficients: (A,B)=(%.8f,%.8f)\n", A, B);
		break;
	case DIAMOND:
		//The diamond action is S = A*D + B*(4-chain)
		//Diamonds are intervals containing two mutually unrelated elements
		//4-chains are intervals containing two mutually related elements

		network->network_observables.action = diamond_action(network->adj, awork, network->network_properties.N, A, B);
		break;
	case ANTIPERCOLATION:
		//The antipercolation weight is used for the action
		//We use S=R-L, where R represents (transitive + non-transitive relations)
		//and L represents the number of links
		//This is equivalent to using the RELATION weight and setting A=0, B=1
		network->network_observables.action = (double)(nrel - nlink) * log(2.0);
		break;
	default:
		fprintf(stderr, "Not implemented!\n");
		return false;
	}

	if (!network->network_properties.flags.exchange_replicas)
		printf_mpi(rank, "\tInitial Action: %f\n", network->network_observables.action);
	if (network->network_properties.flags.cluster_flips)
		printf_mpi(rank, "\tCluster Flip Rate: %f\n", network->network_properties.cluster_rate);
	if (!rank) printf_std();

	//Record initial observables
	recordMarkovObservables(network, workspace, workspace2, cluster, lengths, Uwork, Vwork, acorr, lags, observables, nlink, nrel, npairs, activeproc, false, acorr_converged, file, dspace, mspace, plist, rowdim, rowoffset);

	ProgressBar pb = ProgressBar(network->network_properties.sweeps);
	uint64_t num_accept = 0, num_trials = 0;
	Stopwatch sw = Stopwatch(), sw2;
	stopwatchStart(&sw);

	//Choose temperature steps
	#ifdef MPI_ENABLED
	if (network->network_properties.flags.exchange_replicas && select_optimal_temps) {
		//printf_mpi(rank, "Executing Algorithm: Adaptive Temperature Selection [ REPLICA EXCHANGE MONTE CARLO ]\n");
		printf_mpi(rank, "\n\tSelecting optimal temperature spacings...\n");
		//Tune the spacing between B[i], B[i+1]
		unsigned tuning_sweeps = 1000;
		//double acc_min = 0.18, acc_max = 0.28;
		double acc_min = 0.15, acc_max = 0.5;
		double discount = 0.9, discount_factor = 0.0;
		double max_beta = 100.0;
		double acc;
		printf_mpi(rank, "\t\tB[0] = %f\n", betas[0]);

		int nt_max = num_active_procs;
		for (int i = 0; i < nt_max - 1; i++) {
			num_active_procs = i + 2;
			if (rank <= i + 1)
				activeproc = true;
			else
				activeproc = false;

			double dB = betas[i+1] - betas[i];
			if (dB <= 0.0) {
				dB = betas[i] - betas[i-1];
				betas[i+1] = betas[i] + dB;
				if (activeproc)
					network->network_properties.beta = betas[rank];
			}

			//Thermalize
			for (unsigned sweep = 0; activeproc && sweep < 100; sweep++)
				markovSweep(network, workspace, awork, workspace2, cluster, lengths, Uwork, Vwork, steps_per_sweep, nlink, nrel, num_accept, num_trials, stdim, A, B);

			memset(num_swaps, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
			memset(num_swap_attempts, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
			for (unsigned sweep = 0; sweep < tuning_sweeps; sweep++) {
				if (activeproc)
					markovSweep(network, workspace, awork, workspace2, cluster, lengths, Uwork, Vwork, steps_per_sweep, nlink, nrel, num_accept, num_trials, stdim, A, B);
				replicaExchange(network, betas, num_swaps, num_swap_attempts, nlink, nrel, rank, num_active_procs);
			}
			acc = (double)num_swaps[i*num_active_procs+i+1] / num_swap_attempts[i*num_active_procs+i+1];
			printf_mpi(rank, "\n[%f - %f]\t%f\n", betas[i], betas[i+1], acc);

			discount = 0.9, discount_factor = 0.0;
			while (acc < acc_min || acc > acc_max) {
				if (discount_factor)
					discount = pow(discount, discount_factor);

				if (acc < acc_min) {
					dB = dB * 0.5;
					discount_factor = 1.2;
				} else if (acc > acc_max) {
					dB = dB * 2.0;
					if (discount_factor)
						dB *= discount;
				} else
					break;

				betas[i+1] = betas[i] + dB;
				printf_mpi(rank, "\tNew dB: %f [%s]\n", betas[i+1] - betas[i], acc < acc_min ? "DECREASING" : "INCREASING");
				if (betas[i+1] > max_beta) {
					if (rank >= i + 1)
						activeproc = false;
					for (int j = i + 1; j < num_active_procs; j++)
						betas[j] = 0.0;
					num_active_procs = i + 1;
					break;
				}

				if (activeproc)
					network->network_properties.beta = betas[rank];

				//Thermalize
				for (unsigned sweep = 0; activeproc && sweep < 100; sweep++)
					markovSweep(network, workspace, awork, workspace2, cluster, lengths, Uwork, Vwork, steps_per_sweep, nlink, nrel, num_accept, num_trials, stdim, A, B);

				memset(num_swaps, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
				memset(num_swap_attempts, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
				for (unsigned sweep = 0; sweep < tuning_sweeps; sweep++) {
					if (activeproc)
						markovSweep(network, workspace, awork, workspace2, cluster, lengths, Uwork, Vwork, steps_per_sweep, nlink, nrel, num_accept, num_trials, stdim, A, B);
					replicaExchange(network, betas, num_swaps, num_swap_attempts, nlink, nrel, rank, num_active_procs);
				}
				acc = (double)num_swaps[i*num_active_procs+i+1] / num_swap_attempts[i*num_active_procs+i+1];
				printf_mpi(rank, "\t[%f - %f]\t%f\n", betas[i], betas[i+1], acc);
			}
			//printf_mpi(rank, "[%f - %f]\t%f\n", betas[i], betas[i+1], acc);
			printf_mpi(rank, "\t\tB[%d] = %f\n", i + 1, betas[i+1]);
		}
		num_accept = num_trials = 0;
		memset(num_swaps, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
		memset(num_swap_attempts, 0, sizeof(uint64_t) * num_active_procs * num_active_procs);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	/* Markov Chain */
	for (unsigned sweep = 0; sweep < network->network_properties.sweeps - 1; sweep++) {
		if (activeproc)
			//Perform single sweep
			markovSweep(network, workspace, awork, workspace2, cluster, lengths, Uwork, Vwork, steps_per_sweep, nlink, nrel, num_accept, num_trials, stdim, A, B);

		//Measure and record observables
		recordMarkovObservables(network, workspace, workspace2, cluster, lengths, Uwork, Vwork, acorr, lags, observables, nlink, nrel, npairs, activeproc, !acorr_converged && sweep && !(sweep % ACORR_UPDATE_PERIOD), acorr_converged, file, dspace, mspace, plist, rowdim, rowoffset);

		//Replica exchange
		#ifdef MPI_ENABLED
		if (network->network_properties.flags.exchange_replicas)
			replicaExchange(network, betas, num_swaps, num_swap_attempts, nlink, nrel, rank, num_active_procs);
		MPI_Barrier(MPI_COMM_WORLD);
		#endif

		if (!rank)
			updateProgress(pb, sweep);
	}

	stopwatchStop(&sw);

	if (!rank) {
		completeProgress(pb);
		printf_cyan();
	}
	printf_mpi(rank, "\tMonte Carlo Runtime: %5.6f sec/sweep\n", sw.elapsedTime / network->network_properties.sweeps);
	if (num_active_procs == 1)
		printf_mpi(rank, "\tAcceptance Rate: %.5f%%\n", 100 * (double)num_accept / num_trials);
	if (!rank) printf_std();

	if (num_active_procs == 1) {
		printf_mpi(rank, "\n\tAutocorrelation Times:\n");
		printf_mpi(rank, "\t----------------------\n");
		printf_mpi(rank, "\tObservable\tInt.\t\tExp.\n");
		if (!rank) printf_cyan();
		for (size_t i = 0; i < observables.size(); i++)
			printf_mpi(rank, "\t%s\t\t%f\t%f\n", observables[i].name.substr(0,7).c_str(), observables[i].tau_int, observables[i].tau_exp);
		if (!rank) printf_std();
	}

	//Analysis and thermodynamic variables
	double max_tau_exp = 0.0, max_tau_int = 0.0;
	for (size_t i = 0; i < observables.size(); i++) {
		max_tau_exp = std::max(max_tau_exp, observables[i].tau_exp);
		max_tau_int = std::max(max_tau_int, observables[i].tau_int);
	}
	
	size_t num_discard = acorr_converged ? ceil(20 * max_tau_exp) : 0;
	size_t stride = acorr_converged ? ceil(2 * max_tau_int) : 1;
	size_t num_samples = (observables[0].data.size() - num_discard) / stride;

	if (num_active_procs == 1) {
		printf_mpi(rank, "\n\tStatistical Results:\n");
		printf_mpi(rank, "\t--------------------\n");
		printf_mpi(rank, "\tObservable\tMean\t\tStdDev.\n");
		if (!rank) printf_cyan();
	}

	//Average and standard deviation of each observable
	for (size_t i = 0; i < observables.size(); i++) {
		observables[i].mean = gsl_stats_mean(observables[i].data.data() + num_discard, stride, num_samples);
		observables[i].stddev = gsl_stats_sd(observables[i].data.data() + num_discard, stride, num_samples);
		if (num_active_procs == 1)
			printf_mpi(rank, "\t%s\t\t%f\t%f\n", observables[i].name.substr(0,7).c_str(), observables[i].mean, observables[i].stddev);
	}

	//Specific heat and jackknife error
	double Cv, Cv_err;
	specific_heat(Cv, Cv_err, observables[0].data.data(), observables[0].mean, observables[0].stddev, network->network_properties.beta, num_samples, stride);
	if (num_active_procs == 1)
		printf_mpi(rank, "\tCv\t\t%f\t%f\n", Cv, Cv_err);

	//Reduced free energy and jackknife error
	double FE, FE_err;
	free_energy(FE, FE_err, observables[0].data.data(), observables[0].mean, network->network_properties.beta, num_samples, stride);
	FE /= npairs;
	FE_err /= npairs;
	if (num_active_procs == 1)
		printf_mpi(rank, "\tHFE\t\t%f\t%f\n", -FE, FE_err);

	//Reduced entropy and jackknife error
	double entr, entr_err;
	entropy(entr, entr_err, observables[0].mean, observables[0].stddev, -FE * npairs / network->network_properties.beta, FE_err * npairs / network->network_properties.beta, network->network_properties.beta, npairs, num_samples);
	if (num_active_procs == 1)
		printf_mpi(rank, "\tentropy\t\t%f\t%f\n", entr, entr_err);

	//Susceptibility and jackknife error
	double chi, chi_err;
	specific_heat(chi, chi_err, observables[1].data.data(), observables[1].mean, observables[1].stddev, network->network_properties.beta, num_samples, stride);
	if (num_active_procs == 1) {
		printf_mpi(rank, "\tsuscep.\t\t%f\t%f\n", chi, chi_err);
		printf_std();
	}

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	H5Pclose(plist);
	#endif
	
	if (network->network_properties.flags.exchange_replicas) {
		printf_mpi(rank, "\n\tExchange Statistics:\n");
		printf_mpi(rank, "\t--------------------\n");
		for (int i = 0; i < num_active_procs; i++)
			for (int j = i + 1; j < num_active_procs; j++)
				if (num_swap_attempts[i*num_active_procs+j])
					printf_mpi(rank, "\t[%.5f] <--> [%.5f]:\t%.5f\n", betas[i], betas[j], (float)num_swaps[i*num_active_procs+j] / num_swap_attempts[i*num_active_procs+j]);
		printf_mpi(rank, "\n\tAction Spectrum:\n");
		//printf_mpi(rank, "\n\tHeat Spectrum:\n");
		printf_mpi(rank, "\t--------------\n");
		printf_mpi(rank, "\tBeta\t\tAction\n");
		//printf_mpi(rank, "\tBeta\t\tCv\n");
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
		sleep(1);
		for (int i = 0; i < num_active_procs; i++) {
			if (rank == i) {
				printf("\t%f\t%f\n", network->network_properties.beta, observables[0].mean);
				//printf("\t%f\t%f +/- %f\n", network->network_properties.beta, Cv, Cv_err);
				fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			//sleep(1);
		}
	}

	//Write the refined data as attributes
	//HDF5 does not support asynchronous attribute writing, so
	//we first share the data among all processes
	std::vector<double> attrdata(nprocs, 0.0);
	hid_t aspace, attr;

	rowdim[0] = num_active_procs;
	rowdim[1] = 1;
	rowoffset[0] = rowoffset[1] = 0;
	aspace = H5Screate_simple(num_active_procs > 1 ? 2 : 1, rowdim, NULL);

	for (unsigned i = 0; i < observables.size(); i++) {
		//Integrated autocorrelation times
		#ifdef MPI_ENABLED
		MPI_Allgather(&observables[i].tau_int, 1, MPI_DOUBLE, attrdata.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
		#else
		attrdata[0] = observables[i].tau_int;
		#endif
		attr = H5Acreate(observables[i].dset, "tau_int", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
		H5Aclose(attr);

		//Exponential autocorrelation times
		#ifdef MPI_ENABLED
		MPI_Allgather(&observables[i].tau_exp, 1, MPI_DOUBLE, attrdata.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
		#else
		attrdata[0] = observables[i].tau_exp;
		#endif
		attr = H5Acreate(observables[i].dset, "tau_exp", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
		H5Aclose(attr);

		//Mean values (after filtering)
		#ifdef MPI_ENABLED
		MPI_Allgather(&observables[i].mean, 1, MPI_DOUBLE, attrdata.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
		#else
		attrdata[0] = observables[i].mean;
		#endif
		attr = H5Acreate(observables[i].dset, "mean", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
		H5Aclose(attr);
		
		//Standard deviations (after filtering)
		#ifdef MPI_ENABLED
		MPI_Allgather(&observables[i].stddev, 1, MPI_DOUBLE, attrdata.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
		#else
		attrdata[0] = observables[i].stddev;
		#endif
		attr = H5Acreate(observables[i].dset, "stddev", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
		H5Aclose(attr);
	}

	if (network->network_properties.flags.exchange_replicas) {
		//Temperature values
		attr = H5Acreate(group, "beta", H5T_IEEE_F32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F32LE, betas);
		H5Aclose(attr);
	}

	//Increase the dataspace to two columns
	//The first column is the mean and the second is the standard deviation
	H5Sclose(aspace);
	rowdim[1] = 2;
	aspace = H5Screate_simple(2, rowdim, NULL);
	attrdata.clear();
	attrdata.resize(nprocs * 2);

	//Specific heat
	double data[2] = { Cv, Cv_err };
	#ifdef MPI_ENABLED
	MPI_Allgather(data, 2, MPI_DOUBLE, attrdata.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);
	#else
	memcpy(attrdata.data(), data, sizeof(double) * 2);
	#endif
	attr = H5Acreate(group, "specific_heat", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
	H5Aclose(attr);

	//Free energy
	data[0] = -FE;
	data[1] = FE_err;
	#ifdef MPI_ENABLED
	MPI_Allgather(data, 2, MPI_DOUBLE, attrdata.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);
	#else
	memcpy(attrdata.data(), data, sizeof(double) * 2);
	#endif
	attr = H5Acreate(group, "free_energy", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
	H5Aclose(attr);

	//Entropy
	data[0] = entr;
	data[1] = entr_err;
	#ifdef MPI_ENABLED
	MPI_Allgather(data, 2, MPI_DOUBLE, attrdata.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);
	#else
	memcpy(attrdata.data(), data, sizeof(double) * 2);
	#endif
	attr = H5Acreate(group, "entropy", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
	H5Aclose(attr);
	
	//Susceptibility
	data[0] = chi;
	data[1] = chi_err;
	#ifdef MPI_ENABLED
	MPI_Allgather(data, 2, MPI_DOUBLE, attrdata.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);
	#else
	memcpy(attrdata.data(), data, sizeof(double) * 2);
	#endif
	attr = H5Acreate(group, "susceptibility", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr, H5T_IEEE_F64LE, attrdata.data());
	H5Aclose(attr);

	H5Sclose(aspace);
	save_h5_matrix_close(file, group, dspace, mspace, dsets);

	workspace.clear();
	workspace.shrink_to_fit();

	awork.clear();
	awork.shrink_to_fit();
	
	if (network->network_properties.flags.exchange_replicas) {
		FREE(betas);
		FREE(num_swaps);
		FREE(num_swap_attempts);
	}

	FREE(lengths);
	FREE(network->network_observables.cardinalities);

	printf_mpi(rank, "Task Completed.\n");
	fflush(stdout);

	return true;
}

void markovSweep(Network * const network, Bitvector &workspace, Bitvector &awork, FastBitset &workspace2, FastBitset &cluster, int * const lengths, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, uint64_t steps_per_sweep, uint64_t &nlink, uint64_t &nrel, uint64_t &num_accept, uint64_t &num_trials, const int stdim, double A, double B)
{
	for (uint64_t step = 0; step < steps_per_sweep; step++) {
		if (network->network_properties.flags.cluster_flips && network->network_properties.mrng.rng() < network->network_properties.cluster_rate) {
			if (network->network_properties.gt != _2D_ORDER)
				//clusterMove1 will select a random chain or antichain, invert it,
				//and then perform a Metropolis update. This move does not work with
				//the 2D orders.
				clusterMove1(network->adj, network->links, cluster, workspace, awork, workspace2, lengths, network->network_observables.cardinalities, network->network_properties.mrng, network->network_properties.N, nlink, nrel, network->network_observables.action, num_accept, network->network_properties.wf, stdim, network->network_properties.beta, network->network_properties.epsilon, A, B);
			else
				//clusterMove2 reverses a random range in a total order
				//when we are using 2D Orders.
				clusterMove2(network->adj, network->links, awork, network->U, network->V, Uwork, Vwork, network->network_observables.cardinalities, network->network_properties.mrng, network->network_properties.N, nlink, nrel, network->network_observables.action, num_accept, network->network_properties.wf, stdim, network->network_properties.beta, network->network_properties.epsilon, A, B); 
		} else {
			if (network->network_properties.gt != _2D_ORDER) {
				//matrixMove1 will flip a random entry in the causal matrix
				//while ignoring transitive relations (which would otherwise be added back by the
				//transitive closure). It then implements a Metropolis update.
				matrixMove1(network->adj, network->links, workspace, awork, network->network_observables.cardinalities, network->network_properties.mrng, network->network_properties.N, nlink, nrel, network->network_observables.action, num_accept, network->network_properties.wf, stdim, network->network_properties.beta, network->network_properties.epsilon, A, B);

				//matrixMove2 will flip any random entry; the state is stored in 'link'
				//It then implements a Metropolis update. This move does not work with
				//the 2D orders.
				//matrixMove2(network->adj, network->links, workspace, awork, network->network_observables.cardinalities, network->network_properties.mrng, network->network_properties.N, nlink, nrel, network->network_observables.action, num_accept, network->network_properties.wf, stdim, network->network_properties.beta, network->network_properties.epsilon, A, B);
			} else
				//orderMove will flip a random pair in one of the total orders
				orderMove(network->adj, network->links, awork, network->U, network->V, Uwork, Vwork, network->network_observables.cardinalities, network->network_properties.mrng, network->network_properties.N, nlink, nrel, network->network_observables.action, num_accept, network->network_properties.wf, stdim, network->network_properties.beta, network->network_properties.epsilon, A, B); 
		}

		num_trials++;
	}
}

#ifdef MPI_ENABLED
void replicaExchange(Network * const network, float * const betas, uint64_t * const num_swaps, uint64_t * const num_swap_attempts, uint64_t &nlink, uint64_t &nrel, int rank, int num_active_procs)
{
	for (int i = 0; i < num_active_procs - 1; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		int r0 = -1, r1 = -1;
		if (!rank) {
			//First replica
			r0 = network->network_properties.mrng.rng() * num_active_procs;
			//Second replica
			if (r0 == 0)
				r1 = r0 + 1;
			else if ((int)r0 == num_active_procs - 1)
				r1 = r0 - 1;
			else
				r1 = r0 + (2 * (int)(2 * network->network_properties.mrng.rng()) - 1);
		}
		MPI_Bcast(&r0, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&r1, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (r0 == -1 && r1 == -1) return;

		double other_action = NAN;
		bool accept = false;
		if (rank == r0)
			MPI_Recv(&other_action, 1, MPI_DOUBLE, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		else if (rank == r1)
			MPI_Send(&network->network_observables.action, 1, MPI_DOUBLE, r0, 0, MPI_COMM_WORLD);
		if (rank == r0) {
			//Metropolis step
			double dS = -(network->network_observables.action - other_action) * (betas[(unsigned)r0] - betas[(unsigned)r1]);
			accept = dS <= 0 || exp(-dS) > network->network_properties.mrng.rng();
		}
		MPI_Bcast(&accept, 1, MPI_C_BOOL, r0, MPI_COMM_WORLD);

		if (accept) {
			//Swap the actions, link/relation variables, and link matrices
			if (rank == r0) {
				MPI_Sendrecv_replace(&network->network_observables.action, 1, MPI_DOUBLE, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv_replace(&nlink, 1, MPI_UINT64_T, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv_replace(&nrel, 1, MPI_UINT64_T, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (network->network_properties.gt == _2D_ORDER) {
					MPI_Sendrecv_replace(network->U.data(), network->network_properties.N, MPI_UNSIGNED, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Sendrecv_replace(network->V.data(), network->network_properties.N, MPI_UNSIGNED, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					for (int i = 0; i < network->network_properties.N; i++) {
						MPI_Sendrecv_replace(network->links[i].getAddress(), network->links[i].getNumBlocks(), BlockTypeMPI, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Sendrecv_replace(network->adj[i].getAddress(), network->adj[i].getNumBlocks(), BlockTypeMPI, r1, 0, r1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
				}
			} else if (rank == r1) {
				MPI_Sendrecv_replace(&network->network_observables.action, 1, MPI_DOUBLE, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv_replace(&nlink, 1, MPI_UINT64_T, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv_replace(&nrel, 1, MPI_UINT64_T, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (network->network_properties.gt == _2D_ORDER) {
					MPI_Sendrecv_replace(network->U.data(), network->network_properties.N, MPI_UNSIGNED, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Sendrecv_replace(network->V.data(), network->network_properties.N, MPI_UNSIGNED, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					for (int i = 0; i < network->network_properties.N; i++) {
						MPI_Sendrecv_replace(network->links[i].getAddress(), network->links[i].getNumBlocks(), BlockTypeMPI, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Sendrecv_replace(network->adj[i].getAddress(), network->adj[i].getNumBlocks(), BlockTypeMPI, r0, 0, r0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
				}
			}
			num_swaps[num_active_procs*r0+r1]++;
			num_swaps[num_active_procs*r1+r0]++;
		}
		num_swap_attempts[num_active_procs*r0+r1]++;
		num_swap_attempts[num_active_procs*r1+r0]++;
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
#endif

void recordMarkovObservables(Network * const network, Bitvector &workspace, FastBitset &workspace2, FastBitset &cluster, int * const lengths, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, std::vector<double> &acorr, std::vector<double> &lags, std::vector<MarkovData> &observables, uint64_t nlink, uint64_t nrel, uint64_t npairs, bool activeproc, bool update_autocorr, bool &acorr_converged, hid_t &file, hid_t &dspace, hid_t &mspace, hid_t &plist, hsize_t * const rowdim, hsize_t * rowoffset)
{
	//Update the adjacency matrix before measuring observables
	if (activeproc && network->network_properties.gt == _2D_ORDER) {
		ordered_labeling(network->U, network->V, Uwork, Vwork);
		for (int i = 0; i < network->network_properties.N; i++)
			network->adj[i].reset();
		for (int i = 0; i < network->network_properties.N; i++) {
			for (int j = i + 1; j < network->network_properties.N; j++) {
				if ((Uwork[i] < Uwork[j] && Vwork[i] < Vwork[j]) || (Uwork[j] < Uwork[i] && Vwork[j] < Vwork[i])) {
					network->adj[i].set(j);
					network->adj[j].set(i);
				}
			}
		}
	}

	//Action
	observables[0].data.push_back(network->network_observables.action);

	//Ordering fraction
	double ordering_fraction = (double)nrel / npairs;
	observables[1].data.push_back(ordering_fraction);

	//Link fraction
	double link_fraction = (double)nlink / npairs;
	observables[2].data.push_back(link_fraction);

	//Height
	FastBitset longest_chain(network->network_properties.N);
	int len = 0;
	network->network_observables.longest_chain_length = 1;
	for (int i = 0; activeproc && i < network->network_properties.N - 1; i++) {
		if (!!i && !!network->adj[i].partial_count(0, i)) continue;	//Skip if 'i' not a minimal element
		for (int j = i + 1; j < network->network_properties.N; j++) {
			//Skip if 'j' not a maximal element
			if (j < network->network_properties.N - 1 && !!network->adj[j].partial_count(j + 1, network->network_properties.N - j)) continue;
			if (!network->adj[i].read(j)) continue;	//Skip if the relation (i,j) does not exist
			len = longestChain_v3(network->adj, workspace, cluster, &workspace2, lengths, network->network_properties.N, i, j, 0);
			if (len > network->network_observables.longest_chain_length) {
				network->network_observables.longest_chain_length = len;
				network->network_observables.longest_pair = std::make_pair(i, j);
				cluster.clone(longest_chain);
			}
		}
	}
	observables[3].data.push_back(network->network_observables.longest_chain_length);

	//Aspect ratio
	FastBitset largest_antichain(network->network_properties.N);
	uint64_t seed;
	int wid = 0, largest_width = 1;
	if (!longest_chain.any())	//It is an antichain
		largest_width = network->network_properties.N;
	else if (activeproc) {
		while (longest_chain.any()) {
			longest_chain.unset(seed = longest_chain.next_bit());
			wid = maximalAntichain(cluster, network->adj, network->network_properties.N, seed);
			if (wid > largest_width)
				largest_width = wid;
		}
	}
	double aspect_ratio = (double)network->network_observables.longest_chain_length / largest_width;
	observables[4].data.push_back(aspect_ratio);

	//Minimal, maximal elements
	int num_minimal = 0, num_maximal = 0;
	for (int i = 0; i < network->network_properties.N; i++) {
		if (!i || !network->adj[i].partial_count(0, i))
			num_minimal++;
		if (i == network->network_properties.N - 1 || !network->adj[i].partial_count(i + 1, network->network_properties.N - i))
			num_maximal++;
	}
	double frac_minimal = (double)num_minimal / network->network_properties.N;
	observables[5].data.push_back(frac_minimal);
	double frac_maximal = (double)num_maximal / network->network_properties.N;
	observables[6].data.push_back(frac_maximal);

	//Print to file
	for (size_t i = 0; i < observables.size(); i++) {
		save_h5_matrix_row(activeproc ? &observables[i].data.back() : NULL, observables[i].dset, dspace, mspace, rowdim, rowoffset, plist);
		rowoffset[0]--;
	}
	rowoffset[0]++;

	//Without a flush, if the program is interrupted, the partially
	//written file will be corrupted and unreadable
	//This *does* incur a delay, however.
	if (!MARKOV_SAMPLE_ALL)
		H5Fflush(file, H5F_SCOPE_LOCAL);

	//Estimate autocorrelations
	if (update_autocorr && activeproc) {
		double tau_exp_err;
		acorr_converged = true;
		for (size_t i = 0; i < observables.size(); i++) {
			autocorrelation(observables[i].data.data(), acorr.data(), lags.data(), observables[i].data.size() + 1, observables[i].tau_exp, tau_exp_err, observables[i].tau_int, observables[i].tau_err);
			if (observables[i].tau_err > 0.1)
				acorr_converged = false;
		}
	}
}


std::pair<unsigned,unsigned> randomLink(Bitvector &links, FastBitset &workspace, MersenneRNG &mrng, const uint64_t nlinks)
{
	if (nlinks == 0)
		return std::make_pair((unsigned)links.size(), (unsigned)links.size());

	uint64_t rlink = mrng.rng() * nlinks;
	if (DEBUG_EVOL)
		printf("Choosing random link [%" PRIu64 "]\n", rlink);

	uint64_t counter = 0;
	unsigned row = 0;
	while ((counter += links[row++].count_bits()) <= rlink) {}
	counter -= links[--row].count_bits();

	uint64_t column_index = rlink - counter;
	unsigned column = 0;
	links[row].clone(workspace);
	for (uint64_t i = 0; i <= column_index; i++)
		workspace.unset(column = workspace.next_bit());

	std::pair<unsigned, unsigned> random_link = std::make_pair(row, column);
	if (DEBUG_EVOL) {
		printf("random link: (%u, %u)\n", row, column);
		fflush(stdout);
	}

	return random_link;
}

//Adding the causal relation i->j will produce a new transitive relation
std::pair<unsigned, unsigned> randomAntiRelation(Bitvector &adj, Bitvector &links, Bitvector &workspace, const int N, MersenneRNG &mrng, uint64_t &nrel)
{
	for (int i = 0; i < N; i++)
		adj[i].clone(workspace[i]);

	//This first loop sets A(i,j)=1 when the addition of i->j would produce a new link
	//Afterwards, any A(i,j)=0 indicates a place a new transitive relation can be added
	uint64_t counter;
	for (int i = 0; i < N - 1; i++) {
		for (int j = i + 1; j < N; j++) {
			counter = 0;
			if (adj[i].read(j)) continue;
			//Check for common pairs BOTH are linked to
			//(a) both i,j to the future of k
			for (int k = 0; k < i; k++)
				counter += links[k].read(i) & links[k].read(j);
			//(b) both i,j to the past of k
			for (int k = j + 1; k < N; k++)
				counter += links[i].read(k) & links[j].read(k);
			//Check for common pairs one is linked to, the other is related to
			//(c) k linked to j, k related to i but not linked
			for (int k = 0; k < i; k++)
				counter += links[k].read(j) & adj[k].read(i) & !links[k].read(i);
			//(d) i linked to k, j related to k but not linked
			for (int k = j + 1; k < N; k++)
				counter += (links[i].read(k) & adj[j].read(k) & !links[j].read(k));
			//(e) Check for path completion
			for (int m = 0; m < i; m++)
				for (int n = j + 1; n < N; n++)
					counter += adj[m].read(i) & adj[j].read(n) & links[m].read(n);
			if (!counter) {
				workspace[i].set(j);
				nrel++;
			}
		}
	}

	uint64_t max_rel = (uint64_t)N * (N - 1) >> 1;
	//printf("nrel: %" PRIu64 "\n", nrel);
	if (max_rel == nrel) {
		//printf_dbg("Returning with no anti-relation found.\n");
		return std::make_pair((unsigned)N, (unsigned)N);
	}

	uint64_t rdest = mrng.rng() * (max_rel - nrel);
	unsigned row = 0;
	counter = 0;
	//printf("rdest: %" PRIu64 "\n", rdest);

	while (counter <= rdest) {
		counter += N - row - 1 - workspace[row].partial_count(row + 1, N - row - 1);
		row++;
	}
	row--;
	counter -= N - row - 1 - workspace[row].partial_count(row + 1, N - row - 1);

	uint64_t column_index = rdest - counter;
	unsigned column = 0;

	FastBitset &fb = row == 0 ? workspace[row+1] : workspace[0];
	workspace[row].clone(fb);
	fb.flip();
	for (unsigned i = 0; i <= row; i++)
		fb.unset(i);
	for (unsigned i = 0; i <= column_index; i++)
		fb.unset(column = fb.next_bit());

	std::pair<unsigned, unsigned> random_anti_relation = std::make_pair(row, column);
	if (DEBUG_EVOL) {
		printf("random anti relation: (%u, %u)\n", row, column);
		fflush(stdout);
	}

	return random_anti_relation;
}

//Adding the causal relation i->j will produce a new link
std::pair<unsigned, unsigned> randomAntiLink(Bitvector &adj, Bitvector &links, Bitvector &workspace, const int N, MersenneRNG &mrng, uint64_t &nrel)
{
	for (int i = 0; i < N; i++)
		adj[i].clone(workspace[i]);

	//This first loop sets A(i,j)=1 when the addition of i->j would produce a new relation
	//Afterwards, any A(i,j)=0 indicates a place a new link can be added
	uint64_t counter;
	for (int i = 0; i < N - 1; i++) {
		for (int j = i + 1; j < N; j++) {
			counter = 0;
			if (adj[i].read(j)) continue;
			//Check for common pairs BOTH are linked to
			//(a) both i,j to the future of k
			for (int k = 0; k < i; k++)
				counter += links[k].read(i) & links[k].read(j);
			//(b) both i,j to the past of k
			for (int k = j + 1; k < N; k++)
				counter += links[i].read(k) & links[j].read(k);
			//Check for common pairs one is linked to, the other is related to
			//(c) k linked to j, k related to i but not linked
			for (int k = 0; k < i; k++)
				counter += links[k].read(j) & adj[k].read(i) & !links[k].read(i);
			//(d) i linked to k, j related to k but not linked
			for (int k = j + 1; k < N; k++)
				counter += (links[i].read(k) & adj[j].read(k) & !links[j].read(k));
			//(e) Check for path completion
			for (int m = 0; m < i; m++)
				for (int n = j + 1; n < N; n++)
					counter += adj[m].read(i) & adj[j].read(n) & links[m].read(n);
			if (counter) {
				workspace[i].set(j);
				nrel++;
			}
		}
	}

	/*uint64_t cnt = 0;
	for (int i = 0; i < N; i++)
		cnt += adj[i].count_bits();
	cnt >>= 1;
	printf("nrel in workspace: %" PRIu64 "\n", cnt);*/

	if (DEBUG_EVOL) {
		printf_dbg("Effective Adjacency Matrix:\n");
		for (int j = 0; j < N; j++)
			workspace[j].printBitset();
	}

	uint64_t max_rel = (uint64_t)N * (N - 1) >> 1;
	//printf("nrel: %" PRIu64 "\n", nrel);
	if (max_rel == nrel) {
		//printf_dbg("Returning with no anti-link found.\n");
		return std::make_pair((unsigned)N, (unsigned)N);
	}
	
	//printf("nrel: %u\n", (unsigned)nrel);
	uint64_t rdest = mrng.rng() * (max_rel - nrel);
	unsigned row = 0;
	counter = 0;

	if (DEBUG_EVOL)
		printf("Choosing random anti-link [%" PRIu64 "]\n", rdest);

	while (counter <= rdest) {
		//printf("row: %u\tcounter: %" PRIu64 "\n", row, counter);
		counter += N - row - 1 - workspace[row].partial_count(row + 1, N - row - 1);
		row++;
	}
	//printf("HERE\n"); fflush(stdout);
	row--;
	counter -= N - row - 1 - workspace[row].partial_count(row + 1, N - row - 1);

	uint64_t column_index = rdest - counter;
	unsigned column = 0;
	if (DEBUG_EVOL) {
		printf("column index: %" PRIu64 "\n", column_index);
		fflush(stdout);
	}

	FastBitset &fb = row == 0 ? workspace[row+1] : workspace[0];
	workspace[row].clone(fb);
	fb.flip();
	for (unsigned i = 0; i <= row; i++)
		fb.unset(i);
	for (unsigned i = 0; i <= column_index; i++)
		fb.unset(column = fb.next_bit());

	std::pair<unsigned, unsigned> random_anti_link = std::make_pair(row, column);
	if (DEBUG_EVOL) {
		printf("random anti link: (%u, %u)\n", row, column);
		fflush(stdout);
	}

	return random_anti_link;
}

//Choose a random link
void linkMove1(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B)
{
	std::pair<unsigned, unsigned> entry = randomLink(links, workspace[0], mrng, nlink);
	if (entry.first == N && entry.second == N)
		return;

	//Store old adjacency matrix
	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);

	//Remove link
	links[entry.first].unset(entry.second);
	nlink--;

	//Transitive closure updates the adjacency matrix
	transitiveClosure(adj, links, N);

	//Update the number of relations
	uint64_t nrel_old = nrel;
	nrel = 0;
	for (unsigned i = 0; i < N; i++)
		nrel += adj[i].count_bits();
	nrel >>= 1;

	//Calculate the change in action
	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		links[entry.first].set(entry.second);
		nlink++;
		nrel = nrel_old;
		for (unsigned i = 0; i < N; i++)
			workspace[i].clone(adj[i]);
	}
}

//Choose a random anti-link
void linkMove2(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B)
{
	uint64_t nrel_eff = nrel;
	std::pair<unsigned, unsigned> entry = randomAntiLink(adj, links, workspace, N, mrng, nrel_eff);
	if (entry.first == N && entry.second == N)
		return;

	//Store old adjacency matrix
	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);

	//Add link
	links[entry.first].set(entry.second);
	nlink++;

	//Transitive closure updates the adjacency matrix
	transitiveClosure(adj, links, N);

	//Update the number of relations
	uint64_t nrel_old = nrel;
	nrel = 0;
	for (unsigned i = 0; i < N; i++)
		nrel += adj[i].count_bits();
	nrel >>= 1;

	//Calculate the change in action
	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		links[entry.first].unset(entry.second);
		nlink--;
		nrel = nrel_old;
		for (unsigned i = 0; i < N; i++)
			workspace[i].clone(adj[i]);
	}
}

//This move can modify the number of links and relations; it is the set of those prohibited by linkMove1 and linkMove2
void linkMove3(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, double beta, const double epsilon, float A, float B)
{
	uint64_t nrel_eff = nrel;
	std::pair<unsigned, unsigned> entry = randomAntiRelation(adj, links, workspace, N, mrng, nrel_eff);
	if (entry.first == N && entry.second == N)
		return;

	//Store old adjacency matrix
	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);
	uint64_t nrel_old = nrel, nlink_old = nlink;

	//Add relation
	adj[entry.first].set(entry.second);

	//Transitive reduction updates the link matrix
	transitiveReduction(links, adj, N);
	nlink = 0;
	for (unsigned i = 0; i < N; i++)
		nlink += links[i].count_bits();

	//Transitive closure adds new transitive relations induced by the relation addition
	transitiveClosure(adj, links, N);
	nrel = 0;
	for (unsigned j = 0; j < N; j++)
		nrel += adj[j].count_bits();
	nrel >>= 1;

	//Calculate the change in action
	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		for (unsigned i = 0; i < N; i++)
			workspace[i].clone(adj[i]);
		transitiveReduction(links, adj, N);
	}
}

//This move modifies the adjacency matrix, without removing transitive relations
void matrixMove1(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B)
{
	//We want a matrix with 0's for transitive relations, 1's everywhere else
	//Do this by subtracting the link matrix from the adjacency matrix, then inverting
	uint64_t counter = 0;
	for (unsigned i = 0; i < N; i++) {
		adj[i].clone(workspace[i]);
		workspace[i].setDifference(links[i]);
		workspace[i].flip();
		workspace[i].unset(i);
		counter += workspace[i].count_bits();
	}

	if (counter == 0)
		return;

	uint64_t rlink = counter * mrng.rng();
	counter = 0;
	unsigned row = 0;
	while ((counter += workspace[row++].count_bits()) <= rlink) {}
	counter -= workspace[--row].count_bits();

	uint64_t column_index = rlink - counter;
	unsigned column = 0;
	FastBitset &fb = row == 0 ? workspace[row+1] : workspace[0];
	workspace[row].clone(fb);
	for (unsigned i = 0; i <= column_index; i++)
		fb.unset(column = fb.next_bit());

	//Store the old adjacency matrix
	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);
	uint64_t nrel_old = nrel, nlink_old = nlink;

	//Flip the selected bit
	adj[row].flip(column);

	//Transitive reduction updates the link matrix
	transitiveReduction(links, adj, N);
	nlink = 0;
	for (unsigned i = 0; i < N; i++)
		nlink += links[i].count_bits();

	//Transitive closure adds new transitive relations
	transitiveClosure(adj, links, N);
	nrel = 0;
	for (unsigned j = 0; j < N; j++)
		nrel += adj[j].count_bits();
	nrel >>= 1;

	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);
	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		for (unsigned i = 0; i < N; i++)
			workspace[i].clone(adj[i]);
		transitiveReduction(links, adj, N);
	}
}

//State is encoded in the link matrix
void matrixMove2(Bitvector &adj, Bitvector &links, Bitvector &workspace, Bitvector &awork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B)
{
	//Select an element at random
	uint64_t npairs = ((uint64_t)N * (N - 1)) >> 1;
	uint64_t flip_id = mrng.rng() * npairs;
	unsigned row = 0;
	unsigned cnt = 0;
	while (cnt < flip_id) {
		if (cnt + N - row - 1 < flip_id) {
			cnt += N - row - 1;
			row++;
		} else
			break;
	}	

	unsigned col = flip_id - cnt + row + 1;
	links[row].flip(col);

	uint64_t nrel_old = nrel, nlink_old = nlink;

	transitiveReduction(adj, links, N);
	nlink = 0;
	for (unsigned i = 0; i < N - 1; i++)
		nlink += adj[i].partial_count(i + 1, N - i - 1);

	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);
	transitiveClosure(adj, workspace, N);
	nrel = 0;
	for (unsigned i = 0; i < N - 1; i++)
		nrel += adj[i].partial_count(i + 1, N - i - 1);

	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		links[row].flip(col);
	}
}

void orderMove(Bitvector &adj, Bitvector &links, Bitvector &awork, std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> Uwork, std::vector<unsigned> Vwork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B)
{
	//Select a random total order
	bool flipU = mrng.rng() < 0.5;
	std::vector<unsigned> &W = flipU ? U : V;

	//Select a random pair
	unsigned e0 = mrng.rng() * N;
	unsigned e1 = mrng.rng() * (N - 1);
	e1 += (e0 == e1);

	std::swap(W[e0], W[e1]);

	ordered_labeling(U, V, Uwork, Vwork);

	for (unsigned i = 0; i < N; i++)
		adj[i].reset();

	uint64_t nrel_old = nrel, nlink_old = nlink;

	nrel = 0;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = i + 1; j < N; j++) {
			if ((Uwork[i] < Uwork[j] && Vwork[i] < Vwork[j]) || (Uwork[j] < Uwork[i] && Vwork[j] < Vwork[i])) {
				adj[i].set(j);
				adj[j].set(i);
				nrel++;
			}
		}
	}

	nlink = 0;
	transitiveReduction(links, adj, N);
	for (unsigned i = 0; i < N - 1; i++)
		nlink += links[i].partial_count(i + 1, N - i - 1);
	
	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		std::swap(W[e0], W[e1]);
	}
}

void clusterMove1(Bitvector &adj, Bitvector &links, FastBitset &cluster, Bitvector &workspace, Bitvector &awork, FastBitset &workspace2, int * const &lengths, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B)
{
	//printf("Executing cluster update!\n");
	//Select a seed element at random
	unsigned seed = mrng.rng() * N;
	unsigned chain = (unsigned)(mrng.rng() * 2);
	if (chain) {
		//printf("> Generating a chain.\n");
		//Identify a random chain starting at 'seed' and ending at a random future relation
		unsigned start = seed;
		workspace[0].reset();
		links[start].partial_clone(workspace[0], start + 1, N - start - 1);
		unsigned end_idx = mrng.rng() * workspace[0].count_bits();
		unsigned end_cnt = 0;
		unsigned end = UINT_MAX;
		while (end_cnt++ < end_idx)
			workspace[0].unset(end = workspace[0].next_bit());

		if (end == UINT_MAX) {
			cluster.reset();
			cluster.set(seed);
		} else
			longestChain_v3(links, workspace, cluster, &workspace2, lengths, N, start, end, 0);

		//printf("\tLength: %u\n", (unsigned)cluster.count_bits());
	} else {
		//printf("Generating an antichain.\n");
		//Identify a random antichain starting at 'seed'
		maximalAntichain(cluster, links, N, seed);
		//printf("\tLength: %u\n", (unsigned)cluster.count_bits());
	}

	//Flip all pairs identified in the cluster
	for (unsigned i = 0; i < N - 1; i++)
		for (unsigned j = i + 1; j < N; j++)
			if (cluster.read(i) && cluster.read(j))
				links[i].flip(j);
	
	uint64_t nrel_old = nrel, nlink_old = nlink;

	transitiveReduction(adj, links, N);
	nlink = 0;
	for (unsigned i = 0; i < N - 1; i++)
		nlink += adj[i].partial_count(i + 1, N - i - 1);

	for (unsigned i = 0; i < N; i++)
		adj[i].clone(workspace[i]);
	transitiveClosure(adj, workspace, N);
	nrel = 0;
	for (unsigned i = 0; i < N - 1; i++)
		nrel += adj[i].partial_count(i + 1, N - i - 1);

	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		for (unsigned i = 0; i < N - 1; i++)
			for (unsigned j = i + 1; j < N; j++)
				if (cluster.read(i) && cluster.read(j))
					links[i].flip(j);
	}
}

void clusterMove2(Bitvector &adj, Bitvector &links, Bitvector &awork, std::vector<unsigned> &U, std::vector<unsigned> &V, std::vector<unsigned> &Uwork, std::vector<unsigned> &Vwork, uint64_t * const cardinalities, MersenneRNG &mrng, unsigned N, uint64_t &nlink, uint64_t &nrel, double &action, uint64_t &num_accept, const WeightFunction wf, const int stdim, const double beta, const double epsilon, const double A, const double B)
{
	//Select a random total order
	bool flipU = mrng.rng() < 0.5;
	std::vector<unsigned> &W = flipU ? U : V;

	//Select a random pair
	unsigned e0 = mrng.rng() * N;
	unsigned e1 = mrng.rng() * (N - 1);
	e1 += (e0 == e1);

	//Reverse the range
	unsigned range_start = std::min(e0, e1);
	unsigned range_end = std::max(e0, e1) + 1;
	std::reverse(W.begin() + range_start, W.begin() + range_end);

	ordered_labeling(U, V, Uwork, Vwork);

	for (unsigned i = 0; i < N; i++)
		adj[i].reset();

	uint64_t nrel_old = nrel, nlink_old = nlink;

	nrel = 0;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = i + 1; j < N; j++) {
			if ((Uwork[i] < Uwork[j] && Vwork[i] < Vwork[j]) || (Uwork[j] < Uwork[i] && Vwork[j] < Vwork[i])) {
				adj[i].set(j);
				adj[j].set(i);
				nrel++;
			}
		}
	}

	nlink = 0;
	transitiveReduction(links, adj, N);
	for (unsigned i = 0; i < N - 1; i++)
		nlink += links[i].partial_count(i + 1, N - i - 1);
	
	bool accept = metropolis(adj, links, awork, cardinalities, action, stdim, beta, epsilon, wf, N, A, B, nrel, nlink, num_accept, mrng);

	if (!accept) {
		//Restore old state
		nlink = nlink_old;
		nrel = nrel_old;
		std::reverse(W.begin() + range_start, W.begin() + range_end);
	}
}

