/////////////////////////////
//(C) Will Cunningham 2020 //
//    Perimeter Institute  //
/////////////////////////////

#include "Anneal.h"
#include "Subroutines.h"
#include "Operations.h"

//Population annealing algorithm
//Uses adaptive schedule, multi-histogram reweighting

bool annealNetwork_v2(Network * const network, CuResources * const cu, CaResources * const ca)
{
	const int rank = network->network_properties.cmpi.rank;
	const int nprocs = network->network_properties.cmpi.num_mpi_threads;
	const int N = network->network_properties.N;

	if (!network->network_properties.flags.popanneal)
		return false;

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	printf_mpi(rank, "Executing Population Annealing [v2]...\n");
	if (!rank) fflush(stdout);

	//Population Annealing Parameters
	const unsigned R0 = network->network_properties.R0;
	const unsigned nBmax = 1000;		//Maximum steps in beta
	const double beta_init = 0.0;		//Temperature range
	const double beta_final = network->network_properties.beta;
	const unsigned runs = network->network_properties.runs;
	uint64_t rng_seed = network->network_properties.seed;
	uint64_t tot_pop = 0;			//Sum of population sizes

	double *B = MALLOC<double,true>(nBmax);	//Temperature steps
	B[0] = beta_init;
	unsigned b;
	unsigned Rmax = 2 * R0;
	unsigned *R = MALLOC<unsigned,true>(nBmax);	//Population sizes
	R[0] = R0;
	double *Reff_t = MALLOC<double,true>(nBmax);	//Effective size after resampling
	double *Reff_s = MALLOC<double,true>(nBmax);	//Entropic effective size
	double *Reff_r = MALLOC<double,true>(nBmax);	//Effective size after relaxing
	Reff_t[0] = Reff_s[0] = Reff_r[0] = (double)R0;

	//Schedule Strategy
	bool adaptive_temperature = true;
	bool adaptive_sweep = true;
	//Static Schedule Parameters
	double dB = 0.01;		//Used if adaptive_temperature = false
	unsigned *theta = MALLOC<unsigned,true>(nBmax);
	theta[0] = network->network_properties.sweeps;
	//Dynamic Schedule Parameters
	double min_overlap = 0.85;	//Minimum action overlap
	double max_overlap = 0.87;	//Maximum action overlap

	//Action Parameters
	//double A[2] = { 1.0, 0.0 };
	const double A[2] = { network->network_properties.couplings[0], network->network_properties.couplings[1] };
	uint64_t num_pairs = ((uint64_t)N * (N - 1)) >> 1;
	unsigned steps_per_sweep = num_pairs / 4;
	//unsigned steps_per_sweep = 1;
	double mean_action;

	//MPI Architecture
	//We initially split the replicas among the processes
	//During resampling, replicas do not move between processes, 
	//which may create a minor load imbalance; for large R0
	//this should become negligible

	//Number of replicas on each machine
	unsigned *Rloc = MALLOC<unsigned,true>(nBmax);
	Rloc[0] = ceil((float)R[0] / nprocs);
	if ((rank+1) * Rloc[0] > R[0])
		Rloc[0] -= (rank+1) * Rloc[0] - R[0];

	//CUDA Architecture

	//Threads per block when accessing replicas
	size_t nthreads = N < 128 ? N : N < 256 ? 128 : 256;
	//Number of blocks (replicas) studied (per MPI process)
	size_t nblocks_rep = Rloc[0];
	size_t nblocks_obs = ceil((float)nblocks_rep / 1024);
	dim3 threads_per_block_rep(nthreads, 1, 1);	//Threads per block for
						  	//accessing replicas
	dim3 threads_per_block_obs(1024, 1, 1);		//Threads per block for
							//accessing observables
	dim3 blocks_per_grid_rep(nblocks_rep, 1, 1);
	dim3 blocks_per_grid_obs(nblocks_obs, 1, 1);
	static const size_t cache_size = 4;	//Each row uses X unsigned integers (N allowed up to N=32*X)
	if (N > cache_size << 5) {
		fprintf(stderr, "Please manually increase the cache size in Anneal.cu.\n");
		return false;
	}

	unsigned gpu_id = 0;
	if (cu->dev_count > 1) {
		//We will use only 1 GPU per MPI thread
		//To use multiple GPUs, create one MPI thread for each
		gpu_id = rand() % cu->dev_count;
		cuCtxPushCurrent(cu->cuContext[gpu_id]);
	}

	//CUDA Event Timers
	CUevent start, stop;
	float elapsed_time;
	checkCudaErrors(cuEventCreate(&start, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&stop, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventRecord(start, 0));

	//Replica Variables
	//Note: Each replica uses ceil(N/32) uint's per row and N uints per column
	//These are stored in column-major order to achieve coalescence
	size_t uints_per_replica = ceil((float)N / 32) * N;
	size_t bitblocks = uints_per_replica / N;
	CUdeviceptr d_adj = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
	CUdeviceptr d_link = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
	CUdeviceptr d_U = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
	CUdeviceptr d_V = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
	CUdeviceptr d_action = CUDA_MALLOC<double, 0>(nblocks_rep);
	CUdeviceptr d_nrel = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	CUdeviceptr d_nlink = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	CUdeviceptr d_offspring = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	CUdeviceptr d_parsum = CUDA_MALLOC<unsigned, 0>(2 * nblocks_rep);
	CUdeviceptr d_families = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	CUdeviceptr d_rng = CUDA_MALLOC<RNGState, 0>(Rmax);

	//Observables
	//observables[0] = <E>
	//observables[1] = <E^2>
	//observables[2] = <R>
	//observables[3] = <L>
	CUdeviceptr d_observables = CUDA_MALLOC<double, 0>(4);
	//Mean action of all replicas
	CUdeviceptr d_action_biased = CUDA_MALLOC<double, 0>(1);
	//Buffers used for general data
	void *data = MALLOC<double>(Rmax);
	std::vector<double> v_data(Rmax, 0);
	//double h_obs[4];

	//Counters and Sums
	CUdeviceptr d_Q = CUDA_MALLOC<double, 0>(1);	//Partition function ratio
	CUdeviceptr d_Ri = CUDA_MALLOC<unsigned, 0>(1);	//Population size(s)
	CUdeviceptr d_Reff = CUDA_MALLOC<double, 0>(1);	//Effective population size after resampling
	int *sendcnt = MALLOC<int>(nprocs);
	int *displs = MALLOC<int>(nprocs);
	displs[0] = 0;

	//Family tree
	unsigned *families = MALLOC<unsigned>(Rmax);	//Each entry gives the family ID of a particular replica
	unsigned *famcnt = MALLOC<unsigned>(R0);	//Each entry gives the number of replicas with that family number
	unsigned *nfamilies = MALLOC<unsigned>(nBmax);	//Number of unique families represented by at least one replica
	nfamilies[0] = R0;

	//Histogram Data
	//Used for an adaptive temperature schedule
	CUdeviceptr d_overlap = CUDA_MALLOC<double>(1);		//Energy histogram overlap

	//Thermodynamic Variables / Observables
	//These are the outputs
	//They are also used for multi-histogram reweighting when runs > 1
	//Note: the 'err' indicates the standard deviation of the variable
	double *action = MALLOC<double, true>(nBmax * runs);	//Action (energy)
	double *action_err = MALLOC<double, true>(nBmax * runs);	
	double *ofrac = MALLOC<double, true>(nBmax * runs);	//Ordering fraction
	double *ofrac_err = MALLOC<double, true>(nBmax * runs);	
	double *links = MALLOC<double, true>(nBmax * runs);	//Link fraction
	double *links_err = MALLOC<double, true>(nBmax * runs);	

	double *Cv = MALLOC<double, true>(nBmax * runs);	//Specific heat
	double *Cv_err = MALLOC<double, true>(nBmax * runs);
	double *lnQ = MALLOC<double, true>(nBmax * runs);	//Log of partition function ratio
	double *bF = MALLOC<double, true>(nBmax * runs);	//Dimensionless free energy
	double *bF_err = MALLOC<double, true>(nBmax * runs);	
	double *entr = MALLOC<double, true>(nBmax * runs);	//Statistical entropy
	double *entr_err = MALLOC<double, true>(nBmax * runs);

	//Autocorrelation Data
	double *acorr = MALLOC<double>(Rmax);
	double *lags = MALLOC<double>(Rmax);
	for (unsigned i = 1; i <= Rmax; i++)
		lags[i-1] = log((double)i);
	double *tau_max = MALLOC<double>(nBmax);	//Largest autocorrelation time in the system (used for calculating lag)
	double tau_err_max;	
	double tau_int, tau_exp;	//Integrated and exponential autocorrelation times
	double tau_exp_err, tau_int_err;

	printf_mpi(rank, "\n\tSimulation Parameters:\n");
	printf_mpi(rank, "\t----------------------\n");
	if (!rank) printf_cyan();
	printf_mpi(rank, "\tGraph Type:        [%s]\n", gt_strings[network->network_properties.gt]);
	printf_mpi(rank, "\tWeight Function:   [%s]\n", wf_strings[network->network_properties.wf]);
	printf_mpi(rank, "\tCauset Elements:   [%u]\n", N);
	printf_mpi(rank, "\tTarget Population: [%u]\n", R[0]);
	/*if (mpi_threads > 1) {
		MPI_Barrier(MPI_COMM_WORLD);
		printf("\t > Rank [%u] has [%u]\n", rank, Rloc);
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
	}*/
	if (!adaptive_sweep)
		printf_mpi(rank, "\tRelaxation Sweeps: [%u]\n", theta[0]);
	printf_mpi(rank, "\tAction Coef.:      [%.8f, %.8f]\n", A[0], A[1]);
	printf_mpi(rank, "\tBeta Range:        [%.3f]-[%.3f]\n", beta_init, beta_final);
	printf_mpi(rank, "\tMax. Beta Steps:   [%u]\n", nBmax);
	printf_mpi(rank, "\tIndependent Runs:  [%u]\n", runs);
	printf_mpi(rank, "\tTemp. Sched.:      [%s]\n", adaptive_temperature ? "DYNAMIC" : "STATIC");
	printf_mpi(rank, "\tSweep. Sched.:     [%s]\n", adaptive_sweep ? "DYNAMIC" : "STATIC");
	size_t cr_size = 0;	//Memory used to simulate replicas
	cr_size += 4ULL * uints_per_replica;	//Adjacency matrix
	cr_size += 4ULL * uints_per_replica;	//Link matrix
	cr_size += 8ULL * N;			//U, V
	cr_size += 4ULL * uints_per_replica;	//New adjacency matrix
	cr_size += 4ULL * uints_per_replica;	//New link matrix
	cr_size += 28;	//Action, relations, links, offspring, and partial update (per replica)
	printf_mpi(rank, "\tMemory/Replica:    [%.2f kB]\n", (float)cr_size / 1024.0);
	printf_mpi(rank, "\tPopulation Memory: [%.2f MB]\n", (float)R[0] * cr_size / 1048576.0);
	printf_mpi(rank, "\tRandom Seed:       [%" PRIu64 "]\n", rng_seed);
	if (!rank) {
		printf_std();
		printf("\n");
		fflush(stdout);
	}

	//File I/O
	/*FILE *f_theta, *f_R, *f_Reff, *f_thermo;
	if (!rank) {
		f_theta = fopen("sweep_schedule.txt", "w");
		f_R = fopen("num_replicas.txt", "w");
		f_Reff = fopen("effective_replicas.txt", "w");
		f_thermo = fopen("thermodynamics.txt", "w");
		fprintf(f_thermo, "#run, beta, action, specific_heat, free energy (beta*F), entropy, ordering fraction, links\n");
	}*/

	//HDF5
	//Note: only the master process (rank 0) will
	//handle data I/O
	//This is because attributes cannot be handled in parallel
	hid_t file, group, dspace;
	hsize_t rowdim[2] = { Rmax, nBmax };
	hsize_t rowoffset[2] = { 0, 0 };
	std::string filename = network->network_properties.datdir;
	std::string groupname = "markov";
	filename.append("observables.h5");

	hid_t dset_action, dset_nrel, dset_nlink, dset_family, dset_famcnt;
	if (!rank) {
		file = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		group = H5Gcreate(file, groupname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		dspace = H5Screate_simple(2, rowdim, NULL);

		dset_action = H5Dcreate(group, "action", getH5DataType<double>(), dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		dset_nrel = H5Dcreate(group, "ordering_fraction", getH5DataType<double>(), dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		dset_nlink = H5Dcreate(group, "link_fraction", getH5DataType<double>(), dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		dset_family = H5Dcreate(group, "family", getH5DataType<unsigned>(), dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		dset_famcnt = H5Dcreate(group, "family_count", getH5DataType<unsigned>(), dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	}

	//Necessary since the CUDA_MALLOC calls are asynchronous
	checkCudaErrors(cuCtxSynchronize());

	//Initialize RNG
	RNGInit<<<(unsigned)ceil((float)Rmax/1024), threads_per_block_obs>>>((RNGState*)d_rng, rng_seed, Rmax);

	//Independent Runs (Sequential)
	for (unsigned run = 0; run < runs; run++) {
		printf_mpi(rank, "Starting Run [%u/%u].\n", run+1, runs);
ceil((float)R[0] / nprocs);
		//Initialize the system
		//unsigned *h_adj;
		checkCudaErrors(cuMemsetD32(d_nrel, 0, nblocks_rep));
		checkCudaErrors(cuMemsetD32(d_nlink, 0, nblocks_rep));
		if (network->network_properties.gt == _2D_ORDER) {
			ReplicaInit2D<cache_size<<5><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_U, (unsigned*)d_V, (unsigned*)d_families, (RNGState*)d_rng, N, ceil((float)R[0] / nprocs) * rank);
			/*unsigned *U = MALLOC<unsigned>(N * nblocks_rep);
			unsigned *V = MALLOC<unsigned>(N * nblocks_rep);
			checkCudaErrors(cuMemcpyDtoH(U, d_U, sizeof(unsigned) * N * nblocks_rep));
			checkCudaErrors(cuMemcpyDtoH(V, d_V, sizeof(unsigned) * N * nblocks_rep));
			checkCudaErrors(cuCtxSynchronize());*/
			/*printf("Initial coordinates:\n");
			for (unsigned r = 0; r < 32; r++) {
				printf_cyan();
				printf("Replica [%u]\n", r);
				printf_std();
				for (int i = 0; i < N; i++)
					printf("%u %u\n", U[r*N+i], V[r*N+i]);
				printf("\n");
			}*/
			/*std::vector<unsigned> Uvec, Vvec;
			Uvec.resize(N);	Vvec.resize(N);
			memcpy(Uvec.data(), U, sizeof(unsigned) * N);
			memcpy(Vvec.data(), V, sizeof(unsigned) * N);
			ordered_labeling(Uvec, Vvec, Uvec, Vvec);*/
			/*printf("\nSorted coordinates (CPU):\n");
			for (int i = 0; i < N; i++)
				printf("%u %u\n", Uvec[i], Vvec[i]);*/
			OrderToMatrix<cache_size<<5><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_U, (unsigned*)d_V, (unsigned*)d_nrel, N, bitblocks);
			/*h_adj = MALLOC<unsigned>(nblocks_rep * uints_per_replica);
			checkCudaErrors(cuMemcpyDtoH(h_adj, d_adj, sizeof(unsigned) * nblocks_rep * uints_per_replica));
			checkCudaErrors(cuMemcpyDtoH(U, d_U, sizeof(unsigned) * N * nblocks_rep));
			checkCudaErrors(cuMemcpyDtoH(V, d_V, sizeof(unsigned) * N * nblocks_rep));
			checkCudaErrors(cuCtxSynchronize());*/
			/*printf("\nSorted coordinates:\n");
			for (int i = 0; i < N; i++)
				printf("%u %u\n", U[i], V[i]);*/
			/*for (unsigned r = 0; r < nblocks_rep; r++) {
				for (unsigned i = 0; i < N - 1; i++) {	//Row
					for (unsigned j = i + 1; j < N; j++) {	//Column
						unsigned block = (j >> 5) * N + i;	//Entry of adj that (i,j) belongs to
						unsigned idx = j & 31;
						unsigned bit = (h_adj[r*uints_per_replica+block] >> idx) & 1;
						assert (bit == (U[r*N+i] < U[r*N+j] && V[r*N+i] < V[r*N+j]) || (U[r*N+j] < U[r*N+i] && V[r*N+j] < V[r*N+i]));
					}
				}
			}*/
			//printf_dbg("No errors in OrderToMatrix\n");
		} else {
			ReplicaInitPacked<<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_families, (RNGState*)d_rng, N, bitblocks);
			ReplicaClosurePacked<cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, N, bitblocks);
			ReplicaCountPacked<cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_nrel, N, bitblocks);
		}

		/*unsigned *nrel = MALLOC<unsigned>(nblocks_rep);
		unsigned *nlink = MALLOC<unsigned>(nblocks_rep);
		checkCudaErrors(cuMemcpyDtoH(nrel, d_nrel, sizeof(unsigned) * nblocks_rep));
		checkCudaErrors(cuMemcpyDtoH(nlink, d_nlink, sizeof(unsigned)));*/
		//printf("Initial relations, links: %f\t%f\n", (double)nrel[0] / num_pairs, (double)nlink[0] / num_pairs);
		//checkCudaErrors(cuMemsetD32(d_nrel, 0, nblocks_rep));

		checkCudaErrors(cuMemcpyDtoD(d_link, d_adj, sizeof(unsigned) * uints_per_replica * nblocks_rep));
		ReplicaReductionPacked<cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_link, N, bitblocks);
		ReplicaCountPacked<cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_link, (unsigned*)d_nlink, N, bitblocks);

		/*unsigned *h_link = MALLOC<unsigned>(nblocks_rep * uints_per_replica);
		checkCudaErrors(cuMemcpyDtoH(h_link, d_link, sizeof(unsigned) * uints_per_replica * nblocks_rep));
		checkCudaErrors(cuCtxSynchronize());
		for (unsigned r = 0; r < nblocks_rep; r++) {
			unsigned offset = uints_per_replica * r;
			for (int i = 0; i < N - 1; i++) {
				for (int j = i + 1; j < N; j++) {
					unsigned block = (j >> 5) * N + i;	//Entry of adj that (i,j) belongs to
					unsigned idx = j & 31;
					unsigned bit = (h_adj[offset+block] >> idx) & 1;
					if (!bit) {
						assert (((h_link[offset+block]>>idx) & 1) == 0);
						continue;
					}
					bool any = false;
					for (int k = i + 1; k < j; k++)
						if ((h_adj[offset+(k>>5)*N+i]>>(k&31)) & (h_adj[offset+(j>>5)*N+k]>>(j&31)) & 1)
							any = true;
					if (any)
						assert (((h_link[offset+(j>>5)*N+i]>>(j&31)) & 1) == 0);
					else
						assert (((h_link[offset+(j>>5)*N+i]>>(j&31)) & 1) == 1);
				}
			}
		}*/
		//printf_dbg("No errors in ReplicaReductionPacked\n");
		/*checkCudaErrors(cuMemcpyDtoH(nrel, d_nrel, sizeof(unsigned) * nblocks_rep));
		checkCudaErrors(cuMemcpyDtoH(nlink, d_nlink, sizeof(unsigned) * nblocks_rep));
		checkCudaErrors(cuCtxSynchronize());
		printf("Initial relations, links: %f\t%u\n", (double)nrel[0] / num_pairs, nlink[0]);*/
		/*printf("\nInitial relations, links:\n");
		for (unsigned i = 0; i < nblocks_rep; i++)
			printf("%u %u\n", nrel[i], nlink[i]);*/

		switch (network->network_properties.wf) {
		case RELATION:
			//Relation action
			RelationAction<<<blocks_per_grid_obs, threads_per_block_obs>>>((double*)d_action, (double*)d_observables, (unsigned*)d_nrel, (unsigned*)d_nlink, Rloc[0], A[0], A[1]);
			/*if (run == 3) {
			printf("\nInitial action:\n");
			checkCudaErrors(cuMemcpyDtoH(h_action, d_action, sizeof(double) * nblocks_rep));
			for (unsigned i = 0; i < nblocks_rep; i++)
				printf("%f\n", h_action[i]);
			}*/
			break;
		case RELATION_PAIR:
			//Relation pair action
			//A[0] /= num_pairs;
			ReplicaActionPairPacked<cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((double*)d_action, (double*)d_observables, (unsigned*)d_adj, (unsigned*)d_nrel, N, bitblocks, A[0], A[1]);
			break;
		default:
			assert (false);
		}
		checkCudaErrors(cuCtxSynchronize());

		//Index for temperature steps
		b = 0;

		//Calculate initial observables
		gather_data((double*)data, d_action, sendcnt, displs, Rloc[b], nprocs, rank);
		if (!run && !rank) annealH5Write((double*)data, file, dset_action, dspace, rowdim, rowoffset, R[b]);
		action[nBmax*run] = mean_action = gsl_stats_mean((double*)data, 1, R[b]);
		action_err[nBmax*run] = gsl_stats_sd_m((double*)data, 1, R[b], mean_action);

		gather_data((unsigned*)data, d_nrel, sendcnt, displs, Rloc[b], nprocs, rank);
		for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
		if (!run && !rank) annealH5Write(v_data.data(), file, dset_nrel, dspace, rowdim, rowoffset, R[b]);
		ofrac[nBmax*run] = gsl_stats_mean(v_data.data(), 1, R[b]);
		ofrac_err[nBmax*run] = gsl_stats_sd_m(v_data.data(), 1, R[b], ofrac[nBmax*run]);

		gather_data((unsigned*)data, d_nlink, sendcnt, displs, Rloc[b], nprocs, rank);
		for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
		if (!run && !rank) annealH5Write(v_data.data(), file, dset_nlink, dspace, rowdim, rowoffset, R[b]);
		links[nBmax*run] = gsl_stats_mean(v_data.data(), 1, Rloc[b]);
		links_err[nBmax*run] = gsl_stats_sd_m(v_data.data(), 1, Rloc[b], links[nBmax*run]);
	
		gather_data((unsigned*)data, d_families, sendcnt, displs, Rloc[b], nprocs, rank);
		if (!run && !rank) annealH5Write((unsigned*)data, file, dset_family, dspace, rowdim, rowoffset, R[b]);

		std::fill(famcnt, famcnt + R0, 1);
		if (!run && !rank) annealH5Write(famcnt, file, dset_famcnt, dspace, rowdim, rowoffset, R0);
		if (!run && !rank) H5Fflush(file, H5F_SCOPE_LOCAL);
		rowoffset[1]++;
		
		//Specific heat is initially zero since B[0] = 0
		//lnQ is initially zero since no resampling has occurred yet
		bF[nBmax*run] = -log(2.0); //This is the reduced free energy, divided by num_pairs
		entr[nBmax*run] = -bF[nBmax*run];	//This is the reduced entropy
		tau_max[0] = 0.5;

		//Update temperature
		if (!run) {
			if (adaptive_temperature)
				adaptive_beta_stepsize(dB, d_action, d_Q, d_overlap, R0, R[b], Rloc[b], B[b], beta_final, min_overlap, max_overlap, mean_action, blocks_per_grid_obs, threads_per_block_obs, rank);
			else
				dB = min(dB, beta_final);
			B[b+1] = B[b] + dB;
		}

		/*if (!rank) {
			fprintf(f_R, "%u %f %u\n", run, B[b], R[b]);
			fprintf(f_Reff, "%u %f %u\n", run, B[b], R[b]);
			fprintf(f_thermo, "%u %.16f %.16f %.16f %.16f %.16f\n", run, B[b], action[nBmax*run], Cv[nBmax*run], bF[nBmax*run], entr[nBmax*run], ofrac[nBmax*run], links[nBmax*run]);
		}*/

		//Annealing
		double q;
		while (B[b+1] <= beta_final) {
			printf_mpi(rank, " > %f\t%u\t%f\n", B[b], R[b], mean_action); fflush(stdout);
			b++;

			checkCudaErrors(cuMemsetD32(d_Q, 0, 2));
			QKernel<<<blocks_per_grid_obs, threads_per_block_obs>>>((double*)d_action, Rloc[b-1], B[b] - B[b-1], mean_action, (double*)d_Q);
			checkCudaErrors(cuMemcpyDtoH(&q, d_Q, sizeof(double)));
			checkCudaErrors(cuCtxSynchronize());
			#ifdef MPI_ENABLED
			MPI_Allreduce(MPI_IN_PLACE, &q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			#endif
			lnQ[nBmax*run+b] = -(B[b] - B[b-1]) * mean_action + log(q) - log((double)R[b-1]);
			
			checkCudaErrors(cuMemsetD32(d_Ri, 0, 1));
			//checkCudaErrors(cuMemsetD32(d_Reff, 0, 2));
			TauKernel_v2<<<blocks_per_grid_obs, threads_per_block_obs>>>((double*)d_action, (unsigned*)d_offspring, (unsigned*)d_parsum, (RNGState*)d_rng, R[0], R[b-1], Rloc[b-1], lnQ[nBmax*run+b], B[b] - B[b-1], (unsigned*)d_Ri, (double*)d_Reff);
			checkCudaErrors(cuMemcpyDtoH(&R[b], d_Ri, sizeof(unsigned)));
			//checkCudaErrors(cuMemcpyDtoH(&Reff, d_Reff, sizeof(double)));

			Rloc[b] = R[b];
			//printf("Rank [%u] has Rloc[%u] = %u\n", rank, b, Rloc[b]); fflush(stdout);
			if (Rloc[b] >= Rmax) {
				printf("Exceeded local limit on rank [%u]\n", rank);
				printf("Local Limit: %u\n", Rmax);
				printf("Value: %u\n", Rloc[b]);
				exit(1);
			}
			#ifdef MPI_ENABLED
			MPI_Allreduce(MPI_IN_PLACE, &R[b], 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
			//MPI_Allreduce(MPI_IN_PLACE, &Reff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			#endif
			//Reff = 1.0 / Reff;
			if (R[b] > Rmax) {
				printf_mpi(rank, "Exceeded global limit!\n");
				printf_mpi(rank, "Global Limit: %u\n", Rmax);
				printf_mpi(rank, "Value: %u\n", R[b]);
				exit(1);
			}

			/*if (!rank) {
				//printf("Number of offspring: %u\n", R[b]);
				fprintf(f_R, "%u %f %u\n", run, B[b], R[b]);
				//printf("Effective population size: %f\n", Reff);
			}*/

			PartialSum<<<blocks_per_grid_obs, threads_per_block_obs>>>((unsigned*)d_offspring, (unsigned*)d_parsum, Rloc[b-1]);
			checkCudaErrors(cuCtxSynchronize());

			nblocks_rep = Rloc[b];
			CUdeviceptr d_adj_new = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
			CUdeviceptr d_link_new = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
			CUdeviceptr d_U_new = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
			CUdeviceptr d_V_new = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
			CUdeviceptr d_action_new = CUDA_MALLOC<double, 0>(nblocks_rep);
			CUdeviceptr d_nrel_new = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			CUdeviceptr d_nlink_new = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			CUdeviceptr d_offspring_new = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			CUdeviceptr d_parsum_new = CUDA_MALLOC<unsigned, 0>(2 * nblocks_rep);
			CUdeviceptr d_families_new = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			checkCudaErrors(cuCtxSynchronize());
			
			if (network->network_properties.gt == _2D_ORDER) {
				ResampleReplicas2D<<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_U_new, (unsigned*)d_U, (unsigned*)d_V_new, (unsigned*)d_V, (double*)d_action_new, (double*)d_action, (unsigned*)d_offspring, (unsigned*)d_parsum, (unsigned*)d_families_new, (unsigned*)d_families, Rloc[b-1], N);
			} else {
				ResampleReplicasPacked<<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj_new, (unsigned*)d_adj, (unsigned*)d_link_new, (unsigned*)d_link, (double*)d_action_new, (double*)d_action, (unsigned*)d_offspring, (unsigned*)d_parsum, (unsigned*)d_families_new, (unsigned*)d_families, Rloc[b-1], N, bitblocks);
			}
			checkCudaErrors(cuCtxSynchronize());

			CUDA_FREE(d_adj);		d_adj = d_adj_new;
			CUDA_FREE(d_link);		d_link = d_link_new;
			CUDA_FREE(d_U);			d_U = d_U_new;
			CUDA_FREE(d_V);			d_V = d_V_new;
			CUDA_FREE(d_action);		d_action = d_action_new;
			CUDA_FREE(d_nrel);		d_nrel = d_nrel_new;
			CUDA_FREE(d_nlink);		d_nlink = d_nlink_new;
			CUDA_FREE(d_offspring);		d_offspring = d_offspring_new;
			CUDA_FREE(d_parsum);		d_parsum = d_parsum_new;
			CUDA_FREE(d_families);		d_families = d_families_new;
			checkCudaErrors(cuCtxSynchronize());

			/*checkCudaErrors(cuMemcpyDtoH(h_action, d_action, sizeof(double) * Rloc));
			checkCudaErrors(cuCtxSynchronize());
			double acc = 0;
			for (unsigned i = 0; i < Rloc; i++)
				acc += h_action[i];
			acc /= Rloc;
			printf_dbg("Mean action (directly after resampling): %f\n", acc);
			for (unsigned i = 0; i < 10; i++)
				printf("%f\n", h_action[i]);*/
			
			nblocks_obs = ceil((float)nblocks_rep / 1024);
			blocks_per_grid_rep.x = nblocks_rep;
			blocks_per_grid_obs.x = nblocks_obs;

			//DEBUG
			//checkCudaErrors(cuCtxSynchronize());
			//checkCudaErrors(cuEventRecord(start, 0));
			//END DEBUG

			//Effective Replicas
			checkCudaErrors(cuMemcpyDtoH(families, d_families, sizeof(unsigned) * nblocks_rep));
			checkCudaErrors(cuCtxSynchronize());
			memset(famcnt, 0, sizeof(unsigned) * R0);
			for (unsigned i = 0; i < nblocks_rep; i++) {
				famcnt[families[i]]++;
				//printf("Replica [%u] belongs to family [%u]\n", i, families[i]);
			}
			#ifdef MPI_ENABLED
			MPI_Allreduce(MPI_IN_PLACE, famcnt, R0, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
			#endif
			Reff_t[b] = Reff_s[b] = 0.0;
			for (unsigned i = 0; i < R0; i++) {
				Reff_t[b] += POW2((double)famcnt[i] / R[b]);
				if (famcnt[i]) {
					Reff_s[b] -= famcnt[i] * log((double)famcnt[i] / R[b]) / R[b];
					nfamilies[b]++;
				}
			}
			Reff_t[b] = 1.0 / Reff_t[b];
			Reff_s[b] = exp(Reff_s[b]);
			if (!rank) {
				printf_cyan();
				printf_mpi(rank, "   Effective population size (resampling): R/rho_t = %f\n", Reff_t[b]);
				printf_mpi(rank, "   Effective population size (entropic):   R/rho_s = %f\n", Reff_s[b]);
				printf_mpi(rank, "   Number of unique families: %u\n", nfamilies[b]);
				printf_std();
				//if (!calc_autocorr)
				//	fprintf(f_Reff, "%u %f %u\n", run, B[b], (unsigned)ceil(Reff_t));
			}

			//The variable '_theta' will be different from 'theta'
			//when we are using a dynamic sweep schedule
			theta[b] = theta[0];
			if (adaptive_sweep && b > 1)
				theta[b] *= ceil((float)R[b] / Reff_t[b]);
			printf_mpi(rank, "   > Relaxing with [%u] sweeps at b=[%f].\n", theta[b], B[b]);
			//if (!run && !rank)
			//	fprintf(f_theta, "%f %u\n", B[b], theta[b]);

			if (network->network_properties.gt == RANDOM && network->network_properties.wf == RELATION_PAIR)
				RelaxReplicasPacked<RANDOM, RELATION_PAIR, cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_link, NULL, NULL, (unsigned*)d_nrel, (unsigned*)d_nlink, (double*)d_action, (RNGState*)d_rng, theta[b], steps_per_sweep, N, B[b], A[0], A[1], bitblocks);
			else if (network->network_properties.gt == RANDOM && network->network_properties.wf == RELATION)
				RelaxReplicasPacked<RANDOM, RELATION, cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_link, NULL, NULL, (unsigned*)d_nrel, (unsigned*)d_nlink, (double*)d_action, (RNGState*)d_rng, theta[b], steps_per_sweep, N, B[b], A[0], A[1], bitblocks);
			else if (network->network_properties.gt == _2D_ORDER && network->network_properties.wf == RELATION)
				RelaxReplicasPacked<_2D_ORDER, RELATION, cache_size><<<blocks_per_grid_rep, threads_per_block_rep>>>((unsigned*)d_adj, (unsigned*)d_link, (unsigned*)d_U, (unsigned*)d_V, (unsigned*)d_nrel, (unsigned*)d_nlink, (double*)d_action, (RNGState*)d_rng, theta[b], steps_per_sweep, N, B[b], A[0], A[1], bitblocks);
			else
				assert (false);

			/*checkCudaErrors(cuMemcpyDtoH(h_action, d_action, sizeof(double) * Rloc));
			checkCudaErrors(cuCtxSynchronize());
			double acc = 0;
			for (unsigned i = 0; i < Rloc; i++)
				acc += h_action[i];
			acc /= Rloc;
			printf_dbg("Mean action (directly after relaxing): %f\n", acc);
			for (unsigned i = 0; i < 10; i++)
				printf("%f\n", h_action[i]);
			printChk();*/

			//DEBUG
			//checkCudaErrors(cuCtxSynchronize());
			//checkCudaErrors(cuEventRecord(stop, 0));
			//checkCudaErrors(cuEventSynchronize(stop));
			//checkCudaErrors(cuEventElapsedTime(&elapsed_time, start, stop));
			//END DEBUG

			//First calculate the longest autocorrelation time in the system
			//We also write the data to file
			tau_int = tau_int_err = tau_max[b] = tau_err_max = 0.0;
			gather_data((double*)data, d_action, sendcnt, displs, Rloc[b], nprocs, rank);
			autocorrelation((double*)data, acorr, lags, R[b], tau_exp, tau_exp_err, tau_int, tau_int_err);
			if (!rank && !run) annealH5Write((double*)data, file, dset_action, dspace, rowdim, rowoffset, R[b]);
			tau_max[b] = std::max(tau_int, tau_max[b]);
			tau_err_max = std::max(tau_int_err, tau_err_max);

			gather_data((unsigned*)data, d_nrel, sendcnt, displs, Rloc[b], nprocs, rank);
			for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
			autocorrelation(v_data.data(), acorr, lags, R[b], tau_exp, tau_exp_err, tau_int, tau_int_err);
			if (!rank && !run) annealH5Write((unsigned*)data, file, dset_nrel, dspace, rowdim, rowoffset, R[b]);
			tau_max[b] = std::max(tau_int, tau_max[b]);
			tau_err_max = std::max(tau_int_err, tau_err_max);

			gather_data((unsigned*)data, d_nlink, sendcnt, displs, Rloc[b], nprocs, rank);
			for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
			autocorrelation(v_data.data(), acorr, lags, R[b], tau_exp, tau_exp_err, tau_int, tau_int_err);
			if (!rank && !run) annealH5Write((unsigned*)data, file, dset_nlink, dspace, rowdim, rowoffset, R[b]);
			tau_max[b] = std::max(tau_int, tau_max[b]);
			tau_err_max = std::max(tau_int_err, tau_err_max);

			tau_int = tau_max[b];
			tau_int_err = tau_err_max;
			Reff_r[b] = (double)R[b] / (2.0 * tau_int);	//Number of independent samples after relaxing
			if (tau_int_err > 0.1) {
				printf_red();
				printf("   > Value not converged. Try increasing the population size.\n");
				printf_std();
			}

			gather_data((unsigned*)data, d_families, sendcnt, displs, Rloc[b], nprocs, rank);
			if (!run && !rank) annealH5Write((unsigned*)data, file, dset_family, dspace, rowdim, rowoffset, R[b]);
			if (!run && !rank) annealH5Write(famcnt, file, dset_famcnt, dspace, rowdim, rowoffset, R0);
			rowoffset[1]++;	//Increment the column for the next beta

			//Calculate observables (after autocorrelation filtering)
			unsigned stride = (unsigned)ceil(R[b] / Reff_r[b]);
			unsigned Reff_o = ceil((double)R[b] / stride);
			if (!rank) printf_cyan();
			printf_mpi(rank, "   Effective population size (strided):    R/str   = %u\n", Reff_o);
			if (!rank) printf_std();
			gather_data((double*)data, d_action, sendcnt, displs, Rloc[b], nprocs, rank);
			action[nBmax*run+b] = mean_action = gsl_stats_mean((double*)data, stride, Reff_o);
			action_err[nBmax*run+b] = gsl_stats_sd_m((double*)data, stride, Reff_o, mean_action);

			gather_data((unsigned*)data, d_nrel, sendcnt, displs, Rloc[b], nprocs, rank);
			for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
			ofrac[nBmax*run+b] = gsl_stats_mean(v_data.data(), stride, Reff_o);
			ofrac_err[nBmax*run+b] = gsl_stats_sd_m(v_data.data(), stride, Reff_o, ofrac[nBmax*run+b]);

			gather_data((unsigned*)data, d_nlink, sendcnt, displs, Rloc[b], nprocs, rank);
			for (unsigned i = 0; i < R[b]; i++) v_data[i] = (double)(((unsigned*)data)[i]) / num_pairs;
			links[nBmax*run+b] = gsl_stats_mean(v_data.data(), stride, Reff_o);
			links_err[nBmax*run+b] = gsl_stats_sd_m(v_data.data(), stride, Reff_o, links[nBmax*run+b]);

			//Save thermodynamic variables
			//Specific heat
			gather_data((double*)data, d_action, sendcnt, displs, Rloc[b], nprocs, rank);
			specific_heat(Cv[nBmax*run+b], Cv_err[nBmax*run+b], (double*)data, action[nBmax*run+b], action_err[nBmax*run+b], B[b], Reff_o, stride);
			//Dimensionless free energy (reduced)
			free_energy(bF[nBmax*run+b], bF_err[nBmax*run+b], (double*)data, action[nBmax*run+b], B[b], Reff_o, stride);
			bF[nBmax*run+b] = bF[nBmax*run+b-1] - lnQ[nBmax*run+b];
			bF[nBmax*run+b] /= num_pairs;
			bF_err[nBmax*run+b] /= num_pairs;
			//Statistical entropy (reduced)
			entropy(entr[nBmax*run+b], entr_err[nBmax*run+b], action[nBmax*run+b], action_err[nBmax*run+b], -bF[nBmax*run+b] * num_pairs / B[b], -bF_err[nBmax*run+b] * num_pairs / B[b], B[b], num_pairs, Reff_o);
			entr[nBmax*run+b] = B[b] * mean_action -bF[nBmax*run+b];
			entr[nBmax*run+b] /= num_pairs;
			entr_err[nBmax*run+b] /= num_pairs;
			//if (!rank)
			//	fprintf(f_thermo, "%u %.16f %.16f %.16f %.16f %.16f\n", run, B[b], action[nBmax*run+b], Cv[nBmax*run+b], bF[nBmax*run+b], entr[nBmax*run+b], ofrac[nBmax*run+b], links[nBmax*run+b]);

			//Load Balancing
			#ifdef MPI_ENABLED
			float balance = (float)Rloc[b] / R[b];
			MPI_Allreduce(MPI_IN_PLACE, &balance, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
			if (balance < 0.2 / nprocs) {
				printf_mpi(rank, "Load balancing after [%u] steps.\n", b);
				//printf("Rank [%u] has [%u] replicas\n", rank, Rloc);
				balance_replicas(d_adj, d_link, d_U, d_V, d_nrel, d_nlink, d_action, d_offspring, d_parsum, d_families, network->network_properties.gt, N, R[b], Rloc[b], uints_per_replica, nblocks_rep, nblocks_obs, blocks_per_grid_rep, blocks_per_grid_obs, rank, nprocs);
				//printf("Rank [%u] now has [%u] replicas.\n", rank, Rloc);
			}
			#endif

			//Update temperature
			if (!run) {
				if (adaptive_temperature)
					adaptive_beta_stepsize(dB, d_action, d_Q, d_overlap, R0, R[b], Rloc[b], B[b], beta_final, min_overlap, max_overlap, mean_action, blocks_per_grid_obs, threads_per_block_obs, rank);
				B[b+1] = B[b] + dB;
			}

			tot_pop += R[b];
		}

		//Reset variables for the next run
		if (run < runs - 1) {
			nblocks_rep = Rloc[0];
			nblocks_obs = ceil((float)nblocks_rep / 1024);
			blocks_per_grid_rep.x = nblocks_rep;
			blocks_per_grid_obs.x = nblocks_obs;
			b = 0;

			CUDA_FREE(d_adj);
			CUDA_FREE(d_link);
			CUDA_FREE(d_U);
			CUDA_FREE(d_V);
			CUDA_FREE(d_action);
			CUDA_FREE(d_nrel);
			CUDA_FREE(d_nlink);
			CUDA_FREE(d_offspring);
			CUDA_FREE(d_parsum);
			CUDA_FREE(d_families);

			checkCudaErrors(cuCtxSynchronize());

			d_adj = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
			d_link = CUDA_MALLOC<unsigned, 0>(uints_per_replica * nblocks_rep);
			d_U = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
			d_V = CUDA_MALLOC<unsigned, 0>(N * nblocks_rep);
			d_action = CUDA_MALLOC<double, 0>(nblocks_rep);
			d_nrel = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			d_nlink = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			d_offspring = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			d_parsum = CUDA_MALLOC<unsigned, 0>(2 * nblocks_rep);
			d_families = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
			memset(nfamilies, 0, sizeof(unsigned) * nBmax);

			checkCudaErrors(cuCtxSynchronize());
		}

		//Synchronization
		#if MPI_ENABLED
		MPI_Barrier(MPI_COMM_WORLD);
		#endif
	}

	/*if (!rank) {
		fflush(f_theta);	fclose(f_theta);
		fflush(f_R);		fclose(f_R);
		fflush(f_thermo);	fclose(f_thermo);
		fflush(f_Reff);		fclose(f_Reff);
	}*/

	//Calculate weighted averages
	if (runs > 1 && !rank) {
		printf("\nCalculating weighted averages.\n");

		double *action_weighted = MALLOC<double, true>(b + 1);
		double *Cv_weighted = MALLOC<double, true>(b + 1);
		double *entropy_weighted = MALLOC<double, true>(b + 1);
		double *ofrac_weighted = MALLOC<double, true>(b + 1);
		double *links_weighted = MALLOC<double, true>(b + 1);

		FILE *f_action = fopen("action_weighted.txt", "w");
		FILE *f_Cv = fopen("specific_heat_weighted.txt", "w");
		FILE *f_entropy = fopen("entropy_weighted.txt", "w");
		FILE *f_Req = fopen("equilibrium_size.txt", "w");
		FILE *f_ofrac = fopen("ordering_fraction_weighted.txt", "w");
		FILE *f_links = fopen("links_weighted.txt", "w");

		//for (unsigned i = 0; i < runs; i++)
		//	for (unsigned j = 0; j <= b; j++)
		//		printf("bF[RUN %u / b=%.3f] = %.6f\n", i, B[j], bF[i*nBmax+j]);

		//Calculate the variance in the free energy
		for (unsigned i = 0; i <= b; i++) {
			if (i) {
				double bF_var = gsl_stats_variance(bF + i, nBmax, runs);
				double rhoF = R0 * bF_var * POW2(num_pairs);
				fprintf(f_Req, "%f %u\n", B[i], (unsigned)ceil(rhoF));
				printf_cyan();
				printf("Equilibration population size at b=[%f]: rho_f = %f\n", B[i], rhoF);
				printf_std();
				if (R0 < 100 * rhoF) {
					printf_red();
					printf("Increase R0 to at least [%u].\n", (unsigned)ceil(100 * rhoF));
					printf_std();
				}
			}

			double max_bF = 0.0, norm = 0.0;
			for (unsigned r = 0; r < runs; r++)
				max_bF = max(max_bF, -bF[r*nBmax+i]);
			for (unsigned r = 0; r < runs; r++)
				norm += exp(-bF[r*nBmax+i] - max_bF);
			for (unsigned r = 0; r < runs; r++) {
				double omega = exp(-bF[r*nBmax+i] - max_bF) / norm;
				action_weighted[i] += action[r*nBmax+i] * omega;
				Cv_weighted[i] += Cv[r*nBmax+i] * omega;
				entropy_weighted[i] += entr[r*nBmax+i] * omega;
				ofrac_weighted[i] += ofrac[r*nBmax+i] * omega;
				links_weighted[i] += links[r*nBmax+i] * omega;
			}

			fprintf(f_action, "%f %f\n", B[i], action_weighted[i]);
			fprintf(f_Cv, "%f %f\n", B[i], Cv_weighted[i]);
			fprintf(f_entropy, "%f %f\n", B[i], entropy_weighted[i]);
			fprintf(f_ofrac, "%f %f\n", B[i], ofrac_weighted[i]);
			fprintf(f_links, "%f %f\n", B[i], links_weighted[i]);
		}

		fflush(f_action);	fclose(f_action);
		fflush(f_Cv);		fclose(f_Cv);
		fflush(f_entropy);	fclose(f_entropy);
		fflush(f_Req);		fclose(f_Req);
		fflush(f_ofrac);	fclose(f_ofrac);
		fflush(f_links);	fclose(f_links);

		FREE(action_weighted);
		FREE(Cv_weighted);
		FREE(entropy_weighted);
		FREE(ofrac_weighted);
		FREE(links_weighted);
	}

	if (!rank) {
		FILE *f_beta = fopen("temperature_schedule.txt", "w");
		for (unsigned i = 0; i <= b; i++)
			fprintf(f_beta, "%f\n", B[i]);
		fflush(f_beta);	fclose(f_beta);
	}

	//Save attribute variables to HDF5
	hid_t attr, aspace;
	if (!rank) {
		rowdim[0] = b + 1;
		rowdim[1] = 1;
		rowoffset[0] = rowoffset[1] = 0;
		aspace = H5Screate_simple(2, rowdim, NULL);

		//Temperature
		attr = H5Acreate(group, "beta", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, B);
		H5Aclose(attr);

		//Sweeps
		attr = H5Acreate(group, "sweeps", H5T_STD_U32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_STD_U32LE, theta);
		H5Aclose(attr);

		//Population sizes
		attr = H5Acreate(group, "population_size", H5T_STD_U32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_STD_U32LE, R);
		H5Aclose(attr);

		attr = H5Acreate(group, "resampling_size", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, Reff_t);
		H5Aclose(attr);

		attr = H5Acreate(group, "entropic_size", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, Reff_s);
		H5Aclose(attr);

		attr = H5Acreate(group, "relaxation_size", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, Reff_r);
		H5Aclose(attr);

		attr = H5Acreate(group, "num_families", H5T_STD_U32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_STD_U32LE, nfamilies);
		H5Aclose(attr);

		//Autocorrelation Time
		attr = H5Acreate(group, "tau_int", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, tau_max);
		H5Aclose(attr);

		//Observables
		attr = H5Acreate(dset_action, "mean", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, action);
		H5Aclose(attr);

		attr = H5Acreate(dset_action, "stddev", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, action_err);
		H5Aclose(attr);

		attr = H5Acreate(dset_nrel, "mean", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, ofrac);
		H5Aclose(attr);

		attr = H5Acreate(dset_nrel, "stddev", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, ofrac_err);
		H5Aclose(attr);

		attr = H5Acreate(dset_nlink, "mean", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, links);
		H5Aclose(attr);

		attr = H5Acreate(dset_nlink, "stdderr", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, links_err);
		H5Aclose(attr);

		//Thermodynamic variables
		attr = H5Acreate(group, "specific_heat", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, Cv);
		H5Aclose(attr);

		attr = H5Acreate(group, "specific_heat_err", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, Cv_err);
		H5Aclose(attr);

		attr = H5Acreate(group, "free_energy", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, bF);
		H5Aclose(attr);

		attr = H5Acreate(group, "free_energy_err", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, bF_err);
		H5Aclose(attr);

		attr = H5Acreate(group, "entropy", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, entr);
		H5Aclose(attr);

		attr = H5Acreate(group, "entropy_err", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, entr_err);
		H5Aclose(attr);

		H5Sclose(aspace);

		//Overlap parameters used for adaptive temperature selection
		//We estimate the energy histogram overlap at adjacent temperatures will
		//fall within the range [min_overlap,max_overlap]
		rowdim[0] = 2;
		((double*)data)[0] = min_overlap;
		((double*)data)[1] = max_overlap;
		aspace = H5Screate_simple(2, rowdim, NULL);
		attr = H5Acreate(group, "histogram_overlap", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, data);
		H5Aclose(attr);

		//Couplings used in the calculation of the action
		attr = H5Acreate(group, "action_couplings", H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, A);
		H5Aclose(attr);
		H5Sclose(aspace);

	}

	//Local population size (per MPI process)
	//This information is useful for understanding load balancing
	#ifdef MPI_ENABLED
	rowdim[0] = nprocs;
	rowdim[1] = b + 1;
	aspace = H5Screate_simple(2, rowdim, NULL);
	FREE(data);
	data = MALLOC<unsigned>((b + 1) * nprocs);
	MPI_Gather(Rloc, b + 1, MPI_UNSIGNED, data, b + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	if (!rank) {
		attr = H5Acreate(group, "local_size", H5T_STD_U32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_STD_U32LE, data);
		H5Aclose(attr);
	}
	H5Sclose(aspace);
	#endif

	CUDA_FREE(d_adj);
	CUDA_FREE(d_link);
	CUDA_FREE(d_U);
	CUDA_FREE(d_V);
	CUDA_FREE(d_action);
	CUDA_FREE(d_nrel);
	CUDA_FREE(d_nlink);
	CUDA_FREE(d_offspring);
	CUDA_FREE(d_parsum);
	CUDA_FREE(d_families);
	CUDA_FREE(d_rng);
	CUDA_FREE(d_observables);
	CUDA_FREE(d_action_biased);
	CUDA_FREE(d_Q);
	CUDA_FREE(d_Ri);
	CUDA_FREE(d_Reff);
	CUDA_FREE(d_overlap);

	FREE(data);
	FREE(acorr);
	FREE(lags);
	FREE(tau_max);

	FREE(action);
	FREE(action_err);
	FREE(entr);
	FREE(entr_err);
	FREE(Cv);
	FREE(Cv_err);
	FREE(lnQ);
	FREE(bF);
	FREE(bF_err);
	FREE(ofrac);
	FREE(ofrac_err);
	FREE(links);
	FREE(links_err);
	FREE(families);
	FREE(famcnt);
	FREE(nfamilies);

	FREE(B);
	FREE(R);
	FREE(Rloc);
	FREE(Reff_t);
	FREE(Reff_s);
	FREE(Reff_r);

	FREE(sendcnt);
	FREE(displs);

	if (!rank) {
		H5Dclose(dset_action);
		H5Dclose(dset_nrel);
		H5Dclose(dset_nlink);
		H5Dclose(dset_family);
		H5Dclose(dset_famcnt);
		H5Sclose(dspace);
		H5Gclose(group);
		H5Fclose(file);
	}

	checkCudaErrors(cuCtxSynchronize());
	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
	checkCudaErrors(cuEventRecord(stop, 0));
	checkCudaErrors(cuEventSynchronize(stop));
	checkCudaErrors(cuEventElapsedTime(&elapsed_time, start, stop));

	printf_mpi(network->network_properties.cmpi.rank, "\nElapsed Time: %.8f sec\n", elapsed_time / 1000);
	printf_mpi(network->network_properties.cmpi.rank, " > %.6f ns/step\n", elapsed_time * 1e6 / (theta[0] * steps_per_sweep * tot_pop));
	FREE(theta);

	return true;
}

//Packed Operations
//1 matrix entry = 1 bit
//Optimized for coalesced access

__global__ void RNGInit(RNGState *rng, uint64_t rng_seed, unsigned R)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < R)
		curand_init(rng_seed, id, 0, &rng[id]);
}

__global__ void ReplicaInitPacked(unsigned *adj, unsigned *families, RNGState *rng, const int N, const size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= N) return;

	RNGState localrng = rng[id];
	for (size_t i = 0; i < bitblocks; i++)
		adj[replica_offset+i*N+row] = curand(&localrng);
	rng[id] = localrng;

	families[id] = id;
}

template<size_t TBSIZE>
__global__ void ReplicaInit2D(unsigned *U, unsigned *V, unsigned *families, RNGState *rng, const int N, const int fam_offset)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned id = bid * N + tid;
	
	__shared__ unsigned shr_U[TBSIZE], shr_V[TBSIZE];
	RNGState localrng;
	if (!tid)
		localrng = rng[bid];
	if (tid < N)
		shr_U[tid] = shr_V[tid] = tid;	//Initialize as std::iota does
	__syncthreads();

	//Fisher-Yates shuffle of U, V
	//First half of threads shuffle U; second half shuffle V
	/*unsigned shift = 1;
	unsigned pos = tid << 1;
	if (DBG && bid >= 16) shift = N;
	while (shift <= N >> 1) {
		if (tid < N >> 1) {
			if (curand(&localrng) & 1)
				swap(shr_U, pos, pos + shift);
		} else if (tid < N) {
			if (curand(&localrng) & 1)
				swap(shr_V, pos - N, pos - N + shift);
		}
		shift <<= 1;
		pos = (pos & ~shift) | ((pos & shift) >> 1);
		__syncthreads();
	}*/

	//Only one RNG per thread block
	if (!tid) {
		for (unsigned i = N - 1; i > 0; i--) {
			swap(shr_U, i, (unsigned)(curand_uniform(&localrng) * i));
			swap(shr_V, i, (unsigned)(curand_uniform(&localrng) * i));
		}
	}
	__syncthreads();

	if (tid < N) {
		U[id] = shr_U[tid];
		V[id] = shr_V[tid];
		if (!tid) {
			rng[bid] = localrng;
			families[bid] = fam_offset + bid;
		}
	}
}

template<size_t TBSIZE>
__global__ void OrderToMatrix(unsigned *adj, unsigned *U, unsigned *V, unsigned *nrel, const int N, const size_t bitblocks)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned id = bid * N + tid;
	unsigned wid = tid >> 5;
	unsigned lane = tid & 31;

	static const bool DBG = false;

	__shared__ unsigned shr_U[TBSIZE], shr_V[TBSIZE];
	if (tid < N) {
		shr_U[tid] = U[id];
		shr_V[tid] = V[id];
	} else
		shr_U[tid] = shr_V[tid] = 0;
	__syncthreads();

	BitonicShared(shr_U, shr_V, N, tid);

	unsigned loc_U = shr_U[tid];
	unsigned loc_V = shr_V[tid];
	unsigned nrel_loc = 0;
	for (unsigned i = wid; i < (N >> 5) && tid < N; i++) {	//Column
		//Construct row 'tid' and columns 'i*32' through '(i+1)*32-1'
		unsigned blockval = 0;
		for (unsigned j = 0; j < 32; j++) {
			unsigned col = (i << 5) | j;	
			if (col > tid && col < N && ((loc_U < shr_U[col] && loc_V < shr_V[col]) || (shr_U[col] < loc_U && shr_V[col] < loc_V)))
				blockval |= 1 << j;
		}
		adj[(bid*bitblocks+i)*N+tid] = blockval;
		nrel_loc += __popc(blockval);
	}

	for (unsigned offset = 16; offset > 0; offset >>= 1)
		nrel_loc += __shfl_down_sync(0xFFFFFFFF, nrel_loc, offset, 32);

	if (!lane)
		atomicAdd(nrel + bid, nrel_loc);

	if (DBG && tid < N) {
		U[id] = shr_U[tid];
		V[id] = shr_V[tid];
	}
}

template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaCountPacked(unsigned *adj, unsigned *cnt, const int N, const size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned replica_offset = bid * bitblocks * N;
	unsigned lane = row & 31;
	unsigned wid = row >> 5;
	unsigned rowmask = (N & 31) ? (1u << (N & 31)) - 1 : UINT_MAX;
	unsigned cntmask = ~((1u << ((row + 1) & 31)) - 1);

	unsigned adj_row[ROW_CACHE_SIZE];
	unsigned local_cnt = 0;

	for (unsigned i = 0; i < bitblocks && row < N; i++)
		adj_row[i] = adj[replica_offset+i*N+row];

	RelationCount<ROW_CACHE_SIZE>(local_cnt, adj_row, N, bitblocks, row, wid, lane, rowmask, cntmask);

	if (!row)
		atomicAdd(cnt + bid, local_cnt);
}

template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaClosurePacked(unsigned *adj, const int N, const size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned lane = row & 31;
	unsigned wid = row >> 5;

	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	unsigned num_iters = ceilf(__log2f(N));
	for (unsigned n = 0; n < num_iters; n++) {
		for (int i = 0; i < N; i++) {	//Rows
			//Read row 'i' into shared memory
			if (row == i) {
				for (size_t j = 0; j < bitblocks; j++)
					s_row[j] = adj[replica_offset+j*N+row];
			}
			__syncthreads();

			//Thread 'j' multiplies column 'j' in s_row by
			//row 'j', column 'i'
			unsigned C_ijk = 0, C_ij = 0;
			unsigned active = (row > i) & (row < N);
			unsigned mask = __ballot_sync(0xFFFFFFFF, active);
			for (int j = i + 1; j < N; j++) {	//Columns
				C_ijk = active & (row < j) &
					((s_row[wid] >> lane) &
					(adj[replica_offset+(j>>5)*N+row] >> (j & 31)) & 1);
				C_ij = __any_sync(mask, C_ijk);
				if (!lane)
					s_res[wid] = C_ij;
				__syncthreads();
				//if (!lane)	//Column j, row i is updated
				//	atomicOr(&adj[replica_offset+(j>>5)*N+i], s_res[wid] << (j & 31));
				if (row == i)
					for (unsigned k = 0; k < bitblocks; k++)
						adj[replica_offset+(j>>5)*N+i] |= s_res[k] << (j & 31);
				__syncthreads();
			}
			__syncthreads();
		}
	}
}

template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaReductionPacked(unsigned *adj, const int N, const size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned lane = row & 31;
	unsigned wid = row >> 5;

	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	for (int i = 0; i < N - 1; i++) {	//Rows
		//Read row 'i' into shared memory
		if (row == i) {
			for (size_t j = 0; j < bitblocks; j++)
				s_row[j] = adj[replica_offset+j*N+row];
		}
		__syncthreads();

		//Thread 'j' multiplies column 'j' in s_row by
		//row 'j', column 'i'
		unsigned C_ijk = 0, C_ij = 0;
		unsigned active = (row > i) & (row < N);
		unsigned mask = __ballot_sync(0xFFFFFFFF, active);
		for (int j = i + 1; j < N; j++) {
			unsigned adj_val = row < N ? (adj[replica_offset+(j>>5)*N+row] >> (j & 31)) & 1 : 0;
			C_ijk = active & (row < j) &
				((s_row[wid] >> lane) &	//Row 'i' column 'k'
				adj_val);	//Row 'k' column 'j'
			C_ij = __any_sync(mask, C_ijk);
			if (!lane)
				s_res[wid] = C_ij;
			__syncthreads();
			//s_res now contains a '1' somewhere if the Alexandroff set A(i,j) is non-zero
			//If entry (i,j) is non-zero, and this shared value is non-zero, set entry (i,j) to 0 (set difference)
			if (row == i) {
				unsigned res = 0;
				for (unsigned k = 0; k < bitblocks; k++)
					res |= s_res[k];
				adj_val = ~(!(adj_val & (adj_val ^ res)) << (j & 31));
				adj[replica_offset+(j>>5)*N+i] &= adj_val;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

__global__ void RelationAction(double *action, double *observables, unsigned *nrel, unsigned *nlink, const unsigned R, const double A0, const double A1)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;

	if (idx < R) {
		unsigned r = nrel[idx];
		unsigned l = nlink[idx];
		double act = A0 * l + A1 * (r - l);
	
		action[idx] = act;
		atomicAdd(observables, act);
		atomicAdd(observables + 1, act * act);
	}
}

template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaActionPairPacked(double *action, double *observables, unsigned *adj, unsigned *nrel, const int N, const size_t bitblocks, const double A, const double B)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned lane = row & 31;
	unsigned wid = row >> 5;

	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	unsigned sum = 0;
	unsigned jb, kb, j, t1, t2;

	for (unsigned i = 0; i < N - 2; i++) {	//First element (row)
		if (row == i)
			for (unsigned j = 0; j < bitblocks; j++)
				s_row[j] = adj[replica_offset+j*N+i];
		__syncthreads();

		if (row >= i && row < N) {
			for (unsigned j_block = (i + 1) >> 5; j_block < bitblocks; j_block++) {	//Second element (column)
				jb = s_row[j_block];
				unsigned j_bit = (j_block == ((i + 1) >> 5)) * ((i + 1) & 31);
				for (; j_bit < min(32, N); j_bit++) {
					t1 = jb >> j_bit;
					j = (j_block << 5) | j_bit;
					for (unsigned k_block = (row + 1) >> 5; k_block < bitblocks && j < N; k_block++) {	//Fourth element (column)
						kb = adj[replica_offset+k_block*N+row];
						unsigned k_bit = (k_block == ((row + 1) >> 5)) * ((row + 1) & 31);
						for (; k_bit < min(32, N); k_bit++) {
							unsigned k = (k_block << 5) | k_bit;
							if (i == row && k <= j) continue;
							t2 = kb >> k_bit;
							sum += ((t1 ^ t2) & 1) && (row < k);
						}
					}
				}
			}
		}
	}

	for (unsigned offset = 16; offset > 0; offset >>= 1)
		sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, 32);
	
	if (!lane)
		s_res[wid] = sum;
	__syncthreads();

	for (unsigned stride = 1; stride < bitblocks; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)) && wid + stride < bitblocks)
			s_res[wid] += s_res[wid+stride];
		__syncthreads();
	}

	if (!row) {
		double local_action = A * s_res[0] + B * nrel[blockIdx.x];
		action[blockIdx.x] = local_action;
		atomicAdd(observables, local_action);
		atomicAdd(observables + 1, local_action * local_action);
	}
}

__global__ void QKernel(double *action, int R, double dB, double mean_action, double *Q)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;

	double factor = idx < R ? exp(-dB * (action[idx] - mean_action)) : 0.0;

	//Reduce 'factor' across all threads in a warp
	for (unsigned offset = 16; offset > 0; offset >>= 1)
		factor += __shfl_down_sync(0xFFFFFFFF, factor, offset, 32);

	//Reduce across warps in a thread block
	__shared__ double s_factor[32];
	unsigned lane = tid & 31;
	unsigned wid = tid >> 5;
	if (!lane)
		s_factor[wid] = factor;
	__syncthreads();

	for (unsigned stride = 1; stride < blockDim.x >> 5; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)))
			s_factor[wid] += s_factor[wid+stride];
		__syncthreads();
	}

	if (!tid)
		atomicAdd(&Q[0], s_factor[0]);
}

__global__ void TauKernel_v2(double *action, unsigned *offspring, unsigned *partial_sums, RNGState *rng, unsigned R0, unsigned R, unsigned Rloc, double lnQ, double dB, unsigned *Rnew, double *Reff)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;
	unsigned wid = tid >> 5;
	unsigned lane = tid & 31;

	RNGState localrng = rng[idx];
	__shared__ unsigned s_parS[32];
	//__shared__ double s_parR[32];
	unsigned parS = 0;
	//double parR = 0.0;

	//Calculate number of offspring for each replica
	if (idx < Rloc) {
		double mu = exp(-dB * action[idx] - lnQ) * R0 / R;
		double mufloor = floor(mu);
		if (curand_uniform_double(&localrng) < (mu - mufloor))
			parS = offspring[idx] = mufloor + 1;
		else
			parS = offspring[idx] = mufloor;
		//parR = pow((double)parS / R, 2.0);
	}

	//Reduce this for the total
	for (unsigned offset = 16; offset > 0; offset >>= 1) {
		parS += __shfl_down_sync(0xFFFFFFFF, parS, offset, 32);
		//parR += __shfl_down_sync(0xFFFFFFFF, parR, offset, 32);
	}

	if (!lane) {
		s_parS[wid] = parS;
		//s_parR[wid] = parR;
	}
	__syncthreads();

	for (unsigned stride = 1; stride < blockDim.x >> 5; stride <<= 1) {
		if (!lane && !(wid % (stride << 1))) {
			s_parS[wid] += s_parS[wid+stride];
			//s_parR[wid] += s_parR[wid+stride];
		}
		__syncthreads();
	}

	if (!tid) {
		parS = s_parS[0];
		partial_sums[idx] = parS;
		atomicAdd(Rnew, parS);

		//parR = s_parR[0];
		//atomicAdd(Reff, parR);
	}

	rng[idx] = localrng;
}

__global__ void PartialSum(unsigned *offspring, unsigned *partial_sums, unsigned R)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;
	unsigned lane = tid & 31;
	unsigned wid = tid >> 5;

	__shared__ unsigned s_parS[32];
	__shared__ unsigned val;
	unsigned parS, myParSum = 0;

	for (unsigned j = 0; j < bid; j += blockDim.x) {
		parS = (tid + j < bid) ? partial_sums[(tid+j)*blockDim.x] : 0;
		for (unsigned offset = 16; offset > 0; offset >>= 1)
			parS += __shfl_down_sync(0xFFFFFFFF, parS, offset, 32);
		if (!lane)
			s_parS[wid] = parS;
		__syncthreads();
		for (unsigned stride = 1; stride < blockDim.x >> 5; stride <<= 1) {
			if (!lane && !(wid % (stride << 1)))
				s_parS[wid] += s_parS[wid+stride];
			__syncthreads();
		}
		if (!tid)
			val = s_parS[0];
		__syncthreads();
		myParSum += val;
	}

	if (idx < R) {
		for (unsigned j = blockDim.x * bid; j < idx; j++)
			myParSum += offspring[j];
		partial_sums[R+idx] = myParSum;
	}
}

__global__ void ResampleReplicas2D(unsigned *U_new, unsigned *U, unsigned *V_new, unsigned *V, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned *families_new, unsigned *families, unsigned R, unsigned N)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned id = bid * N + tid;

	unsigned j = partial_sums[R+bid];
	unsigned jnext = j + offspring[bid];
	if (tid < N) {
		for (; j < jnext; j++) {
			U_new[j*N+tid] = U[id];
			V_new[j*N+tid] = V[id];
			if (!tid) {
				action_new[j] = action[bid];
				families_new[j] = families[bid];
			}
		}
	}
}

__global__ void ResampleReplicasPacked(unsigned *adj_new, unsigned *adj, unsigned *link_new, unsigned *link, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned *families_new, unsigned *families, unsigned R, unsigned N, size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;

	unsigned j = partial_sums[R+blockIdx.x];
	unsigned jnext = j + offspring[blockIdx.x];
	if (row < N) {
		for (; j < jnext; j++) {
			for (unsigned i = 0; i < bitblocks; i++) {
				adj_new[(i+j*bitblocks)*N+row] = adj[replica_offset+i*N+row];
				link_new[(i+j*bitblocks)*N+row] = link[replica_offset+i*N+row];
			}
			if (!row) {
				action_new[j] = action[blockIdx.x];
				families_new[j] = families[blockIdx.x];
			}
		}
	}
}

//This kernel works when N <= 32*ROW_CACHE_SIZE
//observables[0] = <E>
//observables[1] = <E^2>
//observables[2] = <R>
//observables[3] = <L>
template<GraphType gt, WeightFunction wf, size_t ROW_CACHE_SIZE>
__global__ void RelaxReplicasPacked(unsigned *adj, unsigned *link, unsigned *U, unsigned *V, unsigned *nrel, unsigned *nlink, double *action, RNGState *rng, unsigned sweeps, unsigned steps, unsigned N, double beta, double A0, double A1, size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned lane = row & 31;
	unsigned wid = row >> 5;
	unsigned rowmask = (N & 31) ? (1u << (N & 31)) - 1 : UINT_MAX;
	unsigned cntmask = ~((1u << ((row + 1) & 31)) - 1);

	RNGState localrng;
	if (!row)
		localrng = rng[blockIdx.x];
	//beta = 0.0;

	unsigned adj_row[ROW_CACHE_SIZE], link_row[ROW_CACHE_SIZE];
	memset(adj_row, 0, sizeof(unsigned) * ROW_CACHE_SIZE);
	memset(link_row, 0, sizeof(unsigned) * ROW_CACHE_SIZE);

	__shared__ unsigned shr_U[ROW_CACHE_SIZE<<5], shr_V[ROW_CACHE_SIZE<<5];

	unsigned closure_iters = ceilf(__log2f(N));

	__shared__ unsigned flip_id;			//ID of element selected for flip
	__shared__ bool accept;
	unsigned current_nrel, new_nrel, current_nlink, new_nlink;
	double current_action, new_action, dS;
	unsigned loc_U, loc_V;
	unsigned loc_U_sorted, loc_V_sorted;

	if (!row) {
		current_action = action[blockIdx.x];
		current_nrel = nrel[blockIdx.x];
		current_nlink = nlink[blockIdx.x];
		new_nrel = new_nlink = 0;
		//printf("Initial action: %f\n", current_action);
		//printf("beta: %f\n", beta);
		//if (!blockIdx.x && !row) printf("%u\n", current_nlink);
	}

	if (row < N) {
		if (gt == _2D_ORDER) {
			loc_U = shr_U[row] = U[blockIdx.x*N+row];
			loc_V = shr_V[row] = V[blockIdx.x*N+row];
		} else {
			for (unsigned i = wid; i < bitblocks; i++) {
				//adj_row[i] = adj[replica_offset+i*N+row];
				link_row[i] = link[replica_offset+i*N+row];
			}
		}
	}
	__syncthreads();

	for (unsigned sweep = 0; sweep < sweeps; sweep++) {
		for (unsigned step = 0; step < steps; step++) {
		//for (unsigned step = 0; step < 1; step++) {
			unsigned col_block = bitblocks;
			unsigned col_bit = 32;
			unsigned order_id, flip_id0, flip_id1;

			if (gt == _2D_ORDER) {
				if (row < N) {
					loc_U = shr_U[row];
					loc_V = shr_V[row];
				}
				__syncthreads();

				if (!row) {
					//Select one of the two total orders and a random pair
					RandomOrderPair(order_id, flip_id0, flip_id1, &localrng, N);
					//Execute the update
					swap(order_id ? shr_U : shr_V, flip_id0, flip_id1);
					//if (!blockIdx.x) printf("swapping [%u] and [%u] in order [%s]\n", flip_id0, flip_id1, order_id ? "U" : "V");
				}
				__syncthreads();

				//Generate a natural ordering
				BitonicShared(shr_U, shr_V, N, row);

				//Build the adjacency matrix
				if (row < N) {
					loc_U_sorted = shr_U[row];
					loc_V_sorted = shr_V[row];
				}
				for (unsigned i = wid; i < (N >> 5) && row < N; i++) {
					//Construct row 'tid' and columns 'i*32' through '(i+1)*32-1'
					unsigned blockval = 0;
					for (unsigned j = 0; j < 32; j++) {
						unsigned col = (i << 5) | j;	
						if (col > row && col < N && ((loc_U_sorted < shr_U[col] && loc_V_sorted < shr_V[col]) || (shr_U[col] < loc_U_sorted && shr_V[col] < loc_V_sorted)))
							blockval |= 1 << j;
					}
					adj_row[i] = blockval;
				}
				__syncthreads();

				//Restore the shared memory (undo topological sort)
				if (row < N) {
					shr_U[row] = loc_U;
					shr_V[row] = loc_V;
				}
				__syncthreads();

				//Count the number of relations after the update
				RelationCount<ROW_CACHE_SIZE>(new_nrel, adj_row, N, bitblocks, row, wid, lane, rowmask, cntmask);
				//Generate the link matrix
				Reduction<ROW_CACHE_SIZE>(link_row, adj_row, N, bitblocks, row, wid, lane);
				//Count the number of links after the update
				RelationCount<ROW_CACHE_SIZE>(new_nlink, link_row, N, bitblocks, row, wid, lane, rowmask, cntmask);
				//if (!blockIdx.x && !row) printf("%u\n", new_nlink);
			} else {
				//RandomBit<ROW_CACHE_SIZE>(&flip_id, col_block, col_bit, adj_row, link_row, &localrng, N, bitblocks, row, wid, lane, rowmask, cntmask);
				RandomBitAny<ROW_CACHE_SIZE>(&flip_id, col_block, col_bit, &localrng, N, row);

				//Re-load adj from shared memory cache
				//if (row < N)
				//	for (unsigned i = wid; i < bitblocks; i++)
				//		adj_row[i] = adj[replica_offset+i*N+row];

				//Execute the flip
				if (col_block != bitblocks)
					//adj_row[col_block] ^= 1u << col_bit;
					link_row[col_block] ^= 1u << col_bit;

				//Transitive reduction
				//Reduction<ROW_CACHE_SIZE>(link_row, adj_row, N, bitblocks, row, wid, lane);
				Reduction<ROW_CACHE_SIZE>(adj_row, link_row, N, bitblocks, row, wid, lane);
				RelationCount<ROW_CACHE_SIZE>(new_nlink, adj_row, N, bitblocks, row, wid, lane, rowmask, cntmask);

				//Transitive closure
				Closure<ROW_CACHE_SIZE>(adj_row, N, bitblocks, closure_iters, row, wid, lane);
				RelationCount<ROW_CACHE_SIZE>(new_nrel, adj_row, N, bitblocks, row, wid, lane, rowmask, cntmask);
			}

			//Calculate new action
			switch (wf) {
			case RELATION:
				//Relation action
				new_action = A0 * new_nlink + A1 * (new_nrel - new_nlink);
				break;
			case RELATION_PAIR:
			{
				unsigned cnt = 0;
				RelationPairCount<ROW_CACHE_SIZE>(cnt, adj_row, N, bitblocks, row, wid, lane);
				new_action = A0 * (double)cnt + A1 * new_nrel;
				//Add the anti-percolation weight
				//new_action += (double)(nrel - nlink) * log(2.0);
				break;
			}
			case ANTIPERCOLATION:
				new_action = (double)(new_nrel - new_nlink) * log(2.0);
				break;
			default:
				assert (false);
			}
			//if (!row) printf("\nProposed new action: %f\n", new_action);

			//Metropolis step
			if (!row) {
				dS = beta * (new_action - current_action);
				accept = dS <= 0 || exp(-dS) > curand_uniform_double(&localrng);
				if (accept) {
					current_action = new_action;
					current_nrel = new_nrel;
					current_nlink = new_nlink;
					if (gt == _2D_ORDER)
						swap(order_id ? shr_U : shr_V, flip_id0, flip_id1);
				}
				//if (!blockIdx.x && accept) printf("Matrix move accepted (GPU) %f\n", new_action);
				//if (!blockIdx.x && !accept) printf("Matrix move rejected (GPU)\n");
			}
			__syncthreads();

			if (accept) {
				//Write adj
				//for (unsigned i = wid; i < bitblocks; i++)
				//	adj[replica_offset+i*N+row] = adj_row[i];
				//__syncthreads();

				//Transitive reduction, link <- adj
				//Reduction<ROW_CACHE_SIZE>(link_row, adj_row, N, bitblocks, row, replica_offset, wid, lane);
				//if (col_block != bitblocks)
				//	link[replica_offset+col_block*N+row] = link_row[col_block];
			} else {
				//Simply re-load adj
				//for (unsigned i = wid; i < bitblocks; i++)
				//	adj_row[i] = adj[replica_offset+i*N+row];
				if (gt != _2D_ORDER) {
					//Reverse the flip
					if (col_block != bitblocks)
						link_row[col_block] ^= 1u << col_bit;
				}
			}
		}
	}

	if (row < N) {
		if (gt == _2D_ORDER) {
			U[blockIdx.x*N+row] = shr_U[row];
			V[blockIdx.x*N+row] = shr_V[row];
		} else {
			for (unsigned i = wid; i < bitblocks; i++) {
				adj[replica_offset+i*N+row] = adj_row[i];
				link[replica_offset+i*N+row] = link_row[i];
			}
		}
	}

	if (!row) {
		rng[blockIdx.x] = localrng;
		action[blockIdx.x] = current_action;
		nrel[blockIdx.x] = current_nrel;
		nlink[blockIdx.x] = current_nlink;
		//if (!blockIdx.x)
		//	printf("Final action [%u]: %f\n", blockIdx.x, current_action);
	}
}

__global__ void MeasureObservables(double *observables, double *biased_action, double *action, unsigned *nrel, unsigned *nlink, unsigned R, unsigned npairs, unsigned stride)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;
	unsigned lane = tid & 31;

	unsigned use_value = !(idx % stride);
	double local_biased_action = (idx < R) ? action[idx] : 0.0;
	double local_action = local_biased_action * use_value;
	double local_square_action = local_action * local_action;
	unsigned local_nrel = (idx < R) ? nrel[idx] * use_value : 0;
	unsigned local_nlink = (idx < R) ? nlink[idx] * use_value : 0;

	//Accumulate values per warp
	for (unsigned offset = 16; offset > 0; offset >>= 1) {
		local_biased_action += __shfl_down_sync(0xFFFFFFFF, local_biased_action, offset, 32);
		local_action += __shfl_down_sync(0xFFFFFFFF, local_action, offset, 32);
		local_square_action += __shfl_down_sync(0xFFFFFFFF, local_square_action, offset, 32);
		local_nrel += __shfl_down_sync(0xFFFFFFFF, local_nrel, offset, 32);
		local_nlink += __shfl_down_sync(0xFFFFFFFF, local_nlink, offset, 32);
	}

	if (!lane) {
		atomicAdd(observables, local_action);
		atomicAdd(observables + 1, local_square_action);
		atomicAdd(observables + 2, local_nrel / npairs);
		atomicAdd(observables + 3, local_nlink / npairs);
		atomicAdd(biased_action, local_biased_action);
	}
}

__global__ void HistogramOverlap(double *action, unsigned R0, unsigned R, unsigned Rloc, double lnQ, double dB, double *overlap, double mean_energy)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;
	unsigned lane = tid & 31;
	unsigned wid = tid >> 5;

	double partial_overlap = (idx < Rloc) ? min(1.0, exp(-dB * action[idx] - lnQ) * R0 / R) : 0.0;

	__shared__ double shr_partial_overlap[32];
	for (unsigned offset = 16; offset > 0; offset >>= 1)
		partial_overlap += __shfl_down_sync(0xFFFFFFFF, partial_overlap, offset, 32);
	if (!lane)
		shr_partial_overlap[wid] = partial_overlap;
	__syncthreads();

	for (unsigned stride = 1; stride < blockDim.x >> 5; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)))
			shr_partial_overlap[wid] += shr_partial_overlap[wid+stride];
		__syncthreads();
	}

	if (!tid)
		atomicAdd(overlap, shr_partial_overlap[0]);
}

double calc_histogram_overlap(CUdeviceptr &d_action, CUdeviceptr &d_Q, CUdeviceptr &d_overlap, const unsigned R0, const unsigned R, const unsigned Rloc, const double dB, const double mean_action, const dim3 blocks_per_grid, const dim3 threads_per_block, unsigned rank)
{
	double q, lnQ, overlap;

	checkCudaErrors(cuMemsetD32(d_Q, 0, 2));
	QKernel<<<blocks_per_grid, threads_per_block>>>((double*)d_action, Rloc, dB, mean_action, (double*)d_Q);
	checkCudaErrors(cuMemcpyDtoH(&q, d_Q, sizeof(double)));
	#ifdef MPI_ENABLED
	MPI_Allreduce(MPI_IN_PLACE, &q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	#endif
	lnQ = -dB * mean_action + log(q) - log((double)R);

	checkCudaErrors(cuMemsetD32(d_overlap, 0, 2));
	HistogramOverlap<<<blocks_per_grid, threads_per_block>>>((double*)d_action, R0, R, Rloc, lnQ, dB, (double*)d_overlap, mean_action);
	checkCudaErrors(cuMemcpyDtoH(&overlap, d_overlap, sizeof(double)));
	#ifdef MPI_ENABLED
	MPI_Allreduce(MPI_IN_PLACE, &overlap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	#endif

	return overlap / R;
}

void adaptive_beta_stepsize(double &dB, CUdeviceptr d_action, CUdeviceptr d_Q, CUdeviceptr d_overlap, const unsigned R0, const unsigned R, const unsigned Rloc, const double beta, const double beta_final, const double min_overlap, const double max_overlap, const double mean_action, const dim3 blocks_per_grid, const dim3 threads_per_block, unsigned rank)
{
	double overlap, dBmin = 0.0, dBmax = dB, dBmean;
	while (1) {
		overlap = calc_histogram_overlap(d_action, d_Q, d_overlap, R0, R, Rloc, dBmax, mean_action, blocks_per_grid, threads_per_block, rank);
		//printf_mpi(rank, "overlap1: %f\n", overlap);
		if ((overlap >= max_overlap) && (beta + dBmax < beta_final))
			dBmax *= 1.1;
		else
			break;
	}

	if (overlap >= min_overlap)
		dBmean = dBmax;
	else {
		double dBm0 = 0.5 * (dBmin + dBmax);
		while (1) {
			dBmean = 0.5 * (dBmin + dBmax);
			overlap = calc_histogram_overlap(d_action, d_Q, d_overlap, R0, R, Rloc, dBmean, mean_action, blocks_per_grid, threads_per_block, rank);
			if (overlap < min_overlap)
				dBmax = dBmean;
			else if (overlap >= max_overlap)
				dBmin = dBmean;
			else
				break;
			//The bisection method has failed
			//This happens when q=R, or dB is too small
			//It causes underflow errors
			//if (dBmean <= 1e-6) {
			if (fabs(overlap - ((float)R0 / R)) / overlap < 1.0e-5) {
				dBmean = dBm0;
				//fprintf(stderr, "Bisection failed. Continuing with dB = %.5e\n", dBmean);
				break;
			}
		}
	}

	if ((beta < beta_final) && (beta + dBmean > beta_final))
		dB = beta_final - beta;
	else
		dB = dBmean;
}

/*double autocorrelation(double *h_action_fine, double *acorr, unsigned theta, unsigned theta_max, unsigned &max_lag, unsigned rank)
{
	memset(acorr, 0, sizeof(double) * (theta_max - 2));
	acorr[0] = 1;
	for (unsigned lag = 1; lag < theta - 2; lag++) {
		acorr[lag] = gsl_stats_correlation(h_action_fine, 1, h_action_fine + lag, 1, theta - lag);
		if (acorr[lag] != acorr[lag])
			acorr[lag] = 0.0;
	}

	//if (b == 1 && !rank) {
	//	FILE *f_acorr = fopen("autocorr_coeff.txt", "w");
	//	fprintf(f_acorr, "1.0\n");
	//	for (unsigned lag = 1; lag < theta - 2; lag++)
	//		fprintf(f_acorr, "%f\n", acorr[lag]);
	//	fflush(f_acorr);	fclose(f_acorr);
	//}

	double tau_int = acorr[0] / 2;
	max_lag = theta;
	for (unsigned lag = 1; lag < theta - 2; lag++) {
		tau_int += acorr[lag];
		if (lag >= 6 * tau_int) {
			max_lag = lag;
			break;
		}
	}

	return tau_int;
}*/

#ifdef MPI_ENABLED
void balance_replicas(CUdeviceptr &d_adj, CUdeviceptr &d_link, CUdeviceptr &d_U, CUdeviceptr &d_V, CUdeviceptr &d_nrel, CUdeviceptr &d_nlink, CUdeviceptr &d_action, CUdeviceptr &d_offspring, CUdeviceptr &d_parsum, CUdeviceptr &d_families, GraphType gt, unsigned N, unsigned R, unsigned &Rloc, size_t uints_per_replica, size_t &nblocks_rep, size_t &nblocks_obs, dim3 &blocks_per_grid_rep, dim3 &blocks_per_grid_obs, unsigned rank, unsigned mpi_threads)
{
	//Move the variables to the master process
	int *sendcnt = NULL, *offset = NULL;
	sendcnt = MALLOC<int>(mpi_threads);
	offset = MALLOC<int>(mpi_threads);
	MPI_Gather(&Rloc, 1, MPI_INT, sendcnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Host Buffers
	unsigned *h_adj = MALLOC<unsigned>(uints_per_replica * R);
	unsigned *h_link = MALLOC<unsigned>(uints_per_replica * R);
	unsigned *h_U = MALLOC<unsigned>(N * R);
	unsigned *h_V = MALLOC<unsigned>(N * R);
	unsigned *h_nrel = MALLOC<unsigned>(R);
	unsigned *h_nlink = MALLOC<unsigned>(R);
	double *h_action = MALLOC<double>(R);
	unsigned *h_families = MALLOC<unsigned>(R);

	//Copy data from GPU to RAM
	if (gt == _2D_ORDER) {
		checkCudaErrors(cuMemcpyDtoH(h_U, d_U, sizeof(unsigned) * N * nblocks_rep));
		checkCudaErrors(cuMemcpyDtoH(h_V, d_V, sizeof(unsigned) * N * nblocks_rep));
	} else {
		checkCudaErrors(cuMemcpyDtoH(h_adj, d_adj, sizeof(unsigned) * uints_per_replica * nblocks_rep));
		checkCudaErrors(cuMemcpyDtoH(h_link, d_link, sizeof(unsigned) * uints_per_replica * nblocks_rep));
	}
	checkCudaErrors(cuMemcpyDtoH(h_nrel, d_nrel, sizeof(unsigned) * nblocks_rep));
	checkCudaErrors(cuMemcpyDtoH(h_nlink, d_nlink, sizeof(unsigned) * nblocks_rep));
	checkCudaErrors(cuMemcpyDtoH(h_action, d_action, sizeof(double) * nblocks_rep));
	checkCudaErrors(cuMemcpyDtoH(h_families, d_families, sizeof(unsigned) * nblocks_rep));

	//Move data to the master process
	//Gather relation, link, action, family values
	if (!rank) {
		offset[0] = 0;
		for (unsigned i = 1; i < mpi_threads; i++)
			offset[i] = offset[i-1] + sendcnt[i-1];
			
		MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_nrel, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_nlink, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_action, sendcnt, offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_families, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	} else {
		MPI_Gatherv(h_nrel, Rloc, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		MPI_Gatherv(h_nlink, Rloc, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		MPI_Gatherv(h_action, Rloc, MPI_DOUBLE, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		MPI_Gatherv(h_families, Rloc, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
	}

	if (gt == _2D_ORDER) {
		//Gather U, V total orders
		if (!rank) {
			for (unsigned i = 0; i < mpi_threads; i++) {
				sendcnt[i] *= N;
				offset[i] *= N;
			}
			MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_U, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_V, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		} else {
			MPI_Gatherv(h_U, Rloc * N, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
			MPI_Gatherv(h_V, Rloc * N, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		}
	} else {
		//Gather adjacency, link values
		if (!rank) {
			for (unsigned i = 0; i < mpi_threads; i++) {
				sendcnt[i] *= uints_per_replica;
				offset[i] *= uints_per_replica;
			}
			MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_adj, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			MPI_Gatherv(MPI_IN_PLACE, 0, NULL, h_link, sendcnt, offset, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		} else {
			MPI_Gatherv(h_adj, Rloc * uints_per_replica, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
			MPI_Gatherv(h_link, Rloc * uints_per_replica, MPI_UNSIGNED, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
		}
	}

	//Free GPU memory
	CUDA_FREE(d_adj);
	CUDA_FREE(d_link);
	CUDA_FREE(d_U);
	CUDA_FREE(d_V);
	CUDA_FREE(d_nrel);
	CUDA_FREE(d_nlink);
	CUDA_FREE(d_action);
	CUDA_FREE(d_offspring);
	CUDA_FREE(d_parsum);
	CUDA_FREE(d_families);

	//Determine new partitioning
	Rloc = ceil((float)R / mpi_threads);
	unsigned new_offset = Rloc * rank;
	if ((rank+1) * Rloc > R)
		Rloc -= (rank+1) * Rloc - R;
	nblocks_rep = Rloc;
	nblocks_obs = ceil((float)nblocks_rep / 1024);
	blocks_per_grid_rep.x = nblocks_rep;
	blocks_per_grid_obs.x = nblocks_obs;

	//Move the data back to the slave processes
	MPI_Allgather(&Rloc, 1, MPI_INT, sendcnt, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&new_offset, 1, MPI_INT, offset, 1, MPI_INT, MPI_COMM_WORLD);
	if (!rank) {
		MPI_Scatterv(h_nrel, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
		MPI_Scatterv(h_nlink, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
		MPI_Scatterv(h_action, sendcnt, offset, MPI_DOUBLE, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
		MPI_Scatterv(h_families, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
	} else {
		MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_nrel, Rloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_nlink, Rloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, sendcnt, offset, MPI_DOUBLE, h_action, Rloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_families, Rloc, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	}

	if (gt == _2D_ORDER) { 
		for (unsigned i = 0; i < mpi_threads; i++) {
			sendcnt[i] *= N;
			offset[i] *= N;
		}

		if (!rank) {
			MPI_Scatterv(h_U, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
			MPI_Scatterv(h_V, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
		} else {
			MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_U, Rloc * N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_V, Rloc * N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		}
	} else {
		for (unsigned i = 0; i < mpi_threads; i++) {
			sendcnt[i] *= uints_per_replica;
			offset[i] *= uints_per_replica;
		}

		if (!rank) {
			MPI_Scatterv(h_adj, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
			MPI_Scatterv(h_link, sendcnt, offset, MPI_UNSIGNED, MPI_IN_PLACE, 0, NULL, 0, MPI_COMM_WORLD);
		} else {
			MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_adj, Rloc * uints_per_replica, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			MPI_Scatterv(NULL, sendcnt, offset, MPI_UNSIGNED, h_link, Rloc * uints_per_replica, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		}
	}

	//Re-allocate the GPU memory
	d_adj = CUDA_MALLOC<unsigned, 0>(nblocks_rep * uints_per_replica);
	d_link = CUDA_MALLOC<unsigned, 0>(nblocks_rep * uints_per_replica);
	d_U = CUDA_MALLOC<unsigned, 0>(nblocks_rep * N);
	d_V = CUDA_MALLOC<unsigned, 0>(nblocks_rep * N);
	d_nrel = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	d_nlink = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	d_action = CUDA_MALLOC<double, 0>(nblocks_rep);
	d_offspring = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	d_parsum = CUDA_MALLOC<unsigned, 0>(2 * nblocks_rep);
	d_families = CUDA_MALLOC<unsigned, 0>(nblocks_rep);
	checkCudaErrors(cuCtxSynchronize());

	//Move data from RAM to GPU
	if (gt == _2D_ORDER) {
		checkCudaErrors(cuMemcpyHtoD(d_U, h_U, sizeof(unsigned) * nblocks_rep * N));
		checkCudaErrors(cuMemcpyHtoD(d_V, h_V, sizeof(unsigned) * nblocks_rep * N));
	} else {
		checkCudaErrors(cuMemcpyHtoD(d_adj, h_adj, sizeof(unsigned) * nblocks_rep * uints_per_replica));
		checkCudaErrors(cuMemcpyHtoD(d_link, h_link, sizeof(unsigned) * nblocks_rep * uints_per_replica));
	}
	checkCudaErrors(cuMemcpyHtoD(d_nrel, h_nrel, sizeof(unsigned) * nblocks_rep));
	checkCudaErrors(cuMemcpyHtoD(d_nlink, h_nlink, sizeof(unsigned) * nblocks_rep));
	checkCudaErrors(cuMemcpyHtoD(d_action, h_action, sizeof(double) * nblocks_rep));
	checkCudaErrors(cuMemcpyHtoD(d_families, h_families, sizeof(unsigned) * nblocks_rep));
	checkCudaErrors(cuCtxSynchronize());

	//Free buffers
	FREE(h_adj);
	FREE(h_link);
	FREE(h_U);
	FREE(h_V);
	FREE(h_nrel);
	FREE(h_nlink);
	FREE(h_action);
	FREE(h_families);
	FREE(sendcnt);
	FREE(offset);
}
#endif

//Unpacked Operations
//These were developed for testing/debugging
//They are no longer supported, but kept for reference
//Ignore everything below here

__global__ void ReplicaInit(unsigned *adj, uint64_t rng_seed, uint64_t initial_sequence, const int N)
{
	unsigned row = threadIdx.y;
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	if (row >= N) return;

	RNGState localrng;
	curand_init(rng_seed, initial_sequence+(col_start+row), 0, &localrng);
	for (unsigned i = 0; i < N; i++)	//Columns in a single replica
		adj[replica_offset+i*N+row] = i == row ? 0 : i < row || curand_uniform_double(&localrng) < 0.8 ? 0 : 1;
}

template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaClosure(unsigned *adj, const int N, const int iter)
{
	unsigned row = threadIdx.y;
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	__shared__ unsigned e0[ROW_CACHE_SIZE];
	if (!row)
		for (unsigned i = 0; i < N; i++)
			e0[i] = adj[replica_offset+i*N+iter];
	__syncthreads();

	int e1;
	for (unsigned i = 0; i < N; i++) {
		__syncthreads();
		if (!e0[i] || row >= N) continue;
		e1 = adj[replica_offset+iter*N+row];
		if (!e1) continue;
		adj[replica_offset+i*N+row] = 1;
	}
}

template<size_t ROW_CACHE_SIZE>
__global__ void SymmetrizePacked(unsigned *adj, const int N, const size_t bitblocks)
{
	unsigned row = threadIdx.x;
	unsigned col_start = blockIdx.x * bitblocks;
	unsigned replica_offset = col_start * N;
	unsigned lane = row & 31;

	__shared__ unsigned s_row[ROW_CACHE_SIZE];
	for (unsigned i = 0; i < N - 1; i++) {
		if (i == row)
			for (unsigned j = 0; j < bitblocks; j++)
				s_row[j] = adj[replica_offset+j*N+i];
		__syncthreads();

		unsigned val = (s_row[row>>5] >> lane) & 1;	//Row 'i' column 'tid'
		atomicOr(&adj[replica_offset+(i>>5)+row], val << (i & 31));
		__syncthreads();
	}
}

//Iter refers to the first element studied, and its corresponding
//row gets loaded into shared memory
//The row (thread id) refers to the second element studied
template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaReduction(unsigned *adj, const int N, const int iter)
{
	unsigned row = threadIdx.y;
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	__shared__ unsigned e0[ROW_CACHE_SIZE];
	if (row == iter)
		for (unsigned i = iter + 1; i < N; i++)
			e0[i] = adj[replica_offset+i*N+iter];
	__syncthreads();

	if (row > iter && row < N && e0[row])
		for (unsigned i = iter + 1; i < row; i++)
			if (e0[i] && adj[replica_offset+row*N+i])
				adj[replica_offset+row*N+iter] = 0;
}

//'iter' refers to the first element (row)
//while 'row' (thread id) refers to the third element (also a row)
//Then we loop over the pair of columns in each thread.
//It is hard-coded that this must be called with 128 threads per block
template<size_t ROW_CACHE_SIZE>
__global__ void ReplicaActionPair(unsigned *adj, double *action, double *averages, const int N, const double A, const int iter)
{
	unsigned row = threadIdx.y;	
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	__shared__ unsigned e0[ROW_CACHE_SIZE];
	if (row == iter)
		for (unsigned i = iter; i < N; i++)
			e0[i] = adj[replica_offset+i*N+iter];	//First relation(s)
	__syncthreads();

	unsigned sum = 0;
	if (row >= iter && row < N)
		for (unsigned i = iter + 1; i < N; i++)	//Second element (column)
			for (unsigned j = i + 1; j < N; j++)	//Fourth element (column)
				sum += e0[i] ^ adj[replica_offset+j*N+row];

	//Shuffle reduction (sum for each block stored in thread 0)
	for (unsigned offset = 16; offset > 0; offset >>= 1)
		sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, 32);
	if (!(row & 31))
		e0[row] = sum;
	__syncthreads();

	if (!row) {
		double local_action = A * (e0[0] + e0[1] + e0[2] + e0[3]);
		action[blockIdx.x] += local_action;
		atomicAdd(averages, local_action / gridDim.x);
	}
}

__global__ void TauKernel(double *action, unsigned *offspring, unsigned *partial_sums, unsigned R0, unsigned R, double lnQ, double dB, unsigned *Rnew, uint64_t rng_seed, uint64_t initial_sequence)
{
	unsigned tid = threadIdx.x;
	unsigned bid = blockIdx.x;
	unsigned idx = bid * blockDim.x + tid;

	RNGState localrng;
	curand_init(rng_seed, initial_sequence + idx, 0, &localrng);

	//Calculate number of offspring for each replica
	unsigned parS;
	if (idx < R) {
		double mu = exp(-dB * action[idx] - lnQ) * R0 / R;
		double mufloor = floor(mu);
		if (curand_uniform_double(&localrng) < (mu - mufloor))
			parS = offspring[idx] = mufloor + 1;
		else
			parS = offspring[idx] = mufloor;
	} else
		parS = 0;

	//Reduce this for the total
	for (unsigned offset = 16; offset > 0; offset >>= 1)
		parS += __shfl_down_sync(0xFFFFFFFF, parS, offset, 32);

	__shared__ unsigned s_parS[32];
	unsigned lane = tid & 31;
	unsigned wid = tid >> 5;
	if (!lane)
		s_parS[wid] = parS;
	__syncthreads();

	for (unsigned stride = 1; stride < blockDim.x >> 5; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)))
			s_parS[wid] += s_parS[wid+stride];
		__syncthreads();
	}

	if (!tid) {
		parS = s_parS[0];
		partial_sums[idx] = parS;
		atomicAdd(Rnew, parS);
	}
}

__global__ void ResampleReplicas(unsigned *adj_new, unsigned *adj, unsigned *link_new, unsigned *link, double *action_new, double *action, unsigned *offspring, unsigned *partial_sums, unsigned R, unsigned N)
{
	unsigned row = threadIdx.y;	
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	unsigned j = partial_sums[R+blockIdx.x];
	unsigned jnext = j + offspring[blockIdx.x];
	for (; j < jnext; j++) {
		for (unsigned i = 0; i < N && row < N; i++) {
			adj_new[(i+j*N)*N+row] = adj[replica_offset+i*N+row];
			link_new[(i+j*N)*N+row] = link[replica_offset+i*N+row];
		}
		if (!row)
			action_new[j] = action[blockIdx.x];
	}
}

template<size_t ROW_CACHE_SIZE>
__global__ void RelaxReplicas(unsigned *adj, unsigned *link, double *action, double *averages, unsigned sweeps, uint64_t rng_seed, uint64_t initial_sequence, unsigned N, double beta, double A)
{
	unsigned row = threadIdx.y;	
	unsigned col_start = blockIdx.x * N;
	unsigned replica_offset = col_start * N;

	unsigned lane = row & 31;
	unsigned wid = row >> 5;

	RNGState localrng;
	curand_init(rng_seed, initial_sequence+col_start+row, 0, &localrng);

	//Each thread pulls a row of 'adj' into register memory
	unsigned adj_row[ROW_CACHE_SIZE];
	memset(adj_row, 0, sizeof(unsigned) * ROW_CACHE_SIZE);
	//Used for reduction/closure
	__shared__ unsigned shr_row[ROW_CACHE_SIZE];

	unsigned steps = N * (N - 1) >> 1;
	__shared__ unsigned s_cnt[4];
	__shared__ unsigned flip_id;
	__shared__ double current_action;
	__shared__ bool accept;
	unsigned cnt, localcnt, col, rowcnt, start;
	double new_action, dS;

	if (!row)
		current_action = action[blockIdx.x];
	__syncthreads();

	for (unsigned i = row; i < N; i++)
		adj_row[i] = adj[replica_offset+i*N+row];

	for (unsigned sweep = 0; sweep < sweeps; sweep++) {
	//for (unsigned sweep = 0; sweep < 1; sweep++) {
		for (unsigned step = 0; step < steps; step++) {
		//for (unsigned step = 0; step < 1; step++) {
			//Pick a random element of 'adj' such that it
			//is not a transitive relation
			cnt = 0;
			for (unsigned i = row; i < N; i++) {
				adj_row[i] &= adj_row[i] ^ link[replica_offset+i*N+row];
				adj_row[i] ^= 1;
				if (i == row)
					adj_row[i] = 0;
				cnt += adj_row[i];
			}
			localcnt = cnt;

			//Prefix Sum
			for (unsigned i = 1; i <= 32; i <<= 1) {
				unsigned shfl_cnt = __shfl_up_sync(0xFFFFFFFF, cnt, i, 32);
				cnt += lane >= i ? shfl_cnt : 0.0;
			}
			if (lane == 31)
				s_cnt[wid] = cnt;
			__syncthreads();
			for (unsigned i = 0; i < wid; i++)
				cnt += s_cnt[i];
			if (row == blockDim.y - 1)
				flip_id = curand_uniform_double(&localrng) * cnt;
			__syncthreads();

			col = N;
			if (flip_id >= cnt - localcnt && flip_id < cnt) {
				start = cnt - localcnt;
				rowcnt = 0, col = 0;
				for (col = 0; col < N && rowcnt + start < flip_id; col++)
					if (adj_row[col])
						rowcnt++;
			}

			//Re-load adj
			for (unsigned i = row; i < N; i++)
				adj_row[i] = adj[replica_offset+i*N+row];

			if (col != N)
				adj_row[col] ^= 1;	//This is the flip
			__syncthreads();

			//Forgot to include transitive reduction here
			assert (false);

			//Transitive closure
			for (unsigned i = 0; i < N; i++) {
				if (row == i)
					for (unsigned j = 0; j < N; j++)
						shr_row[j] = adj_row[j];
				__syncthreads();
				for (unsigned j = 0; j < N && row < N; j++)
					if (shr_row[j] & adj_row[i])
						adj_row[j] = 1;
				__syncthreads();
			}

			//Calculate new action
			cnt = 0;
			for (unsigned i = 0; i < N - 1; i++) {
				localcnt = 0;
				if (row == i)
					for (unsigned j = 0; j < N; j++)
						shr_row[j] = adj_row[j];
				__syncthreads();
				if (row > i && row < N)
					for (unsigned j = i + 1; j < N; j++)
						for (unsigned k = j + 1; k < N; k++)
							localcnt += shr_row[j] ^ adj_row[k];
				for (unsigned j = 16; j > 0; j >>= 1)
					localcnt += __shfl_down_sync(0xFFFFFFFF, localcnt, j, 32);
				if (!lane)
					s_cnt[wid] = localcnt;
				__syncthreads();
				if (!row)
					cnt += s_cnt[0] + s_cnt[1] + s_cnt[2] + s_cnt[3];
			}

			//Metropolis step
			if (!row) {
				new_action = A * cnt;
				//printf("Replica [%d] count: %u\n", blockIdx.x, cnt);
				dS = beta * (new_action - current_action);
				accept = dS <= 0 || exp(-dS) > curand_uniform_double(&localrng);
				if (accept)
					current_action = new_action;
			}
			__syncthreads();

			if (accept) {
				//Write adj to global memory
				for (unsigned i = row; i < N; i++)
					adj[replica_offset+i*N+row] = adj_row[i];
				__syncthreads();

				//Transitive reduction, adj -> link
				for (unsigned i = 0; i < N - 1; i++) {
					if (row == i)
						for (unsigned j = i + 1; j < N; j++)
							shr_row[j] = adj_row[j];
					__syncthreads();
					for (unsigned j = i + 1; j < row && row < N; j++)
						if (shr_row[row] && shr_row[j] && adj[replica_offset+row*N+j])
							link[replica_offset+row*N+i] = 0;
					__syncthreads();
				}
			} else {
				//Simply re-load adj from global memory
				for (unsigned i = row; i < N; i++)
					adj_row[i] = adj[replica_offset+i*N+row];
			}
			__syncthreads(); //Remove after debugging
		}
	}

	if (!row) {
		action[blockIdx.x] = current_action;
		atomicAdd(averages, current_action / gridDim.x);
	}
}

bool annealNetwork(Network * const network, CuResources * const cu, CaResources * const ca)
{
	const int rank = network->network_properties.cmpi.rank;
	const int N = network->network_properties.N;

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
	/*printf("Inside annealing function (Rank %d)\n", network->network_properties.cmpi.rank);
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);*/

	printf_mpi(rank, "Executing Population Annealing [v1]...\n");
	if (!rank) fflush(stdout);

	//Temperature data
	double beta_init = 0.0;		//Initial beta
	double beta_final = 1.0;	//Final beta
	double dB_init = 0.005;		//Initial beta step
	const int nBmax = 10000;	//Max beta steps
	double B[nBmax]/*, Binc[nBmax]*/;	//Stored beta values, step sizes
	B[0] /*= Binc[0]*/ = beta_init;

	//Coupling for action
	double A = 1.0;
	unsigned steps_per_sweep = N * (N - 1) >> 1;
	uint64_t num_rel_pairs = ((uint64_t)N * (N - 1) * (((uint64_t)N * (N - 1) >> 1) - 1)) >> 2;
	A /= num_rel_pairs;

	//Overlap data
	//double min_overlap = 0.85;	//Minimum action overlap
	//double max_overlap = 0.87;	//Maximum action overlap

	//Population annealing parameters
	unsigned R0 = 256;		//Target population size
	unsigned theta = 1;		//Equilibration (Metropolis) steps
	unsigned runs = 1; 		//Independent runs
	uint64_t initial_sequence = 0;	//Used to seed RNG in kernel
	uint64_t tot_pop = 0;		//Sum of population sizes
	uint64_t rng_seed = network->network_properties.mrng.rng() * ULONG_MAX;
	//uint64_t rng_seed = 14911893754678870016ULL;

	//Observables
	//double action[nBmax];		//Action at each beta
	int R[nBmax];			//Population size at each beta
	R[0] = R0;

	//CUDA arch parameters
	dim3 threads_per_block(1, 128, 1);	//Used for replica updates
	dim3 blocks_per_grid(R[0], 1, 1);
	dim3 threads_per_block2(1024, 1, 1);	//Used for observables
	dim3 blocks_per_grid2((int)ceil((float)R[0] / 1024), 1, 1);
	static const size_t cache_size = 128;	//Each row uses X unsigned integers (N allowed up to N=X)

	unsigned gpu_id = rand() % cu->dev_count;
	cuCtxPushCurrent(cu->cuContext[gpu_id]);
	cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
	printf_mpi(rank, "Using GPU [%u]\n", gpu_id);

	//CUDA event timers
	CUevent start, stop;
	float elapsed_time;
	checkCudaErrors(cuEventCreate(&start, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventCreate(&stop, CU_EVENT_DEFAULT));
	checkCudaErrors(cuEventRecord(start, 0));

	//Helper variables
	CUdeviceptr d_averages = CUDA_MALLOC<double>(1);//For now just store action
	CUdeviceptr d_Q = CUDA_MALLOC<double>(1);	//Partition function ratio
	CUdeviceptr d_Ri = CUDA_MALLOC<int>(1);		//Population size
	CUdeviceptr d_overlap = CUDA_MALLOC<double>(1);	//Energy histogram overlap

	printf_mpi(rank, "\n\tSimulation Parameters:\n");
	printf_mpi(rank, "\t----------------------\n");
	if (!rank) printf_cyan();
	printf_mpi(rank, "\tCauset Elements:   [%u]\n", N);
	printf_mpi(rank, "\tTarget Population: [%u]\n", R[0]);
	printf_mpi(rank, "\tRelaxation Sweeps: [%u]\n", theta);
	printf_mpi(rank, "\tBeta Range:        [%.3f]-[%.3f]\n", beta_init, beta_final);
	printf_mpi(rank, "\tMax. Beta Steps:   [%u]\n", nBmax);
	printf_mpi(rank, "\tIndependent Runs:  [%u]\n", runs);
	size_t cr_size = 0;	//Memory used to simulate replicas
	cr_size += 4ULL * N * N;	//Adjacency matrix
	cr_size += 4ULL * N * N;	//Link matrix
	cr_size += 4ULL * N * N;	//New adjacency matrix
	cr_size += 4ULL * N * N;	//New link matrix
	cr_size += 16;	//Action, offspring, and partial update (per replica)
	printf_mpi(rank, "\tMemory/Replica:    [%.2f kB]\n", (float)cr_size / 1024.0);
	printf_mpi(rank, "\tPopulation Memory: [%.2f MB]\n", (float)R[0] * cr_size / 1048576.0);
	printf_mpi(rank, "\tRandom Seed:       [%" PRIu64 "]\n", rng_seed);
	if (!rank) {
		printf_std();
		printf("\n");
		fflush(stdout);
	}

	//N rows, NxR[0] columns, i.e., R[0] NxN matrices
	//stacked horizontally and indexed in column-major order
	unsigned *adj = MALLOC<unsigned,true>(N * N * R[0] * 100);
	//One value per replica, updated each temperature
	double *action = MALLOC<double,true>(100*R[0]);
	unsigned *offspring = MALLOC<unsigned>(100*R[0]);
	unsigned *partial_sums = MALLOC<unsigned,true>(100 * 2 * R[0]);
	//One value per temperature
	double *lnQ = MALLOC<double,true>(nBmax);

	CausetReplicas d_reps, d_reps_new;
	double mean_action, q;
	for (unsigned i = 0; i < runs; i++) {
		mean_action = 0.0;

		//Allocate replicas
		d_reps.adj = CUDA_MALLOC<unsigned>(N * N * R[0]);
		d_reps.link = CUDA_MALLOC<unsigned>(N * N * R[0]);
		d_reps.action = CUDA_MALLOC<double>(R[0]);
		d_reps.offspring = CUDA_MALLOC<unsigned>(R[0]);
		d_reps.partial_sums = CUDA_MALLOC<unsigned>(2 * R[0]);

		//Initialize replicas with random data
		ReplicaInit<<<blocks_per_grid, threads_per_block>>>((unsigned int*)d_reps.adj, rng_seed, initial_sequence, N);
		initial_sequence += R[0] * threads_per_block.y;
		checkCudaErrors(cuCtxSynchronize());
		/*checkCudaErrors(cuMemcpyDtoH(adj, d_reps.adj, sizeof(unsigned) * N * N * R[0]));
		printf("\nInitial random data:\n");
		for (unsigned j = 0; j < N; j++) {	//Rows
			for (unsigned k = 0; k < N * R[0]; k++)	//Columns
				printf("%u", adj[k*N+j]);
			printf("\n");
		}
		fflush(stdout);*/

		//Calculate initial energy
		//First perform a transitive closure
		//printf("\nAfter transitive closure:\n");
		for (unsigned j = 0; j < N; j++)
			ReplicaClosure<cache_size><<<blocks_per_grid, threads_per_block>>>((unsigned*)d_reps.adj, N, j);
		checkCudaErrors(cuCtxSynchronize());
		/*checkCudaErrors(cuMemcpyDtoH(adj, d_reps.adj, sizeof(unsigned) * N * N * R[0]));
		for (unsigned j = 0; j < N; j++) {		//Rows
			adj[j*N+j] = 0;
			for (unsigned k = 0; k < N * R[0]; k++)	//Columns
				printf("%u", adj[k*N+j]);
			printf("\n");
		}
		fflush(stdout);*/

		/*Bitvector h_adj;
		FastBitset fb(N);
		for (unsigned j = 0; j < N; j++) {
			h_adj.push_back(fb);
			for (unsigned k = 0; k < N; k++)
				if (adj[k*N+j] && j != k)
					h_adj[j].set(k);
		}*/
		//Symmetrize
		/*for (unsigned j = 1; j < N; j++)		//Rows
			for (unsigned k = 0; k < j; k++)	//Columns
				if(adj[j*N+k])
					h_adj[j].set(k);*/
		/*printf("\nBitset representation:\n");
		for (unsigned j = 0; j < N; j++)
			h_adj[j].printBitset();*/

		//printf("\nAfter transitive reduction:\n");
		cuMemcpyDtoD(d_reps.link, d_reps.adj, sizeof(unsigned) * N * N * R[0]);
		for (unsigned j = 0; j < N - 1; j++)
			ReplicaReduction<cache_size><<<blocks_per_grid, threads_per_block>>>((unsigned*)d_reps.link, N, j);
		checkCudaErrors(cuCtxSynchronize());
		//checkCudaErrors(cuMemcpyDtoH(adj, d_reps.adj, sizeof(unsigned) * N * N * R[0]));
		/*for (unsigned j = 0; j < N; j++) {		//Rows
			adj[j*N+j] = 0;
			for (unsigned k = 0; k < N * R[0]; k++)	//Columns
				printf("%u", adj[k*N+j]);
			printf("\n");
		}
		fflush(stdout);*/

		/*printf("\nShould be:\n");
		Bitvector h_adj2;
		for (unsigned j = 0; j < N; j++)
			h_adj2.push_back(fb);
		transitiveReduction(h_adj2, h_adj, N);
		for (unsigned j = 0; j < N; j++)
			h_adj2[j].printBitset();*/

		//printf("\nAfter action calculation [PAIR]\n");
		checkCudaErrors(cuMemsetD32(d_averages, 0, 2));
		for (unsigned j = 0; j < N - 1; j++)
			ReplicaActionPair<cache_size><<<blocks_per_grid, threads_per_block>>>((unsigned*)d_reps.adj, (double*)d_reps.action, (double*)d_averages, N, A, j);
		checkCudaErrors(cuCtxSynchronize());
		checkCudaErrors(cuMemcpyDtoH(&mean_action, d_averages, sizeof(double)));
		/*checkCudaErrors(cuMemcpyDtoH(action, d_reps.action, sizeof(double) * R[0]));
		for (unsigned j = 0; j < 10 && j < R[0]; j++)
			printf("Replica [%d] has action: %.5f\n", j, action[j]);
		printf("\n");
		fflush(stdout);*/

		/*printf_cyan();
		printf("Should find:\n");
		for (unsigned nreps = 0; nreps < 10; nreps++) {
			uint64_t cnt = 0;
			uint64_t rep_offset = nreps * N * N;
			for (unsigned j = 0; j < N - 1; j++)
				for (unsigned m = j; m < N; m++)
					for (unsigned k = j + 1; k < N; k++)
						for (unsigned n = k + 1; n < N; n++)
							cnt += adj[rep_offset+k*N+j] ^ adj[rep_offset+n*N+m];
			printf("%" PRIu64 "\n", cnt);
		}
		printf_std();
		fflush(stdout);*/

		unsigned b = 1, bprev = 0/*, nb*/;	//Indexing over beta values in B[]
						//We use an adaptive step size, up to nb=nBmax steps
		double dB = dB_init;
		B[b] /*= Binc[b]*/ = B[bprev] + dB;

		while (B[b] <= beta_final) {
			//Calculate partition function ratio, Q
			checkCudaErrors(cuMemsetD32(d_Q, 0, 2));
			QKernel<<<blocks_per_grid2, threads_per_block2>>>((double*)d_reps.action, R[b-1], B[b] - B[b-1], mean_action, (double*)d_Q);
			checkCudaErrors(cuCtxSynchronize());

			checkCudaErrors(cuMemcpyDtoH(&q, d_Q, sizeof(double)));
			lnQ[b] = -(B[b] - B[b-1]) * mean_action + log(q) - log((double)R[b-1]);
			//printf("Q value: %f\n", q);
			//printf("lnQ: %f\n", lnQ[b]);

			checkCudaErrors(cuMemsetD32(d_Ri, 0, 1));
			TauKernel<<<blocks_per_grid2, threads_per_block2>>>((double*)d_reps.action, (unsigned*)d_reps.offspring, (unsigned*)d_reps.partial_sums, R[0], R[b-1], lnQ[b], B[b] - B[b-1], (unsigned*)d_Ri, rng_seed, initial_sequence);
			initial_sequence += R[b-1];
			checkCudaErrors(cuCtxSynchronize());
			checkCudaErrors(cuMemcpyDtoH(&R[b], d_Ri, sizeof(unsigned)));
			printf("Number of offspring: %u\n", R[b]);
			printf("New Population Memory: [%.2f MB]\n", (float)R[b] * cr_size / 1048576.0);

			PartialSum<<<blocks_per_grid2, threads_per_block2>>>((unsigned*)d_reps.offspring, (unsigned*)d_reps.partial_sums, R[b-1]);
			checkCudaErrors(cuCtxSynchronize());
			/*checkCudaErrors(cuMemcpyDtoH(offspring, d_reps.offspring, sizeof(unsigned) * R[b-1]));
			checkCudaErrors(cuMemcpyDtoH(partial_sums, d_reps.partial_sums, sizeof(unsigned) * 2 * R[b-1]));
			printf("Offpsring / Partial Sums:\n");
			for (unsigned j = 0; j < R[b-1]; j++)
				printf("%u  %u\n", offspring[j], partial_sums[j+R[b-1]]);
			printf("\n");*/

			d_reps_new.adj = CUDA_MALLOC<unsigned>(N * N * R[b]);
			d_reps_new.link = CUDA_MALLOC<unsigned>(N * N * R[b]);
			d_reps_new.action = CUDA_MALLOC<double>(R[b]);
			d_reps_new.offspring = CUDA_MALLOC<unsigned>(R[b]);
			d_reps_new.partial_sums = CUDA_MALLOC<unsigned>(2 * R[b]);

			ResampleReplicas<<<blocks_per_grid, threads_per_block>>>((unsigned*)d_reps_new.adj, (unsigned*)d_reps.adj, (unsigned*)d_reps_new.link, (unsigned*)d_reps.link, (double*)d_reps_new.action, (double*)d_reps.action, (unsigned*)d_reps.offspring, (unsigned*)d_reps.partial_sums, R[b-1], N);
			checkCudaErrors(cuCtxSynchronize());
			/*checkCudaErrors(cuMemcpyDtoH(action, d_reps_new.action, sizeof(double) * R[b]));
			for (unsigned j = 0; j < R[b]; j++)
				printf("Replica [%d] has action: %.5f\n", j, action[j]);
			printf("\n"); fflush(stdout);*/

			CUDA_FREE(d_reps.adj);
			CUDA_FREE(d_reps.link);
			CUDA_FREE(d_reps.action);
			CUDA_FREE(d_reps.offspring);
			CUDA_FREE(d_reps.partial_sums);
			d_reps = d_reps_new;

			blocks_per_grid.x = R[b];
			RelaxReplicas<cache_size><<<blocks_per_grid, threads_per_block>>>((unsigned*)d_reps.adj, (unsigned*)d_reps.link, (double*)d_reps.action, (double*)d_averages, theta, rng_seed, initial_sequence, N, B[b], A);
			initial_sequence += R[b] * threads_per_block.y;
			checkCudaErrors(cuCtxSynchronize());
			checkCudaErrors(cuMemcpyDtoH(&mean_action, d_averages, sizeof(double)));

			bprev = b;
			tot_pop += R[b];
			b++;

			if (b >= nBmax) {
				fprintf(stderr, "Error: number of temperature steps exceeded nBmax=%d.\nIncrease the population size, decrease the value of min_overlap, or increase nBmax.\n", nBmax);
				return false;
			}

			B[b] = B[bprev] + dB;

		}

		CUDA_FREE(d_reps.adj);
		CUDA_FREE(d_reps.link);
		CUDA_FREE(d_reps.action);
		CUDA_FREE(d_reps.offspring);
		CUDA_FREE(d_reps.partial_sums);
	}

	FREE(adj);
	FREE(action);
	FREE(offspring);
	FREE(partial_sums);
	FREE(lnQ);

	CUDA_FREE(d_averages);
	CUDA_FREE(d_Q);
	CUDA_FREE(d_Ri);
	CUDA_FREE(d_overlap);

	checkCudaErrors(cuCtxSynchronize());
	checkCudaErrors(cuEventRecord(stop, 0));
	checkCudaErrors(cuEventSynchronize(stop));
	checkCudaErrors(cuEventElapsedTime(&elapsed_time, start, stop));

	#ifdef MPI_ENABLED
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
	printf_mpi(network->network_properties.cmpi.rank, "\nElapsed Time: %.8f sec\n", elapsed_time / 1000);
	printf_mpi(network->network_properties.cmpi.rank, " > %.5e sec/step\n", elapsed_time / (1000000 * theta * steps_per_sweep * tot_pop));

	checkCudaErrors(cuEventDestroy(start));
	checkCudaErrors(cuEventDestroy(stop));
	cuCtxPopCurrent(&cu->cuContext[gpu_id]);

	return false;
}

template<typename T>
void gather_data(T *data, CUdeviceptr d_data, int * const sendcnt, int * const displs, unsigned Rloc, int nprocs, int rank)
{
	memset(displs, 0, sizeof(int) * nprocs);
	memset(sendcnt, 0, sizeof(int) * nprocs);
	memset(data, 0, sizeof(T) * Rloc);

	#ifdef MPI_ENABLED
	MPI_Allgather(&Rloc, 1, MPI_INT, sendcnt, 1, MPI_INT, MPI_COMM_WORLD);
	#endif
	for (unsigned i = 1; i < nprocs; i++)
		displs[i] = displs[i-1] + sendcnt[i-1];
	checkCudaErrors(cuMemcpyDtoH(data + displs[rank], d_data, sizeof(T) * Rloc));
	checkCudaErrors(cuCtxSynchronize());
	#ifdef MPI_ENABLED
	MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data, sendcnt, displs, getMPIDataType<T>(), MPI_COMM_WORLD);
	#endif
}

template<typename T>
void annealH5Write(T *data, hid_t file, hid_t dset, hid_t dspace, hsize_t *rowdim, hsize_t *rowoffset, unsigned num_values)
{
	rowdim[0] = num_values;
	rowdim[1] = 1;

	hid_t mspace = H5Screate_simple(2, rowdim, NULL);
	H5Sselect_hyperslab(dspace, H5S_SELECT_SET, rowoffset, NULL, rowdim, NULL);
	H5Dwrite(dset, getH5DataType<T>(), mspace, dspace, H5P_DEFAULT, data);
	H5Fflush(file, H5F_SCOPE_LOCAL);
	H5Sclose(mspace);
}
