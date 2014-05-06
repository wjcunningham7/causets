#ifndef AUTOCORR2_CUH_
#define AUTOCORR2_CUH_

#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>

#include <autocorr2.h>

static void avglessi(int n, double x[], double xai[])
{
	int i1, i2;

	if (n <= 1)
		fprintf(stderr, "bad n avglessi\n");

	for (i1 = 0; i1 < n; i1++) {
		if (n > 10000)
			if (!(i1 % 10000)) 
				printf("avglessi step %d\n", i1);

		xai[i1] = 0.0;
		for (i2 = 0; i2 < n; i2++)
			if (i2 != i1)
				xai[i1] += x[i2];

		xai[i1] /= (double)(n - 1);
	}
}
  
static double get_mean(int n, double x[])
{
	if (n <= 0)
		fprintf(stderr, "bad n get_mean");

	int i;
	double dum = 0.0;

	for (i = 0; i < n; i++)
		dum += x[i];

	dum /= (double)n;

	return dum;
}

static double jackknife(int n, double xdot, double xai[])
{
	int i;
	double dumt, dums = 0.0;

	for (i = 0; i < n; i++) {
		dumt = xai[i] - xdot;
		dums += dumt * dumt;
	}

	dums *= ((double)(n - 1)) / ((double)n);
	dums = sqrt(dums);
	
	return dums;
}

autocorr2::autocorr2()
{
	init(1);
}

autocorr2::autocorr2(int blksize)
{
	init(blksize);
}

autocorr2::~autocorr2()
{
	if (analyzed_flag)
		delete_arrays();
}

void autocorr2::init(int blksize)
{
	jackknife_block_size = blksize;
	havedata_flag = 0;
	analyzed_flag = 0;
}

void autocorr2::accum_data(double x)
{
	data.push_back(x);	
	havedata_flag = 1;
}

void autocorr2::analysis()
{
	if (analyzed_flag)
		fprintf(stderr, "cannot analyze twice");
	if (!havedata_flag)
		fprintf(stderr, "you never gave me data");
	nsample=data.size();
	jackknife_blocks = nsample / jackknife_block_size;

	//Set t_max such that 5 blocks can be removed,
	//so that the error estimate will mean something.
	t_max = nsample - 5 * jackknife_block_size - 1;
	if (t_max <= 0)
		fprintf(stderr, "block size too large in autocorr2::analysis");
	setup_arrays();
	for (int blk = 0; blk < jackknife_blocks; blk++)
		analysis(blk);

	//This call fills in ac[] and ac_err[].
	do_jackknife();
	analyzed_flag = 1;
}

void autocorr2::setup_arrays()
{
	int blk;
  
	avg = new double[jackknife_blocks];
	x = new double[jackknife_blocks];
	xai = new double[jackknife_blocks];
  
	corr = new double*[jackknife_blocks];
	count = new int*[jackknife_blocks];
  
	for (blk = 0; blk < jackknife_blocks; blk++) {
		corr[blk] = new double[t_max+1];
		count[blk] = new int[t_max+1];
	}

	ac = new double[t_max+1];
	ac_err = new double[t_max+1];
}

void autocorr2::delete_arrays()
{
	int blk;
  
	delete [] avg;
	delete [] x;
	delete [] xai;
  
	for (blk = 0; blk < jackknife_blocks; blk++) {
		delete [] corr[blk];
		delete [] count[blk];
	}

	delete [] corr;
	delete [] count;

	delete [] ac;
	delete [] ac_err;
}

void autocorr2::analysis(int blk)
{
	avg[blk] = get_avg(blk);
	corr[blk][0] = get_corr0(blk);
	for (int t = 1; t <= t_max; t++)
		corr[blk][t] = get_corrhat(blk, t);
	corr[blk][0] = 1.0;
}

//Average over all data except the data contained in block blk.
double autocorr2::get_avg(int blk)
{
	if ((nsample - jackknife_block_size) <= 0)
		fprintf(stderr, "block size too big in autocorr2::get_avg");

	double avg_tmp = 0.0;
	int i_skip_min = blk * jackknife_block_size;
	int i_skip_max = (blk + 1) * jackknife_block_size - 1;

	//Note that i_skip_max may be larger than nsample - 1.
	int icount = 0;
	for (int i = 0; i < nsample; i++) {
		if (i >= i_skip_min && i <= i_skip_max)
			continue;
    		avg_tmp += data[i];
		icount++;
	}

	//icount is guaranteed to be nonzero.
	avg_tmp /= 1.0 * icount;
	return avg_tmp;
}

double autocorr2::get_corr0(int blk)
{
	if ((nsample - jackknife_block_size) <= 0)
		fprintf(stderr, "block size too big in autocorr2::get_corr0");

	double corr0_tmp = 0.0;
	int i_skip_min = blk * jackknife_block_size;
	int i_skip_max = (blk + 1) * jackknife_block_size - 1;
	int icount = 0;

	for (int i = 0; i < nsample; i++) {
		if (i >= i_skip_min && i <= i_skip_max)
			continue;
		corr0_tmp += (data[i] - avg[blk]) * (data[i] - avg[blk]);
		icount++;
	}

	//icount is guaranteed to be nonzero.
	corr0_tmp /= 1.0 * icount;
	return corr0_tmp;
}

double autocorr2::get_corrhat(int blk, int t)
{
	double corr_tmp = 0.0;
	int i_skip_min = blk * jackknife_block_size;
	int i_skip_max = (blk + 1) * jackknife_block_size - 1;
	int icount = 0;
	int i;

	for (i = 0; i < i_skip_min - t; i++) {
		corr_tmp += (data[i+t] - avg[blk]) * (data[i] - avg[blk]);
		icount++;
	}

	for (i = i_skip_max + 1; i < nsample - t; i++) {
		corr_tmp += (data[i+t] - avg[blk]) * (data[i] - avg[blk]);
		icount++;
	}

	if (icount)
		corr_tmp /= corr[blk][0] * icount;

	count[blk][t] = icount;
	return corr_tmp;
}

void autocorr2::do_jackknife()
{
	double x_err, xdot;
	int icount, blk;

	ac[0] = 1.0;
	ac_err[0] = 0.0;

	for (int t = 1; t <= t_max; t++) {
		icount=0;
		for (blk = 0; blk < jackknife_blocks; blk++) {
			if (count[blk][t]) {
				x[icount] = corr[blk][t];
				icount++;
			}
		}

		avglessi(icount, x, xai);
		xdot = get_mean(icount, x);
		x_err = jackknife(icount, xdot, xai);
		ac[t] = xdot;
		ac_err[t] = x_err;
	}
}

void autocorr2::fout_txt(std::ofstream &f)
{
	int t;
	f.precision(18);

	for (t = 0; t <= t_max; t++)
			f << t << ' ' << ac[t] << ' ' << ac_err[t] << std::endl;
}

#endif
