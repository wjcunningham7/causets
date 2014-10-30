#ifndef AUTOCORR2_H_
#define AUTOCORR2_H_

#include <fstream>
#include <vector>

class autocorr2
{
	private:
		std::vector<double> data;

		double **corr;
		double *avg;
		int **count;
		double *ac;
		double *ac_err;
		double *x;
		double *xai;

		int jackknife_block_size;
		int nsample;
		int t_max;
		int jackknife_blocks;

		int havedata_flag;
		int analyzed_flag;

	public:
		autocorr2();
		autocorr2(int blksize);
		~autocorr2();

		void accum_data(double x);
		void analysis();
		void fout_txt(std::ofstream &f);

	private:
		void init(int blksize);
		void setup_arrays();
		void delete_arrays();
		void analysis(int blk);
		double get_avg(int blk);
		double get_corr0(int blk);
		double get_corrhat(int blk, int t);
		void do_jackknife();
};

void avglessi(int n, double x[], double xai[]);
double get_mean(int n, double x[]);
double jackknife(int n, double xdot, double xai[]);

#endif
