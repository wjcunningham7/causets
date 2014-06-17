#include "Subroutines_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Input:  Index from vectorized upper diagonal matrix
//Output: j*N+i where i,j are matrix indices
__host__ __device__ int vec2MatIdx(const int &N, const int &vecIdx)
{
	int i = 0, j = 0;
	int k;
	
	if (vecIdx < N - 1) {
		//First row in matrix
		j = 0;
		i = vecIdx + 1;
	} else if (vecIdx == N * (N - 1) / 2 - 1) {
		//Last element in matrix
		j = N - 2;
		i = N - 1;
	} else {
		for (k = 1; k < N - 2; k++) {
			if (vecIdx < (k + 1) * N - 3 * k) {
				j = k;
				if (k == 1)
					i = vecIdx - (N - 3);
				else
					i = vecIdx - j * (N - 4) - 2;
			}
		}
	}
	
	return j * N + i;
}
