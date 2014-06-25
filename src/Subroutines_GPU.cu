#include "Subroutines_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Input:  Index from vectorized upper diagonal matrix
//Output: i*N+j where i,j are matrix indices
//Efficiency: O(N)
__host__ __device__ uint64_t vec2MatIdx(const int &N, const uint64_t &vecIdx)
{
	int i = 0, j = 0;
	int delta = 1;
	int k;
	
	for (k = 0; k < N - 1; k++) {
		if (vecIdx < (k + 1) * static_cast<uint64_t>(N) - delta) {
			i = k;
			j = static_cast<int>(vecIdx - (k * static_cast<uint64_t>(N)) + delta);
			break;
		}
		delta += k + 2;
	}

	return i * static_cast<uint64_t>(N) + j;
}
