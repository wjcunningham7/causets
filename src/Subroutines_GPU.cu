#include "Subroutines_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Input:  Index from vectorized upper diagonal matrix
//Output: i*N+j where i,j are matrix indices
//Efficiency: O(N)
/*__host__ __device__ uint64_t vec2MatIdx(const int &N, const uint64_t &vecIdx)
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
}*/

//Bitonic Sort
//Borrowed from https://gist.github.com/mre/1392067

__global__ void BitonicSort(uint64_t *edges, int j, int k)
{
        //Sorting Parameters
        unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned ixj = i ^ j;

        //Threads with the lowest IDs sort the list
        if (i < ixj) {
                //Sort Ascending
                if (!(i & k) && edges[i] > edges[ixj])
                        swap(edges, i, ixj);

                //Sort Descending
                if ((i & k) && edges[i] < edges[ixj])
                        swap(edges, i, ixj);
        }   
}

__device__ void swap(uint64_t * const &edges, const unsigned int &i, const unsigned int &j)
{
	uint64_t temp = edges[i];
	edges[i] = edges[j];
	edges[j] = temp;
}

//Parallel Prefix Sum
//Borrowed from https://gist.github.com/wh5a/4500706

__global__ void Scan(int *input, int *output, int *buf, int elements)
{
	__shared__ int s_vals[BLOCK_SIZE << 1];
	unsigned int tid = threadIdx.x;
	unsigned int start = (blockDim.x * blockIdx.x) << 1;

	//Read 'input' to shared memory
	if (start + tid < elements)
		s_vals[tid] = input[start + tid];
	else
		s_vals[tid] = 0;

	if (start + blockDim.x + tid < elements)
		s_vals[blockDim.x + tid] = input[start + blockDim.x + tid];
	else
		s_vals[blockDim.x + tid] = 0;
	__syncthreads();

	//Primary Reduction
	int stride, index;
	for (stride = 1; stride <= blockDim.x; stride <<= 1) {
		index = (stride * (tid + 1) << 1) - 1;
		if (index < blockDim.x << 1)
			s_vals[index] += s_vals[index - stride];
		__syncthreads();
	}

	//Secondary Reduction
	for (stride = blockDim.x >> 1; stride; stride >>= 1) {
		index = (stride * (tid + 1) << 1) - 1;
		if (index + stride < blockDim.x << 1)
			s_vals[index + stride] += s_vals[index];
		__syncthreads();
	}

	if (start + tid < elements)
		output[start + tid] = s_vals[tid];

	if (start + blockDim.x + tid < elements)
		output[start + blockDim.x + tid] = s_vals[blockDim.x + tid];

	if (buf && tid == 0)
		buf[blockIdx.x] = s_vals[(blockDim.x << 1) - 1];
}

__global__ void PostScan(int *input, int *buf, int elements)
{
	unsigned int tid = threadIdx.x;
	unsigned int start = blockDim.x * blockIdx.x << 1;

	if (blockIdx.x) {
		if (start + tid < elements)
			input[start + tid] += buf[blockIdx.x - 1];

		if (start + blockDim.x + tid < elements)
			input[start + blockDim.x + tid] += buf[blockIdx.x - 1];
	}
}


















