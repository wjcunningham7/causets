/////////////////////////////
//(C) Will Cunningham 2020 //
//    Perimeter Institute  //
/////////////////////////////

#include "Causet.h"

#ifndef ANNEAL_KERNELS_H

#if ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
__device__ inline double atomicAdd(double *address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

__device__ inline void swap(unsigned *data, unsigned i, unsigned j)
{
	unsigned temp = data[i];
	data[i] = data[j];
	data[j] = temp;
}

//Topologically sort a 2D order
__device__ inline void BitonicShared(unsigned *U, unsigned *V, const int N, unsigned tid)
{
	for (unsigned k = 2; k <= N; k <<= 1) {
		for (unsigned j = k >> 1; j > 0; j >>= 1) {
			unsigned ixj = tid ^ j;
			if (ixj > tid && tid < N) {
				if (!(tid & k)) {
					if (U[tid] + V[tid] > U[ixj] + V[ixj]) {
						swap(U, tid, ixj);
						swap(V, tid, ixj);
					}
				} else {
					if (U[tid] + V[tid] < U[ixj] + V[ixj]) {
						swap(U, tid, ixj);
						swap(V, tid, ixj);
					}
				}
			}
			__syncthreads();
		}
	}
}

//Implements OrderMove
__device__ inline void RandomOrderPair(unsigned &order_id, unsigned &flip_id0, unsigned &flip_id1, RNGState *rng, const int N)
{
	//Select one of the two total orders
	order_id = curand_uniform_double(rng) < 0.5;

	//Select a random pair
	flip_id0 = curand_uniform_double(rng) * N;
	flip_id1 = curand_uniform_double(rng) * (N - 1);
	flip_id1 += (flip_id0 == flip_id1);
}

//Implements MatrixMove2
template<size_t ROW_CACHE_SIZE>
__device__ inline void RandomBitAny(unsigned *flip_id, unsigned &flip_col_block, unsigned &flip_col_bit, RNGState *rng, const int N, const unsigned row)
{
	//Pick any random element: (N choose 2) choose 1
	if (!row)
		*flip_id = curand_uniform_double(rng) * ((N * (N - 1)) >> 1);
	__syncthreads();

	unsigned cnt = 0;
	for (unsigned i = 0; i < row; i++)
		cnt += N - i - 1;

	unsigned flip = *flip_id;
	if (flip >= cnt && flip < cnt + N - row) {
		unsigned localcnt = flip - cnt;
		flip_col_block = (localcnt + row + 1) >> 5;
		flip_col_bit = (localcnt + row + 1) & 31;
	}
}

//Implements MatrixMove1
template<size_t ROW_CACHE_SIZE>
__device__ inline void RandomBit(unsigned *flip_id, unsigned &flip_col_block, unsigned &flip_col_bit, unsigned *adj, unsigned *link, RNGState *rng, const int N, const size_t bitblocks, const unsigned row, const unsigned wid, const unsigned lane, const unsigned rowmask, const unsigned cntmask)
{
	__shared__ unsigned s_cnt[ROW_CACHE_SIZE];
	unsigned cnt = 0, localcnt;

	//Pick random element of 'adj' which is not a transitive relation
	for (unsigned i = wid; i < bitblocks; i++) {
		adj[i] = ~(adj[i] & (adj[i] ^ link[i]));
		adj[i] &= (-(i != wid)) | cntmask;
		adj[i] &= (-(i != bitblocks - 1)) | rowmask;
		cnt += __popc(adj[i]);
	}
	localcnt = cnt;

	//Prefix sum
	for (unsigned i = 1; i <= 32; i <<= 1)
		cnt += (lane >= i) * __shfl_up_sync(0xFFFFFFFF, cnt, i, 32);
	if (lane == 31)
		s_cnt[wid] = cnt;
	__syncthreads();
	for (unsigned i = 0; i < wid; i++)
		cnt += s_cnt[i];
	__syncthreads();
	if (row == blockDim.x - 1)
		s_cnt[0] = cnt;
	__syncthreads();
	if (!row)
		*flip_id = curand_uniform_double(rng) * cnt;
	__syncthreads();

	//Identify the (row,col) location corresponding to flip_id
	unsigned flip = *flip_id;
	if (flip >= cnt - localcnt && flip < cnt) {	//This indicates the bit of interest
							//is in the row controlled by thread 'row'
		unsigned rowcnt = cnt - localcnt;
		for (flip_col_block = (row + 1) >> 5; flip_col_block < bitblocks && flip_col_bit == 32; flip_col_block++) {
			adj[flip_col_block] &= (-(flip_col_block != ((row + 1) >> 5))) | cntmask;
			adj[flip_col_block] &= (-(flip_col_block != (bitblocks - 1))) | rowmask;
			unsigned block_cnt = __popc(adj[flip_col_block]);
			rowcnt += (rowcnt + block_cnt < flip) * block_cnt;
			for (unsigned i = 0; i < flip - rowcnt && rowcnt + block_cnt >= flip; i++)
				adj[flip_col_block] &= ~(1u << (flip_col_bit = __ffs(adj[flip_col_block])));
		}
	}
}

template<size_t ROW_CACHE_SIZE>
__device__ inline void Closure(unsigned *adj, const int N, const size_t bitblocks, const unsigned iters, const unsigned row, const unsigned wid, const unsigned lane)
{
	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	unsigned C_ijk, C_ij, active, mask;

	for (unsigned n = 0; n < iters; n++) {
		for (int i = 0; i < N; i++) {	//Rows
			if (row == i)
				for (size_t j = 0; j < bitblocks; j++)
					s_row[j] = adj[j];
			__syncthreads();

			C_ijk = 0, C_ij = 0;
			active = (row > i) & (row < N);
			mask = __ballot_sync(0xFFFFFFFF, active);
			for (unsigned j_block = (i + 1) >> 5; j_block < bitblocks; j_block++) {	//Columns
				unsigned jb = adj[j_block];
				unsigned j_bit = (j_block == ((i + 1) >> 5)) * ((i + 1) & 31);
				for (; j_bit < min(32, N); j_bit++) {
					unsigned j = (j_block << 5) | j_bit;
					C_ijk = active & (row < j) & (j < N) &
						((s_row[wid] >> lane) &
						(jb >> j_bit) & 1);
					C_ij = __any_sync(mask, C_ijk);
					if (!lane)
						s_res[wid] = C_ij;
					__syncthreads();
					if (row == i)
						for (unsigned k = 0; k < bitblocks; k++)
							adj[j_block] |= s_res[k] << j_bit;
					__syncthreads();
				}
			}
			__syncthreads();
		}
	}
}

template<size_t ROW_CACHE_SIZE>
__device__ inline void Reduction(unsigned *link, unsigned *adj, const int N, const size_t bitblocks, const unsigned row, const unsigned wid, const unsigned lane)
{
	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	unsigned C_ijk, C_ij, active, mask;

	for (unsigned i = 0; i < bitblocks && row < N; i++)
		link[i] = adj[i];

	for (int i = 0; i < N - 1; i++) {
		if (row == i)
			for (size_t j = 0; j < bitblocks; j++)
				s_row[j] = link[j];
		__syncthreads();

		//Thread 'j' multiplies column 'j' in s_row by
		//row 'j', column 'i'
		C_ijk = 0, C_ij = 0;
		active = (row > i) & (row < N);
		mask = __ballot_sync(0xFFFFFFFF, active);
		/*for (int j_block = (i + 1) >> 5; j_block < bitblocks; j_block++) {	//Columns (blocks)
			unsigned jb = row < N ? adj[j_block] : 0;
			unsigned j_bit = (j_block == ((i + 1) >> 5)) * ((i + 1) & 31);
			for (; j_bit < min(32, N); j_bit++) {	//Columns (bits)
				unsigned j = (j_block << 5) | j_bit;
				if (!blockIdx.x) printf("[TID %u] (i,j) = (%u, %u)\n", row, i, j);
				unsigned adj_val = (jb >> j_bit) & 1;
				C_ijk = (active & row < j & j < N) &
					((s_row[wid] >> lane) &	
					adj_val);
				C_ij = __any_sync(mask, C_ijk);
				if (!lane)
					s_res[wid] = C_ij;
				__syncthreads();
				if (row == i) {
					unsigned res = 0;
					for (unsigned k = 0; k < bitblocks; k++)
						res |= s_res[k];
					adj_val = adj_val & (adj_val ^ res);
					if (adj_val)
						link[j_block] |= adj_val << j_bit;
					else
						link[j_block] &= ~(1u << j_bit);
				}
			}
		}*/

		for (int j = i + 1; j < N; j++) {
			unsigned adj_val = row < N ? (link[j>>5] >> (j & 31)) & 1 : 0;
			C_ijk = active & (row < j) &
				((s_row[wid] >> lane) &	adj_val);
			C_ij = __any_sync(mask, C_ijk);
			if (!lane)
				s_res[wid] = C_ij;
			__syncthreads();

			if (row == i) {
				unsigned res = 0;
				for (unsigned k = 0; k < bitblocks; k++)
					res |= s_res[k];
				adj_val = ~(!(adj_val & (adj_val ^ res)) << (j & 31));
				/*if (adj_val)
					link[j>>5] |= adj_val << (j & 31);
				else
					link[j>>5] &= ~(1u << (j & 31));*/
				link[j>>5] &= adj_val;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

template<size_t ROW_CACHE_SIZE>
__device__ inline void RelationPairCount(unsigned &cnt, unsigned *adj, const int N, const size_t bitblocks, const unsigned row, const unsigned wid, const unsigned lane)
{
	__shared__ unsigned s_row[ROW_CACHE_SIZE], s_res[ROW_CACHE_SIZE];
	cnt = 0;

	for (unsigned i = 0; i < N - 2; i++) {	//First element (row)
		if (row == i)
			for (unsigned j = wid; j < bitblocks; j++)
				s_row[j] = adj[j];
		__syncthreads();

		if (row >= i && row < N) {
			for (unsigned j_block = (i + 1) >> 5; j_block < bitblocks; j_block++) {	//Second element (column)
				unsigned jb = s_row[j_block];
				unsigned j_bit = (j_block == ((i + 1) >> 5)) * ((i + 1) & 31);
				for (; j_bit < min(32, N); j_bit++) {
					unsigned t1 = jb >> j_bit;
					unsigned j = (j_block << 5) | j_bit;
					for (unsigned k_block = (row + 1) >> 5; k_block < bitblocks && j < N; k_block++) {	//Fourth element (column)
						unsigned kb = adj[k_block];
						unsigned k_bit = (k_block == ((row + 1) >> 5)) * ((row + 1) & 31);
						for (; k_bit < min(32, N); k_bit++) {
							unsigned k = (k_block << 5) + k_bit;
							unsigned t2 = kb >> k_bit;
							cnt += ((t1 ^ t2) & 1) & (row < k) & !(i == row && k <= j);
						}
					}
				}
			}
		}
	}

	for (unsigned offset = 16; offset > 0; offset >>= 1)
		cnt += __shfl_down_sync(0xFFFFFFFF, cnt, offset, 32);

	if (!lane)
		s_res[wid] = cnt;
	__syncthreads();

	for (unsigned stride = 1; stride < bitblocks; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)) && wid + stride < bitblocks)
			s_res[wid] += s_res[wid+stride];
		__syncthreads();
	}

	if (!row)
		cnt = s_res[0];
}

template<size_t ROW_CACHE_SIZE>
__device__ inline void RelationCount(unsigned &cnt, unsigned *adj, const int N, const size_t bitblocks, const unsigned row, const unsigned wid, const unsigned lane, const unsigned rowmask, const unsigned cntmask)
{
	__shared__ unsigned s_res[ROW_CACHE_SIZE];
	cnt = 0;
	unsigned val;
	for (unsigned i = wid; i < bitblocks && row < N; i++) {
		val = adj[i];
		val &= (-(i != wid)) | cntmask;
		val &= (-(i != bitblocks - 1)) | rowmask;
		cnt += __popc(val);
	}
	//for (unsigned i = row + 1; i < N; i++)
	//	cnt += (adj[i>>5] >> (i & 31)) & 1;

	for (unsigned offset = 16; offset > 0; offset >>= 1)
		cnt += __shfl_down_sync(0xFFFFFFFF, cnt, offset, 32);

	if (!lane)
		s_res[wid] = cnt;
	__syncthreads();

	for (unsigned stride = 1; stride < bitblocks; stride <<= 1) {
		if (!lane && !(wid % (stride << 1)) && wid + stride < bitblocks)
			s_res[wid] += s_res[wid+stride];
		__syncthreads();
	}

	if (!row)
		cnt = s_res[0];
}

#endif
