#ifndef SCAN_H
#define SCAN_H

#define NOMINMAX

#include <vector>
#include <cuda_runtime.h>

namespace mycuda
{

	//////////////////////////////////////////////////
	/// SCAN OPERATIONS
	template <typename T> struct plus
	{
		__device__ T operator ()(T const& a, T const& b) const
		{
			return a + b;
		}
	};

	template <typename T> struct mul
	{
		__device__ T operator ()(T const& a, T const& b) const
		{
			return a * b;
		}
	};

	template <typename T> struct min
	{
		__device__ T operator ()(T const& a, T const& b) const
		{
			return std::min(a, b);
		}
	};

	template <typename T> struct max
	{
		__device__ T operator ()(T const& a, T const& b) const
		{
			return std::max(a, b);
		}
	};

	//////////////////////////////////////////////////
	/// SCAN SINGLE THREAD BLOCK W/O BOUNDS CHECKING
	template <typename T, typename Op> __global__ void scan_block(T const* in_array, T* out_array, Op op, T init)
	{
		extern __shared__ T shmem[];

		shmem[threadIdx.x] = op(in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x],
			in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x + 1]);

		__syncthreads();

		for (int stride = 1; stride <= (blockDim.x >> 1); stride <<= 1)
		{
			if (threadIdx.x<blockDim.x/(2*stride))
			{
				shmem[2*(threadIdx.x+1)*stride-1] = op(shmem[2*(threadIdx.x+1)*stride-1], shmem[(2*threadIdx.x+1)*stride-1]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
			shmem[blockDim.x - 1] = init;

		__syncthreads();

		for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
		{
			if (threadIdx.x < blockDim.x/(2*stride))
			{
				T temp = shmem[(2*threadIdx.x+1)*stride-1];
				shmem[(2*threadIdx.x+1)*stride-1] = shmem[2*(threadIdx.x+1)*stride-1];
				shmem[2*(threadIdx.x+1)*stride-1] = op(shmem[2*(threadIdx.x+1)*stride-1], temp);
			}

			__syncthreads();
		}

		out_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x + 1] = op(shmem[threadIdx.x], in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x]);
		out_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x] = shmem[threadIdx.x];
	}


	//////////////////////////////////////////////////
	/// SCAN SINGLE THREAD BLOCK WITH BOUNDS CHECKING
#define FETCH_GLOBAL(array,idx,size) ((idx<size)?(array[idx]):T())
	template <typename T, typename Op> __global__ void scan_block_bounds_aware(T const* in_array, T* out_array, size_t size, Op op, T init)
	{
		extern __shared__ T shmem[];

		int offset = 2*blockDim.x*blockIdx.x;
		int address = offset + 2*threadIdx.x;
		int address1 = offset + 2*threadIdx.x + 1;

		shmem[threadIdx.x] = op(FETCH_GLOBAL(in_array, address, size), 
			FETCH_GLOBAL(in_array, address1, size));

		__syncthreads();

		for (int stride = 1; stride <= (blockDim.x >> 1); stride <<= 1)
		{
			if (threadIdx.x<blockDim.x/(2*stride))
			{
				shmem[2*(threadIdx.x+1)*stride-1] = op(shmem[2*(threadIdx.x+1)*stride-1], shmem[(2*threadIdx.x+1)*stride-1]);
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
			shmem[blockDim.x - 1] = init;

		__syncthreads();

		for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
		{
			if (threadIdx.x < blockDim.x/(2*stride))
			{
				T temp = shmem[(2*threadIdx.x+1)*stride-1];
				shmem[(2*threadIdx.x+1)*stride-1] = shmem[2*(threadIdx.x+1)*stride-1];
				shmem[2*(threadIdx.x+1)*stride-1] = op(shmem[2*(threadIdx.x+1)*stride-1], temp);
			}

			__syncthreads();
		}

		if (address1 < size)
		{
			out_array[address1] = op(shmem[threadIdx.x], in_array[address]);
			out_array[address] = shmem[threadIdx.x];
		}
		else if (address < size)
		{
			out_array[address] = shmem[threadIdx.x];
		}
	}

#define LOG_NUM_BANKS 5  
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
	////////////////////////////////////////////////////////////////
	/// SCAN SINGLE THREAD BLOCK WITH BOUNDS CHECKING UNROLLED
	template <typename T, size_t BS, typename Op> __global__ void scan_block_bounds_aware_part(T const* in_array, T* out_array, size_t size, T* out_part_sums, Op op, T init)
	{
		extern __shared__ T shmem[];

		
		int address = 2 * blockDim.x*blockIdx.x + threadIdx.x;
		int address1 = 2 * blockDim.x*blockIdx.x + threadIdx.x + blockDim.x;

		/// Use offsets to avoid shmem bank conflicts
		int shmemAddr = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x);
		int shmemAddr1 = threadIdx.x + blockDim.x + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x);

		/// Double shmem size to achieve perfect coalescing
		shmem[shmemAddr] = FETCH_GLOBAL(in_array, address, size);
		shmem[shmemAddr1] = FETCH_GLOBAL(in_array, address1, size);

		__syncthreads();

		/// Fully unrolled reduce
#define REDUCE(tid, stride) \
	if (tid < BS/(stride)){ \
	int addr1 = 2*((tid)+1)*(stride)-1 + CONFLICT_FREE_OFFSET(2*(tid+1)*(stride)-1); \
	int addr2 = (2*(tid)+1)*(stride)-1 + CONFLICT_FREE_OFFSET((2*(tid)+1)*(stride)-1); \
	shmem[addr1] = op(shmem[addr1], shmem[addr2]); \
	}

		if (BS >= 1) { REDUCE(threadIdx.x, 1); __syncthreads(); }
		if (BS >= 2) { REDUCE(threadIdx.x, 2); __syncthreads(); }
		if (BS >= 4) { REDUCE(threadIdx.x, 4); __syncthreads(); }
		if (BS >= 8) { REDUCE(threadIdx.x, 8); __syncthreads(); }
		if (BS >= 16) { REDUCE(threadIdx.x, 16); __syncthreads(); }

		if (BS >= 32){ REDUCE(threadIdx.x, 32); }
		if (BS >= 64) { REDUCE(threadIdx.x, 64); }
		if (BS >= 128) { REDUCE(threadIdx.x, 128); }
		if (BS >= 256) { REDUCE(threadIdx.x, 256); }
		if (BS >= 512) { REDUCE(threadIdx.x, 512); }
		if (BS >= 1024) { REDUCE(threadIdx.x, 1024); }

		T part_sum = init;

		if (threadIdx.x == 0)
		{
			part_sum = shmem[2 * blockDim.x - 1 + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
			shmem[2 * blockDim.x - 1 + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1) ] = init;
		}

		/// Fully unrolled upsweep
#define UPSWEEP(tid,stride) \
	if (tid < BS / (stride)) { \
	int addr1 = (2*tid+1)*stride-1 + CONFLICT_FREE_OFFSET((2*tid+1)*stride-1);\
	int addr2 = 2*(tid+1)*stride-1 + CONFLICT_FREE_OFFSET(2*(tid+1)*stride-1);\
	temp = shmem[addr1]; \
	shmem[addr1] = shmem[addr2]; \
	shmem[addr2] = op(shmem[addr2], temp); \
	}

		T temp = init;

		if (BS >= 1024) {UPSWEEP(threadIdx.x, 1024);}
		if (BS >= 512) {UPSWEEP(threadIdx.x, 512);}
		if (BS >= 256) {UPSWEEP(threadIdx.x, 256);}
		if (BS >= 128) {UPSWEEP(threadIdx.x, 128);}
		if (BS >= 64) {UPSWEEP(threadIdx.x, 64);}

		if (BS >= 32) {UPSWEEP(threadIdx.x, 32); __syncthreads();}
		if (BS >= 16) {UPSWEEP(threadIdx.x, 16); __syncthreads();}
		if (BS >= 8) {UPSWEEP(threadIdx.x, 8); __syncthreads();}
		if (BS >= 4) {UPSWEEP(threadIdx.x, 4); __syncthreads();}
		if (BS >= 2) {UPSWEEP(threadIdx.x, 2); __syncthreads();}
		if (BS >= 1) {UPSWEEP(threadIdx.x, 1); __syncthreads();}


		if (address1 < size)
		{
			out_array[address1] = shmem[shmemAddr1];
		}

		out_array[address] = shmem[shmemAddr];

		if (threadIdx.x == 0)
		{
			out_part_sums[blockIdx.x] = part_sum;
		}
	}

	////////////////////////////////////////////////////////////////
	/// HELPERS
	template <typename T, typename Op> __global__ void distribute_sums(T const* in_sums, T* out_array, size_t size, Op op)
	{
		int globalId = 2*blockDim.x*blockIdx.x + threadIdx.x;
		int globalId1 = 2*blockDim.x*blockIdx.x + threadIdx.x + blockDim.x;

		if (globalId < size)
			out_array[globalId] = op(out_array[globalId], in_sums[blockIdx.x]);

		if (globalId1 < size)
			out_array[globalId1] = op(out_array[globalId1], in_sums[blockIdx.x]);
	}

	template <size_t num> struct is_pow2
	{
		enum { Yes = !(num & (num-1)), No = (num & (num-1)) };
	};

	template <typename T> struct scan_level_t
	{
		size_t arraySize;
		T*     d_array;

		scan_level_t(size_t sz, T* d_arr) : 
		arraySize(sz), d_array(d_arr) {}
	};

	//////////////////////////////////////////////////////////
	/// CPU-SIDE SCAN LAUNCHER 
	template <typename T, size_t BLOCK_SIZE, typename Op> void scan_array(T const* in_d_array, T* out_d_array, size_t size, Op op, T const& init)
	{
		static_assert(BLOCK_SIZE <= 1024 , "Specified block size is not supported");
		static_assert(is_pow2<BLOCK_SIZE>::Yes, "Non power of two block sizes are unsupported");

		int block_size_2 = (BLOCK_SIZE << 1); 
		if (size == block_size_2)
		{
			scan_block<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(in_d_array, out_d_array, op, init);
		}
		else if (size < block_size_2)
		{
			scan_block_bounds_aware<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(in_d_array, out_d_array, size, op, init);
		}
		else
		{
			std::vector<scan_level_t<T> > levels;

			size_t level_size = size;

			levels.push_back(scan_level_t<T>(size, out_d_array));

			while( level_size > block_size_2 )
			{
				level_size = (level_size + block_size_2 - 1) / (block_size_2);

				T* d_level_array = NULL;
				cudaMalloc(&d_level_array, sizeof(T)*level_size);

				levels.push_back(scan_level_t<T>(level_size, d_level_array));
			}

			int NLEVELS = levels.size();

			for (int i=0;i<NLEVELS-1;++i)
			{
				int grid_size = levels[i+1].arraySize;
				scan_block_bounds_aware_part<T, BLOCK_SIZE, Op><<<grid_size, BLOCK_SIZE, 2*BLOCK_SIZE*sizeof(T) + CONFLICT_FREE_OFFSET(2*BLOCK_SIZE*sizeof(T)-1) + 1>>>((i==0)?in_d_array:levels[i].d_array, levels[i].d_array, levels[i].arraySize, levels[i+1].d_array, op, init);
			}

			scan_block_bounds_aware<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(levels[NLEVELS-1].d_array, levels[NLEVELS-1].d_array, levels[NLEVELS-1].arraySize, op, init);

			for (int i=levels.size()-2; i>=0;--i)
			{
				int grid_size = levels[i+1].arraySize; 
				distribute_sums<<<grid_size, BLOCK_SIZE>>>(levels[i+1].d_array, levels[i].d_array, levels[i].arraySize, op);
			}
		}
	}

}

#endif 