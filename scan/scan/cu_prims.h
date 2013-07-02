#ifndef CU_PRIMS_H
#define CU_PRIMS_H

#include <vector>
#include "utils.h"
#include <cuda_runtime.h>

//////////////////////////////////////////////////
/// SCAN SINGLE THREAD BLOCK W/O BOUNDS CHECKING
template <typename T> __global__ void scan_block(T const* in_array, T* out_array)
{
	extern __shared__ T shmem[];

	shmem[threadIdx.x] = in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x] + 
						 in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x + 1];

	__syncthreads();

	for (int stride = 1; stride <= (blockDim.x >> 1); stride <<= 1)
	{
		if (threadIdx.x<blockDim.x/(2*stride))
		{
			shmem[2*(threadIdx.x+1)*stride-1] += shmem[(2*threadIdx.x+1)*stride-1];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		shmem[blockDim.x - 1] = 0;

	__syncthreads();

	for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
	{
		if (threadIdx.x < blockDim.x/(2*stride))
		{
			T temp = shmem[(2*threadIdx.x+1)*stride-1];
			shmem[(2*threadIdx.x+1)*stride-1] = shmem[2*(threadIdx.x+1)*stride-1];
			shmem[2*(threadIdx.x+1)*stride-1] += temp;
		}

		__syncthreads();
	}

	out_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x + 1] = shmem[threadIdx.x] + in_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x];
	out_array[2*blockDim.x*blockIdx.x + 2*threadIdx.x] = shmem[threadIdx.x];
}


//////////////////////////////////////////////////
/// SCAN SINGLE THREAD BLOCK WITH BOUNDS CHECKING
#define FETCH_GLOBAL(array,idx,size) ((idx<size)?(array[idx]):T())
template <typename T> __global__ void scan_block_bounds_aware(T const* in_array, T* out_array, size_t size)
{
	extern __shared__ T shmem[];

	int offset = 2*blockDim.x*blockIdx.x;
	int address = offset + 2*threadIdx.x;
	int address1 = offset + 2*threadIdx.x + 1;

	shmem[threadIdx.x] = T();
	shmem[threadIdx.x] = FETCH_GLOBAL(in_array, address, size) + 
						 FETCH_GLOBAL(in_array, address1, size);

	__syncthreads();

	for (int stride = 1; stride <= (blockDim.x >> 1); stride <<= 1)
	{
		if (threadIdx.x<blockDim.x/(2*stride))
		{
			shmem[2*(threadIdx.x+1)*stride-1] += shmem[(2*threadIdx.x+1)*stride-1];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		shmem[blockDim.x - 1] = 0;

	__syncthreads();

	for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
	{
		if (threadIdx.x < blockDim.x/(2*stride))
		{
			T temp = shmem[(2*threadIdx.x+1)*stride-1];
			shmem[(2*threadIdx.x+1)*stride-1] = shmem[2*(threadIdx.x+1)*stride-1];
			shmem[2*(threadIdx.x+1)*stride-1] += temp;
		}

		__syncthreads();
	}

	if (address1 < size)
	{
		out_array[address1] = shmem[threadIdx.x] + in_array[address];
		out_array[address] = shmem[threadIdx.x];
	}
	else if (address < size)
	{
		out_array[address] = shmem[threadIdx.x];
	}
}

//////////////////////////////////////////////////
/// SCAN SINGLE THREAD BLOCK WITH BOUNDS CHECKING
template <typename T> __global__ void scan_block_bounds_aware_part(T const* in_array, T* out_array, size_t size, T* out_part_sums)
{
	extern __shared__ T shmem[];

	int address = 2*blockDim.x*blockIdx.x + 2*threadIdx.x;
	int address1 = 2*blockDim.x*blockIdx.x + 2*threadIdx.x + 1;

	shmem[threadIdx.x] = FETCH_GLOBAL(in_array, address, size) + 
						 FETCH_GLOBAL(in_array, address1, size);
	__syncthreads();

	for (int stride = 1; stride <= (blockDim.x >> 1); stride <<= 1)
	{
		if (threadIdx.x<blockDim.x/(2*stride))
		{
			shmem[2*(threadIdx.x+1)*stride-1] += shmem[(2*threadIdx.x+1)*stride-1];
		}
		__syncthreads();
	}

	T part_sum = 0;

	if (threadIdx.x == 0)
	{
		part_sum = shmem[blockDim.x - 1];
		shmem[blockDim.x - 1] = 0;
	}

	__syncthreads();

	for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1)
	{
		if (threadIdx.x < blockDim.x/(2*stride))
		{
			T temp = shmem[(2*threadIdx.x+1)*stride-1];
			shmem[(2*threadIdx.x+1)*stride-1] = shmem[2*(threadIdx.x+1)*stride-1];
			shmem[2*(threadIdx.x+1)*stride-1] += temp;
		}

		__syncthreads();
	}

	if (address1 < size)
	{
		out_array[address1] = shmem[threadIdx.x] + in_array[address];
		out_array[address] = shmem[threadIdx.x];
	}
	else if (address < size)
	{
		out_array[address] = shmem[threadIdx.x];
	}

	if (threadIdx.x == 0)
	{
		out_part_sums[blockIdx.x] = part_sum;
	}
}

template <typename T> __global__ void distribute_sums(T const* in_sums, T* out_array, size_t size)
{
	int globalId = 2*blockDim.x*blockIdx.x + threadIdx.x;
	int globalId1 = 2*blockDim.x*blockIdx.x + threadIdx.x + blockDim.x;

	if (globalId < size)
		out_array[globalId] += in_sums[blockIdx.x];
	
	if (globalId1 < size)
		out_array[globalId1] += in_sums[blockIdx.x];
}


template <typename T> struct scan_level_t
{
	size_t arraySize;
	T*     d_array;
	
	scan_level_t(size_t sz, T* d_arr) : 
	arraySize(sz), d_array(d_arr) {}
};

//////////////////////////////////////////////////////////
/// CPU-SIDE SCAN LAUNCHER
template <typename T> void scan_array(T const* in_d_array, T* out_d_array, size_t size, size_t block_size)
{
	int block_size_2 = (block_size << 1); 
	if (size == block_size_2)
	{
		scan_block<<<1, block_size, block_size*sizeof(T)>>>(in_d_array, out_d_array);
	}
	else if (size < block_size_2)
	{
	    scan_block_bounds_aware<<<1, block_size, block_size*sizeof(T)>>>(in_d_array, out_d_array, size);
	}
	else
	{
		std::vector<scan_level_t<T>> levels;

		size_t level_size = size;

		levels.push_back(scan_level_t<T>(size, out_d_array));

		while( level_size > block_size_2)
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
			scan_block_bounds_aware_part<<<grid_size, block_size, block_size*sizeof(T)>>>((i==0)?in_d_array:levels[i].d_array, levels[i].d_array, levels[i].arraySize, levels[i+1].d_array);
		}

		scan_block_bounds_aware<<<1, block_size, block_size*sizeof(T)>>>(levels[NLEVELS-1].d_array, levels[NLEVELS-1].d_array, levels[NLEVELS-1].arraySize );

		for (int i=levels.size()-2; i>=0;--i)
		{
			int grid_size = levels[i+1].arraySize; 
			distribute_sums<<<grid_size, block_size>>>(levels[i+1].d_array, levels[i].d_array, levels[i].arraySize);
		}
	}
}

#endif 