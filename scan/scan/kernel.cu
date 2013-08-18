#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <thrust\scan.h>
#include <thrust\device_vector.h>

#include "utils.h"
#include "scan.h"

#define ARRAY_SIZE 262144
#define BLOCK_SIZE 256

typedef int NUMBER;

float randf()
{
	return ((float)rand())/RAND_MAX * 10 + 1;
}

int randi()
{
	return rand() % 10;
}

template <typename T> void scan_gold(T const* in_array, T* out_array, size_t size)
{
	out_array[0] = 0;
	for (size_t i=1;i<size;++i)
		out_array[i] = out_array[i-1] + in_array[i-1];
}

int scan_test()
{
	srand((unsigned int)time(NULL));

	GPUTimer timer;
	CPUTimer timer1;

	NUMBER* d_array = NULL;
	NUMBER* h_array = NULL;
	NUMBER* h_array_gold = NULL;

	h_array = (NUMBER*)malloc(ARRAY_SIZE*sizeof(NUMBER));
	h_array_gold = (NUMBER*)malloc(ARRAY_SIZE*sizeof(NUMBER)); 
	cudaMalloc(&d_array, ARRAY_SIZE*sizeof(NUMBER));
	
	fill_array<NUMBER>(h_array, ARRAY_SIZE, randi);

	timer1.start();
	scan_gold(h_array, h_array_gold, ARRAY_SIZE);
	timer1.stop();

	cudaMemcpy(d_array, h_array, ARRAY_SIZE*sizeof(NUMBER), cudaMemcpyHostToDevice);
	timer.start();
	mycuda::scan_array<NUMBER, BLOCK_SIZE>(d_array, d_array, ARRAY_SIZE, mycuda::plus<NUMBER>(), 0);
	timer.stop();
	cudaMemcpy(h_array, d_array, ARRAY_SIZE*sizeof(NUMBER), cudaMemcpyDeviceToHost);

	bool bCorrect = compare_arrays<NUMBER>(h_array, h_array_gold, ARRAY_SIZE) == 0;

	printf("Comparison result: %s\n", bCorrect ? "CORRECT" : "INCORRECT"); 
	printf("Problem size: %d block_size: %d\n", ARRAY_SIZE, BLOCK_SIZE);
	printf("Time elapsed for GPU version: %f ms\n", timer.elapsed());
	printf("Time elapsed for CPU version: %f ms\n", timer1.elapsed());
	printf("Speedup: %f \n", timer1.elapsed()/timer.elapsed());

	cudaFree(d_array);

	fill_array<NUMBER>(h_array, ARRAY_SIZE, randi);

	timer1.start();
	scan_gold(h_array, h_array_gold, ARRAY_SIZE);
	timer1.stop();

	thrust::device_vector<NUMBER> d_v(h_array, h_array + ARRAY_SIZE);

	timer.start();
	thrust::exclusive_scan(d_v.begin(), d_v.end(), d_v.begin(), 0, thrust::plus<int>());
	timer.stop();

	thrust::copy(d_v.begin(), d_v.end(), h_array);

	bCorrect = compare_arrays<NUMBER>(h_array, h_array_gold, ARRAY_SIZE) == 0;

	printf("Comparison result: %s\n", bCorrect ? "CORRECT" : "INCORRECT"); 
	printf("Problem size: %d block_size: %d\n", ARRAY_SIZE, 0);
	printf("Time elapsed for GPU version: %f ms\n", timer.elapsed());
	printf("Time elapsed for CPU version: %f ms\n", timer1.elapsed());
	printf("Speedup: %f \n", timer1.elapsed()/timer.elapsed());

	free(h_array);
	free(h_array_gold);
	
	return 0;
}


int main()
{
	scan_test();
	return 0;
}