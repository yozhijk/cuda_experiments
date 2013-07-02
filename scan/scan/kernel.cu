#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#include "utils.h"
#include "cu_prims.h"

#define ARRAY_SIZE 50000000
#define BLOCK_SIZE 512

typedef float NUMBER;

float randf()
{
	return ((float)rand())/RAND_MAX * 100.f;
}

int randi()
{
	return rand() % 100;
}

template <typename T> void scan_gold(T const* in_array, T* out_array, size_t size)
{
	out_array[0] = 0;
	for (size_t i=1;i<size;++i)
		out_array[i] = out_array[i-1] + in_array[i-1];
}

int main()
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
	
	fill_array<NUMBER>(h_array, ARRAY_SIZE, randf);

	timer1.start();
	scan_gold(h_array, h_array_gold, ARRAY_SIZE);
	timer1.stop();

	cudaMemcpy(d_array, h_array, ARRAY_SIZE*sizeof(NUMBER), cudaMemcpyHostToDevice);
	timer.start();
	scan_array(d_array, d_array, ARRAY_SIZE, BLOCK_SIZE);
	timer.stop();
	cudaMemcpy(h_array, d_array, ARRAY_SIZE*sizeof(NUMBER), cudaMemcpyDeviceToHost);

	bool bCorrect = compare_arrays<NUMBER>(h_array, h_array_gold, ARRAY_SIZE, 0.5f) == 0;

	printf("Comparison result: %s\n", bCorrect ? "CORRECT" : "INCORRECT"); 
	printf("Problem size: %d block_size: %d\n", ARRAY_SIZE, BLOCK_SIZE);
	printf("Time elapsed for GPU version: %f ms\n", timer.elapsed());
	printf("Time elapsed for CPU version: %f ms\n", timer1.elapsed());
	printf("Speedup: %f ", timer1.elapsed()/timer.elapsed());

	cudaFree(d_array);
	free(h_array);
	free(h_array_gold);
	
	return 0;
}
