#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <assert.h>
#include <Windows.h>

template <typename T> void fill_array(T* arr, size_t size, T (*fill_func)())
{
	for (size_t i=0;i<size;++i)
		arr[i] = fill_func();
}

template <typename T> void print_array(T const* arr, size_t size, char const* format)
{
	for (size_t i=0;i<size;++i)
		printf(format, arr[i]);
}

template <typename T> int compare_arrays(T const* arr1, T const* arr2, size_t size, T eps = T())
{
	for (size_t i=0;i<size;++i)
	{
		if (abs(arr1[i]-arr2[i]) > eps)
			return 1;
	}

	return 0;
}

template <typename T> void print_device_array(T const* d_array, size_t array_size, size_t offset, size_t print_size, char const* format)
{
	assert (array_size);
	T* h_array = (T*)malloc(array_size * sizeof(T));

	cudaMemcpy(h_array, d_array, array_size * sizeof(T), cudaMemcpyDeviceToHost);
	print_array(h_array + offset, print_size, format);

	free(h_array);
}

class GPUTimer
{
public:
	GPUTimer();
	~GPUTimer();
	
	void start();
	void stop();
	float elapsed() const;
	
private:
	GPUTimer(GPUTimer const&);
	GPUTimer& operator=(GPUTimer const&);
	
	cudaEvent_t startEvent_;
	cudaEvent_t stopEvent_;
	float 		elapsed_;
}; 

inline GPUTimer::GPUTimer() : 
elapsed_(0.f)
{
	cudaEventCreate(&startEvent_);
	cudaEventCreate(&stopEvent_);
}

inline GPUTimer::~GPUTimer()
{
	cudaEventDestroy(startEvent_);
	cudaEventDestroy(stopEvent_);
}

inline void GPUTimer::start()
{
	cudaEventRecord(startEvent_);
}

inline void GPUTimer::stop()
{
	cudaEventRecord(stopEvent_);
	cudaEventSynchronize(stopEvent_);
	cudaEventElapsedTime(&elapsed_, startEvent_, stopEvent_);
}

inline float GPUTimer::elapsed() const
{
	return elapsed_;
}

class CPUTimer
{
public:
	CPUTimer();
	~CPUTimer();
	
	void start();
	void stop();
	float elapsed() const;
	
private:
	CPUTimer(CPUTimer const&);
	CPUTimer& operator=(CPUTimer const&);
	
	LARGE_INTEGER freq_;
	LARGE_INTEGER start_;
	float 		elapsed_;
}; 

inline CPUTimer::CPUTimer() : 
elapsed_(0.f)
{
	QueryPerformanceFrequency(&freq_);
}

inline CPUTimer::~CPUTimer()
{
	
}

inline void CPUTimer::start()
{
	QueryPerformanceCounter(&start_);
}

inline void CPUTimer::stop()
{
	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);
	elapsed_ = float(end.QuadPart - start_.QuadPart)/(float(freq_.QuadPart)/1000.f);
}

inline float CPUTimer::elapsed() const
{
	return elapsed_;
}

#endif