#include "helper.cuh"
#include <iostream>
#include <time.h>
using namespace std;


typedef struct Argument {
  bool exec_gpu;
  size_t vec_size;
  bool version1;
  int block_x, block_y;
};

void argumentParser(int argc, char* argv[], Argument& cmd);

void queryDevice(cudaDeviceProp& prop);
void sumArraysOnHost(int* A, int* B, int* C, const int N);
void initialData(int* ip, int size);
void checkResult(int* hostRef, int* gpuRef, const int N);

int main(int argc, char* argv[]) { 
	cout << "hello word\n";
	return 0; 
}

void queryDevice(cudaDeviceProp& prop) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Maximum size of each dimension of a block: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum size of each dimension of a grid: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Global memory available on device in gigabytes: %d\n",
           prop.totalGlobalMem / 1073741824);
    printf("  Shared memory available per block in kilobytes : %d\n",
           prop.sharedMemPerBlock / 1024);
    printf("  32-bit registers available per block : %d\n\n",
           prop.regsPerBlock);
  }
}

void checkResult(int* hostRef, int* gpuRef, const int N) {
  for (int i = 0; i < N; i++) {
    if (hostRef[i] != gpuRef[i]) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2d gpu %5.2d at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if (match) printf("Arrays match.\n\n");
}

void initialData(int* ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = rand() & 0xFF;
  }
}

void sumArraysOnHost(int* A, int* B, int* C, const int N) {
  for (int idx = 0; idx < N; idx++) C[idx] = A[idx] + B[idx];
}

void argumentParser(int argc, char* argv[], Argument& cmd) {

}

__global__ void reduceNeighbored(int* vec_a, int* result, size_t numElement) {
  size_t tid = threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + (blockDim.x * blockIdx.x);
  // boundary check
  if (idx >= numElement) return;
  // in-place reduction in global memory
  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceNeighboredLess(int* vec_a, int* result, size_t numElement) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x;
  // boundary check

  if (idx >= numElement) return;
  // in-place reduction in global memory
  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) local[index] += local[index + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceInterleaved(int* vec_a, int* result, size_t numElement) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x;
  // boundary check

  if (idx >= numElement) return;
  // in-place reduction in global memory
  for (size_t stride = blockIdx.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceUnrolling2(int* vec_a, int* result, size_t numElement) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x;
  // unrolling 2
  if (idx + blockDim.x < numElement) local[idx] += local[idx + blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockIdx.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceUnrollingWrap8(int* vec_a, int* result, size_t numElement) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x * 8;
  // unrolling 8


  if (idx + blockDim.x < numElement) local[idx] += local[idx + blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockIdx.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}