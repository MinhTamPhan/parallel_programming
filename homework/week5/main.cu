#include <time.h>

#include <iostream>

#include "helper.cuh"
using namespace std;
#define EPSILON 1e-5

void queryDevice(cudaDeviceProp& prop);
float* ramdom_init_vec(size_t vec_size);
void mean_two_vec_host(float* A, float* B, float* C, int nElem);
bool check_result(const float* vec_a, const float* vec_b, size_t numElements);
__global__ void mean_two_vec_no_stream(const float* vec_a, const float* vec_b,
                                       float* vec_c, int numElements);

int main(int argc, char* argv[]) {
  // srand(0);
  cudaDeviceProp prop;
  queryDevice(prop);
  int ipower = 10;
  int kernalType = 0;
  if (argc > 2) {
    ipower = atoi(argv[1]);
    kernalType = atoi(argv[2]);
  }
  int nElem = 1 << ipower + 1;
  float *A, *B, *C;
  A = ramdom_init_vec(nElem);
  B = ramdom_init_vec(nElem);
  C = (float*)safe_malloc_host<float>(nElem);
  mean_two_vec_host(A, B, C, nElem);

  safe_free_host_ptr<float*>(3, A, B, C);
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

float* ramdom_init_vec(size_t vec_size) {
  float* vec = (float*)safe_malloc_host<float>(vec_size);
  for (int i = 0; i < vec_size; ++i) {
    vec[i] = rand() / (float)RAND_MAX;
  }
  return vec;
}

void mean_two_vec_host(float* A, float* B, float* C, int numE) {
  for (size_t i = 0; i < numE; i++) {
    C[i] = (A[i] + B[i]) / 2;
  }
}

bool check_result(const float* vec_a, const float* vec_b, size_t numElements) {
  for (int i = 0; i < numElements; ++i)
    if (fabs(vec_a[i] - vec_b[i]) > EPSILON) return false;
  return true;
}

__global__ void mean_two_vec_no_stream(const float* vec_a, const float* vec_b,
                                       float* vec_c, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) vec_c[i] = (vec_a[i] + vec_b[i]) / 2;
}