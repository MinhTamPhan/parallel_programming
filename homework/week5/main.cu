#include <time.h>

#include <iostream>

#include "helper.cuh"
using namespace std;
#define EPSILON 1e-5

void queryDevice(cudaDeviceProp& prop);
float* ramdomInitVec(size_t vec_size);
void meanTwoVecHost(float* A, float* B, float* C, int nElem);
bool checkResult(const float* vec_a, const float* vec_b, size_t numElements);
__global__ void meanTwoVecStream(const float* vec_a, const float* vec_b,
                                 float* vec_c, int numElements);
__global__ void streamOutput(const float* d_out, const float* h_out,
                             int numElements);
__global__ void streamCalcMean(const float* d_A, const float* d_B,
                               const float* d_C, int numElements);

void noStreamImp(float* A, float* B, float* C, int nElem);
void twoStreamImp(float* A, float* B, float*& C, int nElem);

int main(int argc, char* argv[]) {
  // srand(0);
  cudaDeviceProp prop;
  queryDevice(prop);
  int ipower = 10;
  if (argc > 1) ipower = atoi(argv[1]);
  int nElem = 1 << ipower + 1;
  float *A, *B, *C;
  A = ramdomInitVec(nElem);
  B = ramdomInitVec(nElem);
  C = (float*)safe_malloc_host<float>(nElem);
  meanTwoVecHost(A, B, C, nElem);
  float* output = (float*)safe_malloc_host<float>(nElem);
  GpuTimer timer;
  timer.Start();
  noStreamImp(A, B, output, nElem);
  /*bool isValid = checkResult(C, output, nElem);
  if (isValid)
    printf("calculate mean vector on device no stream: Test PASSED\n");*/
  timer.Stop();
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecNoStream time = %f ms\n", noStreamTime);

  float* output2;  //  = (float*)safe_malloc_host<float>(nElem);
  twoStreamImp(A, B, output2, nElem);
  bool isValid = checkResult(C, output2, nElem);
  /*for (int i = 0; i < nElem; ++i) {
    printf("wrong at %d\n", output2[i]);
  }*/
  if (isValid)
    printf("calculate mean vector on device no stream: Test PASSED\n");
  safe_free_host_ptr<float*>(3, A, B, C);
  safe_free_host_ptr<float*>(1, output);
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

float* ramdomInitVec(size_t vec_size) {
  float* vec = (float*)safe_malloc_host<float>(vec_size);
  for (int i = 0; i < vec_size; ++i) {
    vec[i] = rand() / (float)RAND_MAX;
  }
  return vec;
}

void meanTwoVecHost(float* A, float* B, float* C, int numE) {
  for (size_t i = 0; i < numE; i++) {
    C[i] = (A[i] + B[i]) / 2;
  }
}

bool checkResult(const float* vecA, const float* vecB, size_t numElements) {
  for (int i = 0; i < numElements; ++i)
    if (fabs(vecA[i] - vecB[i]) > EPSILON) return false;
  return true;
}

__global__ void meanTwoVecStream(const float* vecA, const float* vecB,
                                 float* vecC, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) vecC[i] = (vecA[i] + vecB[i]) / 2;
}

void noStreamImp(float* A, float* B, float* C, int nElem) {
  // allocate device
  float *d_A, *d_B, *d_C;
 
  size_t size = nElem * sizeof(float);
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_A, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_B, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_C, size));

  CHECK_ERR_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CHECK_ERR_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
  int threadsPerBlock = 512;  // 256;
  int blocksPerGrid = (nElem + threadsPerBlock - 1) / threadsPerBlock;

  // default stream 0
  meanTwoVecStream<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, nElem);
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  CHECK_ERR_CUDA(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
  
}

void twoStreamImp(float* A, float* B, float*& C, int nElem) {
  float *d_A, *d_B, *d_C;

  size_t size = nElem * sizeof(float);
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_A, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_B, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_C, size));
  CHECK_ERR_CUDA(cudaMallocHost((void**)&C, size));

  CHECK_ERR_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CHECK_ERR_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  cudaStream_t stream1, stream2;
  CHECK_ERR_CUDA(cudaStreamCreate(&stream1));
  CHECK_ERR_CUDA(cudaStreamCreate(&stream2));
  int streamSize = 4096;
  int threadsPerBlock = 2048;
  int blocksPerGrid = streamSize / threadsPerBlock;
  int nLoop = (nElem + streamSize - 1) / streamSize;
  GpuTimer timer;
  timer.Start();
  for (size_t i = 0; i < nLoop; i++) {
    int offset = streamSize * i;
    float* stepA = d_A + offset;
    float* stepB = d_B + offset;
    float* stepC = d_C + offset;
    float* h_C = C + offset;
    if (offset > nElem) streamSize = nElem % streamSize;
    meanTwoVecStream<<<streamSize, threadsPerBlock, 0, stream1>>>(
        stepA, stepB, stepC, streamSize);
    CHECK_ERR_CUDA(cudaStreamSynchronize(stream1));
    CHECK_ERR_CUDA(cudaMemcpyAsync(h_C, stepC, streamSize * sizeof(float),
                        cudaMemcpyDeviceToHost, stream2));
    CHECK_ERR_CUDA(cudaStreamSynchronize(stream2));
  }
  CHECK_ERR_CUDA(cudaStreamSynchronize(stream2));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream1));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream2));
  timer.Stop();
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecNoStream time = %f ms\n", noStreamTime);
}