#include <time.h>

#include <iostream>
#include <cuda/std/utility>

#include "helper.cuh"
using namespace std;
#define EPSILON 1e-5

void queryDevice(cudaDeviceProp& prop);
float* ramdomInitVec(size_t vec_size);
void meanTwoVecHost(float* A, float* B, float* C, int nElem);
bool checkResult(const float* vec_a, const float* vec_b, size_t numElements);
__global__ void meanTwoVecStream(const float* vec_a, const float* vec_b,
                                 float* vec_c, int numElements);

void noStreamImp(float* A, float* B, float* C, int nElem);
void twoStreamImp(float* A, float* B, float*& C, int nElem);
void twoStreamImp2(float* A, float* B, float*& C, int nElem);

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
  timer.Stop();
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecNoStream time = %f ms\n", noStreamTime);

  float* output2 = (float*)safe_malloc_host<float>(nElem);
  twoStreamImp(A, B, output2, nElem);

  float* output3 = (float*)safe_malloc_host<float>(nElem);
  twoStreamImp2(A, B, output3, nElem);
  
  safe_free_host_ptr<float*>(3, A, B, C);
  safe_free_host_ptr<float*>(3, output, output2, output3);
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
  //cuprintf("meanTwoVecStream %d", vecA[0]);
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
  CHECK_ERR_CUDA(cudaFree(d_A));
  CHECK_ERR_CUDA(cudaFree(d_B));
  CHECK_ERR_CUDA(cudaFree(d_C));
}

void twoStreamImp(float* A, float* B, float*& C, int nElem) {
  float *d_A, *d_B, *d_C;

  size_t size = nElem * sizeof(float);
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_A, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_B, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_C, size));

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
    size_t offset = streamSize * i;
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
  }
  timer.Stop();
  CHECK_ERR_CUDA(cudaStreamSynchronize(stream2));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream1));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream2));
  CHECK_ERR_CUDA(cudaFree(d_A));
  CHECK_ERR_CUDA(cudaFree(d_B));
  CHECK_ERR_CUDA(cudaFree(d_C));
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecNoStream time = %f ms\n", noStreamTime);
}

void twoStreamImp2(float* A, float* B, float*& C, int nElem) {
  float *d_A, *d_B, *d_C, *t_C;

  size_t size = nElem * sizeof(float);
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_A, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_B, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_C, size));

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
    //CHECK_ERR_CUDA(cudaStreamSynchronize(stream1));
    CHECK_ERR_CUDA(cudaMemcpyAsync(h_C, stepC, streamSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream2));
    swap(stream1, stream2);
  }
  CHECK_ERR_CUDA(cudaStreamSynchronize(stream2));
  CHECK_ERR_CUDA(cudaStreamSynchronize(stream1));
  timer.Stop();
  CHECK_ERR_CUDA(cudaStreamDestroy(stream1));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream2));
  CHECK_ERR_CUDA(cudaFree(d_A));
  CHECK_ERR_CUDA(cudaFree(d_B));
  CHECK_ERR_CUDA(cudaFree(d_C));
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecTwoStream issue order, time = %f ms\n", noStreamTime);
}

void threeStreamImp(float* A, float* B, float*& C, int nElem) {
  float *d_A, *d_B, *d_C;

  size_t size = nElem * sizeof(float);
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_A, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_B, size));
  CHECK_ERR_CUDA(cudaMalloc((void**)&d_C, size));

  CHECK_ERR_CUDA(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
  CHECK_ERR_CUDA(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
  cudaStream_t stream1, stream2, stream3;
  CHECK_ERR_CUDA(cudaStreamCreate(&stream1));
  CHECK_ERR_CUDA(cudaStreamCreate(&stream2));
  CHECK_ERR_CUDA(cudaStreamCreate(&stream3));
  int streamSize = 4096;
  int threadsPerBlock = 2048;
  int blocksPerGrid = streamSize / threadsPerBlock;
  int nLoop = (nElem + streamSize - 1) / streamSize;
  GpuTimer timer;
  timer.Start();
  for (size_t i = 0; i < nLoop; i++) {
    size_t offset = streamSize * i;
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
  }
  timer.Stop();
  CHECK_ERR_CUDA(cudaStreamSynchronize(stream2));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream1));
  CHECK_ERR_CUDA(cudaStreamDestroy(stream2));
  CHECK_ERR_CUDA(cudaFree(d_A));
  CHECK_ERR_CUDA(cudaFree(d_B));
  CHECK_ERR_CUDA(cudaFree(d_C));
  float noStreamTime = timer.Elapsed();
  printf("execute meanTwoVecNoStream time = %f ms\n", noStreamTime);
}
