#include "helper.cuh"
#include <iostream>
#include <time.h>
using namespace std;

void queryDevice(cudaDeviceProp& prop);
int sumArraysOnHost(int* A, const int N);
void initialData(int* ip, int size);
void checkResult(int hostRef, int* gpuRef, const int N);
void printVec(int* vec, const size_t num_element);
void initialDeviceMem(int* hA, const size_t numE, int*& dA, int*& dRes,
                      const size_t numRes);

__global__ void reduceNeighbored(int* vec_a, int* result, const size_t nElem);
__global__ void reduceNeighboredLess(int* vec_a, int* result,
                                     const size_t nElem);
__global__ void reduceInterleaved(int* vec_a, int* result, const size_t nElem);
__global__ void reduceUnrolling2(int* vec_a, int* result, const size_t nElem);
__global__ void reduceUnrollingWrap8(int* vec_a, int* result,
                                     const size_t nElem);
template <unsigned int iBlockSize>
__global__ void reduceCompleteUroll(int* vec_a, int* result,
                                    const size_t nElem);

int main(int argc, char* argv[]) { 
  srand(0);
  cudaDeviceProp prop;
  queryDevice(prop);
  int ipower = 10;
  if (argc > 1) ipower = atoi(argv[1]);
  int nElem = 1 << ipower;
  size_t nBytes = nElem * sizeof(int);
  if (ipower < 18) {
    printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower,
           (float)nBytes / (1024.0f));
  } else {
    printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, ipower,
           (float)nBytes / (1024.0f * 1024.0f));
  }
  int* hA, *hRes;
  hA = (int*)malloc(nBytes);
  initialData(hA, nElem);
  int hostRes = sumArraysOnHost(hA, nElem);
  //printf("%d\n", hostRes);
  int threadsPerBlock = 512;  // 256;
  if (argc == 3) {
    threadsPerBlock = atoi(argv[2]);
  }
  int blocksPerGrid = (nElem + threadsPerBlock - 1) / threadsPerBlock;
  int* dA, *dRes;
  hRes = (int*)malloc(blocksPerGrid);
  GpuTimer timer;

  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceNeighbored<<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  //checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  float time = timer.Elapsed();
  printf("Processing reduceNeighbored time (%s): %f ms\n\n", "use device", time);

  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceNeighboredLess<<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  //checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  time = timer.Elapsed();
  printf("Processing reduceNeighboredLess time (%s): %f ms\n\n", "use device",
         time);

  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceInterleaved<<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  //checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  time = timer.Elapsed();
  printf("Processing reduceInterleaved time (%s): %f ms\n\n", "use device",
         time);

  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceUnrolling2<<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  //checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  time = timer.Elapsed();
  printf("Processing reduceUnrolling 2 warp time (%s): %f ms\n\n", "use device",
         time);

  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceUnrollingWrap8<<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  //checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  time = timer.Elapsed();
  printf("Processing reduceUnrolling Wrap 8 time (%s): %f ms\n\n", "use device",
         time);
  timer.Start();
  initialDeviceMem(hA, nElem, dA, dRes, blocksPerGrid);
  reduceCompleteUroll<512><<<blocksPerGrid, threadsPerBlock>>>(dA, dRes, nElem);
  safe_copy_device(hRes, dRes, blocksPerGrid, cudaMemcpyDeviceToHost);
  // checkResult(hostRes, hRes, blocksPerGrid);
  safe_free_device<int*>(2, dA, dRes);
  cudaDeviceSynchronize();
  timer.Stop();
  time = timer.Elapsed();
  printf("Processing reduceCompleteUroll Wrap 8 time (%s): %f ms\n\n", "use device",
         time);

  safe_free_host_ptr<int*>(1, hA);
  //free(hRes);
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

void checkResult(int hostRef, int* gpuRef, const int N) {
  int sum = 0;
  bool match = false;
  for (int i = 0; i < N; i++) {
    sum += gpuRef[i];
  }
  match = sum == hostRef;
  if (!match) {
    printf("gpuRef do not match!\n");
    printf("host %5.2d gpu %5.2d\n", hostRef, sum);
  }else printf("gpuRef and hostRef match.\n");
}

void initialData(int* ip, int size) {
  // generate different seed for random number
  // time_t t;
  // srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = rand() & 0xFF;
  }
}

int sumArraysOnHost(int* A, const int N) {
  int res = 0;
  for (int idx = 0; idx < N; idx++) res  += A[idx];
  return res;
}

void printVec(int* vec, const size_t num_element) {
  printf("vector: \n");
  for (size_t i = 0; i < num_element; i++) {
    printf("%d \t", vec[i]);
  }
  printf("\n\n");
}

void initialDeviceMem(int* hA, const size_t numE, int* &dA, int* &dRes,
                      const size_t numRes) {
  //printf("initialDeviceMem numE %s numRes %d\n", numE, numRes);
  safe_malloc_device<int>(dA, numE);
  safe_copy_device(dA, hA, numE, cudaMemcpyKind::cudaMemcpyHostToDevice);
  safe_malloc_device(dRes, numRes);
}

__global__ void reduceNeighbored(int* vec_a, int* result, const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + (blockDim.x * blockIdx.x);
  // boundary check
  if (idx >= nElem) return;
  // in-place reduction in global memory
  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceNeighboredLess(int* vec_a, int* result,
                                     const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x;
  // boundary check

  if (idx >= nElem) return;
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

__global__ void reduceInterleaved(int* vec_a, int* result, const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x;
  // boundary check

  if (idx >= nElem) return;
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceUnrolling2(int* vec_a, int* result, const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x * 2;
  // unrolling 2
  if (idx + blockDim.x < nElem) local[idx] += local[idx + blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduceUnrollingWrap8(int* vec_a, int* result, const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x * 8;
  // unrolling 8
  if (idx + 7 * blockDim.x < nElem) {
    int a1 = vec_a[idx];
    int a2 = vec_a[idx + blockDim.x];
    int a3 = vec_a[idx + 2 * blockDim.x];
    int a4 = vec_a[idx + 3 * blockDim.x];
    int b1 = vec_a[idx + 4 * blockDim.x];
    int b2 = vec_a[idx + 5 * blockDim.x];
    int b3 = vec_a[idx + 6 * blockDim.x];
    int b4 = vec_a[idx + 7 * blockDim.x];
    local[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // unrolling warp
  if (tid < 32) {
    volatile int* vmem = local;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUroll(int* vec_a, int* result, const size_t nElem) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int* local = vec_a + blockIdx.x * blockDim.x * 8;
  // unrolling 8
  if (idx + 7 * blockIdx.x < nElem) {
    int a1 = vec_a[idx];
    int a2 = vec_a[idx + blockDim.x];
    int a3 = vec_a[idx + 2 * blockDim.x];
    int a4 = vec_a[idx + 3 * blockDim.x];
    int b1 = vec_a[idx + 4 * blockDim.x];
    int b2 = vec_a[idx + 5 * blockDim.x];
    int b3 = vec_a[idx + 6 * blockDim.x];
    int b4 = vec_a[idx + 7 * blockDim.x];
    local[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // unrolling warp
  if (iBlockSize >= 1024 && tid == 512) local[tid] += local[tid + 512];
  __syncthreads();
  if (iBlockSize >= 512 && tid == 256) local[tid] += local[tid + 256];
  __syncthreads();
  if (iBlockSize >= 256 && tid == 128) local[tid] += local[tid + 128];
  __syncthreads();
  if (iBlockSize >= 128 && tid == 64) local[tid] += local[tid + 64];
  __syncthreads();
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}