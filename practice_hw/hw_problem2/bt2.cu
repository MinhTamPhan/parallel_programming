#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
using namespace std;

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
  }

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

__global__ void reduceBlksKernel1(int *in, int n, int *out) {
  // TODO
  size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
  for (size_t stride = 1; stride < 2 * blockDim.x; stride *= 2) {
    if ((threadIdx.x % stride) == 0)
      if (i + stride < n) in[i] += in[i + stride];
    __syncthreads();// synchonize within each block
  }
  if (threadIdx.x == 0) out[blockIdx.x] = in[blockIdx.x * blockDim.x * 2];
}

__global__ void reduceBlksKernel2(int *in, int n, int *out) {
  // TODO
  size_t numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
  for (size_t stride = 1; stride < 2 * blockDim.x; stride *= 2) {
    size_t i = numElemsBeforeBlk + threadIdx.x * 2 * stride;
    if (threadIdx.x < blockDim.x / stride)
      if (i + stride < n) in[i] += in[i + stride];
    __syncthreads();  // synchonize within each block
  }
  if (threadIdx.x == 0) out[blockIdx.x] = in[numElemsBeforeBlk];
}

__global__ void reduceBlksKernel3(int *in, int n, int *out) {
  // TODO
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
  // convert global data pointer to the local pointer of this block
  int *idata = in + blockIdx.x * blockDim.x * 2;
  // boundary check
  if (idx >= n) return;

  for (size_t stride = blockDim.x; stride > 0; stride >>= 1) {
    if (tid < stride)
      idata[tid] += idata[tid + stride];
    __syncthreads();  // synchonize within each block
  }
  if (threadIdx.x == 0) out[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *in, size_t n, int *out) {
  // set thread ID
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = in + blockIdx.x * blockDim.x * 2;
  // unrolling 2 data blocks
  if (idx + blockDim.x < n) in[idx] += in[idx + blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    // synchronize within threadblock
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) out[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *in, size_t n, int *out) {
  // set thread ID
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = in + blockIdx.x * blockDim.x * 8;
  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    int a1 = in[idx];
    int a2 = in[idx + blockDim.x];
    int a3 = in[idx + 2 * blockDim.x];
    int a4 = in[idx + 3 * blockDim.x];
    int b1 = in[idx + 4 * blockDim.x];
    int b2 = in[idx + 5 * blockDim.x];
    int b3 = in[idx + 6 * blockDim.x];
    int b4 = in[idx + 7 * blockDim.x];
    in[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  __syncthreads();
  // in-place reduction in global memory
  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    // synchronize within threadblock
    __syncthreads();
  }
  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }
  // write result for this block to global mem
  if (tid == 0) out[blockIdx.x] = idata[0];
}

int reduce(int const *in, int n, bool useDevice = false,
           dim3 blockSize = dim3(1), int kernelType = 1) {
  int result = 0;  // Init
  if (useDevice == false) {
    result = in[0];
    for (int i = 1; i < n; i++) result += in[i];
  } else  // Use device
  {
    // Allocate device memories
    int *d_in, *d_out;
    // TODO: Compute gridSize from n and 
    dim3 gridSize(((n + blockSize.x - 1) / blockSize.x)/2 + 1);
    if (kernelType == 5)
      gridSize = dim3(((n + blockSize.x - 1) / blockSize.x) / 8 + 1);
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));

    // Copy data to device memory
    CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

    // Call kernel
    GpuTimer timer;
    timer.Start();
    if (kernelType == 1)
      reduceBlksKernel1<<<gridSize, blockSize>>>(d_in, n, d_out);
    else if (kernelType == 2)
      reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, n, d_out);
    else if (kernelType == 3)
      reduceBlksKernel3<<<gridSize, blockSize>>>(d_in, n, d_out);
    else if (kernelType == 4)
      reduceUnrolling2<<<gridSize, blockSize>>>(d_in, n, d_out);
    else if (kernelType == 5) {
      reduceUnrollWarps8<<<gridSize, blockSize>>>(d_in, n, d_out);
    } 
    timer.Stop();
    float kernelTime = timer.Elapsed();
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memory
    int *out = (int *)malloc(gridSize.x * sizeof(int));
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int),
                     cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    // Host do the rest of the work
   
    timer.Start();
    result = out[0];
    for (int i = 1; i < gridSize.x; i++) {
      result += out[i];
    }
    timer.Stop();
    float postKernelTime = timer.Elapsed();
    // Free memory
    free(out);

    // Print info
    printf("\nKernel %d\n", kernelType);
    printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
    printf("Kernel time = %f ms, post-kernel time = %f ms\n", kernelTime,
           postKernelTime);
  }

  return result;
}

void checkCorrectness(int r1, int r2) {
  if (r1 == r2)
    printf("CORRECT :)\n");
  else
    printf("INCORRECT :(\n");
}

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
  printf("Max num warps per SM: %d\n",
         devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
  printf("****************************\n\n");
}
int main(int argc, char **argv) {
  printDeviceInfo();

  // Set up input size
  int n = (1 << 24) + 1;
  printf("Input size: %d\n", n);

  // Set up input data
  int *in = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    // Generate a random integer in [0, 255]
    in[i] = (int)(rand() & 0xFF);
  }

  // Reduce NOT using device
  int correctResult = reduce(in, n);
  // Reduce using device, kernel1
  dim3 blockSize(512);  // Default
  if (argc == 2) blockSize.x = atoi(argv[1]);
  int result1 = reduce(in, n, true, blockSize, 1);
  checkCorrectness(result1, correctResult);

  // Reduce using device, kernel2
  int result2 = reduce(in, n, true, blockSize, 2);
  checkCorrectness(result2, correctResult);

  // Reduce using device, kernel3
  int result3 = reduce(in, n, true, blockSize, 3);
  checkCorrectness(result3, correctResult);

  // Reduce using device, kernel4 reduceUnrolling2
  int result4 = reduce(in, n, true, blockSize, 4);
  checkCorrectness(result4, correctResult);

  // Reduce using device, kernel4 reduceUnrolling8
  int result5 = reduce(in, n, true, blockSize, 5);
  checkCorrectness(result5, correctResult);

  // Free memories
  free(in);
}