﻿#include <string.h>

#include "helper.cuh"
#define NF 100
#define NI (1 << 24)
#define NO (NI - NF + 1)
__constant__ float d_flt[NF];
__shared__ float d_in[NI];

void printDeviceInfo();
void doConvolution(int type);
__global__ void convOnDevice1(float *d_in, float *d_out);
__global__ void convOnDevice2(float *d_in, float *d_out);
__global__ void convOnDevice3(float *d_in, float *d_out);
__global__ void reduceOnDevice4(int *in, int *out, int n);

int main(int argc, char *argv[]) {
  printDeviceInfo();
  if (strcmp(argv[1], "convolution")) {
    doConvolution(1);
    doConvolution(2);
    doConvolution(3);
  } else {
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
    int *in = (int *)malloc(nBytes);
    for (int i = 0; i < nElem; i++) {
      // Generate a random integer in [0, 255]
      in[i] = (int)(rand() & 0xFF);
    }
  }
  return 0;
}

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK_ERR_CUDA(cudaGetDeviceProperties(&devProv, 0));
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

__global__ void convOnDevice3(float *d_in, float *d_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NO) {
    float s = 0;
    for (int j = 0; j < NF; j++) {
      s += d_flt[j] * d_in[i + j];
    }
    d_out[i] = s;
  }
}

void doConvolution(int type) {
  // Set up data for input and filter
  float *in, *flt, *out;
  out = (float *)malloc(NO * sizeof(float));
  in = (float *)malloc(NI * sizeof(float));
  for (int i = 0; i < NI; i++) {
    // Generate a random integer in [0, 255]
    in[i] = rand() / RAND_MAX;
  }
  flt = (float *)malloc(NF * sizeof(float));
  for (int i = 0; i < NF; i++) {
    // Generate a random integer in [0, 255]
    flt[i] = rand() / RAND_MAX;
  }
  dim3 blockSize(512);
  dim3 gridSize((NO - 1) / blockSize.x + 1);
  float *d_in, *d_out;
  // Allocate device memories
  GpuTimer timer;
  timer.Start();
  if (type == 1) { // Tích chập 1 chiều sử dụng bộ nhớ toàn cục
    cudaMalloc(&d_in, NI * sizeof(float));
    cudaMalloc(&d_out, NO * sizeof(float));
    // Copy data from host memories to device memories
    cudaMemcpy(d_in, in, NI * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_flt, flt, NF * sizeof(float));
    convOnDevice3<<<gridSize, blockSize>>>(d_in, d_out);
  } else if (type == 2) { // Tích chập 1 chiều sử dụng bộ nhớ hằng
  
  } else if (type == 2) {  // Tích chập 1 chiều sử dụng thanh ghi vs bộ nhớ toàn cục
  }
  timer.Stop();
  float kernelTime = timer.Elapsed();

  // Copy results from device memory to host memory
  cudaMemcpy(out, d_out, NO * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memories
  cudaFree(d_in);
  cudaFree(d_out);
}

__global__ void reduceOnDevice4(int *in, int *out, int n) {
  // Each block loads data from GMEM to SMEM
  __shared__ int blkData[2 * 256];
  int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
  blkData[threadIdx.x] = in[numElemsBeforeBlk + threadIdx.x];
  blkData[blockDim.x + threadIdx.x] =
      in[numElemsBeforeBlk + blockDim.x + threadIdx.x];
  __syncthreads();
  // Each block does reduction with data on SMEM
  for (int stride = blockDim.x; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      blkData[threadIdx.x] += blkData[threadIdx.x + stride];
    }
    __syncthreads();  // Synchronize within threadblock
  }
  // Each block writes result from SMEM to GMEM
  if (threadIdx.x == 0) out[blockIdx.x] = blkData[0];
}