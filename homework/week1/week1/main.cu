#include <math.h>
#include <string.h>
#include "helper.cuh"
#include <time.h>

constexpr auto EPSILON = 1e-5;
// func add two vector parallel on cuda
__global__ void vectorAdd(const float *vec_a, const float *vec_b, float *vec_c,
                          int numElements);
void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c, size_t numElements);
/* wrap logic add two vector on GPU
**-step 1 alocate vector on device 
**-step 2 copy vector from host to device
**-step 3 call func parallel on cuda
**-step 4 copy vector result from device to host
*/
int add_vec_on_device(float *vec_a, float *vec_b, float *vec_c,size_t numElements);
/*
**query info GPU device
*/
void query_device();
/*
**Verify that the result vector is correct
*/
bool chek_result(const float *vec_a, const float *vec_b, const float *vec_c,
                 size_t numElements);
float *ramdom_init_vec(size_t vec_size);
void argument_parser(int argc, char *argv[], size_t &vec_size, bool &isGPU);

int main(int argc, char *argv[]) {
  query_device();
  printf("execute program: %s, calculator add two vector\n", argv[0]);
  size_t vec_size = 0;
  bool execute_on_gpu;
  argument_parser(argc, argv, vec_size, execute_on_gpu);
  printf("vector size %zu\n", vec_size);

  float *h_a, *h_b, *h_c;  // host variable start prefix h_
  h_a = ramdom_init_vec(vec_size);
  h_b = ramdom_init_vec(vec_size);
  h_c = (float*)safe_malloc_host(vec_size * sizeof(float));
  int err = 0;
  if (execute_on_gpu) {
    err = add_vec_on_device(h_a, h_b, h_c, vec_size);
    if (is_failed(err)) printf("Failed to add vector on device\n");
  } else {
    clock_t tStart = clock();
    add_vec_on_host(h_a, h_b, h_c, vec_size);
    printf("Add vector size %d on CPU Time taken: %.4fs\n", vec_size,
           (double)(clock() - tStart) / CLOCKS_PER_SEC);
  }
  if (is_success(err)) {
    if (chek_result(h_a, h_b, h_c, vec_size)) printf("Test PASSED\n");
    else printf("Failed Test vec_size = %uz, on %s\n", vec_size, execute_on_gpu ? "GPU": "CPU");
  }
  safe_free_host_ptr<float *>(3, h_a, h_b, h_c);
  return 0;
}

__global__ void vectorAdd(const float *vec_a, const float *vec_b, float *vec_c,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) vec_c[i] = vec_a[i] + vec_b[i];
}

void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c,
                    size_t numElements) {
  for (size_t i = 0; i < numElements; i++) vec_c[i] = vec_a[i] + vec_b[i];
}

int _init_device(float *&d_vec, float *vec_a, size_t numElements) {
  size_t size = numElements * sizeof(float);
  cudaError_t cudaStatus = cudaMalloc((void **)&d_vec, size);

  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(cudaStatus));
    return -1;
  }
  cudaStatus = cudaMemcpy(d_vec, vec_a, numElements * sizeof(float),
                          cudaMemcpyKind::cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    printf("cuda memcopy error: %s in %s at line %d!\n",
           cudaGetErrorString(cudaStatus), __FILE__, __LINE__);
    return -2;
  }
  return 0;
}

void _free_device(float *d_a, float *d_b, float *d_c) {
  safe_free_device(d_a);
  safe_free_device(d_b);
  safe_free_device(d_c);
}

int add_vec_on_device(float *vec_a, float *vec_b, float *vec_c,
                      size_t numElements) {
  float *d_a = NULL, *d_b = NULL, *d_c = NULL;
  int err = _init_device(d_a, vec_a, numElements);
  if (is_failed(err)) {
    printf("init d_a failed error: %d\n", err);
    _free_device(d_a, d_b, d_c);
    return -5;
  }
  err = _init_device(d_b, vec_b, numElements);
  if (is_failed(err)) {
    _free_device(d_a, d_b, d_c);
    printf("init d_b failed error: %d\n", err);
    return -5;
  }
  cudaError_t cudaStatus = cudaMalloc((void **)&d_c, numElements * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    printf("cuda cudaMalloc error: %s in %s at line %d!\n",
           cudaGetErrorString(cudaStatus), __FILE__, __LINE__);
    _free_device(d_a, d_b, d_c);
    return -2;
  }
  int threadsPerBlock = 512;  // 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  clock_t tStart = clock();
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(cudaStatus));
    _free_device(d_a, d_b, d_c);
    return -3;
  }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(cudaStatus));
    _free_device(d_a, d_b, d_c);
    return -4;
  }
  printf("Add vector size %d on GPU Time taken: %.4fs\n",
         numElements, (double)(clock() - tStart) / CLOCKS_PER_SEC);
  // Copy output vector from GPU buffer to host memory.
  err = safe_copy_device<float>(vec_c, d_c, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  if (is_failed(err)) {
    _free_device(d_a, d_b, d_c);
    return -5;
  }
  _free_device(d_a, d_b, d_c);
  return 0;
}

void query_device() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }
}

bool chek_result(const float *vec_a, const float *vec_b, const float *vec_c,
                 size_t numElements) {
  for (int i = 0; i < numElements; ++i)
    if (fabs(vec_a[i] + vec_b[i] - vec_c[i]) > EPSILON) return false;
  return true;
}

float *ramdom_init_vec(size_t vec_size) {
  float *vec = (float *)safe_malloc_host(vec_size * sizeof(float));
  for (int i = 0; i < vec_size; ++i) {
    vec[i] = rand() / (float)RAND_MAX;
  }
  return vec;
}

void argument_parser(int argc, char *argv[], size_t &vec_size, bool &isGPU) {
  if (argc < 4) {
    printf("usge: addVec.exe -n 200 -gpu[cpu]\n");
    printf("-n: is length vector > 0\n");
    exit(EXIT_FAILURE);
  } else {
    long int size = atol(argv[2]);
    if (size < 0) {
      printf("usge: addVec.exe -n 200 -gpu[cpu]\n");
      printf("-n: is length vector > 0\n");
      exit(EXIT_FAILURE);
    }
    vec_size = size;
    if (strcmp(argv[3], "-gpu") == 0) isGPU = true;
  }
}