#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "helper.cuh"
constexpr auto EPSILON = 1e-5;

inline unsigned long time_diff(long start, long end) { return end - start; }

typedef struct Argument {
  bool exec_gpu;
  size_t vec_size;
  bool version1;
};

void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c,
                     size_t numElements);
void query_device(cudaDeviceProp &prop);

int main(int argc, char *argv[]) {
  srand(100);
  Argument cmd;
  cudaDeviceProp prop;
  query_device(prop);
  int nx_thread = 32;
  
  return 0;
}

void query_device(cudaDeviceProp &prop) {
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


void argument_parser(int argc, char *argv[], Argument &cmd) {
  //-cpu -ma 5 -na 6 -mb 6 -nb 5
  if (argc < 9) {
    printf("usge: mulmatrix.exe -cpu[gpu] -ma 5 -na 6 -mb 6 -nb 5\n");
    printf("-n: is length vector > 0\n");
    exit(EXIT_FAILURE);
  } else {
    size_t i = 0;
    while (i < argc) {
     if (strcmp(argv[i], "-cpu") == 0)
        cmd.exec_gpu = false;
     else if (strcmp(argv[i], "-gpu") == 0)
        cmd.exec_gpu = true;
     }
    i++;
  }
}

bool chek_result(const float *vec_a, const float *vec_b, const float *vec_c,
                 size_t numElements) {
  for (int i = 0; i < numElements; ++i)
    if (fabs((double)vec_a[i] + vec_b[i] - vec_c[i]) > EPSILON) return false;
  return true;
}

float *ramdom_init_vec(size_t vec_size) {
  float *vec = (float *)safe_malloc_host(vec_size * sizeof(float));
  for (int i = 0; i < vec_size; ++i) {
    vec[i] = rand() / (float)RAND_MAX;
  }
  return vec;
}


void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c,
                     size_t numElements) {
  for (size_t i = 0; i < numElements; i++) vec_c[i] = vec_a[i] + vec_b[i];
}
void _free_device(float *d_a, float *d_b, float *d_c) {
  safe_free_device(d_a);
  safe_free_device(d_b);
  safe_free_device(d_c);
}