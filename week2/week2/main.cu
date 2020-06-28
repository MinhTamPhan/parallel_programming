#include <math.h>
#include <string.h>
#include <time.h>

#include "helper.cuh"

#define time_diff(start, end) (start - end)

typedef struct MatrixDim {
  int x, y;
};

typedef struct Argument {
  bool exec_gpu;
  MatrixDim dim_a;
  MatrixDim dim_b;
  bool cpu_version1;
};

/*
**query info GPU device
*/
void query_device();
void host_matrix_multiplication(float *A, float *B, float *&result,
                                MatrixDim dim_A, MatrixDim dim_B);
void host_matrix_multiplication_version2(float *A, float *B, float *&result,
                                         MatrixDim dim_A, MatrixDim dim_B);
float *ramdom_init_vec(int vec_size);

void argument_parser(int argc, char *argv[], Argument &cmd);

//inline long time_diff(clock_t start, clock_t end) { return start - end; }

int main(int argc, char *argv[]) {
  Argument cmd;
  clock_t start = clock();
  query_device();
  //printf("execute time = %d\n", time_diff(clock(), start));
  argument_parser(argc, argv, cmd);
  //float *a, *b, *c;  // host variable start prefix h_
  //size_t vec_size = 10;
  //a = ramdom_init_vec(vec_size);
  //b = ramdom_init_vec(vec_size);
  //long execute_time = 0;
  //if (!cmd.exec_gpu) {
  //  if (cmd.cpu_version1) {
  //    clock_t start = clock();
  //    host_matrix_multiplication(a, b, c, cmd.dim_A, cmd.dim_b);
  //    execute_time = time_diff(start, clock()); 
  //  } else {
  //    clock_t start = clock();
  //    host_matrix_multiplication_version2(a, b, c, cmd.dim_A, cmd.dim_b);
  //    execute_time = time_diff(start, clock());
  //  }
  //} else {
  //  clock_t start = clock();
  //  host_matrix_multiplication_version2(a, b, c, cmd.dim_A, cmd.dim_b);
  //  execute_time = time_diff(start, clock());
  //}
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

void host_matrix_multiplication(float *A, float *B, float *&result,
                                MatrixDim dim_A, MatrixDim dim_B) {
  result = (float *)safe_malloc_host(sizeof(float) * dim_A.y * dim_B.y);
  for (size_t i = 0; i < dim_A.y; i++)
    for (size_t j = 0; j < dim_B.x; j++)
      for (size_t k = 0; k < dim_A.x; k++)
        result[i * dim_A.y + j] += A[i * dim_A.x + k] * B[k * dim_B.x + j];
}

void host_matrix_multiplication_version2(float *A, float *B, float *&result,
                                         MatrixDim dim_A, MatrixDim dim_B) {
  result = (float *)safe_malloc_host(sizeof(float) * dim_A.y * dim_B.y);
  for (size_t i = 0; i < dim_A.y; i++)
    for (size_t k = 0; k < dim_A.x; k++)
      for (size_t j = 0; j < dim_B.x; j++)
        result[i * dim_A.y + j] += A[i * dim_A.x + k] * B[k * dim_B.x + j];
}

float *ramdom_init_vec(int vec_size) {
  float *vec = (float *)safe_malloc_host(vec_size * sizeof(float));
  for (int i = 0; i < vec_size; ++i) vec[i] = rand() / (float)RAND_MAX;
  return vec;
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
      else if (strcmp(argv[i], "-ma") == 0) {
        i++;
        cmd.dim_a.y = atoi(argv[i]);
      } else if (strcmp(argv[i], "-na") == 0) {
        i++;
        cmd.dim_a.x = atoi(argv[i]);
      } else if (strcmp(argv[i], "-mb") == 0) {
        i++;
        cmd.dim_b.y = atoi(argv[i]);
      } else if (strcmp(argv[i], "-nb") == 0) {
        i++;
        cmd.dim_b.x = atoi(argv[i]);
      }
      i++;
    }
  }
  printf("execute program with %s\n", cmd.exec_gpu ? "gpu": "cpu");
  printf("matrix A %d rows, %d column %d\n", cmd.dim_a.y, cmd.dim_a.x);
  printf("matrix B %d rows, %d column %d\n\n", cmd.dim_b.y, cmd.dim_b.x);
}
