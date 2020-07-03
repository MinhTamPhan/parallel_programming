#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "helper.cuh"

inline unsigned long time_diff(long start, long end) { return end - start; }

typedef struct MatrixDim {
  unsigned int x, y;
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
void query_device(cudaDeviceProp &prop);
int _init_matrix_device(float *&device_matrix, float *host_matrix,
                        const MatrixDim &dim);
__global__ void device_matrix_multiplication(float *d_a, float *d_b, float *d_c,
                                             size_t ma, size_t na, size_t mb);
void _free_device(float *d_a, float *d_b, float *d_c);
void host_matrix_multiplication(float *A, float *B, float *&result,
                                const MatrixDim &dim_A, const MatrixDim &dim_B);
void host_matrix_multiplication_version2(float *A, float *B, float *&result,
                                         const MatrixDim &dim_A,
                                         const MatrixDim &dim_B);
float *ramdom_init_matrix(const MatrixDim &dim);

void argument_parser(int argc, char *argv[], Argument &cmd);
void print_matrix(float *A, const MatrixDim &dim);

int main(int argc, char *argv[]) {
  srand(100);
  Argument cmd;
  cudaDeviceProp prop;
  query_device(prop);
  int nx_thread = 32;
  // printf("execute time = %d\n", time_diff(clock(), start));
  argument_parser(argc, argv, cmd);
  float *h_a, *h_b, *h_c;  // host variable start prefix h_
  float *d_a = nullptr, *d_b = nullptr,
        *d_c = nullptr;  // host variable start prefix d_
  h_a = ramdom_init_matrix(cmd.dim_a);
  h_b = ramdom_init_matrix(cmd.dim_b);
  MatrixDim dim_c;
  clock_t start;
  dim_c.x = cmd.dim_a.y;
  dim_c.y = cmd.dim_b.x;
  /*printf("Matrix multiplication\nMatrix A:\n");
  print_matrix(h_a, cmd.dim_a);
  printf("Matrix B:\n");
  print_matrix(h_b, cmd.dim_b);*/
  long execute_time = 0;
  if (!cmd.exec_gpu) {
    if (cmd.cpu_version1) {
      start = clock();
      host_matrix_multiplication(h_a, h_b, h_c, cmd.dim_a, cmd.dim_b);
    } else {
      start = clock();
      host_matrix_multiplication_version2(h_a, h_b, h_c, cmd.dim_a, cmd.dim_b);
    }
    execute_time = time_diff(start, clock());
    //printf("Matrix result:\n");
    //print_matrix(h_c, dim_c);
  } else {
    int err = _init_matrix_device(d_a, h_a, cmd.dim_a);
    if (is_success(err)) {
      err = _init_matrix_device(d_b, h_b, cmd.dim_b);
      if (is_success(err)) {
        size_t size = sizeof(float) * cmd.dim_a.y * cmd.dim_b.y;
        h_c = (float *)safe_malloc_host(size);
        memset((void *)h_c, 0, size);
        err = _init_matrix_device(d_c, h_c, cmd.dim_b);
        if (is_success(err)) {
          dim3 blockSize(prop.maxThreadsPerBlock / nx_thread, nx_thread);
          dim3 gridSize((cmd.dim_a.y - 1) / blockSize.x + 1,
                        (cmd.dim_b.y - 1) / blockSize.y + 1);
          start = clock();
          device_matrix_multiplication<<<gridSize, blockSize>>>(
              d_a, d_b, d_c, cmd.dim_a.y, cmd.dim_a.x, cmd.dim_b.y);
          cudaError_t cudaStatus = cudaGetLastError();
          if (cudaStatus != cudaSuccess) {
            fprintf(stderr,
                    "Failed to launch matrix multiplication kernel (error code "
                    "%s)!\n",
                    cudaGetErrorString(cudaStatus));
            _free_device(d_a, d_b, d_c);
          }
          cudaStatus = cudaDeviceSynchronize();
          if (cudaStatus != cudaSuccess) {
            fprintf(stderr,
                    "Failed to launch matrix multiplication kernel (error code "
                    "%s)!\n",
                    cudaGetErrorString(cudaStatus));
            _free_device(d_a, d_b, d_c);
          }
          float *res =
              (float *)safe_malloc_host(sizeof(float) * dim_c.x * dim_c.y);
          err = safe_copy_device(res, d_c, dim_c.x * dim_c.y,
                                 cudaMemcpyDeviceToHost);
          execute_time = time_diff(start, clock());
          printf("matrix result \n");
          //print_matrix(res, dim_c);
        } else
          printf("Failed to matrix multiplication by host");
      } else
        printf("Failed to matrix multiplication by host");
    } else
      printf("Failed to matrix multiplication by host");
    _free_device(d_a, d_b, d_c);
  }
  safe_free_host_ptr<float *>(3, h_a, h_b, h_c);
  printf(
      "Matrix multiplication execute time : %ld minisecond (%f second - %f "
      "minute) \n",
      execute_time, (execute_time / 1000.0), (execute_time / 1000.0) / 60);
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

void host_matrix_multiplication(float *A, float *B, float *&result,
                                const MatrixDim &dim_A,
                                const MatrixDim &dim_B) {
  size_t size = sizeof(float) * dim_A.x * dim_B.y;
  result = (float *)safe_malloc_host(size);
  memset((void *)result, 0, size);
  for (size_t i = 0; i < dim_A.y; i++)
    for (size_t j = 0; j < dim_B.x; j++)
      for (size_t k = 0; k < dim_A.x; k++)
        result[i * dim_A.y + j] += A[i * dim_A.x + k] * B[k * dim_B.x + j];
}

void host_matrix_multiplication_version2(float *A, float *B, float *&result,
                                         const MatrixDim &dim_A,
                                         const MatrixDim &dim_B) {
  size_t size = sizeof(float) * dim_A.y * dim_B.y;
  result = (float *)safe_malloc_host(size);
  memset((void *)result, 0, size);
  for (size_t i = 0; i < dim_A.y; i++)
    for (size_t k = 0; k < dim_A.x; k++)
      for (size_t j = 0; j < dim_B.x; j++)
        result[i * dim_A.y + j] += A[i * dim_A.x + k] * B[k * dim_B.x + j];
}

float *ramdom_init_matrix(const MatrixDim &dim) {
  float *res =
      (float *)safe_malloc_host(size_t(dim.x) * size_t(dim.y) * sizeof(float));
  for (int i = 0; i < dim.x * dim.y; ++i) res[i] = rand() / (float)RAND_MAX;
  return res;
}

void argument_parser(int argc, char *argv[], Argument &cmd) {
  //-cpu -ma 5 -na 6 -mb 6 -nb 5
  if (argc < 9) {
    printf("usge: mulmatrix.exe -cpu[gpu] -ma 5 -na 6 -mb 6 -nb 5\n");
    printf("-n: is length vector > 0\n");
    exit(EXIT_FAILURE);
  } else {
    size_t i = 0;
    cmd.cpu_version1 = true;
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
      } else if (strcmp(argv[i], "-v2") == 0) {
        cmd.cpu_version1 = false;
      }
      i++;
    }
  }
  printf("execute program with %s\n", cmd.exec_gpu ? "gpu" : "cpu");
  if (!cmd.exec_gpu)
    printf("execute cpu version %d\n", cmd.cpu_version1 ? 1 : 2);
  else
    printf("execute gpu\n");
  printf("matrix A %d rows, %d column\n", cmd.dim_a.y, cmd.dim_a.x);
  printf("matrix B %d rows, %d column\n\n", cmd.dim_b.y, cmd.dim_b.x);
}

void print_matrix(float *A, const MatrixDim &dim) {
  for (size_t i = 0; i < dim.y; i++) {
    for (size_t j = 0; j < dim.x; j++) {
      printf("%6.2f    ", A[i * dim.x + j]);
    }
    printf("\n");
  }
  printf("\n\n");
}

int _init_matrix_device(float *&device_matrix, float *host_matrix,
                        const MatrixDim &dim) {
  size_t size = (size_t)dim.x * (size_t)dim.y;
  int err = safe_malloc_device(device_matrix, size);
  if (is_failed(err)) {
    fprintf(stderr, "Failed to allocate device vector C (error code %d)!\n",
            err);
    return -1;
  } else
    err = safe_copy_device(device_matrix, host_matrix, size,
                           cudaMemcpyKind::cudaMemcpyHostToDevice);
  if (is_failed(err)) {
    fprintf(stderr, "Failed to allocate device vector C (error code %d)!\n",
            err);
    return -1;
  }
  return 0;
}

__global__ void device_matrix_multiplication(float *d_a, float *d_b, float *d_c,
                                             size_t ma, size_t na, size_t mb) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix < ma && iy < mb) {
    for (size_t j = 0; j < mb; j++) {
      d_c[iy * ma + ix] += d_a[iy * na + j] * d_b[j * ma + ix];
    }
  }
  // printf("device_matrix_multiplication %d\n", na);
}

void _free_device(float *d_a, float *d_b, float *d_c) {
  safe_free_device(d_a);
  safe_free_device(d_b);
  safe_free_device(d_c);
}