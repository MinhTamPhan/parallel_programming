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
  int block_x, block_y;
};
void argument_parser(int argc, char *argv[], Argument &cmd);

void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c,
                     size_t num_element);
void query_device(cudaDeviceProp &prop);
void print_vec(int *vec, size_t num_element);
int *ramdom_init_vec(size_t vec_size);

__global__ void reduce_neighbored(int *vec_a, int *result, size_t num_element);
__global__ void reduce_neighbored_less(int *vec_a, int *result, size_t num_element);
int _init_vector_device(int *&vec, int *init_vec, int *&result,
                        size_t vec_size);
void _free_device(int *d_a, int *d_b);

int main(int argc, char *argv[]) {
  srand(100);
  Argument cmd;
  cudaDeviceProp prop;
  query_device(prop);
  argument_parser(argc, argv, cmd);
  int nx_thread = 32;
  int *h_a, *h_result;  // host variable start prefix h_
  h_a = ramdom_init_vec(cmd.vec_size);
  h_result = (int *)safe_malloc_host<int>(cmd.vec_size);
  long start, execute_time;
  if (!cmd.exec_gpu) {
    printf("program execute cpu pass\n");
  } else {
    printf("program execute gpu\n");
    int *d_a, *d_result;
    start = clock();
    int err = _init_vector_device(d_a, h_a, d_result, cmd.vec_size);
    if (is_success(err)) {
      dim3 blockSize(cmd.block_x, cmd.block_y);
      dim3 gridSize((cmd.vec_size - 1) / blockSize.x + 1);
      if (cmd.version1)
        reduce_neighbored<<<gridSize, blockSize>>>(d_a, d_result, cmd.vec_size);
      else
        reduce_neighbored_less<<<gridSize, blockSize>>>(d_a, d_result,
                                                        cmd.vec_size);
      cudaError_t cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
                "Failed to launch matrix multiplication kernel (error code "
                "%s)!\n",
                cudaGetErrorString(cudaStatus));
      } else {
        err = safe_copy_device(h_result, d_result, cmd.vec_size,
                               cudaMemcpyDeviceToHost);
        if (is_failed(err)) printf("Failed to matrix multiplication by kernel\n");
      }
      execute_time = time_diff(start, clock());
    } else {
      printf("faild to init vector device\n");
    }
    _free_device(d_a, d_result);
  }
  safe_free_host_ptr<float *>(2, h_a, h_result);
  printf(
      "execute time : %ld minisecond (%f second - %f "
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

void print_vec(int *vec, size_t num_element) {
  printf("vector: \n");
  for (size_t i = 0; i < num_element; i++) {
    printf("%d \t", vec[i]);
  }
  printf("\n\n");
}

__global__ void reduce_neighbored(int *vec_a, int *result, size_t num_element) {
  size_t tid = threadIdx.x;
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int *local = vec_a + (blockDim.x * blockIdx.x);
  // boundary check
  if (idx >= num_element) return;
  // in-place reduction in global memory
  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) local[tid] += local[tid + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

__global__ void reduce_neighbored_less(int *vec_a, int *result,
                                       size_t num_element) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to local data pointer of this block
  int *local = vec_a + blockIdx.x * blockDim.x;
  // boundary check
 
  if (idx >= num_element) return;
  // in-place reduction in global memory
  for (size_t stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x)
      local[index] += local[index + stride];
    // synchonize within block
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) result[blockIdx.x] = local[0];
}

void argument_parser(int argc, char *argv[], Argument &cmd) {
  //-cpu -vecsize 5
  if (argc < 5) {
    printf("usge: 18424059.exe -cpu[gpu] -vecsize 4 -bsize 32 32\n");
    printf("-n: is length vector > 0\n");
    exit(EXIT_FAILURE);
  } else {
    size_t i = 0;
    cmd.version1 = true;
    while (i < argc) {
      if (strcmp(argv[i], "-cpu") == 0)
        cmd.exec_gpu = false;
      else if (strcmp(argv[i], "-gpu") == 0)
        cmd.exec_gpu = true;
      else if (strcmp(argv[i], "-vecsize") == 0) {
        cmd.vec_size = atoi(argv[++i]);
      }
      else if (strcmp(argv[i], "-v2") == 0)
        cmd.version1 = false;
      else if (strcmp(argv[i], "-bsize") == 0) {
        cmd.block_x = atoi(argv[++i]);
        cmd.block_y = atoi(argv[++i]);
      }
      i++;
    }
  }
  // alway usge gpu
  cmd.exec_gpu = true;
  printf("execute program with %s, vector size: %d, vesion %s\n", cmd.exec_gpu ? "gpu" : "cpu",
      cmd.vec_size, cmd.version1 ? "1": "2");
}

bool chek_result(const float *vec_a, const float *vec_b, const float *vec_c,
                 size_t num_element) {
  for (int i = 0; i < num_element; ++i)
    if (fabs((double)vec_a[i] + vec_b[i] - vec_c[i]) > EPSILON) return false;
  return true;
}

int *ramdom_init_vec(size_t vec_size) {
  int *vec = (int *)safe_malloc_host<int>(vec_size);
  for (int i = 0; i < vec_size; ++i) {
    vec[i] = rand();
  }
  return vec;
}

void add_vec_on_host(float *vec_a, float *vec_b, float *vec_c,
                     size_t num_element) {
  for (size_t i = 0; i < num_element; i++) vec_c[i] = vec_a[i] + vec_b[i];
}

void _free_device(int *d_a, int *d_b) {
  safe_free_device(d_a);
  safe_free_device(d_b);
}

int _init_vector_device(int *&vec, int *init_vec, int *&result, size_t vec_size) {

  int err = safe_malloc_device(vec, vec_size);
  if (is_failed(err)) {
    fprintf(stderr, "Failed to allocate device vector init_vec (error code %d)!\n",
            err);
    return -1;
  } else {
    err = safe_copy_device(vec, init_vec, vec_size,
                           cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (is_failed(err)) {
      fprintf(stderr, "Failed to copy host to device vector init_vec (error code %d)!\n",
              err);
      return -1;
    }
    err = safe_malloc_device(result, vec_size);
    if (is_failed(err)) {
      fprintf(stderr, "Failed to allocate device vector result (error code %d)!\n",
              err);
      return -1;
    }
  }
  return 0;
}