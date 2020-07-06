#include <cuda_runtime_api.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

constexpr auto CUDA_SUCCESS = 0;
constexpr auto CUDA_MALOC_FAILED = -1;
constexpr auto CUDA_MEMCOPY_FAILED = -2;
constexpr auto CUDA_FREE_FAILED = -3;

#define CHECK_ERR_CUDA(call)                                                \
  {                                                                         \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      printf("cuda error: %s in %s at line %d!\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                                           \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }

static bool is_failed(int error_code) { return error_code < 0; }

static bool is_success(int error_code) { return error_code >= 0; }

template <typename T>
static void* safe_malloc_host(size_t n) {
  void* p = malloc(n * sizeof(T));
  if (!p) {
    fprintf(stderr, "[%s:%ul] Failed to allocate host (%ul bytes)\n", __FILE__,
            __LINE__, (unsigned long)n);
    exit(EXIT_FAILURE);
  }
  return p;
}

template <typename T>
static int safe_malloc_device(T*& dev_ptr, size_t size,
                              void (*handle_exception)() = nullptr) {
  cudaError_t err = cudaMalloc((void**)&dev_ptr, size * sizeof(T));
  if (err != cudaSuccess) {
    printf("cuda allocate error: %s in %s at line %d!\n",
           cudaGetErrorString(err), __FILE__, __LINE__);
    if (handle_exception != nullptr) {
      (*handle_exception)();
      exit(EXIT_FAILURE);
    }
    return CUDA_MALOC_FAILED;
  }
  return CUDA_SUCCESS;
}

template <typename T>
static int safe_copy_device(T* dest, T* src, size_t size,
                            cudaMemcpyKind cpy_kind,
                            void (*handle_exception)() = nullptr) {
  cudaError_t err = cudaMemcpy(dest, src, size * sizeof(T), cpy_kind);
  if (err != cudaSuccess) {
    printf("cuda memcpy error: %s in %s at line %d!\n", cudaGetErrorString(err),
           __FILE__, __LINE__);
    if (handle_exception != nullptr) {
      (*handle_exception)();
      exit(EXIT_FAILURE);
    }
    return CUDA_MEMCOPY_FAILED;
  }
  return CUDA_SUCCESS;
}

template <typename T>
static int safe_free_device(T* dev_ptr, void (*handle_exception)() = nullptr) {
  cudaError_t err = cudaFree(dev_ptr);
  if (err != cudaSuccess) {
    printf("cuda free error: %s in %s at line %d!\n", cudaGetErrorString(err),
           __FILE__, __LINE__);
    if (handle_exception != nullptr) {
      (*handle_exception)();
      exit(EXIT_FAILURE);
    }
    return CUDA_FREE_FAILED;
  }
  return CUDA_SUCCESS;
}

template <typename T>
static int safe_free_host_ptr(int count, ...) {
  va_list list;
  va_start(list, count);
  for (int i = 0; i < count; i++) free(va_arg(list, T));
  va_end(list);
  return count;
}