#include "helper.cuh"

/*
Not use SMEM.
*/
__global__ void computeHistKernel1(uint32_t *in, int n, int *hist, int nBins)
{
    // TODO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[in[i]], 1);
}

__global__ void computeHistKernelRadix(uint32_t *in, int n, int *hist, int nBins, int bit)
{
    // TODO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        //int bin = (in[i] >> bit) & (nBins - 1);
        atomicAdd(&hist[(in[i] >> bit) & (nBins - 1)], 1);
    }
}

/*
Use SMEM.
*/
__global__ void computeHistKernel2(const uint32_t *in, int n, int *hist, int nBins)
{
    // TODO
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_hist[]; // Size: nBins elements
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
        s_hist[bin] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&s_hist[in[i]], 1);
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
        atomicAdd(&hist[bin], s_hist[bin]);
}

__global__ void computeHistKernelRadix2(const uint32_t *in, int n, int *hist, int nBins, int bit)
{
    // TODO
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_hist[]; // Size: nBins elements
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
        s_hist[bin] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&s_hist[(in[i] >> bit) & (nBins - 1)], 1);
    __syncthreads();

    // Each block adds its local hist to global hist using atomic on GMEM
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
        atomicAdd(&hist[bin], s_hist[bin]);
}

void computeHist(const uint32_t *in, int n, int *hist, int nBins, bool useDevice = false,
                 dim3 blockSize = dim3(1), int kernelType = 1, int bit = 0)
{

    GpuTimer timer;
    timer.Start();

    if (useDevice == false)
    {
        printf("\nHistogram by host\n");
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; i++)
            hist[(in[i] >> bit) & (nBins - 1)]++;
        // hist[in[i]]++;
    }
    else
    { // Use device
        printf("\nHistogram by device, kernel %d, ", kernelType);

        // Allocate device memories
        uint32_t *d_in;
        int *d_hist;
        CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
        CHECK(cudaMalloc(&d_hist, nBins * sizeof(uint32_t)));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // TODO: Initialize d_hist using cudaMemset
        CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));

        // Call kernel
        dim3 gridSize((n - 1) / blockSize.x + 1);
        printf("block size: %d, grid size: %d\n", blockSize.x, gridSize.x);
        if (kernelType == 1)
        {
            // computeHistKernel1<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins);
            //if(bit != 0)
            computeHistKernelRadix<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
            //else computeHistKernel1<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins);
        }
        else
        { // kernelType == 2
            size_t smemSize = nBins * sizeof(int);
            // computeHistKernelRadix2<<<gridSize, blockSize, smemSize>>>(d_in, n, d_hist, nBins);
            computeHistKernelRadix2<<<gridSize, blockSize, smemSize>>>(d_in, n, d_hist, nBins, bit);
        }
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Copy result from device memories
        CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_hist));
    }

    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
}

void computeHistDevice(const uint32_t *in, int n, int *hist, int nBins, bool useDevice = false,
                 dim3 blockSize = dim3(1), int kernelType = 1, int bit = 0)
{

    GpuTimer timer;
    timer.Start();

    // Use device
    // printf("\nHistogram by device, kernel %d, ", kernelType);

    // Allocate device memories
    uint32_t *d_in;
    int *d_hist;
    CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(uint32_t)));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // TODO: Initialize d_hist using cudaMemset
    CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));

    // Call kernel
    dim3 gridSize((n - 1) / blockSize.x + 1);
    //printf("block size: %d, grid size: %d\n", blockSize.x, gridSize.x);
    if (kernelType == 1) {
        // computeHistKernel1<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins);
        //if(bit != 0)
        computeHistKernelRadix<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
        //else computeHistKernel1<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins);
    }
    else { // kernelType == 2
        size_t smemSize = nBins * sizeof(int);
        // computeHistKernelRadix2<<<gridSize, blockSize, smemSize>>>(d_in, n, d_hist, nBins);
        computeHistKernelRadix2<<<gridSize, blockSize, smemSize>>>(d_in, n, d_hist, nBins, bit);
    }
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));

    timer.Stop();
    // printf("Processing time: %.3f ms\n", timer.Elapsed());
}

void checkCorrectness(int *out, int *correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}
