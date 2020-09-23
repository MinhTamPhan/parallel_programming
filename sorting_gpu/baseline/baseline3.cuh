#include "../src/helper.cuh"
#include "../src/hist.cuh"
#include "../src/scan.cuh"


__global__ void coutingSortKernel(const uint32_t *in, int n, int *hist, int nBins, int bit) {
    // TODO
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_data[]; // Size: nBins elements
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
        s_data[bin] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&s_data[(in[i] >> bit) & (nBins - 1)], 1);
    __syncthreads();
}

__global__ void scanBlkKernelEx(int * in, int n, int * out, int * blkSums) {   
    // TODO
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0)
        s_data[threadIdx.x] = in[i - 1];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();

    // 2. Each block does scan with data on SMEM
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int neededVal;
        if (threadIdx.x >= stride)
            neededVal = s_data[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            s_data[threadIdx.x] += neededVal;
        __syncthreads();
    }

    // 3. Each block write results from SMEM to GMEM
    if (i < n)
        out[i] = s_data[threadIdx.x];
    if (blkSums != nullptr && threadIdx.x == 0)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n, uint32_t * out) {

    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;
    uint32_t *d_in;
    int *d_hist, *d_blkSums;
    CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(uint32_t)));
    dim3 blockSize(512); // Default
    dim3 gridSize((n - 1) / blockSize.x + 1);
    size_t smemSize = nBins * sizeof(int);
    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
    // In each loop, sort elements according to the current digit from src to dst 
    // (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
        // TODO: Compute histogram
        memset(hist, 0, nBins * sizeof(int));
        int bin;
         // TODO: Compute histogram by device
        CHECK(cudaMemcpy(d_in, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));
        coutingSortKernel<<<gridSize, blockSize, smemSize>>>(d_in, n, d_hist, nBins, bit);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Copy result from device memories
        //CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

        // TODO: Scan histogram (exclusively)
        // TODO: Compute histogram by device
        histScan[0] = 0;
        
        // for (int bin = 1; bin < nBins; bin++)
        //     histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        scanBlkKernelEx(d_hist, n, nBins, histScan, true, dim3(blockSize));
        // scan(hist, nBins, histScan, true, dim3(nBins));
        // TODO: Scatter elements to correct locations
        for (int i = 0; i < n; i++) {
            // bin = (src[i] >> bit) & (nBins - 1);
            bin = src[i] / (1 << (bit)) % (1 << nBits);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Copy result to out
    memcpy(out, src, n * sizeof(uint32_t)); 

    // Free memory
    free(originalSrc);
    free(hist);
    free(histScan);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
}

// Radix Sort
void sort(const uint32_t * in, int n,  uint32_t * out, bool useDevice=false, int blockSize=1) {
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false) {
        printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else {// use device
        printf("\nRadix Sort by device\n");
        sortByThrust(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}
