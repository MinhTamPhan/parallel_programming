#include "../src/helper.cuh"

__global__ void scanKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int nBins, int bit) {   
    // TODO
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0)
        s_data[threadIdx.x] = ((in[i - 1] >> bit) & (nBins - 1));
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

__global__ void computeHist(uint32_t *in, int n, uint32_t *hist, int nBins, int bit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *pIn = &in[blockDim.x * blockIdx.x]
    if (i < n) {
        atomicAdd(&hist[(pIn[i] >> bit) & (nBins - 1)], 1);
    }
}

void radixSortLv1NoShared(const uint32_t * in, int n, uint32_t * out) {
    dim3 blockSize(512); // Default
    dim3 gridSize((n - 1) / blockSize.x + 1);
    int nBits = 2; // Assume: nBits in {1, 2, 4, 8, 16} // k = 1
    int nBins = 1 << nBits; // 2^nBits
    size_t nBytes = n * sizeof(uint32_t);
    uint32_t *d_in, *d_hist;
    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(uint32_t))); 
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
        //size_t smem = gridSize.x * nBins * sizeof(uint32_t);
        computeHist<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
    }
}