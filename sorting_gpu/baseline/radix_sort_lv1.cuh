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
    uint32_t *pIn = &in[blockDim.x * blockIdx.x];
    uint32_t *pHist = &hist[nBins * blockIdx.x];
    if (i < n) {
        atomicAdd(&pHist[(pIn[threadIdx.x] >> bit) & (nBins - 1)], 1);
    }
}

__global__ void transpose_naive(uint32_t *odata, uint32_t* idata, int width, int height) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xIndex = i % width;
    unsigned int yIndex = i % height;
   
    if (xIndex < width && yIndex < height) {
        unsigned int index_in  = xIndex + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in]; 
    }
}

__global__ void scanBlkKernelCnt(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int nBins, int bit) {   
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


// TODO: You can define necessary functions here
__global__ void addPrevBlkSumCnt(uint32_t * blkSumsScan, uint32_t * blkScans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}


void radixSortLv1NoShared(const uint32_t * in, int n, uint32_t * out, int k) {
    dim3 blockSize(49);
    //dim3 blockSize(512); // Default
    dim3 gridSize((n - 1) / blockSize.x + 1);
    // int nBits = k;
    int nBins = 1 << k;
    size_t nBytes = n * sizeof(uint32_t), hByte = nBins * sizeof(uint32_t) * gridSize.x;
    uint32_t *d_in, *d_hist, *hScan, *blkSums;
    uint32_t *d_hist_t;

    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_hist, hByte)); 
    CHECK(cudaMalloc(&d_hist_t, hByte));
    CHECK(cudaMalloc(&hScan, hByte));
    CHECK(cudaMalloc(&blkSums, sizeof(uint32_t) * gridSize.x));
    CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        CHECK(cudaMemset(d_hist, 0, hByte));
        computeHist<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // CHECK(cudaMemcpy(out, d_hist , hByte, cudaMemcpyDeviceToHost));
        transpose_naive<<<gridSize, blockSize>>>(d_hist_t, d_hist, 4, 3);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        scanBlkKernelCnt<<<gridSize, blockSize, 12 * 4>>>(d_hist_t, nBins * gridSize.x , hScan, blkSums, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out, hScan , hByte, cudaMemcpyDeviceToHost));
        break;
    }
}