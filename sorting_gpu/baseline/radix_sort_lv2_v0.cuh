#include "../src/helper.cuh"

__global__ void computeHistUseSMem(uint32_t *in, int n, uint32_t *hist, int nBins, int bit) {

    extern __shared__ uint32_t s_hist[];
    if (threadIdx.x < nBins)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *pIn = &in[blockDim.x * blockIdx.x];
    if (i < n) {
        atomicAdd(&s_hist[(pIn[threadIdx.x] >> bit) & (nBins - 1)], 1);
    }
    __syncthreads();
    int width = (n - 1) / blockDim.x + 1; // gridDim.x
    if (threadIdx.x < nBins)
        hist[threadIdx.x * width  + blockIdx.x] = s_hist[threadIdx.x];
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

__global__ void scanBlkKernelCnt(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums) {
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


// TODO: You can define necessary functions here
__global__ void addPrevBlkSumCnt(uint32_t * blkSumsScan, uint32_t * blkScans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}

// TODO: You can define necessary functions here
__global__ void scatter(uint32_t * in, int n, int nBits, int bit, int nBins, uint32_t *histScan, uint32_t * out)
{
    extern __shared__ uint32_t smem[];
    uint32_t *sIn = smem;
    uint32_t *dst = &sIn[blockDim.x];
    uint32_t *inScan = &dst[blockDim.x];
    uint32_t *startIndex = &inScan[blockDim.x];

    uint32_t numEleInBlock = blockDim.x;
    if((blockIdx.x + 1) * blockDim.x > n)
       numEleInBlock = n % blockDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x < numEleInBlock)
        sIn[threadIdx.x] = in[id];
    __syncthreads();
    // TODO: B1 - sort radix with k = 1
    for (int b = 0; b < nBits; b++) {
        // scan
        inScan[threadIdx.x] = threadIdx.x == 0 ? 0 : (sIn[threadIdx.x - 1] >> ( b+ bit)) & 1;
        __syncthreads();
        for (int stride = 1; stride < numEleInBlock; stride *= 2)
        {
            int val = 0;
            if (threadIdx.x >= stride)
                val = inScan[threadIdx.x - stride];
            __syncthreads();
            inScan[threadIdx.x] += val;
            __syncthreads();
        }
        __syncthreads();
        // scatter
        if (threadIdx.x < numEleInBlock) {
            int nZeros = numEleInBlock - inScan[numEleInBlock - 1] -  ((sIn[numEleInBlock - 1] >> ( b + bit)) & 1); 
            int rank =  ((sIn[threadIdx.x] >> ( b + bit)) & 1) == 0 ?  threadIdx.x - inScan[threadIdx.x] : nZeros + inScan[threadIdx.x];
            dst[rank] = sIn[threadIdx.x];
        }
        __syncthreads();        
        //swap theo kiểu copy vì có nhiều thread cùng lúc swap con trỏ sẽ gây ra lỗi
        sIn[threadIdx.x] = dst[threadIdx.x];
        __syncthreads();
    }
    
    startIndex[(sIn[0] >> bit) & (nBins - 1)] = 0;
    if (threadIdx.x > 0 && threadIdx.x < numEleInBlock) {
        int currDigit = ((sIn[threadIdx.x] >> bit) & (nBins - 1));
        int preDigit = ((sIn[threadIdx.x - 1] >> bit) & (nBins - 1));
        if (currDigit != preDigit)
            startIndex[currDigit] = threadIdx.x;
    }
    __syncthreads();
    if (threadIdx.x < numEleInBlock) {
       
        int digit = ((sIn[threadIdx.x] >> bit) & (nBins - 1));
        int localRank = threadIdx.x - startIndex[digit];
        int beginOut = histScan[digit * gridDim.x + blockIdx.x];
        int rank = beginOut + localRank;
        out[rank] = sIn[threadIdx.x];
    }
}


void radixSortLv2V0(const uint32_t * in, int n, uint32_t * out, int k = 2, dim3 blockSize=dim3(512)) {
    dim3 gridSize((n - 1) / blockSize.x + 1);
    int nBins = 1 << k;
    
    int nhist = gridSize.x * nBins;
    size_t nBytes = n * sizeof(uint32_t), hByte = nhist * sizeof(uint32_t);
    uint32_t *d_in, *d_hist, *d_scan, *d_blkSums = nullptr, *d_out;
    uint32_t *d_hist_t;

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    CHECK(cudaMalloc(&d_hist, hByte));
    CHECK(cudaMalloc(&d_hist_t, hByte));
    CHECK(cudaMalloc(&d_scan, hByte));
   
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        CHECK(cudaMemcpy(d_in, src, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, hByte));
        int smemHist = nBins * sizeof(uint32_t);
        computeHistUseSMem<<<gridSize, blockSize, smemHist>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        dim3 blockSizeScan(512);
        dim3 gridSizeScan((nhist - 1) / blockSizeScan.x + 1);
        int smemScan = blockSizeScan.x  * sizeof(uint32_t);
        if (gridSizeScan.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, sizeof(uint32_t) * gridSizeScan.x));
        }
        scanBlkKernelCnt<<<gridSizeScan, blockSizeScan, smemScan>>>(d_hist, nhist, d_scan, d_blkSums);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        if (gridSizeScan.x > 1) {
            // 2. Compute each block's previous sum
            //    by scanning array of blocks' sums
            size_t temp =  gridSizeScan.x * sizeof(uint32_t);
            int * h_blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(h_blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSizeScan.x; i++)
                h_blkSums[i] += h_blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, h_blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSumCnt<<<gridSizeScan.x - 1, blockSizeScan>>>(d_blkSums, d_scan, nhist);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            free(h_blkSums);
        }

        int smemScatter = (blockSize.x * 3 + nBins) * sizeof(int);
        scatter<<<gridSize, blockSize, smemScatter>>>(d_in, n, k, bit, nBins, d_scan, d_out);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(dst, d_out , nBytes, cudaMemcpyDeviceToHost));
        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }
    // Copy result to out
    memcpy(out, src, nBytes);

    free(originalSrc);

    // Free memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_hist_t));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));
}