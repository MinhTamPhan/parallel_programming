#include "../src/helper.cuh"

__global__ void computeHist(uint32_t *in, int n, uint32_t *hist, int nBins, int bit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *pIn = &in[blockDim.x * blockIdx.x];//lấy các phần tử trong block đàng xét
    uint32_t *pHist = &hist[nBins * blockIdx.x];//khai báo hist của block nào 
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
__global__ void Scatter(uint32_t * in, int n, int nBits, int bit, int nBins, uint32_t *histScan, uint32_t * out)
{
    extern __shared__ int s_data[];
    int * s_in = s_data;
    int * s_hist = (int *)&s_in[blockDim.x];
    int *dst = (int *)&s_hist[blockDim.x];
    int *dst_ori = (int *)&dst[blockDim.x];
    int *startIndex = (int *)&dst_ori[blockDim.x];
    int * hist = (int *)&startIndex[blockDim.x];
    int * scan = (int *)&hist[blockDim.x];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < n)
    {
        s_in[threadIdx.x] = in[id];
        s_hist[threadIdx.x] = (s_in[threadIdx.x] >> bit) & (nBins - 1); // get bit, lấy giá trị di đang xét
    }
    else 
        s_hist[threadIdx.x] = nBins - 1;
    // TODO: B1 - sort radix with k = 1
    for (int b = 0; b < nBits; b++)
    {
        // compute hist
        hist[threadIdx.x] = (s_hist[threadIdx.x] >> b) & 1;
        __syncthreads();
        // scan
        if (threadIdx.x == 0)
            scan[0] = 0;
        else
            scan[threadIdx.x] = hist[threadIdx.x - 1];
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int val = 0;
            if (threadIdx.x >= stride)
                val = scan[threadIdx.x - stride];
            __syncthreads();
            scan[threadIdx.x] += val;
            __syncthreads();
        }
        __syncthreads();
        // scatter
        int nZeros = blockDim.x - scan[blockDim.x - 1] - hist[blockDim.x - 1];
        int rank = 0;
        if (hist[threadIdx.x] == 0)
            rank = threadIdx.x - scan[threadIdx.x];
        else
            rank = nZeros + scan[threadIdx.x];
        dst[rank] = s_hist[threadIdx.x];
        dst_ori[rank] = s_in[threadIdx.x];
        __syncthreads();        
        // copy or swap
        s_hist[threadIdx.x] = dst[threadIdx.x];
        s_in[threadIdx.x] = dst_ori[threadIdx.x];
    }
    __syncthreads();
    // TODO: B2
    if (threadIdx.x == 0)
        startIndex[s_hist[0]] = 0;
    else
    {
        if (s_hist[threadIdx.x] != s_hist[threadIdx.x - 1])
            startIndex[s_hist[threadIdx.x]] = threadIdx.x;
    }
    __syncthreads();
    // TODO: B3 và B4
    if (id < n)
    {
        int preRank = threadIdx.x - startIndex[s_hist[threadIdx.x]];
        int bin = ((s_in[threadIdx.x] >> bit) & (nBins - 1));
        int scan = histScan[bin * gridDim.x + blockIdx.x];
        int rank = scan + preRank;
        out[rank] = s_in[threadIdx.x];
    }
}


void radixSortLv2_1(const uint32_t * in, int n, uint32_t * out, int k = 2, dim3 blockSize=dim3(512)) {
    dim3 gridSize((n - 1) / blockSize.x + 1);
    // int nBits = k;
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
        computeHist<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        transpose_naive<<<gridSize.x, nBins>>>(d_hist_t, d_hist, nBins, gridSize.x);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        dim3 blockSizeScan(512);
        dim3 gridSizeScan((nhist - 1) / blockSizeScan.x + 1);
        int smemScan = blockSizeScan.x  * sizeof(uint32_t);
        if (gridSizeScan.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, sizeof(uint32_t) * gridSizeScan.x));
        }
        scanBlkKernelCnt<<<gridSizeScan, blockSizeScan, smemScan>>>(d_hist_t, nhist, d_scan, d_blkSums);
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

        int smemScatter = blockSize.x * 7 * sizeof(int);
        Scatter<<<gridSize, blockSize, smemScatter>>>(d_in, n, k, bit, nBins, d_scan, d_out);
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

void sort(const uint32_t * in, int n, uint32_t * out, int nBits, int blockSize, bool useDevice=false)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
        printf("\nRadix sort by device\n");
        radixSortLv2_1(in, n, out, nBits, blockSize);
    }
    else 
    {
        printf("\nRadix Sort by device(Thrust)\n");
        sortByThrust(in, n, out);
    	
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}