#include "../src/helper.cuh"
#include "../src/hist.cuh"
#include "../src/scan.cuh"

__global__ void addBlkSums(int * in, int n, int* blkSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        in[i] += blkSums[blockIdx.x - 1];
}

__global__ void Scatter(uint32_t * in, int n, int nBits, int bit, int nBins, int *histScan, uint32_t * out)
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

__global__ void computeHistKernel(uint32_t * in, int n, int * hist, int nBins, int bit)
{
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_bin[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    s_bin[threadIdx.x] = 0;
    __syncthreads();
    if (i < n)
    {
        int bin = (in[i] >> bit) & (nBins - 1);
        atomicAdd(&s_bin[bin], 1);
    }
    __syncthreads();
    if (threadIdx.x < nBins)
        hist[threadIdx.x * gridDim.x + blockIdx.x] += s_bin[threadIdx.x];
}
       
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums, int mode = 1)
{   
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
        s_data[blockDim.x - 1 - threadIdx.x] = in[i - 1];
    else
        s_data[blockDim.x - 1 - threadIdx.x] = 0;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x < blockDim.x - stride)
            val = s_data[threadIdx.x + stride];
        __syncthreads();
        s_data[threadIdx.x] += val;
        __syncthreads();
    }
    if (i < n)
        out[i] = s_data[blockDim.x - 1 - threadIdx.x];
    if (blkSums != NULL)
        blkSums[blockIdx.x] = s_data[0];
}

void radixSort(const uint32_t * in, int n, uint32_t * out, int nBits, int blockSize)
{
    int nBins = 1 << nBits;
    dim3 blkSize1(blockSize); // block size for histogram kernel
    dim3 blkSize2(blockSize); // block size for scan kernel
    dim3 gridSize1((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    dim3 gridSize2((nBins * gridSize1.x - 1) / blkSize2.x + 1);

    // TODO: initialize
    int * blkSums = (int *)malloc(gridSize2.x * sizeof(int));
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src;

    uint32_t * d_src, *d_dst;
    int *d_scan, *d_blkSums; 

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
	CHECK(cudaMalloc(&d_scan, nBins * gridSize1.x * sizeof(int)));
	CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));

    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    size_t sMemSize2 = blkSize2.x * sizeof(int);
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
        CHECK(cudaMemset(d_scan, 0, nBins * gridSize1.x * sizeof(int)));
        computeHistKernel<<<gridSize1, blkSize1, blkSize1.x * sizeof(int)>>>(d_src, n, d_scan, nBins, bit);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        // TODO: Scan
        scanBlkKernelEx<<<gridSize2, blkSize2, sMemSize2>>>(d_scan, nBins * gridSize1.x, d_scan, d_blkSums);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize2.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize2.x; i++)
            blkSums[i] += blkSums[i - 1];        
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize2.x * sizeof(int), cudaMemcpyHostToDevice));
        addBlkSums<<<gridSize2, blkSize2>>>(d_scan, nBins * gridSize1.x, d_blkSums);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        // TODO: Scatter
        Scatter<<<gridSize1, blkSize1, (blkSize1.x * 7 * sizeof(int))>>>(d_src, n, nBits, bit, nBins, d_scan, d_dst);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        // TODO: Swap "src" and "dst"
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp; 
    }
    // TODO: Copy result to "out"
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Free memories
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));
    
    free(blkSums);
    free(originalSrc);

    //int nBins = 1 << nBits;
    //dim3 blkSize(blockSize); // block size for histogram kernel
    //dim3 gridSize((n - 1) / blkSize.x + 1);
    //dim3 gridSizescan((nBins * gridSize.x - 1) / blkSize.x + 1);
//
    //// TODO
    //int * blkSums = (int *)malloc(gridSizescan.x * sizeof(int));
    //uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    //memcpy(src, in, n * sizeof(uint32_t));
    //uint32_t * originalSrc = src; // Use originalSrc to free memory later
//
//
    //uint32_t * d_src, *d_dst;
    //int *d_hist, *d_scan, *d_blkSums;
//
    //CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    //CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
	//CHECK(cudaMalloc(&d_hist, nBins * gridSize.x * sizeof(int)));
	//CHECK(cudaMalloc(&d_scan, nBins * gridSize.x * sizeof(int)));
	//CHECK(cudaMalloc(&d_blkSums, gridSizescan.x * sizeof(int)));
//
    //size_t smemSize = blkSize.x * sizeof(int);
    ////int nZeros;
    ////int rank;
//
    //for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    //{
    //	// TODO: Compute "hist" of the current digit
    //    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //    // TODO: Initialize d_hist using cudaMemset
    //    CHECK(cudaMemset(d_hist, 0, nBins * gridSize.x * sizeof(int)));
    //    //computeHistKernel<<<gridSize1, blkSize1, sMem>>>(d_src, n, d_hist, nBins, bit);
    //    computeHistKernelRadix2<<<gridSize, blockSize, smemSize>>>(d_src, n, d_hist, nBins, bit);
    //    cudaDeviceSynchronize();
	//    CHECK(cudaGetLastError());
//
    //    //scanBlkKernel<<<gridSize2, blkSize2, sMem>>>(d_hist, nBins * gridSize1.x, d_scan, d_blkSums);
    //    scanBlkKernel<<<gridSizescan, blkSize, smemSize>>>(d_hist, nBins * gridSize.x, d_scan, d_blkSums);
    //    cudaDeviceSynchronize();
	//    CHECK(cudaGetLastError());
//
    //    //CHECK(cudaMemcpy(blkSums, d_blkSums, gridSizescan.x * sizeof(int), cudaMemcpyDeviceToHost));
    //    //for (int i = 1; i < gridSizescan.x; i++)
    //    //    blkSums[i] += blkSums[i - 1];
    //    //CHECK(cudaMemcpy(d_blkSums, blkSums, gridSizescan.x * sizeof(int), cudaMemcpyHostToDevice));
    //    //addBlkSums<<<gridSizescan, blkSize>>>(d_scan, nBins * gridSize.x, d_blkSums);
    //    //cudaDeviceSynchronize();
	//    //CHECK(cudaGetLastError());
//
    //    //CHECK(cudaMemcpy(scan, d_scan, nBins * gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
//
    //    // TODO: Scatter
    //    Scatter<<<gridSize, blkSize, (blkSize.x * 7 * sizeof(int))>>>(d_src, n, nBits, bit, nBins, d_scan, d_dst);
    //    cudaDeviceSynchronize();
	//    CHECK(cudaGetLastError());
    //    
    //    // TODO: Swap "src" and "dst"
    //    uint32_t * temp = d_src;
    //    d_src = d_dst;
    //    d_dst = temp; 
    //}
    //// TODO: Copy result to "out"
    //CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    ////memcpy(out, src, n * sizeof(uint32_t));
    //// Free memories
    //CHECK(cudaFree(d_src));
    //CHECK(cudaFree(d_hist));
    //CHECK(cudaFree(d_scan));
    //CHECK(cudaFree(d_blkSums));
    //
    //free(blkSums);
    ////free(hist);
    ////free(scan);
    //free(originalSrc);
}

// Radix Sort
void sort(const uint32_t * in, int n, uint32_t * out, int nBits, int blockSize, bool useDevice=false)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
        printf("\nRadix sort by device\n");
        radixSort(in, n, out, nBits, blockSize);
    }
    else 
    {
        printf("\nRadix Sort by device(Thrust)\n");
        sortByThrust(in, n, out);
    	
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}