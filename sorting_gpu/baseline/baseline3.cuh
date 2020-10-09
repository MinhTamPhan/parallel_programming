#include "../src/helper.cuh"
#include "../src/scan.cuh"

// __global__ void computeHistKernel(const uint32_t *in, int n, int *hist, int nBins, int bit) {
//     // TODO
//     // Each block computes its local hist using atomic on SMEM
//     extern __shared__ int s_data[]; // Size: nBins elements
//     for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
//         s_data[bin] = 0;
//     __syncthreads();
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n)
//         atomicAdd(&s_data[(in[i] >> bit) & (nBins - 1)], 1);
//     __syncthreads();
// }

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

// __global__ void addPrevBlkSum(uint32_t * blkSumsScan, uint32_t * blkScans, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
//     if (i < n)
//         blkScans[i] += blkSumsScan[blockIdx.x];
// }

void coutingSort2(const uint32_t * in, int n, uint32_t * out) {
    GpuTimer timer;
    timer.Start();

    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16} // k = 1
    int nBins = 1 << nBits; // 2^nBits

    uint32_t * inScan = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;
    uint32_t *d_in;
    uint32_t *d_blkSums, *d_out;
    size_t nBytes = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    // Copy data to device memories
    dim3 blockSize(512);
    dim3 gridSize((n - 1) / blockSize.x + 1);
    if (gridSize.x > 1) {
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    } else {
        d_blkSums = nullptr;
    }

    size_t smemSize = blockSize.x * sizeof(int);
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
        CHECK(cudaMemcpy(d_in, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        scanKernel<<<gridSize, blockSize, smemSize>>>(d_in, n, d_out, d_blkSums, nBins, bit);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        if (gridSize.x > 1) {
            // 2. Compute each block's previous sum
            //    by scanning array of blocks' sums
            // TODO
            size_t temp = gridSize.x * sizeof(int);
            int * blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSize.x; i++)
                blkSums[i] += blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            //addPrevBlkSum<<<gridSize.x - 1, blockSize>>>(d_blkSums, d_out, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());

            free(blkSums);
        }

        CHECK(cudaMemcpy(inScan, d_out, nBytes, cudaMemcpyDeviceToHost));
        int nZeros = n - (inScan[n - 1] + ((in[n - 1] >> bit) & 1));
        int rank;
        printf("nZeros %d\n", nZeros);
        for (int i = 0; i < n; i++) {
            rank = (src[i] >> bit) & (nBins - 1) ? nZeros + inScan[i] : i - inScan[i];
            // bin = src[i] / (1 << (bit)) % (1 << nBits);
            dst[rank] = src[i];
            // histScan[bin]++;
        }
        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }
    //computeHistKernelRadix<<<gridSize, blockSize>>>(d_in, n, d_hist, nBins, bit);
    // TODO: Initialize d_hist using cudaMemset
    //CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));

    // Call kernel

    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
    free(originalSrc);
    CHECK(cudaFree(d_in));
}


void coutingSort(const uint32_t * in, int n, uint32_t * out) {
    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16} // k = 1

    uint32_t * inScan = (uint32_t *)malloc(n * sizeof(uint32_t));
    // int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
    // In each loop, sort elements according to the current digit from src to dst
    // (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {

        dim3 blockSize(512); // Default
        inScan[0] = 0;
        // histScan[0] = 0;
        // for (int bin = 1; bin < nBins; bin++)
        //     histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        // scanExclusiveCounting(src, n, inScan, bit);
        scanExclusiveCounting(src, n, inScan, bit, true, dim3(blockSize));
        // scan(hist, nBins, histScan, true, dim3(nBins));
        // TODO: Scatter elements to correct locations

        int rank;
        int nZeros = n - (inScan[n - 1] + ((src[n - 1] >> bit) & 1));
        for (int i = 0; i < n; i++) {
            rank = ((src[i] >> bit) & 1) ? nZeros + inScan[i] : i - inScan[i];
            // bin = src[i] / (1 << (bit)) % (1 << nBits);
            dst[rank] = src[i];
            // histScan[bin]++;
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
    free(inScan);
}

void coutingSort3(const uint32_t * in, int n, uint32_t * out) {
    GpuTimer timer;
    timer.Start();
    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16} // k = 1

    uint32_t * inScan = (uint32_t *)malloc(n * sizeof(uint32_t));
    // int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    dim3 blockSize(512); // Default
    uint32_t * d_in, * d_out, * d_blkSums;
    size_t nBytes = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    dim3 gridSize((n - 1) / blockSize.x + 1);

    //CHECK(cudaMemcpy(d_in, src, nBytes, cudaMemcpyHostToDevice));
    //size_t smem = blockSize.x * sizeof(int);

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
    // In each loop, sort elements according to the current digit from src to dst
    // (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {

        //dim3 blockSize(512); // Default
        inScan[0] = 0;
        CHECK(cudaMemcpy(d_out, inScan, nBytes, cudaMemcpyHostToDevice));
        // histScan[0] = 0;
        // for (int bin = 1; bin < nBins; bin++)
        //     histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        // scanExclusiveCounting(src, n, inScan, bit);
        //scanExclusiveCounting(src, n, inScan, bit, true, dim3(blockSize));
        if (gridSize.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
        }
        else {
            d_blkSums = nullptr;
        }
        CHECK(cudaMemcpy(d_in, src, nBytes, cudaMemcpyHostToDevice));
        size_t smem = blockSize.x * sizeof(int);
        scanBlkKernelCnt<<<gridSize, blockSize, smem>>>(d_in, n, d_out, d_blkSums, bit);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        if (gridSize.x > 1) {
            // 2. Compute each block's previous sum
            //    by scanning array of blocks' sums
            // TODO
            size_t temp = gridSize.x * sizeof(int);
            int * blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSize.x; i++)
                blkSums[i] += blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSumCnt<<<gridSize.x - 1, blockSize>>>(d_blkSums, d_out, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());

            free(blkSums);
        }

        CHECK(cudaMemcpy(inScan, d_out, nBytes, cudaMemcpyDeviceToHost));
        // scan(hist, nBins, histScan, true, dim3(nBins));
        // TODO: Scatter elements to correct locations

        int rank;
        int nZeros = n - (inScan[n - 1] + ((src[n - 1] >> bit) & 1));
        for (int i = 0; i < n; i++) {
            rank = ((src[i] >> bit) & 1) ? nZeros + inScan[i] : i - inScan[i];
            // bin = src[i] / (1 << (bit)) % (1 << nBits);
            dst[rank] = src[i];
            // histScan[bin]++;
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
    free(inScan);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_blkSums));
    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
}


// Radix Sort
void sort(const uint32_t * in, int n,  uint32_t * out, bool useDevice=false, int blockSize=1) {
    GpuTimer timer;
    timer.Start();

    if (useDevice == false) {
        printf("\nRadix Sort by device implement\n");
        coutingSort3(in, n, out);
    }
    else {// use device
        printf("\nRadix Sort by device\n");
        sortByThrust(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

