#include "../src/helper.cuh"

__global__ void scanBlkKernelEx(int * in, int n, int * out, int * blkSums) {   
    // TODO
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
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



// parallel counting Sort
void countingSortByDevice(const uint32_t * in, int n, uint32_t * out) {

    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));
    int * histScan2 = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
    // In each loop, sort elements according to the current digit from src to dst 
    // (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
        // TODO: Compute histogram
        memset(hist, 0, nBins * sizeof(int));
        int bin;
         // TODO: Compute histogram by device
        // for (int i = 0; i < n; i++) {
        //     bin = (src[i] >> bit) & (nBins - 1);
        //     // bin = src[i] / (1 << (bit)) % (1 << nBits);
        //     hist[bin]++;
        // }
        dim3 blockSize(512); // Default
        computeHistDevice(src, n, hist, nBins, true, blockSize, 1, bit);
        

        // TODO: Scan histogram (exclusively)
        // TODO: Compute histogram by device
        histScan[0] = 0;
        // for (int bin = 1; bin < nBins; bin++)
        //     histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        scanExclusive(hist, nBins, histScan);
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
}
