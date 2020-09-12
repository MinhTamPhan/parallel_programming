// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n, uint32_t * out) {

    int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
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
        for(int i = 0; i < n; i++) {
            bin = src[i] / (1 << (bit)) % (1 << nBits);
            //bin == (src[i] >> bit) & (nBins - 1);
            hist[bin] += 1;
        }
        // TODO: Scan histogram (exclusively)
        //memset(histScan, 0, nBins * sizeof(int));
        histScan[0] = 0;
        for(int i = 1; i < nBins; i++)
            histScan[i] = hist[i - 1] + histScan[i - 1];

        // TODO: Scatter elements to correct locations
        for (int i = 0; i < n; i++) {
            bin = src[i] / (1 << (bit)) % (1 << nBits);
            //bin = (src[i] >> bit) & (nBins - 1);
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
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize) {
// TODO
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
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != correctOut[i]) {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n) {
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}
