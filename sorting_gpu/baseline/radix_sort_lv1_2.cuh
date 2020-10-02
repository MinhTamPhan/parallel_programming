#include "../src/helper.cuh"

void radixSort(const uint32_t * in, int n, uint32_t * out, int nBits, int blockSizes)
{
    dim3 blockSize(blockSizes);
    dim3 gridSize((n - 1) / blockSize.x + 1); // grid size for histogram kernel 
    // TODO
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * gridSize.x * sizeof(int));
    int *histScan = (int * )malloc(nBins * gridSize.x * sizeof(int));
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
        memset(hist, 0, nBins * gridSize.x * sizeof(int));
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < blockSize.x; j++)
            if (i * blockSize.x + j < n)
            {
                int bin = (src[i * blockSize.x + j] >> bit) & (nBins - 1);
                hist[i * nBins + bin]++;
            }
        }

        // TODO: scan
        int pre = 0;
        for (int j = 0; j < nBins; j++){
            for (int i = 0; i < gridSize.x; i++)
            {
                histScan[i * nBins + j] = pre;
                pre = pre + hist[i * nBins + j];
            }
        }

        // TODO: Scatter
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < blockSize.x; j++)
            {
                int id = i * blockSize.x + j;
                if (id < n)
                {
                    int bin = i * nBins + ((src[id] >> bit) & (nBins - 1));
                    dst[histScan[bin]] = src[id];
                    histScan[bin]++;  // (neu cung bin thi ghi ben canh)
                }
            }
        }
        // TODO: Swap "src" and "dst"
        uint32_t * temp = src;
        src = dst;
        dst = temp; 
    }
    // TODO: Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));
    // Free memories
    free(hist);
    free(histScan);
    free(originalSrc);
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
        sortByThrust(in, n, out, blockSize);
    	
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}