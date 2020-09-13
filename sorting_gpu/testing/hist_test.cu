#include "../src/hist.cuh"

int main(int argc, char ** argv) {
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE AND THE NUMBER OF BINS
    int n = (1 << 24) + 1;
    int nBins = 256;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    int * in = (int *)malloc(n * sizeof(int));
    int * hist = (int *)malloc(nBins * sizeof(int)); // Device result
    int * correctHist = (int *)malloc(nBins * sizeof(int)); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = (int)(rand() & (nBins - 1)); // random int in [0, nBins-1]

    // DETERMINE BLOCK SIZE
    dim3 blockSize(512); // Default
    if (argc == 2)
        blockSize.x = atoi(argv[1]);

    // COMPUTE HISTOGRAM BY HOST
    computeHist(in, n, correctHist, nBins);
    
    // COMPUTE HISTOGRAM BY DEVICE, KERNEL 1
    computeHist(in, n, hist, nBins, true, blockSize, 1);
    checkCorrectness(hist, correctHist, nBins);

    // COMPUTE HISTOGRAM BY DEVICE, KERNEL 2
    memset(hist, 0, nBins * sizeof(int)); // Reset output
    computeHist(in, n, hist, nBins, true, blockSize, 2);
	checkCorrectness(hist, correctHist, nBins);

    // FREE MEMORIES
    free(in);
    free(hist);
    free(correctHist);
    
    return EXIT_SUCCESS;
}