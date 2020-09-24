#include "../src/scan.cuh"
#include <stdio.h>
__constant__ int nZeros;

void rankCal(int * in, int n, int * out, int * inscan, bool useDevice=false, dim3 blkSize=dim3(1)) {
    GpuTimer timer; 
    timer.Start();
    if (useDevice == false) {
    	printf("\nRank by host\n");
	    for (int i = 0; i < n; i++)
            if(in[i]==0)
                out[i] = i - inscan[i];
            else
	            out[i] = nZeros + inscan[i];
	    
    }
    else { // Use device
    	printf("\nRank by device\n");

	}
    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
}

int main(int argc, char ** argv) {

    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 511; // For test by eye
    //int n = (1 << 24) + 1;
    int n = (1 << 10) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    int * inscan = (int *)malloc(bytes); // Device result
    int * correctinscan = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand() % 2;
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    scanExclusive(in, n, correctinscan, false, blockSize);
    //printArray(correctinscan, n); // For test by eye
    
    // SORT BY DEVICE
    scanExclusive(in, n, inscan, true, blockSize);
    //printArray(inscan, n); // For test by eye
    // checkCorrectness(inscan, correctinscan, n);

    //COUNT ZERO IN ARRAY IN

    int nZeros = n - inscan[n-1] - in[n-1];
    printf("\nZerros in array in: %d\n", nZeros);

    int * rank = (int *)malloc(bytes);
    int * correctrank = (int *)malloc(bytes);

    //RANK BY HOST
    rankCal(in, n, correctrank, inscan, false, blockSize);

    //RANK BY DEVICE
    rankCal(in, n, rank, inscan, true, blockSize);

    // FREE MEMORIES
    free(in);
    // free(out);
    // free(correctOut);
    
    return EXIT_SUCCESS;
}