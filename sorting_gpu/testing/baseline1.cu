#include "../src/helper.cuh"
#include "../baseline/sorting_cpu.cuh"

int main(int argc, char ** argv) {

    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 511; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
   
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);
    double timeH = 0, timeD = 0;
    for (int i = 0; i < 20; i++) {
        for (int i = 0; i < n; i++)
            //in[i] = rand() % 255; // For test by eye
            in[i] = rand();

        // SORT BY HOST
        timeH += sort(in, n, correctOut);
        //printArray(correctOut, n); // For test by eye
        
        // SORT BY DEVICE
        timeD += sort(in, n, out, true, blockSize);
        //printArray(out, n); // For test by eye
        checkCorrectness(out, correctOut, n);
    }
    printf("================================ avg time after %d run================================ \n", 20);
	printf("avgTime host Imp (hist + scan parallel): %f ms\n", timeH / 20.0);
	printf("avgThrus: %f ms\n", timeD/20.0);
    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}