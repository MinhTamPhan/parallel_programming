#include "../src/helper.cuh"
#include "../baseline/radix_sort_lv1_2.cuh"

int main(int argc, char ** argv) {

    // PRINT OUT DEVICE INFO
    printDeviceInfo();

	// SET UP INPUT SIZE
    int nBits = 1;
    int n = (1 << 24) + 1;
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nInput size: %d\n", n);
    printf("nBits = %d\n", nBits);
    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
       in[i] = rand();    
	
	// DETERMINE BLOCK SIZES
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY DEVICE
    sort(in, n, correctOut, nBits, blockSize);
    // SORT BY DIVECE(THRUST)
	sort(in, n, out, nBits, blockSize, true);
	checkCorrectness(out, correctOut, n);
    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}