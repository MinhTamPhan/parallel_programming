#include "../src/scan.cu"

int main(int argc, char ** argv) {
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    int * out = (int *)malloc(bytes); // Device result
    int * correctOut = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = (int)(rand() & 0xFF) - 127; // random int in [-127, 128]

    // DETERMINE BLOCK SIZE
    dim3 blockSize(512); 
    if (argc == 2)
    {
        blockSize.x = atoi(argv[1]);
    }

    // SCAN BY HOST
    scan(in, n, correctOut, false, blockSize);
    
    // SCAN BY DEVICE
    scan(in, n, out, true, blockSize);
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}