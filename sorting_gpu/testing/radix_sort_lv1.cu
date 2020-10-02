#include "../src/helper.cuh"
#include "../baseline/radix_sort_lv1.cuh"

void hist(uint32_t * in, int n, uint32_t *hist, int bit, int nBins) {
    int bin;
    for (int i = 0; i < n; i++) {
        bin = (in[i] >> bit) & (nBins - 1);
        hist[bin]++;
    }
}

void histTranspose(uint32_t * in, int h, int w, uint32_t * out) {
    int i, j; 
    for (i = 0; i < w; i++) 
        for (j = 0; j < h; j++) 
            out[i * h + j] = in[j * w + i];
}

// __global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height) {
//     // Handle to thread block group
//     __shared__ float tile[TILE_DIM][TILE_DIM+1];

//     int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
//     int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
//     int index_in = xIndex + (yIndex)*width;

//     xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//     yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//     int index_out = xIndex + (yIndex)*height;

//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//     {
//         tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
//     }

//     __syncthreads();


//     for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//         odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
//     }
// }

int main(int argc, char ** argv) {

    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    n = 100;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    int nBins = 1 << 2;
    size_t bytes = n * sizeof(uint32_t);
    size_t hByte = nBins * sizeof(uint32_t) * 3;
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(hByte); 
    uint32_t * correctOutTranspose = (uint32_t *)malloc(hByte); 
    memset(correctOut, 0 , n);
    // SET UP INPUT DATA
    // for (int i = 0; i < n; i++)
    //     in[i] = rand();

    // hist(in, 49, correctOut, 0, nBins);
    // hist(&in[49], 49, correctOut + nBins, 0, nBins);
    // hist(&in[49 * 2], 2, correctOut + nBins * 2, 0, nBins);
    // // printArray(correctOut, nBins * 3);
    // histTranspose(correctOut, 3, 4, correctOutTranspose);
    // // printArray(correctOutTranspose, nBins * 3);

    // uint32_t * histScan = (uint32_t *)malloc(nBins * 3); 
    // histScan[0] = 0;
    // for (int bin = 1; bin < 12; bin++)
    //     histScan[bin] = histScan[bin - 1] + ((correctOutTranspose[bin - 1] >> 0) & (nBins - 1));    

    // printArray(histScan, 12);

    radixSortLv1NoShared(in, n, out, 2);
    
    printArray(out, 100);
    // checkCorrectness(out, histScan, 12);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}