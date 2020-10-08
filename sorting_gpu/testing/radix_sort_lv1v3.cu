#include  "../baseline/radix_sort_lv1_v3.cuh"

void hist(uint32_t * in, int n, uint32_t *hist, int bit, int nBins) {
    int bin;
    for (int i = 0; i < n; i++) {
        bin = (in[i] >> bit) & (nBins - 1);
        hist[bin]++;
    }
}

void transposeHist(uint32_t * in, int h, int w, uint32_t * out) {
    int i, j;
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
            out[i * h + j] = in[j * w + i];
}


void scatterH(uint32_t * in, uint32_t * const scans , int n, uint32_t *out, int nBins, int bit, int block){
    uint32_t* left = new uint32_t[49];
    int begin = block * 49;
    memset(left, 0, 49 );
    for (int i = 0; i < 49; ++i) {
        int idx = i + begin;
        if (idx < n) {
            int digit = (in[idx] >> bit) & (nBins - 1);
            int j = begin;
            while (j < n && j < idx){
                int jdigit = (in[j] >> bit) & (nBins - 1);
                if ( digit == jdigit)
                    left[i]++;
                j++;
            }
        }
    }


    for (int i = 0; i < 49; ++i) {
        if (begin + i < n) {
            // lấy digit đang xét;
            int digit = (in[begin + i] >> bit) & (nBins - 1);

            int preRank = scans[digit  * 3 + block];
            int rank = left[i] + preRank;
            out[rank] = in[begin + i];
        }
    }

}

int main(int argc, char ** argv) {
    
	int n = (1 << 24) + 1;
    // n = 100;
	int k = 8;
	dim3 blockSize = dim3(512);
	if (argc >= 2){
		blockSize = atoi(argv[1]);
    }
    size_t bytes = n * sizeof(uint32_t);
	uint32_t * in =  (uint32_t *)malloc(bytes);
	uint32_t * outImp =  (uint32_t *)malloc(bytes);
	uint32_t * outThrus = (uint32_t *)malloc(bytes);

	int nLoop = 20;
	GpuTimer timer;
	float time;
	float avgTimeImp = 0, avgThrus = 0;
    int loop = 0;
	while(loop < nLoop) {
		for (int i = 0; i < n; i++)
			in[i] = rand();
		printf("radixSortLv1 my implement Internal transpose + SMEM  scatter.Input size: %d, k = %d, nLoop = %d\n\n\n", n, k, loop + 1);
		timer.Start();
		radixSortLv1V2(in, n, outImp, k, blockSize);
        
		timer.Stop();
		time = timer.Elapsed();
		avgTimeImp += time / nLoop;
		timer.Start();
		sortByThrust(in, n, outThrus);
        timer.Stop();
		time = timer.Elapsed();
		// printf("Time sortByThrust: %.3f ms\n",time);
		avgThrus += time / nLoop;
		checkCorrectness(outImp, outThrus, n);
        loop++;
	}

    printf("================================ avg time after %d run================================ \n", nLoop);
	printf("avgTimeImp: %f\n", avgTimeImp);
	printf("avgThrus: %f\n", avgThrus);

    // FREE MEMORIES
    free(in);
    free(outImp);
    free(outImp);
    free(outThrus);
    return EXIT_SUCCESS;
}