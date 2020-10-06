#include <bits/stdc++.h>
using namespace std;

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

void printArray(const uint32_t * a, int n) {
    for (int i = 0; i < n; i++)
        printf("%4i ", a[i]);
    printf("\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != correctOut[i]) {

            printf("INCORRECT :( out = %d, correctOut = %d\n", out[i] , correctOut[i]);
            return;
        }
    }
    printf("CORRECT :)\n");
}

void scatter(uint32_t * in, uint32_t * const scans , int n, uint32_t *out, int nBins, int bit, int block){
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

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    n = 100;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    int k = 4;
    int nBins = 1 << k;
    uint32_t * in = new uint32_t[n];
    uint32_t * out = new uint32_t[n]; // Device result
    uint32_t * histOut = new uint32_t[nBins * 3];
    uint32_t * histTranspose =  new uint32_t[nBins * 3];
    uint32_t * histScan = new uint32_t[nBins * 3];
    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    uint32_t * src = new uint32_t[n];
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        memset(histOut, 0 , 3 * nBins * sizeof(int));
        hist(src, 49, histOut, bit, nBins);
        hist(&src[49], 49, histOut + nBins, bit, nBins);
        hist(&src[49 * 2], 2, histOut + nBins * 2, bit, nBins);
        transposeHist(histOut, 3, nBins, histTranspose); // 3 dòng nBins cột

        histScan[0] = 0;
        for (int bin = 1; bin < nBins * 3; bin++)
            histScan[bin] = histScan[bin - 1] + histTranspose[bin - 1];

        scatter(src, histScan, n, dst, nBins, bit, 0);
        scatter(src, histScan, n, dst, nBins, bit, 1);
        scatter(src, histScan, n, dst, nBins, bit, 2);
        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }


    // Copy result to out
    memcpy(out, src, n * sizeof(uint32_t));

    sort(in, in + n);
//    printArray(out, 100);
     checkCorrectness(out, in, n);

    // FREE MEMORIES
    delete in;
    delete out;
    delete histOut;
    delete histTranspose;
    delete histScan;
    delete originalSrc;

    return 0;
}