#include "../src/helper.cuh"

__global__ void computeHistUseSMem(uint32_t *in, int n, uint32_t *hist, int nBins, int bit) {

    extern __shared__ uint32_t s_hist[];
    if (threadIdx.x < nBins)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *pIn = &in[blockDim.x * blockIdx.x];
    if (i < n) {
        atomicAdd(&s_hist[(pIn[threadIdx.x] >> bit) & (nBins - 1)], 1);
    }
    __syncthreads();
    int width = (n - 1) / blockDim.x + 1;
    if (threadIdx.x < nBins)
        hist[threadIdx.x * width  + blockIdx.x] = s_hist[threadIdx.x];
}


__global__ void scanBlkKernelCnt(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums) {
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0)
        s_data[threadIdx.x] = in[i - 1];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();

    // 2. Each block does scan with data on SMEM
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int neededVal;
        if (threadIdx.x >= stride)
            neededVal = s_data[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            s_data[threadIdx.x] += neededVal;
        __syncthreads();
    }

    // 3. Each block write results from SMEM to GMEM
    if (i < n)
        out[i] = s_data[threadIdx.x];
    if (blkSums != nullptr && threadIdx.x == 0)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}


// TODO: You can define necessary functions here
__global__ void addPrevBlkSumCnt(uint32_t * blkSumsScan, uint32_t * blkScans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}


// TODO: You can define necessary functions here
__global__ void scatter(uint32_t * in, uint32_t * scans, int n, uint32_t *out, int nBins, int bit, int withScan) {
    // TODO
    // ý tưởng đùng SMEM mỗi thread sẽ tự tính rank cho phần tử mình phụ trách
    // Mỗi thread lặp từ 0 đến threadId của mình đếm có bao nhiêu phần tử bằng mình gọi là left
    // rank[threadId] = left ;// gọi là rank nội bộ. dùng rank này cộng với vị trí bắt đầu của digit đang xét có trong mảng scans sẽ ra rank thật sự trong mảng output
    extern __shared__ int s_data[]; // Size: blockDim.x element default = 0
    int *src = s_data;
    int *inScan = &s_data[blockDim.x];
    int *dst = &s_data[blockDim.x * 2];
    int *startPos = &s_data[blockDim.x * 3];
    startPos[0] = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0)
        s_data[threadIdx.x] = in[i - 1];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();
    int numEleInBlock =  blockDim.x;
    // mỗi vòng for thực hiện couting sort với k = 1, trong nội bộ block của mình
    for (size_t i = 0; i <= 2; i ++) {
        // thực hiện exclusive
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int neededVal;
            if (threadIdx.x >= stride)
                neededVal = ((s_data[threadIdx.x - stride]  >> bit) & 1);
            __syncthreads();
            if (threadIdx.x >= stride)
                inScan[threadIdx.x] += neededVal;
            __syncthreads();
        }
        int nZeros = numEleInBlock - (inScan[numEleInBlock - 1] + ((src[numEleInBlock - 1] >> bit) & 1));
        int rank = ((src[threadIdx.x] >> bit) & 1) ? nZeros + inScan[i] : i - inScan[i];
        dst[rank] = src[threadIdx.x]; 
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }
    // tính vị trí bắt đầu
    if (threadIdx.x > 0) {
        int currDigit =  (src[threadIdx.x] >> bit) & (nBins - 1));
        int preDigit = (src[threadIdx.x - 1] >> bit) & (nBins - 1));
        // nếu digit đang set khác với digit trước nó thì set giá trị vào rank vào mảng start pos
        if (currDigit != preDigit) startPos[currDigit] = threadIdx.x;
    }
    // int* left = &s_data[0];
    // left[threadIdx.x] = 0;
    // int begin = blockIdx.x * blockDim.x;
    // int idx = threadIdx.x + begin;
    // if (idx < n) { // nếu vị trí cần xét còn trong mảng hợp lệ
	// 	int digit = (in[idx] >> bit) & (nBins - 1); // lấy digit ở phần tử của thread đang xét;
	// 	int j = begin; // duyệt từ vị trí bắt đầu tới phần tử của thread đang xét
    //     while (j < n && j < idx) {
	// 		int jdigit = (in[j] >> bit) & (nBins - 1); // lấy digit của phần tử đang tính
	// 		if ( digit == jdigit)
	// 			left[threadIdx.x]++;  // không cần syncthreads, hay atomic vì các thread chạy độc lập
	// 		j++; // các biến j là biến cục bộ của mỗi thread, việc cộng thêm ở thread này k ảnh hưởng tới thread khác
	// 	}
    // }

    // if (idx < n) {
	// 	int digit = (in[idx] >> bit) & (nBins - 1);  // lấy digit của phần tử dang xét
    //     int begin_out = scans[digit * withScan + blockIdx.x];//s_scan[digit];//scans[digit * withScan + blockIdx.x];
    //     int rank = begin_out + left[threadIdx.x];
    //     out[rank] = in[idx];
    // }
}

void radixSortLv2V0(const uint32_t * in, int n, uint32_t * out, int k = 2, dim3 blockSize=dim3(512)) {
    dim3 gridSize((n - 1) / blockSize.x + 1);
    // int nBits = k;
    int nBins = 1 << k;

    int nhist = gridSize.x * nBins;
    size_t nBytes = n * sizeof(uint32_t), hByte = nhist * sizeof(uint32_t);
    uint32_t *d_in, *d_hist, *d_scan, *d_blkSums = nullptr, *d_out;

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    CHECK(cudaMalloc(&d_hist, hByte));
    CHECK(cudaMalloc(&d_scan, hByte));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += k) {
        CHECK(cudaMemcpy(d_in, src, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, hByte));
        int smemHist = nBins * sizeof(uint32_t);
        computeHistUseSMem<<<gridSize, blockSize, smemHist>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        dim3 blockSizeScan(512);
        dim3 gridSizeScan((nhist - 1) / blockSizeScan.x + 1);
        int smemScan = blockSizeScan.x  * sizeof(uint32_t);
        if (gridSizeScan.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, sizeof(uint32_t) * gridSizeScan.x));
        }
        scanBlkKernelCnt<<<gridSizeScan, blockSizeScan, smemScan>>>(d_hist, nhist, d_scan, d_blkSums);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        if (gridSizeScan.x > 1) {
            // 2. Compute each block's previous sum
            //    by scanning array of blocks' sums
            size_t temp =  gridSizeScan.x * sizeof(uint32_t);
            int * h_blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(h_blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSizeScan.x; i++)
                h_blkSums[i] += h_blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, h_blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSumCnt<<<gridSizeScan.x - 1, blockSizeScan>>>(d_blkSums, d_scan, nhist);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            free(h_blkSums);
        }

        int smemScatter = blockSize.x * sizeof(uint32_t);
        scatter<<<gridSize, blockSize, smemScatter>>>(d_in, d_scan, n, d_out, nBins, bit, gridSize.x);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(dst, d_out , nBytes, cudaMemcpyDeviceToHost));
        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }
    // Copy result to out
    memcpy(out, src, nBytes);

    free(originalSrc);

    // Free memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));
}