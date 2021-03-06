#include "helper.cuh"
/*
Scan within each block's data (work-inefficient), write results to "out", and 
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums) {   
    // TODO
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        s_data[threadIdx.x] = in[i];
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

__global__ void scanBlkKernelEx(int * in, int n, int * out, int * blkSums) {   
    // TODO
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
__global__ void addPrevBlkSum(int * blkSumsScan, int * blkScans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
    if (i < n)
        blkScans[i] += blkSumsScan[blockIdx.x];
}


void scan(int * in, int n, int * out, bool useDevice=false, dim3 blkSize=dim3(1)) {
    GpuTimer timer; 
    timer.Start();
    if (useDevice == false) {
    	printf("\nScan by host\n");

		out[0] = in[0];
	    for (int i = 1; i < n; i++)
	        out[i] = out[i - 1] + in[i];
	    
    }
    else { // Use device
    	printf("\nScan by device\n");
        // 1. Scan locally within each block, 
        //    and collect blocks' sums into array
        
        int * d_in, * d_out, * d_blkSums;
        size_t nBytes = n * sizeof(int);
        CHECK(cudaMalloc(&d_in, nBytes)); 
        CHECK(cudaMalloc(&d_out, nBytes)); 
        dim3 gridSize((n - 1) / blkSize.x + 1);
        if (gridSize.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
        }
        else {
            d_blkSums = nullptr;
        }
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));

        size_t smem = blkSize.x * sizeof(int);
        scanBlkKernel<<<gridSize, blkSize, smem>>>(d_in, n, d_out, d_blkSums);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        if (gridSize.x > 1) {
            // 2. Compute each block's previous sum 
            //    by scanning array of blocks' sums
            // TODO
            size_t temp = gridSize.x * sizeof(int);
            int * blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSize.x; i++)
                blkSums[i] += blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSum<<<gridSize.x - 1, blkSize>>>(d_blkSums, d_out, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            
            free(blkSums);
        }

        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(d_blkSums));
	}
    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
}



void scanExclusive(int * in, int n, int * out, bool useDevice=false, dim3 blkSize=dim3(1)) {
    GpuTimer timer; 
    timer.Start();
    if (useDevice == false) {
    	printf("\nScan by host\n");

		out[0] = 0;
	    for (int i = 1; i < n; i++)
	        out[i] = out[i - 1] + in[i - 1];
	    
    }
    else { // Use device
    	printf("\nScan by device\n");
        // 1. Scan locally within each block, 
        //    and collect blocks' sums into array
        
        int * d_in, * d_out, * d_blkSums;
        size_t nBytes = n * sizeof(int);
        CHECK(cudaMalloc(&d_in, nBytes)); 
        CHECK(cudaMalloc(&d_out, nBytes)); 
        dim3 gridSize((n - 1) / blkSize.x + 1);
        if (gridSize.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
        }
        else {
            d_blkSums = nullptr;
        }
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));

        size_t smem = blkSize.x * sizeof(int);
        scanBlkKernelEx<<<gridSize, blkSize, smem>>>(d_in, n, d_out, d_blkSums);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        if (gridSize.x > 1) {
            // 2. Compute each block's previous sum 
            //    by scanning array of blocks' sums
            // TODO
            size_t temp = gridSize.x * sizeof(int);
            int * blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSize.x; i++)
                blkSums[i] += blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSum<<<gridSize.x - 1, blkSize>>>(d_blkSums, d_out, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            
            free(blkSums);
        }

        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(d_blkSums));
	}
    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
}


__global__ void scanBlkKernelCnt(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums, int bit) {   
    // TODO
    // 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size: blockDim.x element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0)
        s_data[threadIdx.x] = ((in[i - 1] >> bit) & 1);
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



void scanExclusiveCounting(uint32_t * in, int n, uint32_t * out, int bit, bool useDevice=false, dim3 blkSize=dim3(1)) {
    GpuTimer timer; 
    timer.Start();
    if (useDevice == false) {
    	printf("\nScan by host\n");

		out[0] = 0;
	    for (int i = 1; i < n; i++)
	        out[i] = out[i - 1] + ((in[i - 1] >> bit) & 1);
	    
    }
    else { // Use device
    	// printf("\nScan by device\n");
        // 1. Scan locally within each block, 
        //    and collect blocks' sums into array
        
        uint32_t * d_in, * d_out, * d_blkSums;
        size_t nBytes = n * sizeof(uint32_t);
        CHECK(cudaMalloc(&d_in, nBytes)); 
        CHECK(cudaMalloc(&d_out, nBytes)); 
        dim3 gridSize((n - 1) / blkSize.x + 1);
        if (gridSize.x > 1) {
            CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
        }
        else {
            d_blkSums = nullptr;
        }
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));

        size_t smem = blkSize.x * sizeof(int);
        scanBlkKernelCnt<<<gridSize, blkSize, smem>>>(d_in, n, d_out, d_blkSums, bit);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        if (gridSize.x > 1) {
            // 2. Compute each block's previous sum 
            //    by scanning array of blocks' sums
            // TODO
            size_t temp = gridSize.x * sizeof(int);
            int * blkSums = (int*)malloc(temp);
            CHECK(cudaMemcpy(blkSums, d_blkSums, temp, cudaMemcpyDeviceToHost));
            for (int i = 1; i < gridSize.x; i++)
                blkSums[i] += blkSums[i-1];
            CHECK(cudaMemcpy(d_blkSums, blkSums, temp, cudaMemcpyHostToDevice));

            // 3. Add each block's previous sum to its scan result in step 1
            addPrevBlkSumCnt<<<gridSize.x - 1, blkSize>>>(d_blkSums, d_out, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            
            free(blkSums);
        }

        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(d_blkSums));
	}
    timer.Stop();
    // printf("Processing time: %.3f ms\n", timer.Elapsed());
}



// void checkCorrectness(int * out, int * correctOut, int n) {
//     for (int i = 0; i < n; i++)
//     {
//         if (out[i] != correctOut[i])
//         {
//             printf("INCORRECT :(\n");
//             return;
//         }
//     }
//     printf("CORRECT :)\n");
// }
