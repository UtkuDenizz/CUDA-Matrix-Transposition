#include <stdio.h>
#include <cuda_runtime.h>

// sizes of matris
#define ROWS 2048
#define COLS 1024
#define TILE_DIM 16
// naive gpu kernel - slow
__global__ void transposeNaive(float *in, float *out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // strided write  - slow 
        out[x * rows + y] = in[y * cols + x];
    }
}

// optimized gpu kernel - fast
__global__ void transposeOptimized(float *in, float *out, int rows, int cols) {
    // hafiza bankasi çakışmalarını (bank conflicts) önlemek için +1 eklenebilir [TILE_DIM][TILE_DIM+1]
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    //  read from global memory to shared memory  (yan yana threadler yan yana okur - coalesced)
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];

    __syncthreads(); // make sure all the threads are finished writing 

    // calculate transposed coordinates and rotate blocks
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // write from shared memory to global memory (yan yana threadler yan yana yazar - coalesced)
    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int size = ROWS * COLS * sizeof(float);
    float *data_in, *data_out;

    // 2. section -  unified memory (cudaMallocManaged)
    // cpu and gpu can use this adress
    cudaMallocManaged(&data_in, size);
    cudaMallocManaged(&data_out, size);

    // random data
    for (int i = 0; i < ROWS * COLS; i++) data_in[i] = (float)i;

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((COLS + TILE_DIM - 1) / TILE_DIM, (ROWS + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    //  naive test 
    cudaEventRecord(start);
    transposeNaive<<<dimGrid, dimBlock>>>(data_in, data_out, ROWS, COLS);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float timeNaive = 0;
    cudaEventElapsedTime(&timeNaive, start, stop);

    // optimized test
    cudaEventRecord(start);
    transposeOptimized<<<dimGrid, dimBlock>>>(data_in, data_out, ROWS, COLS);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float timeOpt = 0;
    cudaEventElapsedTime(&timeOpt, start, stop);

    printf("Matrix: %d x %d\n", ROWS, COLS);
    printf("Naive GPU Time:     %.3f ms\n", timeNaive);
    printf("Optimized GPU Time: %.3f ms\n", timeOpt);
    printf("Speedup:            %.2fx\n", timeNaive / timeOpt);

    cudaFree(data_in);
    cudaFree(data_out);
    return 0;
}
