#include <cuda_runtime.h>

#define TM 4
__global__ void matmul(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
    int row = blockIdx.x;
    int col = blockIdx.y;
    __shared__ int sA[blocksize*blocksize];
    __shared__ int sB[blocksize*blocksize];

    a += row * blocksize * N;
    b += col * blocksize;
    c += row * blocksize * N + col * blocksize;

    int tRow = threadIdx.x / blocksize;
    int tCol = threadIdx.x % blocksize;

    int tmp[TM] = 0;
    int Btmp = 0;
    //if check
    //load into shared mem

    for(int i=0; i<N; i+= blocksize){
        sA[tRow * blocksize + tCol] = A[trow * N + tCol];
        sB[tRow * blocksize + tCol] = B[trow * N + tCol];
    
        __syncthreads();

        //do dot product

        for(int i = 0; i<blocksize; i++){
            Btmp = sB[i*blocksize + tCol];
            for(j=0; j<TM; j++){
                result[j] += Btmp * sA[(tRow * TM + j)*blocksize + i];
            }
        }

        __syncthreads();

        //move to next point
        A += blocksize;
        B += blocksize * N;
    }

    for(i=0; i<TM; i++){
        C[(tRow * TM + i)*blocksize + tCol] = tmp[i];
    }

}

__global__ void reduction(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = 12;
    int sum = 0;
    
    for (int i = 0; i < batchSize / 4; i++) {
        const int4 val = gdata[tid * (batchSize / 4) + i];
        if (tid * batchSize + i * 4 < N) {
            sum += val.x + val.y + val.z + val.w;
        }
    }
    sums[tid] = sum;
    __syncthreads();


    for(int s=blockDim.x / 2; s > 32; s/=2){
        int idx = threadIdx.x * s * 2;
        if(idx < blockDim.x){
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    volatile int *volatile_sums = sums;
    for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
        if (tid < activeThreads) {
            volatile_sums[tid] += volatile_sums[tid + activeThreads];
        }
        __syncwarp();
    }

    if(threadIdx.x == 0){
        atomicAdd(gdata, volatile_sums[0]);
    }
}

