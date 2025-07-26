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

/*
Kernels it took to build up to these; todo: detail how to get from naive -> optimized for new readers!

__global__ void matmul_naive(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp = 0;
    //if row < n && col < n
    for(int i = 0; i<N; i++){
        tmp += A[row * N + i] * B[i*N + col];
    }
    C[row*N + col] = tmp;
}

__global__ void matmul_coalesce(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
    int row = blockIdx.x * blocksize + threadIdx.x / blocksize;
    int col = blockIdx.y * blocksize + threadIdx.y % blocksize;

    int tmp = 0;
    //if row < n && col < n
    for(int i = 0; i<N; i++){
        tmp += A[row * N + i] * B[i*N + col];
    }
    C[row*N + col] = tmp;
}
//launch param: blockDim: 1024 (1D)

__global__ void matmul_shared(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
    int row = blockIdx.x;
    int col = blockIdx.y;
    __shared__ int sA[blocksize*blocksize];
    __shared__ int sB[blocksize*blocksize];

    a += row * blocksize * N;
    b += col * blocksize;
    c += row * blocksize * N + col * blocksize;

    int tRow = threadIdx.x / blocksize;
    int tCol = threadIdx.x % blocksize;

    int tmp = 0;
    //if check
    //load into shared mem

    for(int i=0; i<N; i+= blocksize){
        sA[tRow * blocksize + tCol] = A[trow * N + tCol];
        sB[tRow * blocksize + tCol] = B[trow * N + tCol];
    
        __syncthreads();

        //do dot product

        for(int i = 0; i<blocksize; i++){
            tmp += sA[tRow * blocksize + i] * sB[i*blocksize + tCol];
        }

        __syncthreads();

        //move to next point
        A += blocksize;
        B += blocksize * N;
    }

    C[tRow * N + tCol] = tmp;


}
#define TM 4
__global__ void matmul_blocktiled(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
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
            //tmp += sA[tRow * blocksize + i] * sB[i*blocksize + tCol];
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


__global__ void reduction_naive(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = gdata[tid];
    __syncthreads();

    for(int s=1; s < blockDim.x; s*=2){
        if(threadIdx.x % (2 * s) == 0){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        gdata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_nodiv(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = gdata[tid];
    __syncthreads();

    for(int s=1; s < blockDim.x; s*=2){
        int idx = threadIdx.x * s * 2;
        if(idx < blockDim.x){
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        gdata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_noconf(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = gdata[tid];
    __syncthreads();

    for(int s=blockDim.x / 2; s > 0; s/=2){
        int idx = threadIdx.x * s * 2;
        if(idx < blockDim.x){
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        gdata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_noidle(int* gdata, int N){ //halve the number of blocks launched (so grid size)
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[threadIdx.x] = gdata[i] + gdata[i + blockDim.x];
    __syncthreads();

    for(int s=blockDim.x / 2; s > 0; s/=2){
        int idx = threadIdx.x * s * 2;
        if(idx < blockDim.x){
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        gdata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_unroll_onepass(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[threadIdx.x] = gdata[i] + gdata[i + blockDim.x];
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

__global__ void reduction_batch(int* gdata, int N){
    __shared__ int sdata[SHMEM_SIZE];
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = 12;
    int sum = 0;
    
    for (int j = 0; j < batchSize; j++) {
        if (tid * batchSize + j < N) {
            sum += d_in[tid * batchSize + j];
        }
    }
    sums[threadIdx.x] = sum;
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

__global__ void reduction_vec(int* gdata, int N){
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
}*/