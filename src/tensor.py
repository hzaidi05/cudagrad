import numpy as np
import cupy as cp

matmul_kernel = cp.RawKernel(r'''
extern "C" __global__ void matmul(const int* __restrict__ a, const int* __restrict__ b, int* c, int N){
    int row = blockIdx.x;
    int col = blockIdx.y;
    int TM = 4; //todo: make modifiable
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
''', 'matmul')

class Tensor:
    def __init__(self, data, device='CPU', requires_grad=True):
        if device == 'GPU':
            self.data = cp.array(data, dtype=cp.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.grad = cp.zeros_like(self.data) if device == 'GPU' else np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.device = device
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, device=self.device)
        
        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            if self.data.shape != out.data.shape:
                grad_self = grad_self.sum(axis=tuple(range(grad_self.ndim - self.data.ndim)), keepdims=True)
            if other.data.shape != out.data.shape:
                grad_other = grad_other.sum(axis=tuple(range(grad_other.ndim - other.data.ndim)), keepdims=True)
            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data * other.data, device=self.device)
        
        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            if self.data.shape != out.data.shape:
                grad_self = grad_self.sum(axis=tuple(range(grad_self.ndim - self.data.ndim)), keepdims=True)
            if other.data.shape != out.data.shape:
                grad_other = grad_other.sum(axis=tuple(range(grad_other.ndim - other.data.ndim)), keepdims=True)
            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if self.device == 'GPU':
            M, N = self.data.shape
            N2, K = other.data.shape
            
            # Check dimension compatibility
            if N != N2:
                raise ValueError(f"Incompatible dimensions for matrix multiplication: {self.data.shape} and {other.data.shape}")
                
            out_data = cp.zeros((M, K), dtype=cp.float32)
            threads_per_block = (16, 16)
            blocks_per_grid = ((K + 15) // 16, (M + 15) // 16)
            
            # Call the CUDA kernel
            matmul_kernel((blocks_per_grid), (threads_per_block), (
                M, N, K, 1.0, self.data, other.data, 1.0, out_data)
            )
            
            out = Tensor(out_data, device='GPU')
        else:
            out = Tensor(self.data @ other.data, device='CPU')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)
        self.grad = cp.ones_like(self.data) if self.device == 'GPU' else np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
