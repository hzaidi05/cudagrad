import numpy as np
import cupy as cp

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
            #todo
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
