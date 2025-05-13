import random
import numpy as np
import cupy as cp
from tensor import Tensor

class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = cp.zeros_like(p.data) if p.device == 'GPU' else np.zeros_like(p.data)

class Neuron(Module):
    def __init__(self, nin, device='CPU'):
        self.device = device
        self.w = Tensor(np.random.randn(nin, 1) * 0.1, device=device)
        self.b = Tensor(np.zeros((1, 1)), device=device)
    
    def __call__(self, x):
        out = self.w.T @ x + self.b  # Supports batched inputs
        return out.relu()
    
    def parameters(self):
        return [self.w, self.b]

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, device='CPU'):
        sizes = [nin] + nouts
        self.layers = [Neuron(sizes[i], device=device) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Example Usage
if __name__ == "__main__":
    x = Tensor(np.random.randn(3, 2), device='GPU')  # Batch of 3 samples, 2 features each
    model = MLP(2, [3, 1], device='GPU')
    y = model(x)
    print("y.data:", y.data)
