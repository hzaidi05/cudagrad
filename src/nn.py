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
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
            
        act = x @ self.w + self.b
        
        return x.relu(act)
    
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
        self.device = device
        layer_sizes = [nin] + nouts
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i+1]
            
            self.layers.append(Dense(in_features, out_features, device=device))
    
    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
            
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# A simple dense layer that handles dimensions correctly
class Dense(Module):
    def __init__(self, in_features, out_features, device='CPU'):
        self.device = device
        
        # Initialize with correct shape: (in_features, out_features)
        if device == 'GPU':
            self.W = Tensor(cp.random.randn(in_features, out_features) * 0.1, device=device)
            self.b = Tensor(cp.zeros(out_features), device=device)
        else:
            self.W = Tensor(np.random.randn(in_features, out_features) * 0.1, device=device)
            self.b = Tensor(np.zeros(out_features), device=device)
    
    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, device=self.device)
            
        # Linear transformation
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]

# Example Usage
if __name__ == "__main__":
    x = Tensor(np.random.randn(3, 2), device='GPU')  # Batch of 3 samples, 2 features each
    model = MLP(2, [3, 1], device='GPU')
    y = model(x)
    print("y.data:", y.data)
