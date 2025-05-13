import torch
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from src.tensor import Tensor
from src.nn import MLP
from google.colab import files


def benchmark_inference(batch_size=1000, input_size=128, device='CPU', iterations=10):
    x_np = np.random.randn(batch_size, input_size)
    x_cp = cp.asarray(x_np) if device == 'GPU' else x_np
    
    model = MLP(input_size, [64, 32, 16, 1], device=device)
    
    # Warm-up
    if device == 'GPU':
        for _ in range(5):
            _ = model(x_cp)
            cp.cuda.Device(0).synchronize()
    
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        y = model(x_cp)
        if device == 'GPU':
            cp.cuda.Device(0).synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
    
    return total_time / iterations

def test_sanity_check():

    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

if __name__ == "__main__":
    batch_sizes = [32, 100, 500, 512, 1000, 5000, 10000]
    cpu_times = [benchmark_inference(bs, 128, 'CPU', 10) for bs in batch_sizes]
    gpu_times = [benchmark_inference(bs, 128, 'GPU', 10) for bs in batch_sizes]
    
    plt.plot(batch_sizes, cpu_times, label='CPU', marker='o')
    plt.plot(batch_sizes, gpu_times, label='GPU', marker='s')
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Time (s)")
    plt.title("CPU vs GPU Inference Performance")
    plt.legend()
    #plt.show()
    plt.savefig("abc.png")
    files.download("abc.png") 