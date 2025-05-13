import numpy as np
import cupy as cp
from src.tensor import Tensor
from src.nn import MLP
import urllib.request
import gzip
import os

def download_mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    for name, file in files.items():
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file)

def load_mnist():
    download_mnist()
    
    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0
    
    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    train_images = read_images('train-images-idx3-ubyte.gz')
    train_labels = read_labels('train-labels-idx1-ubyte.gz')
    test_images = read_images('t10k-images-idx3-ubyte.gz')
    test_labels = read_labels('t10k-labels-idx1-ubyte.gz')
    
    return train_images, train_labels, test_images, test_labels

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def train_mnist():
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Convert to one-hot encoding
    train_labels_one_hot = one_hot_encode(train_labels)
    test_labels_one_hot = one_hot_encode(test_labels)
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.01
    epochs = 5
    
    # Create model
    model = MLP(784, [128, 64, 10], device='GPU')  # 784 input features (28x28), 10 output classes
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels_one_hot = train_labels_one_hot[indices]
        
        # Mini-batch training
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels_one_hot[i:i+batch_size]
            
            # Forward pass
            x = Tensor(batch_images, device='GPU')
            y = Tensor(batch_labels, device='GPU')
            
            # Get predictions
            out = model(x)
            
            # Compute loss (cross entropy)
            loss = -cp.sum(y.data * cp.log(cp.clip(out.data, 1e-7, 1-1e-7))) / batch_size
            
            # Backward pass
            model.zero_grad()
            out.grad = (out.data - y.data) / batch_size
            out.backward()
            
            # Update weights
            for p in model.parameters():
                p.data -= learning_rate * p.grad
            
            # Track metrics
            total_loss += loss
            predictions = cp.argmax(out.data, axis=1)
            correct += cp.sum(predictions == cp.argmax(y.data, axis=1))
            total += batch_size
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {total_loss/len(train_images):.4f}")
        print(f"Accuracy: {100*correct/total:.2f}%")
        
        # Evaluate on test set
        test_correct = 0
        test_total = 0
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]
            batch_labels = test_labels_one_hot[i:i+batch_size]
            
            x = Tensor(batch_images, device='GPU')
            y = Tensor(batch_labels, device='GPU')
            
            out = model(x)
            predictions = cp.argmax(out.data, axis=1)
            test_correct += cp.sum(predictions == cp.argmax(y.data, axis=1))
            test_total += len(batch_images)
        
        print(f"Test Accuracy: {100*test_correct/test_total:.2f}%")
        print("-------------------")

if __name__ == "__main__":
    train_mnist() 