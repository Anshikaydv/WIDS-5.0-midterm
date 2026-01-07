# WIDS-5.0-midterm
This repository contains my learning and practice code from the WIDS 5.0 program.
It contains PyTorch tensors and then implementing neural network with PyTorch 

# PyTorch Tensors
Here is Tensors basic 

```
import torch
```
# Creating tensors
```
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
```
Scalar → 0D tensor (single number)  
Vector → 1D tensor  
Matrix → 2D tensor  
```
print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
```

# Tensor properties
```
print("Shape of vector:", vector.shape)
print("Datatype:", vector.dtype)

```
shape → tells dimensions
dtype → tells data type (int64, float32, etc.)

# Tensor Attributes
```
import torch

x = torch.randn(3, 4)

print("Tensor:", x)
print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)
print("Device:", x.device)
```
shape → size in each dimension
ndim → number of dimensions
dtype → float32 / float64
device → CPU or GPU

# Tensor Initialization
```
import torch

# Zeros
a = torch.zeros(3, 3)

# Ones
b = torch.ones(2, 4)

# Random values (normal distribution)
c = torch.randn(3, 3)

# Random values (uniform distribution)
d = torch.rand(3, 3)

print(a)
print(b)
print(c)
print(d)
```
# Tensor Operations
```
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float32)

# Addition
print(x + y)

# Subtraction
print(x - y)

# Element-wise multiplication
print(x * y)

# Element-wise division
print(x / y)
```
These are element-wise operations  
Shape of x and y must match  
Used heavily in loss computation & optimization  
# Tensor Multiplication
```
import torch

A = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)

B = torch.tensor([[5, 6],
                  [7, 8]], dtype=torch.float32)
```
```
print(A * B)
```
Multiplies corresponding elements  
NOT matrix multiplication
```
print(torch.matmul(A, B))
```
OR
```
print(A @ B)
```
Uses linear algebra rule
Used in neural networks:
    ```
    output = input @ weights + bias
    ```
# Dot Product 
``` x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

dot = torch.dot(x, y)
print(dot)
```
Dot product = weighted sum
# Reshaping & View
```
x = torch.arange(1, 7)

x_reshaped = x.view(2, 3)
print(x_reshaped)
```
view() reshapes tensor  
No data copied, just reinterpreted  
Used to flatten images before fully connected layers  
# NumPy ↔ PyTorch Bridge
```
import torch
import numpy as np

# NumPy array
np_array = np.array([1, 2, 3])

# Convert NumPy → Tensor
torch_tensor = torch.from_numpy(np_array)

print(torch_tensor)
```
Tensor → NumPy
```
tensor = torch.tensor([4, 5, 6])

np_from_tensor = tensor.numpy()
print(np_from_tensor)
```
They share memory  
Changing one changes the other
```
np_array[0] = 100
print(torch_tensor)
```
