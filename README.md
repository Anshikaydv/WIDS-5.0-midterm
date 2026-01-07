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

