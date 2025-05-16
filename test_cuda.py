import torch
import cupy as cp

# PyTorch CUDA test
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# PyTorch tensor on GPU
x_torch = torch.rand(3, 3, device='cuda')
print(f"PyTorch tensor:\n{x_torch}")


# CuPy array on GPU
x_cupy = cp.random.rand(3, 3)
print(f"CuPy array:\n{x_cupy}")
