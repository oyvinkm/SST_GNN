import torch

print("CUDA is available:", torch.cuda.is_available())
print("CUDA has version:", torch.version.cuda)
print("torch has version:", torch.__version__)
