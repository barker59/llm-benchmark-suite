import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        print(f"GPU {i} capability: sm_{capability[0]}{capability[1]}")
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available")
