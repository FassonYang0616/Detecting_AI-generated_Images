import torch

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("CUDA device: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

"""这个程序用来确认pytorch可以正常的和到Cuda进行连接"""