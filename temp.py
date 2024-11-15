import torch
import time
print("hello world")
# 生成一个随机矩阵
a = torch.randn(1000, 1000).to('cuda')

# CPU计算
start = time.time()
for i in range(10):
    torch.mm(a.cpu(), a.cpu())
end = time.time()
cpu_time = end-start

# GPU计算
start = time.time()
for i in range(10):
    torch.mm(a, a)
torch.cuda.synchronize()
end = time.time()
gpu_time = end-start

print(f"CPU计算时间: {cpu_time:.6f}秒")
print(f"GPU计算时间: {gpu_time:.6f}秒")
print(f"GPU加速比: {cpu_time/gpu_time:.6f}x")
