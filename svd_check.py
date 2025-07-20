import torch
import time

A = torch.rand(1000, 3, 3).cuda()
print("start")
tic = time.time()
u, s, v = torch.svd(A)
toc = time.time()
print(toc-tic)