import torch
from pykeops.torch import LazyTensor
import time

M, N, D = 10000, 20, 3
x = torch.randn(M, D, requires_grad=True).cuda()


for _ in range(10):
    tic = time.time()
    rand_indices = torch.randint(0, M, (M, N), device=x.device)
    y = x[rand_indices]

    # Dij = ((x.view(M, 1, D) - y)**2).sum(dim=2).topk(5, largest=False).indices
    # toc = time.time()
    # print(toc-tic)
    # print(Dij.shape)

    tic = time.time()
    x_i = LazyTensor(x.view(M, 1, 1, D))
    y_j = LazyTensor(y.view(M, N, 1, D))
    knn = ((x_i - y_j)**2).sum(dim=-1).argKmin(10, dim=1)
    toc = time.time()
    print(toc-tic)
    print(knn.shape)
