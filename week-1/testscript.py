import torch
n = torch.arange(0, 30000)
enum = torch.pow(-1,n)
denom = (2*n+1)
print(4*torch.sum(enum/denom))