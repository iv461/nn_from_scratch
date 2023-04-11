
import torch

lin = torch.nn.Linear(1, 10)
for name, param in lin.named_parameters():
    print(f"Param {name} of shape {param.shape}: {param}")
