# import torch
# data = torch.load('src\model\weights\epoch0.pt')
# print(data)


import torch

# Langsung muat seluruh model
model = torch.load("C:\Users\ASUS\Documents\disertasi\LocObj\src\model\weights" )       # ← GANTI

# model = torch.load('src\model\weights\epoch0.pt')
check = model.eval()

# print(check)
