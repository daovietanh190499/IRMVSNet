import torch
from model import IRMVSNet
from dataset import MVSDataset
import time

'''

Args: 
  images: Nx3xHxW tensor. H, W should be multiple of 16
  intrinsics: Nx3x3 tensor
  extrinsics: Nx4x4 tensor
  depth_start: float
  depth_interval: float
  depth_num: float

  depth ranges are computed by: depth_start + range(depth_num) * depth_interval

Return:
  probs: tensor of shape (H/4)x(W/4)
  depths: tensor of shape (H/4)x(W/4)
'''

model = IRMVSNet()

# optional: put model into gpu
model.to(torch.device('cuda'))

val_dataset = MVSDataset("../blended_mvs/", '../blended_mvs/val.txt', 'val', 10, 128)

def totorch(data):
    data["imgs"] = torch.from_numpy(data["imgs"]).to("cuda")
    data["K"] = torch.from_numpy(data["K"]).to("cuda")
    data["Rt"] = torch.from_numpy(data["Rt"]).to("cuda")
    data["depth_values"] = torch.from_numpy(data["depth_values"]).to("cuda")
    data["depth"] = torch.from_numpy(data["depth"]).to("cuda")
    data["mask"] = torch.from_numpy(data["mask"]).to("cuda")
    return data

sample = val_dataset[0]
sample_cuda = totorch(sample)
depth_gt = sample_cuda["depth"]
mask = sample_cuda["mask"]

start = time.time()
print(start)
with torch.no_grad():
    _, depths, probs = model(sample_cuda["imgs"], sample_cuda["K"], sample_cuda["Rt"], sample_cuda["depth_values"])
    print(depths)
    _, depths, probs = model(sample_cuda["imgs"], sample_cuda["K"], sample_cuda["Rt"], sample_cuda["depth_values"])
    print(depths)
print(time.time() - start)