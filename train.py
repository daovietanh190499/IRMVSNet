from dataset import MVSDataset
from model import IRMVSNet, irmvsnet_loss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from homo_warping import *

device = "cuda"

batch_size = 1

torch.manual_seed(1)
torch.cuda.manual_seed(1)

train_dataset = MVSDataset("../blended_mvs/", '../blended_mvs/train.txt', 'train', 3, 128)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, drop_last=True)

val_dataset = MVSDataset("../blended_mvs/", '../blended_mvs/val.txt', 'val', 10, 128)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, drop_last=False)

model = IRMVSNet((train_dataset[0]["imgs"][0].shape[1], train_dataset[0]["imgs"][0].shape[2]), device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)

torch.autograd.set_detect_anomaly(True)

def tocuda(data):
    data["imgs"] = data["imgs"].squeeze(0).to(device)
    data["K"] = data["K"].squeeze(0).to(device)
    data["Rt"] = data["Rt"].squeeze(0).to(device)
    data["depth_values"] = data["depth_values"].squeeze(0).to(device)
    data["depth"] = data["depth"].squeeze(0).to(device)
    data["mask"] = data["mask"].squeeze(0).to(device)
    return data

start_epoch = 0
epochs = 100

milestones = [int(epoch_idx) for epoch_idx in "10,12,14:2".split(':')[0].split(',')]
lr_gamma = 1 / float("10,12,14:2".split(':')[1])
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

f = open("log.txt", 'w')

for epoch_idx in range(start_epoch, epochs):
    print('Epoch {}:'.format(epoch_idx))
    
    for batch_idx, sample in enumerate(train_loader):
            
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        sample_cuda = tocuda(sample)

        p, depth_est, prob = model(sample_cuda["imgs"], sample_cuda["K"], sample_cuda["Rt"], sample_cuda["depth_values"])
        loss = irmvsnet_loss(p, sample_cuda["depth"], sample_cuda["depth_values"])
        
        loss.backward()
        optimizer.step()

        print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, epochs, batch_idx,
                                                                                     len(train_loader), loss.item(),
                                                                                     time.time() - start_time))
        f.write('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f} \n'.format(epoch_idx, epochs, batch_idx,
                                                                                     len(train_loader), loss.item(),
                                                                                     time.time() - start_time))
        if ((batch_idx + 1) + epoch_idx*len(train_loader)) % 10000 == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}_{:0>8}.ckpt".format("pretrained", epoch_idx, batch_idx))

            lr_scheduler.step()
    
    for batch_idx, sample in enumerate(val_loader):
        start_time = time.time()
        model.eval()
        
        sample_cuda = tocuda(sample)
        
        with torch.no_grad():
            p, depth_est, prob = model(sample_cuda["imgs"], sample_cuda["K"], sample_cuda["Rt"], sample_cuda["depth_values"])
            loss = irmvsnet_loss(p, sample_cuda["depth"], sample_cuda["depth_values"])

        print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, epochs, batch_idx,
                                                                                 len(val_loader), loss.item(),
                                                                                 time.time() - start_time))
        f.write('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f} \n'.format(epoch_idx, epochs, batch_idx,
                                                                                 len(val_loader), loss.item(),
                                                                                 time.time() - start_time))

    