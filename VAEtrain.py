import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.VAEreducer import UnetVAE


device='cuda'
data_path='data/UESAT_RGB_53/Screenshots0528'
batch_size = 16
model = UnetVAE(40).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

img_paths = []
epochs=5
model.train()
print('training')
for i in range(20):
    fea_data=torch.load(os.path.join(data_path, f"uesatL_sam64_full{i+1}.pt"))
    print(f'data group {i+1}')
    dataset = TensorDataset(fea_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    overall_loss = 0
    for batch_idx, x0 in tqdm(enumerate(data_loader)):
        x = x0[0].to(device)
        optimizer.zero_grad()
        x_hat, mean, log_var,_ = model(x)
        lossdict = model.loss_function(x_hat, x, mean, log_var)
        loss=lossdict['loss']
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print("\tEpoch", i + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
print(overall_loss)
torch.save(model.state_dict(),'pretrained/UnetVAEsam.pth')
