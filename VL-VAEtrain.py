import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.VAEreducer import DualVAE
import csv
import logging
import time

device='cuda'
data_path='data/UESAT_RGB_53/MMdata'
work_dir='work_dirs_vae/DualVAEclip'
batch_size = 64
model = DualVAE(input_dim=512,latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
weights = {
    'recon_image': 0.2,
    'recon_text': 0.2,
    'kl': 0.5,
    'contrastive': 0.1
}
img_paths = []
epochs=5
losses = []
start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
work_dir=os.path.join(work_dir,start_time)
os.makedirs(work_dir)
logging.basicConfig(filename=os.path.join(work_dir,'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("data path:"+data_path)
logger.info("model: DualVAE")
w_info=""
for key, value in weights.items():
    w_info += f"{key}: {value}, "
logger.info("weight:"+w_info)
model.train()
print('training')
st=time.time()
for i in range(1):
    fea_data=np.load(os.path.join(data_path, "uesatL_clip_full.npz"))
    features = torch.tensor(fea_data["features"], dtype=torch.float32)
    dataset = TensorDataset(features)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    overall_loss = 0
    for batch_idx, x0 in enumerate(data_loader):
        # x = x0[0].to(device)
        x,y = torch.chunk(x0[0].to(device), 2, dim=-1)
        optimizer.zero_grad()
        x_hat, y_hat, mean, log_var,latent_emb = model(x,y)
        total_loss, recon_loss_image, recon_loss_text, kl_loss, contrastive_loss =  model.loss_function(x_hat,y_hat,x,y,mean, log_var,latent_emb, weights)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2).item()
        #         if grad_norm > 1e2:  # 如果梯度值过大，打印警告
        #             log_message=f"Warning: Gradient norm for {name} is {grad_norm}"
        #             logger.info(log_message)
        #             print(log_message) 
        losses.append([total_loss.item(), recon_loss_image.item(), recon_loss_text.item(), kl_loss.item(), contrastive_loss.item()])
        if batch_idx % 10 == 0:  # Print every 10 batches
            elapsed_time = time.time() - st
            progress = (batch_idx + 1) / len(data_loader)
            remaining_time = elapsed_time * (1 - progress) / progress
            log_message = (f'Batch {batch_idx}/{len(data_loader) - 1}, '
                           f'Loss: {total_loss.item():.4f}, Progress: {progress:.2%}, '
                           f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s')
            logger.info(log_message)
            print(log_message)

    # print("\tEpoch", i + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
with open(os.path.join(work_dir,'DualVAElosses.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Step', 'Total Loss', 'Recon Loss Image', 'Recon Loss Text', 'KL Loss', 'Contrastive Loss'])
    for step, loss in enumerate(losses):
        writer.writerow([step] + loss)

print(overall_loss)
torch.save(model.state_dict(),os.path.join(work_dir,'VAE.pth'))
logger.info("successful")
