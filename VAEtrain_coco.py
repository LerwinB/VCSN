import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
from models.VAEreducer import UnetVAE
from dataset import  COCOset
from segment_anything import sam_model_registry
import argparse
import logging
from torch.cuda.amp import autocast
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/coco164k')
parser.add_argument('--task_name', type=str, default='SAM-ViT-B')
parser.add_argument('--model_type', type=str, default='vit_h')
parser.add_argument('--checkpoint', type=str, default='pretrained/sam_vit_h_4b8939.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dirs/UnetVAE')
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--aliyun', default=False)
args = parser.parse_args()

device=args.device
data_path=args.data_path
batch_size = args.batch_size
max_iters = args.max_iters
work_dir = args.work_dir
if args.aliyun:
    aliyun=args.aliyun
    oss_dir = os.path.join('/mnt',work_dir)
start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
print(start_time)
os.makedirs(os.path.join(args.work_dir,start_time))
logging.basicConfig(filename=os.path.join(args.work_dir,start_time,'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("data path:"+data_path)
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
model = UnetVAE(40).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cocodataset = COCOset(args.data_path)
indices = list(range(max_iters))
subset = Subset(cocodataset, indices)
# uedataset = UESAT_segset(args.data_path,args.data_list)
dataloader = DataLoader(dataset=subset,batch_size=args.batch_size,shuffle=True)
img_paths = []

model.train()
print('training')

overall_loss = 0
st = time.time()
for batch_idx, (images,labels,imname) in enumerate(dataloader):
    input_image = sam_model.preprocess(images.to(device))
    with torch.no_grad():
        with autocast():
            feature=sam_model.image_encoder(input_image)
    x = feature.to(device)
    optimizer.zero_grad()
    x_hat, mean, log_var,_ = model(x)
    lossdict = model.loss_function(x_hat, x, mean, log_var)
    loss=lossdict['loss']
    if batch_idx % 10 == 0:  # Print every 10 batches
        elapsed_time = time.time() - st
        progress = (batch_idx + 1) / len(dataloader)
        remaining_time = elapsed_time * (1 - progress) / progress
        log_message = (f'Batch {batch_idx}/{len(dataloader) - 1}, '
                        f'Loss: {loss.item():.4f}, Progress: {progress:.2%}, '
                        f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s')
        logger.info(log_message)
        print(log_message)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(),os.path.join(work_dir,start_time,'UnetVAEsam.pth'))
if aliyun:
    import shutil
    print("copy work_dir to OOS")
    shutil.copytree(work_dir, oss_dir)
    print("successful")
