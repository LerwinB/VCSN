
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision import transforms
import argparse
from sklearn.metrics import pairwise_distances
from skimage import io
import random
import Strategy
from dataset import UESAT_VLset
from transformers import CLIPProcessor, CLIPModel
import clip
from models.VAEreducer import DualVAE
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str, default='data/gssaptest', help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/MMdata')
parser.add_argument('--data_list', type=str, default='uesatL_MMdata_full.txt')
parser.add_argument('--data_pre', type=str, default=True)
parser.add_argument('--task_name', type=str, default='ViT-B-53all')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='pretrained/vit-base-patch16-224-in21k')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# train
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

#  set up model for fine-tuning 
device = args.device
batch_size = args.batch_size
model_save_path = os.path.join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)
model, processor = clip.load("ViT-B/32", device=device)
resize_transform = transforms.Resize((224, 224))
# processor = CLIPProcessor.from_pretrained(args.checkpoint)
# model = CLIPModel.from_pretrained(args.checkpoint)

#sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)


select_fuc = Strategy.KMeansSampling()

uedataset = UESAT_VLset(args.data_path,args.data_list)
ueloader = DataLoader(dataset=uedataset,batch_size=batch_size,shuffle=True)
vaemodel = DualVAE(512,32).to(device)
vaemodel.load_state_dict(torch.load('work_dirs_vae/DualVAEclip/20241015_224116/VAE.pth'))
# vaemodel=torch.load('work_dirs_vae/DualVAEclip/20241015_224116/VAE.pth')


features=[]
imnames=[]
latents = []
average_pool_kernel_size = (7, 7)
average_pool_stride = average_pool_kernel_size[0] // 2
#getting features
print("getting features")
if args.data_pre:
    # use prepared data
    fea_data=np.load(os.path.join(args.data_path,'uesatL_clip_vae_full.npz'))
    features=fea_data["features"]
    latents=fea_data["latents"]
    imnames=fea_data["img_names"]
else:
    num=1
    for batch_idx, (images,labels,text,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):


        # model input: (1, 3, 1024, 1024)
        
        # input_image = processor(images).unsqueeze(0).to(device) # (1, 3, 1024, 1024)
        #assert input_image.shape == (4, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        input_image = resize_transform(images).to(device)
        text_token=clip.tokenize(text).to(device)
        with torch.no_grad():
            feature=model.encode_image(input_image)
            text_emb = model.encode_text(text_token)
            x_hat, y_hat, mean, log_var,latent_emb = vaemodel(feature.float(),text_emb.float())
            concatenated_tensor = torch.cat((feature, text_emb), dim=1)

        # img_fea:(4,512) text_emb(4,512)
        #imgs.append(img)
        features.extend(concatenated_tensor.cpu().numpy())
        latents.extend(latent_emb.cpu().numpy())
        imnames.extend(imname)
        # if len(imnames) >=50000:
            
        #     features = np.stack(features, axis=0)
        #     np.savez_compressed(
        #         os.path.join(args.data_path, f"uesatL_clip_full{num}.npz"),
        #         features=features,
        #         img_names=imnames
        #     )
        #     num+=1
        #     del features,imnames
        #     features=[]
        #     imnames = []
    features = np.stack(features, axis=0)
    latents = np.stack(latents,axis=0)
    np.savez_compressed(
        os.path.join(args.data_path, f"uesatL_clip_vae_full.npz"),
        #imgs=imgs,
        features=features,
        latents=latents,
        img_names=imnames
    )
print("active select")
selected=None
for j in [5,10,15,20]:
    selected=select_fuc.select_batch(latents.astype(np.float32).reshape(latents.shape[0], -1),int(len(latents)*j*0.01),selected_idx=selected)
    print(f"save selected batch {j}")

    with open(os.path.join(args.data_path,  f"clip{j}_vae_kmeans.txt"), "w") as file:
        # 遍历列表，写入每个字符串
        for i in selected:
            file.write(imnames[i].rstrip('.png\n').split('/')[-1]+'\n')
