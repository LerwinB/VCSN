
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
from sklearn.metrics import pairwise_distances
from skimage import io
import random
import Strategy
from dataset import UESAT_segset
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l,EfficientNet_V2_L_Weights
import umap
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str, default='data/gssaptest', help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/Screenshots0528')
parser.add_argument('--data_list', type=str, default='src_path_all.txt')
parser.add_argument('--data_pre', type=str, default=True)
parser.add_argument('--task_name', type=str, default='Resnetall')
parser.add_argument('--model_type', type=str, default='resnet')
parser.add_argument('--checkpoint', type=str, default='pretrained/vit-base-patch16-224-in21k')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# train
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()

#  set up model for fine-tuning 
device = args.device
batch_size = args.batch_size
model_save_path = os.path.join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model = resnet50(weights=weights)
modules = list(model.children())[:-2]  # 不包括最后的avgpool和fc
model = torch.nn.Sequential(*modules)
model.to(device=device)
preprocess = weights.transforms()

#sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)


select_fuc = Strategy.Coreset()
reducer = umap.UMAP( n_neighbors=30,min_dist=0.0,n_components=64,random_state=42)

uedataset = UESAT_segset(args.data_path,args.data_list)
ueloader = DataLoader(dataset=uedataset,batch_size=batch_size,shuffle=True)
features=[]
imnames=[]
average_pool_kernel_size = (7, 7)
average_pool_stride = average_pool_kernel_size[0] // 2
avgpool = torch.nn.AdaptiveAvgPool2d((3, 3))
#getting features
print("getting features")
if args.data_pre:
    # use prepared data
    features=[]
    img_paths = []
    for i in range(11):
        
        fea_data=np.load(os.path.join(args.data_path, f"uesatL_resnet_full{i+1}.npz"))
        features.extend(fea_data["features"])
        img_paths.extend(fea_data["img_names"])
    
else:
    num=1
    for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):


        # model input: (1, 3, 1024, 1024)
        input_image = preprocess(images.to(device)) # (1, 3, 1024, 1024)
        #assert input_image.shape == (4, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        
        with torch.no_grad():
            feature=model(input_image)
            feature_s=avgpool(feature)
            #feature_s = F.avg_pool2d(feature, average_pool_kernel_size, average_pool_stride)
        #imgs.append(img)
        features.extend(feature_s.cpu().numpy())
        imnames.extend(imname)
        if len(imnames) >=50000:
            
            features = np.stack(features, axis=0)
            np.savez_compressed(
                os.path.join(args.data_path, f"uesatL_resnet_full{num}.npz"),
                features=features,
                img_names=imnames
            )
            num+=1
            del features,imnames
            features=[]
            imnames = []
    features = np.stack(features, axis=0)
    np.savez_compressed(
        os.path.join(args.data_path, f"uesatL_resnet_full{num}.npz"),
        #imgs=imgs,
        features=features,
        img_names=imnames
    )
print("active select")
features = np.array(features)
print("reduce dem")
reduce_fea=reducer.fit_transform(features.reshape(features.shape[0], -1))
print("clustering")
selected=select_fuc.select_batch(reduce_fea,len(features)//20)
print("save selected")
np.savez_compressed(
    os.path.join(args.data_path,  "uesatL_resnet_selected.npz"),
    #imgs=[imgs[i] for i in selected],
    features=[features[i] for i in selected],
    img_names=[img_paths[i] for i in selected]
)
with open(os.path.join(args.data_path,  "uesatL_resnet_selected.txt"), "w") as file:
    # 遍历列表，写入每个字符串
    for i in selected:
        file.write(img_paths[i] + "\n")

