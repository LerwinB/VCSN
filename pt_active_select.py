
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
from dataset import UESAT_segset, COCOset
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l,EfficientNet_V2_L_Weights
# import umap
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str, default='data/gssaptest', help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/Screenshots0528')
parser.add_argument('--data_list', type=str, default='src_path_all.txt')
parser.add_argument('--data_pre', type=bool, default=False)
parser.add_argument('--task_name', type=str, default='Coreset')
parser.add_argument('--dataset_name', type=str, default='coco')
parser.add_argument('--model_type', type=str, default='resnet')
parser.add_argument('--suffix', type=str, default='.png')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--aliyun', default=False)
args = parser.parse_args()

#  set up model for fine-tuning 
device = args.device
batch_size = args.batch_size
task_name = args.task_name
dataset_name = args.dataset_name
suffix = args.suffix
if args.aliyun:
    aliyun=args.aliyun
    oss_dir = os.path.join('/mnt',args.work_dir)

if args.data_pre:
    # use prepared data
    print("load preprocessed data")
    features=[]
    img_paths = []
    fea_data=np.load(os.path.join(args.data_path, task_name+".npz"))
    features=fea_data["features"]
    img_paths=fea_data["img_names"]

else:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # model = resnet50(weights=weights)
    modules = list(model.children())[:-2]  # 不包括最后的avgpool和fc
    model = torch.nn.Sequential(*modules)
    model.to(device=device)
    preprocess = weights.transforms()

    #sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)



    # reducer = umap.UMAP( n_neighbors=30,min_dist=0.0,n_components=64,random_state=42)
    if dataset_name == 'uesat':
        dataset = UESAT_segset(args.data_path,args.data_list)
    elif dataset_name == 'coco':
        dataset = COCOset(args.data_path)
    print(dataset.__len__)
    ueloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    features=[]
    imnames=[]
    average_pool_kernel_size = (7, 7)
    average_pool_stride = average_pool_kernel_size[0] // 2
    avgpool = torch.nn.AdaptiveAvgPool2d((3, 3))
    #getting features
    print("getting features")
    num=1
    for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):


        # model input: (1, 3, 1024, 1024)
        input_image = preprocess(images.to(device)) # (1, 3, 1024, 1024)
        #assert input_image.shape == (4, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        
        with torch.no_grad():
            feature=model(input_image)
            avg_tensor=avgpool(feature)
        input_tensor_permuted = avg_tensor.permute(0, 2, 3, 1)
        pooling_layer = torch.nn.AdaptiveAvgPool2d((3, 64))
        pooled_tensor = pooling_layer(input_tensor_permuted)
        output_tensor = pooled_tensor.permute(0, 3, 1, 2)
        feature_s = torch.mean(output_tensor, dim=(2, 3))
            #feature_s = F.avg_pool2d(feature, average_pool_kernel_size, average_pool_stride)
        #imgs.append(img)
        # input (b,2048,3,3) output =(b,64)
        
        features.extend(feature_s.cpu().numpy())
        imnames.extend(imname)
        # if len(imnames) >=50000:
            
        #     features = np.stack(features, axis=0)
        #     np.savez_compressed(
        #         os.path.join(args.data_path, f"uesatL_resnet_full{num}.npz"),
        #         features=features,
        #         img_names=imnames
        #     )
        #     num+=1
        #     del features,imnames
        #     features=[]
        #     imnames = []
    features = np.stack(features, axis=0)
    np.savez_compressed(
        os.path.join(args.data_path, task_name+".npz"),
        #imgs=imgs,
        features=features,
        img_names=imnames
    )
print("active select")
select_fuc = Strategy.KMeansSampling()
#reduce_fea=reducer.fit_transform(features.reshape(features.shape[0], -1))
print("clustering")
reduced_fea=np.array(features)
selected=None
for j in [5,10,15,20]:
    selected=select_fuc.select_batch(reduced_fea.reshape(reduced_fea.shape[0], -1),int(len(reduced_fea)*j*0.01),selected_idx=selected)
    print(f"save selected batch {j}")
    with open(os.path.join("data/UESAT_RGB_53/MMdata",  task_name+f"_{j}_kmeans.txt"), "w") as file:
        # 遍历列表，写入每个字符串
        for i in selected:
            file.write(img_paths[i].rstrip(suffix+'\n').split('/')[-1]+'\n')
