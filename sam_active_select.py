
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from sklearn.metrics import pairwise_distances
from skimage import io
import random
import Strategy
from dataset import UESAT_segset
# import umap
from cuml.manifold import UMAP
from torch.cuda.amp import autocast
from models.VAEreducer import UnetVAE
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str, default='data/gssaptest', help='path to training npy files; two subfolders: npy_gts and npy_embs')
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/Screenshots0528')
parser.add_argument('--data_list', type=str, default='src_path_all.txt')
parser.add_argument('--task_name', type=str, default='uesatL_SAM_Unetvae_full')
parser.add_argument('--data_pre', default=False)
parser.add_argument('--model_type', type=str, default='vit_h')
parser.add_argument('--checkpoint', type=str, default='pretrained/sam_vit_h_4b8939.pth')
parser.add_argument('--vae_model', type=str, default='work_dirs_vae/coco_samvae/UnetVAEsam.pth')
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
task_name = args.task_name
model_save_path = os.path.join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)
if args.data_pre:
    print("use prepared data:"+task_name)
    # use prepared data
    features=[]
    img_paths = []
    for i in range(11):
        fea_data=np.load(os.path.join(args.data_path, task_name+f"{i+1}.npz"))
        features.extend(fea_data["features"])
        img_paths.extend(fea_data["img_names"])

else:
    print("load model")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)



    uedataset = UESAT_segset(args.data_path,args.data_list)
    ueloader = DataLoader(dataset=uedataset,batch_size=args.batch_size,shuffle=True)

    # vaemodel = UnetVAE(40).to(device)
    # vaemodel.load_state_dict(torch.load('pretrained/UnetVAEsam.pth'))
    vaemodel =torch.load(args.vae_model).to(device)
    vaeoptimizer = torch.optim.Adam(vaemodel.parameters(), lr=1e-3)
    vaemodel.train()
    features=[]
    imnames=[]
    average_pool_kernel_size = (32, 32)
    average_pool_stride = average_pool_kernel_size[0] // 2
    #getting features
    print("getting features")


    num=1
    overall_loss=0
    for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):


        # model input: (1, 3, 1024, 1024)
        input_image = sam_model.preprocess(images.to(device)) # (1, 3, 1024, 1024)
        #assert input_image.shape == (4, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
        
        with torch.no_grad():
            with autocast():
                feature=sam_model.image_encoder(input_image)

            x_hat, mean, log_var,x_re = vaemodel(feature)

            # feature = F.avg_pool2d(feature, average_pool_kernel_size, average_pool_stride)
        #imgs.append(img)
        features.extend(torch.flatten(x_re[0],start_dim=1).cpu().numpy())
        imnames.extend(imname)

        if len(imnames) >=50000:
            
            features = np.stack(features, axis=0)
            np.savez_compressed(
                os.path.join(args.data_path, task_name+f"{num}.npz"),
                features=features,
                img_names=imnames
            )
            num+=1
            del features,imnames
            features=[]
            imnames = []
    features = np.stack(features, axis=0)
    np.savez_compressed(
        os.path.join(args.data_path,  task_name+f"{num}.npz"),
        #imgs=imgs,
        features=features,
        img_names=imnames
    )
    img_paths = imnames
'''
        features.append(x_re.cpu())
        imnames.extend(imname)
        if len(imnames) >=4000:
            
            torch_features = torch.cat(features, dim=0)
            torch.save(torch_features,os.path.join(args.data_path, f"uesatL_sam64_full{num}.pt"))
            np.savez_compressed(
                os.path.join(args.data_path, f"uesatL_sam64_full{num}.npz"),
                img_names=imnames
            )
            num+=1
            del features,imnames
            features=[]
            imnames = []
'''

print("active select")
#umap_model = UMAP(n_components=1024, n_neighbors=80, random_state=42,  verbose=True)
#hdbscan_model = DBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=False, min_cluster_size=20, verbose=True)
#reducer = umap.UMAP( n_neighbors=30,min_dist=0.0,n_components=64,random_state=42)
select_fuc = Strategy.KMeansSampling()
features = np.array(features)
print("reduce dem")
#features=umap_model.fit_transform(features.reshape(features.shape[0], -1))
#reduce_fea=reducer.fit_transform(features.reshape(features.shape[0], -1))
print("clustering")
#selected = hdbscan_model.fit_predict(reduce_fea)
selected=None
for j in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
    selected=select_fuc.select_batch(features.reshape(features.shape[0], -1),int(len(features)*j*0.01),selected_idx=selected)
    print(f"save selected batch {j}")
    with open(os.path.join("data/UESAT_RGB_53/MMdata",  f"sam{j}_VAE3n_kmeans.txt"), "w") as file:
        # 遍历列表，写入每个字符串
        for i in selected:
            file.write(img_paths[i].rstrip('.png\n').split('/')[-1]+'\n')
'''
selected.to_json('cluster_labels.json')
selected_np = selected.to_numpy()
np.savez_compressed(
    os.path.join(args.data_path,  "uesatL_sam_selected.npz"),
    #imgs=[imgs[i] for i in selected],
    features=[features[i] for i in selected_np],
    img_names=[img_paths[i] for i in selected_np]
)
'''