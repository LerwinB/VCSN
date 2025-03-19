from dataset import UESAT_segset, COCOset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms as T
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import numpy as np

def resize_labels(labels, size):
    # labels: 输入的标签张量，形状为 (B, H, W)
    # size: 一个元组，指定新的尺寸 (new_H, new_W)
    
    resized_labels = []
    for label in labels:
        # label: (H, W)
        # 使用最近邻插值来调整标签大小
        resized_label = TF.resize(label.unsqueeze(0), size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        resized_labels.append(resized_label)

    # 将列表转换回张量
    return torch.stack(resized_labels)
def loss_fn(outputs, targets):
    # outputs: 模型的输出，形状应为 [B, C, H, W]，其中 C 是类别数
    # targets: 真实的标签，形状应为 [B, H, W]，且数据类型为 torch.long
    # 返回每个图像的损失，形状为 [B]
    
    # 计算交叉熵损失，reduction设置为'none'来保持损失的形状与批次大小一致
    losses = F.cross_entropy(outputs, targets, reduction='none')
    
    # 由于交叉熵损失会返回每个像素的损失，我们需要对每张图像的所有像素损失求平均
    losses = losses.mean(dim=[1, 2])
    
    return losses


def calculate_entropy(probs, eps=1e-10):
    # probs: 概率张量，形状为 [B, N, H, W]
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        raise ValueError("probs contains NaN or Inf values")
    # 避免数值问题，确保概率非负
    probs = torch.clamp(probs, min=eps, max=1.0)
    
    # 计算熵
    entropy = -probs * torch.log(probs)
    
    # 对所有类别求和得到每个像素的熵，然后对所有像素求和
    entropy = entropy.sum(dim=1)  # 求和所有类别
    entropy = entropy.view(probs.size(0), -1).mean(dim=1)  # 将 H 和 W 维度合并，然后求和
    
    return entropy

class ucsample:
    def __init__(self,n):
        self.n=n
    def entropy_sampling(self,entropy,loss):
        selected_idx=np.argsort(entropy)[-self.n:]
        return selected_idx
    def ULAL_sampling(self,entropy,loss):
        a=1
        score=entropy+a*loss
        selected_idx=np.argsort(score)[-self.n:]
        return selected_idx


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/MMdata')
parser.add_argument('--dataset', type=str, default='uesat')
parser.add_argument('--method', type=str, default='entropy')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dir')


args = parser.parse_args()

device='cuda:0'
data_path=args.data_path
batch_size=5
dataset_name = args.dataset
selected_size = 4 # 1:5%,2: 10%,3: 15%,4: 20%
if dataset_name == 'uesat':
    dataset = UESAT_segset(args.data_path)
    num_classes = 21
elif dataset_name == 'coco':
    dataset = COCOset(args.data_path)
    num_classes = 171

if args.method == 'entropy':
    sampling = ucsample(selected_size).entropy_sampling
elif args.method == 'ULAL':
    sampling = ucsample(selected_size).ULAL_sampling
ueloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21)
# model.classifier[4] = torch.nn.Conv2d(256, 171, kernel_size=(1, 1), stride=(1, 1))
model.to(device=device)
model.train()  # 切换到训练模式
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()
preprocess = weights.transforms()
# wpreprocess = weights.target_transforms()

selected_batch5=[]
selected_batch10=[]
selected_batch15=[]
selected_batch20=[]
i=1
lossall=0
mark = 0
imnames=[]
entropys = []
for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):
    
    
    optimizer.zero_grad()
    # with autocast():
    if True:
        input_images=preprocess(images)
        input_images=input_images.to(device)
        outputs = model(input_images)['out']
        
        target = resize_labels(labels=labels[...,0],size=outputs.size()[-2:])
        target = target.long().to(device)
        # loss = loss_fn(outputs, target.long())
        loss = criterion(outputs, target)
        # print("loss:", loss.item())
        # sampled_loss = loss.mean()

        probs = torch.softmax(outputs, dim=1)  # 计算预测概率
        # entropy = calculate_entropy(probs=probs).cpu()
        
        entropy0=calculate_entropy(probs=probs).cpu().detach().numpy()
        entropys.append(entropy0)
        imnames.extend(imname)
        mark+=1
    if mark == 4:
        entropy = np.hstack(entropys)
        sidx = sampling(entropy=entropy,loss=loss.cpu().detach().numpy())
        
        selected_batch5.append(imnames[sidx[0]])
        selected_batch10.append(imnames[sidx[0,1]])
        selected_batch15.append(imnames[sidx[0,1,2]])
        selected_batch20.append(imnames[sidx[0,1,2,3]])
        mark=0
        entropys = []
        imnames = []
        loss.backward()
        optimizer.step()
    # scaler.scale(sampled_loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    # if i ==20:
    #     lossall=lossall/20
    #     scaler.scale(lossall).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     lossall = 0
    #     i = 0
    # else:
    #     i+=1
    #     lossall+=sampled_loss

    del input_images, outputs, probs, target, loss
    torch.cuda.empty_cache()
for i in [5,10,15,20]:
    with open(os.path.join(args.data_path, args.method+f"_{i}.txt"), "w") as file:
        for img_path in selected_batch:
                file.write(img_path.rstrip('.png\n').split('/')[-1]+'\n')
# with open('work_dirs/Deeplabv3_Entropy/Entropy5.txt','w') as file:
#     file.writelines(selected_batch)
# with open(os.path.join(args.data_path,dataset_name+ f"COCO_ULAL_{selected_size}_kmeans.txt"), "w") as file:
#     for img_path in selected_batch:
#                 file.write(img_path.rstrip('.jpg\n').split('/')[-1]+'\n')

torch.save(model.state_dict(),os.path.join(args.work_dir,'ULAL_CKP',args.method+'.pth'))

    


