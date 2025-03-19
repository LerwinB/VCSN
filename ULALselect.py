from dataset import UESAT_segset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
import torchvision.transforms as T
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

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
    
    # 避免数值问题，确保概率非负
    probs = torch.clamp(probs, min=eps, max=1.0)
    
    # 计算熵
    entropy = -probs * torch.log(probs)
    
    # 对所有类别求和得到每个像素的熵，然后对所有像素求和
    entropy = entropy.sum(dim=1)  # 求和所有类别
    entropy = entropy.view(probs.size(0), -1).mean(dim=1)  # 将 H 和 W 维度合并，然后求和
    
    return entropy


def entropy_sampling(entropy,loss,n):
    _,selected_idx=torch.topk(entropy,n)
    return selected_idx


def ULAL_sampling(entropy,loss,n):
    a=1
    score=entropy+a*loss
    _,selected_idx=torch.topk(score,n)
    return selected_idx

device='cuda'
data_path='data/UESAT_RGB_53/Screenshots0528'
data_list='src_path_all.txt'
batch_size=20
selected_size = 3 # 1:5%,2: 10%,3: 15%,4: 20%

uedataset = UESAT_segset(data_path,data_list)
ueloader = DataLoader(dataset=uedataset,batch_size=batch_size,shuffle=True)

weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(pretrained=False, progress=True)
model.to(device=device)
model.train()  # 切换到训练模式
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()
preprocess = weights.transforms()

selected_batch=[]
i=1
lossall=0
for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):
    optimizer.zero_grad()
    with autocast():
        input_images=preprocess(images)
        input_images=input_images.to(device)
        outputs = model(input_images)['out']
        probs = torch.softmax(outputs, dim=1)  # 计算预测概率
        entropy = calculate_entropy(probs=probs)

        target = resize_labels(labels=labels.to(device),size=outputs.size()[-2:])
        loss = loss_fn(outputs, target.long())

        sidx= ULAL_sampling(entropy=entropy,loss=loss,n=selected_size)
        sampled_loss = loss.mean()
    
        scaler.scale(sampled_loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    for i in sidx:
        selected_batch.append(imname[i])
    del input_images, outputs, probs, target, loss
    torch.cuda.empty_cache()

with open('work_dirs/Deeplabv3_ULAL/ULAL15.txt','w') as file:
    file.writelines(selected_batch)

torch.save(model.state_dict(),'work_dirs/Deeplabv3_ULAL/ULAL15.pth')

    


