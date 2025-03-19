from dataset import UESAT_segset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
import torchvision.transforms as T
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU


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

data_path='data/UESAT_RGB_53/Screenshots0528'
data_list='src_path_test_mini.txt'
batch_size=20
selected_size = 1 # 1:5%,2: 10%,3: 15%,4: 20%
device = 'cuda'
uedataset = UESAT_segset(data_path,data_list)
ueloader = DataLoader(dataset=uedataset,batch_size=batch_size,shuffle=True)
miou_metric = MeanIoU(num_classes=21)
#miou_metric = torchmetrics.IoU(num_classes=19, reduction='elementwise_mean', absent_score=0)

weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(pretrained=False, progress=True)
model.load_state_dict(torch.load('work_dirs/Deeplabv3_ULAL/ULAL5.pth'))
model.eval()


model.to(device=device)
miou_metric.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
preprocess = weights.transforms()

selected_batch=[]
for batch_idx, (images,labels,imname) in enumerate(tqdm(ueloader,desc='compute embedding')):
    input_images=preprocess(images)
    input_images=input_images.to(device)

    with torch.no_grad():
        output = model(input_images)['out']
        preds = torch.argmax(output, dim=1)
    target = resize_labels(labels=labels,size=output.size()[-2:])
    target = target.long().to(device=device)
    miou_metric.update(preds, target=target)

final_miou = miou_metric.compute()
print(f"Mean IoU on validation set: {final_miou}")
#torch.save(model.state_dict(),'work_dirs/Deeplabv3_Entropy5/Entropy5.pth')

    


