import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from dataset import UESAT_clsset
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import logging
import argparse
import time
import os

# set seeds
torch.manual_seed(2024)


# set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/UESAT_RGB_53/MMdata')
parser.add_argument('--data_list', type=str, default=None)
parser.add_argument('--modelname', type=str, default='Resnet')
parser.add_argument('--work_dir', type=str, default='./work_dirs/Resnet_full_uesat')
# train
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()
datapath = args.data_path
datalist = args.data_list
modelname = args.modelname
num_classes = 54
start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
print(start_time)
os.makedirs(os.path.join(args.work_dir,start_time))
logging.basicConfig(filename=os.path.join(args.work_dir,start_time,'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info("data path:"+datapath)
if datalist:
    logger.info("data list:"+datalist)
# Define transformations for the training and validation sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
print("Load the datasets")
train_dataset = UESAT_clsset(data_dir=datapath,train=True,name=datalist,transform=train_transforms)
val_dataset = UESAT_clsset(data_dir=datapath,train=False,transform=val_transforms)
logger.info(f"train dataset size:{len(train_dataset)}")
logger.info(f"val dataset size:{len(val_dataset)}")
# train_dataset = datasets.ImageFolder(root='path/to/train', transform=train_transforms)
# val_dataset = datasets.ImageFolder(root='path/to/val', transform=val_transforms)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pretrained ResNet50 model
print("Load the pretrained ResNet50 model")
if modelname == 'Resnet':
    model = models.resnet50(pretrained=True)

    # Modify the final layer to match the number of classes in your dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace `num_classes` with the actual number of classes
elif modelname == 'ViT':
    model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = args.num_epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate and print detailed information
        if batch_idx % 10 == 0:  # Print every 10 batches
            elapsed_time = time.time() - start_time
            progress = (batch_idx + 1) / len(train_loader)
            remaining_time = elapsed_time * (1 - progress) / progress
            log_message = (f'Epoch {epoch}/{num_epochs - 1}, Batch {batch_idx}/{len(train_loader) - 1}, '
                           f'Loss: {loss.item():.4f}, Progress: {progress:.2%}, '
                           f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s')
            logger.info(log_message)
            print(log_message)
    
    epoch_loss = running_loss / len(train_dataset)
    logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    corrects = 0
    top5_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            _, top5_preds = outputs.topk(5, 1, True, True)  # Get top 5 predictions
            top5_corrects += torch.sum(top5_preds.eq(labels.view(-1, 1)).any(dim=1))  # Check if the label is in top 5

            all_preds.append(preds)
            all_labels.append(labels)

    val_loss = val_running_loss / len(val_dataset)
    val_acc = corrects.double() / len(val_dataset)
    top5_acc = top5_corrects.double() / len(val_dataset)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate additional metrics
    val_precision = torchmetrics.functional.precision(all_preds, all_labels, average='macro',task='multiclass', num_classes=num_classes)
    val_recall = torchmetrics.functional.recall(all_preds, all_labels, average='macro',task='multiclass', num_classes=num_classes)
    val_f1 = torchmetrics.functional.f1_score(all_preds, all_labels, average='macro',task='multiclass', num_classes=num_classes)

    # Print additional metrics
    log_message = f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}'
    logger.info(log_message)
    print(log_message)

# save model
torch.save(model.state_dict(), os.path.join(args.work_dir,start_time,'model.pth'))
print("finish")