import os
import re
import torch
import torchvision
from PIL import Image
import random
import numpy as np
from torchvision.datasets import VisionDataset
# from mmseg.datasets.basesegdataset import BaseSegDataset
class UESAT_segset(torch.utils.data.Dataset): 
    def __init__(self, data_dir,train = True, transform=None, target_transform=None):
        if train:
            self.file_path = os.path.join(data_dir,'images/train')
            self.dicts = os.listdir(os.path.join(data_dir,'images/train'))
        else:
            self.file_path = os.path.join(data_dir,'images/minival')
            self.dicts = os.listdir(os.path.join(data_dir,'images/minival'))
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform


    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = os.path.join(self.file_path,self.dicts[index].rstrip('\n'))
        image_ori=Image.open(img_path)
        if self.transform:
            image = self.transform(image_ori,return_tensors="pt")
        else:
            image = np.array(Image.open(img_path).resize((1024,1024)))
            image = self.process(image)

        label = np.array(Image.open(os.path.join(self.file_path,self.dicts[index].rstrip('\n').replace('src','label'))).resize((1024,1024)))
        return (image,label,self.dicts[index])

    def __len__(self):
        return len(self.dicts)
    
class HIL_segset(torch.utils.data.Dataset): 
    def __init__(self, data_dir,train = True, transform=None, target_transform=None):
        if train:
            self.file_path = os.path.join(data_dir,'images/training')
            self.dicts = os.listdir(os.path.join(data_dir,'images/training'))
        else:
            self.file_path = os.path.join(data_dir,'images/minival')
            self.dicts = os.listdir(os.path.join(data_dir,'images/minival'))
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform


    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = os.path.join(self.file_path,self.dicts[index].rstrip('\n'))
        image_ori=Image.open(img_path)
        if self.transform:
            image = self.transform(image_ori,return_tensors="pt")
        else:
            image = np.array(Image.open(img_path).resize((1024,1024)))
            image = self.process(image)

        label = np.array(Image.open(os.path.join(self.file_path,self.dicts[index].rstrip('\n').replace('src','label'))).resize((1024,1024)))
        return (image,label,self.dicts[index])

    def __len__(self):
        return len(self.dicts)

class UESAT_VLset(torch.utils.data.Dataset):
    def __init__(self, data_dir, name, train = True, transform=None, target_transform=None):
        self.file_path = data_dir
        if train:
            f=open(os.path.join(data_dir, name),"r")
        else:
            f=open(os.path.join(self.file_path, "test.txt"),"r")
        self.dicts = f.readlines()
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform
        f.close()

    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = os.path.join(self.file_path,'images','train', self.dicts[index].rstrip('\n')+'.png')
        mask_path = os.path.join(self.file_path,'annotations','train', self.dicts[index].rstrip('\n')+'.png')
        text_path = os.path.join(self.file_path,'text', self.dicts[index].rstrip('\n')+'.txt')
        image_ori=Image.open(img_path)
        if self.transform:
            image = self.transform(image_ori,return_tensors="pt")
        else:
            image = np.array(Image.open(img_path).resize((1024,1024)))
            image = self.process(image)

        label = np.array(Image.open(mask_path).resize((1024,1024)))
        with open(text_path) as t:
            text=t.readline()
        return (image,label,text,self.dicts[index])

    def __len__(self):
        return len(self.dicts)

class UESAT_feas(torch.utils.data.Dataset):
    def __init__(self, data_dir, name,train = True, transform=None, target_transform=None):
        self.file_path = data_dir
        if train:
            f=open(os.path.join(self.file_path, name),"r")
        else:
            f=open(os.path.join(self.file_path, "test.txt"),"r")
        self.dicts = f.readlines()
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform
        f.close()

    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = os.path.join(self.file_path, 'color',self.dicts[index].rstrip('\n'))
        image = np.array(Image.open(img_path).resize((1024,1024)))
        image = self.process(image)

        if self.transform:
            image = self.transform(image)

        label = np.array(Image.open(os.path.join(self.file_path, 'label',self.dicts[index].rstrip('\n'))).resize((1024,1024)))
        return (image,label,self.dicts[index])

    def __len__(self):
        return len(self.dicts)

class UESAT_mmseg(VisionDataset):

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        f=open(os.path.join(self.root, name+".txt"),"r")
        self.images = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'color', self.images[index].rstrip('\n'))
        mask_path = os.path.join(self.root, 'label',self.images[index].rstrip('\n'))

        img = Image.open(img_path).convert('RGB')# Convert to RGB
        mask = Image.open(mask_path)  
        img=img.resize((512,512))
        mask=mask.resize((512,512))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            gt_sem_seg = self.target_transform(mask)
        # Convert the RGB values to class indices
        labels = np.array(mask)
        
        data_samples = dict(
            labels=labels, img_path=img_path, mask_path=mask_path, gt_sem_seg=gt_sem_seg)
        return img, data_samples

    def __len__(self):
        return len(self.images)
    
# class MMEngineUeset(BaseSegDataset):
#     def __init__(self, pytorch_dataset):
#         super().__init__()
#         self.pytorch_dataset = pytorch_dataset
    
#     def __len__(self):
#         return len(self.pytorch_dataset)
    
#     def __getitem__(self, idx):
#         # 这里调用 PyTorch dataset 的 __getitem__
#         # 可能需要根据 MMEngine 的需求进一步处理数据
#         return self.pytorch_dataset[idx]

class COCOset(torch.utils.data.Dataset): 
    def __init__(self, data_dir, train = True, transform=None, target_transform=None):
        
        if train:
            self.file_path = os.path.join(data_dir,'images/train2017')
        else:
            self.file_path = os.path.join(data_dir,'images/val2017')
        self.dicts = os.listdir(self.file_path)
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform

    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = os.path.join(self.file_path,self.dicts[index])
        image_ori=Image.open(img_path)
        if self.transform:
            image = self.transform(image_ori,return_tensors="pt")
        else:
            image = np.array(Image.open(img_path).resize((1024,1024)))
            image = self.process(image)
        label_path = os.path.join(self.file_path.replace('images','annotations'),self.dicts[index].replace('.jpg','_labelTrainIds.png'))
        label = np.array(Image.open(label_path).resize((1024,1024), Image.NEAREST))
        return (image,label,self.dicts[index])

    def __len__(self):
        return len(self.dicts)
    
class UESAT_clsset(torch.utils.data.Dataset): 
    string_to_id = {
        '01GSSAP': 1,
        '02Mycroft': 2,
        '03TETRA_1': 3,
        '04TETRA-2': 4,
        '04TETRA-3_4': 5,
        '06TETRA-3_4': 6,
        'ROOSTER': 7,
        '08EAGLE': 8,
        '09ESPAStar-HP': 9,
        '10Jackal': 10,
        'Gaofen-13': 11,
        '12CH01': 12,
        '13Cartosat-2': 13,
        '14ACCESS': 14,
        '15ChandraX': 15,
        '16geoeye-1': 16,
        '17IKONOS': 17,
        '18deimos-2_new': 18,
        'Gaofen-1': 19,
        '20Gaofen-2': 20,
        '21cehui-1': 21,
        '22worldview-1': 22,
        '23Pleiades': 23,
        '24KOMPSAT-3': 24,
        '25ORS-1': 25,
        '26ALOS': 26,
        '27Worldview-2': 27,
        '28Worldview-3': 28,
        '29Worldview-4': 29,
        '30QuickBird': 30,
        '31ThalesAleni': 31,
        '32aiji2': 32,
        '33GEO-Kompsat-2A': 33,
        '34GEO-Kompsat-2B': 34,
        '35KOMPSAT-5': 35,
        '43AdvancedOrion10': 36,
        '44TDRS-13': 37,
        '45LUCAS': 38,
        '46tianlian-1': 39,
        '47SBIRSGEO': 40,
        '48Nemesis1_2': 41,
        '49AEHF': 42,
        '50WGS': 43,
        '51Telstar18Vantage': 44,
        '52Telstar19Vantage': 45,
        '53Telstar14R': 46,
        '54Telstar12V': 47,
        '55Intelsat22': 48,
        '56MUOS-1': 49,
        '57SiriusXM_8': 50,
        '58Arachne': 51,
        '59MEV-1': 52,
        '60ALOS3': 53,
    }
    def __init__(self, data_dir, name=None,train = True, transform=None, target_transform=None):
        self.file_path = data_dir

        if train:
            if name:
                with open(os.path.join(data_dir,name)) as f:
                    files = f.readlines()
                    self.dicts = [os.path.join(data_dir,'images/train',file.strip('\n')+'.png') for file in files]
            else:
                files = os.listdir(os.path.join(data_dir,'images/train'))
                self.dicts = [os.path.join(data_dir,'images/train',file) for file in files]
        else:
            files = os.listdir(os.path.join(data_dir,'images/minival'))
            self.dicts = [os.path.join(data_dir,'images/minival',file) for file in files]
        print(len(self.dicts))
        self.transform = transform
        #= torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.target_transform = target_transform


    def process(self,image_data):
    # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        image_data_pre = np.uint8(image_data.transpose(2, 0, 1))
        return image_data_pre

    def __getitem__(self,index):

        img_path = self.dicts[index]
        image_ori=Image.open(img_path)
        if self.transform:
            image = self.transform(image_ori)
        else:
            image = np.array(Image.open(img_path).resize((1024,1024)))
            image = self.process(image)
        cat = os.path.basename(self.dicts[index]).split("_c")[0]
        label = self.string_to_id.get(cat,None)
        # label = np.array(Image.open(os.path.join(self.file_path,self.dicts[index].rstrip('\n').replace('src','label'))).resize((1024,1024)))
        return image,label

    def __len__(self):
        return len(self.dicts)