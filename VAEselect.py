import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.VAEreducer import VAE


class CustomDataset(Dataset):
    def __init__(self, features, imnames):
        self.features = features
        self.imnames = imnames

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        img_name = self.imnames[idx]
        return feature, img_name

device='cuda'
data_path='data/UESAT_RGB_53/Screenshots0528'
batch_size = 16
model = VAE(40).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



img_paths = []
epochs=5
model.train()
print('training')
for i in range(126):
    fea_data=torch.load(os.path.join(data_path, f"uesatL_sam64_full{i+1}.pt"))
    with np.load(os.path.join(data_path, f"uesatL_sam64_full{i+1}.npz")) as data:
        imnames = data['img_names']
    print(f'data group {i+1}')
    assert len(fea_data) == len(imnames), "数量不匹配，torch_features 和 imnames 必须具有相同的长度"
    dataset = CustomDataset(fea_data,imnames)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    overall_loss = 0
    for batch_idx, x0 ,imname in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            x = x0[0].to(device)
            x_hat, mean, log_var,x_re = model(x)


print("clustering")
#selected = hdbscan_model.fit_predict(reduce_fea)
selected=None
for j in [5,10,15,20]:
    selected=select_fuc.select_batch(features.reshape(features.shape[0], -1),int(len(features)*j*0.01),selected_idx=selected)
    print(f"save selected batch {j}")
    with open(os.path.join(args.data_path, f"uesatL_sam{j}_VAE2_kmeans_selected.txt"), "w") as file:
        # 遍历列表，写入每个字符串
        for i in selected:
            file.write(img_paths[i])

    with open(os.path.join("data/UESAT_RGB_53/MMdata",  f"sam{j}_VAE2_kmeans.txt"), "w") as file:
        # 遍历列表，写入每个字符串
        for i in selected:
            file.write(img_paths[i].rstrip('.png\n').split('/')[-1]+'\n')
print(overall_loss)
torch.save(model.state_dict(),'pretrained/VAEsam3.pth')
