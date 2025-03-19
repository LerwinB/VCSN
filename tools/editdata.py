import os
import shutil
from tqdm import tqdm
import random
data_root='../data/UESAT_RGB_53'

with open('src_path_all.txt') as file:
    paths=file.readlines()
for path in tqdm(paths):
    if 'GSSAP' in path:
        path=path.rstrip('\n')
        old1=os.path.join(data_root,'Screenshots0528',path)
        old2=os.path.join(data_root,'Screenshots0528',path.replace('src','label'))
        newpath=path.split('/')[-1]
        new1=os.path.join(data_root,'MMdata/images/train',newpath)
        new2=os.path.join(data_root,'MMdata/annotations/train',newpath)
        shutil.copy2(old1,new1)
        shutil.copy2(old2,new2)
with open('src_path_all_test.txt') as file:
    paths=file.readlines()
minipaths=random.sample(paths,len(paths)//100)
for path in tqdm(minipaths):
    if 'GSSAP' in path:
        path=path.rstrip('\n')
        old1=os.path.join(data_root,'Screenshots0528',path)
        old2=os.path.join(data_root,'Screenshots0528',path.replace('src','label'))
        newpath=path.split('/')[-1]
        new1=os.path.join(data_root,'MMdata/images/minitest',newpath)
        new2=os.path.join(data_root,'MMdata/annotations/minitest',newpath)
        shutil.copy2(old1,new1)
        shutil.copy2(old2,new2)
print('finish')
