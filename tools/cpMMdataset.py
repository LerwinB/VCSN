import os
import shutil
from tqdm import tqdm
import random
data_root='data/UESAT_RGB_53'
'''
with open('src_path_all.txt') as file:
    paths=file.readlines()
for path in tqdm(paths):
    path=path.rstrip('\n')
    old1=os.path.join(data_root,'Screenshots0528',path)
    old2=os.path.join(data_root,'Screenshots0528',path.replace('src','label'))
    newpath=path.split('/')[-1]
    new1=os.path.join(data_root,'MMdata/images/train',newpath)
    new2=os.path.join(data_root,'MMdata/annotations/train',newpath)
    shutil.copy2(old1,new1)
    shutil.copy2(old2,new2)'''
with open('tools/src_path_all_test.txt') as file:
    paths=file.readlines()
minipaths=random.sample(paths,50000)
with open('tools/src_path_allval.txt','w') as file:
    file.writelines(minipaths)
for path in tqdm(minipaths):
    path=path.rstrip('\n')
    old1=os.path.join(data_root,'Screenshots0528',path)
    old2=os.path.join(data_root,'Screenshots0528',path.replace('src','label'))
    newpath=path.split('/')[-1]
    new1=os.path.join(data_root,'MMdata/images/allval',newpath)
    new2=os.path.join(data_root,'MMdata/annotations/allval',newpath)
    shutil.copy2(old1,new1)
    shutil.copy2(old2,new2)
print('finish')
