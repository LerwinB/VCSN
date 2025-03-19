import os
import shutil
from tqdm import tqdm
import random

data_root='../data/UESAT_RGB_53'

with open('src_path_all.txt') as file:
    paths=file.readlines()
minipaths=random.sample(paths,len(paths)//1000)
for path in tqdm(minipaths):
    path=path.rstrip('\n')
    old1=os.path.join(data_root,'Screenshots0528',path)
    old2=os.path.join(data_root,'Screenshots0528',path.replace('src','label'))
    newpath=path.split('/')[-1]
    new1=os.path.join(data_root,'MMdata/images/test1',newpath)
    new2=os.path.join(data_root,'MMdata/annotations/test1',newpath)
    shutil.copy2(old1,new1)
    shutil.copy2(old2,new2)