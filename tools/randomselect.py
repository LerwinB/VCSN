import os
import random
data_root = 'data/coco164k'
with open(os.path.join(data_root,'coco_all.txt'),'r') as file:
    paths=file.readlines()
pathsnew=[]
paths20=random.sample(paths,len(paths)//5)
paths15=random.sample(paths20,int(len(paths)*0.15))
paths10=random.sample(paths15,len(paths)//10)
paths5=random.sample(paths10,len(paths)//20)
pathdict = {5:paths5, 10:paths10,15:paths15,20:paths20}
for i in [5,10,15,20]:
    with open(os.path.join(data_root,f'COCO_random{i}.txt'),'w') as f:
        f.writelines(pathdict[i])
    files =[]
# for path in paths:
#     if path.rstrip('\n'):
#         pathsnew.append(path.rstrip('.png\n').split('/')[-1]+'\n')
# with open('data/UESAT_RGB_53/MMdata/uesatL_MMdata_full.txt','w') as file:
#     file.writelines(pathsnew)
