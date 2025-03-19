import os

with open('work_dirs/Deeplabv3_ULAL/ULAL10.txt') as file:
    paths=file.readlines()
pathsnew=[]
for path in paths:
    if path.rstrip('\n'):
        pathsnew.append(path.rstrip('.png\n').split('/')[-1]+'\n')
with open('data/UESAT_RGB_53/MMdata/ULAL10.txt','w') as file:
    file.writelines(pathsnew)
    