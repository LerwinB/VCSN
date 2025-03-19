import os
data_root='data/UESAT_RGB_53/MMdata'
txtpath='sam5_noUmap_kmeans.txt'
with open(os.path.join(data_root,txtpath)) as file:
    paths=file.readlines()
a=set(paths)
for path in paths:
    fullpath=os.path.join(data_root,'images','train',path.rstrip('\n')+'.png')
    fullpath1=os.path.join(data_root,'annotations','train',path.rstrip('\n')+'.png')
    if not os.path.exists(fullpath):
        print(fullpath)
    if not os.path.exists(fullpath1):
        print(fullpath1)