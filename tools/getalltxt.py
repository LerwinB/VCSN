import os
import random
data_root="data/UESAT_RGB_53/Screenshots0528"
dirs = os.listdir(data_root)
paths_all=[]
paths_test=[]
for dir in dirs:
    with open(os.path.join(data_root,dir,'srcpath.txt')) as file:
        paths = file.readlines()
        paths_new = [path[2:] for path in paths]
    test_size = int(len(paths_new) * 0.2)
    test_set = random.sample(paths_new,test_size)
    train_set = [item for item in paths_new if item not in test_set]
    paths_all.extend(train_set)
    paths_test.extend(test_set)

with open("tools/src_path_all.txt",'w') as file:
    file.writelines(paths_all)

with open("tools/src_path_all_test.txt",'w') as file:
    file.writelines(paths_test)

