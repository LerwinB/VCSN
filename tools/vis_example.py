import os
from collections import defaultdict
import random
import shutil

data_root="data/UESAT_RGB_53/MMdata/images/train"
data_paths=os.listdir(data_root)
imgs=[]

grouped_strings = defaultdict(list)

# 将字符串按前两位数字分组
for s in data_paths:
    if 'd0_L0' in s:
        prefix = s[:2]
        grouped_strings[prefix].append(s)

# 从每组中随机选择一个字符串
random_selections = []
for prefix, strings in grouped_strings.items():
    random_selections.append(random.choice(strings))

for select in random_selections:
    shutil.copy2(os.path.join(data_root,select),os.path.join('data/vis_example/src',select))