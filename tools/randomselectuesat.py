import os
import random
data_root = 'data/UESAT_RGB_53/MMdata'
with open(os.path.join('tools/src_path_all.txt'),'r') as file:
    paths0=file.readlines()
    paths = [path.rstrip('.png\n').split('/')[-1]+'\n' for path in paths0]
pathsnew=[]
percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# 计算每个百分比对应的索引位置，并生成子集
random.shuffle(paths)
grouped_data = []
for p in percentages:
    index = int(len(paths) * p / 100)  # 计算当前百分比对应的索引位置
    subset = paths[:index]  # 截取前index个元素作为子集
    grouped_data.append(subset)

# 输出分组结果
for i, group in enumerate(grouped_data):
    with open(os.path.join(data_root,f'uesat_Random{percentages[i]}.txt'),'w') as f:
        f.writelines(group)
    print(f"{percentages[i]}%: {len(group)}")
