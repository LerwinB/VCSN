import os
import numpy as np
from PIL import Image
from tqdm import tqdm
def calculate_label_distribution(data_dir):
    """
    统计数据集中每个标签（不包括0背景）的像素占比。

    参数：
    data_dir (str): 标签图像所在目录，每个像素的值代表标签。

    返回：
    dict: 标签占比，每个标签对应其占比（不包括背景）。
    """
    label_counts = {}
    total_pixels = 0

    # 遍历所有图像
    for file_name in tqdm(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        
        # 打开图像并转换为numpy数组
        label_image = np.array(Image.open(file_path))
        
        # 统计图像中各个标签的像素数量
        unique_labels, counts = np.unique(label_image, return_counts=True)
        
        # 更新统计结果，忽略背景标签0
        for label, count in zip(unique_labels, counts):
            if label != 0:  # 忽略背景标签
                label_counts[label] = label_counts.get(label, 0) + count
                total_pixels += count

    # 计算占比
    label_distribution = {label: count / total_pixels for label, count in label_counts.items()}
    return label_distribution

# 使用示例
data_dir = "data/HIL_resized/annotations/validation"  # 替换为标签图像所在目录
label_distribution = calculate_label_distribution(data_dir)
print("各标签的像素占比（不包括背景）:", label_distribution)
