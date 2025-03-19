import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import shutil
import random
data_path = '/media/buaa22/DATA/zhaoshengyun/HIL_SD/'
data_path_new = '/media/buaa22/DATA/zhaoshengyun/HIL_resized/'
data_files = os.listdir(os.path.join(data_path_new, 'images', 'training'))

# data_files = [f for f in data_files if f.endswith('masked.png')]
data_files = [f for f in data_files if f.startswith('jackal')]
def edit_mask(file):
    label_path = os.path.join(data_path_new, 'annotations', 'training', file)
    with Image.open(label_path).convert("L") as label:
        mask_array = np.array(label)

# 修改特定值
# 例如，将所有像素值为100的改为200
        mask_array[mask_array == 4] = 8

# 将数组转换回PIL图像
        mask_modified = Image.fromarray(mask_array)

# 保存或显示结果
        mask_modified.save(label_path)
        # 将标签中的 7 像素值替换为 8
        # label = label.point(lambda p: p if p != 4 else 8)

def process_image(file):
    file_path = os.path.join(data_path, 'images', 'training', file)
    label_path = os.path.join(data_path, 'annotations', 'training', file)
    
    with Image.open(file_path) as img, Image.open(label_path) as label:
        # 获取原始尺寸
        width, height = img.size
        # 计算填充尺寸
        max_dim = max(width, height)
        
        # 创建新的方形图像和标签，背景颜色为黑色
        new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        new_label = Image.new("L", (max_dim, max_dim), 0)
        
        # 将原始图像和标签粘贴到新的方形图像和标签的中心
        new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        new_label.paste(label, ((max_dim - width) // 2, (max_dim - height) // 2))
        
        # 缩放到 1024x1024
        img_resized = new_img.resize((1024, 1024))
        label_resized = new_label.resize((1024, 1024), Image.NEAREST)
        
        # 保存缩放后的图像和标签
        resized_file_path = os.path.join(data_path_new, 'images', 'training', file)
        resized_label_path = os.path.join(data_path_new, 'annotations', 'training', file)
        
        img_resized.save(resized_file_path)
        label_resized.save(resized_label_path)

# 使用 ThreadPoolExecutor 并行处理图像和标签
# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(process_image, data_files), total=len(data_files)))

# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(edit_mask, data_files), total=len(data_files)))

training_images_path = os.path.join(data_path_new, 'images', 'training')
validation_images_path = os.path.join(data_path_new, 'images', 'validation')
training_annotations_path = os.path.join(data_path_new, 'annotations', 'training')
validation_annotations_path = os.path.join(data_path_new, 'annotations', 'validation')

# 获取 training 目录中的所有文件
training_files = os.listdir(training_images_path)

# 随机采样 20% 的文件
sampled_files = random.sample(training_files, int(0.2 * len(training_files)))

# 移动文件
for file in sampled_files:
    # 移动图像文件
    src_image_file = os.path.join(training_images_path, file)
    dst_image_file = os.path.join(validation_images_path, file)
    shutil.move(src_image_file, dst_image_file)
    
    # 移动注释文件
    annotation_file = file  # 假设注释文件的命名规则
    src_annotation_file = os.path.join(training_annotations_path, annotation_file)
    dst_annotation_file = os.path.join(validation_annotations_path, annotation_file)
    shutil.move(src_annotation_file, dst_annotation_file)



# 更新 train.txt 文件
# 更新 val.txt 文件
# data_files = [f.rstrip('.png') + '\n' for f in data_files]
# with open(os.path.join(data_path, 'val.txt'), 'w') as f:
#     f.writelines(data_files)