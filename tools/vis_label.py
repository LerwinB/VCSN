import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# 定义 palette，每个标签对应的颜色
palette = [
    [0, 0, 0],       # 背景
    [255, 0, 0],     # 类别 1
    [0, 255, 0],     # 类别 2
    [255, 255, 0],   # 类别 3
    [0, 0, 255],     # 类别 4
    [255, 0, 255],   # 类别 5
    [0, 0, 124],     # 类别 6
    [202, 202, 202], # 类别 7
    [255, 255, 255], # 类别 8
    [0, 255, 255],   # 类别 9
    [0, 124, 0]      # 类别 10
]

def apply_palette(label_array):
    """
    根据给定的 palette 将标签数组映射为彩色图像。
    
    Args:
        label_array (numpy.ndarray): 标签图的数组表示.
        
    Returns:
        Image: 使用 palette 映射的彩色图像.
    """
    # 创建一个彩色的空数组，用于存放调色后的标签
    color_label = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    
    # 遍历 palette 中的每个颜色，并将标签映射到颜色
    for label_value, color in enumerate(palette):
        color_label[label_array == label_value] = color

    return Image.fromarray(color_label)

def overlay_image_and_label(image_path, label_path, output_path):
    """
    将标签图与原始图像叠加，并保存到指定输出路径。

    Args:
        image_path (str): 原始图像路径。
        label_path (str): 标签图路径。
        output_path (str): 叠加图像的保存路径。
    """
    image = Image.open(image_path).convert("RGBA")
    label = Image.open(label_path).convert("L")  # 单通道灰度图
    label_array = np.array(label)

    # 应用 palette 映射标签
    label_color = apply_palette(label_array).convert("RGBA")
    
    # 设置标签图的透明度
    label_color.putalpha(128)  # 半透明

    # 叠加标签图到原图上
    overlay = Image.alpha_composite(image, label_color)

    # 保存叠加后的图像
    overlay = overlay.convert("RGB")  # 转为 RGB 保存
    overlay.save(output_path)
# print(f"已保存: {output_path}")

def process_images(input_image_dir, input_label_dir, output_dir):
    """
    多线程处理多个图像，将标签图叠加到原图上并保存。

    Args:
        input_image_dir (str): 原始图像文件夹路径。
        input_label_dir (str): 标签图文件夹路径。
        output_dir (str): 叠加图像的保存文件夹路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像和标签文件
    image_files = [f for f in os.listdir(input_image_dir) if f.startswith('eagel')]


    # 确保图像和标签文件名对应
    image_files.sort()


    # 多线程处理
    # with ThreadPoolExecutor() as executor:
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_image_dir, image_file)
        label_path = os.path.join(input_label_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        overlay_image_and_label(image_path, label_path, output_path)
        # executor.submit(overlay_image_and_label, image_path, label_path, output_path)

# 使用示例
input_image_dir = 'data/HIL_resized/images/training'
input_label_dir = 'data/HIL_resized/annotations/training'
output_dir = 'data/HIL_resized/eaglevis'

process_images(input_image_dir, input_label_dir, output_dir)
