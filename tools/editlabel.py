import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_image(filename, input_dir, output_dir, label_map):
    if filename.endswith('.png'):  # 假设标签图是 .png 格式
        # 打开标签图并转换为 numpy 数组
        label_path = os.path.join(input_dir, filename)
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # 根据 label_map 调整标签
        adjusted_label = label_array.copy()
        for old_label, new_label in label_map.items():
            adjusted_label[label_array == old_label] = new_label

        # 保存调整后的标签图
        adjusted_label_img = Image.fromarray(adjusted_label.astype(np.uint8))
        output_path = os.path.join(output_dir, filename)
        adjusted_label_img.save(output_path)

def main(input_dir, output_dir, label_map):
    os.makedirs(output_dir, exist_ok=True)

    filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda filename: process_image(filename, input_dir, output_dir, label_map), filenames), total=len(filenames)))

if __name__ == "__main__":

    
    label_map = {
        10: 10,
        11: 2,
        12: 2,
        13: 9,
    }
    
    datapath = 'data/UESAT_RGB_53/MMdata/annotations/allval'  # 原始标签图文件夹路径
    outputpath = 'data/UESAT_RGB_53/MMdata/labels/allval'  # 调整后标签图的保存路径
    main(datapath, outputpath, label_map)