import cv2
import numpy as np
import json
import os
from tqdm import tqdm

def convert_mask_to_coco(image_path, category_dict, image_id):
    # 1. 加载掩码
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # 初始化 COCO 格式的结构
    coco_output = {
        "images": [{"id": image_id, "file_name": image_path, "width": width, "height": height}],
        "annotations": [],
        "categories": [{"id": id, "name": name} for name, id in category_dict.items()]
    }

    annotation_id = 1  # 初始化 annotation_id

    for color, category_id in category_dict.items():
        # 2. 根据颜色过滤出对应类别的掩码
        mask = np.all(img == color, axis=-1).astype(np.uint8)
        
        # 3. 计算边界框和分割区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 6:
                continue  # 跳过小的噪声轮廓
            x, y, w, h = cv2.boundingRect(contour)
            segmentation = contour.flatten().tolist()

            # 创建标注
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "segmentation": [segmentation],
                "area": w * h,
                "iscrowd": 0
            }
            coco_output['annotations'].append(annotation)
            annotation_id += 1  # 递增 annotation_id

    return coco_output
# 路径设置
output_json = 'uesat_detection.json'  # 输出的JSON文件
data_root='data/UESAT_RGB_53'
# 获取所有掩码图像的文件列表
with open('tools/src_path_all_test.txt') as file:
    paths=file.readlines()

# 初始化COCO格式的字典
coco_format = {
    "info": {
        "description": "AMSD Dataset",
        "url": "http://example.com",
        "version": "1.0",
        "year": 2024,
        "contributor": "BUAA-SHMCT",
        "date_created": "2024-07-02"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Example License",
            "url": "http://example.com/license"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": []
}
classes=( 'background', 'panel', 'aerial','spout','camera','panel support','star sensor','docking','body','arm','part','rodaerial','unkown aerial','others'),
palette=[[0, 0, 0],[255, 0, 0], [0, 255, 0],[255, 255, 0],[0, 0, 255],[255, 0, 255],[0, 0, 124],[202, 202, 202],[255, 255, 255],[124, 0, 0],[0, 124, 0],[124, 124, 0],[0, 124, 124],[0, 255, 255]]

# 假设我们有固定的类别信息
color2index = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (255, 255, 0): 3,
    (0, 0, 255): 4,
    (255, 0, 255): 5,
    (0, 0, 124): 6,
    (255, 0, 0): 1,
    (202, 202, 202): 7,
    (255, 255, 255): 8,
    (124, 0, 0): 9,
    (0, 185, 0): 10,
    (185, 185, 0): 11,
    (0, 124, 124): 12,
    (0, 255, 255): 13,
}

categories = [
    {"id": 0, "name": "satellite", "supercategory": "spacecraft"},
    {"id": 1, "name": "panel", "supercategory": "spacecraft"},
    {"id": 2, "name": "aerial", "supercategory": "spacecraft"},
    {"id": 3, "name": "spout", "supercategory": "spacecraft"},
    {"id": 4, "name": "camera", "supercategory": "spacecraft"},
    {"id": 5, "name": "panel support", "supercategory": "spacecraft"},
    {"id": 6, "name": "star sensor", "supercategory": "spacecraft"},
    {"id": 7, "name": "docking", "supercategory": "spacecraft"},
    {"id": 8, "name": "body", "supercategory": "spacecraft"},
    {"id": 9, "name": "arm", "supercategory": "spacecraft"},
    {"id": 10, "name": "part", "supercategory": "spacecraft"},
    {"id": 11, "name": "rodaerial", "supercategory": "spacecraft"},
    {"id": 12, "name": "unkown aerial", "supercategory": "spacecraft"},
    {"id": 13, "name": "others", "supercategory": "spacecraft"},
    # 添加更多类别
]

categories1 = [
    {"id": 1, "name": "satellite", "supercategory": "spacecraft"},
    {"id": 2, "name": "Space Probes", "supercategory": "spacecraft"},
    {"id": 3, "name": "Space Stations", "supercategory": "spacecraft"},
    {"id": 4, "name": "Crewed Spacecraft", "supercategory": "spacecraft"},
    {"id": 5, "name": "Cargo Spacecraft", "supercategory": "spacecraft"},
    {"id": 6, "name": "earth", "supercategory": "Celestial Body"},
    {"id": 7, "name": "moon", "supercategory": "Celestial Body"},
    {"id": 8, "name": "meteorite", "supercategory": "Celestial Body"},
    # 添加更多类别
]
coco_format["categories"] = categories

image_id = 1
annotation_id = 1

for path in tqdm(paths):
    # 读取掩码图像
    mask_file=os.path.join(data_root,'Screenshots0528',path.replace('src','label_color').strip())
    convert_mask_to_coco(mask_file,color2index,image_id=image_id,)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    file_name = os.path.basename(mask_file)
    
    # 添加图像信息
    coco_format["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    })
    
    # 找到当前标签的所有像素
    obj_mask = (mask != 0).astype(np.uint8)
    
    # 查找对象的轮廓
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, w, h]
        area = w * h
        
        # 添加标注信息
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # 假设标签值对应类别ID
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        annotation_id += 1
    
    image_id += 1

# 写入JSON文件
with open(output_json, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO格式的JSON文件已生成: {output_json}")
