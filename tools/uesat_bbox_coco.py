import numpy as np
import cv2
import json
import os

from tqdm import tqdm

def convert_mask_to_coco(data_root, image_paths, category_dict):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
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
        ],
    }

    annotation_id = 1  # 初始化 annotation_id
    image_id = 1  # 初始化 image_id

    for image_path in tqdm(image_paths):
        # 1. 加载掩码
        mask_file=os.path.join(data_root,'Screenshots0528',image_path.replace('src','label_color').strip())
        img = cv2.imread(mask_file)
        height, width, _ = img.shape

        # 添加图像信息
        coco_output["images"].append({"id": image_id, "file_name": os.path.basename(image_path), "width": width, "height": height})

        for color, category_id in category_dict.items():
            # 2. 根据颜色过滤出对应类别的掩码
            if category_id == 0:
                mask = np.all(img > color, axis=-1).astype(np.uint8)
            else:
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

        image_id += 1  # 递增 image_id

    return coco_output

# 示例字典：掩码颜色（RGB）和类别 ID 对应关系
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
# 图片路径列表

with open('tools/src_path_all_test.txt') as file:
    paths=file.readlines()
data_root="data/UESAT_RGB_53"
# 调用函数生成 COCO 标签
coco_data = convert_mask_to_coco(data_root,paths, color2index)

# 保存到 JSON 文件
with open("uesat_detection_coco.json", "w") as f:
    json.dump(coco_data, f, indent=4)