import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(image_path, data_root, category_dict, image_id):
    # 加载掩码
    mask_file = os.path.join(data_root, 'Screenshots0528', image_path.replace('src', 'label_color').strip())
    img = cv2.imread(mask_file)
    height, width, _ = img.shape

    annotations = []

    for color, category_id in category_dict.items():
        # 根据颜色过滤出对应类别的掩码
        if category_id == 0:
            mask = np.all(img > color, axis=-1).astype(np.uint8)
        else:
            mask = np.all(img == color, axis=-1).astype(np.uint8)

        # 计算边界框和分割区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 6:
                continue  # 跳过小的噪声轮廓
            x, y, w, h = cv2.boundingRect(contour)
            #segmentation = contour.flatten().tolist()

            # 创建标注
            annotation = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                #"segmentation": [segmentation],
                "area": w * h,
                "iscrowd": 0
            }
            annotations.append(annotation)

    return image_id, annotations, height, width

def convert_mask_to_coco(data_root, image_paths, category_dict):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "spacecraft"} 
            for i, name in enumerate([
                "satellite", "panel", "aerial", "spout", "camera", 
                "panel support", "star sensor", "docking", "body", 
                "arm", "part", "rodaerial", "unkown aerial", "others"
            ])
        ]
    }

    annotation_id = 1  # 初始化 annotation_id
    image_id = 1  # 初始化 image_id

    # 使用多线程处理图像
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, img_path, data_root, category_dict, image_id): img_path for image_id, img_path in enumerate(image_paths, start=1)}

        for future in tqdm(as_completed(futures), total=len(futures)):
            img_id, annotations, height, width = future.result()
            coco_output["images"].append({"id": img_id, "file_name": os.path.basename(futures[future]).strip(), "width": width, "height": height})
            for annotation in annotations:
                annotation["id"] = annotation_id
                coco_output['annotations'].append(annotation)
                annotation_id += 1  # 递增 annotation_id

    return coco_output

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

with open('tools/src_path_minival.txt') as file:
    paths=file.readlines()
data_root="data/UESAT_RGB_53"
# 调用函数生成 COCO 标签
coco_data = convert_mask_to_coco(data_root,paths, color2index)

# 保存到 JSON 文件
with open(os.path.join(data_root,"uesat_detection_coco.json"), "w") as f:
    json.dump(coco_data, f, indent=4)