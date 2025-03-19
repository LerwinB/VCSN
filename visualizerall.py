from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmdet.apis import init_detector, inference_detector
import os
# import mmcv
from tqdm import tqdm
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

# palette=[[255, 0, 0], [0, 255, 0],[255, 255, 0],[0, 0, 255],[255, 0, 255],[0, 0, 124],[202, 202, 202],[255, 255, 255],[0, 255, 255],[0, 124, 0]]
palette = [[0,0,0],[255,0,0]]
def visdata(img_path, model,task):
    # 推理给定图像
    result = inference_model(model,img_path)
    # result to image 
    seg_map = result.pred_sem_seg.data.cpu().numpy()
    seg_map = seg_map.squeeze(0)
    # color_seg = mmcv.visualization.palette2rgb(seg_map, palette)
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_map == label, :] = color

    out_file = img_path.replace('image', 'result').replace('.png', task+'.png')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    Image.fromarray(color_seg).save(out_file)

def vislabel(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    label = Image.open(img_path.replace('image', 'label')).convert('L')
    label = np.array(label)
    color_seg = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        color_seg[label == idx, :] = color
    out_file = img_path.replace('image', 'result').replace('.png', 'label.png')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    Image.fromarray(color_seg).save(out_file)
    original_image = cv2.imread(img_path)
    # 颜色设定：例如将目标类别标为红色
    overlay_color = np.array([0, 0, 255])  # 红色
    # 将掩膜应用到颜色上
    colored_mask = np.zeros_like(original_image)
    for i in range(3):  # R, G, B 叠加颜色
        colored_mask[:, :, i] = label * overlay_color[i]

    # 将掩膜以一定透明度叠加到原图上
    alpha = 0.5  # 透明度
    overlayed_image = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

    cv2.imwrite(img_path.replace('image', 'vis').replace('.png', 'label_vis.png'), overlayed_image)

def vispanel(img_path, model,task):
    result = inference_model(model,img_path)
    seg_map = result.pred_sem_seg.data.cpu().numpy()
    seg_map = seg_map.squeeze(0)
    # 目标类别
    target_class = 1
    original_image = cv2.imread(img_path)
    # 生成掩膜，只保留目标类别
    mask = (seg_map == target_class).astype(np.uint8)
    mask_out_file = img_path.replace('image', 'result').replace('.png', task+'_mask.png')
    red_mask = np.zeros_like(original_image)
    red_mask[:, :, 2] = mask * 255  # 只保留红色通道
    cv2.imwrite(mask_out_file, red_mask)  # 保存掩膜图像

    # 颜色设定：例如将目标类别标为红色
    overlay_color = np.array([0, 0, 255])  # 红色
      # 读取原图
    # 将掩膜应用到颜色上
    colored_mask = np.zeros_like(original_image)
    for i in range(3):  # R, G, B 叠加颜色
        colored_mask[:, :, i] = mask * overlay_color[i]

    # 将掩膜以一定透明度叠加到原图上
    alpha = 0.5  # 透明度
    overlayed_image = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
    cv2.imwrite(img_path.replace('image', 'result').replace('.png', task+'.png'), overlayed_image)


def visdet(img_path, model,task):
    # 推理给定图像
    out_file = img_path.replace('image', 'result').replace('.png', task+'det.png')
    result = inference_detector(model,img_path)
    # show_result_pyplot(model, img, result, score_thr=0.3)
    # show_result_pyplot(model, img_path, result)
    # 解析边界框和标签
    bbox_results, _ = result if isinstance(result, tuple) else (result, None)
    bbox_result = bbox_results.pred_instances  # 只取第一张图像的结果
    label_names = [
    'camera', 'aerial','spout','panel','panel support','star sensor','docking','body','star sensor','part','rodaerial','unkown aerial','others'
    ]
    # bboxes = np.vstack(bbox_result)
    bboxes = bbox_result.bboxes.cpu()
    labels = bbox_result.labels.cpu()
    scores = bbox_result.scores.cpu()
    # 设置置信度阈值
    score_thr = 0.7
    class_names = getattr(model, 'CLASSES', ['panel', 'class2', 'class3'])  # 获取类别名称
    img=cv2.imread(img_path)
    # 绘制边界框
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        label = labels[i]
        score = scores[i]

        # 如果得分低于阈值则跳过
        if score < score_thr:
            continue

        # 提取边界框坐标
        x1, y1, x2, y2 = map(int, bbox)

        # 选择颜色和标签文本
        color = (0, 255, 0)  # 绿色，用于表示边界框
        label_text = label_names[label] if label < len(label_names) else "unknown"
        label_with_score = f'{label_text}, Score: {score:.2f}'

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 在边界框上方绘制标签
        cv2.putText(img, label_with_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存图像

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转回 BGR 格式
    cv2.imwrite(out_file, img)


    # vis_iamge = show_result_pyplot(model, img_path, result,show=False,with_labels=False, save_dir='data/UESAT_vis/results',out_file=img_path.replace('images', 'results'))
# config_path = 'configs/deeplabv3plus/deeplabv3plus_r50_uesat10.py'
# config_path = 'configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py'
config_path = 'det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py'
checkpoint_path = 'work_dirs/deeplabv3plus_r50_Random5_uesatrgb/epoch_5.pth'

checkpoint_paths= {
        'Full':'work_dirs_det/yolov3_full2_uesat/epoch_14.pth',
}
# checkpoint_paths= {
#     'VAE3':'work_dirs/deeplabv3plus_r50_uesat10_VAE3n5/epoch_5.pth',
#     'Random':'work_dirs/deeplabv3plus_r50_uesat10_random5/epoch_5.pth',
#     'FULL':'work_dirs/deeplabv3plus_r50_uesat10_FULL/epoch_5.pth',
#     'HIL-full':'work_dirs/deeplabv3plus_r50_uesat10_HILfull/epoch_5.pth',
#     'HIL5':'work_dirs/deeplabv3plus_r50_uesat10_HIL5/epoch_5.pth',
#     'VAE+HIL5':'work_dirs/deeplabv3plus_r50_uesat10_VAE35+HIL/epoch_5.pth'
#     }
# checkpoint_paths= {

#     'VAE3e20':'work_dirs/deeplabv3plus_r50_sam5_VAE3_e20/epoch_20.pth',
#     'Random':'work_dirs/deeplabv3plus_r50_Random5_uesatrgb/epoch_5.pth',
#     'FULL':'work_dirs/deeplabv3plus_r50_full_uesatrgb/epoch_5.pth',
#     'Real':'work_dirs/deeplab_e5-b8_uesat11_realtrain/epoch_5.pth',
# 'HIL115':'work_dirs/deeplab_e5-b8_uesat11_HILVAE5/epoch_5.pth'
#     }
# checkpoint_paths= {
#     'VAE3':'work_dirs/deeplab_e5-b8_uesat11_VAEn_5/epoch_5.pth',
#     'VAE311':'work_dirs/deeplab_e5-b8_uesat11_VAEn_5/epoch_5.pth',
#     'VAE11+HIL5':'work_dirs/deeplab_e5-b8_uesat11_VAE3n5+HIL5/epoch_5.pth',
    
    # }
# datapath='data/Realdataset/image'
datapath='data/UESAT_vis/images'
os.makedirs(datapath.replace('image','result'),exist_ok=True)
allimgs=os.listdir(datapath)
# imgs=random.sample(allimgs,20)
# imgs=['0024.png','0058.png','0076.png','0014.png','0016.png','Soyuz0213.png']
imgs=allimgs
# for img in tqdm(imgs):
#     img_path = os.path.join(datapath,img)
#     vislabel(img_path)
for task in checkpoint_paths:
    model = init_detector(config_path, checkpoint_paths[task], device='cuda:1')
    for img in tqdm(imgs):
        img_path = os.path.join(datapath,img)
        visdet(img_path, model,task)
    print('task:',task,'finished')

