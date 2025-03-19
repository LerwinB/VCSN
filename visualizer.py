from mmseg.apis import init_model, inference_model, show_result_pyplot
from PIL import Image
import os
# import mmcv
from tqdm import tqdm
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

palette=[[0,0,0],[255, 0, 0], [0, 255, 0],[255, 255, 0],[0, 0, 255],[255, 0, 255],[0, 0, 124],[202, 202, 202],[255, 255, 255],[0, 255, 255],[0, 124, 0]]
# palette = [[0,0,0],[255,0,0]]
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


    # vis_iamge = show_result_pyplot(model, img_path, result,show=False,with_labels=False, save_dir='data/UESAT_vis/results',out_file=img_path.replace('images', 'results'))
# config_path = 'configs/deeplabv3plus/deeplabv3plus_r50_uesat10.py'
config_path = 'configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py'
checkpoint_path = 'work_dirs/deeplabv3plus_r50_Random5_uesatrgb/epoch_5.pth'
# checkpoint_paths= {
#     'VAE3':'work_dirs/deeplabv3plus_r50_uesat10_VAE3n5/epoch_5.pth',
#     'Random':'work_dirs/deeplabv3plus_r50_uesat10_random5/epoch_5.pth',
#     'FULL':'work_dirs/deeplabv3plus_r50_uesat10_FULL/epoch_5.pth',
#     'HIL-full':'work_dirs/deeplabv3plus_r50_uesat10_HILfull/epoch_5.pth',
#     'HIL5':'work_dirs/deeplabv3plus_r50_uesat10_HIL5/epoch_5.pth',
#     'VAE+HIL5':'work_dirs/deeplabv3plus_r50_uesat10_VAE35+HIL/epoch_5.pth'
#     }
checkpoint_paths= {

    'VAE3e20':'work_dirs/deeplabv3plus_r50_sam5_VAE3_e20/epoch_20.pth',
    'Random':'work_dirs/deeplabv3plus_r50_Random5_uesatrgb/epoch_5.pth',
    'FULL':'work_dirs/deeplabv3plus_r50_full_uesatrgb/epoch_5.pth',
    'Real':'work_dirs/deeplab_e5-b8_uesat11_realtrain/epoch_5.pth',
'HIL115':'work_dirs/deeplab_e5-b8_uesat11_HILVAE5/epoch_5.pth'
    }
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
    model = init_model(config_path, checkpoint_paths[task], device='cuda:0')
    for img in tqdm(imgs):
        img_path = os.path.join(datapath,img)
        visdata(img_path, model,task)
        # vispanel(img_path, model,task)
    print('task:',task,'finished')

