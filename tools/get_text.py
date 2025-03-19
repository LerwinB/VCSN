'''
    automate describe image into text
'''
import os
from tqdm import tqdm
import random
import json
from string import Template
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
def get_direction_from_euler_camera(yaw, pitch, roll):
    """
    根据欧拉角获取物体在相机坐标系中的前后左右上下方向
    """
    # 初始前方向向量（相机坐标系下的Z轴负方向）
    r = R.from_euler('xyz', [[yaw, pitch, roll]], degrees=True)
    R_m = r.as_matrix()
    Y1 = R_m[0][1]
    max_index = np.argmax(np.abs(Y1))
    if Y1[max_index]>0:
        mark =0
    else:
        mark =1

    directions = ['front','back','right','left','top','bottom']

    
    return directions[max_index*2+mark]
def get_appends(path):
    
    mask_image = Image.open(path)
    mask_array = np.array(mask_image)
    unique_values = np.unique(mask_array)
    objs=[]

    for value in unique_values:
        part_name = class_mapping.get(value, 'Unknown Part')
        objs.append(part_name)
    result_string = ", ".join(objs)
    return result_string


class_mapping = {
    0: 'background',
    1: 'panel',
    2: 'aerial',
    3: 'spout',
    4: 'camera',
    5: 'panel support',
    6: 'star sensor',
    7: 'docking',
    8: 'body',
    9: 'arm',
    10: 'part',
    11: 'rodaerial',
    12: 'unknown aerial',
    13: 'others'
}

data_root='data/UESAT_RGB_53'
template = Template("The image depicts a satellite named '$name' in dark space. The image shows the $direct side of the satellite. The satellite extends additional appendages, include $appendages.")
with open(os.path.join(data_root,'Screenshots0528','src_path_all.txt')) as file:
    paths=file.readlines()

for path in tqdm(paths):
    path=path.rstrip('.png\n')
    json_path=os.path.join(data_root,'Screenshots0528',path.replace('src','json')+'.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    satname = data['SatInfo']['SatName']
    satrot = data['Location']['SatRot']
    direction = get_direction_from_euler_camera(satrot['X'],satrot['Y'],satrot['Z'])
    mask_path=os.path.join(data_root,'Screenshots0528',path.replace('src','label')+'.png')

    appendage = get_appends(mask_path)
    formatted_string = template.substitute(name=satname[2:], appendages=appendage,direct=direction)
    newpath=path.split('/')[-1]
    txt_path=os.path.join(data_root,'MMdata/text',newpath+'.txt')
    with open(txt_path, 'w') as file:
        file.write(formatted_string)