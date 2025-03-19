import json
import os
import csv
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler2quat(roll, pitch, yaw):
    """
    将欧拉角转换为四元数
    :param roll: 滚转角
    :param pitch: 俯仰角
    :param yaw: 偏航角
    :return: 四元数 [qx, qy, qz, qw]
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def quat2euler(quat):
    """
    将四元数转换为欧拉角
    :param quat: 四元数 [qx, qy, qz, qw]
    :return: 欧拉角 [roll, pitch, yaw]
    """
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=True)

def euler2quatt(pitch, yaw, roll):
    """Convert euler angles in degrees to a quaternion"""

    cos_pitch = np.cos(pitch * 0.5)
    sin_pitch = np.sin(pitch * 0.5)
    cos_yaw = np.cos(yaw * 0.5)
    sin_yaw = np.sin(yaw * 0.5)
    cos_roll = np.cos(roll * 0.5)
    sin_roll = np.sin(roll * 0.5)

    qx = -sin_yaw * cos_roll * cos_pitch - cos_yaw * sin_roll * sin_pitch
    qy = -cos_yaw * sin_roll * cos_pitch + sin_yaw * cos_roll * sin_pitch
    qz = sin_yaw * sin_roll * cos_pitch - cos_yaw * cos_roll * sin_pitch
    qw = cos_yaw * cos_roll * cos_pitch + sin_yaw * sin_roll * sin_pitch

    return np.array([qx, qy, qz, qw])

r_camera, p_camera, y_camera = 0, 0, -90
r_camera = np.radians(r_camera)
p_camera = np.radians(p_camera)
y_camera = np.radians(y_camera)
quat_camera = euler2quat(r_camera, p_camera, y_camera)
quat_camera_inv = R.from_quat(quat_camera).inv().as_quat()

with open('tools/src_path_minival.txt') as file:
    paths=file.readlines()
data_root="data/UESAT_RGB_53/Screenshots0528"
jsonpaths = [os.path.join(data_root,path.replace('src','json').replace('.png\n','.json')) for path in paths]
with open('tools/pose_minival_gt.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # 写入标题行
    writer.writerow(['filename','x', 'y', 'z', 'q1', 'q2', 'q3','q4','raw','pitch','yaw'])

    for jsonpath in tqdm(jsonpaths):
        with open(jsonpath, 'r') as file:
            data = json.load(file)

        # 提取 CameraLoc 和 SatRot
        camera_loc = data['Location']['CameraLoc']
        sat_rot = data['Location']['SatRot']
        r_target, p_target, y_target = sat_rot['X'],sat_rot['Y'],sat_rot['Z']
        r_target = np.radians(r_target)
        p_target = np.radians(p_target)
        y_target = np.radians(y_target)
        quat_target = euler2quat(r_target, p_target, y_target)
        quat_relative = R.from_quat(quat_camera_inv) * R.from_quat(quat_target)
        euler_relative = quat2euler(quat_relative.as_quat())

        # print(euler_relative)
        r=euler_relative[0]
        p=euler_relative[1]
        y=euler_relative[2]
        # r,p,y=-39.4406,-11.1028, -139.621
        # 度转弧度
        r = np.radians(r)
        p = np.radians(p)
        y = np.radians(y)
        sat_quat=euler2quatt(-r,-p,y)

        # sat_quat = quat_relative.as_quat()

        writer.writerow([
                os.path.basename(jsonpath.rstrip('.json')),
                camera_loc['X'], camera_loc['Z'], camera_loc['Y'], 
                sat_quat[0], sat_quat[1], sat_quat[2], sat_quat[3],euler_relative[0],euler_relative[1],euler_relative[2]
            ])
    print(jsonpaths[0])