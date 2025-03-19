import tensorflow as tf
print(tf.__version__)
from tensorflow.python.client import device_lib

# 查看可用设备列表
print(device_lib.list_local_devices())
import keras
print(keras.__version__)
import keras.backend as K
import keras.layers as KL
import keras.engine as KE

import os
if os.path.exists('/MMdata/images'):
    print('data right')
if os.path.exists('/MMdata/MMdata'):
    print('/MMdata/MMdata')
from urso import Urso ,Uesat
from config import Config
import utils
import cv2

dataset = Uesat()
config = Config()
dataset.load_dataset('/MMdata', config, "sam5_VAE3_kmeans_pose.txt")
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
image_id = random.choice(dataset.image_ids)
image_original = dataset.load_image(image_id)
#image_path= '/media/computer/study/CZL/code training/pose and motion tracking through optical_flow/datasets/test/gassp1_174.png'
#image_original = Image.open(image_path)
loc_gt = dataset.load_location(image_id)
q_gt = dataset.load_quaternion(image_id)
print(loc_gt)
# Retrieve K (calibration matrix)
# K = dataset.camera.K
fov_x = 0.08
fov_y = 0.08
width = 1024  # number of horizontal[pixels]
height = 1024  # number of vertical[pixels]
# Focal lengths
fx = width / (2 * np.tan(fov_x / 2))
fy = - height / (2 * np.tan(fov_y / 2))

K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
print(K)
#fov_x = 90.0 * np.pi / 180
#fov_z = 73.7 * np.pi / 180
#width = 1280  # number of horizontal[pixels]
#height = 960  # number of vertical[pixels]
# Focal lengths
#fx = width / (2 * np.tan(fov_x / 2))
#fz =  -height / (2 * np.tan(fov_z / 2))
#K = np.matrix([[fx, width / 2, 0], [0, height / 2, fz], [0, 1, 0]])
# 3. Visualize original image + gt
fig, ax_1 = plt.subplots(1,1,figsize=(12, 8))
ax_1.imshow(image_original)
ax_1.set_xticks([])
ax_1.set_yticks([])
axis_length = 200
utils.visualize_axes(ax_1, q_gt, loc_gt, K, axis_length)
#visualize_axes is left handvisualize_axes1 is right hand 
plt.savefig('0000test.png')
print('success')