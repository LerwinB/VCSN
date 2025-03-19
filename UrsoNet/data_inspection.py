from urso import Urso ,Uesat
from config import Config
import utils
import cv2

# Dataset location
dataset_path = 'data/UESAT_RGB_53/MMdata'

# Load dataset
# dataset = Urso()
dataset = Uesat()
config = Config()
# dataset.load_dataset(dataset_path, config, "sam5_VAE3_kmeans_pose.txt")
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
# Load image and gt pose
# image_id = random.choice(dataset.image_ids)
# image_original = dataset.load_image(image_id)
#image_path= '/media/computer/study/CZL/code training/pose and motion tracking through optical_flow/datasets/test/gassp1_174.png'
#image_original = Image.open(image_path)
# loc_gt = dataset.load_location(image_id)
# q_gt = dataset.load_quaternion(image_id)
datapath='data/UESAT_vis/images/45LUCAS_c1_d0_L2_000830.png'
loc_gt = np.array([0,0,1000])
q_gt1 = np.array([-0.13876961136098365,0.6920979383338811,0.7083382686560569,0.000579546419])#pred
q_gt = np.array([-0.14974911,  0.68354692,  0.71426424, -0.01286089])
image_original = cv2.imread(datapath)
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
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
axis_length = 200
utils.visualize_axes(plt.gca(), q_gt, loc_gt, K, axis_length, 'g')
utils.visualize_axes(plt.gca(), q_gt1, loc_gt, K, axis_length, 'r')
# visualize_axes is left hand, visualize_axes1 is right hand
plt.savefig(datapath.replace('images', 'vis').replace('.png', 'pose.png'))
plt.show()