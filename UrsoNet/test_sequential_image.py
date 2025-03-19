import os
import numpy as np
import os.path
import skimage
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import se3lib
import utils
from config import Config
import net
import urso
import speed
# Create model
OrientationParamOptions = ['quaternion', 'euler_angles', 'angle_axis']
config = Config()
config.ORIENTATION_PARAM= OrientationParamOptions
config.ORI_BINS_PER_DIM = 24 # only used in classifcation mode
config.EPOCHS = 100
config.NR_DENSE_LAYERS = 1 # Number of fully connected layers used on top of the feature network
config.LEARNING_RATE = 0.001
config.BOTTLENECK_WIDTH = 32
config.BRANCH_SIZE = 1024
config.BACKBONE = 'resnet50'
config.OPTIMIZER = "SGD"
config.REGRESS_ORI = False
config.REGRESS_LOC = True
config.REGRESS_KEYPOINTS = False
logs = './models/logs'
model = net.UrsoNet(mode="inference", config=config,
                        model_dir=logs)
_, weights_path = model.get_last_checkpoint(args.weights)
model.load_weights(weights_path, weights_path, by_name=True)
delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
var = delta ** 2 / 12
image_folder = './deimos2withtrajectory'
gt_file = './deimos2withtrajectory/gt.csv'

# 读取gt.csv文件
gt_df = pd.read_csv(gt_file)

# 循环读取每一张图片及其对应的标签
for index, row in gt_df.iterrows():
    # 根据图片的命名规则生成图片文件名
    image_filename = f'{index:04d}_rgb.png'
    image_path = os.path.join(image_folder, image_filename)
    
    if os.path.exists(image_path):
        # 读取图片
        image_ori = cv2.imread(image_path)
        
        if image is not None:
            # 获取位姿标签
            x, y, z, q1, q2, q3, q4 = row[['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']]

# Run detection
results = model.detect([image_ori], verbose=1)

# Retrieve location
loc_est = results[0]['loc']
else:
    loc_pmf = utils.stable_softmax(results[0]['loc'])

    # Compute location mean according to first moment
    loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

    # Compute loc encoding error
    loc_encoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
    loc_encoded_err = np.linalg.norm(loc_encoded_gt - loc_gt)

# Retrieve orientation
if model.config.REGRESS_ORI:

    if model.config.ORIENTATION_PARAM == 'quaternion':
            q_est = results[0]['ori']
    elif model.config.ORIENTATION_PARAM == 'euler_angles':
            q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
    elif model.config.ORIENTATION_PARAM == 'angle_axis':
        theta = np.linalg.norm(results[0]['ori'])
        if theta < 1e-6:
            v = [0,0,0]
        else:
            v = results[0]['ori']/theta
        q_est = se3lib.angleaxis2quat(v,theta)
else:
    ori_pmf = utils.stable_softmax(results[0]['ori'])

    # Compute mean quaternion
    q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

    # Multimodal estimation
    # Uncomment this block to try the EM framework
    # nr_EM_iterations = 5
    # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
    # print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
    #
    # q_est_1 = Q_mean[0, :]
    # q_est_2 = Q_mean[1, :]
    # utils.polar_plot(q_est_1, q_est_2)
print('q_est', q_est)
print('Qm error', q_est-q_gt)
print('postim error',loc_est-loc_gt)
# Compute Errors
angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
loc_err = np.linalg.norm(loc_est - loc_gt)

print('GT location: ', loc_gt)
print('Est location: ', loc_est)
print('Processed Image:', info['path'])
print('Est orientation: ', q_est)
print('GT_orientation: ', q_gt)

print('Location error: ', loc_err)
print('Angular error: ', angular_err)
