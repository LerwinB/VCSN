
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from sklearn.metrics import pairwise_distances
from skimage import io
from tqdm import tqdm
import random
from cuml import KMeans,DBSCAN
# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

class VaeNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,256,kernel_size=3)

class Coreset:
    def _updated_distances(self,cluster_centers, features, min_distances):
        x = features[cluster_centers, :]
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def select_batch(self,features, N, selected_idx):
        new_batch = []
        if selected_idx:
            selected_indices=selected_idx
            new_batch = selected_idx
            selected_size= len(selected_idx)
        else:
            selected_indices=list(range(N//20))
            selected_size = 0
        min_distances = self._updated_distances(selected_indices, features, None)
        for _ in tqdm(range(N-selected_size)):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
        return new_batch

class KMeansSampling:

    def select_batch(self, features, N, selected_idx):

        cluster_learner = KMeans(n_clusters=N)
        cluster_idxs = cluster_learner.fit_predict(features)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (features - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(features.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(N)])

        return q_idxs

class DBSCANSampling:

    def select_batch(self, features, N, selected_idx):

        cluster_learner = DBSCAN(n_clusters=N)
        cluster_idxs = cluster_learner.fit_predict(features)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (features - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(features.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(N)])

        return q_idxs


class RandomSampling:

    def select_batch(self,features, N):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], N, replace=False)


