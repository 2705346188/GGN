import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.cluster import KMeans
import scipy
import pdb
import os
#from Extreme_Clustering import Extreme_Clustreing
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#from cupy.linalg import norm


def get_dc(data, m, size, dev):
    data = data.to(dev) 
    all_zeros = torch.all(data == 0, dim=1)  
    indices = torch.nonzero(all_zeros).squeeze()
    print(indices)
    has_nan = torch.isnan(data).any()
    y = pdist(data.detach().cpu().numpy(),'euclidean')
    has_nan = torch.isnan(torch.Tensor(y)).any()
    z = squareform(cp.asarray(y).get())
    z = torch.Tensor(z)
    has_nan = torch.isnan(z).any()
    t1, t2 = torch.topk(z, m + 1, largest=False) 
    t3 = t2
    t2 = data[t2]
    a = t1[:,1].mean()
    return t2, a, t1,t3 


def get_center_1(data, cluster_num,adj,k,m_node,dev):

  
    adj = torch.cat((adj,torch.zeros(cluster_num,adj.shape[1]).to(dev)))
    temp = torch.cat((torch.zeros(adj.shape[1],cluster_num),torch.zeros(cluster_num,cluster_num)))
    adj = torch.cat((adj, temp.to(dev)),dim=1)
    data1 = data.detach()
    ori_data = pdist(data1.cpu().numpy(),'euclidean')
    ori_data = squareform(ori_data)
    ori_data = torch.Tensor(ori_data)
    ori_data = ori_data.to(dev)
    m = m_node
    _, dc, _, loc = get_dc(data, m, 2, dev)
    ori_data_tmp = ori_data.clone()
    ori_data_tmp = ori_data_tmp.to(dev)
    dc = dc.to(dev)
    dc = dc.item()
    for j in range(0, loc.shape[0]):
        ori_data[j][loc[j][1:cluster_num+1]] = 0
    ori_data = ori_data_tmp - ori_data
    ori_data = ori_data.detach().cpu().numpy()
    ori_data = cp.array(ori_data)
    loc = cp.asarray(loc)
    index = cp.arange(ori_data.size).reshape(ori_data.shape)
    nearest_dis = cp.take_along_axis(ori_data, loc[:, 1:2], axis=1)
    ori_data = np.where(ori_data != 0  , nearest_dis / (ori_data * ori_data), ori_data)
    mask = index <= cp.argsort(ori_data, axis=1)[:, -m:].max(axis=1, keepdims=True)
    ori_data = cp.where(mask, 0, ori_data)
    ori_data =torch.Tensor(ori_data)
    norms = torch.norm(ori_data, p=2, dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    ori_data = ori_data / norms
    ori_data = ori_data.to(dev)

    kmeans = KMeans(
        n_clusters=cluster_num, init="k-means++"
    ).fit(data.detach().cpu().numpy())
    center = kmeans.cluster_centers_
    center1 = torch.tensor(center)
    data = data.to(dev)
    
    if adj.shape[0] != data.shape[0]:
        data1 = torch.cat((data.to(dev), center1.to(dev)), dim=0)
    else:
        data1 = data.to(dev)

    z_3 = torch.zeros(center1.shape[0], center1.shape[0])
    n = data.shape[0]
    m = center1.shape[0]
    data = data.unsqueeze(1).expand(n, m, -1)  
    center1 = center1.unsqueeze(0).expand(n, m, -1)  
    distances_1 = torch.sqrt(torch.sum((data.to(dev) - center1.to(dev)) ** 2, dim=2))
    max_indices = torch.argmax(distances_1, dim=1)
    
    distances_1 = distances_1.detach().cpu().numpy()
    index = cp.arange(distances_1.size).reshape(distances_1.shape)
    #max_indices = cp.arange(max_indices.detach().cpu().numpy())
    max_indices = (max_indices.detach().cpu().numpy()).reshape(-1, 1)
    index = cp.asnumpy(index)
    index[:, 0:1] = max_indices.astype(index.dtype)
   
    distances_1 = cp.asarray(distances_1)
    nearest_dis = cp.take_along_axis(distances_1, index[:, 0:1], axis=1)
 
    distances_1 = cp.where(distances_1 != 0,nearest_dis/ (distances_1 * distances_1), distances_1)
    distances_1 = torch.Tensor(distances_1)
    norms = torch.norm(distances_1, p=2, dim=1, keepdim=True)
    norms[norms == 0] = 1
    distances_1 = distances_1 / norms
    distances_2 = torch.t(distances_1) 
    if adj.shape[0] != ori_data.shape[0]:
        final_data = torch.cat((ori_data.to(dev),distances_2.to(dev)),dim=0)
        medium_data = torch.cat((distances_1.to(dev),z_3.to(dev)))
        final_data = torch.cat((final_data.to(dev),medium_data.to(dev)),dim=1)
    else:
        final_data = ori_data

    final_data = torch.add(adj,torch.mul(k, final_data))
    return data1,final_data