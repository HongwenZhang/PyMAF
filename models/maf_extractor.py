# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py

import os
from pickle import NONE
import re
from numpy.lib.twodim_base import triu_indices_from
from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.cfgs import cfg
from utils.geometry import projection
from core import constants, path_config
import json

import logging
logger = logging.getLogger(__name__)

from utils.iuvmap import iuv_img2map, iuv_map2img, seg_img2map
from .smpl import get_smpl_tpose
from utils.imutils import j2d_processing


class Mesh_Sampler(nn.Module):
    ''' Mesh Up/Down-sampling
    '''

    def __init__(self, type='smpl', level=2, device=torch.device('cuda'), option=None):
        super().__init__()

        self.model_type = type
        if type == 'flame':
            sampling_idx = torch.LongTensor(list(np.load('data/flame_downsampling.npy')))
            self.register_buffer('sampling_idx', sampling_idx)
        else:
            # downsample SMPL mesh and assign part labels
            if type == 'smpl':
                # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
                smpl_mesh_graph = np.load('data/smpl_downsampling.npz', allow_pickle=True, encoding='latin1')

                A = smpl_mesh_graph['A']
                U = smpl_mesh_graph['U']
                D = smpl_mesh_graph['D'] # shape: (2,)
            elif type == 'mano':
                mano_mesh_graph = np.load('data/mano_downsampling.npz', allow_pickle=True, encoding='latin1')

                A = mano_mesh_graph['A']
                U = mano_mesh_graph['U']
                D = mano_mesh_graph['D'] # shape: (2,)

            # downsampling
            ptD = []
            for lv in range(len(D)):
                d = scipy.sparse.coo_matrix(D[lv])
                i = torch.LongTensor(np.array([d.row, d.col]))
                v = torch.FloatTensor(d.data)
                ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
            
            # downsampling mapping from 6890 points to 431 points
            # ptD[0].to_dense() - Size: [1723, 6890] , [195, 778]
            # ptD[1].to_dense() - Size: [431, 1723] , [49, 195]
            if level == 2:
                Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431
            elif level == 1:
                Dmap = ptD[0].to_dense() # 
            self.register_buffer('Dmap', Dmap)

            # upsampling
            ptU = []
            for lv in range(len(U)):
                d = scipy.sparse.coo_matrix(U[lv])
                i = torch.LongTensor(np.array([d.row, d.col]))
                v = torch.FloatTensor(d.data)
                ptU.append(torch.sparse.FloatTensor(i, v, d.shape))
            
            # upsampling mapping from 431 points to 6890 points
            # ptU[0].to_dense() - Size: [6890, 1723]
            # ptU[1].to_dense() - Size: [1723, 431]
            if level == 2:
                Umap = torch.matmul(ptU[0].to_dense(), ptU[1].to_dense()) # 431 -> 6890
            elif level == 1:
                Umap = ptU[0].to_dense() # 
            self.register_buffer('Umap', Umap)

    def downsample(self, x):
        if self.model_type == 'flame':
            return x[:, self.sampling_idx].contiguous()
        else:
            return torch.matmul(self.Dmap.unsqueeze(0), x) # [B, 431, 3]
    
    def upsample(self, x):
        return torch.matmul(self.Umap.unsqueeze(0), x) # [B, 6890, 3]

    def forward(self, x, mode='downsample'):
        if mode == 'downsample':
            return self.downsample(x)
        elif mode == 'upsample':
            return self.upsample(x)


class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator
    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, filter_channels, device=torch.device('cuda'), iwp_cam_mode=True, option=None):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        self.last_op = nn.ReLU(True)

        self.iwp_cam_mode = iwp_cam_mode

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load('data/smpl_downsampling.npz', allow_pickle=True, encoding='latin1')

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D'] # shape: (2,)

        # downsampling
        ptD = []
        for level in range(len(D)):
            d = scipy.sparse.coo_matrix(D[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        
        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431
        self.register_buffer('Dmap', Dmap)

        # upsampling
        ptU = []
        for level in range(len(U)):
            d = scipy.sparse.coo_matrix(U[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptU.append(torch.sparse.FloatTensor(i, v, d.shape))
        
        # upsampling mapping from 431 points to 6890 points
        # ptU[0].to_dense() - Size: [6890, 1723]
        # ptU[1].to_dense() - Size: [1723, 431]
        Umap = torch.matmul(ptU[0].to_dense(), ptU[1].to_dense()) # 431 -> 6890
        self.register_buffer('Umap', Umap)


    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)

        return y

    def sampling(self, points, im_feat=None, z_feat=None, add_att=False, reduce_dim=True):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        # if im_feat is None:
        #     im_feat = self.im_feat

        batch_size = im_feat.shape[0]
        point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=False)[..., 0]

        if reduce_dim:
            mesh_align_feat = self.reduce_dim(point_feat)
            return mesh_align_feat
        else:
            return point_feat

    def forward(self, p, im_feat, cam=None, add_att=False, reduce_dim=True, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.
        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            im_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        # if cam is None:
        #     cam = self.cam
        p_proj_2d = projection(p, cam, retain_z=False, iwp_mode=self.iwp_cam_mode)
        if self.iwp_cam_mode:
            # Normalize keypoints to [-1,1]
            p_proj_2d = p_proj_2d / (224. / 2.)
        else:
            p_proj_2d = j2d_processing(p_proj_2d, cam['kps_transf'])            
        mesh_align_feat = self.sampling(p_proj_2d, im_feat, add_att=add_att, reduce_dim=reduce_dim)
        return mesh_align_feat
