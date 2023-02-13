# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/base_dataset.py

from __future__ import division
from genericpath import exists
import imp

import os
import cv2
from numpy.testing._private.utils import print_assert_equal
import torch
import random
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from utils.cam_params import read_cam_params, homo_vector
import joblib

from core import path_config, constants
from core.cfgs import cfg
from utils.imutils import crop, flip_img, flip_pose, flip_aa, flip_kp, transform, get_transform, get_rot_transf, rot_aa
from models.smpl import SMPL, get_part_joints
from utils.geometry import projection, perspective_projection, estimate_translation
from utils.cam_params import f_pix2vfov, vfov2f_pix
import torch.nn.functional as F
import scipy.misc
from kornia.filters import motion_blur, gaussian_blur2d
from PIL import Image


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/path_config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super().__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = path_config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self.smpl_mode = (cfg.MODEL.MESH_MODEL == 'smpl')
        self.smplx_mode = (cfg.MODEL.MESH_MODEL == 'smplx')

        assert cfg.TRAIN.BHF_MODE in ['body_only', 'hand_only', 'face_only', 'body_hand', 'full_body']
        self.body_only_mode = (cfg.TRAIN.BHF_MODE == 'body_only')
        self.hand_only_mode = (cfg.TRAIN.BHF_MODE == 'hand_only')
        self.face_only_mode = (cfg.TRAIN.BHF_MODE == 'face_only')
        self.body_hand_mode = (cfg.TRAIN.BHF_MODE == 'body_hand')
        self.full_body_mode = (cfg.TRAIN.BHF_MODE == 'full_body')

        bhf_names = []
        if cfg.TRAIN.BHF_MODE in ['body_only', 'body_hand', 'full_body']:
            bhf_names.append('body')
        if cfg.TRAIN.BHF_MODE in ['hand_only', 'body_hand', 'full_body']:
            bhf_names.append('hand')
        if cfg.TRAIN.BHF_MODE in ['face_only', 'full_body']:
            bhf_names.append('face')
        self.bhf_names = bhf_names

        # bhf_names = cfg.TRAIN.BHF_MODE.split(',')
        # self.bhf_names = bhf_names
        # self.body_only_mode = len(bhf_names) == 1 and 'body' in bhf_names
        # self.hand_only_mode = len(bhf_names) == 1 and 'hand' in bhf_names
        # self.face_only_mode = len(bhf_names) == 1 and 'face' in bhf_names
        # self.body_hand_mode = len(bhf_names) == 2 and 'body' in bhf_names and 'hand' in bhf_names
        # self.full_body_mode = len(bhf_names) == 3 and 'body' in bhf_names and 'hand' in bhf_names and 'face' in bhf_names

        # if self.body_only_mode:
        #     self.bhf_mode = 'body_only'
        # elif self.hand_only_mode:
        #     self.bhf_mode = 'hand_only'
        # elif self.face_only_mode:
        #     self.bhf_mode = 'face_only'
        # elif self.body_hand_mode:
        #     self.bhf_mode = 'body_hand'
        # elif self.full_body_mode:
        #     self.bhf_mode = 'full_body'

        if cfg.TRAIN.USE_EFT:
            if self.full_body_mode or self.hand_only_mode or self.face_only_mode or self.body_hand_mode:
                dataset_label_paths = path_config.SMPLX_DATASET_FILES
            else:
                dataset_label_paths = path_config.EFT_DATASET_FILES
        else:
            dataset_label_paths = path_config.DATASET_FILES
        if not is_train and dataset == 'h36m-p2' and options.eval_pve:
            self.data = np.load(dataset_label_paths[is_train]['h36m-p2-mosh'], allow_pickle=True)
        else:
            self.data = np.load(dataset_label_paths[is_train][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']
        self.dataset_dict = {dataset: 0}

        logger.info('len of {}: {}'.format(self.dataset, len(self.imgname)))

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        try:
            self.orig_shape = np.concatenate([self.data['img_height'][:, None], self.data['img_width'][:, None]], axis=1)
        except:
            self.orig_shape = np.zeros((len(self.imgname), 2), dtype=np.float32) # (N, 2)

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            if cfg.MODEL.MESH_MODEL == 'smplx' and 'xpose' in self.data:
                self.pose = self.data['xpose'].astype(np.float) # (N, 72)
                self.betas = self.data['xshape'].astype(np.float) # (N, 10)
            else:
                self.pose = self.data['pose'].astype(np.float) # (N, 72)
                self.betas = self.data['shape'].astype(np.float) # (N, 10)

            if 'valid_fit' in self.data:
                self.has_smpl = self.data['valid_fit'].astype(np.float)
            elif 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.pose = np.zeros((len(self.imgname), 72), dtype=np.float32) # (N, 72)
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32) # (N, 10)
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        try:
            self.global_orient = self.data['global_orient'].astype(np.float)
            self.pose[:, :3] = self.data['global_orient'].astype(np.float)
        except KeyError:
            self.global_orient = self.pose[:, :3]

        # if self.dataset in path_config.HAND_DATASET_NAMES:
        if cfg.MODEL.MESH_MODEL == 'mano':
            try:
                self.lh_global_orient = self.data['lh_global_orient'].astype(np.float)
            except KeyError:
                self.lh_global_orient = np.zeros((len(self.imgname), 3), dtype=np.float32)
            try:
                self.rh_global_orient = self.data['rh_global_orient'].astype(np.float)
            except KeyError:
                self.rh_global_orient = np.zeros((len(self.imgname), 3), dtype=np.float32)
            
            try:
                self.lh_shape = self.data['lh_shape']
            except KeyError:
                self.lh_shape = np.zeros((len(self.imgname), 10), dtype=np.float32)
            try:
                self.rh_shape = self.data['rh_shape']
            except KeyError:
                self.rh_shape = np.zeros((len(self.imgname), 10), dtype=np.float32)
        
        # if self.dataset in path_config.FACE_DATASET_NAMES:
        if cfg.MODEL.MESH_MODEL == 'flame':
            try:
                self.fa_global_orient = self.data['fa_global_orient'].astype(np.float)
            except KeyError:
                self.fa_global_orient = np.zeros((len(self.imgname), 3), dtype=np.float32)
            
            try:
                self.fa_shape = self.data['fa_shape']
            except KeyError:
                self.fa_shape = np.zeros((len(self.imgname), 10), dtype=np.float32)

        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        # Get SMPL 2D keypoints
        try:
            self.smpl_2dkps = self.data['smpl_2dkps']
            self.has_smpl_2dkps = 1
        except KeyError:
            self.has_smpl_2dkps = 0

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            if cfg.MODEL.ALL_GENDER:
                gender = self.data['gender']
                gender_dict = {'male': 0, 'female': 1, 'neutral': 2, 'm': 0, 'f': 1, 'n': 2}
                self.gender = np.array([gender_dict[str(g)] for g in gender]).astype(np.int32)
            else:
                # set all to neutral
                self.gender = 2 * np.ones(len(self.imgname)).astype(np.int32)
        except KeyError:
            # self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
            self.gender = 2 * np.ones(len(self.imgname)).astype(np.int32)

        # camera info
        try:
            if cfg.MODEL.USE_GT_FL:
                self.f_pix = self.data['f_pix']
            else:
                self.f_pix = self.data['f_pix_pred']
                print('fpix pred', self.f_pix.shape)
        except KeyError:
            self.f_pix = np.ones(len(self.imgname)) * 5000.

        try:
            self.pitch = self.data['pitch']
        except KeyError:
            self.pitch = np.zeros(len(self.imgname))

        try:
            self.left_hand_pose = self.data['left_hand_pose']
            if 'valid_fit_lh' in self.data:
                self.has_lh = self.data['valid_fit_lh'].astype(np.float)
            else:
                self.has_lh = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.left_hand_pose = np.zeros((len(self.imgname), 15 * 3), dtype=np.float32)
            self.has_lh = np.zeros(len(self.imgname), dtype=np.float32)

        try:
            self.right_hand_pose = self.data['right_hand_pose']
            if 'valid_fit_rh' in self.data:
                self.has_rh = self.data['valid_fit_rh'].astype(np.float)
            else:
                self.has_rh = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.right_hand_pose = np.zeros((len(self.imgname), 15 * 3), dtype=np.float32)
            self.has_rh = np.zeros(len(self.imgname), dtype=np.float32)

        try:
            self.jaw_pose = self.data['jaw_pose']
            self.leye_pose = self.data['leye_pose']
            self.reye_pose = self.data['reye_pose']
            self.expression = self.data['expression']

            if 'valid_fit_fa' in self.data:
                self.has_fa = self.data['valid_fit_fa'].astype(np.float)
            else:
                self.has_fa = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.jaw_pose = np.zeros((len(self.imgname), 3), dtype=np.float32)
            self.leye_pose = np.zeros((len(self.imgname), 3), dtype=np.float32)
            self.reye_pose = np.zeros((len(self.imgname), 3), dtype=np.float32)
            self.expression = np.zeros((len(self.imgname), 10), dtype=np.float32)

            self.has_fa = np.zeros(len(self.imgname), dtype=np.float32)

        self.has_hf = np.stack([self.has_lh, self.has_rh, self.has_fa], axis=1)

        try:
            self.rhand_kp2d = self.data['rhand_kp2d']
        except KeyError:
            self.rhand_kp2d = np.zeros((len(self.imgname), 21, 3), dtype=np.float32)

        try:
            self.rhand_kp3d = self.data['rhand_kp3d']
        except KeyError:
            self.rhand_kp3d = np.zeros((len(self.imgname), 21, 4), dtype=np.float32)

        try:
            self.lhand_kp3d = self.data['lhand_kp3d']
        except KeyError:
            self.lhand_kp3d = np.zeros((len(self.imgname), 21, 4), dtype=np.float32)

        try:
            self.face_kp3d = self.data['face_kp3d']
        except KeyError:
            self.face_kp3d = np.zeros((len(self.imgname), 68, 4), dtype=np.float32)

        try:
            self.lhand_kp2d = self.data['lhand_kp2d']
        except KeyError:
            self.lhand_kp2d = np.zeros((len(self.imgname), 21, 3), dtype=np.float32)

        try:
            self.face_kp2d = self.data['face_kp2d']
        except KeyError:
            self.face_kp2d = np.zeros((len(self.imgname), 68, 3), dtype=np.float32)

        try:
            self.feet_kp2d = self.data['feet_kp2d']
        except KeyError:
            self.feet_kp2d = np.zeros((len(self.imgname), 6, 3), dtype=np.float32)

        if self.dataset in ['ehf']:
            self.smplx_verts = self.data['smplx_verts']
            self.smpl_verts = self.data['smpl_verts']
            self.lhand_verts = self.data['lhand_verts']
            self.rhand_verts = self.data['rhand_verts']
            self.face_verts = self.data['face_verts']

        try:
            self.det_lhand_kp2d = self.data['det_lhand_kp2d']
            self.det_rhand_kp2d = self.data['det_rhand_kp2d']
            self.det_face_kp2d = self.data['det_face_kp2d']
            # self.det_feet_kp2d = self.data['det_feet_kp2d']
        except KeyError:
            self.det_lhand_kp2d = np.zeros((len(self.imgname), 21, 3), dtype=np.float32)
            self.det_rhand_kp2d = np.zeros((len(self.imgname), 21, 3), dtype=np.float32)
            self.det_face_kp2d = np.zeros((len(self.imgname), 68, 3), dtype=np.float32)
            # raise

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        blur = 0            # blur

        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= self.options.flip_factor:
                flip = 1

            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
            
            # blur with probability 
            if np.random.uniform() <= self.options.blur_factor:
                blur = 1

        return flip, pn, rot, sc, blur

    def rgb_add_noise(self, rgb_img, pn):
        """add pixel noise rgb image and do augmentation."""
        # in the rgb image we add pixel noise in a channel-wise manner
        if self.options.noise_factor > 0:
            rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
            rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
            rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        return rgb_img

    def rgb_crop(self, rgb_img, center, scale, res, rot):
        crop_img_resized, crop_img, crop_shape = crop(rgb_img, center, scale, res, rot=rot)
        return crop_img_resized, crop_img, crop_shape

    def rgb_transpose(self, crop_img, flip=0):
        """Process rgb image and do augmentation."""
        # in the rgb image we add pixel noise in a channel-wise manner
        # rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        # rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        # rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # crop
        # crop_img_resized, crop_img, crop_shape = crop(rgb_img, center, scale, res, rot=rot)
        # flip the image
        if flip:
            # crop_img_resized = flip_img(crop_img_resized)
            crop_img = flip_img(crop_img)
            # rgb_img = flip_img(rgb_img)
        # (3,224,224),float,[0,1]
        # crop_img_resized = np.transpose(crop_img_resized.astype('float32'), (2,0,1)) / 255.0
        crop_img = np.transpose(crop_img.astype('float32'), (2,0,1)) / 255.0
        # rgb_img = np.transpose(rgb_img.astype('float32'), (2,0,1)) / 255.0
        return crop_img

    def rgb_processing(self, rgb_img, center, scale, res, rot, flip):
        """Process rgb image and do augmentation."""
        # in the rgb image we add pixel noise in a channel-wise manner
        # rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        # rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        # rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # crop
        crop_img_resized, crop_img, crop_shape = crop(rgb_img, center, scale, res, rot=rot)
        # flip the image
        if flip:
            crop_img_resized = flip_img(crop_img_resized)
            crop_img = flip_img(crop_img)
            # rgb_img = flip_img(rgb_img)
        # (3,224,224),float,[0,1]
        crop_img_resized = np.transpose(crop_img_resized.astype('float32'), (2,0,1)) / 255.0
        crop_img = np.transpose(crop_img.astype('float32'), (2,0,1)) / 255.0
        # rgb_img = np.transpose(rgb_img.astype('float32'), (2,0,1)) / 255.0
        return crop_img_resized, crop_img, crop_shape

    def j2d_processing(self, kp, t, f, is_smpl=False, is_hand=False, is_face=False, is_feet=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        kp = kp.copy()
        nparts = kp.shape[0]
        # res = [constants.IMG_RES, constants.IMG_RES]
        # t = get_transform(center, scale, res, rot=rot)
        for i in range(nparts):
            pt = kp[i,0:2]
            new_pt = np.array([pt[0], pt[1], 1.]).T
            new_pt = np.dot(t, new_pt)
            kp[i,0:2] = new_pt[:2]
            # kp[i,0:2] = new_pt[:2].astype(int) + 1
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
            if is_hand:
                kp = flip_kp(kp, type='hand')
            elif is_face:
                kp = flip_kp(kp, type='face')
            elif is_feet:
                kp = flip_kp(kp, type='feet')
            else:
                kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f, is_smpl=False, is_hand=False, is_face=False):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        S = S.copy()
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
            # S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
            S[:, :3] = np.einsum('ij,kj->ki', rot_mat, S[:, :3])
        # flip the x coordinates
        if f:
            if is_hand:
                S = flip_kp(S, type='hand')
            elif is_face:
                S = flip_kp(S, type='face')
            else:
                S = flip_kp(S, is_smpl)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose
    
    def global_rot_processing(self, pose, r, f):
        """Process global oritation and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_aa(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        if 'body' in self.bhf_names:
            if cfg.DATA.RESCALE_B != 1.:
                scale *= cfg.DATA.RESCALE_B

        # Get augmentation parameters
        flip, pn, rot, sc, blur = self.augm_params()

        if self.hand_only_mode:
            if not self.has_hf[index][1]:
                flip = 1
            elif not self.has_hf[index][0]:
                flip = 0

            if 'coco' in self.dataset:
                min_hand_keypoints = 10
                valid_lh = np.sum(self.lhand_kp2d[index][:, -1] > 0) > min_hand_keypoints
                valid_rh = np.sum(self.rhand_kp2d[index][:, -1]> 0) > min_hand_keypoints
                if not valid_rh:
                    flip = 1
                elif not valid_lh:
                    flip = 0
        
        if self.dataset in ['ehf']:
            flip = 0

        if self.hand_only_mode:
            if self.dataset not in ['freihand']:
                kp_2d = self.lhand_kp2d[index].copy() if flip else self.rhand_kp2d[index].copy()
                # scale and center
                bbox = [min(kp_2d[:, 0]), min(kp_2d[:, 1]),
                        max(kp_2d[:, 0]), max(kp_2d[:, 1])]

                center = np.array([(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.])
                scale = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

            if cfg.DATA.RESCALE_H != 1.:
                scale *= cfg.DATA.RESCALE_H

        if self.face_only_mode:
            if self.dataset not in path_config.FACE_DATASET_NAMES:
                kp_2d = self.face_kp2d[index].copy()
                kp_2d_valid = kp_2d[kp_2d[:, -1]>0.2, :]
                if len(kp_2d_valid) > 2:
                    kp_2d = kp_2d_valid
                # scale and center
                bbox = [min(kp_2d[:, 0]), min(kp_2d[:, 1]),
                        max(kp_2d[:, 0]), max(kp_2d[:, 1])]

                center = np.array([(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.])
                scale = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.
        
            if cfg.DATA.RESCALE_F != 1.:
                scale *= cfg.DATA.RESCALE_F

        if self.options.upper_crop_factor > 0:
            if np.random.uniform() <= self.options.upper_crop_factor:
                kp_2d = self.keypoints[index].copy()[-24:]
                kp_2d = np.concatenate([kp_2d[2:4], kp_2d[6:]])
                kp_2d = kp_2d[kp_2d[:, -1] > 0.1]
                if len(kp_2d) > 5:
                    # re-calculate scale and center
                    bbox = [min(kp_2d[:, 0]), min(kp_2d[:, 1]),
                            max(kp_2d[:, 0]), max(kp_2d[:, 1])]

                    center = np.array([(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.])
                    scale = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

        kp_is_smpl = True if self.dataset == 'surreal' else False

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            pose = self.pose_processing(pose, rot, flip)
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # if self.dataset in path_config.HAND_DATASET_NAMES:
        if cfg.MODEL.MESH_MODEL == 'mano':
            global_orient = self.lh_global_orient[index].copy() if flip else self.rh_global_orient[index].copy()
        elif self.dataset in path_config.FACE_DATASET_NAMES:
            global_orient = self.fa_global_orient[index].copy()
        else:
            global_orient = self.global_orient[index].copy()
        item['global_orient'] = self.global_rot_processing(global_orient, rot, flip)

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        if self.dataset in ['agora']:
            img_dir, img_base_name = os.path.split(self.imgname[index])
            if not self.is_train:
                img_dir += '/{:03d}'.format(index // 1000)
            img_crop_name = img_base_name.split('.')[0] + '_{:06d}.png'.format(index)

            imgpath_crop = join(self.img_dir, 'crop', img_dir, 'body', img_crop_name)
            if os.path.exists(imgpath_crop):
                orig_shape = self.orig_shape[index]
                try:
                    img_crop = cv2.imread(imgpath_crop)[:,:,::-1].copy().astype(np.float32)
                except:
                    logger.error('fail while loading {}'.format(imgpath_crop))
                crop_shape = np.array(img_crop.shape)[:2]
                if self.is_train:
                    img_crop = self.rgb_add_noise(img_crop, pn)
                img = self.rgb_transpose(img_crop, flip)
            else:
                try:
                    img_orig = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
                    orig_shape = np.array(img_orig.shape)[:2]
                except:
                    logger.error('fail while loading {}'.format(imgname))

                if self.is_train:
                    img_orig = self.rgb_add_noise(img_orig, pn)
                img, img_crop, crop_shape = self.rgb_crop(img_orig, center, sc*scale, [constants.IMG_RES, constants.IMG_RES], rot)

                os.makedirs(os.path.split(imgpath_crop)[0], exist_ok=True)
                cv2.imwrite(imgpath_crop, img[:,:,::-1])

                img = self.rgb_transpose(img, flip)
        else:
            try:
                img_orig = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
                orig_shape = np.array(img_orig.shape)[:2]
            except:
                logger.error('fail while loading {}'.format(imgname))
            if self.is_train:
                img_orig = self.rgb_add_noise(img_orig, pn)
            try:
                # img, img_crop1, crop_shape = self.rgb_processing(img_orig, center, sc*scale, [constants.IMG_RES, constants.IMG_RES], rot, flip)
                img, img_crop, crop_shape = self.rgb_crop(img_orig, center, sc*scale, [constants.IMG_RES, constants.IMG_RES], rot)
                img = self.rgb_transpose(img, flip)
            except:
                img = np.random.rand(3, constants.IMG_RES, constants.IMG_RES) * 0.1
                logger.warning('fail while cropping {}'.format(imgname))

        # Store image before normalization to use it in visualization
        img_main = self.normalize_img(torch.from_numpy(img).float())
        if blur:
            if np.random.uniform() <= 0.5:
                # add motion_blur
                ks = np.random.randint(6)*4 + 7
                angle = np.random.rand()*180 - 90
                direction = np.random.randint(21)*0.1 - 1
                img_main = motion_blur(img_main[None], kernel_size=ks, angle=angle, direction=direction).squeeze()
            else:
                # add gaussian_blur2d
                ks = np.random.randint(5)*2 + 9
                sgm = np.random.randint(5)*0.2 + 1.2
                img_main = gaussian_blur2d(img_main[None], kernel_size=(ks,ks), sigma=(sgm,sgm)).squeeze()
        item['img'] = img_main
        if self.hand_only_mode:
            item['img_rhand'] = img_main
        elif self.face_only_mode:
            item['img_face'] = img_main
        else:
            item['img_body'] = img_main
        if cfg.TRAIN.HR_IMG and self.full_body_mode:
            # img_hr, img_crop2, _ = self.rgb_processing(img_orig, center, sc*scale, [constants.IMG_RES * 8, constants.IMG_RES * 8], rot, flip)
            # img_hr = scipy.misc.imresize(img_crop, [constants.IMG_RES * 8, constants.IMG_RES * 8])
            img_hr = np.array(Image.fromarray(img_crop.astype(np.uint8)).resize([constants.IMG_RES * 8, constants.IMG_RES * 8]))
            img_hr = self.rgb_transpose(img_hr, flip)
            img_crop = self.rgb_transpose(img_crop, flip)
            # img_orig = flip_img(img_orig) if flip else img_orig
            # img_orig = np.transpose(img_orig.astype('float32'), (2,0,1)) / 255.0
            item['img_hr'] = self.normalize_img(torch.from_numpy(img_hr).float())
            # item['img_orig'] = self.normalize_img(torch.from_numpy(img_orig).float())
        elif self.full_body_mode:
            img_crop = self.rgb_transpose(img_crop, flip)
            item['img_hr'] = item['img']

        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        kps_transf = get_transform(center, sc * scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        rot_transf = get_rot_transf([constants.IMG_RES, constants.IMG_RES], rot)
        # Get 2D SMPL joints
        if self.has_smpl_2dkps:
            smpl_2dkps = self.smpl_2dkps[index].copy()
            smpl_2dkps = self.j2d_processing(smpl_2dkps, kps_transf, f=0)
            smpl_2dkps[smpl_2dkps[:, 2] == 0] = 0
            if flip:
                smpl_2dkps = smpl_2dkps[constants.SMPL_JOINTS_FLIP_PERM]
                smpl_2dkps[:, 0] = - smpl_2dkps[:, 0]
            item['smpl_2dkps'] = torch.from_numpy(smpl_2dkps).float()
        else:
            item['smpl_2dkps'] = torch.zeros(24, 3, dtype=torch.float32)

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            S_root = np.mean(S[2:4, :3], axis=0)[None]
            S[:, :3] -= S_root
            S[S[:,3]>0, 3] = 1
            pose_3d = torch.from_numpy(self.j3d_processing(S, rot, flip, kp_is_smpl)).float()
            item['pose_3d'] = pose_3d
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints_orig = self.keypoints[index].copy()
        # keypoints_bk = self.j2d_processing_bk(keypoints.copy(), center, sc*scale, rot, flip, kp_is_smpl)
        keypoints = self.j2d_processing(keypoints_orig.copy(), kps_transf, flip, kp_is_smpl)
        # print('keypoints_bk', np.sum(keypoints_bk - keypoints))
        item['keypoints'] = torch.from_numpy(keypoints).float()

        keypoints_full_img = torch.from_numpy(keypoints_orig.copy()).float()

        n_hand_kp = len(constants.HAND_NAMES)
        hand_keypoints_orig = np.concatenate([self.lhand_kp2d[index], self.rhand_kp2d[index]]).copy()
        hand_keypoints_full_img = torch.from_numpy(hand_keypoints_orig.copy()).float()

        if flip:
            center[0] = orig_shape[1] - center[0] - 1
            keypoints_full_img[:, 0] = keypoints_full_img[:, 0] / orig_shape[1] - 0.5
            keypoints_full_img = flip_kp(keypoints_full_img)
            keypoints_full_img[:, 0] = (keypoints_full_img[:, 0] + 0.5) * orig_shape[1]

            hand_keypoints_full_img[:, 0] = hand_keypoints_full_img[:, 0] / orig_shape[1] - 0.5
            hand_keypoints_full_img = flip_kp(hand_keypoints_full_img, type='hand')
            hand_keypoints_full_img[:, 0] = (hand_keypoints_full_img[:, 0] + 0.5) * orig_shape[1]

        item['keypoints_full_img'] = keypoints_full_img
        item['lhand_keypoints_full_img'] = hand_keypoints_full_img[:n_hand_kp]
        item['rhand_keypoints_full_img'] = hand_keypoints_full_img[n_hand_kp:]

        item['has_smpl'] = self.has_smpl[index].copy()
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32) 
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        item['kps_transf'] = get_transform(center, sc * scale, [constants.IMG_RES, constants.IMG_RES], rot=rot).astype(np.float32)
        item['rot_transf'] = rot_transf.astype(np.float32)

        if self.smplx_mode or 'hand' in self.bhf_names or 'face' in self.bhf_names:
            item['left_hand_pose'] = flip_aa(self.right_hand_pose[index].copy()) if flip else self.left_hand_pose[index].copy()
            item['right_hand_pose'] = flip_aa(self.left_hand_pose[index].copy()) if flip else self.right_hand_pose[index].copy()
            item['jaw_pose'] = flip_aa(self.jaw_pose[index].copy()) if flip else self.jaw_pose[index].copy()
            item['leye_pose'] = flip_aa(self.reye_pose[index].copy()) if flip else self.leye_pose[index].copy()
            item['reye_pose'] = flip_aa(self.leye_pose[index].copy()) if flip else self.reye_pose[index].copy()
            item['expression'] = self.expression[index].copy()
            has_hf = self.has_hf[index].copy()
            if flip:
                has_hf[:2] = has_hf[:2][::-1]
            item['has_hf'] = has_hf

            if 'hand' in self.bhf_names:
                hand_kp3d = self.j3d_processing(np.concatenate([self.lhand_kp3d[index], self.rhand_kp3d[index]]).copy(), rot, flip, is_hand=True)
                item['lhand_kp3d'] = hand_kp3d[:n_hand_kp]
                item['rhand_kp3d'] = hand_kp3d[n_hand_kp:]

                if self.hand_only_mode:
                    item['lhand_shape'] = self.rh_shape[index].copy() if flip else self.lh_shape[index].copy()
                    item['rhand_shape'] = self.lh_shape[index].copy() if flip else self.rh_shape[index].copy()

            if 'face' in self.bhf_names:
                face_kp3d = self.j3d_processing(self.face_kp3d[index].copy(), rot, flip, is_face=True)
                item['face_kp3d'] = face_kp3d

                if self.face_only_mode:
                    item['face_shape'] = self.fa_shape[index].copy()

            if  'hand' in self.bhf_names or 'face' in self.bhf_names:
                if self.dataset in ['h36m']:
                    det_lhand_kp2d = self.det_lhand_kp2d[index].copy()
                    det_rhand_kp2d = self.det_rhand_kp2d[index].copy()
                    det_face_kp2d = self.det_face_kp2d[index].copy()

                    det_hand_kp2d = self.j2d_processing(np.concatenate([det_lhand_kp2d, det_rhand_kp2d]), kps_transf, flip, is_hand=True)
                    det_face_kp2d = self.j2d_processing(det_face_kp2d, kps_transf, flip, is_face=True)

                    det_hand_kp2d[det_hand_kp2d[:, -1] == 0] = 0.
                    det_face_kp2d[det_face_kp2d[:, -1] == 0] = 0.

                    item['lhand_kp2d'] = det_hand_kp2d[:n_hand_kp]
                    item['rhand_kp2d'] = det_hand_kp2d[n_hand_kp:]
                    item['face_kp2d'] = det_face_kp2d
                    item['feet_kp2d'] = np.zeros((6, 3), dtype=np.float32)
                else:
                    # part_kp2d_dict = {'lhand': self.lhand_kp2d[index], 'rhand': self.rhand_kp2d[index], 'face': self.face_kp2d[index]}
                    if 'hand' in self.bhf_names:
                        hand_kp2d = self.j2d_processing(np.concatenate([self.lhand_kp2d[index], self.rhand_kp2d[index]]).copy(), kps_transf, flip, is_hand=True)
                        item['lhand_kp2d'] = hand_kp2d[:n_hand_kp]
                        item['rhand_kp2d'] = hand_kp2d[n_hand_kp:]

                    if 'face' in self.bhf_names:
                        face_kp2d = self.j2d_processing(self.face_kp2d[index].copy(), kps_transf, flip, is_face=True)
                        item['face_kp2d'] = face_kp2d

            if self.smplx_mode:
                feet_kp2d = self.j2d_processing(self.feet_kp2d[index].copy(), kps_transf, flip, is_feet=True)
                item['feet_kp2d'] = feet_kp2d

        if self.body_hand_mode or self.full_body_mode:
            if self.body_hand_mode:
                part_kp2d_dict = {'lhand': item['lhand_kp2d'], 'rhand': item['rhand_kp2d']}
            elif self.full_body_mode:
                part_kp2d_dict = {'lhand': item['lhand_kp2d'], 'rhand': item['rhand_kp2d'], 'face': item['face_kp2d']}

            if self.body_hand_mode:
                part_names = ['lhand', 'rhand']
            elif self.full_body_mode:
                part_names = ['lhand', 'rhand', 'face']

            for part in part_names:
                kp2d = part_kp2d_dict[part]
                kp2d_valid = kp2d[kp2d[:, 2]>0.]
                if len(kp2d_valid) > 0:
                    bbox = [min(kp2d_valid[:, 0]), min(kp2d_valid[:, 1]),
                            max(kp2d_valid[:, 0]), max(kp2d_valid[:, 1])]
                    center_part = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
                    scale_part = 1.5 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

                if len(kp2d_valid) < 1 or scale_part < 0.01:
                    center_part = [0, 0]
                    scale_part = 0.5
                    kp2d[:, 2] = 0

                    if 'coco' in self.dataset:
                        part_idx = {'lhand': 11, 'rhand': 6, 'face': 19}
                        center_part = item['keypoints'][-24:, :2][part_idx[part]]
                        scale_part = 0.1

                if not torch.is_tensor(center_part):
                    center_part = torch.tensor(center_part).float()

                theta_part = torch.zeros(1, 2, 3)
                theta_part[:, 0, 0] = scale_part
                theta_part[:, 1, 1] = scale_part
                theta_part[:, :, -1] = center_part

                if self.dataset in ['agora']:

                    imgpath_crop = join(self.img_dir, 'crop', img_dir, part, img_crop_name)
                    if os.path.exists(imgpath_crop):
                        if flip:
                            if part == 'lhand':
                                imgpath_crop = join(self.img_dir, 'crop', img_dir, 'rhand', img_crop_name)
                            elif part == 'rhand':
                                imgpath_crop = join(self.img_dir, 'crop', img_dir, 'lhand', img_crop_name)

                        try:
                            img_part = cv2.imread(imgpath_crop)[:,:,::-1].copy().astype(np.float32)
                        except:
                            logger.error('fail while loading {}'.format(imgpath_crop))
                        if self.is_train:
                            img_part = self.rgb_add_noise(img_part, pn)
                        img_part = self.rgb_transpose(img_part, flip)
                        img_part = torch.from_numpy(img_part)
                    else:
                        crop_hf_img_size = torch.Size([1, 3, cfg.MODEL.PyMAF.HF_IMG_SIZE, cfg.MODEL.PyMAF.HF_IMG_SIZE])
                        grid = F.affine_grid(theta_part.detach(), crop_hf_img_size, align_corners=False)
                        img_part = F.grid_sample(torch.from_numpy(img_crop[None]), grid.cpu(), align_corners=False).squeeze(0)

                        img_part_np = np.transpose(img_part.numpy(), (1,2,0)) * 255.

                        assert not flip

                        os.makedirs(os.path.split(imgpath_crop)[0], exist_ok=True)
                        cv2.imwrite(imgpath_crop, img_part_np[:,:,::-1])

                else:
                    crop_hf_img_size = torch.Size([1, 3, cfg.MODEL.PyMAF.HF_IMG_SIZE, cfg.MODEL.PyMAF.HF_IMG_SIZE])
                    grid = F.affine_grid(theta_part.detach(), crop_hf_img_size, align_corners=False)
                    img_part = F.grid_sample(torch.from_numpy(img_crop[None]), grid.cpu(), align_corners=False).squeeze(0)

                # Store image before normalization to use it in visualization
                item[f'img_{part}'] = self.normalize_img(img_part.float())

                theta_i_inv = torch.zeros_like(theta_part)
                theta_i_inv[:, 0, 0] = 1. / theta_part[:, 0, 0]
                theta_i_inv[:, 1, 1] = 1. / theta_part[:, 1, 1]
                theta_i_inv[:, :, -1] = - theta_part[:, :, -1] / theta_part[:, 0, 0].unsqueeze(-1)

                # print(theta_i_inv.shape, stn_centers.shape)
                kp2d = torch.from_numpy(kp2d[None])
                part_kp2d = torch.bmm(theta_i_inv, homo_vector(kp2d[:, :, :2]).permute(0, 2, 1)).permute(0, 2, 1)
                part_kp2d = torch.cat([part_kp2d, kp2d[:, :, 2:3]], dim=-1).squeeze(0)

                item[f'{part}_kp2d_local'] = part_kp2d
                item[f'{part}_theta'] = theta_part[0]
                item[f'{part}_theta_inv'] = theta_i_inv[0]

        if self.dataset in ['ehf']:
            smplx_verts = self.j3d_processing(self.smplx_verts[index].copy(), rot, flip, is_face=True)
            smpl_verts = self.j3d_processing(self.smpl_verts[index].copy(), rot, flip, is_face=True)
            lhand_verts = self.j3d_processing(self.lhand_verts[index].copy(), rot, flip, is_face=True)
            rhand_verts = self.j3d_processing(self.rhand_verts[index].copy(), rot, flip, is_face=True)
            face_verts = self.j3d_processing(self.face_verts[index].copy(), rot, flip, is_face=True)
            item['smplx_verts'] = smplx_verts
            item['smpl_verts'] = smpl_verts
            item['lhand_verts'] = lhand_verts
            item['rhand_verts'] = rhand_verts
            item['face_verts'] = face_verts

        if not cfg.MODEL.USE_IWP_CAM or 'agora' in self.dataset:
            if 'agora' in self.dataset:
                # prediction from SPEC
                # cam_params = joblib.load(os.path.join(path_config.CAM_PARAM_FOLDERS[self.dataset], self.imgname[index] + '.pkl'))
                # GT cam
                cam_params = {'f_pix': self.f_pix[index], 'pitch': self.pitch[index], 'orig_resolution': orig_shape}
                if not cfg.MODEL.PRED_PITCH:
                    cam_params['pitch'] = np.zeros(1)
            elif 'h36m' in self.dataset:
                cam_params = {'f_pix': 1146.8, 'pitch': np.zeros(1), 'orig_resolution': orig_shape}
            elif 'coco' in self.dataset:
                cam_params = {'f_pix': 5000, 'pitch': np.zeros(1), 'orig_resolution': orig_shape}
            else:
                cam_params = {'f_pix': 5000, 'pitch': np.zeros(1), 'orig_resolution': orig_shape}

            cam_rotmat, cam_intrinsics, cam_vfov, cam_pitch, cam_roll, cam_focal_length = read_cam_params(cam_params)

            vfov = f_pix2vfov(cam_focal_length, orig_shape[0])

            item['cam_rotmat'] = cam_rotmat
            item['cam_intrinsics'] = cam_intrinsics
            item['cam_focal_length'] = cam_focal_length
            item['cam_vfov'] = np.float32(vfov)
            item['crop_ratio'] = np.float32(crop_shape[0] / orig_shape[0])

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        # return 1
        return len(self.imgname)
