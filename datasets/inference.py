# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import numpy as np
import os.path as osp
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from torchvision.transforms import Normalize

from utils.smooth_bbox import get_all_bbox_params
from .data_utils.img_utils import get_single_image_crop_demo
from utils.cam_params import read_cam_params, homo_vector


from core import path_config, constants
from core.cfgs import cfg
from utils.imutils import crop, flip_img, flip_pose, flip_aa, flip_kp, transform, get_transform, get_rot_transf, rot_aa


class Inference(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224, pre_load_imgs=None, full_body=False, person_ids=[], wb_kps={}):
        self.pre_load_imgs = pre_load_imgs
        if pre_load_imgs is None:
            self.image_file_names = [
                osp.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg')
            ]
            self.image_file_names = sorted(self.image_file_names)
            self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale_factor = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False
        self.full_body = full_body
        self.person_ids = person_ids

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            if not self.full_body:
                bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
                bboxes[:, 2:] = 150. / bboxes[:, 2:]
                self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

                self.image_file_names = self.image_file_names[time_pt1:time_pt2]
                self.joints2d = joints2d[time_pt1:time_pt2]
                self.frames = frames[time_pt1:time_pt2]
            else:
                bboxes = []
                scales = []
                for j2d in joints2d:
                    kp2d_valid = j2d[j2d[:, 2]>0.]
                    bbox = [min(kp2d_valid[:, 0]), min(kp2d_valid[:, 1]),
                            max(kp2d_valid[:, 0]), max(kp2d_valid[:, 1])]
                    center = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
                    scale = self.scale_factor * 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

                    res = [constants.IMG_RES, constants.IMG_RES]
                    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
                    # Bottom right point
                    br = np.array(transform([res[0]+1, 
                                            res[1]+1], center, scale, res, invert=1))-1

                    center = [(ul[0] + br[0]) / 2., (ul[1] + br[1]) / 2.]
                    width_height = [br[0] - ul[0], br[1] - ul[1]]

                    bbox = np.array(center + width_height)
                    bboxes.append(bbox)
                    scales.append(scale)
                
                self.bboxes = np.stack(bboxes)
                self.scales = np.array(scales)

                self.image_file_names = self.image_file_names
                self.joints2d = joints2d
                self.frames = frames

        if self.full_body:
            joints2d_face = wb_kps['joints2d_face']
            joints2d_lhand = wb_kps['joints2d_lhand']
            joints2d_rhand = wb_kps['joints2d_rhand']

            joints_part = {'lhand': joints2d_lhand, 'rhand': joints2d_rhand, 'face': joints2d_face}

            self.vis_part = {'lhand': wb_kps['vis_lhand'], 'rhand': wb_kps['vis_rhand'], 'face': wb_kps['vis_face']}

            self.bboxes_part = {}
            self.joints2d_part = {}

            for part, joints in joints_part.items():
                self.joints2d_part[part] = joints


    def __len__(self):
        # return len(self.image_file_names)
        return len(self.bboxes)

    def rgb_processing(self, rgb_img, center, scale, res, rot=0., flip=0):
        """Process rgb image and do augmentation."""
        # crop
        crop_img_resized, crop_img, crop_shape = crop(rgb_img, center, scale, res, rot=rot)
        # flip the image
        if flip:
            crop_img_resized = flip_img(crop_img_resized)
            crop_img = flip_img(crop_img)
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

    def __getitem__(self, idx):
        if self.pre_load_imgs is not None:
            img = self.pre_load_imgs
        else:
            # img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
            img_orig = cv2.imread(self.image_file_names[idx])[:,:,::-1].copy().astype(np.float32)
            # img_orig = img.copy()
            orig_height, orig_width = img_orig.shape[:2]

        if not self.full_body:
            bbox = self.bboxes[idx]
            j2d = self.joints2d[idx] if self.has_keypoints else None

            norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                img,
                bbox,
                kp_2d=j2d,
                scale=self.scale_factor,
                crop_size=self.crop_size)

            if self.has_keypoints:
                return norm_img, kp_2d
            else:
                return norm_img
        else:
            item = {}

            scale = self.scale_factor
            rot = 0.
            flip = 0

            j2d = self.joints2d[idx]

            kp2d_valid = j2d[j2d[:, 2]>0.]
            bbox = [min(kp2d_valid[:, 0]), min(kp2d_valid[:, 1]),
                    max(kp2d_valid[:, 0]), max(kp2d_valid[:, 1])]
            center = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
            sc = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

            sc *= cfg.DATA.RESCALE_B

            img, _, crop_shape = self.rgb_processing(img_orig, center, sc*scale, [constants.IMG_RES, constants.IMG_RES])

            # Store image before normalization to use it in visualization
            item['img_body'] = self.normalize_img(torch.from_numpy(img).float())
            item['orig_height'] = orig_height
            item['orig_width'] = orig_width

            item['person_id'] = self.person_ids[idx]

            img_hr, img_crop, _ = self.rgb_processing(img_orig, center, sc*scale, [constants.IMG_RES * 8, constants.IMG_RES * 8])

            kps_transf = get_transform(center, sc * scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)

            lhand_kp2d, rhand_kp2d, face_kp2d = self.joints2d_part['lhand'][idx], self.joints2d_part['rhand'][idx], self.joints2d_part['face'][idx]

            if cfg.MODEL.PyMAF.HF_BOX_ALIGN:

                hand_kp2d = self.j2d_processing(np.concatenate([lhand_kp2d, rhand_kp2d]).copy(), kps_transf, flip, is_hand=True)
                face_kp2d = self.j2d_processing(face_kp2d.copy(), kps_transf, flip, is_face=True)

                n_hand_kp = len(constants.HAND_NAMES)

                part_kp2d_dict = {'lhand': hand_kp2d[:n_hand_kp], 'rhand': hand_kp2d[n_hand_kp:], 'face': face_kp2d}

                for part in ['lhand', 'rhand', 'face']:
                    kp2d = part_kp2d_dict[part]
                    kp2d_valid = kp2d[kp2d[:, 2]>0.]
                    if len(kp2d_valid) > 0:
                        bbox = [min(kp2d_valid[:, 0]), min(kp2d_valid[:, 1]),
                                max(kp2d_valid[:, 0]), max(kp2d_valid[:, 1])]
                        center_part = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
                        scale_part = 2. * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

                    # handle invalid part keypoints
                    if len(kp2d_valid) < 1 or scale_part < 0.01:
                        center_part = [0, 0]
                        scale_part = 0.5
                        kp2d[:, 2] = 0

                    center_part = torch.tensor(center_part).float()

                    theta_part = torch.zeros(1, 2, 3)
                    theta_part[:, 0, 0] = scale_part
                    theta_part[:, 1, 1] = scale_part
                    theta_part[:, :, -1] = center_part

                    crop_hf_img_size = torch.Size([1, 3, cfg.MODEL.PyMAF.HF_IMG_SIZE, cfg.MODEL.PyMAF.HF_IMG_SIZE])
                    grid = F.affine_grid(theta_part.detach(), crop_hf_img_size, align_corners=False)
                    img_part = F.grid_sample(torch.from_numpy(img_crop[None]), grid.cpu(), align_corners=False).squeeze(0)

                    item[f'img_{part}'] = self.normalize_img(img_part.float())

                    theta_i_inv = torch.zeros_like(theta_part)
                    theta_i_inv[:, 0, 0] = 1. / theta_part[:, 0, 0]
                    theta_i_inv[:, 1, 1] = 1. / theta_part[:, 1, 1]
                    theta_i_inv[:, :, -1] = - theta_part[:, :, -1] / theta_part[:, 0, 0].unsqueeze(-1)

                    item[f'{part}_theta_inv'] = theta_i_inv[0]

                    if part in self.vis_part:
                        item[f'vis_{part}'] = self.vis_part[part][idx]
            else:
                n_hand_kp = len(constants.HAND_NAMES)
                assert len(lhand_kp2d) == n_hand_kp
                assert len(rhand_kp2d) == n_hand_kp

                part_kp2d_dict = {'lhand': lhand_kp2d, 'rhand': rhand_kp2d, 'face': face_kp2d}

                for part in ['lhand', 'rhand', 'face']:
                    kp2d = part_kp2d_dict[part]
                    kp2d_valid = kp2d[kp2d[:, 2]>0.]
                    if len(kp2d_valid) > 0:
                        bbox = [min(kp2d_valid[:, 0]), min(kp2d_valid[:, 1]),
                                max(kp2d_valid[:, 0]), max(kp2d_valid[:, 1])]
                        center_part = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
                        scale_part = 1.2 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

                        if part in ['lhand', 'rhand']:
                            scale_part *= cfg.DATA.RESCALE_H
                        elif part in ['face']:
                            scale_part *= cfg.DATA.RESCALE_F

                    # handle invalid part keypoints
                    if len(kp2d_valid) < 1 or scale_part < 0.01:
                        center_part = [0, 0]
                        scale_part = 0.5
                        kp2d[:, 2] = 0

                    img_part, _, crop_shape = self.rgb_processing(img_orig, center_part, scale_part, [constants.IMG_RES, constants.IMG_RES])

                    item[f'img_{part}'] = self.normalize_img(torch.from_numpy(img_part).float())

                    if part in self.vis_part:
                        item[f'vis_{part}'] = self.vis_part[part][idx]

            return item


class ImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)
        return to_tensor(img)

