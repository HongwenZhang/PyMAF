# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/train/trainer.py

import os
from genericpath import exists
from pickle import NONE
from telnetlib import EXOPL
import time
from tkinter.messagebox import NO
import torch
import numpy as np
from torch._C import device
from torch.cuda import random
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from .base_trainer import BaseTrainer
from datasets import MixedDataset, BaseDataset
from models import hmr, pymaf_net, SMPL
from utils.pose_utils import compute_similarity_transform_batch
from utils.geometry import batch_rodrigues, projection, perspective_projection, estimate_translation, rot6d_to_rotmat, rotation_matrix_to_angle_axis

from core import path_config, constants
from .fits_dict import FitsDict, HandFaceFitsDict
from .cfgs import cfg
from utils.train_utils import print_args
from utils.iuvmap import iuv_img2map, iuv_map2img, seg_img2map
from utils.imutils import flip_aa, j2d_processing
from models.smpl import get_model_faces, get_partial_smpl
from utils.vis import vis_batch_image_with_joints
from einops import rearrange

from models.smpl import SMPL, SMPL_MODEL_DIR

try:
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
except:
    pass

# try:
from utils.renderer import PyRenderer
from utils.renderer import OpenDRenderer, IUV_Renderer
# from utils.renderer import OpenDRenderer
# except:
#     print('fail to import Renderer.')

import logging
logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def init_fn(self):
        if self.options.rank == 0:
            self.summary_writer.add_text('command_args', print_args())

        if self.options.regressor == 'hmr':
            # HMR/SPIN model
            self.model = hmr(path_config.SMPL_MEAN_PARAMS, pretrained=True)
            self.mesh_model = SMPL(path_config.SMPL_MODEL_DIR,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             create_transl=False)
        elif self.options.regressor == 'pymaf_net':
            # PyMAF model
            self.model = pymaf_net(path_config.SMPL_MEAN_PARAMS, device=self.device)
            if cfg.MODEL.MESH_MODEL == 'mano':
                self.mesh_model = self.model.smpl_family['hand'].model
            elif cfg.MODEL.MESH_MODEL == 'flame':
                self.mesh_model = self.model.smpl_family['face'].model
            else:
                self.mesh_model = self.model.smpl_family['body'].model
            
            self.smpl_male = SMPL(model_path=path_config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(self.device)
            self.smpl_female = SMPL(model_path=path_config.SMPL_MODEL_DIR,
                            gender='female',
                            create_transl=False).to(self.device)

            self.bhf_names = self.model.bhf_names
            self.part_names = self.model.part_names

            self.hand_only_mode = self.model.hand_only_mode
            self.face_only_mode = self.model.face_only_mode
            self.full_body_mode = self.model.full_body_mode

        if self.options.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            find_unused_parameters = False
            if self.options.gpu is not None:
                torch.cuda.set_device(self.options.gpu)
                self.model.cuda(self.options.gpu)
                self.mesh_model.cuda(self.options.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.options.batch_size = int(self.options.batch_size / self.options.ngpus_per_node)
                self.options.workers = int((self.options.workers + self.options.ngpus_per_node - 1) / self.options.ngpus_per_node)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.options.gpu], 
                                                                       output_device=self.options.gpu, 
                                                                       find_unused_parameters=find_unused_parameters)
            else:
                self.model.cuda()
                self.mesh_model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=find_unused_parameters)
            self.models_dict = {'model': self.model.module}
        else:
            self.model = self.model.to(self.device)
            self.mesh_model = self.mesh_model.to(self.device)
            self.models_dict = {'model': self.model}

        cudnn.benchmark = True

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.focal_length = constants.FOCAL_LENGTH

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)
        
        if self.options.pretrained_backbone is not None:
            self.load_backbone(checkpoint_file=self.options.pretrained_backbone)

        if self.options.pretrained_body or self.options.pretrained_hand or self.options.pretrained_face:
            self.load_pretrained_full_body(checkpoint_body=self.options.pretrained_body,
                                            checkpoint_hand=self.options.pretrained_hand,
                                            checkpoint_face=self.options.pretrained_face,
                                            )

        if len(cfg.TRAIN.FREEZE_ENCODER) > 0 or len(cfg.TRAIN.FREEZE_PART) > 0:
            for name, param in self.model.named_parameters():
                if len(cfg.TRAIN.FREEZE_ENCODER) > 0:
                    for part in cfg.TRAIN.FREEZE_ENCODER.split(','):
                        assert part in ['body', 'hand', 'face']
                        if name.startswith(f'encoders.{part}'):
                            param.requires_grad = False

                if len(cfg.TRAIN.FREEZE_PART) > 0:
                    for part in cfg.TRAIN.FREEZE_PART.split(','):
                        assert part in ['body', 'hand', 'face']
                        for part_module in self.model.part_module_names[part].keys():
                            if name.startswith(part_module):
                                param.requires_grad = False

            opt_params = [param for param in self.model.parameters() if param.requires_grad]
        else:
            opt_params = self.model.parameters()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        print('total model size:', sum([np.prod(p.size()) for p in model_parameters]))


        self.optimizer = torch.optim.Adam(params=opt_params,
                                        lr=cfg.SOLVER.BASE_LR,
                                        weight_decay=0)

        self.optimizers_dict = {'optimizer': self.optimizer}

        self.train_ds = MixedDataset(self.options, is_train=True)
        print('---- Data Summary ----')
        for ds in self.train_ds.datasets:
            print(ds.dataset, ds.pose.shape, ds.betas.shape, 'valid fits: ', int(np.sum(ds.has_smpl)))
        for ds in self.train_ds.datasets:
            print(ds.dataset, 'hands/face valid fits: ', int(np.sum(ds.has_hf[:, 0])), int(np.sum(ds.has_hf[:, 1])), int(np.sum(ds.has_hf[:, 2])))

        self.valid_ds = BaseDataset(self.options, self.options.eval_dataset, is_train=False)

        if self.options.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds)
            val_sampler = None
        else:
            train_sampler = None
            val_sampler = None

        train_shuffle = (train_sampler is None)
        if not cfg.TRAIN.SHUFFLE:
            train_shuffle = False

        self.train_data_loader = DataLoader(
            self.train_ds, 
            batch_size=self.options.batch_size,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.TRAIN.PIN_MEMORY,
            shuffle=train_shuffle,
            sampler=train_sampler
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_ds,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.TRAIN.PIN_MEMORY,
            sampler=val_sampler
        )

        self.evaluation_accumulators = dict.fromkeys(['errors', 'errors_pa', 'errors_pve'])

        self.smpl_mode = (cfg.MODEL.MESH_MODEL == 'smpl')
        self.smplx_mode = (cfg.MODEL.MESH_MODEL == 'smplx')

        # Create renderer
        self.renderer = PyRenderer()

        if not self.smpl_mode:
            self.smpl2limb_vert_faces = get_partial_smpl(device=torch.device('cpu'))

            h_root_idx = constants.HAND_NAMES.index('wrist')
            h_idx = constants.HAND_NAMES.index('middle1')
            f_idx = constants.FACIAL_LANDMARKS.index('nose_middle')
            # self.hf_center_idx = {'lhand': h_idx, 'rhand': h_idx, 'face': f_idx}
            # self.hf_root_idx = {'lhand': h_idx, 'rhand': h_idx, 'face': f_idx}
            self.hf_root_idx = {'lhand': h_root_idx, 'rhand': h_root_idx, 'face': f_idx}

            lh_idx_spin = constants.SPIN_JOINT_NAMES.index('Left Wrist')
            rh_idx_spin = constants.SPIN_JOINT_NAMES.index('Right Wrist')
            f_idx_spin = constants.SPIN_JOINT_NAMES.index('Nose')
            self.hf_root_idx_spin = {'lhand': lh_idx_spin, 'rhand': rh_idx_spin, 'face': f_idx_spin}

        if 'body' in self.bhf_names:
            if cfg.MODEL.PyMAF.AUX_SUPV_ON:
                self.iuv_maker = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, mode='iuv', device=self.device, mesh_type='smpl')
            elif cfg.MODEL.PyMAF.SEG_ON:
                self.iuv_maker = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, mode='seg', device=self.device)

        if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
            self.iuv_maker_part = {}
            if 'rhand' in self.part_names:
                self.iuv_maker_part['rhand'] = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, mode='pncc', device=self.device, mesh_type='mano')
            if 'face' in self.part_names:
                self.iuv_maker_part['face'] = IUV_Renderer(output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, mode='pncc', device=self.device, mesh_type='flame')

        self.decay_steps_ind = 1
        self.decay_epochs_ind = 1

    def load_pretrained_full_body(self, checkpoint_body=None, checkpoint_hand=None, checkpoint_face=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        checkpoint_paths = {'body': checkpoint_body, 'hand': checkpoint_hand, 'face': checkpoint_face}
        for part in ['body', 'hand', 'face']:
            checkpoint_path = checkpoint_paths[part]
            if checkpoint_path is not None:
                print('Loading part model for ', part)
                checkpoint = torch.load(checkpoint_path)['model']
                checkpoint_filtered = {}
                key_start_list = self.models_dict['model'].part_module_names[part].keys()
                for key in list(checkpoint.keys()):
                    for key_start in key_start_list:
                        if key.startswith(key_start):
                            checkpoint_filtered[key] = checkpoint[key]
                self.models_dict['model'].load_state_dict(checkpoint_filtered, strict=False)
                print(f'Checkpoint for {part} loaded.')

    def finalize(self):
        pass
        # self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight, ):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss
    
    def hf_keypoint_loss(self, pred_keypoints, gt_keypoints, valid):
        """ Compute loss on the hand and face keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        if valid.sum() > 0:
            n_kps, dim = pred_keypoints.shape[1:]
            conf = gt_keypoints[valid, :, -1].unsqueeze(-1).clone()
            loss = (conf * self.criterion_keypoints(pred_keypoints[valid], gt_keypoints[valid, :, :-1]))
            loss = loss.sum() / (valid.sum() * n_kps * dim)
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d]
        conf = conf[has_pose_3d]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def vert_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl]
        gt_vertices_with_shape = gt_vertices[has_smpl]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def pose_shape_losses(self, pred_rotmat, pred_betas, gt_rotmat, gt_betas, has_smpl):
        n_joint = pred_rotmat.shape[1]
        pred_rotmat_valid = pred_rotmat[has_smpl]
        # gt_rotmat_valid = batch_rodrigues(gt_pose.contiguous().view(-1,3)).view(-1, n_joint, 3, 3)[has_smpl]
        gt_rotmat_valid = gt_rotmat[has_smpl]
        pred_betas_valid = pred_betas[has_smpl]
        gt_betas_valid = gt_betas[has_smpl]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def param_losses(self, pred_params, gt_params, valid):
        pred_params_valid = pred_params[valid]
        gt_params_valid = gt_params[valid]
        if len(pred_params_valid) > 0:
            loss_regr = self.criterion_regr(pred_params_valid, gt_params_valid)
        else:
            loss_regr = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr

    def body_uv_losses(self, u_pred, v_pred, index_pred, ann_pred, uvia_list, has_iuv=None):
        batch_size = index_pred.size(0)
        device = index_pred.device

        Umap, Vmap, Imap, Annmap = uvia_list

        if has_iuv is not None:
            if torch.sum(has_iuv.float()) > 0:
                u_pred = u_pred[has_iuv] if u_pred is not None else u_pred
                v_pred = v_pred[has_iuv] if v_pred is not None else v_pred
                index_pred = index_pred[has_iuv] if index_pred is not None else index_pred
                ann_pred = ann_pred[has_iuv] if ann_pred is not None else ann_pred
                Imap = Imap[has_iuv]
                if Umap is not None:
                    Umap, Vmap = Umap[has_iuv], Vmap[has_iuv]
                else:
                    Umap, Vmap = None, None
                Annmap = Annmap[has_iuv] if Annmap is not None else Annmap
            else:
                return (torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device))

        Itarget = torch.argmax(Imap, dim=1)
        Itarget = Itarget.view(-1).to(torch.int64)

        index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
        index_pred = index_pred.view(-1, Imap.size(1))

        loss_IndexUV = F.cross_entropy(index_pred, Itarget)

        if cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0 and Umap is not None:
            loss_U = F.smooth_l1_loss(u_pred[Imap > 0], Umap[Imap > 0], reduction='sum') / batch_size
            loss_V = F.smooth_l1_loss(v_pred[Imap > 0], Vmap[Imap > 0], reduction='sum') / batch_size

            loss_U *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
            loss_V *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
        else:
            loss_U, loss_V = torch.zeros(1).to(device), torch.zeros(1).to(device)

        if ann_pred is None:
            loss_segAnn = torch.zeros(1).to(device)
        else:
            Anntarget = torch.argmax(Annmap, dim=1)
            Anntarget = Anntarget.view(-1).to(torch.int64)
            ann_pred = ann_pred.permute([0, 2, 3, 1]).contiguous()
            ann_pred = ann_pred.view(-1, Annmap.size(1))
            loss_segAnn = F.cross_entropy(ann_pred, Anntarget)

        return loss_U, loss_V, loss_IndexUV, loss_segAnn
    
    def pncc_losses(self, pred_pncc, gt_pncc, valid):
        batch_size = pred_pncc.size(0)
        device = pred_pncc.device

        pred_pncc_valid = pred_pncc[valid]
        gt_pncc_valid = gt_pncc[valid]

        if len(pred_pncc_valid) > 0:
            # loss_pncc = F.smooth_l1_loss(pred_pncc_valid[gt_pncc_valid > 0], gt_pncc_valid[gt_pncc_valid > 0], reduction='sum') / len(pred_pncc_valid)
            loss_pncc = F.smooth_l1_loss(pred_pncc_valid, gt_pncc_valid, reduction='sum') / len(pred_pncc_valid)
            loss_pncc *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
        else:
            loss_pncc = torch.zeros(1).to(device)

        return loss_pncc
    
    def train(self, epoch):
        """Training process."""
        if self.options.distributed:
            self.train_data_loader.sampler.set_epoch(epoch)

        self.model.train()

        # Learning rate decay
        if self.decay_epochs_ind < len(cfg.SOLVER.EPOCHS) and epoch == cfg.SOLVER.EPOCHS[self.decay_epochs_ind]:
            lr = self.optimizer.param_groups[0]['lr']
            lr_new = lr * cfg.SOLVER.GAMMA
            print('Decay the learning on epoch {} from {} to {}'.format(epoch, lr, lr_new))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_new
            lr = self.optimizer.param_groups[0]['lr']
            assert lr == lr_new
            self.decay_epochs_ind += 1

        if self.options.rank == 0:
            pbar = tqdm(desc=self.options.log_name + ' Epoch ' + str(epoch),
                                            total=len(self.train_ds) // cfg.TRAIN.BATCH_SIZE,
                                            initial=self.checkpoint_batch_idx)

        # Iterate over all batches in an epoch
        for step, batch in enumerate(self.train_data_loader, self.checkpoint_batch_idx):
            if self.options.rank == 0:
                pbar.update(1)

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and k not in ['img_hr'] else v for k,v in batch.items()}
            self.step_count += 1
            out = self.train_step(batch)

            if self.step_count % 5000 == 0 and self.options.rank == 0:
                # if self.options.train_data == 'surreal':
                if self.options.train_data == 'expose_hand':
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, cfg.TRAIN.BATCH_SIZE, self.step_count,
                                    interval=1, with_optimizer=False)

            # Tensorboard logging every summary_steps steps
            if self.step_count % cfg.TRAIN_VIS_ITER_FERQ == 0 and self.options.rank == 0:
                self.model.eval()
                self.visualize(self.step_count, batch, 'train', **out)
                self.model.train()

        if self.options.rank == 0:
            pbar.close()

        # load a checkpoint only on startup, for the next epochs
        # just iterate over the dataset as usual
        self.checkpoint = None
        return

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'].to(torch.bool) # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].to(torch.bool) # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        opt_pose, opt_betas = gt_pose, gt_betas

        if self.hand_only_mode or self.face_only_mode:
            opt_pose[:, :3] = input_batch['global_orient']

        if not self.smpl_mode:
            has_hf = input_batch['has_hf'].to(torch.bool) # flag that indicates whether hand and face params are valid
            valid_fit_hf = has_hf
            valid_fit_hf_dict = {'lhand': has_hf[:, 0], 'rhand': has_hf[:, 1], 'face': has_hf[:, 2]}

            opt_hfrotmat_dict = {}
            opt_hfshape_dict = {}
            if self.hand_only_mode:
                gt_hfpose = torch.cat([input_batch['left_hand_pose'], input_batch['right_hand_pose']], dim=1)

                opt_hfpose = gt_hfpose
                opt_hfrotmat = batch_rodrigues(opt_hfpose.contiguous().view(-1,3)).view(batch_size, -1, 3, 3)
                assert opt_hfrotmat.shape[1] == 30
                opt_hfrotmat_dict['lhand'] = opt_hfrotmat[:, :15]
                opt_hfrotmat_dict['rhand'] = opt_hfrotmat[:, 15:30]
                opt_hfrotmat_dict['rhand_orient'] = batch_rodrigues(input_batch['global_orient'].contiguous().view(-1,3)).view(batch_size, 3, 3)
                opt_hfshape_dict['rhand_shape'] = input_batch['rhand_shape']
            elif self.face_only_mode:
                gt_hfpose = torch.cat([input_batch['jaw_pose'], input_batch['leye_pose'], input_batch['reye_pose']], dim=1)
                gt_exp = input_batch['expression']
                opt_exp = gt_exp

                opt_hfpose = gt_hfpose
                opt_hfrotmat = batch_rodrigues(opt_hfpose.contiguous().view(-1,3)).view(batch_size, -1, 3, 3)
                assert opt_hfrotmat.shape[1] == 3
                opt_hfrotmat_dict['face'] = opt_hfrotmat
                opt_hfrotmat_dict['face_orient'] = batch_rodrigues(input_batch['global_orient'].contiguous().view(-1,3)).view(batch_size, 3, 3)
                opt_hfshape_dict['face_shape'] = input_batch['face_shape']
            else:
                gt_hfpose = torch.cat([input_batch['left_hand_pose'], input_batch['right_hand_pose'],
                                    input_batch['jaw_pose'], input_batch['leye_pose'], input_batch['reye_pose']
                                    ], dim=1)
                gt_exp = input_batch['expression']
                opt_exp = gt_exp

                opt_hfpose = gt_hfpose
                opt_hfrotmat = batch_rodrigues(opt_hfpose.contiguous().view(-1,3)).view(batch_size, -1, 3, 3)
                assert opt_hfrotmat.shape[1] == 33
                opt_hfrotmat_dict['lhand'] = opt_hfrotmat[:, :15]
                opt_hfrotmat_dict['rhand'] = opt_hfrotmat[:, 15:30]
                opt_hfrotmat_dict['face'] = opt_hfrotmat[:, 30:]

        if self.smpl_mode:
            opt_output = self.mesh_model(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
            opt_vertices = opt_output.vertices
            opt_joints = opt_output.joints
        else:
            if cfg.MODEL.MESH_MODEL == 'mano':
                opt_mano_params = {'betas': opt_hfshape_dict['rhand_shape'], 'global_orient': input_batch['global_orient'], 'right_hand_pose': opt_hfpose[:,45:90],
                                }
                opt_output = self.mesh_model(**opt_mano_params)
                opt_vertices_rh = opt_output.rhand_vertices
                opt_joints_rh = opt_output.rhand_joints
                opt_vertices = None
                opt_joints = None
                input_batch['verts_rh'] = opt_vertices_rh
            elif cfg.MODEL.MESH_MODEL == 'flame':
                opt_flame_params = {'betas': opt_hfshape_dict['face_shape'],
                                   'global_orient': input_batch['global_orient'],
                                   'jaw_pose': opt_hfpose[:,0:3], 
                                   'leye_pose': opt_hfpose[:,3:6],
                                   'reye_pose': opt_hfpose[:,6:9],
                                   'expression': opt_exp,
                                }
                opt_output = self.mesh_model(**opt_flame_params)
                opt_vertices_fa = opt_output.flame_vertices
                opt_joints_face = opt_output.face_joints
                opt_vertices = None
                opt_joints = None
                input_batch['verts_fa'] = opt_vertices_fa
            elif cfg.MODEL.MESH_MODEL == 'smplx':
                opt_smplx_params = {'betas': opt_betas, 'body_pose': opt_pose[:,3:], 'global_orient': opt_pose[:,:3],
                                'left_hand_pose': opt_hfpose[:,:45], 'right_hand_pose': opt_hfpose[:,45:90],
                                'jaw_pose': opt_hfpose[:,90:93], 'leye_pose': opt_hfpose[:,93:96], 'reye_pose': opt_hfpose[:,96:99],
                                'expression': opt_exp,
                                'gender': input_batch['gender']
                                }
                opt_output = self.mesh_model(**opt_smplx_params)

                opt_vertices = opt_output.vertices
                opt_joints = opt_output.joints
                opt_joints_rh = opt_output.rhand_joints

                is_synth = torch.tensor(['agora' in dn for dn in dataset_name]).bool()
                opt_j24 = opt_joints[:, -24:] - torch.mean(opt_joints[:, -24:][:, 2:4], dim=1, keepdim=True)
                input_batch['pose_3d'][is_synth, :, :3] = opt_j24[is_synth]

        input_batch['verts'] = opt_vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        if self.hand_only_mode:
            gt_hand_keypoints_2d_orig = input_batch['rhand_kp2d'].clone()
            gt_hand_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_hand_keypoints_2d_orig[:, :, :-1] + 1)
        elif self.face_only_mode:
            gt_face_keypoints_2d_orig = input_batch['face_kp2d'].clone()
            gt_face_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_face_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        # gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=(self.options.img_res, self.options.img_res))
        if cfg.MODEL.USE_IWP_CAM or (cfg.MODEL.PyMAF.AUX_SUPV_ON or cfg.MODEL.PyMAF.SEG_ON):
            # print(opt_joints[0], gt_keypoints_2d_orig[0])
            if self.hand_only_mode:
                joints_3d = opt_joints_rh
                joints_proj = gt_hand_keypoints_2d_orig
                use_all_kps = True
            elif self.face_only_mode:
                joints_3d = opt_joints_face
                joints_proj = gt_face_keypoints_2d_orig
                use_all_kps = True
            else:
                joints_3d = opt_joints
                joints_proj = gt_keypoints_2d_orig
                use_all_kps = False
            opt_cam_t = estimate_translation(joints_3d, joints_proj, focal_length=self.focal_length, img_size=self.options.img_res, use_all_kps=use_all_kps)
            gt_cam_t_nr = opt_cam_t.detach().clone()
            gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
            gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
            gt_camera[:, 0] = (2. * self.focal_length / self.options.img_res) / gt_cam_t_nr[:, 2]
            if cfg.MODEL.USE_IWP_CAM:
                input_batch['gt_cam'] = gt_camera

        if not cfg.MODEL.USE_IWP_CAM:
            gt_keypoints_2d_full_img = input_batch['keypoints_full_img']
            if self.hand_only_mode:
                gt_hand_keypoints_2d_full_img = input_batch['rhand_keypoints_full_img'].clone()
            elif self.face_only_mode:
                gt_face_keypoints_2d_full_img = input_batch['face_keypoints_full_img'].clone()

            if self.hand_only_mode:
                joints_3d = opt_joints_rh
                joints_proj = gt_hand_keypoints_2d_full_img
                use_all_kps = True
            elif self.face_only_mode:
                joints_3d = opt_joints_face
                joints_proj = gt_face_keypoints_2d_full_img
                use_all_kps = True
            else:
                joints_3d = opt_joints
                joints_proj = gt_keypoints_2d_full_img
                use_all_kps = False

            if cfg.MODEL.PRED_PITCH:
                joints_3d_rot = torch.einsum('bij,bkj->bki', input_batch['cam_rotmat'], joints_3d)
            else:
                joints_3d_rot = joints_3d

            opt_cam_t = estimate_translation(joints_3d_rot.detach(), joints_proj, 
                                             focal_length=input_batch['cam_focal_length'].cpu().numpy(), img_size=input_batch['orig_shape'].cpu().numpy(),
                                             use_all_kps=use_all_kps)

        valid_fit = has_smpl
        # get fitted smpl parameters as pseudo ground truth
        if False:
            valid_fit = self.fits_dict.get_vaild_state(dataset_name, indices.cpu()).to(torch.bool).to(self.device)

            try:
                valid_fit = valid_fit | has_smpl
            except RuntimeError:
                valid_fit = (valid_fit.byte() | has_smpl.byte()).to(torch.bool)

        # Render Dense Correspondences
        if self.options.regressor == 'pymaf_net':
            if 'body' in self.bhf_names and (cfg.MODEL.PyMAF.AUX_SUPV_ON or cfg.MODEL.PyMAF.SEG_ON):
                iuv_image_gt = torch.zeros((batch_size, 3, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)).to(self.device)
                if torch.sum(valid_fit.float()) > 0:
                    iuv_image_gt[valid_fit] = self.iuv_maker.verts2iuvimg(opt_vertices[valid_fit], cam=gt_camera[valid_fit])  # [B, 3, 56, 56]
                input_batch['iuv_image_gt'] = iuv_image_gt
                img2map_fn = seg_img2map if cfg.MODEL.PyMAF.SEG_ON else iuv_img2map
                uvia_list = img2map_fn(iuv_image_gt)

            if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                part_pncc_gt = {}
                for part in self.part_names:
                    pncc_image_gt = torch.zeros((batch_size, 3, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)).to(self.device)
                    valid_p = valid_fit_hf_dict[part]
                    if torch.sum(valid_p.float()) > 0:
                        if part == 'rhand':
                            opt_vertices_part = opt_vertices_rh
                        elif part == 'face':
                            opt_vertices_part = opt_vertices_fa
                        pncc_image_gt[valid_p] = self.iuv_maker_part[part].verts2iuvimg(opt_vertices_part[valid_p], cam=gt_camera[valid_p])  # [B, 3, 56, 56]
                    part_pncc_gt[part] = pncc_image_gt
                    input_batch[f'{part}_pncc_gt'] = pncc_image_gt

        # Feed images in the network to predict camera and SMPL parameters
        if self.options.regressor == 'hmr':
            pred_rotmat, pred_betas, pred_camera = self.model(images)
            # torch.Size([32, 24, 3, 3]) torch.Size([32, 10]) torch.Size([32, 3])
        elif self.options.regressor == 'pymaf_net':
            spec_cam = {}
            if not cfg.MODEL.USE_IWP_CAM:
                spec_cam = {'bbox_scale': input_batch['scale'].float(),
                            'bbox_center': input_batch['center'],
                            'img_w': input_batch['orig_shape'][:, 1],
                            'img_h': input_batch['orig_shape'][:, 0],
                            'crop_res': self.options.img_res,
                            'cam_rotmat': input_batch['cam_rotmat'],
                            'cam_intrinsics': input_batch['cam_intrinsics'],
                            'kps_transf': input_batch['kps_transf'],
                            'rot_transf': input_batch['rot_transf'],
                            'vfov': input_batch['cam_vfov'],
                            'crop_ratio': input_batch['crop_ratio'],
                        }

            preds_dict, _ = self.model(input_batch, rw_cam=spec_cam)

        output = preds_dict

        loss_dict = {}

        if self.options.regressor == 'pymaf_net':
            if cfg.MODEL.PyMAF.AUX_SUPV_ON or cfg.MODEL.PyMAF.SEG_ON:
                dp_out = preds_dict['dp_out']
                for i in range(len(dp_out)):
                    r_i = i - len(dp_out)

                    if cfg.MODEL.PyMAF.AUX_SUPV_ON:
                        u_pred, v_pred, index_pred, ann_pred = dp_out[r_i]['predict_u'], dp_out[r_i]['predict_v'], dp_out[r_i]['predict_uv_index'], dp_out[r_i]['predict_ann_index']
                    elif cfg.MODEL.PyMAF.SEG_ON:
                        u_pred, v_pred, index_pred, ann_pred = None, None, dp_out[r_i]['predict_seg_index'], None
                    if index_pred.shape[-1] == iuv_image_gt.shape[-1]:
                        uvia_list_i = uvia_list
                    else:
                        iuv_image_gt_i = F.interpolate(iuv_image_gt, index_pred.shape[-1], mode='nearest')
                        uvia_list_i = img2map_fn(iuv_image_gt_i)

                    loss_U, loss_V, loss_IndexUV, loss_segAnn = self.body_uv_losses(u_pred, v_pred, index_pred, ann_pred,
                                                                                    uvia_list_i, valid_fit)
                    loss_dict[f'loss_U{r_i}'] = loss_U
                    loss_dict[f'loss_V{r_i}'] = loss_V
                    loss_dict[f'loss_IndexUV{r_i}'] = loss_IndexUV
                    loss_dict[f'loss_segAnn{r_i}'] = loss_segAnn
            
            if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                for part in self.part_names:
                    pncc_out = preds_dict[f'{part}_dpout']
                    for i in range(len(pncc_out)):
                        r_i = i - len(pncc_out)
                        pred_pncc = pncc_out[r_i]['predict_pncc']
                        loss_pncc = self.pncc_losses(pred_pncc, part_pncc_gt[part], valid_fit_hf_dict[part]) * 0.01
                        loss_dict[f'loss_{part}_pncc{r_i}'] = loss_pncc

        len_loop = len(preds_dict['mesh_out']) if self.options.regressor == 'pymaf_net' else 1

        for l_i in range(len_loop):

            if cfg.MODEL.PyMAF.SUPV_LAST:
                if l_i < (len_loop - 1):
                    continue

            if self.options.regressor == 'pymaf_net':
                if l_i == 0:
                    # initial parameters (mean poses)
                    continue
                # pred_rotmat = preds_dict['mesh_out'][l_i]['rotmat']
                # pred_betas = preds_dict['mesh_out'][l_i]['theta'][:, 3:13]
                pred_camera =  preds_dict['mesh_out'][l_i]['theta'][:, :3]

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                    pred_camera[:,2],
                                    2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

            # Compute loss on SMPL parameters
            if cfg.LOSS.POSE_W > 0:
                if 'body' in self.bhf_names:
                    pred_rotmat = preds_dict['mesh_out'][l_i]['rotmat']
                    pred_betas = preds_dict['mesh_out'][l_i]['theta'][:, 3:13]
                    opt_rotmat = batch_rodrigues(opt_pose.contiguous().view(-1,3)).view(-1, pred_rotmat.shape[1], 3, 3)
                    if self.full_body_mode:
                        loss_regr_pose, loss_regr_betas = self.pose_shape_losses(pred_rotmat[:, :20], pred_betas, opt_rotmat[:, :20], opt_betas, valid_fit)
                    else:
                        if cfg.LOSS.FEET_KP_2D_W > 0:
                            seleted_idx = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                            loss_regr_pose, loss_regr_betas = self.pose_shape_losses(pred_rotmat[:, seleted_idx], pred_betas, opt_rotmat[:, seleted_idx], opt_betas, valid_fit)
                        else:
                            loss_regr_pose, loss_regr_betas = self.pose_shape_losses(pred_rotmat, pred_betas, opt_rotmat, opt_betas, valid_fit)
                    loss_regr_pose *= cfg.LOSS.POSE_W
                    loss_regr_betas *= cfg.LOSS.SHAPE_W
                    loss_dict['loss_regr_pose_{}'.format(l_i)] = loss_regr_pose
                    loss_dict['loss_regr_betas_{}'.format(l_i)] = loss_regr_betas

                # if self.smplx_mode:
                if self.hf_loss_on:
                    if self.hand_only_mode:                        
                        loss_dict['loss_regr_rhorient_{}'.format(l_i)] = self.param_losses(preds_dict['mesh_out'][l_i]['pred_orient_rh_rotmat'], opt_hfrotmat_dict['rhand_orient'], valid_fit_hf_dict['rhand']) * cfg.LOSS.POSE_W
                        loss_dict['loss_regr_rhshape_{}'.format(l_i)] = self.param_losses(preds_dict['mesh_out'][l_i]['pred_shape_rh'], opt_hfshape_dict['rhand_shape'], valid_fit_hf_dict['rhand']) * cfg.LOSS.SHAPE_W
                    elif self.face_only_mode:     
                        loss_dict['loss_regr_faorient_{}'.format(l_i)] = self.param_losses(preds_dict['mesh_out'][l_i]['pred_orient_fa_rotmat'], opt_hfrotmat_dict['face_orient'], valid_fit_hf_dict['face']) * cfg.LOSS.POSE_W
                        loss_dict['loss_regr_fashape_{}'.format(l_i)] = self.param_losses(preds_dict['mesh_out'][l_i]['pred_shape_fa'], opt_hfshape_dict['face_shape'], valid_fit_hf_dict['face']) * cfg.LOSS.SHAPE_W
                    elif self.full_body_mode:
                        loss_regr_lwrist = self.param_losses(pred_rotmat[:, 20:21], opt_rotmat[:, 20:21], valid_fit_hf_dict['lhand']) * cfg.LOSS.POSE_W
                        loss_regr_rwrist = self.param_losses(pred_rotmat[:, 21:22], opt_rotmat[:, 21:22], valid_fit_hf_dict['rhand']) * cfg.LOSS.POSE_W
                        loss_dict['loss_regr_wrist_pose_{}'.format(l_i)] = loss_regr_lwrist + loss_regr_rwrist

                        if cfg.MODEL.PyMAF.RES_EBTW:
                            loss_regr_lelbow = self.param_losses(pred_rotmat[:, 18:19], opt_rotmat[:, 18:19], valid_fit_hf_dict['lhand']) * cfg.LOSS.POSE_W
                            loss_regr_relbow = self.param_losses(pred_rotmat[:, 19:20], opt_rotmat[:, 19:20], valid_fit_hf_dict['rhand']) * cfg.LOSS.POSE_W
                            loss_dict['loss_regr_elbow_twist_{}'.format(l_i)] = loss_regr_lelbow + loss_regr_relbow

                    for part in self.part_names:
                        loss_dict['loss_regr_{}_pose_{}'.format(part, l_i)] = self.param_losses(preds_dict['mesh_out'][l_i][f'pred_{part}_rotmat'], 
                                                                                                  opt_hfrotmat_dict[f'{part}'], 
                                                                                                  valid_fit_hf_dict[f'{part}']) * cfg.LOSS.POSE_W

                        if part == 'face':
                            pred_exp =  preds_dict['mesh_out'][l_i]['pred_exp']
                            loss_dict['loss_regr_exp_{}'.format(l_i)] = self.param_losses(pred_exp, opt_exp, valid_fit_hf_dict['face']) * cfg.LOSS.SHAPE_W

            # Compute 2D reprojection loss for the keypoints
            if cfg.LOSS.KP_2D_W > 0 or cfg.LOSS.HF_KP_2D_W > 0:
                if 'body' in self.bhf_names:
                    pred_keypoints_2d = preds_dict['mesh_out'][l_i]['kp_2d']
                    loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                        self.options.openpose_train_weight,
                                                        self.options.gt_train_weight) * cfg.LOSS.KP_2D_W
                    loss_dict['loss_keypoints_{}'.format(l_i)] = loss_keypoints

                # if self.smplx_mode:
                if self.hf_loss_on:  # and cfg.LOSS.HF_KP_2D_W > 0:
                    for part in self.part_names:
                        key_name = part + '_kp2d'
                        pred_kp2d_i = preds_dict['mesh_out'][l_i]['pred_'+key_name]

                        gt_kp2d_i = input_batch[key_name].detach()

                        if self.hand_only_mode or self.face_only_mode:
                            valid = gt_kp2d_i[:, self.hf_root_idx[part], -1] > 0.1

                            if cfg.TRAIN.HF_ROOT_ALIGN:
                                pred_root = pred_kp2d_i[:, self.hf_root_idx[part]].unsqueeze(1)
                                gt_root = gt_kp2d_i[:, self.hf_root_idx[part], :-1].unsqueeze(1)
                                pred_kp2d_i_local = pred_kp2d_i - pred_root
                                gt_kp2d_i_local = torch.cat([gt_kp2d_i[:, :, :-1] - gt_root, gt_kp2d_i[:, :, -1].unsqueeze(-1)], dim=-1)
                                loss_keypoints_part = self.hf_keypoint_loss(pred_kp2d_i_local, gt_kp2d_i_local, valid)
                            else:
                                loss_keypoints_part = self.hf_keypoint_loss(pred_kp2d_i, gt_kp2d_i, valid)

                            loss_dict['loss_{}_{}'.format(key_name, l_i)] = loss_keypoints_part * cfg.LOSS.KP_2D_W
                        else:
                            if cfg.TRAIN.HF_ROOT_ALIGN:
                                valid = gt_kp2d_i[:, self.hf_root_idx[part], -1] > 0.1
                                pred_root = pred_kp2d_i[:, self.hf_root_idx[part]].unsqueeze(1)
                                gt_root = gt_kp2d_i[:, self.hf_root_idx[part], :-1].unsqueeze(1)
                            else:
                                valid = torch.ones(gt_kp2d_i.shape[0]).bool()
                                valid_hf_id = gt_kp2d_i[:, :, -1] > 0.1
                                valid_hf_id = valid_hf_id.float()
                                num_valid_hf = torch.sum(valid_hf_id, dim=-1, keepdim=True)

                                gt_root = torch.sum(gt_kp2d_i[:, :, :2] * valid_hf_id.unsqueeze(-1), dim=1) / (num_valid_hf + 1e-9)
                                gt_root = gt_root.unsqueeze(1)
                                pred_root = torch.sum(pred_kp2d_i * valid_hf_id.unsqueeze(-1), dim=1) / (num_valid_hf + 1e-9)
                                pred_root = pred_root.unsqueeze(1)

                            pred_kp2d_i_local = pred_kp2d_i - pred_root
                            gt_kp2d_i_local = torch.cat([gt_kp2d_i[:, :, :-1] - gt_root, gt_kp2d_i[:, :, -1].unsqueeze(-1)], dim=-1)

                            loss_keypoints_part = self.hf_keypoint_loss(pred_kp2d_i_local, gt_kp2d_i_local, valid)
                            loss_dict['loss_{}_{}'.format(key_name, l_i)] = loss_keypoints_part * cfg.LOSS.HF_KP_2D_W

                        # if part not in ['face'] and cfg.TRAIN.HAND_KP2D_GLOBAL:
                        # if cfg.TRAIN.HAND_KP2D_GLOBAL:
                        if cfg.LOSS.GL_HF_KP_2D_W > 0:
                            loss_dict['loss_{}_global_{}'.format(key_name, l_i)] = self.hf_keypoint_loss(pred_kp2d_i, gt_kp2d_i, torch.ones(gt_kp2d_i.shape[0]).bool()) * cfg.LOSS.GL_HF_KP_2D_W

                if cfg.LOSS.FEET_KP_2D_W > 0:
                    key_name = 'feet_kp2d'
                    pred_kp2d_i = preds_dict['mesh_out'][l_i]['pred_'+key_name]
                    gt_kp2d_i = input_batch[key_name].detach()
                    loss_dict['loss_{}_{}'.format(key_name, l_i)] = self.hf_keypoint_loss(pred_kp2d_i, gt_kp2d_i, torch.ones(gt_kp2d_i.shape[0]).bool()) * cfg.LOSS.FEET_KP_2D_W

            if cfg.MODEL.PyMAF.PRED_VIS_H:
                pred_vis_hands = preds_dict['mesh_out'][l_i]['pred_vis_hands']
                gt_vis_lhand = torch.sum(input_batch['lhand_kp2d'][:, :, -1] > 0.1, dim=1, keepdim=True)
                gt_vis_rhand = torch.sum(input_batch['rhand_kp2d'][:, :, -1] > 0.1, dim=1, keepdim=True)
                gt_vis_hands = torch.cat([gt_vis_lhand, gt_vis_rhand], dim=1) / len(constants.HAND_NAMES)

                loss_dict['loss_vish_{}'.format(l_i)] = F.l1_loss(pred_vis_hands, gt_vis_hands).mean()

            # Compute 3D keypoint loss
            if cfg.LOSS.KP_3D_W > 0:
                if 'body' in self.bhf_names:
                    pred_joints = preds_dict['mesh_out'][l_i]['pred_joints']
                    loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d) * cfg.LOSS.KP_3D_W
                    loss_dict['loss_keypoints_3d_{}'.format(l_i)] = loss_keypoints_3d

                # if self.smplx_mode:
                if self.hf_loss_on and cfg.LOSS.HF_KP_3D_W > 0:
                    for part in self.part_names:
                        key_name = part + '_kp3d'
                        pred_kp3d_i = preds_dict['mesh_out'][l_i]['pred_'+key_name]
                        gt_kp3d_i = input_batch[key_name].detach()

                        pred_kp3d_i_root = pred_kp3d_i[:, self.hf_root_idx[part], :].detach()
                        gt_kp3d_i_root = gt_kp3d_i[:, self.hf_root_idx[part], :-1].detach()

                        pred_kp3d_i = pred_kp3d_i - pred_kp3d_i_root[:, None]
                        gt_kp3d_i = torch.cat([gt_kp3d_i[:, :, :-1] - gt_kp3d_i_root[:, None], gt_kp3d_i[:, :, 3:4]], dim=-1)

                        valid = gt_kp3d_i[:, self.hf_root_idx[part], -1].detach() > 0.
                        loss_dict['loss_{}_{}'.format(key_name, l_i)] = self.hf_keypoint_loss(pred_kp3d_i, gt_kp3d_i, valid) * cfg.LOSS.HF_KP_3D_W

            # Per-vertex loss for the shape
            if 'body' in self.bhf_names:
                pred_vertices = preds_dict['mesh_out'][l_i]['verts']
                if cfg.LOSS.VERT_W > 0:
                    loss_shape = self.vert_loss(pred_vertices, opt_vertices, valid_fit) * cfg.LOSS.VERT_W
                    loss_dict['loss_shape_{}'.format(l_i)] = loss_shape
            if 'hand' in self.bhf_names:
                pred_vertices_rh = preds_dict['mesh_out'][l_i]['verts_rh']

            # Camera
            # force the network to predict positive depth values
            if self.hand_only_mode or self.face_only_mode:
                loss_cam = ((torch.exp(-pred_camera[:,0])) ** 2 ).mean()
            else:
                loss_cam = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
            loss_dict['loss_cam_{}'.format(l_i)] = loss_cam

        for key in loss_dict:
            # print(key, loss_dict[key])
            if len(loss_dict[key].shape) > 0:
                loss_dict[key] = loss_dict[key][0]

        # Compute total loss
        loss = torch.stack(list(loss_dict.values())).sum()

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output.update({'pred_cam_t': pred_cam_t.detach(),
                       'opt_cam_t': opt_cam_t})
        if 'body' in self.bhf_names:
            output.update({'pred_vertices': pred_vertices.detach(),
                            'opt_vertices': opt_vertices})
        if self.hand_only_mode:
            output.update({'pred_vertices_rh': pred_vertices_rh.detach(),
                           'opt_vertices_rh': opt_vertices_rh,
                           'pred_rhand_kp2d': preds_dict['mesh_out'][-1]['pred_rhand_kp2d']
                            })
        if self.face_only_mode:
            output.update({'pred_vertices_fa': preds_dict['mesh_out'][l_i]['verts_fa'].detach(),
                           'opt_vertices_fa': opt_vertices_fa,
                           'pred_face_kp2d': preds_dict['mesh_out'][-1]['pred_face_kp2d']
                            })
        if not cfg.MODEL.USE_IWP_CAM:
            output['spec_cam'] = spec_cam

        loss_dict['loss'] = loss.detach().item()

        if self.step_count % cfg.TRAIN.LOG_FERQ == 0:
            if self.options.multiprocessing_distributed:
                for loss_name, val in loss_dict.items():
                    val = val / self.options.world_size
                    if not torch.is_tensor(val):
                        val = torch.Tensor([val]).to(self.device)
                    dist.all_reduce(val)
                    loss_dict[loss_name] = val
            if self.options.rank == 0:
                for loss_name, val in loss_dict.items():
                    self.summary_writer.add_scalar('losses/{}'.format(loss_name), val, self.step_count)

        return {'preds': output, 'losses': loss_dict}

    def fit(self):
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            self.epoch_count = epoch

            self.hf_loss_on = cfg.TRAIN.HF_LOSS_ON

            self.train(epoch)

            if self.smpl_mode:
                if (epoch + 1) % self.options.eval_every == 0:
                    self.validate()

                    if self.options.rank == 0:
                        performance = self.evaluate()
                        # log the learning rate
                        for param_group in self.optimizer.param_groups:
                            print(f'Learning rate {param_group["lr"]}')
                            self.summary_writer.add_scalar('lr/model_lr', param_group['lr'], global_step=self.epoch_count)

                        is_best = performance < self.best_performance
                        if is_best:
                            logger.info('Best performance achived, saved it!')
                            self.best_performance = performance
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, cfg.TRAIN.BATCH_SIZE, self.step_count, is_best)
            else:
                if (epoch + 1) % self.options.eval_every == 0:
                    self.validate()

                    if self.options.rank == 0:
                        performance = self.evaluate()
                        # log the learning rate
                        for param_group in self.optimizer.param_groups:
                            print(f'Learning rate {param_group["lr"]}')
                            self.summary_writer.add_scalar('lr/model_lr', param_group['lr'], global_step=self.epoch_count)

                        is_best = performance < self.best_performance
                        if is_best:
                            # logger.info('Best performance achived, saved it!')
                            self.best_performance = performance
                            self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, cfg.TRAIN.BATCH_SIZE, self.step_count, is_best)

                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, cfg.TRAIN.BATCH_SIZE, self.step_count,
                                           interval=self.options.save_every, with_optimizer=False)

        return

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        start = time.time()
        logger.info('Start Validation.')

        # initialize
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = []

        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(path_config.JOINT_REGRESSOR_H36M)).float()

        joint_mapper_h36m = constants.H36M_TO_J17 if self.options.eval_dataset == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if self.options.eval_dataset == 'mpi-inf-3dhp' else constants.J24_TO_J14

        if self.options.rank == 0:
            pbar = tqdm(desc='Eval', total=len(self.valid_ds) // cfg.TEST.BATCH_SIZE)
        for i, target in enumerate(self.valid_loader):
            if self.options.rank == 0:
                pbar.update(1)

            inp = target['img'].to(self.device, non_blocking=True)
            J_regressor_batch = J_regressor[None, :].expand(inp.shape[0], -1, -1).contiguous().to(self.device, non_blocking=True)

            # Get GT vertices and model joints
            gt_betas = target['betas'].to(self.device)
            gt_pose = target['pose'].to(self.device)

            if 'h36m' in self.options.eval_dataset:
                gt_out = self.mesh_model(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
                gt_model_joints = gt_out.joints
                gt_vertices = gt_out.vertices
                target['verts'] = gt_vertices
            elif '3dpw' in self.options.eval_dataset:
                # For 3DPW get the 14 common joints from the rendered shape
                gender = target['gender'].to(self.device)
                gt_vertices = self.smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = self.smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                target_j3d = gt_keypoints_3d - gt_pelvis

                target['target_j3d'] = target_j3d
                target['verts'] = gt_vertices

            spec_cam = {}
            input_batch = {'img_body': inp}
            if not cfg.MODEL.USE_IWP_CAM:
                spec_cam = {'bbox_scale': target['scale'].float(),
                            'bbox_center': target['center'],
                            'img_w': target['orig_shape'][:, 1],
                            'img_h': target['orig_shape'][:, 0],
                            'crop_res': self.options.img_res,
                            'cam_rotmat': target['cam_rotmat'],
                            'cam_intrinsics': target['cam_intrinsics'],
                            'kps_transf': target['kps_transf'],
                            'rot_transf': target['rot_transf'],
                            'vfov': target['cam_vfov'],
                            'crop_ratio': target['crop_ratio'],
                        }
                spec_cam = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in spec_cam.items()}
            if self.full_body_mode:
                input_batch['img_hr'] = target['img_hr']
                for part in self.part_names:
                    input_batch[f'img_{part}'] = target[f'img_{part}'].to(self.device, non_blocking=True)
                    input_batch[f'{part}_theta'] = target[f'{part}_theta'].to(self.device, non_blocking=True)
                    input_batch[f'{part}_theta_inv'] = target[f'{part}_theta_inv'].to(self.device, non_blocking=True)
                    input_batch[f'{part}_kp2d_local'] = target[f'{part}_kp2d_local'].to(self.device, non_blocking=True)

            pred_dict, _ = self.model(input_batch, J_regressor=J_regressor_batch, rw_cam=spec_cam)

            if self.options.rank == 0:
                if cfg.TRAIN.VAL_LOOP:
                    preds_list = pred_dict['mesh_out'][1:]
                else:
                    preds_list = pred_dict['mesh_out'][-1:]

                for preds in preds_list:
                    # convert to 14 keypoint format for evaluation
                    n_kp = preds['kp_3d'].shape[-2]
                    pred_j3d = preds['kp_3d'].view(-1, n_kp, 3)
                    pred_verts = preds['verts']

                    if 'h36m' in self.options.eval_dataset:
                        target_j3d = target['pose_3d'].cpu()
                        target_j3d = target_j3d[:, joint_mapper_gt, :-1]
                    elif '3dpw' in self.options.eval_dataset:
                        target_j3d = target['target_j3d'].cpu()
                    target_verts = target['verts']

                    batch_len = target['betas'].shape[0]

                    # Absolute error (MPJPE)
                    errors = torch.sqrt(((pred_j3d.cpu() - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                    S1_hat = compute_similarity_transform_batch(pred_j3d.cpu().numpy(), target_j3d.cpu().numpy())
                    S1_hat = torch.from_numpy(S1_hat).float()
                    errors_pa = torch.sqrt(((S1_hat - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

                    errors_pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

                    self.evaluation_accumulators['errors'].append(errors)
                    self.evaluation_accumulators['errors_pa'].append(errors_pa)
                    self.evaluation_accumulators['errors_pve'].append(errors_pve)

                if (i + 1) % cfg.VAL_VIS_BATCH_FREQ == 0 and self.options.rank == 0:
                    if not cfg.MODEL.USE_IWP_CAM:
                        pred_dict['spec_cam'] = spec_cam
                    self.visualize(i, target, 'valid', pred_dict)

            del pred_dict, _

            batch_time = time.time() - start

        if self.options.rank == 0:
            pbar.close()

    def evaluate(self):
        if cfg.TRAIN.VAL_LOOP:
            step = cfg.MODEL.PyMAF.N_ITER
        else:
            step = 1
        
        # num_poses = len(self.evaluation_accumulators['errors']) * cfg.TEST.BATCH_SIZE // step
        print(f'Averaging...')

        for loop_id in range(step):
            # pred_j3ds = self.evaluation_accumulators['pred_j3d'][loop_id::step]
            # pred_j3ds = np.vstack(pred_j3ds)
            # pred_j3ds = torch.from_numpy(pred_j3ds).float()
            
            # target_j3ds = self.evaluation_accumulators['target_j3d'][loop_id::step]
            # target_j3ds = np.vstack(target_j3ds)
            # target_j3ds = torch.from_numpy(target_j3ds).float()

            # # Absolute error (MPJPE)
            # errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            # S1_hat = compute_similarity_transform_batch(pred_j3ds.numpy(), target_j3ds.numpy())
            # S1_hat = torch.from_numpy(S1_hat).float()
            # errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            # pred_verts = self.evaluation_accumulators['pred_verts'][loop_id::step]
            # pred_verts = np.vstack(pred_verts)
            # pred_verts = torch.from_numpy(pred_verts).float()

            # target_verts = self.evaluation_accumulators['target_verts'][loop_id::step]
            # target_verts = np.vstack(target_verts)
            # target_verts = torch.from_numpy(target_verts).float()
            # errors_pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            errors_pve = self.evaluation_accumulators['errors_pve'][loop_id::step]
            errors = self.evaluation_accumulators['errors'][loop_id::step]
            errors_pa = self.evaluation_accumulators['errors_pa'][loop_id::step]
            errors_pve = np.concatenate(errors_pve)
            errors = np.concatenate(errors)
            errors_pa = np.concatenate(errors_pa)

            m2mm = 1000
            pve = np.mean(errors_pve) * m2mm
            mpjpe = np.mean(errors) * m2mm
            pa_mpjpe = np.mean(errors_pa) * m2mm

            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pve': pve,
            }

            loop_id -= step # to ensure the index of latest prediction is always -1
            log_str = f'Epoch {self.epoch_count}, {self.valid_ds.dataset} #samples: {len(errors)}, step {loop_id}  '
            log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
            logger.info(log_str)

            for k, v in eval_dict.items():
                self.summary_writer.add_scalar(f'eval_error/{k}_{loop_id}', v, global_step=self.epoch_count)

        # empty accumulators
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k].clear()

        # return pa_mpjpe
        return mpjpe

    @torch.no_grad()
    def visualize(self, it, target, stage, preds, losses=None):

        theta = preds['mesh_out'][-1]['theta']
        pred_verts = preds['mesh_out'][-1]['verts'].cpu().numpy() if 'verts' in preds['mesh_out'][-1] else None
        cam_pred = theta[:, :3].detach()

        if self.hand_only_mode:
            pred_verts_rh = preds['mesh_out'][-1]['verts_rh'].cpu().numpy() if 'verts_rh' in preds['mesh_out'][-1] else None
        if self.face_only_mode:
            pred_verts_fa = preds['mesh_out'][-1]['verts_fa'].cpu().numpy() if 'verts_fa' in preds['mesh_out'][-1] else None

        spec_cam = preds['spec_cam'] if not cfg.MODEL.USE_IWP_CAM else None

        cam_gt = target['gt_cam'] if 'gt_cam' in target else cam_pred

        images = target['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        imgs_np = images.cpu().numpy()

        if self.full_body_mode:
            images_hr = target['img_hr']
            images_hr = images_hr * torch.tensor([0.229, 0.224, 0.225], device=images_hr.device).reshape(1,3,1,1)
            images_hr = images_hr + torch.tensor([0.485, 0.456, 0.406], device=images_hr.device).reshape(1,3,1,1)
            imgs_hr_np = images_hr.cpu().numpy()
            # imgs_hr_np = imgs_np

            img_parts = {p: target[f'img_{p}'] for p in self.part_names}
            img_np_parts = {}
            for _part, _img in img_parts.items():
                _img = _img * torch.tensor([0.229, 0.224, 0.225], device=_img.device).reshape(1,3,1,1)
                _img = _img + torch.tensor([0.485, 0.456, 0.406], device=_img.device).reshape(1,3,1,1)
                img_np_parts[_part] = _img.cpu().numpy()
        else:
            imgs_hr_np = imgs_np

        vis_img_full = []
        vis_n = min(len(theta), 16)
        vis_img = []
        for b in range(vis_n):
            cam_sxy = cam_pred[b].cpu().numpy()
            cam_sxy_gt = cam_gt[b].cpu().numpy()

            render_imgs = []

            img_vis = np.transpose(imgs_np[b], (1, 2, 0)) * 255
            img_vis = img_vis.astype(np.uint8)
            render_imgs.append(img_vis)

            # draw keypoints
            if self.hand_only_mode or self.face_only_mode:
                for _p in self.part_names:
                    part_kps_proj_2d = 0.5 * img_vis.shape[0] * (target[f'{_p}_kp2d'][b:b+1] + 1)
                    img_part_vis = torch.from_numpy(np.transpose(img_vis / 255., (2, 0, 1))[None]).float()
                    img_part_vis = vis_batch_image_with_joints(img_part_vis, part_kps_proj_2d.detach().cpu().numpy(), np.ones((1, part_kps_proj_2d.shape[1], 1)), add_text=False)
                    render_imgs.append(img_part_vis)

                    part_kps_proj_2d = 0.5 * img_vis.shape[0] * (preds[f'pred_{_p}_kp2d'][b:b+1] + 1)
                    img_part_vis = torch.from_numpy(np.transpose(img_vis / 255., (2, 0, 1))[None]).float()
                    img_part_vis = vis_batch_image_with_joints(img_part_vis, part_kps_proj_2d.detach().cpu().numpy(), np.ones((1, part_kps_proj_2d.shape[1], 1)), add_text=False)
                    render_imgs.append(img_part_vis)
            else:
                body_kps_proj_2d = 0.5 * img_vis.shape[0] * (target['keypoints'][b:b+1, -24:] + 1)
                img_part_vis = torch.from_numpy(np.transpose(img_vis / 255., (2, 0, 1))[None]).float()
                img_part_vis = vis_batch_image_with_joints(img_part_vis, body_kps_proj_2d.detach().cpu().numpy(), np.ones((1, body_kps_proj_2d.shape[1], 1)), add_text=True)
                render_imgs.append(img_part_vis)
            
                if self.full_body_mode:
                    for _part, _img_part in img_np_parts.items():
                        img_part_vis = np.transpose(_img_part[b], (1, 2, 0)) * 255
                        img_part_vis = img_part_vis.astype(np.uint8)

                        hf_kps_proj_2d = 0.5 * img_part_vis.shape[0] * (target[f'{_part}_kp2d_local'][b:b+1] + 1)
                        if _part == 'face':
                            hf_kps_proj_2d = hf_kps_proj_2d[:, :51]
                        img_part_vis = torch.from_numpy(np.transpose(img_part_vis / 255., (2, 0, 1))[None]).float()
                        img_part_vis = vis_batch_image_with_joints(img_part_vis, hf_kps_proj_2d.detach().cpu().numpy(), np.ones((1, hf_kps_proj_2d.shape[1], 1)), add_text=False)

                        render_imgs.append(img_part_vis)

            # render GT smpl verts
            if self.hand_only_mode:
                mano_verts = target['verts_rh'][b].cpu().numpy()
                mano_verts_pred = pred_verts_rh[b] if pred_verts_rh is not None else None
                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        mano_verts,
                        img=img_vis,
                        cam=cam_sxy_gt,
                        mesh_type='mano'
                    ))
            elif self.face_only_mode:
                flame_verts = target['verts_fa'][b].cpu().numpy()
                flame_verts_pred = pred_verts_fa[b] if pred_verts_fa is not None else None
                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        flame_verts,
                        img=img_vis,
                        cam=cam_sxy_gt,
                        mesh_type='flame'
                    ))
            else:
                smpl_verts = target['smpl_verts'][b].cpu().numpy() if 'smpl_verts' in target else target['verts'][b].cpu().numpy()
                smpl_verts_pred = pred_verts[b] if pred_verts is not None else None

                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        smpl_verts,
                        img=img_vis,
                        cam=cam_sxy_gt if cam_sxy_gt[0] > 0 else cam_sxy  # cam_sxy_gt cam_sxy
                    ))
                else:
                    crop_info = {'bbox_scale': spec_cam['bbox_scale'][b:b+1],
                                'bbox_center': spec_cam['bbox_center'][b:b+1],
                                'img_w': spec_cam['img_w'][b:b+1],
                                'img_h': spec_cam['img_h'][b:b+1],
                                'opt_cam_t': preds['opt_cam_t'][b:b+1] if 'opt_cam_t' in preds else None,
                                }
                    render_imgs.append(self.renderer(
                        smpl_verts,
                        img=img_vis,
                        cam=cam_sxy_gt, # cam_pred[b:b+1],
                        focal_length=(spec_cam['cam_intrinsics'][b, 0, 0], spec_cam['cam_intrinsics'][b, 1, 1]),
                        camera_rotation=spec_cam['cam_rotmat'][b].cpu().numpy(),
                        crop_info=crop_info,
                        iwp_mode=False
                    ))

            if 'body' in self.bhf_names and (cfg.MODEL.PyMAF.AUX_SUPV_ON or cfg.MODEL.PyMAF.SEG_ON):
                dp_out = preds['dp_out'][-1]
                cm = plt.get_cmap('gist_rainbow')  # gist_rainbow  viridis
                if stage == 'train':
                    iuv_image_gt = target['iuv_image_gt'][b].detach().cpu().numpy()
                    iuv_image_gt = np.transpose(iuv_image_gt, (1, 2, 0))
                    if cfg.MODEL.PyMAF.SEG_ON:
                        bg_mask = iuv_image_gt == 0
                        iuv_image_gt = cm(iuv_image_gt[..., 0])[..., :3]
                        iuv_image_gt[bg_mask] = 0
                    iuv_image_gt *= 255
                    iuv_image_gt_resized = resize(iuv_image_gt, (img_vis.shape[0], img_vis.shape[1]), 
                                                        order=0, preserve_range=True, anti_aliasing=False)
                    render_imgs.append(iuv_image_gt_resized.astype(np.uint8))

                if cfg.MODEL.PyMAF.AUX_SUPV_ON:
                    pred_iuv_list = [dp_out['predict_u'][b:b+1], dp_out['predict_v'][b:b+1], \
                                    dp_out['predict_uv_index'][b:b+1], dp_out['predict_ann_index'][b:b+1]]
                elif cfg.MODEL.PyMAF.SEG_ON:
                    pred_iuv_list = [dp_out['predict_seg_index'][b:b+1], dp_out['predict_seg_index'][b:b+1], \
                                    dp_out['predict_seg_index'][b:b+1]]
                iuv_image_pred = iuv_map2img(*pred_iuv_list)[0].detach().cpu().numpy()
                iuv_image_pred = np.transpose(iuv_image_pred, (1, 2, 0))
                if cfg.MODEL.PyMAF.SEG_ON:
                    bg_mask = iuv_image_pred == 0
                    iuv_image_pred = cm(iuv_image_pred[..., 0])[..., :3]
                    iuv_image_pred[bg_mask] = 0
                iuv_image_pred *= 255
                iuv_image_pred_resized = resize(iuv_image_pred, (img_vis.shape[0], img_vis.shape[1]),
                                                        order=0, preserve_range=True, anti_aliasing=False)
                render_imgs.append(iuv_image_pred_resized.astype(np.uint8))
            
            if len(self.part_names) > 0 and cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                cm = plt.get_cmap('gist_rainbow')  # gist_rainbow  viridis
                for part in self.part_names:
                    if stage == 'train':
                        iuv_image_gt = target[f'{part}_pncc_gt'][b].detach().cpu().numpy()
                        iuv_image_gt = np.transpose(iuv_image_gt, (1, 2, 0))
                        iuv_image_gt *= 255
                        iuv_image_gt_resized = resize(iuv_image_gt, (img_vis.shape[0], img_vis.shape[1]), 
                                                            order=0, preserve_range=True, anti_aliasing=False)
                        render_imgs.append(iuv_image_gt_resized.astype(np.uint8))

                    iuv_image_pred = preds[f'{part}_dpout'][-1]['predict_pncc'][b].detach().cpu().numpy()
                    iuv_image_pred[iuv_image_pred<0] = 0
                    iuv_image_pred = np.transpose(iuv_image_pred, (1, 2, 0))
                    iuv_image_pred *= 255
                    iuv_image_pred_resized = resize(iuv_image_pred, (img_vis.shape[0], img_vis.shape[1]),
                                                            order=0, preserve_range=True, anti_aliasing=False)
                    render_imgs.append(iuv_image_pred_resized.astype(np.uint8))

            if self.hand_only_mode:
                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        mano_verts_pred,
                        # faces=get_model_faces('mano'),
                        img=img_vis,
                        # cam=cam_sxy_gt,
                        cam=cam_sxy,
                        mesh_type='mano'
                    ))
            elif self.face_only_mode:
                assert flame_verts_pred is not None
                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        flame_verts_pred,
                        img=img_vis,
                        # cam=cam_sxy_gt,
                        cam=cam_sxy,
                        mesh_type='flame'
                    ))
            else:
                if cfg.MODEL.USE_IWP_CAM:
                    render_imgs.append(self.renderer(
                        smpl_verts_pred,
                        img=img_vis,
                        cam=cam_sxy,
                    ))
                else:
                    render_imgs.append(self.renderer(
                        smpl_verts_pred,
                        img=img_vis,
                        cam=cam_sxy, # cam_pred[b:b+1],
                        focal_length=(spec_cam['cam_intrinsics'][b, 0, 0], spec_cam['cam_intrinsics'][b, 1, 1]),
                        camera_rotation=spec_cam['cam_rotmat'][b].cpu().numpy(),
                        crop_info=crop_info,
                        iwp_mode=False
                    ))

                if self.full_body_mode:
                    img_vis_part = np.transpose(imgs_hr_np[b], (1, 2, 0)) * 255
                    img_vis_part = img_vis_part.astype(np.uint8)
                    for p_i, part in enumerate(self.part_names):
                        p_vids = self.smpl2limb_vert_faces[part]['vids']
                        p_faces = self.smpl2limb_vert_faces[part]['faces']

                        if cfg.DEBUG_W_GT:
                            part_verts = smpl_verts[p_vids]
                        else:
                            part_verts = smpl_verts_pred[p_vids]

                        if cfg.MODEL.USE_IWP_CAM:
                            render_imgs_part = self.renderer(
                                    part_verts,
                                    faces=p_faces,
                                    img=img_vis_part,
                                    cam=cam_sxy, # cam_pred[b:b+1],
                                )
                        else:
                            render_imgs_part = self.renderer(
                                    part_verts,
                                    faces=p_faces,
                                    img=img_vis_part,
                                    cam=cam_sxy, # cam_pred[b:b+1],
                                    focal_length=(spec_cam['cam_intrinsics'][b, 0, 0], spec_cam['cam_intrinsics'][b, 1, 1]),
                                    camera_rotation=spec_cam['cam_rotmat'][b].cpu().numpy(),
                                    crop_info=crop_info,
                                    iwp_mode=False
                                )

                        theta_i = target[f'{part}_theta'][b:b+1].detach().cpu()

                        render_imgs_part_th = torch.from_numpy(render_imgs_part).permute(2, 0, 1).unsqueeze(0).float()
                        # crop_size = render_imgs_part_th.size()
                        crop_size = torch.zeros(1, 3, img_vis.shape[0], img_vis.shape[0]).size()
                        grid = F.affine_grid(theta_i.detach(), crop_size, align_corners=False)
                        crop_imgs_part = F.grid_sample(render_imgs_part_th, grid, align_corners=False)
                        crop_imgs_part = crop_imgs_part[0].permute(1, 2, 0).numpy()

                        render_imgs.append(crop_imgs_part.astype(np.uint8))

            img = np.concatenate(render_imgs, axis=1)
            img = np.transpose(img, (2, 0, 1))
            vis_img.append(img)

        vis_img_full.append(np.concatenate(vis_img, axis=1))

        vis_img_full = np.concatenate(vis_img_full, axis=-1)

        if stage == 'train':
            self.summary_writer.add_image('{}/mesh_pred'.format(stage), vis_img_full, it)
        else:
            self.summary_writer.add_image('{}/mesh_pred_{}'.format(stage, it), vis_img_full, self.epoch_count)
