"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/eval.py
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import os
import cv2
import torch
import argparse
import scipy.io
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import BaseDataset
from models import hmr, SMPL, pymaf_net
from core import constants, path_config
from core.cfgs import parse_args
from utils.imutils import uncrop
from utils.uv_vis import vis_smpl_iuv
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer # used by lsp

import logging
logger = logging.getLogger(__name__)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', choices=['h36m-p1', 'h36m-p2', 'h36m-p2-mosh', 'lsp', '3dpw', 'mpi-inf-3dhp', '3doh50k'],
                                            default='h36m-p2', help='Choose evaluation dataset')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--regressor', type=str, choices=['hmr', 'pymaf_net'], default='pymaf_net', help='Name of the SMPL regressor.')
parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml', help='config file path for PyMAF.')
parser.add_argument('--misc', default=None, type=str, nargs="*", help='other parameters')

parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--eval_pve', default=False, action='store_true', help='evaluate PVE')
parser.add_argument('--vis_demo', default=False, action='store_true', help='result visualization')
parser.add_argument('--ratio', default=1, type=int, help='image size ration for visualization')


def run_evaluation(model, dataset):
    """Run evaluation on the datasets and metrics we report in the paper. """
    
    shuffle = args.shuffle
    log_freq = args.log_freq
    batch_size = args.batch_size
    dataset_name = args.dataset
    result_file = args.result_file
    num_workers = args.num_workers
    device = torch.device('cuda') if torch.cuda.is_available() \
                                else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(path_config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(path_config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(path_config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(path_config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))
    action_idxes = {}
    idx_counter = 0
    # for each action
    act_PVE = {}
    act_MPJPE = {}
    act_paMPJPE = {}

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == 'h36m-p2-mosh' \
       or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == '3doh50k':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = path_config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    cnt = 0
    results_dict = {'id': [], 'pred': [], 'pred_pa': [], 'gt': []}
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_smpl_out = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_vertices_nt = gt_smpl_out.vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]

        if save_results:
            s_id = np.array([int(item.split('/')[-3][-1]) for item in batch['imgname']]) * 10000
            s_id += np.array([int(item.split('/')[-1][4:-4]) for item in batch['imgname']])
            results_dict['id'].append(s_id)

        if dataset_name == 'h36m-p2':
            action = [im_path.split('/')[-1].split('.')[0].split('_')[1] for im_path in batch['imgname']]
            for act_i in range(len(action)):

                if action[act_i] in action_idxes:
                    action_idxes[action[act_i]].append(idx_counter + act_i)
                else:
                    action_idxes[action[act_i]] = [idx_counter + act_i]
            idx_counter += len(action)

        with torch.no_grad():
            if args.regressor == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
                # torch.Size([32, 24, 3, 3]) torch.Size([32, 10]) torch.Size([32, 3])
            elif args.regressor == 'pymaf_net':
                preds_dict, _ = model(images)
                pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
                pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
                pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name or '3doh50k' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            if '3dpw' in dataset_name:
                per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            else:
                per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices_nt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error, pred_keypoints_3d_pa = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            if save_results:
                results_dict['gt'].append(gt_keypoints_3d.cpu().numpy())
                results_dict['pred'].append(pred_keypoints_3d.cpu().numpy())
                results_dict['pred_pa'].append(pred_keypoints_3d_pa)

        if args.vis_demo:
            imgnames = [i_n.split('/')[-1] for i_n in batch['imgname']]

            if args.regressor == 'hmr':
                iuv_pred = None

            images_vis = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images_vis = images_vis + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            vis_smpl_iuv(images_vis.cpu().numpy(), pred_camera.cpu().numpy(), pred_output.vertices.cpu().numpy(),
                        smpl_neutral.faces, iuv_pred, 100 * per_vertex_error, imgnames,
                        os.path.join('./notebooks/output/demo_results', dataset_name, args.checkpoint.split('/')[-3]), args)

        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, pred_camera)
        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        for k in results_dict.keys():
            results_dict[k] = np.concatenate(results_dict[k])
            print(k, results_dict[k].shape)
        
        scipy.io.savemat(result_file +'.mat', results_dict)

    # Print final results during evaluation
    print('*** Final Results ***')
    try:
        print(os.path.split(args.checkpoint)[-3:], args.dataset)
    except:
        pass
    if eval_pose:
        print('PVE: ' + str(1000 * pve.mean()))
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

    if dataset_name == 'h36m-p2':
        print('Note: PVE is not available for h36m-p2. To evaluate PVE, use h36m-p2-mosh instead.')
        for act in action_idxes:
            act_idx = action_idxes[act]
            act_pve = [pve[i] for i in act_idx]
            act_errors = [mpjpe[i] for i in act_idx]
            act_errors_pa = [recon_err[i] for i in act_idx]

            act_errors_mean = np.mean(np.array(act_errors)) * 1000.
            act_errors_pa_mean = np.mean(np.array(act_errors_pa)) * 1000.
            act_pve_mean = np.mean(np.array(act_pve)) * 1000.
            act_MPJPE[act] = act_errors_mean
            act_paMPJPE[act] = act_errors_pa_mean
            act_PVE[act] = act_pve_mean

        act_err_info = ['action err']
        act_row = [str(act_paMPJPE[act]) for act in action_idxes] + [act for act in action_idxes]
        act_err_info.extend(act_row)
        print(act_err_info)
    else:
        act_row = None


if __name__ == '__main__':
    args = parser.parse_args()
    parse_args(args)

    if args.regressor == 'pymaf_net':
        model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=False)
    elif args.regressor == 'hmr':
        model = hmr(path_config.SMPL_MEAN_PARAMS)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        strict = args.regressor != 'hmr'
        model.load_state_dict(checkpoint['model'], strict=strict)

    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(args, args.dataset, is_train=False)

    # Run evaluation
    run_evaluation(model, dataset)
