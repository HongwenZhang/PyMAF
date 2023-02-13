# -*- coding: utf-8 -*-
# This script is borrowed and extended from https://github.com/mkocabas/VIBE/blob/master/demo.py and https://github.com/nkolot/SPIN/blob/master/demo.py
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

from copyreg import pickle
import enum
import os
import copy
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pickle as pkle

import cv2
import time
import json
import shutil
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize
from collections import OrderedDict

from core.cfgs import cfg, parse_args
from models import hmr, pymaf_net
from models.smpl import get_partial_smpl, SMPL, SMPLX
from core import path_config, constants
from datasets.inference import Inference
from utils.renderer import PyRenderer
from utils.imutils import crop
from utils.demo_utils import (
    download_url,
    convert_crop_cam_to_orig_img,
    video_to_images,
    images_to_video,
)
from utils.geometry import convert_to_full_img_cam

from openpifpaf import decoder as ppdecoder
from openpifpaf import network as ppnetwork

from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream

from os.path import join, expanduser


MIN_NUM_FRAMES = 1

def prepare_rendering_results(person_data, nframes):
    frame_results = [{} for _ in range(nframes)]
    for idx, frame_id in enumerate(person_data['frame_ids']):
        person_id = person_data['person_ids'][idx],
        frame_results[frame_id][person_id] = {
            'verts': person_data['verts'][idx],
            'smplx_verts': person_data['smplx_verts'][idx] if 'smplx_verts' in person_data else None,
            'cam': person_data['orig_cam'][idx],
            'cam_t': person_data['orig_cam_t'][idx] if 'orig_cam_t' in person_data else None,
            # 'cam': person_data['pred_cam'][idx],
        }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results


def run_demo(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.image_folder is None:
        video_file = args.vid_file

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            print(f'Donwloading YouTube video \"{video_file}\"')
            video_file = download_url(video_file, '/tmp')

            if video_file is None:
                exit('Youtube url is not valid!')

            print(f'YouTube Video has been downloaded to {video_file}...')

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
        
        output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))

        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    else:
        image_folder = args.image_folder
        num_frames = len(os.listdir(image_folder))
        img_shape = cv2.imread(osp.join(image_folder, os.listdir(image_folder)[0])).shape

        output_path = os.path.join(args.output_folder, osp.split(image_folder)[-1])

    os.makedirs(output_path, exist_ok=True)

    print(f'Input video number of frames {num_frames}')

    total_time = time.time()

    args.device = device
    args.pin_memory = True if torch.cuda.is_available() else False

    # pifpaf person detection 
    pp_det_file_path = os.path.join(output_path, 'pp_det_results.pkl')
    pp_args = copy.deepcopy(args)
    pp_args.force_complete_pose = True
    ppdecoder.configure(pp_args)
    ppnetwork.Factory.configure(pp_args)
    ppnetwork.Factory.checkpoint = pp_args.detector_checkpoint
    Predictor.configure(pp_args)
    Stream.configure(pp_args)

    Predictor.batch_size = pp_args.detector_batch_size
    if pp_args.detector_batch_size > 1:
        Predictor.long_edge = 1000
    Predictor.loader_workers = 1
    predictor = Predictor()
    if args.vid_file is not None:
        capture = Stream(args.vid_file, preprocess=predictor.preprocess)
        capture = predictor.dataset(capture)
    elif args.image_folder is not None:
        image_file_names = sorted([
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        capture = predictor.images(image_file_names)

    tracking_results = {}
    print('Running openpifpaf for person detection...')
    for preds, _, meta in tqdm(capture, total=num_frames // args.detector_batch_size):
        if args.single_person:
            preds = [preds[0]]
        for pid, ann in enumerate(preds):
            if ann.score > args.detection_threshold:
                frame_i = meta['frame_i'] - 1 if 'frame_i' in meta else meta['dataset_index']
                file_name = meta['file_name'] if 'file_name' in meta else image_folder
                person_id = file_name.split('/')[-1].split('.')[0] + '_f' + str(frame_i) + '_p' + str(pid)
                det_wb_kps = ann.data
                det_face_kps = det_wb_kps[23:91]
                tracking_results[person_id] = {
                            'frames': [frame_i],
                            # 'predictions': [ann.json_data() for ann in preds]
                            'joints2d': [det_wb_kps[:17]],
                            'joints2d_lhand': [det_wb_kps[91:112]],
                            'joints2d_rhand': [det_wb_kps[112:133]],
                            'joints2d_face': [np.concatenate([det_face_kps[17:], det_face_kps[:17]])],
                            'vis_face': [np.mean(det_face_kps[17:, -1])],
                            'vis_lhand': [np.mean(det_wb_kps[91:112, -1])],
                            'vis_rhand': [np.mean(det_wb_kps[112:133, -1])],
                        }
    pkle.dump(tracking_results, open(pp_det_file_path, 'wb'))

    bbox_scale = 1.0

    # ========= Define model ========= #
    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, is_train=False).to(device)

    # ========= Load pretrained weights ========= #
    checkpoint_paths = {'body': args.pretrained_body, 'hand': args.pretrained_hand, 'face': args.pretrained_face}
    if args.pretrained_model is not None:
        print(f'Loading pretrained weights from \"{args.pretrained_model}\"')
        checkpoint = torch.load(args.pretrained_model)

        # remove the state_dict overrode by hand and face sub-models
        for part in ['hand', 'face']:
            if checkpoint_paths[part] is not None:
                key_start_list = model.part_module_names[part].keys()
                for key in list(checkpoint['model'].keys()):
                    for key_start in key_start_list:
                        if key.startswith(key_start):
                            checkpoint['model'].pop(key)

        model.load_state_dict(checkpoint['model'], strict=True)
        print(f'loaded checkpoint: {args.pretrained_model}')

    if not all([args.pretrained_body is None, args.pretrained_hand is None, args.pretrained_face is None]):
        for part in ['body', 'hand', 'face']:
            checkpoint_path = checkpoint_paths[part]
            if checkpoint_path is not None:
                print(f'Loading checkpoint for the {part} part.')
                checkpoint = torch.load(checkpoint_path)['model']
                checkpoint_filtered = {}
                key_start_list = model.part_module_names[part].keys()
                for key in list(checkpoint.keys()):
                    for key_start in key_start_list:
                        if key.startswith(key_start):
                            checkpoint_filtered[key] = checkpoint[key]
                model.load_state_dict(checkpoint_filtered, strict=False)
                print(f'Loaded checkpoint for the {part} part.')

    model.eval()

    smpl2limb_vert_faces = get_partial_smpl(args.render_model)

    smpl2part = smpl2limb_vert_faces[args.render_part]['vids']
    part_faces = smpl2limb_vert_faces[args.render_part]['faces']

    # ========= Run pred on each person ========= #
    if args.recon_result_file:
        pred_results = joblib.load(args.recon_result_file)
        print('Loaded results from ' + args.recon_result_file)
    else:
        if args.pre_load_imgs:
            image_file_names = [
                osp.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg')
            ]
            image_file_names = sorted(image_file_names)
            image_file_names = np.array(image_file_names)
            pre_load_imgs = []
            for file_name in image_file_names:
                pre_load_imgs.append(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB))
            pre_load_imgs = np.array(pre_load_imgs)
        else:
            image_file_names = None
        print(f'Running reconstruction on each tracklet...')
        pred_time = time.time()
        pred_results = {}
        bboxes = joints2d = []
        frames= []
        if args.tracking_method == 'pose':
            wb_kps = {'joints2d_lhand': [],
                      'joints2d_rhand': [],
                      'joints2d_face': [],
                      'vis_face': [],
                      'vis_lhand': [],
                      'vis_rhand': [],
                     }
        person_id_list = list(tracking_results.keys())
        for person_id in person_id_list:
            if args.tracking_method == 'bbox':
                raise NotImplementedError
            elif args.tracking_method == 'pose':
                joints2d.extend(tracking_results[person_id]['joints2d'])
                wb_kps['joints2d_lhand'].extend(tracking_results[person_id]['joints2d_lhand'])
                wb_kps['joints2d_rhand'].extend(tracking_results[person_id]['joints2d_rhand'])
                wb_kps['joints2d_face'].extend(tracking_results[person_id]['joints2d_face'])
                wb_kps['vis_lhand'].extend(tracking_results[person_id]['vis_lhand'])
                wb_kps['vis_rhand'].extend(tracking_results[person_id]['vis_rhand'])
                wb_kps['vis_face'].extend(tracking_results[person_id]['vis_face'])

            frames.extend(tracking_results[person_id]['frames'])

        if args.pre_load_imgs:
            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
                pre_load_imgs=pre_load_imgs[frames],
                full_body=True,
                person_ids=person_id_list,
                wb_kps=wb_kps,
            )
        else:
            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
                full_body=True,
                person_ids=person_id_list,
                wb_kps=wb_kps,
            )

        bboxes = dataset.bboxes
        scales = dataset.scales
        frames = dataset.frames

        dataloader = DataLoader(dataset, batch_size=args.model_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_smplx_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], [], []
            orig_height, orig_width = [], []
            person_ids = []
            smplx_params = []

            for batch in tqdm(dataloader):
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

                person_ids.extend(batch['person_id'])
                orig_height.append(batch['orig_height'])
                orig_width.append(batch['orig_width'])

                img_body = batch['img_body']

                batch_size = img_body.shape[0]
                preds_dict, _ = model(batch)

                output = preds_dict['mesh_out'][-1]

                pred_cam.append(output['theta'][:, :3])
                pred_verts.append(output['verts'])
                pred_smplx_verts.append(output['smplx_verts'])
                pred_pose.append(output['theta'][:, 13:85])
                pred_betas.append(output['theta'][:, 3:13])
                pred_joints3d.append(output['kp_3d'])

                smplx_params.append({'shape': output['pred_shape'],
                                        'body_pose': output['rotmat'],
                                        'left_hand_pose': output['pred_lhand_rotmat'],
                                        'right_hand_pose': output['pred_rhand_rotmat'],
                                        'jaw_pose': output['pred_face_rotmat'][:, 0:1],
                                        'leye_pose': output['pred_face_rotmat'][:, 1:2],
                                        'reye_pose': output['pred_face_rotmat'][:, 2:3],
                                        'expression': output['pred_exp'],
                                    })

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_smplx_verts = torch.cat(pred_smplx_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            orig_height = torch.cat(orig_height, dim=0)
            orig_width = torch.cat(orig_width, dim=0)

            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_smplx_verts = pred_smplx_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        orig_height = orig_height.cpu().numpy()
        orig_width = orig_width.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        camera_translation = convert_to_full_img_cam(
                                pare_cam=pred_cam,
                                bbox_height=scales * 200.,
                                bbox_center=bboxes[:, :2],
                                img_w=orig_width,
                                img_h=orig_height,
                                focal_length=5000.,
                            )

        pred_results = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'orig_cam_t': camera_translation,
            'verts': pred_verts,
            'smplx_verts': pred_smplx_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
            'person_ids': person_ids,
            'smplx_params': smplx_params,
        }

        del model

        total_time = time.time() - total_time
        print(f'Total time spent for reconstruction: {total_time:.2f} seconds (including model loading time).')

        print(f'Saving output results to \"{os.path.join(output_path, "output.pkl")}\".')

        joblib.dump(pred_results, os.path.join(output_path, "output.pkl"))

    if not args.no_render:
        # ========= Render results as a single video ========= #
        renderer = PyRenderer(vis_ratio=args.render_vis_ratio)

        output_img_folder = os.path.join(output_path, osp.split(image_folder)[-1] + '_output')
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_img_folder + '/arm', exist_ok=True)

        print(f'Rendering results, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(pred_results, num_frames)

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        color_type = 'purple'

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if args.render_ratio != 1:
                img = resize(img, (int(img.shape[0] * args.render_ratio), int(img.shape[1] * args.render_ratio)), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)

            raw_img = img.copy()

            img_full = img.copy()
            img_arm = img.copy()

            if args.empty_bg:
                empty_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                if args.render_model == 'smplx':
                    frame_verts = person_data['smplx_verts']
                else:
                    frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                crop_info = {'opt_cam_t': person_data['cam_t']}

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                if args.empty_bg:
                    img, empty_img = renderer(
                            frame_verts,
                            img=[img, empty_img],
                            cam=frame_cam,
                            crop_info=crop_info,
                            color_type=color_type,
                            iwp_mode=False,
                            crop_img=False,
                            mesh_filename=mesh_filename
                        )
                else:
                    img_full = renderer(
                        frame_verts,
                        img=img_full,
                        cam=frame_cam,
                        crop_info=crop_info,
                        color_type=color_type,
                        iwp_mode=False,
                        crop_img=False,
                        mesh_type=args.render_model,
                        mesh_filename=mesh_filename
                    )

                    img_arm = renderer(
                        frame_verts[smpl2part],
                        faces=part_faces,
                        img=img_arm,
                        cam=frame_cam,
                        crop_info=crop_info,
                        color_type=color_type,
                        iwp_mode=False,
                        crop_img=False,
                        mesh_filename=mesh_filename
                    )

            if args.with_raw:
                img = np.concatenate([raw_img, img], axis=1)

            if args.empty_bg:
                img = np.concatenate([img, empty_img], axis=1)

            if args.vid_file is not None:
                imsave(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img_full)
                imsave(os.path.join(output_img_folder, 'arm', f'{frame_idx:06d}.png'), img_arm)
            else:
                imsave(os.path.join(output_img_folder, osp.split(img_fname)[-1][:-4]+'.png'), img_full)
                imsave(os.path.join(output_img_folder, 'arm', osp.split(img_fname)[-1][:-4]+'.png'), img_arm)                

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        if args.vid_file is not None:
            vid_name = osp.split(image_folder)[-1] if args.image_folder is not None else os.path.basename(video_file)
            save_name = f'{vid_name.replace(".mp4", "")}_result.mp4'
            save_name = os.path.join(output_path, save_name)

            print(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
            images_to_video(img_folder=output_img_folder + '/arm', output_vid_file=save_name.replace(".mp4", "_arm.mp4"))
            images_to_video(img_folder=image_folder, output_vid_file=save_name.replace(".mp4", "_raw.mp4"))

            # remove temporary files
            shutil.rmtree(output_img_folder)
            shutil.rmtree(image_folder)

    print('================= END =================')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

    print('initializing openpifpaf')
    ppnetwork.Factory.cli(parser)
    ppdecoder.cli(parser)
    Predictor.cli(parser)
    Stream.cli(parser)

    parser.add_argument('--img_file', type=str, default=None,
                        help='Path to a single input image')
    parser.add_argument('--vid_file', type=str, default=None,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='pose', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector_checkpoint', type=str, default='shufflenetv2k30-wholebody',
                        help='detector checkpoint for openpifpaf')
    parser.add_argument('--detector_batch_size', type=int, default=1,
                        help='batch size of person detection')
    parser.add_argument('--detection_threshold', type=float, default=0.55,
                        help='pifpaf detection score threshold.')
    parser.add_argument('--single_person', action='store_true',
                        help='only one person in the scene.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymafx_config.yaml',
                        help='config file path.')
    parser.add_argument('--pretrained_model', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--pretrained_body', default=None, help='Load a pretrained checkpoint for body at the beginning training') 
    parser.add_argument('--pretrained_hand', default=None, help='Load a pretrained checkpoint for hand at the beginning training') 
    parser.add_argument('--pretrained_face', default=None, help='Load a pretrained checkpoint for face at the beginning training') 

    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--render_vis_ratio', type=float, default=1.,
                        help='transparency ratio for rendered results')
    parser.add_argument('--render_part', type=str, default='arm',
                        help='render part mesh')
    parser.add_argument('--render_model', type=str, default='smplx', choices=['smpl', 'smplx'],
                        help='render model type')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    parser.add_argument('--use_gt', action='store_true',
                        help='use the ground truth tracking annotations.')
    parser.add_argument('--anno_file', type=str, default='',
                        help='path to tracking annotation file.')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='path to reconstruction result file.')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='pred-load input images.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()
    parse_args(args)

    print('Running demo...')
    run_demo(args)
