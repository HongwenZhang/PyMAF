# -*- coding: utf-8 -*-
# This script is borrowed and extended from https://github.com/mkocabas/VIBE/blob/master/demo.py
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
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize

from core.cfgs import cfg, parse_args
from models import hmr, pymaf_net
from utils.renderer import OpenDRenderer, PyRenderer
from core import path_config
from datasets.inference import Inference
from utils.demo_utils import (
    download_url,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 1

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.image_folder is None:
        video_file = args.vid_file

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            print(f'Donwloading YouTube video \"{video_file}\"')
            video_file = download_youtube_clip(video_file, '/tmp')

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
    if not args.image_based:
        orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.0
    if args.use_gt:
        with open(args.anno_file) as f:
            tracking_anno = json.load(f)
        tracking_results = {}
        for tracklet in tracking_anno:
            track_id = tracklet['idx']
            frames = tracklet['frames']
            f_id = []
            bbox = []
            for f in frames:
                f_id.append(f['frame_id'])
                x_tl, y_tl = f['rect']['tl']['x'] * orig_width, f['rect']['tl']['y'] * orig_height
                x_br, y_br = f['rect']['br']['x'] * orig_width, f['rect']['br']['y'] * orig_height

                x_c, y_c = (x_br + x_tl) / 2., (y_br + y_tl) / 2.
                w, h = x_br - x_tl, y_br - y_tl
                wh_max = max(w, h)
                x_tl, y_tl = x_c - wh_max / 2., y_c - wh_max / 2.

                bbox.append(np.array([x_c, y_c, wh_max, wh_max]))
            f_id = np.array(f_id)
            bbox = np.array(bbox)
            tracking_results[track_id] = {'frames': f_id, 'bbox': bbox}
    else:
        # run multi object tracker
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define model ========= #
    if args.regressor == 'pymaf_net':
        model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)
    elif args.regressor == 'hmr':
        model = hmr(path_config.SMPL_MEAN_PARAMS).to(device)

    # ========= Load pretrained weights ========= #
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()
    print(f'Loaded pretrained weights from \"{args.checkpoint}\"')

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
            print('image_file_names', pre_load_imgs.shape)
        else:
            image_file_names = None
        print(f'Running reconstruction on each tracklet...')
        pred_time = time.time()
        pred_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            if args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            if args.pre_load_imgs:
                print('image_file_names frames', pre_load_imgs[frames].shape)
                dataset = Inference(
                    image_folder=image_folder,
                    frames=frames,
                    bboxes=bboxes,
                    joints2d=joints2d,
                    scale=bbox_scale,
                    pre_load_imgs=pre_load_imgs[frames]
                )
            else:
                dataset = Inference(
                    image_folder=image_folder,
                    frames=frames,
                    bboxes=bboxes,
                    joints2d=joints2d,
                    scale=bbox_scale,
                )

            if args.image_based:
                img_shape = cv2.imread(osp.join(image_folder, os.listdir(image_folder)[frames[0]])).shape
                orig_height, orig_width = img_shape[:2]

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=args.model_batch_size, num_workers=16)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    # batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    # batch_size, seqlen = batch.shape[:2]
                    batch_size = batch.shape[0]
                    seqlen = 1
                    preds_dict, _ = model(batch)

                    output = preds_dict['smpl_out'][-1]

                    pred_cam.append(output['theta'][:, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:, 13:85].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, 3:13].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)

                del batch

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            pred_results[person_id] = output_dict

        del model

        end = time.time()
        fps = num_frames / (end - pred_time)

        print(f'FPS: {fps:.2f}')
        total_time = time.time() - total_time
        print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        print(f'Saving output results to \"{os.path.join(output_path, "_output.pkl")}\".')

        joblib.dump(pred_results, os.path.join(output_path, "_output.pkl"))

    if not args.no_render:
        # ========= Render results as a single video ========= #
        if args.use_opendr:
            renderer = OpenDRenderer(resolution=(orig_height, orig_width))
        else:
            renderer = PyRenderer(resolution=(orig_width, orig_height))

        output_img_folder = os.path.join(output_path, osp.split(image_folder)[-1] + '_output')
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(pred_results, num_frames)

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        if args.regressor == 'hmr':
            color_type = 'pink'
        elif cfg.MODEL.PyMAF.N_ITER == 0 and cfg.MODEL.PyMAF.AUX_SUPV_ON == False:
            color_type = 'neutral'
        else:
            color_type = 'purple'

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if args.render_ratio != 1:
                img = resize(img, (int(img.shape[0] * args.render_ratio), int(img.shape[1] * args.render_ratio)), anti_aliasing=True)
                img = (img * 255).astype(np.uint8)

            raw_img = img.copy()

            # if args.sideview:
            #     side_img = np.zeros_like(img)
            
            if args.empty_bg:
                empty_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                if args.empty_bg:
                    img, empty_img = renderer(
                            frame_verts[None, :, :] if args.use_opendr else frame_verts,
                            img=[img, empty_img],
                            cam=frame_cam,
                            color_type=color_type,
                            mesh_filename=mesh_filename
                        )
                else:
                    img = renderer(
                        frame_verts[None, :, :] if args.use_opendr else frame_verts,
                        img=img,
                        cam=frame_cam,
                        color_type=color_type,
                        mesh_filename=mesh_filename
                    )

                # if args.sideview:
                #     side_img = renderer(
                #         frame_verts,
                #         img=side_img,
                #         cam=frame_cam,
                #         color_type=color_type,
                #         angle=270,
                #         axis=[0,1,0],
                #     )

            if args.with_raw:
                img = np.concatenate([raw_img, img], axis=1)

            if args.empty_bg:
                img = np.concatenate([img, empty_img], axis=1)

            # if args.sideview:
            #     img = np.concatenate([img, side_img], axis=1)

            # cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)
            if args.image_based:
                imsave(os.path.join(output_img_folder, osp.split(img_fname)[-1][:-4]+'.png'), img)
            else:
                imsave(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = osp.split(image_folder)[-1] if args.image_folder is not None else os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_result.mp4'
        save_name = os.path.join(output_path, save_name)
        if not args.image_based:
            print(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        # shutil.rmtree(output_img_folder)

    # shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')
    parser.add_argument('--regressor', type=str, default='pymaf_net',
                        help='Name of the SMPL regressor.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
                        help='config file path for daspin.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--use_opendr', action='store_true',
                        help='Use opendr to render the predicted SMPL instead of pyrender')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    # parser.add_argument('--sideview', action='store_true',
    #                     help='render meshes from alternate viewpoint.')
    parser.add_argument('--image_based', action='store_true',
                        help='image based reconstruction.')
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

    main(args)
