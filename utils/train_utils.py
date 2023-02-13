import os
import sys
import json
import random
import string
from datetime import datetime
from core.cfgs import cfg

import logging
logger = logging.getLogger(__name__)

def print_args():
    message = ' '.join(sys.argv)
    return message

def prepare_env(args):
    letters = string.ascii_letters
    timestamp = datetime.now().strftime('%b%d-%H-%M-%S-') + ''.join(random.choice(letters) for i in range(3))

    if cfg.DEBUG:
        sub_dir, log_name = '', 'log_debug'
    else:
        if cfg.TRAIN.BHF_MODE == 'full_body':
            sub_dir = 'pymaf-x_' + cfg.MODEL.PyMAF.BACKBONE
        elif cfg.TRAIN.BHF_MODE in ['hand_only', 'face_only']:
            sub_dir = 'pymaf_' + cfg.MODEL.PyMAF.HF_BACKBONE
        else:
            sub_dir = 'pymaf_' + cfg.MODEL.PyMAF.BACKBONE
        
        if args.train_data not in ['h36m', 'agora', 'agora_val']:
            sub_dir += '_mix'

        # [backbone]_[pretrained datasets]_[aux.supv.]_[loop iteration]_[time]_[random number]
        log_name = sub_dir
        if not cfg.MODEL.PyMAF.MAF_ON:
            log_name += '_baseline'

        if cfg.MODEL.MESH_MODEL in ['smpl', 'smplx']:
            log_name += '_as_' if cfg.MODEL.PyMAF.AUX_SUPV_ON else '_'
        elif cfg.MODEL.MESH_MODEL in ['mano', 'flame']:
            log_name += '_hfas_' if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON else '_'
        log_name += 'lp' + str(cfg.MODEL.PyMAF.N_ITER)

        if cfg.MODEL.PyMAF.N_ITER > 0:
            log_name += '_mlp'
            log_name += '-'.join(str(i) for i in cfg.MODEL.PyMAF.MLP_DIM)
        
        if cfg.MODEL.PyMAF.GRID_FEAT:
            log_name += '_grid'

        if cfg.MODEL.PyMAF.ADD_GRID:
            log_name += '_ag'
        
        if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
            # grid maf? attention starts
            log_name += '_gmats{}'.format(cfg.MODEL.PyMAF.GRID_ALIGN.ATT_STARTS)
        
        if cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC:
            log_name += '_fc'

        if cfg.MODEL.PyMAF.SEG_ON:
            if cfg.MODEL.PyMAF.SEG_LAST:
                log_name += '_sels'
            else:
                log_name += '_seg'
        
        if not cfg.MODEL.USE_IWP_CAM:
            log_name += '_fcam'

        log_name += '_' + timestamp

    log_dir = os.path.join(args.log_dir, sub_dir, log_name)

    if not args.resume:
        args.log_name = log_name
        args.log_dir = log_dir
    else:
        args.log_name = args.log_dir.split('/')[-1]

    logger.info('log name: {}'.format(args.log_dir))

    args.summary_dir = os.path.join(args.log_dir, 'tb_summary')
    args.checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)
    if not os.path.exists(args.checkpoint_dir):
        if args.resume:
            raise ValueError('Experiment are set to resume mode, but checkpoint directory does not exist.')
        os.makedirs(args.checkpoint_dir)

    if not args.resume:
        with open(os.path.join(args.log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        with open(os.path.join(args.log_dir, 'cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
    else:
        with open(os.path.join(args.log_dir, "args_resume.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        with open(os.path.join(args.log_dir, 'cfg_resume.yaml'), 'w') as f:
            f.write(cfg.dump())
