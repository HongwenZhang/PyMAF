"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/path_config.py
path configuration
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join, expanduser

H36M_ROOT = join(expanduser('~'), 'Datasets/human/h36m/c2f_vol')
LSP_ROOT = join(expanduser('~'), 'Datasets/human/LSP/lsp_dataset_small')
LSP_ORIGINAL_ROOT = join(expanduser('~'), 'Datasets/human/LSP/lsp_dataset_original')
LSPET_ROOT = join(expanduser('~'), 'Datasets/human/LSP/hr_lspet/hr-lspet')
MPII_ROOT = join(expanduser('~'), 'Datasets/human/mpii')
COCO_ROOT = join(expanduser('~'), 'Datasets/coco')
MPI_INF_3DHP_ROOT = join(expanduser('~'), 'Datasets/human/MPI_INF_3DHP/mpi_inf_3dhp/mpi_inf_3dhp_train_set')
PW3D_ROOT = join(expanduser('~'), 'Datasets/human/3DPW')
UPI_S1H_ROOT = join(expanduser('~'), 'Datasets/human/upi-s1h')
SURREAL_ROOT = join(expanduser('~'), 'Datasets/human/SURREAL/data')
threeDOH50K_ROOT = join(expanduser('~'), 'Datasets/human/3DOH50K')
AGORA_ROOT = join(expanduser('~'), 'Datasets/human/AGORA')
EHF_ROOT = join(expanduser('~'), 'Datasets/human/EHF')

FREIHAND_ROOT = join(expanduser('~'), 'Datasets/hand/FreiHAND')
INTERHAND_ROOT = join(expanduser('~'), 'Datasets/hand/interhand2.6m_mul')

FFHQ_ROOT = join(expanduser('~'), 'Datasets/face/FFHQ/thumbnails128x128')
VGGFACE2_ROOT = join(expanduser('~'), 'Datasets/face/VGG-Face2/data')
STIRLING_ROOT = join(expanduser('~'), 'Datasets/face/Stirling/Subset_2D_FG2018')
NOW_ROOT = join(expanduser('~'), 'Datasets/face/NoW_Dataset/iphone_pictures')

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files, spin fits
DATASET_FILES = [{
                    'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_newpath.npz'),
                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_newpath.npz'),
                    'h36m-p2-mosh': join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
                    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                    # 'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_minival.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_val.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_testset.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_test.npz'),
                },
                {
                    'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_train.npz'),
                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_train.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_trainset.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_train.npz') 
                }
]

# Path to test/train npz files, eft fits
EFT_DATASET_FILES = [{
                    'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_newpath.npz'),
                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_newpath.npz'),
                    'h36m-p2-mosh': join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
                    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                    # 'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_minival.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_val.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_testset.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_test.npz'),
                    'freihand': join(DATASET_NPZ_PATH, 'freihand_evaluation.npz'),
                },
                {
                    'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train_eft.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train_eft.npz'),
                    'coco-full': join(DATASET_NPZ_PATH, 'coco-full_train_eft.npz'),
                    'coco-hf': join(DATASET_NPZ_PATH, 'coco-hf_train_eft.npz'),
                    'coco-hf-x': join(DATASET_NPZ_PATH, 'coco-hf_train_smplx.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_train.npz'),
                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train_eft.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_train.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_trainset.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),
                    'freihand': join(DATASET_NPZ_PATH, 'freihand_training.npz'),
                }
]
# HAS_EFT_DATA = ['coco']

# Path to test/train npz files, smplx fits
SMPLX_DATASET_FILES = [{
                    'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_newpath.npz'),
                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_newpath.npz'),
                    'h36m-p2-mosh': join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_test.npz'),
                    'freihand': join(DATASET_NPZ_PATH, 'freihand_evaluation.npz'),
                    'ffhq': join(DATASET_NPZ_PATH, 'ffhq.npz'),
                    'ehf': join(DATASET_NPZ_PATH, 'ehf_test.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val_wb.npz'),
                    'stirling': join(DATASET_NPZ_PATH, 'stirling.npz'),
                    'now_validation': join(DATASET_NPZ_PATH, 'now_validation.npz'),
                    'now_test': join(DATASET_NPZ_PATH, 'now_test.npz'),
                    'vggface2': join(DATASET_NPZ_PATH, 'vggface2_test.npz'),
                },
                {
                    'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train_expose.npz'),
                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train_expose.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train_eft.npz'),
                    'coco-full': join(DATASET_NPZ_PATH, 'coco-full_train_eft.npz'),
                    'coco-smplx': join(DATASET_NPZ_PATH, 'coco_smplx_train.npz'),
                    'coco-hf': join(DATASET_NPZ_PATH, 'coco-hf_train_eft.npz'),
                    'coco-hf-x': join(DATASET_NPZ_PATH, 'coco-hf_train_smplx.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_train.npz'),
                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train_expose.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_train.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_trainset.npz'),
                    'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),
                    'ffhq': join(DATASET_NPZ_PATH, 'ffhq.npz'),
                    'freihand': join(DATASET_NPZ_PATH, 'freihand_training.npz'),
                    'interhand': join(DATASET_NPZ_PATH, 'interhand-full_train.npz'),
                    'vggface2': join(DATASET_NPZ_PATH, 'vggface2.npz'),
                }
]

DATASET_FOLDERS = {
                    'h36m': H36M_ROOT,
                    'h36m-p1': H36M_ROOT,
                    'h36m-p2': H36M_ROOT,
                    'h36m-p2-mosh': H36M_ROOT,
                    'lsp-orig': LSP_ORIGINAL_ROOT,
                    'lsp': LSP_ROOT,
                    'lspet': LSPET_ROOT,
                    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                    'mpii': MPII_ROOT,
                    'coco': COCO_ROOT,
                    'coco-full': COCO_ROOT,
                    'coco-hf': COCO_ROOT,
                    'coco-hf-x': COCO_ROOT,
                    'coco-smplx': COCO_ROOT,
                    'dp_coco': COCO_ROOT,
                    '3dpw': PW3D_ROOT,
                    'upi-s1h': UPI_S1H_ROOT,
                    'surreal': SURREAL_ROOT,
                    '3doh50k': threeDOH50K_ROOT,
                    'agora': AGORA_ROOT,
                    'freihand': FREIHAND_ROOT,
                    'interhand': INTERHAND_ROOT,
                    'ffhq': FFHQ_ROOT,
                    'ehf': EHF_ROOT,
                    'vggface2': VGGFACE2_ROOT,
                    'stirling': STIRLING_ROOT,
                    'now_validation': NOW_ROOT,
                    'now_test': NOW_ROOT,
                }

DATASET_NAMES = DATASET_FOLDERS.keys()
HAND_DATASET_NAMES = ['freihand', 'interhand']
FACE_DATASET_NAMES = ['ffhq', 'vggface2', 'stirling', 'now_validation', 'now_test']

CAM_PARAM_FOLDERS = {
                    'agora': join(AGORA_ROOT, 'spec_cam'),
                    # 'agora_val': join(AGORA_ROOT, 'spec_cam')
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
FINAL_FITS_DIR = 'data/final_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
