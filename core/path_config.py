"""
This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/path_config.py
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

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
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
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_testset.npz') 
                },
                {
                    'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_train.npz'),
                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                    'surreal': join(DATASET_NPZ_PATH, 'surreal_train.npz'),
                    '3doh50k': join(DATASET_NPZ_PATH, 'threeDOH50K_trainset.npz')
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
                    'dp_coco': COCO_ROOT,
                    '3dpw': PW3D_ROOT,
                    'upi-s1h': UPI_S1H_ROOT,
                    'surreal': SURREAL_ROOT,
                    '3doh50k': threeDOH50K_ROOT
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
FINAL_FITS_DIR = 'data/final_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
