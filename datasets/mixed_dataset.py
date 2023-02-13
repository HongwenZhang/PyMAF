"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/mixed_dataset.py
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
import core.path_config as path_config
from core.cfgs import cfg

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        coco_set_name = 'coco-full' if cfg.TRAIN.USE_EFT else 'coco'
        if options.train_data in path_config.DATASET_NAMES:
            print('Training Data: {} datasets.'.format(options.train_data))
            self.dataset_list = [options.train_data]
        elif options.train_data == 'agora_coco':
            print('Training Data: AGORA and COCO datasets.')
            self.dataset_list = ['agora_val', coco_set_name]
        elif options.train_data == 'h36m_coco':
            print('Training Data: Human3.6M, COCO.')
            self.dataset_list = ['h36m', coco_set_name]
        elif options.train_data == 'h36m_3dpw':
            print('Training Data: Human3.6M, 3DPW.')
            self.dataset_list = ['h36m', '3dpw']
        elif options.train_data == 'expose_coco':
            print('Training Data: ExPose, COCO.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', coco_set_name]
        elif options.train_data == 'expose_coco_hf':
            print('Training Data: ExPose, COCO.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco-hf']
        elif options.train_data == 'inter_freihand':
            print('Training Data: interhand and FreiHand datasets.')
            self.dataset_list = ['interhand', 'freihand']
        elif options.train_data == 'inter_freihand_itw':
            print('Training Data: interhand, FreiHand, and other in-the-wild datasets.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'interhand', 'freihand', 'coco-hf']
        elif options.train_data == 'expose_freihand':
            print('Training Data: Expose and FreiHand datasets.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'freihand']
        elif options.train_data == 'expose_hand':
            print('Training Data: Expose and FreiHand, interhand datasets.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'freihand', 'interhand']
        elif options.train_data == 'expose':
            print('Training Data: Expose datasets.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet']
        elif options.train_data == 'coco_itw':
            print('Training Data: COCO and other in-the-wild datasets.')
            self.dataset_list = ['lsp-orig', 'mpii', 'lspet', coco_set_name]
        elif options.train_data == 'h36m_coco_itw':
            print('Training Data: Human3.6M, COCO, and other in-the-wild datasets.')
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', coco_set_name, 'mpi-inf-3dhp']
            # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'lspet', 'lsp-orig']
        elif options.train_data == '3dpw_coco_itw':
            print('Training Data: 3DPW, COCO, and other in-the-wild datasets.')
            self.dataset_list = ['3dpw', 'lsp-orig', 'mpii', 'lspet', coco_set_name, 'mpi-inf-3dhp']
        elif options.train_data == 'h36m_3dpw_coco_itw':
            print('Training Data: Human3.6M, 3DPW, COCO, and other in-the-wild datasets.')
            self.dataset_list = ['h36m', '3dpw', 'lsp-orig', 'mpii', 'lspet', coco_set_name, 'mpi-inf-3dhp']
        elif options.train_data == '3dpw_coco':
            print('Training Data: 3DPW, and COCO.')
            self.dataset_list = ['3dpw', coco_set_name]

        self.dataset_dict = {dn: idx for idx, dn in enumerate(self.dataset_list)}
        self.num_datasets = len(self.dataset_list)

        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        self.dataset_length = {self.dataset_list[idx]: len(ds) for idx, ds in enumerate(self.datasets)}
        total_length = sum([len(ds) for ds in self.datasets])
        self.length = max([len(ds) for ds in self.datasets])
        if options.train_data in path_config.DATASET_NAMES:
            self.partition = [1.0]
        elif options.train_data == 'agora_coco':
            # agora + COCO
            length_itw = sum([len(ds) for ds in self.datasets[1:]])
            self.partition = [0.5, 0.5*len(self.datasets[1])/length_itw]
        elif options.train_data == 'h36m_coco':
            # h36m + COCO
            length_itw = sum([len(ds) for ds in self.datasets[1:]])
            self.partition = [0.5, 0.5*len(self.datasets[1])/length_itw]
        elif options.train_data == '3dpw_coco':
            # 3DPW + COCO
            length_itw = sum([len(ds) for ds in self.datasets[1:]])
            self.partition = [cfg.TRAIN.PW3D_RATIO, (1. - cfg.TRAIN.PW3D_RATIO)*len(self.datasets[1])/length_itw]
        elif options.train_data == 'coco_itw':
            length_itw = sum([len(ds) for ds in self.datasets[:-1]])
            self.partition = [0.3 * len(self.datasets[i])/length_itw for i in range(len(self.datasets[:-1]))] + [0.7]
        elif options.train_data == 'h36m_coco_itw':
            """
            Data distribution inside each batch:
            50% H36M - 30% ITW - 20% MPI-INF
            """
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            # self.partition = [
            #                     .3,
            #                     .6*len(self.datasets[1])/length_itw,
            #                     .6*len(self.datasets[2])/length_itw,
            #                     .6*len(self.datasets[3])/length_itw,
            #                     .6*len(self.datasets[4])/length_itw,
            #                     0.1]
            self.partition = [
                                .5,
                                .3*len(self.datasets[1])/length_itw,
                                .3*len(self.datasets[2])/length_itw,
                                .3*len(self.datasets[3])/length_itw,
                                .3*len(self.datasets[4])/length_itw,
                                0.2]
            print('h36m length_itw', length_itw, self.partition )
        elif options.train_data == '3dpw_coco_itw':
            """
            Data distribution inside each batch:
            50% 3DPW - 30% ITW - 20% MPI-INF
            """
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.partition = [
                                .5,
                                .3*len(self.datasets[1])/length_itw,
                                .3*len(self.datasets[2])/length_itw,
                                .3*len(self.datasets[3])/length_itw,
                                .3*len(self.datasets[4])/length_itw,
                                0.2]
            print('3dpw length_itw', length_itw, self.partition )
        elif options.train_data == 'h36m_3dpw_coco_itw':
            """
            Data distribution inside each batch:
            40% H36M - 10% 3DPW - 30% ITW - 20% MPI-INF
            """
            length_itw = sum([len(ds) for ds in self.datasets[2:-1]])
            self.partition = [0.5 - cfg.TRAIN.PW3D_RATIO,
                              cfg.TRAIN.PW3D_RATIO,
                              .3*len(self.datasets[2])/length_itw,
                              .3*len(self.datasets[3])/length_itw,
                              .3*len(self.datasets[4])/length_itw,
                              .3*len(self.datasets[5])/length_itw,
                              0.2]
        elif options.train_data == 'inter_freihand':
            self.partition = [0.7, 0.3]
        elif options.train_data == 'inter_freihand_itw':
            length_itw = sum([len(ds) for ds in self.datasets[:3]])
            self.partition = [0.3 * len(self.datasets[i])/length_itw for i in range(len(self.datasets[:3]))] + [0.3, 0.1] + [0.3]
        elif options.train_data == 'expose_freihand':
            length_itw = sum([len(ds) for ds in self.datasets[:-1]])
            self.partition = [0.3 * len(self.datasets[i])/length_itw for i in range(len(self.datasets[:-1]))] + [0.7]
        elif options.train_data == 'expose_hand':
            length_itw = sum([len(ds) for ds in self.datasets[:-2]])
            self.partition = [0.3 * len(self.datasets[i])/length_itw for i in range(len(self.datasets[:-2]))] + [0.2, 0.5]
        elif options.train_data == 'expose':
            length_itw = sum([len(ds) for ds in self.datasets])
            self.partition = [len(self.datasets[i])/length_itw for i in range(len(self.datasets))]
        else:
            length_itw = sum([len(ds) for ds in self.datasets[:-1]])
            self.partition = [0.7 * len(self.datasets[i])/length_itw for i in range(len(self.datasets[:-1]))] + [0.3]
        
        print('partition:', self.partition)

        self.partition = np.array(self.partition).cumsum()

        print('partition sum', self.partition[-1])

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.num_datasets):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
