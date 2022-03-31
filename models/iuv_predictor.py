# code brought in part from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class IUV_predict_layer(nn.Module):
    def __init__(self, feat_dim=256, final_cov_k=3, part_out_dim=25, with_uv=True):
        super().__init__()

        self.with_uv = with_uv
        if self.with_uv:
            self.predict_u = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0
            )

            self.predict_v = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0
            )

        self.predict_ann_index = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=15,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0
        )

        self.predict_uv_index = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=25,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0
        )

        self.inplanes = feat_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return_dict = {}

        predict_uv_index = self.predict_uv_index(x)
        predict_ann_index = self.predict_ann_index(x)

        return_dict['predict_uv_index'] = predict_uv_index
        return_dict['predict_ann_index'] = predict_ann_index

        if self.with_uv:
            predict_u = self.predict_u(x)
            predict_v = self.predict_v(x)
            return_dict['predict_u'] = predict_u
            return_dict['predict_v'] = predict_v
        else:
            return_dict['predict_u'] = None
            return_dict['predict_v'] = None
            # return_dict['predict_u'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)
            # return_dict['predict_v'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)

        return return_dict