# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# ============================================================
# F-Cooper with a PointPillars backbone, as in OpenCOOD
# (opencood/models/point_pillar_fcooper.py). HEAL dropped this model and
# runs F-Cooper through heter_model_baseline (3x3 shrink header, direction
# head, features warped to the ego frame). This port keeps the OpenCOOD
# design: the points of every agent are projected to the ego frame by the
# dataset (proj_first: true), so the per-agent feature maps are already
# aligned and fused by an element-wise max.
#
# Changes from OpenCOOD: the BEV backbone takes and returns the feature
# tensor in this codebase, and the outputs use the cls_preds / reg_preds
# keys of the QuantV2X post-processor and loss.
# ============================================================

import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion


class PointPillarFCooper(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarFCooper, self).__init__()

        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(out_channel,
                                                    args['compression'])

        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """
        modules = [self.pillar_vfe, self.scatter, self.backbone,
                   self.cls_head, self.reg_head]
        if self.compression:
            modules.append(self.naive_compressor)
        if self.shrink_flag:
            modules.append(self.shrink_conv)
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        spatial_features_2d = self.backbone(batch_dict['spatial_features'])

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        return {'cls_preds': psm, 'reg_preds': rm}
