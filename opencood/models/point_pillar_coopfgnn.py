# -*- coding: utf-8 -*-

# ============================================================
# CoopFGNN with the PointPillars backbone of the OpenCOOD baselines
# (F-Cooper / AttFuse: 3-level BEV backbone, 1x1 shrink to 256 channels,
# feature stride 2, anchor heads without direction head), so that only the
# fusion differs from the baselines.
#
# The fusion itself lives in the CoopFGNN repository
# (src/models/coop/coopfgnn_fuse.py, on PYTHONPATH through use_heal). This
# file only wires the batch to it:
#   - each agent keeps its own frame (dataset proj_first: false)
#   - single-agent heads give the confidence C_j that drives the selection,
#     and are supervised in each agent's frame (supervise_single: true)
#   - LiDAR points for the visibility maps are read back from the pillars
#   - world poses come from the 6-DoF lidar poses (pose_to_tfm, the
#     transform the dataset uses), reduced to SE(2)
# ============================================================

import json
import os

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.utils.transformation_utils import pose_to_tfm

from src.models.coop.bev_warp import se2_from_se3
from src.models.coop.coopfgnn_fuse import CoopFGNNFuse
from src.models.coop.visibility import BEVGrid
from src.models.coop.visibility_torch import points_from_pillars


class PointPillarCoopFGNN(nn.Module):
    def __init__(self, args):
        super(PointPillarCoopFGNN, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.shrink_conv = DownsampleConv(args['shrink_header'])
        channels = args['shrink_header']['dim'][-1]

        anchors = args['anchor_number']
        self.cls_head = nn.Conv2d(channels, anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(channels, 7 * anchors, kernel_size=1)
        self.cls_head_single = nn.Conv2d(channels, anchors, kernel_size=1)
        self.reg_head_single = nn.Conv2d(channels, 7 * anchors, kernel_size=1)

        coop = dict(args['coopfgnn'])
        stride = coop.pop('feature_stride')
        r = args['lidar_range']
        grid = BEVGrid(x_range=(r[0], r[3]), y_range=(r[1], r[4]),
                       res=args['voxel_size'][0] * stride)
        if 'tau_r_range' in coop:
            coop['tau_r_range'] = tuple(coop['tau_r_range'])
        if 'pose_noise' in coop:
            coop['pose_noise'] = tuple(coop['pose_noise'])
        if 'miscalib' in coop:
            coop['miscalib'] = tuple(coop['miscalib'])
        self.fusion_net = CoopFGNNFuse(channels, grid, **coop)

        # ============================================================
        # Inference settings of the fusion (link gate tau_q, selection
        # threshold / top-k, pose noise): 'coopfgnn_eval' in the config,
        # overridden by the COOPFGNN_EVAL environment variable (JSON), so
        # that the evaluation job can sweep them without new configs.
        # ============================================================
        eval_args = dict(args.get('coopfgnn_eval', {}))
        eval_args.update(json.loads(os.environ.get('COOPFGNN_EVAL', '{}')))
        if 'noise' in eval_args:
            eval_args['noise'] = tuple(eval_args['noise'])
        self.fusion_net.set_eval(**eval_args)

    @staticmethod
    def agent_poses(lidar_pose):
        """(N, 6) OpenCOOD poses, degrees -> (N, 3) SE(2) x, y, yaw."""
        T = se2_from_se3(pose_to_tfm(lidar_pose.float()))
        return torch.stack([T[:, 0, 2], T[:, 1, 2],
                            torch.atan2(T[:, 1, 0], T[:, 0, 0])], dim=-1)

    @staticmethod
    def agent_types(prior_encoding, record_len):
        """Infrastructure flag of each agent from the padded prior encoding."""
        flags = [prior_encoding[b, :n, 2] for b, n in enumerate(record_len)]
        return torch.cat(flags).round().long()

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        feats = self.shrink_conv(self.backbone(batch_dict['spatial_features']))

        psm_single = self.cls_head_single(feats)
        rm_single = self.reg_head_single(feats)
        conf = torch.sigmoid(psm_single).max(dim=1, keepdim=True)[0]

        points, cloud_idx = points_from_pillars(
            voxel_features, voxel_num_points, voxel_coords)
        agent_type = None
        if 'prior_encoding' in data_dict:
            agent_type = self.agent_types(data_dict['prior_encoding'],
                                          record_len)

        belief, stats = self.fusion_net(
            feats, conf, record_len, self.agent_poses(data_dict['lidar_pose']),
            points, cloud_idx, agent_type=agent_type)

        return {'cls_preds': self.cls_head(belief),
                'reg_preds': self.reg_head(belief),
                'cls_preds_single': psm_single,
                'reg_preds_single': rm_single,
                'comm_bits': stats['bits'],
                'ego_alpha': stats['ego_alpha']}
