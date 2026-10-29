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
#   - with the temporal dataset (v2): latency of each sender from
#     prior_encoding, and the ambiguity A_i^{t-1} of the ego computed from
#     its previous frame
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
from src.models.coop.coopfgnn_fuse import CoopFGNNFuse, MultiScaleCoopFGNNFuse
from src.models.coop.edge_encoder import relative_pose
from src.models.coop.query import ambiguity
from src.models.coop.visibility import BEVGrid
from src.models.coop.visibility_torch import points_from_pillars


class RefineBlock(nn.Module):
    """
    Bottleneck residual block (1x1 reduce, dilated 3x3, 1x1 expand). The
    expand conv starts at zero, so the block starts as the identity.
    """
    def __init__(self, channels, hidden, dilation):
        super(RefineBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.01), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.01), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


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
        # optional direction classifier (dir_args), on both heads
        self.use_dir = 'dir_args' in args
        if self.use_dir:
            bins = args['dir_args']['num_bins']
            self.dir_head = nn.Conv2d(channels, bins * anchors, kernel_size=1)
            self.dir_head_single = nn.Conv2d(channels, bins * anchors,
                                             kernel_size=1)

        # ============================================================
        # refine: residual blocks on the fused ego map before the heads,
        # for wider context after the windowed fusion (no extra message);
        # the dilations give a receptive field of 1 + 2 * sum(dilations)
        # cells
        # ============================================================
        refine = args.get('refine')
        self.refine = nn.Sequential(*[
            RefineBlock(channels, refine.get('hidden', channels // 2), d)
            for d in refine['dilations']]) if refine else None

        coop = dict(args['coopfgnn'])
        self.aligned = coop.get('aligned', False)
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
        # ============================================================
        # multiscale: fuse the three backbone scales (0.8 / 1.6 / 3.2 m with
        # stride 2 / 4 / 8), then decode them to the heads, as CoAlign;
        # otherwise fuse the decoded 0.8 m map only (v1)
        # ============================================================
        self.multiscale = coop.pop('multiscale', False)
        if self.multiscale:
            self.fusion_net = MultiScaleCoopFGNNFuse(
                list(args['base_bev_backbone']['num_filters']), grid, **coop)
        else:
            self.fusion_net = CoopFGNNFuse(channels, grid, **coop)

        # ============================================================
        # Inference settings of the fusion (link gate tau_q, selection
        # threshold / top-k, pose noise): 'coopfgnn_eval' in the config,
        # overridden by the COOPFGNN_EVAL environment variable (JSON), so
        # that the evaluation job can sweep them without new configs.
        # ============================================================
        eval_args = dict(args.get('coopfgnn_eval', {}))
        eval_args.update(json.loads(os.environ.get('COOPFGNN_EVAL') or '{}'))
        # ============================================================
        # late: inference-time late fusion of the neighbours' single-agent
        # predictions with the fused ones (hybrid, as CoSDH intermediatelate).
        # Only in aligned mode, where every agent predicts on the ego anchors.
        # late_min: a neighbour prediction is only considered above this
        # confidence, so that the max over heads does not lift negatives.
        # ============================================================
        self.late = eval_args.pop('late', False)
        self.late_min = eval_args.pop('late_min', 0.0)
        if self.late and not self.aligned:
            raise ValueError('late fusion needs aligned mode (proj_first)')
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
    def per_agent(prior_encoding, record_len, column):
        """(N,) column of the padded (B, max_cav, 3) prior encoding."""
        return torch.cat([prior_encoding[b, :n, column]
                          for b, n in enumerate(record_len)])

    @staticmethod
    def late_fusion(record_len, psm, psm_single, maps, late_min=0.0):
        """
        Per anchor, keep the most confident of the fused prediction and the
        neighbours' single-agent predictions (all on the ego anchors), with
        its other outputs; thresholding and NMS then run as usual.

        maps: list of (fused, single, k) for the outputs with k channels per
        anchor (regression k=7, direction k=num_bins), grouped per anchor.
        late_min: minimum confidence of a neighbour prediction.
        """
        B, A, H, W = psm.shape
        out_psm, out_maps = [], [[] for _ in maps]
        start = 0
        for b, n in enumerate(record_len.tolist()):
            nbr = psm_single[start + 1:start + n]
            if late_min > 0:
                nbr = nbr.masked_fill(torch.sigmoid(nbr) < late_min,
                                      float('-inf'))
            cls = torch.cat([psm[b:b + 1], nbr])
            best = cls.argmax(dim=0, keepdim=True)
            out_psm.append(cls.gather(0, best)[0])
            for out, (fused, single, k) in zip(out_maps, maps):
                m = torch.cat([fused[b:b + 1], single[start + 1:start + n]])
                m = m.view(-1, A, k, H, W)
                out.append(m.gather(
                    0, best.unsqueeze(2).expand(1, A, k, H, W))[0].view(
                        k * A, H, W))
            start += n
        return torch.stack(out_psm), [torch.stack(o) for o in out_maps]

    def encode(self, lidar, record_len, scales=False):
        """
        Pillars -> (N, C, H, W) BEV features, one map per agent; with
        scales=True also the list of the backbone scales before decoding.
        """
        batch_dict = {'voxel_features': lidar['voxel_features'],
                      'voxel_coords': lidar['voxel_coords'],
                      'voxel_num_points': lidar['voxel_num_points'],
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        if not scales:
            return self.shrink_conv(
                self.backbone(batch_dict['spatial_features']))
        levels = self.backbone.get_multiscale_feature(
            batch_dict['spatial_features'])
        decoded = self.shrink_conv(
            self.backbone.decode_multiscale_feature(levels))
        return decoded, levels

    def previous_ambiguity(self, data_dict, record_len):
        """
        A_i^{t-1} of each ego and the ego motion t-1 -> t (temporal dataset).

        The ego's previous frame goes through the same backbone and
        single-agent head without gradient: it is what the ego computed
        and sent one frame earlier. Scenes without a previous frame get a
        zero map, so their query map is U_hat alone.
        """
        if 'prev_ego_lidar' not in data_dict:
            return None, None
        B = record_len.shape[0]
        with torch.no_grad():
            feats = self.encode(data_dict['prev_ego_lidar'],
                                torch.ones_like(record_len))
            conf = torch.sigmoid(self.cls_head_single(feats)).max(
                dim=1, keepdim=True)[0]
            amb = ambiguity(conf) * \
                data_dict['prev_ego_valid'].to(conf).view(B, 1, 1, 1)
            ego = torch.cumsum(record_len, 0) - record_len
            T_t_prev = relative_pose(
                self.agent_poses(data_dict['lidar_pose_clean'][ego]),
                self.agent_poses(data_dict['prev_ego_pose']))
        return amb, T_t_prev

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        if self.multiscale:
            feats, levels = self.encode(data_dict['processed_lidar'],
                                        record_len, scales=True)
        else:
            feats = self.encode(data_dict['processed_lidar'], record_len)

        psm_single = self.cls_head_single(feats)
        rm_single = self.reg_head_single(feats)
        conf = torch.sigmoid(psm_single).max(dim=1, keepdim=True)[0]

        points, cloud_idx = points_from_pillars(
            voxel_features, voxel_num_points, voxel_coords)
        agent_type, dt = None, None
        if 'prior_encoding' in data_dict:
            prior = data_dict['prior_encoding']
            dt = self.per_agent(prior, record_len, 1).to(feats)
            agent_type = self.per_agent(prior, record_len, 2).round().long()
        amb_prev, T_t_prev = self.previous_ambiguity(data_dict, record_len)

        belief, stats = self.fusion_net(
            levels if self.multiscale else feats, conf, record_len,
            self.agent_poses(data_dict['lidar_pose']), points, cloud_idx,
            dt=dt, agent_type=agent_type, amb_prev=amb_prev,
            T_t_prev=T_t_prev)
        if self.multiscale:
            belief = self.shrink_conv(
                self.backbone.decode_multiscale_feature(belief))
        if self.refine is not None:
            belief = self.refine(belief)

        out = {'cls_preds': self.cls_head(belief),
               'reg_preds': self.reg_head(belief),
               'cls_preds_single': psm_single,
               'reg_preds_single': rm_single}
        heads = [('reg_preds', 7)]
        if self.use_dir:
            out['dir_preds'] = self.dir_head(belief)
            out['dir_preds_single'] = self.dir_head_single(feats)
            heads.append(('dir_preds', out['dir_preds'].shape[1] //
                          out['cls_preds'].shape[1]))
        if self.late and not self.training:
            out['cls_preds'], fused = self.late_fusion(
                record_len, out['cls_preds'], psm_single,
                [(out[h], out[h + '_single'], k) for h, k in heads],
                self.late_min)
            for (h, _), m in zip(heads, fused):
                out[h] = m

        out.update({'comm_bits': stats['bits'],
                    'ego_alpha': stats['ego_alpha']})
        return out
