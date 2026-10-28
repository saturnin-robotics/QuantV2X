# -*- coding: utf-8 -*-

# ============================================================
# Intermediate fusion dataset with time, for CoopFGNN v2.
#
# On top of the intermediate dataset (each agent in its own frame when
# proj_first is false):
#   - latency: each non-ego agent sends data captured k frames earlier,
#     k uniform in [0, max_delay] in training and eval_delay in testing.
#     Its point cloud and pose come from frame t - k, so its own features
#     and its single-agent labels stay consistent; the ego ground truth is
#     still built from every agent's annotations at frame t.
#     time_delay = k * frame_dt goes to prior_encoding[..., 1].
#   - previous ego frame: point cloud and clean pose of the ego at t - 1
#     ('prev_ego_lidar', 'prev_ego_pose', 'prev_ego_valid'), from which the
#     model computes the ambiguity A_i^{t-1} that the senders received with
#     the previous message. At the first frame of a scenario the current
#     frame is loaded instead and prev_ego_valid is 0.
#
# Config, fusion args:
#   temporal: {max_delay: 2, eval_delay: 0, frame_dt: 0.1, prev_ego: true}
# ============================================================

import json
import os
import random

import numpy as np
import torch

from opencood.data_utils.datasets.intermediate_fusion_dataset import \
    getIntermediateFusionDataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import pcd_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.pcd_utils import mask_ego_points, shuffle_points


def getIntermediatetemporalFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    base = getIntermediateFusionDataset(cls)

    class IntermediateTemporalFusionDataset(base):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            args = params['fusion']['args'].get('temporal', {})
            self.max_delay = args.get('max_delay', 0)
            self.eval_delay = args.get('eval_delay', 0)
            self.frame_dt = args.get('frame_dt', 0.1)
            self.prev_ego = args.get('prev_ego', True)

        def locate(self, idx):
            """(scenario index, timestamp index) of a sample."""
            scenario_index = 0
            for i, ele in enumerate(self.len_record):
                if idx < ele:
                    scenario_index = i
                    break
            timestamp_index = idx if scenario_index == 0 else \
                idx - self.len_record[scenario_index - 1]
            return scenario_index, timestamp_index

        def load_frame(self, scenario_database, cav_id, timestamp_index):
            """Params and point cloud of one agent at one timestamp."""
            key = self.return_timestamp_key(scenario_database, timestamp_index)
            content = scenario_database[cav_id][key]
            json_file = content['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    params = json.load(f)
            else:
                params = load_yaml(content['yaml'])
            return params, pcd_utils.pcd_to_np(content['lidar'])

        def retrieve_base_data(self, idx):
            data = super().retrieve_base_data(idx)
            scenario_index, ti = self.locate(idx)
            database = self.scenario_database[scenario_index]
            for cav_id, cav in data.items():
                k = 0
                if not cav['ego']:
                    k = random.randint(0, self.max_delay) if self.train \
                        else self.eval_delay
                    k = min(k, ti)
                cav['time_delay'] = k * self.frame_dt
                if k > 0:
                    params, lidar = self.load_frame(database, cav_id, ti - k)
                    cav['params_now'] = cav['params']
                    cav['params'] = params
                    cav['lidar_np'] = lidar
            return data

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            out = super().get_item_single_car(selected_cav_base, ego_cav_base)
            if 'params_now' in selected_cav_base:
                # Ego ground truth from the annotations at frame t
                now = dict(selected_cav_base)
                now['params'] = selected_cav_base['params_now']
                center, mask, ids = self.generate_object_center(
                    [now], ego_cav_base['params']['lidar_pose_clean'])
                out.update({'object_bbx_center': center[mask == 1],
                            'object_bbx_mask': mask,
                            'object_ids': ids})
            return out

        def __getitem__(self, idx):
            out = super().__getitem__(idx)
            if out is None or not self.prev_ego:
                return out
            scenario_index, ti = self.locate(idx)
            database = self.scenario_database[scenario_index]
            ego_id = [c for c, v in database.items() if v['ego']][0]
            params, lidar = self.load_frame(database, ego_id, max(ti - 1, 0))
            lidar = mask_ego_points(shuffle_points(lidar))
            out['ego'].update({
                'prev_ego_lidar': self.pre_processor.preprocess(lidar),
                'prev_ego_pose': np.array(params['lidar_pose'], dtype=np.float64),
                'prev_ego_valid': float(ti > 0)})
            return out

        def collate_batch_train(self, batch):
            out = super().collate_batch_train(batch)
            if out is None or 'prev_ego_lidar' not in batch[0]['ego']:
                return out
            merged = merge_features_to_dict(
                [b['ego']['prev_ego_lidar'] for b in batch])
            out['ego'].update({
                'prev_ego_lidar': self.pre_processor.collate_batch(merged),
                'prev_ego_pose': torch.from_numpy(
                    np.stack([b['ego']['prev_ego_pose'] for b in batch])),
                'prev_ego_valid': torch.tensor(
                    [b['ego']['prev_ego_valid'] for b in batch])})
            return out

    return IntermediateTemporalFusionDataset
