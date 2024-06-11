# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from os.path import join as pjoin
base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, '..'))
import argparse
import numpy as np
import json

from itertools import permutations
from os.path import join as pjoin
from eval_mesh import eval_CD
from utils.articulation_utils import eval_axis_and_state, read_gt, interpret_transforms
from utils.py_utils import list_to_array



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--joint_type', type=str, default='rotate')

    return parser.parse_args()


def main(args):
    gt_info_list = read_gt(pjoin(args.gt_path, 'trans.json'))
    num_joints = len(gt_info_list)
    results = {}

    for gt_name, pred_name in zip(('start', 'end'), ('init', 'last')):
        results[pred_name] = {}

        all_perm = permutations(range(num_joints))

        for p, perm in enumerate(all_perm):
            perm_str = 'p' + ''.join(list(map(str, perm))) + '_' if num_joints > 1 else ''

            with open(pjoin(args.pred_path, f'{pred_name}_{perm_str}motion.json'), 'r') as f:
                pred_info_list = json.load(f)
            pred_info_list = [{key: list_to_array(value) for key, value in pred_info.items()} for pred_info in
                              pred_info_list]

            metric_dict = {}

            for gt_i, pred_i in enumerate(perm):
                joint_str = f'_{gt_i}' if len(perm) > 1 else ''
                gt_joint = {key: value for key, value in gt_info_list[gt_i].items()}
                pred_joint = {key: value for key, value in pred_info_list[pred_i].items()}

                joint_type = gt_joint['type']

                pred_joint, _, __ = interpret_transforms(np.eye(3), np.zeros(3),
                                                         pred_joint['rotation'], pred_joint['translation'],
                                                         joint_type=joint_type)
                angle, distance, theta_diff = eval_axis_and_state(pred_joint, gt_joint, joint_type=joint_type,
                                                                  reverse=gt_name == 'end')
                metric_dict.update({f'axis_angle{joint_str}': angle,
                                    f'axis_dist{joint_str}': distance * 10,
                                    f'theta_diff{joint_str}': theta_diff})

            pred_list = [pjoin(args.pred_path, f'{pred_name}_{suffix}.obj')
                                      for suffix in [f'part_{i}' for i in range(num_joints + 1)] + ['all']]
            pred_s, pred_d_list, pred_w = pred_list[0], pred_list[1:-1], pred_list[-1]

            dyn_names = [''] if num_joints == 1 else [f'_{i}' for i in range(num_joints)]
            gt_list = [pjoin(args.gt_path, gt_name, f'{gt_name}_{mid}rotate.ply')
                                for mid in ['static_'] + [f'dynamic{dyn_name}_' for dyn_name in dyn_names] + ['']]
            gt_s, gt_d_list, gt_w = gt_list[0], gt_list[1:-1], gt_list[-1]

            s, d_list, w = eval_CD(pred_s, pred_d_list, pred_w, gt_s, gt_d_list, gt_w)

            metric_dict['chamfer static'] = s * 1000
            for dyn_name, d in zip(dyn_names, d_list):
                metric_dict[f'chamfer dynamic{dyn_name}'] = d * 1000
            metric_dict['chamfer whole'] = w * 1000

            results[pred_name].update({f'{perm_str}{key}': value for key, value in metric_dict.items()})

    for pred_name in ['init', 'last']:
        for metric in results[pred_name]:
            metric_name = f'{pred_name} {metric}'
            print(f'{metric_name : >20}: {results[pred_name][metric]:.6f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
