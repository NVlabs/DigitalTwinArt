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
sys.path.insert(0, pjoin(base_path, '..'))
sys.path.insert(0, base_path)
import json
from scipy.spatial.transform import Rotation as R
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import open3d as o3d

from argparse import ArgumentParser
from utils.py_utils import list_to_array


def interpret_transforms(base_R, base_t, R, t, joint_type='revolute'):
    """
    base_R, base_t, R, t are all from canonical to world
    rewrite the transformation = global transformation (base_R, base_t) {R' part + t'} --> s.t. R' and t' happens in canonical space
    R', t':
    - revolute: R'p + t' = R'(p - a) + a, R' --> axis-theta representation; axis goes through a = (I - R')^{-1}t'
    - prismatic: R' = I, t' = l * axis_direction
    """
    R = np.matmul(base_R.T, R)
    t = np.matmul(base_R.T, (t - base_t).reshape(3, 1)).reshape(-1)

    if joint_type == 'revolute':
        rotvec = Rotation.from_matrix(R).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=-1)
        axis_direction = rotvec / max(theta, (theta < 1e-8))
        try:
            axis_position = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
        except:   # TO DO find the best solution
            axis_position = np.zeros(3)
        axis_position += axis_direction * np.dot(axis_direction, -axis_position)
        joint_info = {'axis_position': axis_position,
                      'axis_direction': axis_direction,
                      'theta': np.rad2deg(theta),
                      'rotation': R, 'translation': t}

    elif joint_type == 'prismatic':
        theta = np.linalg.norm(t)
        axis_direction = t / max(theta, (theta < 1e-8))
        joint_info = {'axis_direction': axis_direction, 'axis_position': np.zeros(3), 'theta': theta,
                      'rotation': R, 'translation': t}

    return joint_info, R, t


def read_pts_from_obj_file(obj_file):
    tm = trimesh.load(obj_file)
    return np.array(tm.vertices)


def transform_pts_with_rt(pts, rot, trans):
    return (np.matmul(rot, pts.T) + trans.reshape(3, 1)).T


def read_gt(gt_path):
    with open(gt_path, 'r') as f:
        info = json.load(f)

    all_trans_info = info['trans_info']
    if isinstance(all_trans_info, dict):
        all_trans_info = [all_trans_info]
    ret_list = []
    for trans_info in all_trans_info:
        axis = trans_info['axis']
        axis_o, axis_d = np.array(axis['o']), np.array(axis['d'])
        axis_type = trans_info['type']
        l, r = trans_info[axis_type]['l'], trans_info[axis_type]['r']

        if axis_type == 'rotate':
            rotvec = axis_d * np.deg2rad(r - l)
            rot = R.from_rotvec(rotvec).as_matrix()
            trans = np.matmul(np.eye(3) - rot, axis_o.reshape(3, 1)).reshape(-1)
            joint_type = 'revolute'
        else:
            rot = np.eye(3)
            trans = (r - l) * axis_d
            joint_type = 'prismatic'
        ret_list.append({'axis_position': axis_o, 'axis_direction': axis_d, 'theta': r - l, 'joint_type': axis_type, 'rotation': rot, 'translation': trans,
                         'type': joint_type})
    return ret_list


def line_distance(a_o, a_d, b_o, b_d):
    normal = np.cross(a_d, b_d)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:   # parallel
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    else:
        return np.abs(np.dot(normal, a_o - b_o)) / normal_length


def eval_axis_and_state(axis_a, axis_b, joint_type='revolute', reverse=False):
    a_d, b_d = axis_a['axis_direction'], axis_b['axis_direction']

    angle = np.rad2deg(np.arccos(np.dot(a_d, b_d) / np.linalg.norm(a_d) / np.linalg.norm(b_d)))
    angle = min(angle, 180 - angle)

    if joint_type == 'revolute':
        a_o, b_o = axis_a['axis_position'], axis_b['axis_position']
        distance = line_distance(a_o, a_d, b_o, b_d)

        a_r, b_r = axis_a['rotation'], axis_b['rotation']
        if reverse:
            a_r = a_r.T

        r_diff = np.matmul(a_r, b_r.T)
        state = np.rad2deg(np.arccos(np.clip((np.trace(r_diff) - 1.0) * 0.5, a_min=-1, a_max=1)))
    else:
        distance = 0
        a_t, b_t = axis_a['translation'], axis_b['translation']
        if reverse:
            a_t = -a_t

        state = np.linalg.norm(a_t - b_t)

    return angle, distance, state


def geodesic_distance(pred_R, gt_R):
    '''
    q is the output from the network (rotation from t=0.5 to t=1)
    gt_R is the GT rotation from t=0 to t=1
    '''
    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle))
    return angle


def axis_metrics(motion, gt):
    # pred axis
    pred_axis_d = motion['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt['axis_d']
    gt_axis_o = gt['axis_o']
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err


def translational_error(motion, gt):
    dist_half = motion['dist'].cpu()
    dist = dist_half * 2.
    gt_dist = gt['dist']

    axis_d = F.normalize(motion['axis_d'].cpu().squeeze(0), p=2, dim=0)
    gt_axis_d = F.normalize(gt['axis_d'].cpu(), p=2, dim=0)

    err = torch.sqrt(((dist * axis_d - gt_dist * gt_axis_d) ** 2).sum())
    return err


def eval_step_info(path, step_name, gt_path, base_joint=0):
    # gt_joint, gt_rot, gt_trans = read_gt(gt_path)
    gt_joint_list = read_gt(gt_path)
    gt_joint = gt_joint_list[0]

    step_transform = np.load(pjoin(path, 'step_transform', f'{step_name}_part_pose.npz'), allow_pickle=True)['data'].item()

    cfg = yaml.safe_load(open(pjoin(path, 'config_nerf.yml'), 'r'))
    nerf_scale = cfg['sc_factor']
    nerf_trans = np.array(cfg['translation'])
    inv_scale = 1.0 / nerf_scale
    inv_trans = (-nerf_scale * nerf_trans).reshape(1, 3)

    rot = step_transform['rotation'].swapaxes(-1, -2)   # somehow R^T is saved..
    trans = step_transform['translation']
    trans = (trans + inv_trans - np.matmul(rot, inv_trans[..., np.newaxis])[..., 0]) * inv_scale

    num_parts = len(rot)

    joint_dict = {}
    for joint_i in range(num_parts):
        if joint_i == base_joint:
            continue
        cur_joint, cur_rot, cur_trans = interpret_transforms(rot[base_joint], trans[base_joint], rot[joint_i], trans[joint_i])
        print('cur joint', cur_joint)
        print('gt joint', gt_joint)
        angle, distance, theta_diff = eval_axis_and_state(cur_joint, gt_joint)
        print('axis diff angle', angle, 'distance', distance)
        print('joint state diff angle', theta_diff)

        # cur_joint_tensor = {'axis_d': torch.tensor(cur_joint['axis_direction']).reshape(1, 3),
        #                     'axis_o': torch.tensor(cur_joint['axis_position']).reshape(1, 3)}
        # gt_joint_tensor = {'axis_d': torch.tensor(gt_joint['axis_direction']),
        #                    'axis_o': torch.tensor(gt_joint['axis_position'])}
        # print(axis_metrics(cur_joint_tensor, gt_joint_tensor))
        """
        cur_pts = transform_pts_with_rt(part_canon_pts[joint_i], cur_rot, cur_trans)
        center = np.mean(cur_pts, axis=0)
        if 'axis_position' in cur_joint:
            p = cur_joint['axis_position']
            u = cur_joint['axis_direction']
            cur_joint['axis_position'] = p + (np.dot(center, u) - np.dot(p, u)) * u
        cur_pt_dict[joint_i] = cur_pts
        """
        joint_dict[joint_i] = cur_joint


def eval_result(path, step_name, gt_path):
    gt_joint_list = read_gt(gt_path)
    gt_joint = gt_joint_list[0]

    for cnc in ['init', 'last']:
        if cnc == 'last':
            gt_joint['theta'] *= -1.
        with open(pjoin(path, 'results', f'{step_name}', f'{cnc}_motion.json'), 'r') as f:
            axis_info = json.load(f)

        axis_info = {key: list_to_array(value) for key, value in axis_info.items()}
        angle, distance, theta_diff = eval_axis_and_state(axis_info, gt_joint)
        print('axis diff angle', angle, 'distance', distance)
        print('joint state diff angle', theta_diff)

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction unit vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R


def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k)
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)



def main(args):
    if args.name is None:
        eval_result(args.path, f'step_{args.step:07d}', args.gt_path)
    else:
        eval_step_info(args.path, f'step_{args.step:07d}_{args.name}', args.gt_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--step', default=3500, type=int)
    parser.add_argument('--base_joint', default=0, type=int)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--gt_path', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

