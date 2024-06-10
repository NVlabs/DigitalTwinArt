# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import imageio.v2 as imageio
import cv2
import numpy as np
import json
from tqdm import tqdm
import copy
import ruamel.yaml
yaml = ruamel.yaml.YAML()
import time
import open3d as o3d
import multiprocessing
from os.path import join as pjoin
from model import ArtiModel
from utils.py_utils import set_seed, set_logging_file, list_to_array
from utils.geometry_utils import compute_scene_bounds, mask_and_normalize_data

try:
    multiprocessing.set_start_method('spawn')
except:
    pass


def run(cfg_dir, data_dir, save_dir, test_only=False, no_wandb=False, ckpt_path=None,
        num_parts=2, denoise=False):

    with open(pjoin(cfg_dir, 'config.yml'), 'r') as f:
        cfg = yaml.load(f)

    cfg['data_dir'] = data_dir
    cfg['save_dir'] = save_dir
    cfg['slot_num'] = num_parts

    os.makedirs(save_dir, exist_ok=True)
    set_logging_file(pjoin(save_dir, 'log.txt'))

    folders = save_dir.split('/')
    while folders[0] != 'runs':
        folders = folders[1:]
    exp_name = '/'.join(folders[1:])

    cfg['bounding_box'] = np.array(cfg['bounding_box']).reshape(2, 3)

    if 'random_seed' not in cfg:
        cfg['random_seed'] = int(time.time()) % 200003

    set_seed(cfg['random_seed'])

    K = np.loadtxt(pjoin(data_dir, 'cam_K.txt')).reshape(3, 3)

    keyframes = yaml.load(open(pjoin(data_dir, 'init_keyframes.yml'), 'r'))
    keys = list(keyframes.keys())

    frame_ids = []
    timesteps = []
    cam_in_obs = []
    for k in keys:
        cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
        cam_in_obs.append(cam_in_ob)
        timesteps.append(float(keyframes[k]['time']))
        frame_ids.append(k.replace('frame_', ''))
    cam_in_obs = np.array(cam_in_obs)
    timesteps = np.array(timesteps)

    max_timestep = np.max(timesteps) + 1
    normalized_timesteps = timesteps / max_timestep

    frame_names, rgbs, depths, masks = [], [], [], []

    rgb_dir = pjoin(data_dir, 'color_segmented')
    for frame_id in frame_ids:
        rgb_file = pjoin(rgb_dir, f'{frame_id}.png')
        rgb = imageio.imread(rgb_file)
        rgb_wh = rgb.shape[:2]
        depth = cv2.imread(rgb_file.replace('color_segmented', 'depth_filtered'), -1) / 1e3
        depth_wh = depth.shape[:2]
        if rgb_wh[0] != depth_wh[0] or rgb_wh[1] != depth_wh[1]:
            depth = cv2.resize(depth, (rgb_wh[1], rgb_wh[0]), interpolation=cv2.INTER_NEAREST)

        mask = cv2.imread(rgb_file.replace('color_segmented', 'mask'), -1)
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        frame_names.append(rgb_file)
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)

    glcam_in_obs = cam_in_obs

    scene_normalization_path = pjoin(data_dir, 'scene_normalization.npz')
    if os.path.exists(scene_normalization_path):
        scene_info = np.load(scene_normalization_path, allow_pickle=True)
        sc_factor, translation = scene_info['sc_factor'], scene_info['translation']
        pcd_normalized = scene_info['pcd_normalized']
        pcd_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_normalized.astype(np.float64)))
    else:
        sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(frame_names, glcam_in_obs, K,
                                                                                      use_mask=True,
                                                                                      base_dir=save_dir,
                                                                                      rgbs=np.array(rgbs),
                                                                                      depths=np.array(depths),
                                                                                      masks=np.array(masks),
                                                                                      cluster=denoise, eps=0.01,
                                                                                      min_samples=5,
                                                                                      sc_factor=None,
                                                                                      translation_cvcam=None)
        np.savez_compressed(scene_normalization_path, sc_factor=sc_factor, translation=translation,
                            pcd_normalized=np.asarray(pcd_normalized.points))

    cfg['sc_factor'] = float(sc_factor)
    cfg['translation'] = translation

    print("sc factor", sc_factor, 'translation', translation)

    rgbs, depths, masks, poses = mask_and_normalize_data(np.array(rgbs), depths=np.array(depths),
                                                         masks=np.array(masks),
                                                         poses=glcam_in_obs,
                                                         sc_factor=cfg['sc_factor'],
                                                         translation=cfg['translation'])
    cfg['sampled_frame_ids'] = np.arange(len(rgbs))

    nerf = ArtiModel(cfg, frame_names=frame_names, images=rgbs, depths=depths, masks=masks,
                     poses=poses, timesteps=normalized_timesteps, K=K, build_octree_pcd=pcd_normalized,
                     use_wandb=not no_wandb, exp_name=exp_name, max_timestep=int(max_timestep),
                     test_only=test_only)
    nerf.max_timestep = max_timestep

    if ckpt_path is not None:
        assert os.path.exists(ckpt_path)
    else:
        if os.path.exists(pjoin(save_dir, 'ckpt', 'model_latest.pth')):
            ckpt_path = pjoin(save_dir, 'ckpt', 'model_latest.pth')

    if ckpt_path is not None:
        nerf.load_weights(ckpt_path)

    if test_only:
        nerf.test()
    else:
        nerf.initialize_correspondence()
        corr_path = pjoin(data_dir, cfg['correspondence_name'])
        if os.path.exists(corr_path):
            for filename in tqdm(os.listdir(corr_path)):
                if filename.endswith('npz'):
                    cur_corr = np.load(pjoin(corr_path, filename), allow_pickle=True)['data']
                    try:
                        cur_corr = list(cur_corr)
                        nerf.load_correspondence(cur_corr)
                    except:
                        pass
        nerf.finalize_correspondence()

        print("Start training")
        nerf.train()

        with open(pjoin(save_dir, 'config.yml'), 'w') as ff:
            tmp = copy.deepcopy(cfg)
            for k in tmp.keys():
                if isinstance(tmp[k], np.ndarray):
                    tmp[k] = tmp[k].tolist()
            yaml.dump(tmp, ff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--cfg_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--num_parts', type=int, default=2)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir is not None else args.cfg_dir
    run(args.cfg_dir, args.data_dir, save_dir, test_only=args.test_only, no_wandb=args.no_wandb,
        ckpt_path=args.ckpt_path, num_parts=args.num_parts, denoise=args.denoise)
