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
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, pjoin(base_dir, '..'))
import numpy as np
import argparse
import yaml
import cv2
import imageio
from PIL import Image
from tqdm import tqdm

from utils.py_utils import list_to_array
from loftr_wrapper import LoftrRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--src_name', type=str, default=None)
    parser.add_argument('--tgt_name', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--cat', type=str, default=None)

    return parser.parse_args()


def read_img_dict(folder, name, poses=None):
    rgb_subfolder = 'color_raw'
    if not os.path.exists(pjoin(folder, rgb_subfolder)):
        rgb_subfolder = 'color_segmented'
    img_dict = {}
    for key, subfolder in (('rgb', rgb_subfolder), ('mask', 'mask')):
        img_name = pjoin(folder, subfolder, f'{name}.png')
        img = np.array(Image.open(img_name))
        if key == 'mask' and len(img.shape) == 3:
            img = img[..., 0]
        img_dict[key] = img
    if poses is None:
        poses = yaml.safe_load(open(pjoin(folder, 'init_keyframes.yml'), 'r'))
    img_dict['cam2world'] = list_to_array(poses[f'frame_{name}']['cam_in_ob']).reshape(4, 4)

    return img_dict, poses


def run_source_img(loftr, folder, src_name, tgt_names=None, top_k=-1,
                   filter_level_list=['no_filter'], visualize=False, vis_path='test_loftr'):

    src_time = int(src_name.split('_')[0])

    K = np.loadtxt(pjoin(folder, 'cam_K.txt')).reshape(3, 3)

    src_dict, poses = read_img_dict(folder, src_name)

    src_position = src_dict['cam2world'][:3, 3]

    if tgt_names is None:
        all_tgt = []
        for frame_name in poses:
            if poses[frame_name]['time'] != src_time:
                tgt_position = list_to_array(poses[frame_name]['cam_in_ob']).reshape(4, 4)[:3, 3]
                all_tgt.append(('_'.join(frame_name.split('_')[-2:]), ((src_position - tgt_position) ** 2).sum()))
        all_tgt.sort(key=lambda x: x[1])
        if top_k <= 0:
            top_k = len(all_tgt)
        tgt_names = [pair[0] for pair in all_tgt[:top_k]]

    tgt_dicts = [read_img_dict(folder, tgt_name, poses)[0] for tgt_name in tgt_names]

    all_corr = compute_correspondence(loftr, src_dict, tgt_dicts, K, filter_level_list=filter_level_list,
                                      visualize=visualize, vis_path=vis_path)

    results = {filter_level: [{src_name: tgt_corr[0], tgt_name: tgt_corr[1]}
               for tgt_name, tgt_corr in zip(tgt_names, corr)]
               for filter_level, corr in all_corr.items()}

    return results


def draw_corr(rgbA, rgbB, corrA, corrB, output_name):
    vis = np.concatenate([rgbA, rgbB], axis=1)
    radius = 2
    for i in range(len(corrA)):
        uvA = corrA[i]
        uvB = corrB[i].copy()
        uvB[0] += rgbA.shape[1]
        color = tuple(np.random.randint(0, 255, size=(3)).tolist())
        vis = cv2.circle(vis, uvA, radius=radius, color=color, thickness=1)
        vis = cv2.circle(vis, uvB, radius=radius, color=color, thickness=1)
        vis = cv2.line(vis, uvA, uvB, color=color, thickness=1, lineType=cv2.LINE_AA)
    imageio.imwrite(f'{output_name}.png', vis.astype(np.uint8))


def compute_correspondence(loftr, src_dict, tgt_dicts, K, visualize=False, vis_path='test_loftr',
                           filter_level_list=['no_filter']):
    src_mask = src_dict['mask']
    src_rgb = src_dict['rgb']
    img_h, img_w = src_rgb.shape[:2]

    # src_pixels = np.stack(np.where(src_seg > 0), axis=-1)
    fx, _, cx = K[0]
    _, fy, cy = K[1]

    tgt_rgbs = np.stack([tgt_dict['rgb'] for tgt_dict in tgt_dicts], axis=0)
    corres = loftr.predict(np.tile(src_rgb[np.newaxis], (len(tgt_rgbs), 1, 1, 1)), tgt_rgbs)

    def get_valid_mask(mask, coords):
        valid = np.logical_and(np.logical_and(coords[..., 0] >= 0, coords[..., 0] < img_w),
                               np.logical_and(coords[..., 1] >= 0, coords[..., 1] < img_h))
        valid = np.logical_and(valid, mask[coords[..., 1], coords[..., 0]])
        return valid

    os.makedirs('debug', exist_ok=True)

    filtered_corr = {key: [] for key in filter_level_list}

    for i, tgt_dict in enumerate(tgt_dicts):
        tgt_mask, tgt_rgb = tgt_dict['mask'], tgt_dict['rgb']
        cur_corres = corres[i]
        src_coords = cur_corres[:, :2].round().astype(int)
        tgt_coords = cur_corres[:, 2:4].round().astype(int)
        valid_mask = np.logical_and(get_valid_mask(src_mask, src_coords),
                                    get_valid_mask(tgt_mask, tgt_coords))

        loftr_total = len(valid_mask)
        valid_total = sum(valid_mask)

        src_coords = src_coords[np.where(valid_mask)[0]]
        tgt_coords = tgt_coords[np.where(valid_mask)[0]]

        if 'no_filter' in filter_level_list:
            filtered_corr['no_filter'].append((src_coords, tgt_coords))

        if visualize:
            draw_corr(src_rgb, tgt_rgb, src_coords, tgt_coords, pjoin(vis_path, f'{i}_1_no_filter_{valid_total}_of_{loftr_total}'))

    return filtered_corr


def main(args):

    loftr = LoftrRunner()
    if args.batch:
        paris_path = args.data_path
        for folder in os.listdir(paris_path):
            if os.path.isdir(pjoin(paris_path, folder)):
                if args.cat is not None and not folder.startswith(args.cat):
                    continue
                print(f'Running for folder {folder}')
                run_folder(loftr, pjoin(paris_path, folder),
                           src_name=None, tgt_name=None, output_path=pjoin(paris_path, folder, 'correspondence_loftr'),
                           top_k=args.top_k)
    else:
        run_folder(loftr, args.data_path, args.src_name, args.tgt_name, args.output_path, args.top_k)


def run_folder(loftr, folder, src_name, tgt_name, output_path, top_k):
    if src_name is not None:
        src_names = [src_name]
    else:
        all_frames = yaml.safe_load(open(pjoin(folder, 'init_keyframes.yml'), 'r'))
        src_names = ['_'.join(frame_name.split('_')[-2:]) for frame_name in all_frames]

    if tgt_name is not None:
        tgt_names = [tgt_name]
        save_name = f'tgt_{tgt_name}'
    else:
        tgt_names = None
        save_name = f'tgt_all'
        if top_k >= 0:
            save_name = f'{save_name}_top{top_k}'

    os.makedirs(output_path, exist_ok=True)

    pbar = tqdm(src_names)
    for src_name in pbar:
        pbar.set_description(src_name)
        results = run_source_img(loftr, folder, src_name, tgt_names, top_k=top_k, visualize=False)
        for filter_level in results:
            os.makedirs(pjoin(output_path, filter_level), exist_ok=True)
            np.savez_compressed(pjoin(output_path, filter_level, f'src_{src_name}_{save_name}.npz'),
                                data=results[filter_level])


if __name__ == '__main__':
    args = parse_args()
    main(args)