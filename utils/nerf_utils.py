# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import numpy as np

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_camera_rays_np(H, W, K):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -np.ones_like(i)], axis=-1)
    return dirs


def get_pixel_coords_np(H, W, K):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    coords = np.stack([i, H - j - 1], axis=-1)  # assume K[1, 2] * 2 == H, -(j - K[1, 2]) = -j + H - K[1, 2]
    return coords


def get_masks(z_vals, target_d, truncation, cfg, dir_norm=None):
    valid_depth_mask = (target_d >= cfg['near'] * cfg['sc_factor']) & (target_d <= cfg['far'] * cfg['sc_factor'])
    front_mask = (z_vals < target_d - truncation)
    back_mask = (z_vals > target_d + truncation * cfg['neg_trunc_ratio'])

    sdf_mask = (1.0 - front_mask.float()) * (1.0 - back_mask.float()) * valid_depth_mask

    return front_mask.bool(), sdf_mask.bool()


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, cfg, return_mask=False, rays_d=None):
    dir_norm = rays_d.norm(dim=-1, keepdim=True)
    front_mask, sdf_mask = get_masks(z_vals, target_d, truncation, cfg, dir_norm=dir_norm)

    mask = (target_d > cfg['far'] * cfg['sc_factor']) & (predicted_sdf < cfg['fs_sdf'])
    fs_loss = ((predicted_sdf - cfg['fs_sdf']) * mask) ** 2

    mask = front_mask & (target_d <= cfg['far'] * cfg['sc_factor']) & (predicted_sdf < 1)
    empty_loss = torch.abs(predicted_sdf - 1) * mask

    sdf_loss = ((z_vals + predicted_sdf * truncation) * sdf_mask - target_d * sdf_mask) ** 2

    if return_mask:
        return empty_loss, fs_loss, sdf_loss, front_mask, sdf_mask
    return empty_loss, fs_loss, sdf_loss


def ray_box_intersection_batch(origins, dirs, bounds):
    '''
    @origins: (N,3) origin and directions. In the same coordinate frame as the bounding box
    @bounds: (2,3) xyz_min and max
    '''
    if not torch.is_tensor(origins):
        origins = torch.tensor(origins)
        dirs = torch.tensor(dirs)
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds)

    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)
    inv_dirs = 1 / dirs
    bounds = bounds[None].expand(len(dirs), -1, -1)  # (N,2,3)

    sign = torch.zeros((len(dirs), 3)).long().to(dirs.device)  # (N,3)
    sign[:, 0] = (inv_dirs[:, 0] < 0)
    sign[:, 1] = (inv_dirs[:, 1] < 0)
    sign[:, 2] = (inv_dirs[:, 2] < 0)

    tmin = (torch.gather(bounds[..., 0], dim=1, index=sign[:, 0].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                               0]) * inv_dirs[:,
                                                                                                     0]  # (N)
    tmin[tmin < 0] = 0
    tmax = (torch.gather(bounds[..., 0], dim=1, index=1 - sign[:, 0].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                                   0]) * inv_dirs[:, 0]
    tymin = (torch.gather(bounds[..., 1], dim=1, index=sign[:, 1].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                                1]) * inv_dirs[:, 1]
    tymin[tymin < 0] = 0
    tymax = (torch.gather(bounds[..., 1], dim=1, index=1 - sign[:, 1].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                                    1]) * inv_dirs[:, 1]

    ishit = torch.ones(len(dirs)).bool().to(dirs.device)
    ishit[(tmin > tymax) | (tymin > tmax)] = 0
    tmin[tymin > tmin] = tymin[tymin > tmin]
    tmax[tymax < tmax] = tymax[tymax < tmax]

    tzmin = (torch.gather(bounds[..., 2], dim=1, index=sign[:, 2].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                                2]) * inv_dirs[:, 2]
    tzmin[tzmin < 0] = 0
    tzmax = (torch.gather(bounds[..., 2], dim=1, index=1 - sign[:, 2].reshape(-1, 1)).reshape(-1) - origins[:,
                                                                                                    2]) * inv_dirs[:, 2]

    ishit[(tmin > tzmax) | (tzmin > tmax)] = 0
    tmin[tzmin > tmin] = tzmin[tzmin > tmin]  # (N)
    tmax[tzmax < tmax] = tzmax[tzmax < tmax]

    tmin[ishit == 0] = -1
    tmax[ishit == 0] = -1

    return tmin, tmax

