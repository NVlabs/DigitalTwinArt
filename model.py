# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os.path
import cv2
import torch
import numpy as np
import imageio, trimesh
import json
import logging
import wandb
import common
import copy
from itertools import permutations
from tqdm import tqdm
from os.path import join as pjoin
from utils.nerf_utils import ray_box_intersection_batch, \
    get_sdf_loss, get_camera_rays_np, get_pixel_coords_np, to8b
from utils.geometry_utils import to_homo, transform_pts, OctreeManager, get_voxel_pts, \
    DepthFuser, VoxelVisibility, VoxelSDF, sdf_voxel_from_mesh
from network import PartArticulationNet, SHEncoder, GridEncoder, FeatureVolume, NeRFSmall
from utils.articulation_utils import save_axis_mesh, interpret_transforms, eval_axis_and_state, read_gt as read_axis_gt
from eval.eval_mesh import eval_CD, cluster_meshes

"""
train_loop(batch of rays): call [render], compute losses
- train_loop_forward: similar

render: call [batchify_rays], split the results into ['rgb_map'] and others
batchify_rays: call [render_rays] in chunks, concat the results
render_rays: sample points, run [run_network] or [run_network_for_forward_only]
"""


def inverse_transform(transform):
    rot = transform['rot']
    trans = transform['trans']
    return {'rot': rot.T, 'trans': -np.matmul(rot.T, trans.reshape(3, 1)).reshape(-1)}


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def compute_near_far_and_filter_rays(cam_in_world, rays, cfg):
    '''
    @cam_in_world: in normalized space
    @rays: (...,D) in camera
    Return:
        (-1,D+2) with near far
    '''
    D = rays.shape[-1]
    rays = rays.reshape(-1, D)
    dirs_unit = rays[:, :3] / np.linalg.norm(rays[:, :3], axis=-1).reshape(-1, 1)
    dirs = (cam_in_world[:3, :3] @ rays[:, :3].T).T
    origins = (cam_in_world @ to_homo(np.zeros(dirs.shape)).T).T[:, :3]
    bounds = np.array(cfg['bounding_box']).reshape(2, 3)
    tmin, tmax = ray_box_intersection_batch(origins, dirs, bounds)
    tmin = tmin.data.cpu().numpy()
    tmax = tmax.data.cpu().numpy()
    ishit = tmin >= 0
    near = (dirs_unit * tmin.reshape(-1, 1))[:, 2]
    far = (dirs_unit * tmax.reshape(-1, 1))[:, 2]
    good_rays = rays[ishit]
    near = near[ishit]
    far = far[ishit]
    near = np.abs(near)
    far = np.abs(far)
    good_rays = np.concatenate((good_rays, near.reshape(-1, 1), far.reshape(-1, 1)), axis=-1)  # (N,8+2)

    return good_rays


@torch.no_grad()
def sample_rays_uniform(N_samples, near, far, lindisp=False, perturb=True):
    '''
    @near: (N_ray,1)
    '''
    N_ray = near.shape[0]
    t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device).reshape(1, -1)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))  # (N_ray,N_sample)

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=far.device)
        z_vals = lower + (upper - lower) * t_rand
        z_vals = torch.clip(z_vals, near, far)

    return z_vals.reshape(N_ray, N_samples)


class DataLoader:
    def __init__(self, rays, batch_size):
        self.rays = rays
        self.batch_size = batch_size
        self.pos = 0
        self.ids = torch.randperm(len(self.rays))

    def __next__(self):
        if self.pos + self.batch_size < len(self.ids):
            self.batch_ray_ids = self.ids[self.pos:self.pos + self.batch_size]
            out = self.rays[self.batch_ray_ids]
            self.pos += self.batch_size
            return out.cuda()

        self.ids = torch.randperm(len(self.rays))
        self.pos = self.batch_size
        self.batch_ray_ids = self.ids[:self.batch_size]
        return self.rays[self.batch_ray_ids].cuda()


class IndexDataLoader:
    def __init__(self, indices, batch_size):
        self.indices = indices
        self.batch_size = batch_size
        self.pos = 0
        self.ids = torch.randperm(len(self.indices))

    def __next__(self):
        if self.pos + self.batch_size < len(self.ids):
            out = self.indices[self.ids[self.pos:self.pos + self.batch_size]]
            self.pos += self.batch_size
            return out

        self.ids = torch.randperm(len(self.indices))
        self.pos = self.batch_size
        return self.indices[self.ids[:self.batch_size]]


class ArtiModel:
    def __init__(self, cfg, frame_names, images, depths, masks, poses, timesteps, K,
                 build_octree_pcd=None, use_wandb=True, exp_name=None, max_timestep=0,
                 test_only=False):
        '''
        normal_maps: use None
        poses: opengl convention, camera pose w.r.t. object(object frame normalized to [-1, 1] or [0, 1]); z- forward, y up;
        K: cam intrinsics

        '''
        self.cfg = cfg
        self.frame_names = frame_names
        self.frame_name2id = {'_'.join(frame_name.split('/')[-1].split('.')[0].split('_')[-2:]): id for id, frame_name in enumerate(frame_names)}
        self.images = images
        self.depths = depths
        self.masks = masks
        self.poses = poses
        self.timesteps = timesteps
        self.all_timesteps = np.unique(timesteps)
        self.all_timesteps.sort()
        self.all_timesteps = torch.tensor(self.all_timesteps).cuda()
        self.max_timestep = max_timestep
        self.cnc_timesteps = {'init': 0.0, 'last': (self.max_timestep - 1.0) / self.max_timestep}
        assert self.cnc_timesteps['last'] == self.all_timesteps[-1]
        self.K = K.copy()

        self.load_gt()

        self.build_octree_pts = np.asarray(build_octree_pcd.points).copy()  # Make it pickable

        self.save_dir = self.cfg['save_dir']

        self.H, self.W = self.images[0].shape[:2]
        self.tensor_K = torch.tensor(self.K, device='cuda:0', dtype=torch.float32)

        self.octree_m = None
        if self.cfg['use_octree']:
            self.build_octree()

        self.create_nerf()
        self.create_optimizer()

        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])

        self.total_step = self.cfg['n_step']
        self.global_step = 0
        self.freeze_recon_step = self.cfg['freeze_recon_step']

        self.c2w_array = torch.tensor(poses).float().cuda()

        if not test_only:
            rays_ = {cnc_name: [] for cnc_name in self.cnc_timesteps}
            num_rays_ = {cnc_name: 0 for cnc_name in self.cnc_timesteps}
            pixel_to_ray_id = {cnc_name: {} for cnc_name in self.cnc_timesteps}
            for frame_i in tqdm(range(len(self.timesteps))):
                for cnc_name in self.cnc_timesteps:
                    if self.timesteps[frame_i] == self.cnc_timesteps[cnc_name]:
                        frame_rays, frame_pixel_to_ray_id = self.make_frame_rays(frame_i)
                        rays_[cnc_name].append(frame_rays)
                        frame_pixel_to_ray_id[np.where(frame_pixel_to_ray_id >= 0)] += num_rays_[cnc_name]
                        pixel_to_ray_id[cnc_name][frame_i] = frame_pixel_to_ray_id
                        num_rays_[cnc_name] += len(frame_rays)
            self.pixel_to_ray_id = pixel_to_ray_id

            rays_dict = {}
            for cnc_name in self.cnc_timesteps:
                rays_dict[cnc_name] = np.concatenate(rays_[cnc_name], axis=0)

            for cnc_name in self.cnc_timesteps:
                rays_dict[cnc_name] = torch.tensor(rays_dict[cnc_name], dtype=torch.float).cuda()

            self.rays_dict = rays_dict

            self.data_loader = {cnc_name: DataLoader(rays=self.rays_dict[cnc_name], batch_size=self.cfg['N_rand'])
                                for cnc_name in self.rays_dict}

        self.loss_weights = {key: torch.tensor(value).float().cuda() for key, value in self.cfg['loss_weights'].items()}
        self.loss_schedule = {} if 'loss_schedule' not in self.cfg else self.cfg['loss_schedule']

        self.use_wandb = use_wandb and not test_only
        if self.use_wandb:
            wandb.init(project='art-nerf', name=exp_name)
            wandb.init(config=self.cfg)

        self.depth_fuser = {}
        for cnc_name in self.cnc_timesteps:
            cur_frame_idx = np.where(self.timesteps == self.cnc_timesteps[cnc_name])
            self.depth_fuser[cnc_name] = DepthFuser(self.tensor_K, self.c2w_array[cur_frame_idx],
                                                    self.depths[cur_frame_idx].squeeze(-1),
                                                    self.masks[cur_frame_idx].squeeze(-1),
                                                    self.get_truncation(),
                                                    near=self.cfg['near'] * self.cfg['sc_factor'],
                                                    far=self.cfg['far'] * self.cfg['sc_factor'])
        self.load_visibility_grid()

    def load_visibility_grid(self):
        self.visibility_grid = {}
        for cnc_name in self.cnc_timesteps:
            visibility_path = pjoin(self.cfg['data_dir'], f'{cnc_name}_visibility.npz')
            if os.path.exists(visibility_path):
                visibility = np.load(visibility_path, allow_pickle=True)['data']
            else:
                query_pts = get_voxel_pts(self.cfg['sdf_voxel_size'])
                old_shape = tuple(query_pts.shape[:3])
                query_pts = torch.tensor(query_pts.astype(np.float32).reshape(-1, 3)).float().cuda()

                if self.octree_m is not None:
                    vox_size = self.cfg['octree_raytracing_voxel_size'] * self.cfg['sc_factor']
                    level = int(np.floor(np.log2(2.0 / vox_size)))

                    chunk = 160000
                    all_valid = []
                    for i in range(0, query_pts.shape[0], chunk):
                        cur_pts = query_pts[i: i + chunk]
                        center_ids = self.octree_m.get_center_ids(cur_pts, level)
                        valid = center_ids >= 0
                        all_valid.append(valid)
                    valid = torch.cat(all_valid, dim=0)
                else:
                    valid = torch.ones(len(query_pts), dtype=bool).cuda()

                flat = query_pts[valid]
                chunk = 160000
                observed = []
                for i in range(0, flat.shape[0], chunk):
                    observed.append(self.depth_fuser[cnc_name].query(flat[i:i + chunk]))

                observed = torch.cat(observed, dim=0)

                visibility = np.zeros(len(query_pts), dtype=bool)
                visibility[valid.cpu().numpy()] = observed.cpu().numpy()

                np.savez_compressed(visibility_path, data=visibility.reshape(old_shape))

                visibility = visibility.reshape(old_shape)

            self.visibility_grid[cnc_name] = VoxelVisibility(visibility)

    def initialize_correspondence(self):
        self.correspondence = {cnc_name: [] for cnc_name in self.cnc_timesteps}
        self.corr_src_id_slice = 0
        self.corr_tgt_frame_slice = 1
        self.corr_tgt_pixel_silce = [2, 3]

    def load_correspondence(self, corr_list, downsample=10):

        def rev_pixel(pixel):
            return pixel * np.array([1, -1]).reshape(1, 2) + np.array([0, self.H - 1]).reshape(1, 2)

        for corr in corr_list:
            for order in [1, -1]:
                src_name, tgt_name = list(corr.keys())[::order]
                src_pixel, tgt_pixel = corr[src_name], corr[tgt_name]  # smaller coords are at the top - the same index to use for images
                src_pixel = rev_pixel(src_pixel)
                tgt_pixel = rev_pixel(tgt_pixel)
                # however, the coords here are - smaller at the bottom
                cnc_name = {0: 'init', 1: 'last'}[int(src_name.split('_')[0])]
                if src_name not in self.frame_name2id or tgt_name not in self.frame_name2id:
                    continue
                src_frame_id = self.frame_name2id[src_name]
                tgt_frame_id = self.frame_name2id[tgt_name]
                src_idx = src_pixel[:, 1] * self.W + src_pixel[:, 0]
                src_ray_ids = self.pixel_to_ray_id[cnc_name][src_frame_id][src_idx].reshape(-1, 1)
                valid_idx = np.where(src_ray_ids >= 0)[0]
                target_length = max(500, len(valid_idx) // downsample)
                final_idx = np.random.permutation(valid_idx)[:target_length]
                tgt_frame_ids = np.ones_like(src_ray_ids) * tgt_frame_id
                cur_corr = np.concatenate([src_ray_ids, tgt_frame_ids, tgt_pixel], axis=-1)
                self.correspondence[cnc_name].append(cur_corr[final_idx])

    def finalize_correspondence(self):
        self.correspondence = {cnc_name: None if len(corr_list) == 0 else np.concatenate(corr_list, axis=0) for cnc_name, corr_list in self.correspondence.items()}

        upper_limit = self.H * self.W * len(self.frame_names) * 5
        self.correspondence = {cnc_name: None if corr is None else torch.tensor(corr[np.random.permutation(len(corr))[:upper_limit]]).cuda() for cnc_name, corr in self.correspondence.items()}
        self.corr_loader = {cnc_name: None if self.correspondence[cnc_name] is None else DataLoader(rays=self.correspondence[cnc_name], batch_size=self.cfg['N_rand']) for cnc_name in self.correspondence}

    def plot_loss(self, loss_dict, step):
        if self.use_wandb:
            wandb.log(loss_dict, step=step)

    def create_nerf(self, device=torch.device("cuda")):

        models = {}
        for cnc_name in self.cnc_timesteps:
            embed_fn = GridEncoder(input_dim=3, n_levels=self.cfg['num_levels'],
                                   log2_hashmap_size=self.cfg['log2_hashmap_size'],
                                   desired_resolution=self.cfg['finest_res'], base_resolution=self.cfg['base_res'],
                                   level_dim=self.cfg['feature_grid_dim'])
            embed_fn = embed_fn.to(device)
            input_ch = embed_fn.out_dim
            models[f'{cnc_name}_embed_fn'] = embed_fn

            embeddirs_fn = SHEncoder(self.cfg['multires_views'])
            input_ch_views = embeddirs_fn.out_dim
            models[f'{cnc_name}_embeddirs_fn'] = embeddirs_fn

            model = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64,
                              input_ch=input_ch, input_ch_views=input_ch_views).to(device)
            model = model.to(device)
            models[f'{cnc_name}_model'] = model

            embed_bwdflow_fn = FeatureVolume(out_dim=self.cfg['feature_vol_dim'], res=self.cfg['feature_vol_res'], num_dim=3)
            embed_bwdflow_fn = embed_bwdflow_fn.to(device)
            models[f'{cnc_name}_embed_bwdflow_fn'] = embed_bwdflow_fn

            embed_fwdflow_fn = FeatureVolume(out_dim=self.cfg['feature_vol_dim'], res=self.cfg['feature_vol_res'], num_dim=3)
            embed_fwdflow_fn = embed_fwdflow_fn.to(device)
            models[f'{cnc_name}_embed_fwdflow_fn'] = embed_fwdflow_fn

            fwdflow_ch = self.cfg['feature_vol_dim']

            if self.cfg['share_motion'] and cnc_name == 'last':
                inv_transform = lambda: models['init_deformation_model'].get_raw_slot_transform()
            else:
                inv_transform = None
            deformation_model = PartArticulationNet(device=device, feat_dim=fwdflow_ch,
                                                    slot_num=self.cfg['slot_num'],
                                                    slot_hard=self.cfg['slot_hard'],
                                                    gt_transform=None,
                                                    inv_transform=inv_transform,
                                                    fix_base=self.cfg.get('fix_base', True),
                                                    gt_joint_types=None if not self.cfg['use_gt_joint_type'] else self.gt_joint_types)

            deformation_model = deformation_model.to(device)

            models[f'{cnc_name}_deformation_model'] = deformation_model

        self.models = models
        print(models)

    def make_frame_rays(self, frame_id):

        def get_last_ray_slice_idx(rays, num):
            if num == 1:
                return rays.shape[-1] - 1
            else:
                return list(range(rays.shape[-1] - num, rays.shape[-1]))
        mask = self.masks[frame_id, ..., 0].copy()

        rays = get_camera_rays_np(self.H, self.W,
                                  self.K)  # [self.H, self.W, 3]  We create rays frame-by-frame to save memory
        self.ray_dir_slice = get_last_ray_slice_idx(rays, 3)

        rays = np.concatenate([rays, frame_id * np.ones(self.depths[frame_id].shape)], -1)  # [H, W, 18]
        self.ray_frame_id_slice = get_last_ray_slice_idx(rays, 1)

        rays = np.concatenate([rays, self.depths[frame_id]], -1)  # [H, W, 7]
        self.ray_depth_slice = get_last_ray_slice_idx(rays, 1)

        ray_types = np.zeros((self.H, self.W, 1))  # 0 is good; 1 is invalid depth (uncertain)
        invalid_depth = ((self.depths[frame_id, ..., 0] < self.cfg['near'] * self.cfg['sc_factor']) | (
                self.depths[frame_id, ..., 0] > self.cfg['far'] * self.cfg['sc_factor'])) & (mask > 0)
        ray_types[invalid_depth] = 1
        rays = np.concatenate((rays, ray_types), axis=-1)  # 19
        self.ray_type_slice = get_last_ray_slice_idx(rays, 1)

        rays = np.concatenate([rays, get_pixel_coords_np(self.H, self.W, self.K)], axis=-1)
        self.ray_coords_slice = get_last_ray_slice_idx(rays, 2)

        rays = np.concatenate([rays, self.images[frame_id]], -1)  # [H, W, 6]
        self.ray_rgb_slice = get_last_ray_slice_idx(rays, 3)

        rays = np.concatenate([rays, self.masks[frame_id] > 0], -1)  # [H, W, 8]
        self.ray_mask_slice = get_last_ray_slice_idx(rays, 1)

        rays = np.concatenate([rays, self.timesteps[frame_id] * np.ones(self.depths[frame_id].shape)], -1)  # 20
        self.ray_time_slice = get_last_ray_slice_idx(rays, 1)

        n = rays.shape[-1]

        dilate = 60
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        if self.cfg['rays_valid_depth_only']:
            mask[invalid_depth] = 0

        vs, us = np.where(mask > 0)
        cur_rays = rays[vs, us].reshape(-1, n)
        cur_rays = cur_rays[cur_rays[:, self.ray_type_slice] == 0]
        cur_rays = compute_near_far_and_filter_rays(self.poses[frame_id], cur_rays, self.cfg)

        self.ray_near_slice, self.ray_far_slice = get_last_ray_slice_idx(rays, 2)

        if self.cfg['use_octree']:
            rays_o_world = (self.poses[frame_id] @ to_homo(np.zeros((len(cur_rays), 3))).T).T[:, :3]
            rays_o_world = torch.from_numpy(rays_o_world).cuda().float()
            rays_unit_d_cam = cur_rays[:, :3] / np.linalg.norm(cur_rays[:, :3], axis=-1).reshape(-1, 1)
            rays_d_world = (self.poses[frame_id][:3, :3] @ rays_unit_d_cam.T).T
            rays_d_world = torch.from_numpy(rays_d_world).cuda().float()

            vox_size = self.cfg['octree_raytracing_voxel_size'] * self.cfg['sc_factor']
            level = int(np.floor(np.log2(2.0 / vox_size)))
            near, far, _, ray_depths_in_out = self.octree_m.ray_trace(rays_o_world, rays_d_world, level=level)
            near = near.cpu().numpy()
            valid = (near > 0).reshape(-1)
            cur_rays = cur_rays[valid]

        cur_ray_coords = cur_rays[:, self.ray_coords_slice]   # [N, 2], x in [0, W - 1], y in [0, H - 1]
        coords = (cur_ray_coords[:, 1] * self.W + cur_ray_coords[:, 0]).astype(np.int32)
        pixel_to_ray_id = np.ones(self.H * self.W) * -1
        pixel_to_ray_id[coords] = np.arange(len(coords))

        return cur_rays, pixel_to_ray_id

    def build_octree(self):

        pts = torch.tensor(self.build_octree_pts).cuda().float()  # Must be within [-1,1]
        octree_smallest_voxel_size = self.cfg['octree_smallest_voxel_size'] * self.cfg['sc_factor']
        finest_n_voxels = 2.0 / octree_smallest_voxel_size
        max_level = int(np.ceil(np.log2(finest_n_voxels)))
        octree_smallest_voxel_size = 2.0 / (2 ** max_level)

        dilate_radius = int(np.ceil(self.cfg['octree_dilate_size'] / self.cfg['octree_smallest_voxel_size']))
        dilate_radius = max(1, dilate_radius)
        logging.info(f"Octree voxel dilate_radius:{dilate_radius}")
        shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shifts.append([dx, dy, dz])
        shifts = torch.tensor(shifts).cuda().long()  # (27,3)
        coords = torch.floor((pts + 1) / octree_smallest_voxel_size).long()  # (N,3)
        dilated_coords = coords.detach().clone()
        for iter in range(dilate_radius):
            dilated_coords = (dilated_coords[None].expand(shifts.shape[0], -1, -1) + shifts[:, None]).reshape(-1, 3)
            dilated_coords = torch.unique(dilated_coords, dim=0)
        pts = (dilated_coords + 0.5) * octree_smallest_voxel_size - 1
        pts = torch.clip(pts, -1, 1)

        assert pts.min() >= -1 and pts.max() <= 1
        self.octree_m = OctreeManager(pts, max_level)

    def create_optimizer(self):
        params = []
        for k in self.models:
            if self.models[k] is not None:
                params += list(self.models[k].parameters())

        param_groups = [{'name': 'basic', 'params': params, 'lr': self.cfg['lrate']}]

        self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=0, eps=1e-15)
        self.param_groups_init = copy.deepcopy(self.optimizer.param_groups)

    def load_weights(self, ckpt_path):
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        for key in self.models:
            self.models[key].load_state_dict(ckpt[key])

        if 'octree' in ckpt:
            self.octree_m = OctreeManager(octree=ckpt['octree'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt['global_step']

        if self.global_step >= self.freeze_recon_step:
            self.freeze_recon()

    def freeze_recon(self):
        print("----------------freeze recon--------------")
        for cnc_name in self.cnc_timesteps:
            for suffix in ['model', 'embed_fn', 'embeddirs_fn']:
                model_key = f'{cnc_name}_{suffix}'
                if model_key in self.models:
                    for param in self.models[model_key].parameters():
                        param.requires_grad = False

    def save_weights(self, output_path):
        data = {
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
        }
        for key in self.models:
            data[key] = self.models[key].state_dict()

        if self.octree_m is not None:
            data['octree'] = self.octree_m.octree

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(data, output_path)
        print('Saved checkpoints at', output_path)
        latest_path = pjoin(output_dir, 'model_latest.pth')
        if latest_path != output_path:
            os.system(f'cp {output_path} {latest_path}')

    def schedule_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            init_lr = self.param_groups_init[i]['lr']
            new_lrate = init_lr * (self.cfg['decay_rate'] ** (float(self.global_step) / self.total_step))
            param_group['lr'] = new_lrate

    def load_gt(self):
        gt_path = pjoin(self.cfg['data_dir'], 'gt')
        if not os.path.exists(gt_path):
            print("No gt")
            self.gt_dict = None
            return

        gt_joint_list = read_axis_gt(pjoin(gt_path, 'trans.json'))
        gt_rot_list = [gt_joint['rotation'] for gt_joint in gt_joint_list]
        gt_trans_list = [gt_joint['translation'] for gt_joint in gt_joint_list]

        self.gt_joint_types = [gt_joint['type'] for gt_joint in gt_joint_list]

        gt_dict = {'joint': gt_joint_list, 'rot': gt_rot_list, 'trans': gt_trans_list}
        num_joints = len(gt_joint_list)

        for gt_name in ('start', 'end'):
            if len(gt_joint_list) > 1:
                gt_meshes = [pjoin(gt_path, gt_name, f'{gt_name}_{mid}rotate.ply')
                                    for mid in ['', 'static_'] + [f'dynamic_{i}_' for i in range(num_joints)]]
                gt_w, gt_s, gt_d = gt_meshes[0], gt_meshes[1], gt_meshes[2:]
            else:
                gt_w, gt_s, gt_d = [pjoin(gt_path, gt_name, f'{gt_name}_{mid}rotate.ply')
                                    for mid in ['', 'static_', 'dynamic_']]
                gt_d = [gt_d]

            gt_dict[f'mesh_{gt_name}'] = {'s': gt_s, 'd': gt_d, 'w': gt_w}

        self.gt_dict = gt_dict

    def get_truncation(self):
        truncation = self.cfg['trunc']
        truncation *= self.cfg['sc_factor']
        return truncation

    def query_full_sdf(self, cnc_name, queries):
        sdf = self.recon_sdf_dict[cnc_name].query(queries.reshape(-1, 3)) / self.get_truncation()
        sdf = sdf.reshape(queries.shape[:-1])
        return sdf

    def query_visibility(self, cnc_name, queries):
        visibility = self.visibility_grid[cnc_name].query(queries.reshape(-1, 3))
        visibility = visibility.reshape(queries.shape[:-1])
        return visibility

    def backward_flow(self, cnc_name, pts, valid_samples, training=True):

        if valid_samples is None:
            valid_samples = torch.ones((len(pts)), dtype=torch.bool, device=pts.device)

        inputs_flat = pts  # torch.cat([pts, timesteps], dim=-1)

        embedded_bwdflow = torch.zeros((inputs_flat.shape[0], self.models[f'{cnc_name}_embed_bwdflow_fn'].out_dim),
                                       device=inputs_flat.device)

        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            embedded_bwdflow[valid_samples] = self.models[f'{cnc_name}_embed_bwdflow_fn'](
                inputs_flat[valid_samples]).to(embedded_bwdflow.dtype)

        embedded_bwdflow = embedded_bwdflow.float()

        canonical_pts = []
        bwd_attn_hard, bwd_attn_soft = [], []
        raw_cnc, raw_slot_attn, raw_slot_sdf = [], [], []
        all_max_attn = []
        all_total_occ = []
        all_non_max_occ = []
        empty_slot_mask = []
        canonical_pts_cand = []
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            chunk = self.cfg['netchunk']
            for i in range(0, embedded_bwdflow.shape[0], chunk):
                out = self.models[f'{cnc_name}_deformation_model'].back_deform(pts[i: i + chunk], embedded_bwdflow[i: i + chunk])
                xyz_cnc = out['xyz_cnc']  # [N, S, 3]
                num_pts, num_slots = xyz_cnc.shape[:2]
                xyz_cnc = xyz_cnc.reshape(-1, 3)
                raw_cnc.append(xyz_cnc)
                with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
                    embedded_fwd_cnc = self.models[f'{cnc_name}_embed_fwdflow_fn'](xyz_cnc.float()).float()
                fwd_attn_hard, fwd_attn_raw = self.models[f'{cnc_name}_deformation_model'].forw_attn(xyz_cnc, embedded_fwd_cnc, training=training)  # [N * S, S]

                def pick_slot_attn(fwd_attn):
                    fwd_attn = fwd_attn.reshape(num_pts, num_slots, num_slots)
                    fwd_attn = fwd_attn[
                        torch.arange(num_pts).to(fwd_attn.device).long().reshape(-1, 1).repeat(1, num_slots),  # [N, S]
                        torch.arange(num_slots).to(fwd_attn.device).long().reshape(1, -1).repeat(num_pts, 1),  # [N, S]
                        torch.arange(num_slots).to(fwd_attn.device).long().reshape(1, -1).repeat(num_pts, 1)]  # [N, S]
                    return fwd_attn

                fwd_attn_hard = pick_slot_attn(fwd_attn_hard)
                fwd_attn_raw = pick_slot_attn(fwd_attn_raw)

                # [2] candidates, [2, 2] <-- diagonal  --> [S], fwd_attn_soft
                # point 0: prob(point 0 belongs to slot 0) prob(point 0 belongs to slot 1)
                # point 1: prob(point 1 belongs to slot 0) prob(point 1 belongs to slot 1)

                raw_slot_attn.append(fwd_attn_raw)  # for future analysis

                sdf = self.query_full_sdf(cnc_name, xyz_cnc.float())
                weights_from_sdf = self.get_occ_from_full_sdf(sdf)

                weights_from_sdf = weights_from_sdf.reshape(num_pts, num_slots)
                raw_slot_sdf.append(weights_from_sdf)

                dots = fwd_attn_hard * weights_from_sdf  # * weights_from_sdf  # [N, S]
                total_occ = torch.sum(dots, dim=-1)
                non_max_occ = total_occ - torch.max(dots, dim=-1)[0]
                dots = torch.cat([dots, torch.ones_like(dots[:, :1]) * self.cfg['empty_slot_weight']], dim=-1)

                # let the stochasticity only happen in forward pass; just take their results (attn_hard), and run straight-through argmax
                attn = dots / torch.sum(dots, dim=1, keepdim=True)
                max_attn, index = attn.max(dim=1, keepdim=True)  # [N]
                y_hard = torch.zeros_like(attn, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
                attn_hard = y_hard - attn.detach() + attn
                attn_raw = attn

                # make all indices other than the max index have a small value
                all_max_attn.append(max_attn.reshape(-1))
                all_total_occ.append(total_occ)
                all_non_max_occ.append(non_max_occ)

                xyz_base = torch.cat([xyz_cnc.reshape(num_pts, num_slots, 3), pts[i: i + chunk].reshape(-1, 1, 3)], dim=1)

                chosen_cnc = (attn_hard.unsqueeze(-1) * xyz_base).sum(dim=1)
                bwd_attn_hard.append(attn_hard[:, :-1])
                bwd_attn_soft.append(attn_raw[:, :-1])
                canonical_pts.append(chosen_cnc)
                empty_slot_mask.append(attn_hard[:, -1])
                canonical_pts_cand.append(xyz_cnc.reshape(num_pts, num_slots, 3))

            canonical_pts = torch.cat(canonical_pts, dim=0).float()
            if len(bwd_attn_hard) > 0 and bwd_attn_hard[0] is not None:
                bwd_attn_hard = torch.cat(bwd_attn_hard, dim=0).float()
                bwd_attn_soft = torch.cat(bwd_attn_soft, dim=0).float()
            else:
                bwd_attn_hard, bwd_attn_soft = None, None
            if len(raw_cnc) > 0:
                raw_cnc = torch.cat(raw_cnc, dim=0)
                raw_slot_attn = torch.cat(raw_slot_attn, dim=0)
                raw_slot_sdf = torch.cat(raw_slot_sdf, dim=0)
                empty_slot_mask = torch.cat(empty_slot_mask, dim=0)
                all_max_attn = torch.cat(all_max_attn, dim=0)
                all_total_occ = torch.cat(all_total_occ, dim=0)
                all_non_max_occ = torch.cat(all_non_max_occ, dim=0)
                canonical_pts_cand = torch.cat(canonical_pts_cand, dim=0)
            else:
                raw_cnc, raw_slot_attn, raw_slot_sdf, empty_slot_mask, canonical_pts_cand = None, None, None, None, None

            ret_dict = {'canonical_pts': canonical_pts, 'canonical_pts_cand': canonical_pts_cand,
                        'bwd_attn_hard': bwd_attn_hard, 'bwd_attn_soft': bwd_attn_soft,
                        'raw_cnc': raw_cnc, 'raw_slot_attn': raw_slot_attn, 'raw_slot_sdf': raw_slot_sdf,
                        'empty_slot_mask': empty_slot_mask, 'max_attn': all_max_attn, 'total_occ': all_total_occ, 'non_max_occ': all_non_max_occ}

            return ret_dict

    def forward_flow(self, cnc_name, pts, valid_samples, training=True):

        if valid_samples is None:
            valid_samples = torch.ones((len(pts)), dtype=torch.bool, device=pts.device)

        inputs_flat = pts
        embedded_fwdflow = torch.zeros((inputs_flat.shape[0], self.models[f'{cnc_name}_embed_fwdflow_fn'].out_dim),
                                       device=inputs_flat.device)

        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            embedded_fwdflow[valid_samples] = self.models[f'{cnc_name}_embed_fwdflow_fn'](
                inputs_flat[valid_samples]).to(embedded_fwdflow.dtype)

        embedded_fwdflow = embedded_fwdflow.float()

        world_pts = []
        world_pts_cand = []
        fwd_attn_hard, fwd_attn_soft = [], []
        fwd_rot, fwd_trans = [], []
        fwd_rot_cand = []
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            chunk = self.cfg['netchunk']
            for i in range(0, embedded_fwdflow.shape[0], chunk):
                out = self.models[f'{cnc_name}_deformation_model'].forw_deform(pts[i: i + chunk],
                                                                   embedded_fwdflow[i: i + chunk],
                                                                   training=training, gt_attn=None)
                world_pts.append(out['world_pts'])
                world_pts_cand.append(out['world_pts_cand'])
                fwd_attn_hard.append(out['attn_hard'])  # [N, S]
                fwd_attn_soft.append(out['attn_soft'])
                fwd_rot.append(out['rotation'])
                fwd_trans.append(out['translation'])
                fwd_rot_cand.append(out['rotation_cand'])
        world_pts = torch.cat(world_pts, dim=0).float()
        world_pts_cand = torch.cat(world_pts_cand, dim=0).float()
        if fwd_attn_hard[0] is not None:
            fwd_attn_hard = torch.cat(fwd_attn_hard, dim=0).float()
            fwd_attn_soft = torch.cat(fwd_attn_soft, dim=0).float()
            fwd_rot = torch.cat(fwd_rot, dim=0).float()
            fwd_rot_cand = torch.cat(fwd_rot_cand, dim=0).float()
            fwd_trans = torch.cat(fwd_trans, dim=0).float()
        else:
            fwd_attn_hard, fwd_attn_soft = None, None
        return {'world_pts': world_pts, 'world_pts_cand': world_pts_cand,
                'fwd_attn_hard': fwd_attn_hard, 'fwd_attn_soft': fwd_attn_soft,
                'fwd_rot': fwd_rot, 'fwd_trans': fwd_trans, 'fwd_rot_cand': fwd_rot_cand,
                'cnc_features': embedded_fwdflow}

    def project_to_pixel(self, cam_pts):
        projection = torch.matmul(self.tensor_K[:2, :2],
                                  (cam_pts[..., :2] /
                                   torch.clip(-cam_pts[..., 2:3], min=1e-8)).transpose(0, 1)) + self.tensor_K[:2, 2:3]
        projection = projection.transpose(0, 1)
        return projection

    def get_canonical_pts_from_world_pts(self, cnc_name, world_pts, timesteps, valid_samples):
        ret = {}
        first_mask = (timesteps == self.cnc_timesteps[cnc_name]).float()
        if first_mask.mean() == 1:
            canonical_pts = world_pts
        else:
            backward_flow = self.backward_flow(cnc_name, world_pts, valid_samples)
            canonical_pts = backward_flow['canonical_pts']
            for key in ['bwd_attn_soft', 'bwd_attn_hard', 'raw_cnc', 'raw_slot_attn', 'raw_slot_sdf', 'empty_slot_mask',
                        'max_attn', 'total_occ', 'non_max_occ', 'canonical_pts_cand']:
                if key in backward_flow and backward_flow[key] is not None:
                    ret[key] = backward_flow[key]
            canonical_pts = first_mask * world_pts + (1 - first_mask) * canonical_pts

        ret['canonical_pts'] = canonical_pts

        return ret

    def get_world_pts_from_canonical_pts(self, cnc_name, canonical_pts, timesteps, valid_samples, training=True):
        ret = {}
        first_mask = (timesteps == self.cnc_timesteps[cnc_name]).float()

        forward_flow = self.forward_flow(cnc_name, canonical_pts, valid_samples, training=training)
        world_pts = forward_flow['world_pts']
        world_pts_cand = forward_flow['world_pts_cand']

        world_pts = first_mask * canonical_pts + (1 - first_mask) * world_pts
        world_pts_cand = first_mask.unsqueeze(1) * canonical_pts.unsqueeze(1) + (1 - first_mask.unsqueeze(1)) * world_pts_cand
        for key in ['fwd_attn_soft', 'fwd_attn_hard', 'cnc_features', 'fwd_rot', 'fwd_trans', 'fwd_rot_cand']:
            ret[key] = forward_flow[key]

        ret.update({'world_pts': world_pts, 'world_pts_cand': world_pts_cand})

        return ret

    def summarize_loss(self, loss_dict):
        loss = torch.tensor(0.).cuda()
        for loss_name, weight in self.loss_weights.items():
            if weight > 0 and loss_name in loss_dict:
                if loss_name in self.loss_schedule and self.loss_schedule[loss_name] > self.global_step:
                    continue
                loss += loss_dict[loss_name] * weight
        return loss

    def train_epilogue(self, cnc_name, loss_dict):
        loss = self.summarize_loss(loss_dict)

        if (self.global_step + 1) % self.cfg['i_print'] == 0:
            msg = f"Iter: {self.global_step + 1}, {cnc_name}, "
            metrics = {
                'loss': loss.item(),
            }
            metrics.update({loss_name: loss_dict[loss_name].item() for loss_name in loss_dict
                            if loss_name.startswith(cnc_name) or loss_name.startswith('self')})

            for k in metrics.keys():
                msg += f"{k}: {metrics[k]:.7f}, "
            msg += "\n"
            logging.info(msg)

        if (self.global_step + 1) % self.cfg['i_wandb'] == 0 and self.use_wandb:
            self.plot_loss({'total_loss': loss.item()}, self.global_step)
            self.plot_loss({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']},
                           self.global_step)
            self.plot_loss(loss_dict, self.global_step)

        if loss.requires_grad:
            self.optimizer.zero_grad()

            self.amp_scaler.scale(loss).backward()

            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()

            if (self.global_step + 1) % 10 == 0:
                self.schedule_lr()

            if (self.global_step + 1) % self.cfg['i_weights'] == 0 and cnc_name == 'last':
                self.save_weights(output_path=os.path.join(self.save_dir, 'ckpt', f'model_{self.global_step + 1:07d}.pth'))

        if (self.global_step + 1) % self.cfg['i_mesh'] == 0:
            self.export_canonical(cnc_name, per_part=self.global_step >= self.freeze_recon_step)

        if (self.global_step + 1) % self.cfg['i_img'] == 0 and self.global_step < self.freeze_recon_step:
            ids = torch.unique(self.rays_dict[cnc_name][:, self.ray_frame_id_slice]).data.cpu().numpy().astype(int).tolist()
            ids.sort()
            ids = ids[::10][:10]

            os.makedirs(pjoin(self.save_dir, 'step_img'), exist_ok=True)
            dir = f"{self.save_dir}/step_img/step_{self.global_step + 1:07d}_{cnc_name}"
            os.makedirs(dir, exist_ok=True)
            for frame_idx in ids:
                rgb, depth, ray_mask, gt_rgb, gt_depth, _ = self.render_images(cnc_name, frame_idx)
                mask_vis = (rgb * 255 * 0.2 + ray_mask * 0.8).astype(np.uint8)
                mask_vis = np.clip(mask_vis, 0, 255)
                rgb = np.concatenate((rgb, gt_rgb), axis=1)
                far = self.cfg['far'] * self.cfg['sc_factor']
                gt_depth = np.clip(gt_depth, self.cfg['near'] * self.cfg['sc_factor'], far)
                depth_vis = np.concatenate((to8b(depth / far), to8b(gt_depth / far)), axis=1)
                depth_vis = np.tile(depth_vis[..., None], (1, 1, 3))
                row = np.concatenate((to8b(rgb), depth_vis, mask_vis), axis=1)
                img_name = self.frame_names[frame_idx].split('/')[-1].split('.')[-2]
                imageio.imwrite(pjoin(dir, f'{img_name}.png'), row.astype(np.uint8))

    def train_render_loop(self, cnc_name, batch):
        target_s = batch[:, self.ray_rgb_slice]  # Color (N,3)
        target_d = batch[:, self.ray_depth_slice]  # Normalized scale (N)

        extras = self.render_rays(cnc_name=cnc_name, ray_batch=batch,
                                  depth=target_d, lindisp=False, perturb=True)
        loss_dict = {}

        valid_samples = extras['valid_samples']  # (N_ray,N_samples)
        N_rays, N_samples = valid_samples.shape

        rgb = extras['rgb_map']

        valid_samples = extras['valid_samples']  # (N_ray,N_samples)
        z_vals = extras['z_vals']  # [N_rand, N_samples + N_importance]

        sdf = extras['raw'][..., -1]

        valid_rays = (valid_samples > 0).any(dim=-1).bool().reshape(N_rays) & (batch[:, self.ray_type_slice] == 0)
        valid_sample_weights = valid_samples * valid_rays.view(-1, 1)

        rgb_loss = (((rgb - target_s) ** 2 * valid_rays.view(-1, 1))).mean(dim=-1)

        loss_dict['self_rgb'] = rgb_loss.mean()

        truncation = self.get_truncation()

        empty_loss, fs_loss, sdf_loss, front_mask, sdf_mask = get_sdf_loss(z_vals, target_d.reshape(-1, 1).expand(-1, N_samples),
                                                                           sdf, truncation, self.cfg, return_mask=True,
                                                                           rays_d=batch[:, self.ray_dir_slice])

        for loss, loss_name in zip((fs_loss, empty_loss, sdf_loss), ('freespace', 'empty', 'sdf')):
            loss = (loss * valid_sample_weights).mean(dim=-1)
            loss_dict[f'self_{loss_name}'] = loss.mean()

        return loss_dict

    def forward_consistency(self, cnc_name, cnc_pts, cnc_viewdirs=None, valid_samples=None):

        other_cnc_name = [name for name in self.cnc_timesteps if name != cnc_name][0]
        target_timesteps = torch.ones_like(cnc_pts[..., :1]) * self.cnc_timesteps[other_cnc_name]
        target_pts_dict = self.get_world_pts_from_canonical_pts(cnc_name, cnc_pts, target_timesteps, valid_samples.reshape(-1))
        target_pts = target_pts_dict['world_pts']
        target_pts_cand = target_pts_dict['world_pts_cand']

        attn = target_pts_dict['fwd_attn_hard']
        num_slots = target_pts_cand.shape[1]

        target_rot = target_pts_dict['fwd_rot']  # [N, 3, 3]
        target_rot_cand = target_pts_dict['fwd_rot_cand']

        num_pts = len(target_pts)
        target_pts_all = torch.cat([target_pts, target_pts_cand.reshape(-1, 3)], dim=0)

        valid_samples_cand = valid_samples.unsqueeze(1).repeat(1, num_slots)
        valid_samples_all = torch.cat([valid_samples, valid_samples_cand.reshape(-1)], dim=0)

        if cnc_viewdirs is not None:  # [N, 3]
            target_viewdirs = torch.matmul(target_rot, cnc_viewdirs.unsqueeze(-1)).squeeze(-1)   # [N, 3]
            target_viewdirs_cand = torch.matmul(target_rot_cand.unsqueeze(0), cnc_viewdirs.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # [N, P, 3]
            target_viewdirs_all = torch.cat([target_viewdirs, target_viewdirs_cand.reshape(-1, 3)], dim=0)
            sdf_only = False
        else:
            target_viewdirs_all = None
            sdf_only = True

        target_outputs_all, __ = self.query_object_field(other_cnc_name, target_pts_all, valid_samples_all, viewdirs=target_viewdirs_all,
                                                            sdf_only=sdf_only)
        target_outputs, target_outputs_cand = target_outputs_all[:num_pts], target_outputs_all[num_pts:].reshape(-1, num_slots, target_outputs_all.shape[-1])

        target_outputs_post = (attn.unsqueeze(-1) * target_outputs_cand).sum(dim=1)
        target_outputs_post = target_outputs_post.reshape(target_outputs.shape)

        target_outputs = target_outputs_post

        target_full_sdf_cand = self.query_full_sdf(other_cnc_name, target_pts_cand.reshape(-1, 3).float())
        target_full_sdf_cand = target_full_sdf_cand.reshape(-1, num_slots)
        target_full_sdf_post = (attn * target_full_sdf_cand).sum(dim=1)
        target_full_sdf_post = target_full_sdf_post.reshape(-1)
        warped_sdf = target_full_sdf_post
        target_vis_cand = self.query_visibility(other_cnc_name, target_pts_cand.reshape(-1, 3).float()).float()
        warped_vis = (attn * target_vis_cand.reshape(-1, num_slots)).sum(dim=1).reshape(-1)

        ret_dict = {'warped_sdf': warped_sdf, 'fwd_attn': target_pts_dict['fwd_attn_soft'], 'warped_vis': warped_vis}

        if cnc_viewdirs is not None:
            ret_dict['warped_rgb'] = target_outputs[..., :3]
        return ret_dict

    def backward_consistency(self, cnc_name, world_pts, valid_samples):  # world_pts (N, 3)
        world_pts = world_pts.detach()
        other_cnc_name = [name for name in self.cnc_timesteps if name != cnc_name][0]
        target_timesteps = torch.ones_like(world_pts[..., :1]) * self.cnc_timesteps[other_cnc_name]

        canonical_pts_dict = self.get_canonical_pts_from_world_pts(cnc_name, world_pts, target_timesteps, valid_samples)

        ret_dict = {f'bwd_{key}': canonical_pts_dict[key] for key in ['max_attn', 'non_max_occ', 'total_occ']}
        ret_dict['bwd_attn'] = canonical_pts_dict['bwd_attn_soft']

        return ret_dict

    def compute_forward_losses(self, self_dict, forward_dict):
        loss_dict = {}

        weights_from_sdf = self_dict['weights']

        #--------------Consistency------------

        self_sdf = self_dict['sdf']
        warped_sdf = forward_dict['warped_sdf']
        self_vis = self_dict['visibility']
        warped_vis = forward_dict['warped_vis']

        vis_weight = self_vis * warped_vis
        vis_discount = self.cfg.get('vis_discount', 1.)
        vis_weight = (1 - vis_weight) + vis_weight * vis_discount
        weights = weights_from_sdf * vis_weight # * self_vis * warped_vis
        weights = weights / (weights_from_sdf.sum() + 1e-6)
        cns_sdf = ((warped_sdf - self_sdf.detach()).abs() * weights).sum()
        loss_dict[f'cns_sdf'] = cns_sdf

        if 'rgb' in self_dict and 'warped_rgb' in forward_dict:

            self_rgb = self_dict['rgb']
            warped_rgb = forward_dict['warped_rgb']
            cns_rgb = (((warped_rgb - self_rgb.detach()) ** 2).mean(dim=-1) * weights).sum()
            loss_dict['cns_rgb'] = cns_rgb

        return loss_dict

    def compute_backward_losses(self, self_dict, backward_dict):
        loss_dict = {}

        total_occ = backward_dict['bwd_total_occ']
        loss_dict['collision_occ'] = (torch.relu(total_occ - 1) ** 2).mean()

        if 'occ' in self_dict:
            occ = self_dict['occ']

            vis_weight = self_dict['visibility'].float()

            vis_discount = self.cfg.get('vis_discount', 1.)
            vis_weight = (1 - vis_weight) + vis_weight * vis_discount

            loss_dict['cns_occ'] = (((total_occ - occ) ** 2) * vis_weight).mean()

        return loss_dict

    def train_ray_loop(self, cnc_name, batch):
        target_d = batch[:, self.ray_depth_slice]  # Normalized scale (N)

        sample_dict = self.sample_rays(cnc_name, batch, lindisp=False, perturb=True, depth=target_d)

        cnc_pts, cnc_viewdirs, valid_samples = [sample_dict[key].reshape((-1, ) + sample_dict[key].shape[2:])
                                                for key in ['cnc_pts', 'cnc_viewdirs', 'valid_samples']]

        full_sdf = self.query_full_sdf(cnc_name, cnc_pts)
        visibility = self.query_visibility(cnc_name, cnc_pts)
        weights_from_sdf = self.get_weights_from_full_sdf(full_sdf)

        self_outputs, _ = self.query_object_field(cnc_name, cnc_pts, valid_samples, cnc_viewdirs)
        self_rgb = self_outputs[..., :3]
        self_sdf = full_sdf
        self_dict = {'rgb': self_rgb, 'sdf': self_sdf, 'weights': weights_from_sdf, 'visibility': visibility}
        self_dict['occ'] = self.get_occ_from_full_sdf(full_sdf)

        loss_dict = {}

        forward_dict = self.forward_consistency(cnc_name, cnc_pts, cnc_viewdirs, valid_samples)
        loss_dict.update(self.compute_forward_losses(self_dict, forward_dict))

        backward_dict = self.backward_consistency(cnc_name, cnc_pts, valid_samples)
        loss_dict.update(self.compute_backward_losses(self_dict, backward_dict))

        return loss_dict

    def sample_occ(self, cnc_name):
        canonical_pts = torch.rand(self.cfg['occ_sample_space'], 3).cuda() * 2 - 1.
        return canonical_pts

    def train_occ_loop(self, cnc_name):
        cnc_pts = self.sample_occ(cnc_name)

        full_sdf = self.query_full_sdf(cnc_name, cnc_pts)

        weights_from_sdf = self.get_weights_from_full_sdf(full_sdf)

        valid_samples = torch.ones(len(cnc_pts), dtype=torch.bool, device=cnc_pts.device)

        visibility = self.query_visibility(cnc_name, cnc_pts)

        self_sdf = full_sdf
        self_dict = {'sdf': self_sdf, 'weights': weights_from_sdf, 'visibility': visibility}
        self_dict['occ'] = self.get_occ_from_full_sdf(full_sdf)

        loss_dict = {}

        forward_dict = self.forward_consistency(cnc_name, cnc_pts, cnc_viewdirs=None, valid_samples=valid_samples)
        loss_dict.update(self.compute_forward_losses(self_dict, forward_dict))

        backward_dict = self.backward_consistency(cnc_name, cnc_pts, valid_samples)
        loss_dict.update(self.compute_backward_losses(self_dict, backward_dict))

        return loss_dict

    def train_corr_loop(self, cnc_name, corr_batch):
        src_ray_id = corr_batch[:, self.corr_src_id_slice].long()
        tgt_frame_id = corr_batch[:, self.corr_tgt_frame_slice].long()
        tgt_gt_pixel = corr_batch[:, self.corr_tgt_pixel_silce]

        batch = self.rays_dict[cnc_name][src_ray_id]
        target_d = batch[:, self.ray_depth_slice]  # Normalized scale (N)

        sample_dict = self.sample_rays(cnc_name=cnc_name, ray_batch=batch, lindisp=False,
                                       perturb=True, depth=target_d)
        tgt_pred = self.run_network_corr(cnc_name=cnc_name, sample_dict=sample_dict, depth=target_d)

        valid_rays = tgt_pred['valid']
        tgt_world_pts = tgt_pred['target_world_pts']
        valid_rays = valid_rays & (batch[:, self.ray_type_slice] == 0)

        tgt_tf = self.c2w_array[tgt_frame_id]

        tgt_cam_pts = (tgt_tf[:, :3, :3].transpose(-1, -2) @ (tgt_world_pts.unsqueeze(-1) - tgt_tf[:, :3, 3:])).squeeze(-1)
        tgt_pixel = self.project_to_pixel(tgt_cam_pts)

        corr_diff = (tgt_pixel - tgt_gt_pixel) * valid_rays.unsqueeze(-1)
        corr_loss = torch.abs(corr_diff).sum() / valid_rays.sum() / self.H

        loss_dict = {}

        loss_dict['corr'] = corr_loss
        loss_dict[f'{cnc_name}_corr'] = corr_loss

        return loss_dict

    def train_recon(self):
        start_step = self.global_step

        for iter in range(start_step, self.freeze_recon_step):
            for cnc_name in self.cnc_timesteps:
                batch = next(self.data_loader[cnc_name])
                loss_dict = self.train_render_loop(cnc_name, batch.cuda())

                self.train_epilogue(cnc_name, loss_dict)

            self.global_step += 1

        self.freeze_recon()

    def train_arti(self):
        start_step = self.global_step

        start_corr = -1 if 'corr' not in self.loss_schedule else self.loss_schedule['corr']

        for iter in range(start_step, self.total_step):
            for cnc_name in self.cnc_timesteps:
                batch = next(self.data_loader[cnc_name])
                loss_dict = {}

                if 'ray' in self.cfg['train_modes']:
                    ray_loss = self.train_ray_loop(cnc_name, batch.cuda())
                    loss_dict.update({f'ray_{key}': value for key, value in ray_loss.items()})
                    loss_dict.update({f'{cnc_name}_ray_{key}': value for key, value in ray_loss.items()})

                if 'occ' in self.cfg['train_modes']:
                    occ_loss = self.train_occ_loop(cnc_name)
                    loss_dict.update({f'occ_{key}': value for key, value in occ_loss.items()})
                    loss_dict.update({f'{cnc_name}_occ_{key}': value for key, value in occ_loss.items()})

                if iter % self.cfg['train_corr_n_step'] == 0 and iter > start_corr:
                    corr_batch = next(self.corr_loader[cnc_name])
                    loss_dict.update(self.train_corr_loop(cnc_name, corr_batch))

                self.train_epilogue(cnc_name, loss_dict)

            self.global_step += 1

    def load_recon(self):
        recon_path = pjoin(self.save_dir, 'recon')
        os.makedirs(recon_path, exist_ok=True)

        sdf_dict = {}

        nerf_scale = self.cfg['sc_factor']
        nerf_trans = self.cfg['translation'].reshape(1, 3)

        for cnc_name in ['init', 'last']:
            recon_mesh_path = pjoin(recon_path, f'{cnc_name}_all_clustered.obj')
            if not os.path.exists(recon_mesh_path):
                with torch.no_grad():
                    mesh = self.extract_canonical_info(cnc_name, isolevel=0.0,
                                                       voxel_size=self.cfg['mesh_resolution'], disable_octree=False,
                                                       per_part=False)

                vertices_raw = mesh.vertices.copy()
                vertices_raw = vertices_raw / nerf_scale - nerf_trans
                ori_mesh = trimesh.Trimesh(vertices_raw, mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)

                ori_mesh.export(pjoin(recon_path, f'{cnc_name}_all.obj'))

                cluster_meshes(pjoin(recon_path, f'{cnc_name}_all.obj'), [], None)

            sdf_voxel_size = self.cfg['sdf_voxel_size']
            sdf_path = pjoin(recon_path, f'{cnc_name}_{sdf_voxel_size:.3f}.npz')
            if os.path.exists(sdf_path):
                sdf = np.load(sdf_path, allow_pickle=True)['data']
            else:
                mesh = trimesh.load(recon_mesh_path)
                mesh.vertices = (mesh.vertices + nerf_trans) * nerf_scale
                print('mesh vertices range', np.asarray(mesh.vertices).min(axis=0),
                      np.asarray(mesh.vertices).max(axis=0))
                pts, sdf = sdf_voxel_from_mesh(mesh, sdf_voxel_size)
                np.savez_compressed(sdf_path, data=sdf)

            voxel_sdf = VoxelSDF(sdf)

            sdf_dict[cnc_name] = voxel_sdf

        self.recon_sdf_dict = sdf_dict

    def train(self):
        if self.global_step < self.freeze_recon_step:
            self.train_recon()

        self.freeze_recon()

        self.load_recon()

        self.train_arti()

    def test(self):
        assert self.global_step >= self.freeze_recon_step, 'Stage 1 reconstruction incomplete.'

        self.load_recon()

        for cnc_name in self.cnc_timesteps:
            self.export_canonical(cnc_name, per_part=True)

    @torch.no_grad()
    def sample_rays_uniform_occupied_voxels(self, rays_d, depths_in_out, lindisp=False, perturb=False,
                                            depths=None, N_samples=None):
        N_rays = rays_d.shape[0]
        N_intersect = depths_in_out.shape[1]
        dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

        z_in_out = depths_in_out.cuda() * torch.abs(dirs[..., 2]).reshape(N_rays, 1, 1).cuda()

        if depths is not None:
            depths = depths.reshape(-1, 1)
            trunc = self.get_truncation()
            valid = (depths >= self.cfg['near'] * self.cfg['sc_factor']) & (
                        depths <= self.cfg['far'] * self.cfg['sc_factor']).expand(-1, N_intersect)
            valid = valid & (z_in_out > 0).all(dim=-1)  # (N_ray, N_intersect)
            # upper_bound = 10.0 if not near_depth else 0.
            z_in_out[valid] = torch.clip(z_in_out[valid],
                                         min=torch.zeros_like(z_in_out[valid]),
                                         max=torch.ones_like(z_in_out[valid]) * (
                                                     depths.reshape(-1, 1, 1).expand(-1, N_intersect, 2)[
                                                         valid] + trunc))

        depths_lens = z_in_out[:, :, 1] - z_in_out[:, :, 0]  # (N_ray,N_intersect)
        z_vals_continous = sample_rays_uniform(N_samples,
                                               torch.zeros((N_rays, 1), device=z_in_out.device).reshape(-1, 1),
                                               depths_lens.sum(dim=-1).reshape(-1, 1), lindisp=lindisp,
                                               perturb=perturb)  # (N_ray,N_sample)

        N_samples = z_vals_continous.shape[1]
        z_vals = torch.zeros((N_rays, N_samples), dtype=torch.float, device=rays_d.device)
        z_vals = common.sampleRaysUniformOccupiedVoxels(z_in_out.contiguous(), z_vals_continous.contiguous(), z_vals)
        z_vals = z_vals.float().to(rays_d.device)  # (N_ray,N_sample)

        return z_vals, z_vals_continous

    def sample_rays(self, cnc_name, ray_batch, lindisp=False, perturb=False, depth=None):

        N_rays = ray_batch.shape[0]

        rays_d = ray_batch[:, self.ray_dir_slice]   # in world frame
        rays_o = torch.zeros_like(rays_d)
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

        frame_ids = ray_batch[:, self.ray_frame_id_slice].long()

        tf = self.c2w_array[frame_ids]

        rays_o_w = transform_pts(rays_o, tf)
        viewdirs_w = (tf[:, :3, :3] @ viewdirs[:, None].permute(0, 2, 1))[:, :3, 0]
        voxel_size = self.cfg['octree_raytracing_voxel_size'] * self.cfg['sc_factor']
        level = int(np.floor(np.log2(2.0 / voxel_size)))
        near, far, _, depths_in_out = self.octree_m.ray_trace(rays_o_w, viewdirs_w, level=level, debug=0)
        z_vals, _ = self.sample_rays_uniform_occupied_voxels(rays_d=viewdirs,
                                                             depths_in_out=depths_in_out, lindisp=lindisp,
                                                             perturb=perturb, depths=depth,
                                                             N_samples=self.cfg['N_samples'])

        if self.cfg['N_samples_around_depth'] > 0 and depth is not None:
            valid_depth_mask = (depth >= self.cfg['near'] * self.cfg['sc_factor']) & (
                        depth <= self.cfg['far'] * self.cfg['sc_factor'])
            valid_depth_mask = valid_depth_mask.reshape(-1)
            trunc = self.get_truncation()
            near_depth = depth[valid_depth_mask] - trunc
            far_depth = depth[valid_depth_mask] + trunc * self.cfg['neg_trunc_ratio']
            z_vals_around_depth = torch.zeros((N_rays, self.cfg['N_samples_around_depth']),
                                              device=ray_batch.device).float()
            # if torch.sum(inside_mask)>0:
            z_vals_around_depth[valid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],
                                                                        near_depth.reshape(-1, 1),
                                                                        far_depth.reshape(-1, 1), lindisp=lindisp,
                                                                        perturb=perturb)
            invalid_depth_mask = valid_depth_mask == 0

            if invalid_depth_mask.any() and self.cfg['use_octree']:
                z_vals_invalid, _ = self.sample_rays_uniform_occupied_voxels(rays_d=viewdirs[invalid_depth_mask],
                                                                             depths_in_out=depths_in_out[
                                                                                 invalid_depth_mask], lindisp=lindisp,
                                                                             perturb=perturb, depths=None,
                                                                             N_samples=self.cfg[
                                                                                 'N_samples_around_depth'])
                z_vals_around_depth[invalid_depth_mask] = z_vals_invalid
            else:
                z_vals_around_depth[invalid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],
                                                                              near[invalid_depth_mask].reshape(-1, 1),
                                                                              far[invalid_depth_mask].reshape(-1, 1),
                                                                              lindisp=lindisp, perturb=perturb)

            z_vals = torch.cat((z_vals, z_vals_around_depth), dim=-1)
        valid_samples = torch.ones(z_vals.shape, dtype=torch.bool,
                                   device=ray_batch.device)  # During pose update if ray out of box, it becomes invalid

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        N_ray, N_sample = pts.shape[:2]

        tf_flat = tf[:, None].expand(-1, N_sample, -1, -1).reshape(-1, 4, 4)
        cnc_pts = transform_pts(torch.reshape(pts, (-1, 3)), tf_flat)

        valid_samples = valid_samples.bool() & (torch.abs(cnc_pts) <= 1).all(dim=-1).view(N_ray, N_sample).bool()
        cnc_pts = cnc_pts.reshape(N_ray, N_sample, 3)

        cnc_viewdirs = (tf[..., :3, :3] @ viewdirs[..., None])[..., 0]  # (N_ray, 3)
        cnc_viewdirs = cnc_viewdirs.unsqueeze(1).repeat(1, N_sample, 1)

        return {'cnc_pts': cnc_pts, 'z_vals': z_vals, 'tf': tf, 'valid_samples': valid_samples,
                'cnc_viewdirs': cnc_viewdirs}

    def render_rays(self, cnc_name, ray_batch, lindisp=False, perturb=False, depth=None):
        sample_dict = self.sample_rays(cnc_name, ray_batch, lindisp=lindisp, perturb=perturb, depth=depth)

        cnc_pts, cnc_viewdirs, valid_samples, z_vals = [sample_dict[key]
                                                        for key in ['cnc_pts', 'cnc_viewdirs', 'valid_samples', 'z_vals']]

        cur_dict = self.run_network_render(cnc_name, cnc_pts, cnc_viewdirs,
                                           valid_samples=valid_samples)  # [N_rays, N_samples, 4]

        raw, valid_samples = cur_dict['raw'], cur_dict['valid_samples']

        rgb_map, weights = self.raw2outputs(raw, z_vals, valid_samples=valid_samples, depth=depth)

        cur_dict.update({'rgb_map': rgb_map, 'weights': weights, 'z_vals': z_vals})

        return cur_dict

    def get_ray_sample_weights_from_depth(self, z_vals, depth=None, pred_sdf=None, valid_samples=None, truncation=None):
        '''
        z_vals, valid_samples: [N_rays, N_samples]
        depth: [N_rays]
        '''
        if truncation is None:
            truncation = self.get_truncation()

        if depth is not None:
            depth = depth.view(-1, 1)
            sdf_from_depth = (depth - z_vals) / truncation
            sdf = sdf_from_depth
        else:
            sdf = pred_sdf

        weights = torch.sigmoid(sdf * self.cfg['sdf_lambda']) * torch.sigmoid(-sdf * self.cfg['sdf_lambda'])
        mask = (sdf > -self.cfg['neg_trunc_ratio']) & (sdf < 1)
        weights = weights * mask

        if depth is not None:
            invalid = (depth > self.cfg['far'] * self.cfg['sc_factor']).reshape(-1)
            weights[invalid] = 0

        if valid_samples is not None:
            weights[valid_samples == 0] = 0

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

        return weights

    def raw2outputs(self, raw, z_vals, valid_samples=None, depth=None):

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        sdf = raw[..., -1]

        weights = self.get_ray_sample_weights_from_depth(z_vals, depth=depth, pred_sdf=sdf, valid_samples=valid_samples)

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        return rgb_map, weights

    def get_weights_from_full_sdf(self, ori_sdf):
        trunc = self.cfg['sdf_weight_trunc']
        neg_clamp = self.cfg['sdf_weight_neg_clamp']
        sdf = torch.clamp_min(ori_sdf / trunc, -neg_clamp)
        weights = torch.sigmoid(sdf * self.cfg['sdf_lambda']) * torch.sigmoid(-sdf * self.cfg['sdf_lambda'])
        cut_off = self.cfg['sdf_weight_cutoff']
        if cut_off > 0:
            mask = ori_sdf < cut_off
            weights = weights * mask

        return weights

    def get_occ_from_full_sdf(self, sdf):
        thresh = self.cfg['sdf_to_occ_thresh']
        return torch.clamp_min(1 - torch.relu(sdf / thresh + 0.5), 0)   # SDF < -0.005 --> 1, SDF > 0.005 --> 0, [-0.005, 0.005] occ 1 --> 0

    def render_frame(self, cnc_name, rays, depth=None, lindisp=False, perturb=False):
        """Render rays in chunks
        """

        all_ret = []
        chunk = self.cfg['chunk']
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(cnc_name, rays[i:i + chunk],
                                   depth=None if depth is None else depth[i: i + chunk],
                                   lindisp=lindisp, perturb=perturb)
            all_ret.append({key: value for key, value in ret.items() if value is not None})

        def merge_list(l):
            if isinstance(l[0], dict):  # merge a list of dicts
                return {key: merge_list(list(x[key] for x in l)) for key in l[0]}
            elif isinstance(l[0], list):
                return [merge_list(list(x[i] for x in l)) for i in range(len(l[0]))]
            elif isinstance(l[0], torch.Tensor):
                return torch.cat(l, 0)

        all_ret = merge_list(all_ret)

        return all_ret

    def query_object_field(self, cnc_name, pts, valid_samples, viewdirs=None,
                           sdf_only=False, empty_slot_mask=None):
        embedded = torch.zeros((pts.shape[0], self.models[f'{cnc_name}_embed_fn'].out_dim), device=pts.device)
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            embedded[valid_samples] = self.models[f'{cnc_name}_embed_fn'](pts[valid_samples]).to(embedded.dtype)
        embedded = embedded.float()

        # Add view directions
        if not sdf_only and self.models[f'{cnc_name}_embeddirs_fn'] is not None:
            embedded_dirs = self.models[f'{cnc_name}_embeddirs_fn'](viewdirs)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs = []
        with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
            chunk = self.cfg['netchunk']
            for i in range(0, embedded.shape[0], chunk):
                if sdf_only:
                    out = self.models[f'{cnc_name}_model'].forward_sdf(embedded[i:i + chunk]).reshape(-1, 1)
                else:
                    out = self.models[f'{cnc_name}_model'](embedded[i:i + chunk])
                outputs.append(out)
        outputs = torch.cat(outputs, dim=0).float()

        if empty_slot_mask is not None:
            empty_outputs = torch.zeros_like(outputs)
            empty_outputs[..., -1] = 1
            empty_slot_mask = empty_slot_mask.float().reshape(-1, 1)

            outputs = (1 - empty_slot_mask) * outputs + empty_slot_mask * empty_outputs

        return outputs, valid_samples

    def run_network_corr(self, cnc_name, sample_dict, depth):

        canonical_pts = sample_dict['cnc_pts'].reshape(-1, 3)
        valid_samples = sample_dict['valid_samples']

        other_cnc_name = [name for name in self.cnc_timesteps if name != cnc_name][0]
        target_timesteps = torch.ones_like(canonical_pts[..., :1]) * self.cnc_timesteps[other_cnc_name]
        target_pts_dict = self.get_world_pts_from_canonical_pts(cnc_name, canonical_pts, target_timesteps,
                                                                valid_samples.reshape(-1))
        target_world_pts = target_pts_dict['world_pts']

        # per-point 3D correspondence --> sum over the ray

        weights = self.get_ray_sample_weights_from_depth(sample_dict['z_vals'],
                                                         depth=depth, valid_samples=sample_dict['valid_samples'])
        sumw = weights.sum(dim=-1)
        non_zero_weight = (sumw > 0)
        target_world_pts = (target_world_pts.reshape(len(weights), -1, 3) * weights.unsqueeze(-1)).sum(dim=-2)
        return {'target_world_pts': target_world_pts, 'valid': non_zero_weight}

    def run_network_render(self, cnc_name, cnc_pts, cnc_viewdirs, valid_samples=None):
        # only render the current frame; does not go through motion
        # cnc_pts: [#rays, #samples, 3], cnc_viewdirs: [#rays, #samples, 3]
        old_shape = cnc_pts.shape[:-1]

        outputs, cur_valid_samples = self.query_object_field(cnc_name, cnc_pts.reshape(-1, 3),
                                                                      valid_samples.reshape(-1),
                                                                      cnc_viewdirs.reshape(-1, 3))

        ret_dict = {'raw': outputs.reshape(old_shape + (outputs.shape[-1], )),
                    'valid_samples': cur_valid_samples.reshape(old_shape)}

        return ret_dict

    def run_network_density(self, cnc_name, inputs, timestep):
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])

        inputs_flat = torch.clip(inputs_flat, -1, 1)
        valid_samples = torch.ones((len(inputs_flat)), device=inputs.device).bool()
        empty_slot_mask = None

        if timestep != self.cnc_timesteps[cnc_name]:

            timesteps = torch.ones_like(inputs_flat[..., :1]) * timestep

            if not inputs_flat.requires_grad:
                inputs_flat.requires_grad = True

            input_pts = inputs_flat
            canonical_pts_dict = self.backward_flow(cnc_name, input_pts, timesteps, valid_samples, training=False)
            inputs_flat = canonical_pts_dict['canonical_pts']
            inputs_flat = torch.clip(inputs_flat, -1, 1)
            empty_slot_mask = torch.zeros_like(inputs_flat[..., 0]) if 'empty_slot_mask' not in canonical_pts_dict or canonical_pts_dict['empty_slot_mask'] is None else canonical_pts_dict['empty_slot_mask']

        outputs, valid_samples = self.query_object_field(cnc_name, inputs_flat, valid_samples,
                                                                  sdf_only=True,
                                                                  empty_slot_mask=empty_slot_mask)

        return outputs, valid_samples

    def run_network_canonical_info(self, cnc_name, inputs):

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        inputs_flat = torch.clip(inputs_flat, -1, 1)
        valid_samples = torch.ones((len(inputs_flat)), device=inputs.device).bool()

        canonical_pts = inputs_flat
        target_pts_dict = self.get_world_pts_from_canonical_pts(cnc_name, canonical_pts, torch.zeros_like(canonical_pts[..., :1]),
                                                                valid_samples.reshape(-1), training=False)

        ret = {key: target_pts_dict[key] for key in ['fwd_attn_hard']}

        return ret

    def export_canonical(self, cnc_name, per_part=True, eval=True):
        os.makedirs(pjoin(self.save_dir, 'results'), exist_ok=True)
        result_path = pjoin(self.save_dir, 'results', f'step_{self.global_step + 1:07d}')
        os.makedirs(result_path, exist_ok=True)

        with torch.no_grad():
            recon = self.extract_canonical_info(cnc_name, isolevel=0.0,
                                                voxel_size=self.cfg['mesh_resolution'], disable_octree=False,
                                                per_part=per_part)

            if per_part:
                all_mesh, part_meshes, part_rot, part_trans = recon
            else:
                all_mesh = recon

            nerf_scale = self.cfg['sc_factor']
            nerf_trans = self.cfg['translation'].reshape(1, 3)
            inv_scale = 1.0 / nerf_scale
            inv_trans = (-nerf_scale * nerf_trans).reshape(1, 3)

            if per_part:
                all_rot = part_rot.cpu().numpy().swapaxes(-1, -2)
                all_trans = part_trans.cpu().numpy()
                num_joints = len(all_rot) - 1

                all_trans = (all_trans + inv_trans - np.matmul(all_rot, inv_trans[..., np.newaxis])[..., 0]) * inv_scale

                pred_meshes = [pjoin(result_path, f'{cnc_name}_{suffix}.obj')
                               for suffix in ['all'] + [f'part_{i}' for i in range(num_joints + 1)]]
            else:
                pred_meshes = [pjoin(result_path, f'{cnc_name}_all.obj')]
                part_meshes = []

            gt_name = {'init': 'start', 'last': 'end'}[cnc_name]

            for mesh, mesh_path in zip([all_mesh] + part_meshes, pred_meshes):
                if mesh is not None:
                    vertices_raw = mesh.vertices.copy()
                    vertices_raw = inv_scale * (vertices_raw + inv_trans.reshape(1, 3))
                    ori_mesh = trimesh.Trimesh(vertices_raw, mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)

                    ori_mesh.export(mesh_path)

                    mesh_name = '_'.join(mesh_path.split('/')[-1].split('.')[0].split('_')[1:])
                    if mesh_name.startswith('part'):
                        part_i = int(mesh_name.split('_')[-1])
                        cur_rot, cur_trans = all_rot[part_i], all_trans[part_i]  # [3, 3] and [3]
                        vertices_raw = ori_mesh.vertices.copy()
                        vertices_raw = (np.matmul(cur_rot,
                                                  vertices_raw.transpose(1, 0)) + cur_trans.reshape(3, 1)).transpose(1, 0)
                        forward_mesh = trimesh.Trimesh(vertices_raw, ori_mesh.faces, process=False)
                        new_mesh_path = mesh_path.replace(mesh_name, f'{mesh_name}_forward')
                        forward_mesh.export(new_mesh_path)

            all_metric_dict = {}

            if not per_part:
                pred_w = pred_meshes[0]
                if self.gt_dict is not None and eval:
                    gt_w = self.gt_dict[f'mesh_{gt_name}']['w']
                    s, d_list, w = eval_CD(None, [], pred_w, None, [], gt_w)
                    all_metric_dict.update({'chamfer_whole': w * 1000})
                else:
                    cluster_meshes(None, [], pred_w)

            else:

                pred_w, pred_s, pred_d_list = pred_meshes[0], pred_meshes[1], pred_meshes[2:]
                if self.gt_dict is not None:
                    gt_w, gt_s, gt_d_list = list(self.gt_dict[f'mesh_{gt_name}'][key] for key in ['w', 's', 'd'])
                else:
                    eval = False
                    cluster_meshes(pred_s, pred_d_list, pred_w)

                base_rot, base_trans = all_rot[0], all_trans[0]   # only works for one-level below base

                all_perm = permutations(range(num_joints))

                if not eval:
                    all_perm = []
                    for joint_type in ['prismatic', 'revolute']:
                        joint_info_list = []
                        for pred_i in range(num_joints):
                            part_rot, part_trans = all_rot[pred_i + 1], all_trans[pred_i + 1]

                            joint_info, rel_rot, rel_trans = interpret_transforms(base_rot, base_trans, part_rot, part_trans,
                                                                                  joint_type=joint_type)
                            joint_info.update({'rotation': rel_rot, 'translation': rel_trans})
                            save_axis_mesh(joint_info['axis_direction'], joint_info['axis_position'],
                                           pjoin(result_path, f'{cnc_name}_axis_{pred_i}_{joint_type}.obj'))

                            joint_info_list.append(joint_info)

                        with open(pjoin(result_path, f'{cnc_name}_{joint_type}_motion.json'), 'w') as f:
                            json_joint_info = [
                                {key: value if not isinstance(value, np.ndarray) else value.tolist() for key, value in
                                 joint_info.items()} for joint_info in joint_info_list]
                            json.dump(json_joint_info, f)

                for p, perm in enumerate(all_perm):
                    perm_str = 'p' + ''.join(list(map(str, perm))) + '_'
                    if num_joints == 1:
                        perm_str = ''

                    joint_info_list = []
                    joint_metric = {}

                    for gt_i, pred_i in enumerate(perm):
                        joint_str = f'_{gt_i}' if len(perm) > 1 else ''
                        gt_joint = {key: value for key, value in self.gt_dict['joint'][gt_i].items()}
                        part_rot, part_trans = all_rot[pred_i + 1], all_trans[pred_i + 1]

                        joint_type = 'revolute' if gt_joint['joint_type'] == 'rotate' else 'prismatic'

                        joint_info, rel_rot, rel_trans = interpret_transforms(base_rot, base_trans, part_rot, part_trans, joint_type=joint_type)
                        joint_info.update({'rotation': rel_rot, 'translation': rel_trans})

                        joint_info_list.append(joint_info)

                        save_axis_mesh(joint_info['axis_direction'], joint_info['axis_position'],
                                       pjoin(result_path, f'{cnc_name}_axis_{pred_i}_{joint_type}.obj'))

                        angle, distance, theta_diff = eval_axis_and_state(joint_info, gt_joint, joint_type=joint_type,
                                                                          reverse=gt_name == 'end')
                        joint_metric.update({f'axis_angle{joint_str}': angle,
                                             f'axis_dist{joint_str}': distance * 10,
                                             f'theta_diff{joint_str}': theta_diff})

                    try:
                        s, d_list, w = eval_CD(pred_s, [pred_d_list[pred_i] for pred_i in perm], pred_w, gt_s, gt_d_list, gt_w)
                    except:
                        s, d_list, w = -1, [-1 for _ in range(num_joints)], -1

                    metric_dict = {}
                    metric_dict.update(joint_metric)
                    metric_dict.update({'chamfer_static': s * 1000})
                    if len(d_list) > 1:
                        metric_dict.update({f'chamfer_dynamic_{i}': d_list[i] * 1000 for i in range(len(d_list))})
                    else:
                        metric_dict['chamfer_dynamic'] = d_list[0] * 1000
                    metric_dict.update({'chamfer_whole': w * 1000})

                    with open(pjoin(result_path, f'{cnc_name}_{perm_str}motion.json'), 'w') as f:
                        json_joint_info = [{key: value if not isinstance(value, np.ndarray) else value.tolist() for key, value in
                                           joint_info.items()} for joint_info in joint_info_list]
                        json.dump(json_joint_info, f)

                    self.plot_loss({f'{cnc_name}_{perm_str}{key}': value for key, value in metric_dict.items()}, self.global_step)

                    with open(pjoin(result_path, f'{cnc_name}_{perm_str}metrics.json'), 'w') as f:
                        json.dump(metric_dict, f)

                    all_metric_dict.update({f'{perm_str}{key}': value for key, value in metric_dict.items()})

                    write_mode = 'w' if cnc_name == 'init' and p == 0 else 'a'
                    with open(pjoin(result_path, 'all_metrics'), write_mode) as f:
                        for metric in metric_dict:
                            metric_name = f'{cnc_name} {perm_str}{metric}'
                            print(f'{metric_name : >20}: {metric_dict[metric]:.6f}', file=f)

            msg = f"{cnc_name}, iter: {self.global_step}, "
            for k in all_metric_dict.keys():
                msg += f"{k}: {all_metric_dict[k]:.4f}, "
            msg += "\n"
            logging.info(msg)


    @torch.no_grad()
    def extract_canonical_info(self, cnc_name, voxel_size=0.003, isolevel=0.0, disable_octree=False, per_part=True):
        # Query network on dense 3d grid of points
        voxel_size *= self.cfg['sc_factor']  # in "network space"
        voxel_size = max(voxel_size, 0.004)

        bounds = np.array(self.cfg['bounding_box']).reshape(2, 3)
        x_min, x_max = bounds[0, 0], bounds[1, 0]
        y_min, y_max = bounds[0, 1], bounds[1, 1]
        z_min, z_max = bounds[0, 2], bounds[1, 2]
        tx = np.arange(x_min + 0.5 * voxel_size, x_max, voxel_size)
        ty = np.arange(y_min + 0.5 * voxel_size, y_max, voxel_size)
        tz = np.arange(z_min + 0.5 * voxel_size, z_max, voxel_size)
        N = len(tx)
        query_pts = torch.tensor(
            np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32).reshape(-1, 3)).float().cuda()

        if self.octree_m is not None and not disable_octree:
            vox_size = self.cfg['octree_raytracing_voxel_size'] * self.cfg['sc_factor']
            level = int(np.floor(np.log2(2.0 / vox_size)))

            chunk = 160000
            all_valid = []
            for i in range(0, query_pts.shape[0], chunk):
                cur_pts = query_pts[i: i + chunk]
                center_ids = self.octree_m.get_center_ids(cur_pts, level)
                valid = center_ids >= 0
                all_valid.append(valid)
            valid = torch.cat(all_valid, dim=0)
        else:
            valid = torch.ones(len(query_pts), dtype=bool).cuda()

        flat = query_pts[valid]

        sigma = []
        cnc_features, attn_hard = [], []
        chunk = self.cfg['netchunk']
        for i in range(0, flat.shape[0], chunk):
            inputs = flat[i:i + chunk]
            with torch.no_grad():
                outputs, valid_samplh_resos = self.run_network_density(cnc_name=cnc_name, inputs=inputs, timestep=self.cnc_timesteps[cnc_name])
                slot_info = self.run_network_canonical_info(cnc_name=cnc_name, inputs=inputs)
                attn_hard.append(slot_info['fwd_attn_hard'])
            sigma.append(outputs)
        sigma = torch.cat(sigma, dim=0)

        observed = []
        chunk = 120000

        for i in range(0, flat.shape[0], chunk):
            observed.append(self.depth_fuser[cnc_name].query(flat[i:i + chunk]))

        observed = torch.cat(observed, dim=0)
        sigma[~observed] = 1.

        sigma_ = torch.ones((N ** 3)).float()
        sigma_[valid.cpu()] = sigma.reshape(-1).cpu()
        sigma = sigma_.reshape(N, N, N).data.numpy()

        valid = valid.cpu().numpy()
        valid = np.where(valid > 0)[0]

        def get_mesh(sigma):

            from skimage import measure
            try:
                vertices, triangles, normals, values = measure.marching_cubes(sigma, isolevel)
            except Exception as e:
                logging.info(f"ERROR Marching Cubes {e}")
                return None

            # Rescale and translate
            voxel_size_ndc = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]]) / np.array(
                [[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])
            offset = np.array([tx[0], ty[0], tz[0]])
            vertices[:, :3] = voxel_size_ndc.reshape(1, 3) * vertices[:, :3] + offset.reshape(1, 3)

            # Create mesh
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            return mesh

        all_mesh = get_mesh(sigma)

        if not per_part:
            return all_mesh

        attn_hard = torch.cat(attn_hard, dim=0)
        rot, trans = self.models[f'{cnc_name}_deformation_model'].get_slot_motions()

        if all_mesh is not None:
            all_vertices = all_mesh.vertices
            flat = torch.tensor(all_vertices).float().cuda()

            chunk = self.cfg['netchunk']
            vert_attn_hard = []
            for i in range(0, flat.shape[0], chunk):
                inputs = flat[i:i + chunk]
                with torch.no_grad():
                    slot_info = self.run_network_canonical_info(cnc_name, inputs)
                    vert_attn_hard.append(slot_info['fwd_attn_hard'])
            vert_attn_hard = torch.cat(vert_attn_hard, dim=0)
            visual = np.zeros((len(vert_attn_hard), 3))
            color_list = np.array(((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.5, 0.5, 0.5)))
            for i in range(vert_attn_hard.shape[-1]):
                visual[torch.where(vert_attn_hard[:, i] == 1)[0].cpu().numpy()] = color_list[i]
            visual = trimesh.visual.ColorVisuals(mesh=all_mesh, face_colors=None,
                                                 vertex_colors=(visual * 255).astype(np.uint8))
            all_mesh.visual = visual

        part_meshes = []
        for i in range(attn_hard.shape[-1]):
            part_sigma = np.ones_like(sigma.reshape(-1))
            idx = torch.where(attn_hard[:, i] == 1)[0].cpu().numpy()
            part_sigma[valid[idx]] = sigma.reshape(-1)[valid[idx]]
            part_meshes.append(get_mesh(part_sigma.reshape(sigma.shape)))

        return all_mesh, part_meshes, rot, trans

    def render_images(self, cnc_name, img_i, cur_rays=None):
        if cur_rays is None:
            frame_ids = self.rays_dict[cnc_name][:, self.ray_frame_id_slice].cuda()
            cur_rays = self.rays_dict[cnc_name][frame_ids == img_i].cuda()
        gt_depth = cur_rays[:, self.ray_depth_slice]
        gt_rgb = cur_rays[:, self.ray_rgb_slice].cpu()
        ray_type = cur_rays[:, self.ray_type_slice].data.cpu().numpy()

        ori_chunk = self.cfg['chunk']
        self.cfg['chunk'] = copy.deepcopy(self.cfg['render_chunk'])
        with torch.no_grad():
            extras = self.render_frame(cnc_name=cnc_name, rays=cur_rays, lindisp=False, perturb=False,
                                       depth=gt_depth)
        self.cfg['chunk'] = ori_chunk

        sdf = extras['raw'][..., -1]  # full_sdf: will just use the partial one learned with color
        z_vals = extras['z_vals']
        signs = sdf[:, 1:] * sdf[:, :-1]
        empty_rays = (signs > 0).all(dim=-1)
        mask = signs < 0
        inds = torch.argmax(mask.float(), axis=1)
        inds = inds[..., None]
        depth = torch.gather(z_vals, dim=1, index=inds)
        depth[empty_rays] = self.cfg['far'] * self.cfg['sc_factor']
        depth = depth[..., None].data.cpu().numpy()

        rgb = extras['rgb_map']
        rgb = rgb.data.cpu().numpy()

        rgb_full = np.zeros((self.H, self.W, 3), dtype=float)
        depth_full = np.zeros((self.H, self.W), dtype=float)
        ray_mask_full = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        X = cur_rays[:, self.ray_dir_slice].data.cpu().numpy()
        X[:, [1, 2]] = -X[:, [1, 2]]
        projected = (self.K @ X.T).T
        uvs = projected / projected[:, 2].reshape(-1, 1)
        uvs = uvs.round().astype(int)
        uvs_good = uvs[ray_type == 0]
        ray_mask_full[uvs_good[:, 1], uvs_good[:, 0]] = [255, 0, 0]
        uvs_uncertain = uvs[ray_type == 1]
        ray_mask_full[uvs_uncertain[:, 1], uvs_uncertain[:, 0]] = [0, 255, 0]
        rgb_full[uvs[:, 1], uvs[:, 0]] = rgb.reshape(-1, 3)
        depth_full[uvs[:, 1], uvs[:, 0]] = depth.reshape(-1)
        gt_rgb_full = np.zeros((self.H, self.W, 3), dtype=float)
        gt_rgb_full[uvs[:, 1], uvs[:, 0]] = gt_rgb.reshape(-1, 3).data.cpu().numpy()
        gt_depth_full = np.zeros((self.H, self.W), dtype=float)
        gt_depth_full[uvs[:, 1], uvs[:, 0]] = gt_depth.reshape(-1).data.cpu().numpy()

        return rgb_full, depth_full, ray_mask_full, gt_rgb_full, gt_depth_full, extras
