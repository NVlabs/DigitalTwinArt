# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply, load_obj, save_obj
from pytorch3d.structures import Meshes
import torch
import open3d as o3d
from os.path import join as pjoin
import yaml
import numpy as np
import copy

def combine_pred_mesh(paths, exp_path):
    recon_mesh = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        recon_mesh += mesh
    o3d.io.write_triangle_mesh(exp_path, recon_mesh)


def compute_chamfer(recon_pts,gt_pts):
    with torch.no_grad():
        recon_pts = recon_pts.cuda()
        gt_pts = gt_pts.cuda()
        dist, _ = chamfer_distance(recon_pts, gt_pts, batch_reduction=None)
        dist = dist.item()
    return dist


def get_cluster(verts, faces, thresh=0.1):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_1 = copy.deepcopy(mesh)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    largest_cluster_size = cluster_n_triangles.max()

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < largest_cluster_size * thresh
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    # o3d.visualization.draw_geometries([mesh_1])
    return torch.tensor(np.asarray(mesh_1.vertices)).float(), torch.tensor(np.asarray(mesh_1.triangles)).long()


def load_mesh(path):
    if path.endswith('.ply'):
        verts, faces = load_ply(path)
    elif path.endswith('.obj'):
        obj = load_obj(path)
        verts = obj[0]
        faces = obj[1].verts_idx
    return verts, faces


def compute_recon_error(recon_path, gt_path, n_samples=10000, vis=False, recon_transform=None,
                        use_cluster=True):
    try:
        verts, faces = load_mesh(recon_path)
        if use_cluster:
            verts, faces = get_cluster(verts, faces)
            upd_path = f"{'.'.join(recon_path.split('.')[:-1])}_clustered.{recon_path.split('.')[-1]}"
            save_obj(upd_path, verts=verts, faces=faces)
        if recon_transform is not None:
            verts = recon_transform['scale'] * (verts + recon_transform['trans'].reshape(1, 3))
            verts = verts.float()
        recon_mesh = Meshes(verts=[verts], faces=[faces])

        verts, faces = load_ply(gt_path)
        gt_mesh = Meshes(verts=[verts], faces=[faces])

        gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
        recon_pts = sample_points_from_meshes(recon_mesh, num_samples=n_samples)

        if vis:
            pts = gt_pts.clone().detach().squeeze().numpy()
            gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            o3d.io.write_point_cloud("gt_points.ply", gt_pcd)
            pts = recon_pts.clone().detach().squeeze().numpy()
            recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            o3d.io.write_point_cloud("recon_points.ply", recon_pcd)

        return (compute_chamfer(recon_pts, gt_pts) + compute_chamfer(gt_pts, recon_pts)) * 0.5
    except:
        return -1


def compute_point_to_mesh_error(recon_pts, gt_path, n_samples=10000):
    verts, faces = load_ply(gt_path)
    gt_mesh = Meshes(verts=[verts], faces=[faces])

    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    return (compute_chamfer(recon_pts, gt_pts) + compute_chamfer(gt_pts, recon_pts)) * 0.5


def eval_CD(pred_s_ply, pred_d_ply_list, pred_w_ply, gt_s_ply, gt_d_ply_list, gt_w_ply):
    # combine the part meshes as a whole
    if pred_w_ply is None:
        pred_w_ply = 'temp_pred_w.ply'
        combine_pred_mesh([pred_s_ply] + pred_d_ply_list, pred_w_ply)

    # compute synmetric distance
    chamfer_dist_s = compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False)
    chamfer_dist_w = compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False)
    chamfer_dist_d_list = [compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False)
                           for pred_d_ply, gt_d_ply in zip(pred_d_ply_list, gt_d_ply_list)]

    return chamfer_dist_s, chamfer_dist_d_list, chamfer_dist_w


def cluster_meshes(pred_s_ply, pred_d_ply_list, pred_w_ply):
    for recon_path in [pred_s_ply, pred_w_ply] + pred_d_ply_list:
        try:
            verts, faces = load_mesh(recon_path)
            verts, faces = get_cluster(verts, faces)
            upd_path = f"{'.'.join(recon_path.split('.')[:-1])}_clustered.{recon_path.split('.')[-1]}"
            save_obj(upd_path, verts=verts, faces=faces)
        except:
            continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, default=None)
    parser.add_argument('--gt_s', type=str)
    parser.add_argument('--gt_d', type=str)
    parser.add_argument('--gt_w', type=str)
    parser.add_argument('--pred_folder', type=str, default=None)
    parser.add_argument('--pred_s', type=str)
    parser.add_argument('--pred_d', type=str)
    parser.add_argument('--pred_w', type=str)
    parser.add_argument('--pred_cfg', type=str, default=None)

    return parser.parse_args()


def main(args):
    def add_prefix(path, prefix):
        if path is None:
            return None
        if prefix is not None:
            return pjoin(prefix, path)
    pred_s, pred_d, pred_w = map(lambda x: add_prefix(x, args.pred_folder), [args.pred_s, args.pred_d, args.pred_w])
    gt_s, gt_d, gt_w = map(lambda x: add_prefix(x, args.gt_folder), [args.gt_s, args.gt_d, args.gt_w])

    if args.pred_cfg is not None:
        cfg = yaml.safe_load(open(args.pred_cfg, 'r'))
        nerf_scale = cfg['sc_factor']
        nerf_trans = np.array(cfg['translation'])
        inv_scale = 1.0 / nerf_scale
        inv_trans = (-nerf_scale * nerf_trans).reshape(1, 3)
        pred_transform = {'trans': inv_trans, 'scale': inv_scale}
    else:
        pred_transform = None

    s, d, w = eval_CD(pred_s, pred_d, pred_w, gt_s, gt_d, gt_w, pred_transform)
    print('chamfer static', s)
    print('chamfer dynamic', d)
    print('chamfer whole', w)


if __name__ == '__main__':
    args = parse_args()
    main(args)