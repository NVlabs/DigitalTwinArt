# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import open3d as o3d
import numpy as np
import ruamel.yaml
import os
import logging
import copy
import joblib
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN

os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_sdf
import torch.nn.functional as F
import time
import trimesh

yaml = ruamel.yaml.YAML()
try:
    import kaolin
except Exception as e:
    print(f"Import kaolin failed, {e}")
try:
    from mycuda import common
except:
    pass

glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth, K):
    invalid_mask = (depth < 0.1)
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)


def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def transform_pts(pts, tf):
    """Transform 2d or 3d points
    @pts: (...,3)
    """
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


class OctreeManager:
    def __init__(self, pts=None, max_level=None, octree=None):
        if octree is None:
            pts_quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=max_level)
            self.octree = kaolin.ops.spc.unbatched_points_to_octree(pts_quantized, max_level, sorted=False)
        else:
            self.octree = octree
        lengths = torch.tensor([len(self.octree)], dtype=torch.int32).cpu()
        self.max_level, self.pyramids, self.exsum = kaolin.ops.spc.scan_octrees(self.octree, lengths)
        self.n_level = self.max_level + 1
        self.point_hierarchies = kaolin.ops.spc.generate_points(self.octree, self.pyramids, self.exsum)
        self.point_hierarchy_dual, self.pyramid_dual = kaolin.ops.spc.unbatched_make_dual(self.point_hierarchies,
                                                                                          self.pyramids[0])
        self.trinkets, self.pointers_to_parent = kaolin.ops.spc.unbatched_make_trinkets(self.point_hierarchies,
                                                                                        self.pyramids[0],
                                                                                        self.point_hierarchy_dual,
                                                                                        self.pyramid_dual)
        self.n_vox = len(self.point_hierarchies)
        self.n_corners = len(self.point_hierarchy_dual)

    def get_level_corner_quantized_points(self, level):
        start = self.pyramid_dual[..., 1, level]
        num = self.pyramid_dual[..., 0, level]
        return self.point_hierarchy_dual[start:start + num]

    def get_level_quantized_points(self, level):
        start = self.pyramids[..., 1, level]
        num = self.pyramids[..., 0, level]
        return self.pyramids[start:start + num]

    def get_trilinear_coeffs(self, x, level):
        quantized = kaolin.ops.spc.quantize_points(x, level)
        coeffs = kaolin.ops.spc.coords_to_trilinear_coeffs(x, quantized, level)  # (N,8)
        return coeffs

    def get_center_ids(self, x, level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        return pidx

    def get_corners_ids(self, x, level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        corner_ids = self.trinkets[pidx]
        is_valid = torch.ones(len(x)).bool().to(x.device)
        bad_ids = (pidx < 0).nonzero()[:, 0]
        is_valid[bad_ids] = 0

        return corner_ids, is_valid

    def trilinear_interpolate(self, x, level, feat):
        '''
        @feat: (N_feature of current level, D)
        '''
        ############!NOTE direct API call cannot back prop well
        # pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        # x = x.unsqueeze(0)
        # interpolated = kaolin.ops.spc.unbatched_interpolate_trilinear(coords=x,pidx=pidx.int(),point_hierarchy=self.point_hierarchies,trinkets=self.trinkets, feats=feat, level=level)[0]
        ##################

        coeffs = self.get_trilinear_coeffs(x, level)  # (N,8)
        corner_ids, is_valid = self.get_corners_ids(x, level)
        # if corner_ids.max()>=feat.shape[0]:
        #     pdb.set_trace()

        corner_feat = feat[corner_ids[is_valid].long()]  # (N,8,D)
        out = torch.zeros((len(x), feat.shape[-1]), device=x.device).float()
        out[is_valid] = torch.sum(coeffs[..., None][is_valid] * corner_feat, dim=1)  # (N,D)

        # corner_feat = feat[corner_ids.long()]   #(N,8,D)
        # out = torch.sum(coeffs[...,None]*corner_feat, dim=1)   #(N,D)

        return out, is_valid

    def draw_boxes(self, level, outfile='/home/bowen/debug/corners.ply'):
        centers = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchies.reshape(-1, 3),
                                                            self.pyramids.reshape(2, -1), level)
        pts = (centers.float() + 0.5) / (2 ** level) * 2 - 1  # Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile.replace("corners", "centers"), pcd)

        corners = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchy_dual, self.pyramid_dual, level)
        pts = corners.float() / (2 ** level) * 2 - 1  # Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile, pcd)

    def ray_trace(self, rays_o, rays_d, level, debug=False):
        """Octree is in normalized [-1,1] world coordinate frame
        'rays_o': ray origin in normalized world coordinate system
        'rays_d': (N,3) unit length ray direction in normalized world coordinate system
        'octree': spc
        @voxel_size: in the scale of [-1,1] space
        Return:
            ray_depths_in_out: traveling times, NOT the Z value
        """
        from mycuda import common

        # Avoid corner cases. issuse in kaolin: https://github.com/NVIDIAGameWorks/kaolin/issues/490 and https://github.com/NVIDIAGameWorks/kaolin/pull/634
        # rays_o = rays_o.clone() + 1e-7

        ray_index, rays_pid, depth_in_out = kaolin.render.spc.unbatched_raytrace(self.octree, self.point_hierarchies,
                                                                                 self.pyramids[0], self.exsum, rays_o,
                                                                                 rays_d, level=level, return_depth=True,
                                                                                 with_exit=True)
        if ray_index.size()[0] == 0:
            ray_depths_in_out = torch.zeros((rays_o.shape[0], 1, 2))
            rays_pid = -torch.ones_like(rays_o[:, :1])
            rays_near = torch.zeros_like(rays_o[:, :1])
            rays_far = torch.zeros_like(rays_o[:, :1])
            return rays_near, rays_far, rays_pid, ray_depths_in_out

        intersected_ray_ids, counts = torch.unique_consecutive(ray_index, return_counts=True)
        max_intersections = counts.max().item()
        start_poss = torch.cat([torch.tensor([0], device=counts.device), torch.cumsum(counts[:-1], dim=0)], dim=0)

        ray_depths_in_out = common.postprocessOctreeRayTracing(ray_index.long().contiguous(), depth_in_out.contiguous(),
                                                               intersected_ray_ids.long().contiguous(),
                                                               start_poss.long().contiguous(), max_intersections,
                                                               rays_o.shape[0])

        rays_far = ray_depths_in_out[:, :, 1].max(dim=-1)[0].reshape(-1, 1)
        rays_near = ray_depths_in_out[:, 0, 0].reshape(-1, 1)

        return rays_near, rays_far, rays_pid, ray_depths_in_out


def find_biggest_cluster(pts, eps=0.005, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(pts)
    ids, cnts = np.unique(dbscan.labels_, return_counts=True)
    best_id = ids[cnts.argsort()[-1]]
    keep_mask = dbscan.labels_ == best_id
    pts_cluster = pts[keep_mask]
    return pts_cluster, keep_mask


def compute_translation_scales(pts, max_dim=2, cluster=True, eps=0.005, min_samples=5):
    if cluster:
        pts, keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    max_xyz = pts.max(axis=0)
    min_xyz = pts.min(axis=0)
    center = (max_xyz + min_xyz) / 2
    sc_factor = max_dim / (max_xyz - min_xyz).max()  # Normalize to [-1,1]
    sc_factor *= 0.9
    translation_cvcam = -center
    return translation_cvcam, sc_factor, keep_mask


def compute_scene_bounds_worker(color_file, K, glcam_in_world, use_mask, rgb=None, depth=None, mask=None):
    if rgb is None:
        depth_file = color_file.replace('images', 'depth_filtered')
        mask_file = color_file.replace('images', 'masks')
        rgb = np.array(Image.open(color_file))[..., :3]
        depth = cv2.imread(depth_file, -1) / 1e3
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    if use_mask:
        if mask is None:
            mask = cv2.imread(mask_file, -1)
        valid = valid & (mask > 0)
    pts = xyz_map[valid].reshape(-1, 3)
    if len(pts) == 0:
        return None
    colors = rgb[valid].reshape(-1, 3)

    pcd = toOpen3dCloud(pts, colors)

    pcd = pcd.voxel_down_sample(0.01)
    new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cam_in_world = glcam_in_world @ glcam_in_cvcam
    new_pcd.transform(cam_in_world)

    return np.asarray(new_pcd.points).copy(), np.asarray(new_pcd.colors).copy()


def compute_scene_bounds(color_files, glcam_in_worlds, K, use_mask=True, base_dir=None, rgbs=None, depths=None,
                         masks=None, cluster=True, translation_cvcam=None, sc_factor=None, eps=0.06, min_samples=1):
    # assert color_files is None or rgbs is None

    if base_dir is None:
        base_dir = os.path.dirname(color_files[0]) + '/../'
    os.makedirs(base_dir, exist_ok=True)

    args = []
    if rgbs is not None:
        for i in range(len(rgbs)):
            args.append((color_files[i], K, glcam_in_worlds[i], use_mask, rgbs[i], depths[i], masks[i]))
    else:
        for i in range(len(color_files)):
            args.append((color_files[i], K, glcam_in_worlds[i], use_mask))

    logging.info(f"compute_scene_bounds_worker start")
    ret = joblib.Parallel(n_jobs=6, prefer="threads")(joblib.delayed(compute_scene_bounds_worker)(*arg) for arg in args)
    logging.info(f"compute_scene_bounds_worker done")

    pcd_all = None
    for r in ret:
        if r is None or len(r[0]) == 0:
            continue
        if pcd_all is None:
            pcd_all = toOpen3dCloud(r[0], r[1])
        else:
            pcd_all += toOpen3dCloud(r[0], r[1])

    pcd = pcd_all.voxel_down_sample(eps / 5)

    pts = np.asarray(pcd.points).copy()

    def make_tf(translation_cvcam, sc_factor):
        tf = np.eye(4)
        tf[:3, 3] = translation_cvcam
        tf1 = np.eye(4)
        tf1[:3, :3] *= sc_factor
        tf = tf1 @ tf
        return tf

    if translation_cvcam is None:
        translation_cvcam, sc_factor, keep_mask = compute_translation_scales(pts, cluster=cluster, eps=eps,
                                                                             min_samples=min_samples)
        tf = make_tf(translation_cvcam, sc_factor)
    else:
        tf = make_tf(translation_cvcam, sc_factor)
        tmp = copy.deepcopy(pcd)
        tmp.transform(tf)
        tmp_pts = np.asarray(tmp.points)
        keep_mask = (np.abs(tmp_pts) < 1).all(axis=-1)

    logging.info(f"compute_translation_scales done")

    pcd = toOpen3dCloud(pts[keep_mask], np.asarray(pcd.colors)[keep_mask])
    pcd_real_scale = copy.deepcopy(pcd)

    with open(f'{base_dir}/normalization.yml', 'w') as ff:
        tmp = {
            'translation_cvcam': translation_cvcam.tolist(),
            'sc_factor': float(sc_factor),
        }
        yaml.dump(tmp, ff)

    pcd.transform(tf)

    return sc_factor, translation_cvcam, pcd_real_scale, pcd


BAD_DEPTH = 99
BAD_COLOR = 128


def mask_and_normalize_data(rgbs, depths, masks, poses, sc_factor, translation):
    '''
    @rgbs: np array (N,H,W,3)
    @depths: (N,H,W)
    @masks: (N,H,W)
    @normal_maps: (N,H,W,3)
    @poses: (N,4,4)
    '''
    depths[depths < 0.1] = BAD_DEPTH
    if masks is not None:
        rgbs[masks == 0] = BAD_COLOR
        depths[masks == 0] = BAD_DEPTH
        masks = masks[..., None]

    rgbs = (rgbs / 255.0).astype(np.float32)
    depths *= sc_factor
    depths = depths[..., None]
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    return rgbs, depths, masks, poses


def get_voxel_pts(voxel_size):
    one_dim = np.linspace(-1, 1, int(np.ceil(2.0 / voxel_size)) + 1)
    x, y, z = np.meshgrid(one_dim, one_dim, one_dim)  # [N, N, N]
    pts = np.stack([x, y, z], axis=-1)  # [N, N, N, 3]
    return pts


def sdf_voxel_from_mesh(mesh, voxel_size):  # mesh is already scaled to be in [-1, 1]
    pts = get_voxel_pts(voxel_size)

    t = time.time()
    sdf = mesh_to_sdf(mesh, pts.reshape(-1, 3), sign_method='depth')  # , sign_method='depth')
    sdf = sdf.reshape(pts.shape[:-1])

    return pts, sdf


class VoxelSDF:
    def __init__(self, sdf):  # sdf is by default from a grid [-1, 1]^3
        self.sdf_grid = torch.FloatTensor(sdf).unsqueeze(0).unsqueeze(0).cuda()

    def query(self, xyz):
        zxy = xyz[..., [2, 0, 1]]
        sdf = F.grid_sample(self.sdf_grid, zxy.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                            mode='bilinear', padding_mode='border', align_corners=False)
        return sdf.reshape(-1)


def extract_mesh(voxel_sdf, voxel_size=0.0099, isolevel=0):
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max = -1, 1
    tx = np.arange(x_min + 0.5 * voxel_size, x_max, voxel_size)
    ty = np.arange(y_min + 0.5 * voxel_size, y_max, voxel_size)
    tz = np.arange(z_min + 0.5 * voxel_size, z_max, voxel_size)
    N = len(tx)
    query_pts = torch.tensor(
        np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32).reshape(-1, 3)).float().cuda()

    sigma = voxel_sdf.query(query_pts.reshape(-1, 3)).reshape(N, N, N).data.cpu().numpy()

    from skimage import measure
    try:
        vertices, triangles, normals, values = measure.marching_cubes(sigma, isolevel)
    except Exception as e:
        print(f"ERROR Marching Cubes {e}")
        return None

    # Rescale and translate
    voxel_size_ndc = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]]) / np.array(
        [[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = voxel_size_ndc.reshape(1, 3) * vertices[:, :3] + offset.reshape(1, 3)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh


def transform_mesh_to_world(mesh, sc_factor, translation):
    nerf_scale = sc_factor
    nerf_trans = translation
    inv_scale = 1.0 / nerf_scale
    inv_trans = (-nerf_scale * nerf_trans).reshape(1, 3)

    vertices_raw = mesh.vertices.copy()
    vertices_raw = inv_scale * (vertices_raw + inv_trans.reshape(1, 3))
    ori_mesh = trimesh.Trimesh(vertices_raw, mesh.faces, process=False)
    return ori_mesh


class DepthFuser:
    def __init__(self, K, c2w, depth, mask, trunc, near=0.01, far=10):
        self.w2c = torch.linalg.inv(c2w)  # [V, 4, 4], w2c
        self.depth = torch.tensor(depth).to(K.device)  # [V, H, W]
        self.mask = torch.tensor(mask).to(K.device)
        self.near = near
        self.far = far
        self.K = K
        self.trunc = trunc
        self.V, self.H, self.W = depth.shape

    def query(self, pts):  # pts [N, 3]
        with torch.no_grad():
            cam_pts = torch.matmul(self.w2c[:, :3, :3].unsqueeze(0), pts.unsqueeze(1).unsqueeze(-1)).squeeze(
                -1)  # [N, V, 3]
            cam_pts = cam_pts + self.w2c[:, :3, 3].unsqueeze(0)  # [N, V, 3]

            cam_depth = -cam_pts[..., 2]

            projection = torch.matmul(self.K[:2, :2].unsqueeze(0).unsqueeze(0),  # [1, 1, 2, 2]
                                      (cam_pts[..., :2] / torch.clip(-cam_pts[..., 2:3], min=1e-8)).unsqueeze(
                                          -1)).squeeze(-1)
            projection = projection + self.K[:2, 2].unsqueeze(0).unsqueeze(0)  # [N, V, 2]

            pixel = torch.round(projection).long()

            valid_pixel = torch.logical_and(
                torch.logical_and(pixel[..., 0] >= 0, pixel[..., 0] < self.W),
                torch.logical_and(pixel[..., 1] >= 0, pixel[..., 1] < self.H))

            py = self.H - 1 - torch.clamp(pixel[..., 1], 0, self.H - 1)
            px = torch.clamp(pixel[..., 0], 0, self.W - 1)

            view_idx = torch.arange(0, self.V).long().to(px.device).reshape(1, -1)
            depth = self.depth[view_idx, py, px]
            mask = self.mask[view_idx, py, px]

            valid_depth = torch.logical_and(depth > self.near, depth < self.far)

            before_depth = cam_depth <= depth + self.trunc

            valid = torch.logical_and(torch.logical_and(valid_pixel, mask), valid_depth)

            observed = torch.logical_and(valid, before_depth)  # [N, V]
            observed = torch.any(observed, dim=1)  # [N]

        return observed


class VoxelVisibility:
    def __init__(self, visibility):
        self.grid = torch.FloatTensor(visibility).unsqueeze(0).unsqueeze(0).cuda()

    def query(self, xyz):
        zxy = xyz[..., [2, 0, 1]]
        visibility = F.grid_sample(self.grid, zxy.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                   mode='nearest', padding_mode='zeros', align_corners=False)
        return visibility.bool().reshape(-1)
