# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import torch.nn.functional as F
from mycuda.torch_ngp_grid_encoder.grid import GridEncoder


class SHEncoder(nn.Module):
    '''Spherical encoding
    '''

    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


class FeatureVolume(nn.Module):
    def __init__(self, out_dim, res, num_dim=3):
        super().__init__()
        self.grid = torch.nn.Parameter(torch.zeros([1, out_dim] + [res] * num_dim,
                                                   dtype=torch.float32))
        self.out_dim = out_dim

    def forward(self, pts):
        feat = F.grid_sample(self.grid, pts[None, None, None, :, :], mode='bilinear',
                             align_corners=True)  # [1, C, 1, 1, N]
        return feat[0, :, 0, 0, :].permute(1, 0)  # [N, C]


class NeRFSmall(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=64, geo_feat_dim=15, num_layers_color=4, hidden_dim_color=64,
                 input_ch=3, input_ch_views=3):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l != num_layers - 1:
                sigma_net.append(nn.ReLU(inplace=True))

        self.sigma_net = nn.Sequential(*sigma_net)
        torch.nn.init.constant_(self.sigma_net[-1].bias, -1.)  # Encourage last layer predict positive SDF

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l != num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        self.color_net = nn.Sequential(*color_net)

    def forward_sdf(self, x):
        '''
        @x: embedded positions
        '''
        h = self.sigma_net(x)
        sigma, geo_feat = h[..., 0], h[..., 1:]
        return sigma

    def forward(self, x):
        x = x.float()
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        h = self.sigma_net(h)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        color = self.color_net(h)

        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


def rotat_from_6d(ortho6d):
    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if (return_mag == True):
            return v, v_mag[:, 0]
        else:
            return v

    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
        return out

    x_raw = ortho6d[:, 0:3]  # batch*3  100
    y_raw = ortho6d[:, 3:6]  # batch*3
    x = normalize_vector(x_raw)  # batch*3  100
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


class GumbelAttn:
    def __init__(self, device, dtype, tau):
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=device, dtype=dtype),
            torch.tensor(1., device=device, dtype=dtype))
        self.tau = tau

    def process_dots(self, dots, training=True):
        if training:
            gumbels = self.gumbel_dist.sample(dots.shape)
            # gumbels = (torch.log(dots.softmax(dim=1) + 1e-7) + gumbels) / self.tau  # ~Gumbel(logits,tau)
            gumbels = (dots + gumbels) / self.tau
        else:
            gumbels = dots
        return gumbels


def compute_attn(logits, style, gumbel_module, training=True):
    raw_attn = logits.softmax(dim=1)

    if 'gumbel' in style and training:
        gumbels = gumbel_module.process_dots(logits, training=training)
        attn = gumbels.softmax(dim=1)  # [1, S, N]
    else:
        attn = raw_attn

    if 'soft' in style and training:
        return attn, raw_attn
    else:
        index = attn.max(dim=1, keepdim=True)[1]
        y_hard = torch.zeros_like(attn, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        attn_hard = y_hard - attn.detach() + attn
        return attn_hard, raw_attn


class PartArticulationNet(torch.nn.Module):
    def __init__(self, device, feat_dim=20, num_layers=3, hidden_dim=64,
                 slot_name='motion_xyz', slot_num=16, slot_hard='hard',
                 gt_transform=None, inv_transform=None, fix_base=True, gt_joint_types=None):
        super(PartArticulationNet, self).__init__()
        self.device = device

        self.slot_name = slot_name
        self.slot_num = slot_num
        self.slot_hard = slot_hard

        self.gumbel_module = GumbelAttn(device=device, dtype=torch.float, tau=1.0)

        # classification network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if 'xyz' in slot_name:
            feat_dim += 3

        self.net = self.create_layer(feat_dim, hidden_dim, num_layers, slot_num)
        self.net.to(device)

        self.gt_joint_types = gt_joint_types   # will use the same parameters, but interpret them based on the joint types

        self.rotation = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).unsqueeze(0).repeat(slot_num, 1))
        self.translation = nn.Parameter(torch.zeros(slot_num, 3))
        self.inv_transform = inv_transform

        self.gt_transform = gt_transform

        self.fix_base = fix_base

    def create_layer(self, dimin, width, layers, dimout):
        if layers == 1:
            return nn.Sequential(nn.Linear(dimin, dimout))
        else:
            return nn.Sequential(
                nn.Linear(dimin, width), nn.ReLU(inplace=True),
                *[nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True)) for _ in range(layers - 2)],
                nn.Linear(width, dimout), )

    def get_raw_slot_transform(self):   # note that the output R is transposed!!
        if self.inv_transform is None:
            rotat_back = self.rotation
            trans_back = self.translation

            if self.fix_base:
                rotat_static = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=rotat_back.dtype,
                                             device=rotat_back.device).reshape(1, 6)
                trans_static = torch.tensor([0, 0, 0], dtype=trans_back.dtype, device=trans_back.device).reshape(1, 3)

                if self.gt_transform is None:
                    rotat_other = rotat_back[1:]
                    trans_other = trans_back[1:]
                else:
                    rotat_other = torch.tensor(self.gt_transform['rot'], dtype=rotat_back.dtype,
                                               device=rotat_back.device)[:2].reshape(1, 6)
                    trans_other = torch.tensor(self.gt_transform['trans'], dtype=trans_back.dtype,
                                               device=trans_back.device).reshape(1, 3)

                rotat_all, trans_all = torch.cat([rotat_static, rotat_other], dim=0), \
                                       torch.cat([trans_static, trans_other], dim=0)
            else:
                rotat_all, trans_all = rotat_back, trans_back

            rotat_all = rotat_from_6d(rotat_all)

            if self.gt_joint_types is not None:
                revolute_rotation = rotat_all

                mat = torch.eye(3).reshape(1, 3, 3).to(rotat_all.device) - rotat_all
                U, S, Vh = torch.linalg.svd(mat.detach())
                U = U * (S.unsqueeze(-2) > 0)
                revolute_translation = torch.matmul(U, torch.matmul(U.transpose(-1, -2),
                                                                    trans_all.unsqueeze(-1))).squeeze(-1)
                prismatic_rotation = torch.eye(3).reshape(1, 3, 3).repeat(self.slot_num, 1, 1).to(rotat_all.device)
                prismatic_translation = trans_all

                revolute_mask = torch.tensor([joint_type == 'revolute' for joint_type in self.gt_joint_types],
                                             dtype=revolute_rotation.dtype, device=revolute_rotation.device)

                rotat_all = revolute_mask.reshape(-1, 1, 1) * revolute_rotation + \
                            (1 - revolute_mask.reshape(-1, 1, 1)) * prismatic_rotation
                trans_all = revolute_mask.reshape(-1, 1) * revolute_translation + \
                            (1 - revolute_mask.reshape(-1, 1)) * prismatic_translation

            return rotat_all, trans_all
        else:
            inv_rot, inv_trans = self.inv_transform()  # [S, 3, 3], [S, 3]
            rot = inv_rot.transpose(-1, -2)
            trans = -torch.matmul(inv_rot, inv_trans.unsqueeze(-1)).squeeze(-1)
            return rot, trans

    def back_deform(self, xyz_smp, xyz_smp_embedded, training=True):
        num_points = len(xyz_smp)
        slot_rotat, slot_trans = self.get_raw_slot_transform()  # [S, xx]
        slot_rotat = slot_rotat.unsqueeze(0).repeat(num_points, 1, 1, 1)
        xyz = xyz_smp.unsqueeze(1).repeat(1, self.slot_num, 1)  # [N, S, 3]
        xyz_cnc = torch.einsum('nscd,nsde->nsce', slot_rotat, (xyz - slot_trans).unsqueeze(-1)).squeeze(-1)  # [N, S, 3]
        return {'xyz_cnc': xyz_cnc, 'inv': True}

    def forw_attn(self, xyz_cnc, xyz_cnc_embedded, training=True):
        feat_forw = xyz_cnc_embedded
        if 'xyz' in self.slot_name:
            feat_forw = torch.cat([feat_forw, xyz_cnc], dim=-1)
        logits = self.net(feat_forw)
        attn_hard, attn_soft = compute_attn(logits, self.slot_hard, self.gumbel_module, training=training)
        attn_hard = attn_hard
        attn_soft = attn_soft
        return attn_hard, attn_soft

    def forw_deform(self, xyz_cnc, xyz_cnc_embedded, training=True, gt_attn=None):
        attn_hard, attn_soft = self.forw_attn(xyz_cnc, xyz_cnc_embedded, training=training)
        if gt_attn is not None:
            attn_hard, attn_soft = gt_attn, gt_attn

        rotat_forw_cand, trans_forw_cand = self.get_raw_slot_transform()

        rotat_forw = (attn_hard.unsqueeze(-1).unsqueeze(-1) * rotat_forw_cand.unsqueeze(0)).sum(dim=1)

        trans_forw = (attn_hard.unsqueeze(-1) * trans_forw_cand.unsqueeze(0)).sum(dim=1)

        xyz_smp_pred = torch.einsum('bcd,bde->bce', rotat_forw.permute(0, 2, 1), xyz_cnc.unsqueeze(-1)).squeeze(-1)
        xyz_smp_pred = xyz_smp_pred + trans_forw

        xyz_smp_pred_cand = torch.matmul(rotat_forw_cand.permute(0, 2, 1).unsqueeze(0),
                                         xyz_cnc.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # 1, S, 3, 3; N, 1, 3, 1 --> N, S, 3
        xyz_smp_pred_cand = xyz_smp_pred_cand + trans_forw_cand.unsqueeze(0)

        return {'attn_hard': attn_hard, 'attn_soft': attn_soft,  # [N, S]
                'world_pts': xyz_smp_pred, 'world_pts_cand': xyz_smp_pred_cand,  # [N, S, 3]
                'rotation': rotat_forw.permute(0, 2, 1), 'translation': trans_forw,
                'rotation_cand': rotat_forw_cand.permute(0, 2, 1)}

    def get_slot_motions(self):  # attn_hard: [N, S], feat_forw: [N, S, C], timesteps: [T, 1]
        rot, trans = self.get_raw_slot_transform()
        return rot, trans


