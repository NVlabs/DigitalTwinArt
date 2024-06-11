/*
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <stdint.h>
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>
#include <set>
#include "common.h"
#include "Eigen/Dense"



/**
 * @brief
 *
 * @tparam scalar_t
 * @param z_sampled
 * @param z_in_out
 * @param z_vals
 * @return __global__
 */
template <typename scalar_t>
__global__ void sample_rays_uniform_occupied_voxels_kernel(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_sampled, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> z_in_out, torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_vals)
{
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  const int i_sample = blockIdx.y * blockDim.y + threadIdx.y;
  if (i_ray>=z_sampled.size(0)) return;
  if (i_sample>=z_sampled.size(1)) return;

  int i_box = 0;
  float z_remain = z_sampled[i_ray][i_sample];
  auto z_in_out_cur_ray = z_in_out[i_ray];
  const float eps = 1e-4;
  const int max_n_box = z_in_out.size(1);

  if (z_in_out_cur_ray[0][0]==0) return;

  while (1)
  {
    if (i_box>=max_n_box)
    {
      if (z_remain<=eps)
      {
        z_vals[i_ray][i_sample] = z_in_out_cur_ray[max_n_box-1][1];
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<z_in_out.size(1);i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }


      return;
    }

    if (z_in_out_cur_ray[i_box][0]==0)
    {
      if (z_remain<=eps && i_box>=1)
      {
        z_vals[i_ray][i_sample] = z_in_out_cur_ray[i_box-1][1];
        return;
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<z_in_out.size(1);i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }
    }

    float box_len = z_in_out_cur_ray[i_box][1]-z_in_out_cur_ray[i_box][0];
    if (z_remain<=box_len)
    {
      z_vals[i_ray][i_sample] = z_in_out_cur_ray[i_box][0] + z_remain;
      return;
    }
    z_remain -= box_len;
    i_box++;
  }
}

at::Tensor sampleRaysUniformOccupiedVoxels(const at::Tensor z_in_out,  const at::Tensor z_sampled, at::Tensor z_vals)
{
  CHECK_INPUT(z_in_out);
  CHECK_INPUT(z_sampled);
  CHECK_INPUT(z_vals);
  AT_ASSERTM(z_vals.sizes()==z_sampled.sizes());

  const int N_rays = z_sampled.sizes()[0];
  const int N_samples = z_sampled.sizes()[1];
  const int threadx = 32;
  const int thready = 32;

  AT_DISPATCH_FLOATING_TYPES(z_in_out.type(), "sample_rays_uniform_occupied_voxels_kernel", ([&]
  {
    sample_rays_uniform_occupied_voxels_kernel<scalar_t><<<{divCeil(N_rays,threadx),divCeil(N_samples,thready)}, {threadx,thready}>>>(z_sampled.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),z_in_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),z_vals.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return z_vals;
}


template<class scalar_t>
__global__ void postprocessOctreeRayTracingKernel(const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ray_index, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> depth_in_out, const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> unique_intersect_ray_ids, const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> start_poss, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depths_in_out_padded)
{
  const int unique_id_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (unique_id_pos>=unique_intersect_ray_ids.size(0)) return;
  const int i_ray = unique_intersect_ray_ids[unique_id_pos];

  int i_intersect = 0;
  auto cur_depths_in_out_padded = depths_in_out_padded[i_ray];
  for (int i=start_poss[unique_id_pos];i<ray_index.size(0);i++)
  {
    if (ray_index[i]!=i_ray) break;
    if (depth_in_out[i][0]==0 || depth_in_out[i][1]==0) break;
    if (depth_in_out[i][0]>depth_in_out[i][1]) continue;
    if (abs(depth_in_out[i][1]-depth_in_out[i][0])<1e-4) continue;

    cur_depths_in_out_padded[i_intersect][0] = depth_in_out[i][0];
    cur_depths_in_out_padded[i_intersect][1] = depth_in_out[i][1];
    i_intersect++;

  }
}

at::Tensor postprocessOctreeRayTracing(const at::Tensor ray_index, const at::Tensor depth_in_out, const at::Tensor unique_intersect_ray_ids, const at::Tensor start_poss, const int max_intersections, const int N_rays)
{
  CHECK_INPUT(ray_index);
  CHECK_INPUT(depth_in_out);
  CHECK_INPUT(start_poss);

  const int n_unique_ids = unique_intersect_ray_ids.sizes()[0];
  at::Tensor depths_in_out_padded = at::zeros({N_rays,max_intersections,2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false));
  dim3 threads = {256};
  dim3 blocks = {divCeil(n_unique_ids,threads.x)};
  AT_DISPATCH_FLOATING_TYPES(depth_in_out.type(), "postprocessOctreeRayTracingKernel", ([&]
  {
    postprocessOctreeRayTracingKernel<scalar_t><<<blocks,threads>>>(ray_index.packed_accessor32<long,1,torch::RestrictPtrTraits>(), depth_in_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), unique_intersect_ray_ids.packed_accessor32<long,1,torch::RestrictPtrTraits>(), start_poss.packed_accessor32<long,1,torch::RestrictPtrTraits>(), depths_in_out_padded.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  return depths_in_out_padded;
}

