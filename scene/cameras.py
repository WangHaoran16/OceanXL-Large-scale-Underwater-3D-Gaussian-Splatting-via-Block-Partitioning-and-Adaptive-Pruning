#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2ViewCUDA, getProjectionMatrixCUDA
from utils.general_utils import PILtoTorch
import cv2


# ── BlockGaussian camera types (used by scene_loader / colmap_loader) ─────────

class ViewInfo:
    def __init__(self, view_id, extrinsic, intrinsic, image_height, image_width, image_filepath, **kwargs):
        self.view_id = view_id
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.image_height = image_height
        self.image_width = image_width
        self.image_filepath = image_filepath
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return None


class RenderCamera(object):
    def __init__(self, image_height, image_width, FoVy, FoVx, extrinsic, intrinsic,
                 world_view_transform, full_proj_transform):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def get_render_camera(image_height, image_width, extrinsic, intrinsic, disturb=None,
                      trans=torch.tensor([0.0, 0.0, 0.0]).cuda(), scale=1.0,
                      znear=0.01, zfar=100.0):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    FoVx = 2 * math.atan(image_width / (2 * fx))
    FoVy = 2 * math.atan(image_height / (2 * fy))
    world_view_transform = getWorld2ViewCUDA(R, T, trans, scale, disturb).permute(1, 0)
    projection_matrix = getProjectionMatrixCUDA(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    return RenderCamera(image_height=image_height, image_width=image_width,
                        FoVy=FoVy, FoVx=FoVx,
                        extrinsic=world_view_transform.permute(1, 0), intrinsic=intrinsic,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform)


# ── gs camera types (used by dataset_readers / Scene for training) ────────────

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        #resized_image_rgb = PILtoTorch(image, resolution)
        if torch.is_tensor(image):
            resized_image_rgb = image
        else:
            if resolution is not None:
                resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0

                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

