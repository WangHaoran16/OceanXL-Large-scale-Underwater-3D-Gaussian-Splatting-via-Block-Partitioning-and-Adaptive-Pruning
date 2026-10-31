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

import torch
import math
from diff_gaussian_rasterization_DoG import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render_DoG(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, scores = None, separate_sh=False,
           override_color=None, use_trained_exp=False, render_size=None, dog_mask=None, use_dog=True, point_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if render_size is None else render_size[0],
        image_width=int(viewpoint_camera.image_width) if render_size is None else render_size[1],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # set scores to the correct size if not passed in
    if scores is None:
        scores = torch.zeros_like(opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        if use_dog:
            raise ValueError("DoG rendering requires CUDA scale/rotation covariance computation.")
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # The DoG CUDA kernel always reads the DC coefficients separately.
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    dog_opacities = pc.get_dog_opacity if use_dog else None
    dog_scales = pc.get_dog_scaling if use_dog else None
    if dog_mask is None:
        # selective DoG (paper Eq. 4): the current selection lives on the model during training
        dog_mask = getattr(pc, "dog_mask", None)
        if dog_mask is not None and dog_mask.shape[0] != means3D.shape[0]:
            dog_mask = None
    if use_dog and dog_mask is not None:
        dog_mask = dog_mask.to(device=means3D.device, dtype=dog_opacities.dtype).reshape(-1, 1)
        dog_opacities = dog_opacities * dog_mask

    if point_mask is not None:
        point_mask = point_mask.to(device=means3D.device, dtype=torch.bool)
        means3D = means3D[point_mask]
        means2D = means2D[point_mask]
        opacity = opacity[point_mask]
        scores = scores[point_mask]
        if use_dog:
            dog_opacities = dog_opacities[point_mask]
            dog_scales = dog_scales[point_mask]

        if pipe.compute_cov3D_python:
            cov3D_precomp = cov3D_precomp[point_mask]
        else:
            scales = scales[point_mask]
            rotations = rotations[point_mask]

        if colors_precomp is not None:
            colors_precomp = colors_precomp[point_mask]
        elif shs is not None:
            shs = shs[point_mask]

        if dc is not None:
            dc = dc[point_mask]


    #threshold = torch.quantile(betas, 0.5)
    #mask = (betas >= threshold).squeeze()

    #means3D = means3D[mask]
    #means2D = means2D[mask]
    #opacity = opacity[mask]
    #scores = scores[mask]
    #betas = betas[mask]

    #if pipe.compute_cov3D_python:
        #cov3D_precomp = cov3D_precomp[mask]
    #else:
        #scales = scales[mask]
        #rotations = rotations[mask]

    #if colors_precomp is not None:
        #colors_precomp = colors_precomp[mask]
    #elif shs is not None:
        #shs = shs[mask]

    #if separate_sh:
        #dc = dc[mask]

    #scaling_factor = scaling_factor[mask]


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if dc is not None:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scores=scores,
            scales=scales,
            dog_opacities=dog_opacities,
            dog_scales=dog_scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scores=scores,
            scales=scales,
            dog_opacities=dog_opacities,
            dog_scales=dog_scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3,
                                                                                                            3, None,
                                                                                                            None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }

    return out
