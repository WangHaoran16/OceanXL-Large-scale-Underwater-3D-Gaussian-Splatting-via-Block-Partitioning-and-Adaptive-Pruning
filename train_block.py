#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Per-block training = gs/train_prune.py, with two changes taken from OceanXL block_train.py:
#   * densification weights: the fixed near/mid/far factors 1 / 2 / 4 of train_prune.py are
#     replaced by UCD, whose f_mid / f_far are learned with SPSA (utils/ucd.py);
#   * after Wise Adaptive Pruning: selective DoG (select_dog_primitives + render_DoG).
# Schedule (40K): densify 0-20K, prune 20K-30K, DoG 30K-40K.
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import uuid
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer_pup import render_pup
from gaussian_renderer import render, network_gui
from gaussian_renderer_DoG import render_DoG
from gaussian_renderer_DoG_report import render_DoG_report
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.schedule_utils import TrainingScheduler
from utils.ucd import UCDSPSA

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

SEPARATE_SH = True  # both CUDA rasterizers read the DC colour from a separate buffer


# ---------------------------------------------------------------- pruning (gs/train_prune.py)

def reconstruction_detection(scene, gaussians, pipe, background, dataset, size):
    with torch.enable_grad():
        scores = torch.zeros_like(gaussians.get_opacity)
        total_loss = 0.0
        flag = 0
        for view in scene.getTrainCameras():
            flag += 1
            gt = view.original_image.cuda()
            per_view_loss = loss_func(view, gaussians, pipe, background, scores, dataset, size, gt)
            total_loss += per_view_loss.item()
        avg_loss = total_loss / flag
    return avg_loss


def loss_func(view, gaussians, pipeline, background, scores, dataset, size, gt):
    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = False
    image = render_pup(view, gaussians, pipeline, background,
                       use_trained_exp=dataset.train_test_exp,
                       separate_sh=SEPARATE_SH,
                       scores=img_scores)['render']
    return l1_loss(gt, image)


def score_func(view, gaussians, pipeline, background, scores, dataset, size, gt):
    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True
    image = render_pup(view, gaussians, pipeline, background,
                       use_trained_exp=dataset.train_test_exp,
                       separate_sh=SEPARATE_SH,
                       scores=img_scores)['render']
    spatial_seed = image.sum()
    spatial_seed.backward()
    per_view_loss = l1_loss(gt, image)
    scores += img_scores.grad
    return per_view_loss


def prune(scene, gaussians, pipe, background, prune_ratio, dataset, size):
    with torch.enable_grad():
        scores = torch.zeros_like(gaussians.get_opacity)
        total_loss = 0.0
        flag = 0
        for view in scene.getTrainCameras():
            flag += 1
            gt = view.original_image.cuda()
            per_view_loss = score_func(view, gaussians, pipe, background, scores, dataset, size, gt)
            total_loss += per_view_loss.item()
        avg_loss = total_loss / flag
    gaussians.prune_gaussians(prune_ratio, scores, scene)
    return avg_loss


# ---------------------------------------------------------------- selective DoG (OceanXL)

def select_dog_primitives(scene, gaussians, pipe, background, dataset, opt, tau, iteration):
    """OceanXL Eq. 4: a_i = 1[ ||dL/d mean2D_i||_2 > tau ], applied progressively.

    One pass over the training views with the current model; Gaussians whose mean 2D positional
    gradient exceeds tau get DoG, on top of those already activated (activation is never undone).
    """
    n = gaussians.get_xyz.shape[0]
    prev = getattr(gaussians, "dog_mask", None)
    if prev is None or prev.shape[0] != n:
        prev = torch.zeros(n, dtype=torch.bool, device="cuda")
    gaussians.dog_mask = prev
    grad_accum = torch.zeros(n, device="cuda")
    count = torch.zeros(n, device="cuda")
    with torch.enable_grad():
        for view in scene.getTrainCameras():
            pkg = render_DoG(view, gaussians, pipe, background,
                             use_trained_exp=dataset.train_test_exp,
                             separate_sh=SEPARATE_SH)
            image, vs, radii = pkg["render"], pkg["viewspace_points"], pkg["radii"]
            visible = radii > 0
            if not visible.any():
                continue
            gt = view.original_image.cuda()
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt)
            loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1.0 - ssim_value)
            loss.backward()
            grad_accum[visible] += torch.norm(vs.grad[visible, :2], dim=-1)
            count[visible] += 1
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
    mean_grad = grad_accum / count.clamp_min(1)
    dog_mask = prev | (mean_grad > tau)
    gaussians.dog_mask = dog_mask
    msg = "iter {}: DoG active for {:,} / {:,} Gaussians ({:.1%}, +{:,} new), tau = {}".format(
        iteration, int(dog_mask.sum()), n, dog_mask.float().mean().item(),
        int(dog_mask.sum() - prev.sum()), tau)
    print("\n[SELECTIVE DoG] " + msg)
    with open(os.path.join(scene.model_path, "dog_selection.txt"), "a") as f:
        f.write(msg + "\n")


def save_with_dog_mask(scene, gaussians, iteration):
    """Save with inactive primitives written as plain Gaussians (DoG opacity ~ 0), so the PLY
    renders the same without the in-memory mask; the live values are restored afterwards."""
    dog_mask = getattr(gaussians, "dog_mask", None)
    if dog_mask is None or dog_mask.shape[0] != gaussians._dog_opacity.shape[0]:
        scene.save(iteration)
        return
    with torch.no_grad():
        live = gaussians._dog_opacity.detach().clone()
        gaussians._dog_opacity[~dog_mask] = -20.0  # sigmoid(-20) ~ 2e-9
    scene.save(iteration)
    with torch.no_grad():
        gaussians._dog_opacity.copy_(live)


# ---------------------------------------------------------------- training

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, block_bbx_expand, freq_loss_weight):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    pruning_count = 0

    ucd = UCDSPSA(f_max=opt.ucd_f_max, a=opt.ucd_spsa_a, c=opt.ucd_spsa_c) if not opt.no_ucd else None

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians,
                                  [cam.original_image for cam in scene.getTrainCameras()])

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_pup(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                           use_trained_exp=dataset.train_test_exp, separate_sh=SEPARATE_SH)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # DoG stage (after pruning): redo the selection every dog_select_interval steps, keep it
        # fixed for the last dog_select_freeze steps (no densify / prune here, indices are stable)
        if opt.optimizing_spa_stop_iter <= iteration <= opt.iterations - opt.dog_select_freeze and \
                (iteration - opt.optimizing_spa_stop_iter) % opt.dog_select_interval == 0:
            select_dog_primitives(scene, gaussians, pipe, background, dataset, opt,
                                  opt.dog_grad_threshold, iteration)

        if iteration < opt.optimizing_spa_stop_iter:
            render_pkg = render_pup(viewpoint_cam, gaussians, pipe, bg,
                                    use_trained_exp=dataset.train_test_exp, separate_sh=SEPARATE_SH)
        else:
            render_pkg = render_DoG(viewpoint_cam, gaussians, pipe, bg,
                                    use_trained_exp=dataset.train_test_exp, separate_sh=SEPARATE_SH)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if freq_loss_weight > 0 and iteration > opt.densify_until_iter:
            F_img = torch.fft.fft2(image, norm='ortho')
            F_gt = torch.fft.fft2(gt_image, norm='ortho')
            loss += freq_loss_weight * (l1_loss(F_img.real, F_gt.real) + l1_loss(F_img.imag, F_gt.imag))

        # Skip backward when no Gaussians are visible: both rasterizers return wrong
        # gradient shapes in that edge case.
        has_visible = (radii > 0).any()
        if has_visible:
            loss.backward()
        else:
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.6 * ema_Ll1depth_for_log  # no depth loss, as in gs train_prune
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "N_GS": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            report_render = render if iteration < opt.optimizing_spa_stop_iter else render_DoG_report
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations,
                            scene, report_render, (pipe, background, 1., SEPARATE_SH, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                save_with_dog_mask(scene, gaussians, iteration)

            # Densification (UCD depth weights, densify only inside the block's expanded box)
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if ucd is not None:
                    f_mid, f_far = ucd.current()
                    depth_factor = UCDSPSA.depth_weights(gaussians.get_xyz, viewpoint_cam, radii > 0, f_mid, f_far)
                    gaussians.add_densification_stats_big(viewspace_point_tensor, visibility_filter, depth_factor)
                    if iteration > opt.densify_from_iter:
                        ucd.record(Ll1.item())
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_big(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, block_bbx_expand)
                    if ucd is not None:
                        ucd.end_interval(iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    if ucd is not None:
                        ucd.mark_dirty()

            # Wise Adaptive Pruning (gs/train_prune.py)
            elif iteration == opt.optimizing_spa_start_iter:
                if opt.optimizing_spa == True:
                    loss_stamp = 1
                    iteration_no_prune = 0
                    pruning_count = 1
                    scheduler.calculate_pruning_target(gaussians.get_xyz.shape[0], opt.prune_ratio2, opt.pruning_target)
                    pruning_ratio = scheduler.get_pruning_rate(iteration, gaussians.get_xyz.shape[0], pruning_count)
                    pruning_count = pruning_count + 1
                else:
                    opacity = gaussians.get_opacity
                    k_threshold = int((1 - pruning_ratio) * opacity.shape[0])
                    _, indices = torch.topk(opacity[:, 0], k=k_threshold, largest=True)
                    mask = torch.ones(opacity.shape[0], dtype=bool)
                    mask[indices] = False
                    gaussians.prune_points(mask)

            elif (iteration - opt.optimizing_spa_start_iter) % opt.pruning_interval == 0 and (
                    iteration > opt.optimizing_spa_start_iter and iteration < opt.optimizing_spa_stop_iter) and pruning_count <= 6:
                per_view_loss = reconstruction_detection(scene, gaussians, pipe, background, dataset, gt_image.shape[-2:])
                print("\n[loss_stamp {}],[per_view_loss {}]".format(loss_stamp, per_view_loss))
                factor = 1.0
                if per_view_loss < factor * loss_stamp or iteration_no_prune >= opt.pruning_interval_maximum:
                    loss_stamp = prune(scene, gaussians, pipe, background, pruning_ratio, dataset, gt_image.shape[-2:])
                    iteration_no_prune = 0
                    pruning_ratio = scheduler.get_pruning_rate(iteration, gaussians.get_xyz.shape[0], pruning_count)
                    print("\n[ITER {}],[Pru_C {}]".format(iteration, pruning_count))
                    pruning_count = pruning_count + 1
                elif per_view_loss > factor * loss_stamp:
                    iteration_no_prune += opt.pruning_interval

            # Optimizer step
            if iteration < opt.iterations and has_visible:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report samples of training set (no test views: every block image is used for training)
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        cams = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        l1_test = 0.0
        psnr_test = 0.0
        for viewpoint in cams:
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
        print("\n[ITER {}] Evaluating train: L1 {} PSNR {}".format(iteration, l1_test / len(cams), psnr_test / len(cams)))
        if tb_writer:
            tb_writer.add_scalar('train/loss_viewpoint - psnr', psnr_test / len(cams), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Per-block training: gs train_prune + UCD + selective DoG")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--blocks_info", type=str, required=True)
    parser.add_argument("--block_id", type=int, required=True)
    parser.add_argument("--freq_loss_weight", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    with open(args.blocks_info, "r") as json_file:
        blocks_info = json.load(json_file)
    import numpy as np
    block_bbx_expand = np.array(blocks_info[str(args.block_id)]["bbx_expand"])
    print("received block_id:", args.block_id)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, block_bbx_expand, args.freq_loss_weight)

    # All done
    print("\nTraining complete.")
