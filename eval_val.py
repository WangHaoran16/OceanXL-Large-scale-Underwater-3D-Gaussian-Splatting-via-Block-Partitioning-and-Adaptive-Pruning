import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer_DoG import render_DoG
from scene import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """BlockGaussian render_views.color_correct: per-channel quadratic colour warp of `img` (H, W, 3)
    onto `ref`, solved by least squares on unsaturated pixels."""
    nc = img.shape[-1]
    img_mat = img.reshape([-1, nc])
    ref_mat = ref.reshape([-1, nc])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))
    mask0 = is_unclipped(img_mat)
    for _ in range(num_iters):
        a_mat = [img_mat[:, c:(c + 1)] * img_mat[:, c:] for c in range(nc)]
        a_mat = np.concatenate(a_mat + [img_mat, np.ones_like(img_mat[:, :1])], axis=-1)
        warp = []
        for c in range(nc):
            b = ref_mat[:, c]
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            w = np.linalg.lstsq(np.where(mask[:, None], a_mat, 0), np.where(mask, b, 0), rcond=-1)[0]
            assert np.all(np.isfinite(w))
            warp.append(w)
        img_mat = np.clip(a_mat @ np.stack(warp, axis=-1), 0, 1)
    return img_mat.reshape(img.shape)


def metrics(img, gt):
    return {"SSIM": ssim(img[None], gt[None]).item(),
            "PSNR": psnr(img[None], gt[None]).mean().item(),
            "LPIPS": lpips(img[None], gt[None], net_type='vgg').item()}


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_renders", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    pipe = pp.extract(args)

    cam_args = lp.extract(args)
    cam_args.train_test_exp = False
    cam_args.data_device = "cpu"
    info = readColmapSceneInfo(os.path.abspath(args.val_dir), "images", "", False, False)
    views = cameraList_from_camInfos(info.train_cameras, 1.0, cam_args, False, False)

    gaussians = GaussianModel(cam_args.sh_degree)
    gaussians.load_ply(args.ply)
    print(f"{gaussians.get_xyz.shape[0]:,} Gaussians, {len(views)} val cameras, "
          f"image {views[0].image_width}x{views[0].image_height}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ply)), "eval_val")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_renders:
        for sub in ("renders", "gt"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    per_view = {}
    with torch.no_grad():
        for view in tqdm(views, desc="Rendering val"):
            name = os.path.splitext(os.path.basename(view.image_name))[0]
            raw = render_DoG(view, gaussians, pipe, bg, separate_sh=True)["render"].clamp(0, 1)
            gt = view.original_image[0:3].cuda()
            cc = color_correct(raw.permute(1, 2, 0).double().cpu().numpy(),
                               gt.permute(1, 2, 0).double().cpu().numpy())
            cc = torch.from_numpy(cc).float().permute(2, 0, 1).contiguous().cuda()
            per_view[name] = metrics(cc, gt)
            if args.save_renders:
                torchvision.utils.save_image(cc, os.path.join(out_dir, "renders", name + ".png"))
                torchvision.utils.save_image(gt, os.path.join(out_dir, "gt", name + ".png"))

    results = {m: sum(v[m] for v in per_view.values()) / len(per_view) for m in ("SSIM", "PSNR", "LPIPS")}
    results["num_views"] = len(views)
    results["image_size"] = [views[0].image_width, views[0].image_height]
    results["num_gaussians"] = int(gaussians.get_xyz.shape[0])
    results["ply_size_MB"] = round(os.path.getsize(args.ply) / 2 ** 20, 1)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "per_view.json"), "w") as f:
        json.dump(per_view, f, indent=2)
    print(json.dumps(results, indent=2))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
