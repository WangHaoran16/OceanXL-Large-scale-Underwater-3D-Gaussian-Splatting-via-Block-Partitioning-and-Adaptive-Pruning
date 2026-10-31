"""One entry point for the whole pipeline, scene parameters come from configs/scenes.yaml.

    conda activate gaussian_splatting
    cd /d D:\\rebuild\\OceanXL_pipeline
    python run.py KWAJ                         # partition -> export -> train -> fuse -> eval
    python run.py KWAJ --steps train fuse eval # only some steps
    python run.py KWAJ --blocks 0 2            # train only these blocks
    python run.py all                          # every scene in scenes.yaml, one after another
    python run.py cr --reuse_partition D:/rebuild/BlockGaussian/output/cr_ours

Steps
  partition  BlockGaussian scene_partition (3d: BSP, 2d: draw the ROI polygon, Enter to close)
  export     per-block COLMAP folders  <out>/block_colmap/block_i
  train      train_block.py per block   <out>/block_gs/block_i   (all train images of the block)
  fuse       crop each block to its bbx and merge  <out>/point_cloud_merged.ply
  eval       render <scene>/val only     <out>/eval_val/results.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = ["partition", "export", "train", "fuse", "eval"]
ITER_KEYS = ["iterations", "densify_from_iter", "densify_until_iter", "opacity_reset_interval",
             "prune_start_iter", "prune_stop_iter", "pruning_interval", "pruning_interval_maximum",
             "dog_select_interval", "dog_select_freeze"]


def load_scene_cfg(name, iterations=None):
    with open(os.path.join(ROOT, "configs", "scenes.yaml"), "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    scenes = {str(k): v for k, v in cfg_all["scenes"].items()}
    if name not in scenes:
        sys.exit(f"unknown scene '{name}', choose from {list(scenes)}")
    cfg = dict(cfg_all["defaults"])
    cfg.update(scenes[name])
    cfg["name"] = name
    cfg["train_dir"] = os.path.normpath(os.path.join(cfg["datasets_root"], cfg["train_dir"]))
    cfg["val_dir"] = os.path.normpath(os.path.join(cfg["datasets_root"], cfg["val_dir"]))
    cfg["out"] = os.path.normpath(os.path.join(cfg["output_root"], name))
    if iterations:  # shorter run (smoke test): scale the whole schedule
        ratio = iterations / cfg["iterations"]
        for k in ITER_KEYS:
            cfg[k] = max(1, int(round(cfg[k] * ratio)))
    return cfg



def run(cmd, tag, env=None):
    print("\n" + "=" * 80 + f"\n[{tag}] " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"[{tag}] exit={ret.returncode}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if ret.returncode != 0:
        sys.exit(f"[{tag}] failed")


def num_blocks(cfg):
    with open(os.path.join(cfg["out"], "blocks_info.json"), "r") as f:
        return int(json.load(f)["num_blocks"])


def latest_ply(model_dir):
    plys = glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*", "point_cloud.ply"))
    return max(plys, key=lambda p: int(p.split("iteration_")[-1].split(os.sep)[0])) if plys else None


# ------------------------------------------------------------------------------------ steps

def step_partition(cfg, reuse_from=None):
    os.makedirs(cfg["out"], exist_ok=True)
    if reuse_from:
        reuse_from = os.path.abspath(reuse_from)
        with open(os.path.join(reuse_from, "blocks_info.json"), "r") as f:
            info = json.load(f)
        dst_pcd = os.path.join(cfg["out"], "block_init_pcd")
        os.makedirs(dst_pcd, exist_ok=True)
        for ply in glob.glob(os.path.join(reuse_from, "block_init_pcd", "*.ply")):
            shutil.copy2(ply, dst_pcd)
        for k, v in info.items():
            if k != "num_blocks":
                v["block_pcd_filepath"] = os.path.join(dst_pcd, os.path.basename(v["block_pcd_filepath"]))
        with open(os.path.join(cfg["out"], "blocks_info.json"), "w") as f:
            json.dump(info, f, indent=4)
        for png in ("Partition_Results.png", "ROI_region.png"):
            if os.path.exists(os.path.join(reuse_from, png)):
                shutil.copy2(os.path.join(reuse_from, png), cfg["out"])
        print(f"[PARTITION] reused {info['num_blocks']} blocks from {reuse_from}")
        return

    env = dict(os.environ)
    if cfg["partition_mode"] == "3d":
        env["MPLBACKEND"] = "Agg"  # nothing to draw by hand: save the figure, don't block on it
    run([sys.executable, "scene_partition.py",
         "-s", cfg["train_dir"], "-o", cfg["out"],
         "--partition_mode", cfg["partition_mode"],
         "--vertical_axis", cfg["vertical_axis"],
         "--max_tree_depth", str(cfg["max_tree_depth"]),
         "--num_points_thresh", str(cfg["num_points_thresh"]),
         "--cover_ratio_thresh", str(cfg["cover_ratio_thresh"]),
         "--expand_ratio", str(cfg["expand_ratio"]),
         "--scene_scale", str(cfg["scene_scale"]),
         "--evaluate", "False"], "PARTITION", env)


def step_export(cfg):
    run([sys.executable, "block_export.py", "-s", cfg["train_dir"], "-o", cfg["out"],
         "--evaluate", "False", "--scene_scale", str(cfg["scene_scale"])], "EXPORT")


def step_train(cfg, blocks=None, force=False):
    blocks_info = os.path.join(cfg["out"], "blocks_info.json")
    ids = blocks if blocks else range(num_blocks(cfg))
    timing = {}
    for b in ids:
        model = os.path.join(cfg["out"], "block_gs", f"block_{b}")
        done = os.path.join(model, "point_cloud", f"iteration_{cfg['iterations']}", "point_cloud.ply")
        if os.path.exists(done) and not force:
            print(f"[TRAIN block_{b}] already trained, skipped (--force to retrain)")
            continue
        cmd = [sys.executable, "train_block.py",
               "-s", os.path.join(cfg["out"], "block_colmap", f"block_{b}"), "-m", model,
               "--blocks_info", blocks_info, "--block_id", str(b),
               "-r", "960",
               "--data_device", cfg["data_device"],
               "--sh_degree", str(cfg["sh_degree"]),
               "--iterations", str(cfg["iterations"]),
               "--position_lr_max_steps", str(cfg["iterations"]),
               "--densify_from_iter", str(cfg["densify_from_iter"]),
               "--densify_until_iter", str(cfg["densify_until_iter"]),
               "--densification_interval", str(cfg["densification_interval"]),
               "--densify_grad_threshold", str(cfg["densify_grad_threshold"]),
               "--opacity_reset_interval", str(cfg["opacity_reset_interval"]),
               "--optimizing_spa_start_iter", str(cfg["prune_start_iter"]),
               "--optimizing_spa_stop_iter", str(cfg["prune_stop_iter"]),
               "--pruning_interval", str(cfg["pruning_interval"]),
               "--pruning_interval_maximum", str(cfg["pruning_interval_maximum"]),
               "--prune_keep_ratio", str(cfg["prune_keep_ratio"]),
               "--dog_grad_threshold", str(cfg["dog_grad_threshold"]),
               "--dog_select_interval", str(cfg["dog_select_interval"]),
               "--dog_select_freeze", str(cfg["dog_select_freeze"]),
               "--ucd_f_max", str(cfg["ucd_f_max"]),
               "--ucd_spsa_a", str(cfg["ucd_spsa_a"]),
               "--ucd_spsa_c", str(cfg["ucd_spsa_c"]),
               "--freq_loss_weight", str(cfg["freq_loss_weight"]),
               "--densify_mode", "free", "--resolution_mode", "const",
               "--disable_viewer", "--quiet"]
        if not cfg["ucd"]:
            cmd.append("--no_ucd")
        t0 = time.time()
        run(cmd, f"TRAIN block_{b}")
        timing[f"block_{b}"] = round((time.time() - t0) / 60, 2)
    if timing:
        with open(os.path.join(cfg["out"], "train_timing.json"), "a") as f:
            f.write(json.dumps(timing) + "\n")


def step_fuse(cfg):
    missing = [b for b in range(num_blocks(cfg))
               if latest_ply(os.path.join(cfg["out"], "block_gs", f"block_{b}")) is None]
    if missing:
        sys.exit(f"[FUSE] blocks not trained yet: {missing}")
    run([sys.executable, "block_fusion.py", "-o", cfg["out"],
         "--sh_degree", str(cfg["sh_degree"]), "--partition_mode", cfg["partition_mode"]], "FUSE")


def step_eval(cfg, save_renders=False):
    cmd = [sys.executable, "eval_val.py",
           "--ply", os.path.join(cfg["out"], "point_cloud_merged.ply"),
           "--val_dir", cfg["val_dir"],
           "-r", "960",
           "--sh_degree", str(cfg["sh_degree"])]
    if save_renders:
        cmd.append("--save_renders")
    run(cmd, "EVAL val")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene name in configs/scenes.yaml, or 'all'")
    parser.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    parser.add_argument("--blocks", nargs="+", type=int, default=None, help="train only these block ids")
    parser.add_argument("--force", action="store_true", help="retrain blocks that are already trained")
    parser.add_argument("--reuse_partition", type=str, default=None,
                        help="copy blocks_info.json + block_init_pcd from an earlier partition instead of partitioning")
    parser.add_argument("--iterations", type=int, default=None,
                        help="override total iterations; the whole schedule is scaled with it")
    parser.add_argument("--save_renders", action="store_true", help="eval: also save val renders / gt")
    parser.add_argument("--output_root", type=str, default=None, help="override output_root of scenes.yaml")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "configs", "scenes.yaml"), "r", encoding="utf-8") as f:
        all_names = [str(k) for k in yaml.safe_load(f)["scenes"]]
    names = all_names if args.scene == "all" else [args.scene]

    for name in names:
        cfg = load_scene_cfg(name, args.iterations)
        if args.output_root:
            cfg["out"] = os.path.normpath(os.path.join(args.output_root, name))
        os.makedirs(cfg["out"], exist_ok=True)
        with open(os.path.join(cfg["out"], "scene_cfg.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"\n##### {name}: {cfg['train_dir']} -> {cfg['out']}  steps={args.steps}")

        t0 = time.time()
        if "partition" in args.steps:
            step_partition(cfg, args.reuse_partition)
        if "export" in args.steps:
            step_export(cfg)
        if "train" in args.steps:
            step_train(cfg, args.blocks, args.force)
        if "fuse" in args.steps:
            step_fuse(cfg)
        if "eval" in args.steps:
            step_eval(cfg, args.save_renders)
        print(f"\n##### {name} done in {(time.time() - t0) / 60:.1f} min, results in {cfg['out']}")


if __name__ == "__main__":
    main()
