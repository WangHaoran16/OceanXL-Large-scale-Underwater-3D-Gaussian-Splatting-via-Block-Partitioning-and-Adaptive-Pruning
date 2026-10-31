"""Prune per-block Gaussians to their bounding boxes, then stream-merge all PLYs."""

import os
import json
import shutil
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
from utils.general_utils import calculate_block_transform


def mask_points_in_bbx(points, bbx, partition_mode):
    if partition_mode == "3d":
        corner_min = torch.tensor(bbx.min(axis=0), dtype=points.dtype, device=points.device)
        corner_max = torch.tensor(bbx.max(axis=0), dtype=points.dtype, device=points.device)
        return torch.all((points >= corner_min) & (points <= corner_max), dim=1)

    W2B, x_extent, y_extent, z_extent = calculate_block_transform(bbx)
    W2B = torch.tensor(W2B, dtype=points.dtype, device=points.device)
    points_block = (W2B[:3, :3] @ points.T).T + W2B[:3, 3]
    return ((points_block[:, 0] >= -x_extent) & (points_block[:, 0] <= x_extent) &
            (points_block[:, 2] >= -z_extent) & (points_block[:, 2] <= z_extent) &
            (points_block[:, 1] >= -y_extent * 1.2) & (points_block[:, 1] <= y_extent * 1.2))


def _read_ply_header(ply_filepath):
    with open(ply_filepath, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file without end_header: {ply_filepath}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        data_offset = f.tell()

    if len(header_lines) < 3 or header_lines[0].strip() != b"ply":
        raise ValueError(f"Invalid PLY header: {ply_filepath}")
    fmt = header_lines[1].strip()
    if fmt not in (b"format binary_little_endian 1.0", b"format binary_big_endian 1.0"):
        raise ValueError(f"Streaming merge only supports binary PLY, got: {fmt}")

    vertex_count = None
    vertex_line_idx = None
    for idx, line in enumerate(header_lines):
        parts = line.decode("ascii", errors="strict").strip().split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            vertex_line_idx = idx
            break
    if vertex_count is None:
        raise ValueError(f"PLY file has no vertex element: {ply_filepath}")

    normalized_header = list(header_lines)
    normalized_header[vertex_line_idx] = b"element vertex <COUNT>\n"
    return header_lines, normalized_header, vertex_line_idx, vertex_count, data_offset


def merge_binary_plys_streaming(ply_filepaths, merged_plypath):
    output_dir = os.path.dirname(merged_plypath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    headers = [_read_ply_header(p) for p in ply_filepaths]
    reference_header = headers[0][1]
    for fp, (_, norm_hdr, _, _, _) in zip(ply_filepaths[1:], headers[1:]):
        if norm_hdr != reference_header:
            raise ValueError(f"Cannot stream-merge PLY files with different headers: {fp}")

    total_vertices = sum(vertex_count for _, _, _, vertex_count, _ in headers)
    output_header = list(headers[0][0])
    output_header[headers[0][2]] = f"element vertex {total_vertices}\n".encode("ascii")

    with open(merged_plypath, "wb") as merged_file:
        merged_file.writelines(output_header)
        for fp, (_, _, _, _, data_offset) in zip(ply_filepaths, headers):
            with open(fp, "rb") as block_file:
                block_file.seek(data_offset)
                shutil.copyfileobj(block_file, merged_file, length=1024 * 1024 * 64)

    return total_vertices


def _find_latest_ply(block_gs_dir):
    """Return path to the latest iteration point_cloud.ply inside block_gs_dir."""
    pcd_root = os.path.join(block_gs_dir, "point_cloud")
    if not os.path.isdir(pcd_root):
        raise FileNotFoundError(f"No point_cloud dir under {block_gs_dir}")
    iterations = sorted(os.listdir(pcd_root))
    if not iterations:
        raise FileNotFoundError(f"No iteration dirs in {pcd_root}")
    ply_path = os.path.join(pcd_root, iterations[-1], "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"point_cloud.ply not found at {ply_path}")
    return ply_path


def fuse_blocks(output_dirpath, sh_degree=3, partition_mode="3d",
                skip_prune=False, merge_raw=False):
    """
    Args:
        output_dirpath: root directory containing blocks_info.json and block_gs/
        sh_degree: spherical harmonics degree (must match training)
        partition_mode: "3d" (axis-aligned bbox) or "2d" (OBB)
        skip_prune: if True, skip bbox pruning step (merge_only mode)
        merge_raw: if True, merge raw trained PLYs without bbox pruning
    """
    blocks_info_path = os.path.join(output_dirpath, "blocks_info.json")
    with open(blocks_info_path, "r") as f:
        blocks_info = json.load(f)
    num_blocks = int(blocks_info["num_blocks"])

    block_gs_root = os.path.join(output_dirpath, "block_gs")
    postfix_dir = os.path.join(output_dirpath, "point_cloud_postfix")
    merged_plypath = os.path.join(output_dirpath, "point_cloud_merged.ply")

    if not skip_prune and not merge_raw:
        os.makedirs(postfix_dir, exist_ok=True)
        points_counter = 0
        with torch.no_grad():
            block_gaussian = GaussianModel(sh_degree=sh_degree)
            for block_id in range(num_blocks):
                block_gs_dir = os.path.join(block_gs_root, f"block_{block_id}")
                ply_in = _find_latest_ply(block_gs_dir)
                bbx = np.array(blocks_info[str(block_id)]["bbx"])

                block_gaussian.load_ply(ply_in)
                mask = mask_points_in_bbx(block_gaussian.get_xyz, bbx, partition_mode)
                print(f"block_{block_id}: {mask.shape[0]} → {mask.sum().item()} pts "
                      f"({100*mask.sum().item()/max(mask.shape[0],1):.1f}% retained)")

                block_gaussian._xyz = block_gaussian._xyz[mask]
                block_gaussian._features_dc = block_gaussian._features_dc[mask]
                block_gaussian._features_rest = block_gaussian._features_rest[mask]
                block_gaussian._opacity = block_gaussian._opacity[mask]
                block_gaussian._scaling = block_gaussian._scaling[mask]
                block_gaussian._rotation = block_gaussian._rotation[mask]
                block_gaussian._dog_opacity = block_gaussian._dog_opacity[mask]  # DoG attributes
                block_gaussian._dog_scaling = block_gaussian._dog_scaling[mask]

                ply_out = os.path.join(postfix_dir, f"point_cloud_{block_id}.ply")
                block_gaussian.save_ply(ply_out)
                points_counter += block_gaussian._xyz.shape[0]

        print(f"Postfix over, scene total points: {points_counter}")

    ply_filepaths = []
    for block_id in range(num_blocks):
        if merge_raw:
            ply_filepaths.append(_find_latest_ply(os.path.join(block_gs_root, f"block_{block_id}")))
        else:
            ply_filepaths.append(os.path.join(postfix_dir, f"point_cloud_{block_id}.ply"))

    print(f"Merging {len(ply_filepaths)} blocks → {merged_plypath} …")
    total = merge_binary_plys_streaming(ply_filepaths, merged_plypath)
    print(f"Merged PLY written: {total} Gaussians  ({merged_plypath})")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse per-block Gaussians into a single PLY.")
    parser.add_argument("--output_dirpath", "-o", type=str, required=True,
                        help="Root output dir (contains blocks_info.json and block_gs/)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--partition_mode", type=str, default="3d",
                        help="'3d' (axis-aligned) or '2d' (OBB)")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip pruning; just merge existing postfix PLYs")
    parser.add_argument("--merge_raw", action="store_true",
                        help="Merge raw trained PLYs without bbox pruning")
    args = parser.parse_args()

    fuse_blocks(
        output_dirpath=args.output_dirpath,
        sh_degree=args.sh_degree,
        partition_mode=args.partition_mode,
        skip_prune=args.merge_only,
        merge_raw=args.merge_raw,
    )
