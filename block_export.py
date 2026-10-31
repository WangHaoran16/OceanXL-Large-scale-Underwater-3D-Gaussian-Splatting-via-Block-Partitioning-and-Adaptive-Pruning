"""Export per-block COLMAP directories from a scene partitioned by scene_partition.py."""

import os
import json
import shutil
import struct
import argparse
import numpy as np
from collections import namedtuple
from scene.scene_loader import Scene

Camera = namedtuple("Camera", "id model_id width height params")
Image = namedtuple("Image", "id qvec tvec camera_id name xys point3D_ids")
Point3D = namedtuple("Point3D", "id xyz rgb error image_ids point2D_idxs")

CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12
}


def _read_next(fid, n, fmt):
    return struct.unpack("<" + fmt, fid.read(n))


def _write_next(fid, data, fmt):
    if not isinstance(data, (list, tuple)):
        data = [data]
    fid.write(struct.pack("<" + fmt, *data))


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num = _read_next(f, 8, "Q")[0]
        for _ in range(num):
            cam_id, model_id, width, height = _read_next(f, 24, "iiQQ")
            n = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(_read_next(f, 8 * n, "d" * n))
            cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
    return cameras


def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        _write_next(f, len(cameras), "Q")
        for cam in cameras.values():
            _write_next(f, [cam.id, cam.model_id, cam.width, cam.height], "iiQQ")
            _write_next(f, cam.params.tolist(), "d" * len(cam.params))


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = _read_next(f, 8, "Q")[0]
        for _ in range(num):
            props = _read_next(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points2D = _read_next(f, 8, "Q")[0]
            xys = np.zeros((num_points2D, 2))
            point3D_ids = np.zeros(num_points2D, dtype=np.int64)
            for i in range(num_points2D):
                x, y, p3d_id = _read_next(f, 24, "ddq")
                xys[i] = [x, y]
                point3D_ids[i] = p3d_id
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def write_images_binary(images, path):
    with open(path, "wb") as f:
        _write_next(f, len(images), "Q")
        for im in images.values():
            _write_next(f, [im.id, *im.qvec.tolist(), *im.tvec.tolist(), im.camera_id], "idddddddi")
            f.write(im.name.encode("utf-8") + b"\x00")
            _write_next(f, len(im.point3D_ids), "Q")
            for (x, y), p3d_id in zip(im.xys, im.point3D_ids):
                _write_next(f, [float(x), float(y), int(p3d_id)], "ddq")


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as f:
        num = _read_next(f, 8, "Q")[0]
        for _ in range(num):
            props = _read_next(f, 43, "QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = props[7]
            track_len = _read_next(f, 8, "Q")[0]
            image_ids = np.zeros(track_len, dtype=np.int32)
            point2D_idxs = np.zeros(track_len, dtype=np.int32)
            for i in range(track_len):
                image_ids[i], point2D_idxs[i] = _read_next(f, 8, "ii")
            points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points3D


def write_points3D_binary(points3D, path):
    with open(path, "wb") as f:
        _write_next(f, len(points3D), "Q")
        for p in points3D.values():
            _write_next(f, [p.id, *p.xyz.tolist(),
                            int(p.rgb[0]), int(p.rgb[1]), int(p.rgb[2]), p.error], "QdddBBBd")
            _write_next(f, len(p.image_ids), "Q")
            for img_id, idx2d in zip(p.image_ids, p.point2D_idxs):
                _write_next(f, [int(img_id), int(idx2d)], "ii")


def export_blocks(scene_dirpath, output_dirpath, evaluate=False, scene_scale=1.0):
    scene = Scene(scene_dirpath, evaluate=evaluate, scene_scale=scene_scale)

    with open(os.path.join(output_dirpath, "blocks_info.json"), "r") as f:
        blocks_info = json.load(f)

    sparse_dir = os.path.join(scene_dirpath, "sparse", "0")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    points3D = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))

    depth_src_dir = os.path.join(scene_dirpath, "depth_any")
    depth_params_path = os.path.join(scene_dirpath, "sparse", "0", "depth_params.json")
    depth_params = None
    if os.path.exists(depth_params_path):
        with open(depth_params_path, "r") as f:
            depth_params = json.load(f)

    name_to_image = {os.path.basename(im.name): im for im in images.values()}

    save_root = os.path.join(output_dirpath, "block_colmap")
    os.makedirs(save_root, exist_ok=True)

    for block_id in range(blocks_info["num_blocks"]):
        block_dir = os.path.join(save_root, f"block_{block_id}")
        image_dir = os.path.join(block_dir, "images")
        depth_dir = os.path.join(block_dir, "depths")
        sparse_out = os.path.join(block_dir, "sparse", "0")

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(sparse_out, exist_ok=True)

        keep_names = []
        for view_id in blocks_info[str(block_id)]["views_id"]:
            src = scene.views_info[view_id].image_filepath
            name = os.path.basename(src)
            if not os.path.exists(src):
                src = os.path.join(scene_dirpath, "images", name)
            keep_names.append(name)
            shutil.copy2(src, os.path.join(image_dir, name))
            depth_src = os.path.join(depth_src_dir, name)
            depth_dst = os.path.join(depth_dir, name)
            if os.path.exists(depth_src):
                shutil.copy2(depth_src, depth_dst)

        keep_images = {}
        for name in keep_names:
            if name in name_to_image:
                im = name_to_image[name]
                keep_images[im.id] = Image(
                    im.id, im.qvec, im.tvec, im.camera_id, name, im.xys, im.point3D_ids)

        if depth_params is not None:
            block_depth_params = {
                os.path.splitext(n)[0]: depth_params[os.path.splitext(n)[0]]
                for n in keep_names
                if os.path.splitext(n)[0] in depth_params
            }
            with open(os.path.join(sparse_out, "depth_params.json"), "w") as f:
                json.dump(block_depth_params, f, indent=2, ensure_ascii=False)

        keep_camera_ids = {im.camera_id for im in keep_images.values()}
        keep_cameras = {cid: cameras[cid] for cid in keep_camera_ids}

        keep_point_ids = set()
        for im in keep_images.values():
            for pid in im.point3D_ids:
                if pid != -1:
                    keep_point_ids.add(int(pid))

        keep_images_set = set(keep_images.keys())
        keep_points = {}
        for pid in keep_point_ids:
            if pid not in points3D:
                continue
            p = points3D[pid]
            mask = np.array([img_id in keep_images_set for img_id in p.image_ids])
            if mask.sum() == 0:
                continue
            keep_points[pid] = Point3D(
                p.id, p.xyz, p.rgb, p.error,
                p.image_ids[mask], p.point2D_idxs[mask])

        write_cameras_binary(keep_cameras, os.path.join(sparse_out, "cameras.bin"))
        write_images_binary(keep_images, os.path.join(sparse_out, "images.bin"))
        write_points3D_binary(keep_points, os.path.join(sparse_out, "points3D.bin"))

        block_pcd_filepath = blocks_info[str(block_id)].get("block_pcd_filepath")
        if block_pcd_filepath is not None and os.path.exists(block_pcd_filepath):
            shutil.copy2(block_pcd_filepath, os.path.join(sparse_out, "points3D.ply"))
        else:
            print(f"warning: block_{block_id} init pcd not found: {block_pcd_filepath}")

        print(f"block_{block_id}: images={len(keep_images)}, "
              f"cameras={len(keep_cameras)}, points3D={len(keep_points)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export per-block COLMAP dirs from a scene partitioned by scene_partition.py.")
    parser.add_argument("--scene_dirpath", "-s", type=str,
                        default="./datasets/ISRO/undistorted/train")
    parser.add_argument("--output_dirpath", "-o", type=str, default=None)
    parser.add_argument("--evaluate", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.output_dirpath is None:
        parser.error("--output_dirpath / -o is required")

    export_blocks(args.scene_dirpath, args.output_dirpath, args.evaluate, args.scene_scale)
