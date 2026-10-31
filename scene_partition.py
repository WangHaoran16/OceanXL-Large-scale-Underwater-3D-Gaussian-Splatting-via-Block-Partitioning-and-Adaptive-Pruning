import os
import json
import argparse
import numpy as np
try:
    import open3d as o3d
except ImportError:  # no open3d wheel (e.g. aarch64): scipy fallback below
    o3d = None
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scene.scene_loader import Scene
import matplotlib.pyplot as plt
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from scene.colmap_loader import read_points3D_binary_, read_points3D_text_, qvec2rotmat
from utils.general_utils import storePly, fetchPly


def fetch_local_pcd(scene_dirpath, block_pcd_filepath, views_id, scene_scale=1.0):
    assert os.path.exists(os.path.join(scene_dirpath, "sparse/0")), \
        "sparse fold is not exist while the scene_type is colmap."
    try:
        cameras_extrinsic_file = os.path.join(scene_dirpath, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(scene_dirpath, "sparse/0", "images.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)

    bin_path = os.path.join(scene_dirpath, "sparse/0/points3D.bin")
    txt_path = os.path.join(scene_dirpath, "sparse/0/points3D.txt")
    try:
        points3D = read_points3D_binary_(bin_path, scene_scale)
    except:
        points3D = read_points3D_text_(txt_path, scene_scale)

    points3D_ids_total = set()
    for view_id in views_id:
        extr = cam_extrinsics[view_id]
        ids = extr.point3D_ids[extr.point3D_ids != -1]
        points3D_ids_total.update(ids)

    points3D_ids_total = list(points3D_ids_total)
    num_pts = len(points3D_ids_total)
    xyzs = np.array([points3D[pt_id].xyz for pt_id in points3D_ids_total])
    rgbs = np.array([points3D[pt_id].rgb for pt_id in points3D_ids_total])
    if num_pts != 0:
        os.makedirs(os.path.dirname(block_pcd_filepath), exist_ok=True)
        storePly(block_pcd_filepath, xyzs, rgbs)

    return len(points3D_ids_total)


def plot_rectangle(rectangle: np.ndarray, color=np.random.rand(3), text=None):
    rectangle_closed = np.vstack([rectangle, rectangle[0]])
    plt.fill(rectangle_closed[:, 0], rectangle_closed[:, 1], color=color, alpha=0.3)
    plt.plot(rectangle_closed[:, 0], rectangle_closed[:, 1], color=color, alpha=0.3)
    if text is not None:
        center_x = (rectangle[0][0] + rectangle[2][0]) / 2
        center_y = (rectangle[0][1] + rectangle[2][1]) / 2
        font_size = int(abs(rectangle[0][0] - rectangle[2][0]) / 10)
        plt.text(center_x, center_y, text, fontsize=font_size, color='blue')


def minimum_area_bounding_rectangle(polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon)
    hull = ConvexHull(polygon)
    hull_points = polygon[hull.vertices]

    min_area = float('inf')
    best_rectangle = None

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        edge = p2 - p1
        angle = np.arctan2(edge[1], edge[0])
        cos_theta = np.cos(-angle)
        sin_theta = np.sin(-angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_points = np.dot(hull_points, rotation_matrix.T)
        min_x, min_y = np.min(rotated_points, axis=0)
        max_x, max_y = np.max(rotated_points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_rectangle = np.array([
                [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
            ])
            best_rectangle = np.dot(best_rectangle, rotation_matrix)

    return best_rectangle


def cross_product(v1, v2):
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


def points_in_rotated_rectangle(points, rect_vertices):
    points = np.array(points)
    rect_vertices = np.array(rect_vertices)
    AB = rect_vertices[1] - rect_vertices[0]
    BC = rect_vertices[2] - rect_vertices[1]
    CD = rect_vertices[3] - rect_vertices[2]
    DA = rect_vertices[0] - rect_vertices[3]
    AP = points - rect_vertices[0]
    BP = points - rect_vertices[1]
    CP = points - rect_vertices[2]
    DP = points - rect_vertices[3]
    cross_AB_AP = cross_product(np.tile(AB, (points.shape[0], 1)), AP)
    cross_BC_BP = cross_product(np.tile(BC, (points.shape[0], 1)), BP)
    cross_CD_CP = cross_product(np.tile(CD, (points.shape[0], 1)), CP)
    cross_DA_DP = cross_product(np.tile(DA, (points.shape[0], 1)), DP)
    mask = (
        (cross_AB_AP <= 0) & (cross_BC_BP <= 0) & (cross_CD_CP <= 0) & (cross_DA_DP <= 0)
    ) | (
        (cross_AB_AP >= 0) & (cross_BC_BP >= 0) & (cross_CD_CP >= 0) & (cross_DA_DP >= 0)
    )
    return mask


def expand_rectangle(rect_vertices, ratio):
    center = np.mean(rect_vertices, axis=0)
    return center + (rect_vertices - center) * (1 + ratio)


def bbox_vertices(corner_min, corner_max):
    x0, y0, z0 = corner_min
    x1, y1, z1 = corner_max
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
        [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)


def expand_bbox(corner_min, corner_max, ratio):
    center = (corner_min + corner_max) / 2
    half_extent = (corner_max - corner_min) / 2 * (1 + ratio)
    return center - half_extent, center + half_extent


def points_in_bbox(points, chunk):
    return np.all(points >= chunk[0], axis=-1) & np.all(points <= chunk[1], axis=-1)


def divide_bbox(chunk, points):
    extent = chunk[1] - chunk[0]
    axis = np.argmax(extent)
    best_ratio = 0.5
    best_balance_score = float("inf")
    for ratio in np.linspace(0.3, 0.7, 17):
        split_value = chunk[0, axis] + extent[axis] * ratio
        chunk_1 = chunk.copy()
        chunk_2 = chunk.copy()
        chunk_1[1, axis] = split_value
        chunk_2[0, axis] = split_value
        count_1 = points_in_bbox(points, chunk_1).sum()
        count_2 = points_in_bbox(points, chunk_2).sum()
        if count_1 == 0 or count_2 == 0:
            continue
        balance_score = abs(count_1 - count_2) / max(count_1, count_2)
        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_ratio = ratio
    split_value = chunk[0, axis] + extent[axis] * best_ratio
    chunk_1 = chunk.copy()
    chunk_2 = chunk.copy()
    chunk_1[1, axis] = split_value
    chunk_2[0, axis] = split_value
    return chunk_1, chunk_2


def bbox_projected_rect(corner_min, corner_max, vertical_axis):
    if vertical_axis == "z":
        return np.array([[corner_min[0], corner_min[1]], [corner_max[0], corner_min[1]],
                         [corner_max[0], corner_max[1]], [corner_min[0], corner_max[1]]])
    if vertical_axis == "y":
        return np.array([[corner_min[0], corner_min[2]], [corner_max[0], corner_min[2]],
                         [corner_max[0], corner_max[2]], [corner_min[0], corner_max[2]]])
    if vertical_axis == "x":
        return np.array([[corner_min[1], corner_min[2]], [corner_max[1], corner_min[2]],
                         [corner_max[1], corner_max[2]], [corner_min[1], corner_max[2]]])
    raise ValueError("vertical_axis should be x/y/z")


def select_scene_area(pts_2d, cameras_position_2d):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Interactive Select Scene ROI Region")
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=0.01, label='Points', alpha=0.05)
    plt.scatter(cameras_position_2d[:, 0], cameras_position_2d[:, 1], c='red', s=1.0, label='Camera')
    polygon_points = []
    polygon_line, = ax.plot([], [], 'yo-')

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        polygon_points.append((x, y))
        x_data, y_data = zip(*polygon_points) if len(polygon_points) > 1 else ([x], [y])
        polygon_line.set_data(x_data, y_data)
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            if len(polygon_points) > 2:
                polygon_points.append(polygon_points[0])
                x_data, y_data = zip(*polygon_points)
                polygon_line.set_data(x_data, y_data)
                fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    return np.array(polygon_points)[:-1, :]


def divide_rect(rect, pts_2d_init=None):
    edge_1_length = np.linalg.norm(rect[0] - rect[1])
    edge_2_length = np.linalg.norm(rect[0] - rect[3])
    split_ratio = 0.5
    if pts_2d_init is not None:
        best_balance_score = float("inf")
        for candidate_ratio in np.linspace(0.3, 0.7, 17):
            r1, r2 = _divide_rect_at_ratio(rect, edge_1_length, edge_2_length, candidate_ratio)
            count_1 = points_in_rotated_rectangle(pts_2d_init, r1).sum()
            count_2 = points_in_rotated_rectangle(pts_2d_init, r2).sum()
            if count_1 == 0 or count_2 == 0:
                continue
            score = abs(count_1 - count_2) / max(count_1, count_2)
            if score < best_balance_score:
                best_balance_score = score
                split_ratio = candidate_ratio
    return _divide_rect_at_ratio(rect, edge_1_length, edge_2_length, split_ratio)


def _divide_rect_at_ratio(rect, edge_1_length, edge_2_length, split_ratio):
    if edge_1_length >= edge_2_length:
        mid_pt_1 = rect[0] + (rect[1] - rect[0]) * split_ratio
        mid_pt_2 = rect[3] + (rect[2] - rect[3]) * split_ratio
        return (np.array([rect[0], mid_pt_1, mid_pt_2, rect[3]]),
                np.array([mid_pt_1, rect[1], rect[2], mid_pt_2]))
    else:
        mid_pt_1 = rect[0] + (rect[3] - rect[0]) * split_ratio
        mid_pt_2 = rect[1] + (rect[2] - rect[1]) * split_ratio
        return (np.array([rect[0], rect[1], mid_pt_2, mid_pt_1]),
                np.array([mid_pt_1, mid_pt_2, rect[2], rect[3]]))


def divide_condition(rect, condition_vars_list):
    pts_2d_init, num_points_thresh = condition_vars_list
    mask = points_in_rotated_rectangle(pts_2d_init, rect)
    if mask.sum() > num_points_thresh:
        return False
    print("block init point number:", mask.sum())
    return True


def recursive_split(rectangle, blocks, tree_depth, max_tree_depth, condition_vars_list):
    flag = divide_condition(rectangle, condition_vars_list)
    if flag or tree_depth > max_tree_depth:
        blocks.append(rectangle)
    else:
        pts_2d_init = condition_vars_list[0]
        rect_1, rect_2 = divide_rect(rectangle, pts_2d_init)
        recursive_split(rect_1, blocks, tree_depth+1, max_tree_depth, condition_vars_list)
        recursive_split(rect_2, blocks, tree_depth+1, max_tree_depth, condition_vars_list)


def divide_bbox_condition(chunk, condition_vars_list):
    points3D_xyz, num_points_thresh = condition_vars_list
    mask = points_in_bbox(points3D_xyz, chunk)
    if mask.sum() > num_points_thresh:
        return False
    print("block init point number:", mask.sum())
    return True


def recursive_split_bbox(chunk, blocks, tree_depth, max_tree_depth, condition_vars_list):
    flag = divide_bbox_condition(chunk, condition_vars_list)
    if flag or tree_depth > max_tree_depth:
        blocks.append(chunk)
    else:
        points3D_xyz = condition_vars_list[0]
        chunk_1, chunk_2 = divide_bbox(chunk, points3D_xyz)
        recursive_split_bbox(chunk_1, blocks, tree_depth+1, max_tree_depth, condition_vars_list)
        recursive_split_bbox(chunk_2, blocks, tree_depth+1, max_tree_depth, condition_vars_list)


def scene_partion(cfg, vertical_axis="z"):
    vertical_axis_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis]
    scene = Scene(cfg.scene_dirpath, evaluate=cfg.evaluate, scene_scale=cfg.scene_scale)
    cameras_positions = [np.linalg.inv(scene.views_info[view_id].extrinsic)[:3, 3]
                         for view_id in scene.train_views_id]
    cameras_positions_3d = np.stack(cameras_positions, axis=0)

    pts_bin_path = os.path.join(cfg.scene_dirpath, "sparse/0/points3D.bin")
    points3D = read_points3D_binary_(pts_bin_path, scale=cfg.scene_scale)
    points3D_ids = np.array(list(points3D.keys()))
    points3D_xyz = np.stack([pt.xyz for pt in points3D.values()], axis=0)

    if o3d is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points3D_xyz)
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        point_cloud = point_cloud.select_by_index(ind)
        points = np.asarray(point_cloud.points)
    else:
        # same rule as open3d remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0):
        # the 20 neighbours include the point itself, the threshold uses the sample std (n - 1)
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(points3D_xyz).query(points3D_xyz, k=20)
        mean_d = dist.mean(axis=1)
        points = points3D_xyz[mean_d <= mean_d.mean() + 2.0 * mean_d.std(ddof=1)]

    partition_mode = getattr(cfg, "partition_mode", "2d")
    if partition_mode == "3d":
        max_tree_depth = cfg.max_tree_depth
        num_points_thresh = cfg.num_points_thresh
        camera_images_info = read_extrinsics_binary(
            os.path.join(cfg.scene_dirpath, "sparse/0/images.bin"))

        initial_chunk = np.stack([points.min(axis=0), points.max(axis=0)]).astype(np.float32)
        chunks = []
        # count / balance the same outlier-filtered points the initial box is built from
        condition_vars_list = [points, num_points_thresh]
        recursive_split_bbox(initial_chunk, chunks, 0, max_tree_depth, condition_vars_list)
        print("candidate block number:", len(chunks))

        if vertical_axis == "z":
            plot_pts_2d = np.stack((points[:, 0], points[:, 1]), axis=1)
            plot_cameras_2d = np.stack((cameras_positions_3d[:, 0], cameras_positions_3d[:, 1]), axis=1)
        elif vertical_axis == "y":
            plot_pts_2d = np.stack((points[:, 0], points[:, 2]), axis=1)
            plot_cameras_2d = np.stack((cameras_positions_3d[:, 0], cameras_positions_3d[:, 2]), axis=1)
        else:
            plot_pts_2d = np.stack((points[:, 1], points[:, 2]), axis=1)
            plot_cameras_2d = np.stack((cameras_positions_3d[:, 1], cameras_positions_3d[:, 2]), axis=1)

        plt.figure("Scene Partion Blocks", figsize=(8, 8))
        plt.scatter(plot_pts_2d[:, 0], plot_pts_2d[:, 1], c='black', s=0.01, label='Points', alpha=0.3)
        plt.scatter(plot_cameras_2d[:, 0], plot_cameras_2d[:, 1], c='red', s=0.5, label='Camera')

        blocks_info_dict = {}
        block_counter = 0
        for block_idx, chunk in enumerate(chunks):
            print("###" * 20)
            print("Candidate block idx: {}".format(block_idx))

            corner_min, corner_max = expand_bbox(chunk[0], chunk[1], cfg.expand_ratio)
            mask = points_in_bbox(points3D_xyz, np.stack([corner_min, corner_max]))

            if mask.sum() < 2000:
                print("Too little sparse pcd in this block, filtered.")
                continue

            bbx = bbox_vertices(chunk[0], chunk[1])
            bbx_expand = bbox_vertices(corner_min, corner_max)

            block_points_ids = points3D_ids[mask]
            block_related_image_ids = set()
            for pt_id in block_points_ids:
                block_related_image_ids.update(list(map(int, points3D[pt_id].image_ids)))

            block_image_ids = []
            base_image_ids = []
            border_image_ids = []
            for image_id in list(block_related_image_ids):
                cover_ratio_view = (len(set(camera_images_info[image_id].point3D_ids) & set(block_points_ids))
                                    / len(set(camera_images_info[image_id].point3D_ids)))
                cover_ratio_block = (len(set(camera_images_info[image_id].point3D_ids) & set(block_points_ids))
                                     / len(set(block_points_ids)))
                if cover_ratio_view > cfg.cover_ratio_thresh or cover_ratio_block > 0.2:
                    block_image_ids.append(image_id)
                    if cover_ratio_view > 0.7:
                        base_image_ids.append(image_id)
                    else:
                        border_image_ids.append(image_id)

            print("num views: {}, num pts in block: {}".format(len(block_image_ids), mask.sum()))
            if len(block_image_ids) < 20:
                print("Too little view related to this block, filtered.")
                continue

            block_pcd_filepath = os.path.join(
                cfg.output_dirpath, "block_init_pcd/block_{:03d}_init_pcd.ply".format(block_counter))
            num_pts = fetch_local_pcd(cfg.scene_dirpath, block_pcd_filepath, block_image_ids, cfg.scene_scale)
            print("Selected, block_id: {}, block sparse pcd num: {}, initial point num: {}, num views: {}".format(
                block_counter, mask.sum(), num_pts, len(block_image_ids)))

            plot_rectangle(bbox_projected_rect(chunk[0], chunk[1], vertical_axis),
                           color=np.random.rand(3), text=None)

            blocks_info_dict[block_counter] = {
                "bbx": bbx.tolist(),
                "bbx_expand": bbx_expand.tolist(),
                "views_id": list(block_image_ids),
                "base_views_id": list(base_image_ids),
                "border_image_ids": list(border_image_ids),
                "num_views": len(block_image_ids),
                "num_base_views": len(base_image_ids),
                "num_border_views": len(border_image_ids),
                "num_pts": len(block_points_ids.tolist()),
                "scene_scale": cfg.scene_scale,
                "block_pcd_filepath": block_pcd_filepath
            }
            block_counter += 1

        print("Scene partion finished, valid block num: {}".format(block_counter))
        blocks_info_dict["num_blocks"] = block_counter
        json_filepath = os.path.join(cfg.output_dirpath, "blocks_info.json")
        with open(json_filepath, "w") as json_file:
            json.dump(blocks_info_dict, json_file, indent=4)

        plt.axis("equal")
        plt.axis("off")
        plt.savefig(os.path.join(cfg.output_dirpath, "Partition_Results.png"))
        plt.show()
        return

    # ── 2D mode ────────────────────────────────────────────────────────────────
    if vertical_axis == "z":
        cameras_position_2d = np.stack((cameras_positions_3d[:, 0], cameras_positions_3d[:, 1]), axis=1)
        pts_2d = np.stack((points[:, 0], points[:, 1]), axis=1)
        pts_2d_init = np.stack((points3D_xyz[:, 0], points3D_xyz[:, 1]), axis=1)
    elif vertical_axis == "y":
        cameras_position_2d = np.stack((cameras_positions_3d[:, 0], cameras_positions_3d[:, 2]), axis=1)
        pts_2d = np.stack((points[:, 0], points[:, 2]), axis=1)
        pts_2d_init = np.stack((points3D_xyz[:, 0], points3D_xyz[:, 2]), axis=1)
    else:
        cameras_position_2d = np.stack((cameras_positions_3d[:, 1], cameras_positions_3d[:, 2]), axis=1)
        pts_2d = np.stack((points[:, 1], points[:, 2]), axis=1)
        pts_2d_init = np.stack((points3D_xyz[:, 1], points3D_xyz[:, 2]), axis=1)

    polygon_pts = select_scene_area(pts_2d, cameras_position_2d)
    bounding_rect = minimum_area_bounding_rectangle(polygon_pts)

    plt.figure("Scene ROI Aera", figsize=(8, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=0.01, alpha=0.05)
    plt.scatter(cameras_position_2d[:, 0], cameras_position_2d[:, 1], c='red', s=0.5)
    plot_rectangle(bounding_rect, color=[0, 1, 0])
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(os.path.join(cfg.output_dirpath, "ROI_region.png"))
    plt.show()

    plt.figure("Scene Partion Blocks", figsize=(8, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='black', s=0.01, alpha=0.3)
    plt.scatter(cameras_position_2d[:, 0], cameras_position_2d[:, 1], c='red', s=0.5)

    max_tree_depth = cfg.max_tree_depth
    num_points_thresh = cfg.num_points_thresh
    camera_images_info = read_extrinsics_binary(
        os.path.join(cfg.scene_dirpath, "sparse/0/images.bin"))

    block_rects = []
    condition_vars_list = [pts_2d_init, num_points_thresh]
    recursive_split(bounding_rect, block_rects, 0, max_tree_depth, condition_vars_list)
    print("candidate block number:", len(block_rects))

    blocks_info_dict = {}
    block_counter = 0
    for block_idx, block_rect in enumerate(block_rects):
        print("###" * 20)
        print("Candidate block idx: {}".format(block_idx))

        rect_expand = expand_rectangle(block_rect, ratio=cfg.expand_ratio)
        mask = points_in_rotated_rectangle(pts_2d_init, rect_expand)

        if mask.sum() < 2000:
            print("Too little sparse pcd in this block, filtered.")
            continue

        pts_block = points3D_xyz[mask]
        vmin = np.percentile(pts_block[:, vertical_axis_idx], 1)
        vmax = np.percentile(pts_block[:, vertical_axis_idx], 99)
        vmax = vmax + (vmax - vmin) * 0.2
        vmin = vmin - (vmax - vmin) * 0.2
        bbx = np.zeros((8, 3))
        bbx[:4, :2], bbx[:4, 2] = block_rect, np.array([vmin] * 4)
        bbx[4:, :2], bbx[4:, 2] = block_rect, np.array([vmax] * 4)
        bbx_expand = np.zeros((8, 3))
        bbx_expand[:4, :2], bbx_expand[:4, 2] = rect_expand, np.array([vmin] * 4)
        bbx_expand[4:, :2], bbx_expand[4:, 2] = rect_expand, np.array([vmax] * 4)
        if vertical_axis == "x":
            bbx[:, [0, 1, 2]] = bbx[:, [2, 0, 1]]
            bbx_expand[:, [0, 1, 2]] = bbx_expand[:, [2, 0, 1]]
        elif vertical_axis == "y":
            bbx[:, [0, 1, 2]] = bbx[:, [0, 2, 1]]
            bbx_expand[:, [0, 1, 2]] = bbx_expand[:, [0, 2, 1]]

        block_points_ids = points3D_ids[mask]
        block_related_image_ids = set()
        for pt_id in block_points_ids:
            block_related_image_ids.update(list(map(int, points3D[pt_id].image_ids)))

        block_image_ids = []
        base_image_ids = []
        border_image_ids = []
        for image_id in list(block_related_image_ids):
            cover_ratio_view = (len(set(camera_images_info[image_id].point3D_ids) & set(block_points_ids))
                                / len(set(camera_images_info[image_id].point3D_ids)))
            cover_ratio_block = (len(set(camera_images_info[image_id].point3D_ids) & set(block_points_ids))
                                 / len(set(block_points_ids)))
            if cover_ratio_view > cfg.cover_ratio_thresh or cover_ratio_block > 0.1:
                block_image_ids.append(image_id)
                if cover_ratio_view > 0.7:
                    base_image_ids.append(image_id)
                else:
                    border_image_ids.append(image_id)

        print("num views: {}, num pts in block: {}".format(len(block_image_ids), mask.sum()))
        if len(block_image_ids) < 20:
            print("Too little view related to this block, filtered.")
            continue

        block_pcd_filepath = os.path.join(
            cfg.output_dirpath, "block_init_pcd/block_{:03d}_init_pcd.ply".format(block_counter))
        num_pts = fetch_local_pcd(cfg.scene_dirpath, block_pcd_filepath, block_image_ids, cfg.scene_scale)
        print("Selected, block_id: {}, block sparse pcd num: {}, initial point num: {}, num views: {}".format(
            block_counter, mask.sum(), num_pts, len(block_image_ids)))
        plot_rectangle(block_rect, color=np.random.rand(3), text=None)

        blocks_info_dict[block_counter] = {
            "bbx": bbx.tolist(),
            "bbx_expand": bbx_expand.tolist(),
            "views_id": list(block_image_ids),
            "base_views_id": list(base_image_ids),
            "border_image_ids": list(border_image_ids),
            "num_views": len(block_image_ids),
            "num_base_views": len(base_image_ids),
            "num_border_views": len(border_image_ids),
            "num_pts": len(block_points_ids.tolist()),
            "scene_scale": cfg.scene_scale,
            "block_pcd_filepath": block_pcd_filepath
        }
        block_counter += 1

    print("Scene partion finished, valid block num: {}".format(block_counter))
    blocks_info_dict["num_blocks"] = block_counter
    json_filepath = os.path.join(cfg.output_dirpath, "blocks_info.json")
    with open(json_filepath, "w") as json_file:
        json.dump(blocks_info_dict, json_file, indent=4)

    plt.axis("equal")
    plt.axis("off")
    plt.savefig(os.path.join(cfg.output_dirpath, "Partition_Results.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction Process of View-based Gaussian Splatting.")
    parser.add_argument("--scene_dirpath", "-s", type=str,
                        default="./datasets/ISRO/undistorted/train",
                        help="scene data dirpath")
    parser.add_argument("--output_dirpath", "-o", type=str, default=None,
                        help="optimized result output dirpath")
    parser.add_argument("--evaluate", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--vertical_axis", type=str, default="z")
    parser.add_argument("--max_tree_depth", type=int, default=6)
    parser.add_argument("--num_points_thresh", type=int, default=400000)
    parser.add_argument("--cover_ratio_thresh", type=float, default=0.3)
    parser.add_argument("--expand_ratio", type=float, default=0.05)
    parser.add_argument("--partition_mode", type=str, default="3d",
                        help="partition mode: '2d' (interactive) or '3d' (BSP, OceanXL default)")
    args = parser.parse_args()

    if args.output_dirpath is None:
        parser.error("--output_dirpath / -o is required")

    os.makedirs(args.output_dirpath, exist_ok=True)
    scene_partion(args, vertical_axis=args.vertical_axis)
