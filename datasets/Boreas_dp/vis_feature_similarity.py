import copy
import numpy as np
import yaml
import os
import torch_scatter
import torch

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d==0.9.0" with python==3.6 to install open3d first.')


learning_map = {
  1: 0,
  5: 0,
  7: 0,
  8: 0,
  10: 0,
  11: 0,
  13: 0,
  19: 0,
  20: 0,
  0: 0,
  29: 0,
  31: 0,
  9: 1,
  14: 2,
  15: 3,
  16: 3,
  17: 4,
  18: 5,
  21: 6,
  2: 7,
  3: 7,
  4: 7,
  6: 7,
  12: 8,
  22: 9,
  23: 10,
  24: 11,
  25: 12,
  26: 13,
  27: 14,
  28: 15,
  30: 16,
}

def _draw_points(points,
                 vis,
                 points_size=1,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyzrgb'):
    """Draw points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    mode = "xyzrgb"
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) # from numpy to open3d
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    o3d.io.write_point_cloud("eg.pcd", pcd)
    return pcd, points_colors


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.
    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path="pic",
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)


    def add_seg_mask(self, seg_mask_colors):
        """Add segmentation mask to visualizer via per-point colorization.
        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        """Visualize the points cloud.
        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


def show_rawdata(pc_path, label_path, pred_path):
    raw_data = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    if label_path is not None:
        raw_label = np.fromfile(label_path, dtype=np.uint8).reshape(-1, 1)
        raw_label = np.vectorize(learning_map.__getitem__)(raw_label)
        pred_label = np.fromfile(pred_path, dtype=np.uint8).reshape(-1, 1)

    else:
        raw_label = np.zeros((len(raw_data), 1)) * 173
        pred_label = np.zeros((len(raw_data), 1)) * 173

    mask = pred_label == raw_label
    colors = []
    for label in mask:
        if label == 1:
            colors.append((173, 173, 173))
        elif label == 0:
            colors.append((255, 0, 0))  # red

    colors = np.vstack(colors)

    if raw_data.shape[1] == 4:
        points = raw_data[:, 0:3]

    else:
        points = raw_data[:, 3:6]
    color_points = np.concatenate((points, colors), axis=1)
    vis_er = Visualizer(color_points)
    vis_er.show(save_path="pic.ply")


# if __name__ == "__main__":
#     pc_path = "xxx.pcd.bin"
#     label_path = "xxx_lidarseg.bin"
#     pred_path = "xxx_lidarseg.bin"
#     show_rawdata(pc_path, label_path, pred_path)

class Visualizer2(object):
    r"""Online visualizer implemented with Open3d.
    Args:
    """

    def __init__(self,
                 ):
        super(Visualizer2, self).__init__()

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame/system
        self.o3d_visualizer.add_geometry(mesh_frame)
    def draw_points(self, points, points_size=2, point_color=[0.5, 0.5, 0.5], mode='xyz', offset=0):
        points = copy.deepcopy(points)
        points[:, 0] += offset
        self.pcd, self.points_colors = _draw_points(
            points, self.o3d_visualizer, points_size, point_color, mode)

    def show(self, save_path=None):
        """Visualize the points cloud.
        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


def caculate_center_points_for_image(original_pc_point, original_pc_2_image, img_W, img_H, device):
    original_pc_point = torch.tensor(original_pc_point, device=device)
    original_pc_2_image = torch.tensor(original_pc_2_image, dtype=torch.int64, device=device)
    # image_num_pt = torch_scatter.scatter_sum(torch.ones_like(original_pc_2_image), original_pc_2_image, dim=-1, dim_size=img_H * img_W)
    image_center_coords = torch_scatter.scatter_mean(original_pc_point, original_pc_2_image, dim=-1, dim_size=img_H * img_W)
    image_center_coords = image_center_coords.cpu().numpy()
    return image_center_coords

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    curr_batch = 0
    i = 0
    j = 0
    once_offset = 300.0
    img_W, img_H = 56, 56
    pc_point = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_pc.npy") # (N, 3)
    pc_similarity = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_2_8231_pc_similarity.npy") # (N, M) M is the number of image features, assume all the similarity is in [0, 1]
    original_pc_point = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_pc_original.npy") # (BigN, 3)
    original_pc_2_curr_points = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_pc_original_pc_2_curr_points.npy") # (BigN)
    original_pc_2_image = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_2_8231_pc_original_pc_2_image.npy") # (BigN)
    image_center_coords = caculate_center_points_for_image(original_pc_point, original_pc_2_image, img_W, img_H, device) # (M, 3)
    pc_overlap_ratio_matrix = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_2_8231_pc_overlap_ratio_matrix.npy") # (M, N)
    img_overlap_ratio_matrix = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_2_8231_img_overlap_ratio_matrix.npy") # (M, N)
    minimum_overlap_ratio_matrix = np.minimum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix) # (M, N)
    maximum_overlap_ratio_matrix = np.maximum(pc_overlap_ratio_matrix, img_overlap_ratio_matrix) # (M, N)
    mean_overlap_ratio_matrix = (pc_overlap_ratio_matrix + img_overlap_ratio_matrix) / 2.0 # (M, N)
    original_pc_2_many_1 = np.load("A:\datasets\CMVPR_visualization\vis1\train_batch9_8231_2_8231_original_pc_2_many_1.npy") # (BigN)

    use_matrix = 'mean'

    assert (img_H * img_W) == len(image_center_coords)

    vis = Visualizer2()
    depth_color = original_pc_point[:, 2:] / (np.max(original_pc_point[:, 2:]) + 1.0)
    original_pc_point_1 = np.concatenate((original_pc_point, 
                                 depth_color, 
                                 depth_color, 
                                 depth_color), axis=-1) # (N, 6)
    vis.draw_points(original_pc_point_1, points_size=2, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=0 * once_offset)

    image_belong_color = np.expand_dims(original_pc_2_image, axis=-1) / (np.max(original_pc_2_image) + 1.0)
    original_pc_point_2 = np.concatenate((original_pc_point,
                                    image_belong_color,
                                    image_belong_color,
                                    image_belong_color), axis=-1) # (BigN, 6)
    vis.draw_points(original_pc_point_2, points_size=2, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=1 * once_offset)
    image_color = np.expand_dims(np.arange(0, img_H * img_W, dtype=np.float32), axis=-1) / (img_H * img_W + 1.0)
    image_center_coords_1 = np.concatenate((image_center_coords,
                                            image_color,
                                            image_color,
                                            image_color), axis=-1) # (M, 6)
    vis.draw_points(image_center_coords_1, points_size=20, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=1 * once_offset)

    node_belong_color = np.expand_dims(original_pc_2_curr_points, axis=-1) / (np.max(original_pc_2_curr_points) + 1.0)
    original_pc_point_3 = np.concatenate((original_pc_point,
                                    node_belong_color,
                                    node_belong_color,
                                    node_belong_color), axis=-1) # (BigN, 6)
    vis.draw_points(original_pc_point_3, points_size=2, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=2 * once_offset)
    node_color = np.expand_dims(np.arange(0, len(pc_point), dtype=np.float32), axis=-1) / (len(pc_point) + 1.0)
    pc_point_1 = np.concatenate((pc_point,
                                node_color,
                                node_color,
                                node_color), axis=-1) # (N, 6)
    vis.draw_points(pc_point_1, points_size=20, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=2 * once_offset)

    if use_matrix == 'mean':
        overlap_matrix_in_use = mean_overlap_ratio_matrix
    elif use_matrix == 'min':
        overlap_matrix_in_use = minimum_overlap_ratio_matrix
    elif use_matrix == 'max':
        overlap_matrix_in_use = maximum_overlap_ratio_matrix
    elif use_matrix == 'pc':
        overlap_matrix_in_use = pc_overlap_ratio_matrix
    elif use_matrix == 'img':
        overlap_matrix_in_use = img_overlap_ratio_matrix
    else:
        raise NotImplementedError
    
    overlap_matrix_in_use_flattened = overlap_matrix_in_use.flatten() # (M * N)
    topk_indices = np.argsort(overlap_matrix_in_use_flattened)[::-1][:10]
    topk_img_indices = topk_indices // len(pc_point)
    topk_pc_indices = topk_indices % len(pc_point)

    for i in range(topk_img_indices.shape[0]):
        curr_node_color = pc_similarity[:, topk_img_indices[i]:]
        curr_node_point = np.concatenate((pc_point,
                                        curr_node_color,
                                        curr_node_color,
                                        curr_node_color), axis=-1)
        vis.draw_points(curr_node_point, points_size=5, point_color=[0.5, 0.5, 0.5], mode='xyzrgb', offset=(3 + i) * once_offset)
        curr_image_point = image_center_coords[topk_img_indices[i]:, :]
        vis.draw_points(curr_image_point, points_size=20, point_color=[0.5, 0.5, 0.5], mode='xyz', offset=(3 + i) * once_offset)
    vis.show(save_path="A:\datasets\CMVPR_visualization\vis1\visualize.npy")
        
    