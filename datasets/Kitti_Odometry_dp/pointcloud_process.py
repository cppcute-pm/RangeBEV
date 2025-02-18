import open3d as o3d
import numpy as np



def voxel_downsample(pointcloud, voxel_grid_downsample_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.asarray(down_pcd.points, dtype=np.float32)  # Nx3
    return down_pcd_points

def FPS_downsample(pointcloud, num_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    down_pcd = pcd.farthest_point_down_sample(num_points)
    down_pcd_points = np.asarray(down_pcd.points, dtype=np.float32)  # Nx3
    return down_pcd_points