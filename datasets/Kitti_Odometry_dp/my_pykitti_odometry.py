from pykitti import odometry
from pykitti.utils import read_calib_file
import os
import numpy as np

class my_odometry(odometry):

    def __init__(self, base_path, sequence, pose_path, **kwargs):
        """Set the path."""
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', sequence)
        self.pose_path = os.path.join(pose_path, 'sequences', sequence)
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_poses()
    
    def _load_poses(self):
        """Load ground truth poses (T_w_cam0) from file."""
        pose_file = os.path.join(self.pose_path, 'poses.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.frames is not None:
                    lines = [lines[i] for i in self.frames]

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  self.sequence + '.')

        self.poses = poses
    
    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = read_calib_file(calib_filepath)

        Tr = np.reshape(filedata['Tr'], (3, 4))

        data['T_ego_LiDAR'] = np.concatenate([Tr, np.array([[0, 0, 0, 1]])], axis=0)

        P_rect_00 = np.reshape(filedata['P0'], (3, 4)) # left grayscale camera
        P_rect_10 = np.reshape(filedata['P1'], (3, 4)) # right grayscale camera
        P_rect_20 = np.reshape(filedata['P2'], (3, 4)) # left color camera
        P_rect_30 = np.reshape(filedata['P3'], (3, 4)) # right color camera

        data['cam0_K'] = P_rect_00[0:3, 0:3]  
        data['cam1_K'] = P_rect_10[0:3, 0:3]  
        data['cam2_K'] = P_rect_20[0:3, 0:3]  
        data['cam3_K'] = P_rect_30[0:3, 0:3]

        data['T_ego_cam0'] = np.eye(4)
        fx = data['cam0_K'][0, 0]
        fy = data['cam0_K'][1, 1]
        cx = data['cam0_K'][0, 2]
        cy = data['cam0_K'][1, 2]
        tz = P_rect_00[2, 3]
        tx = (P_rect_00[0, 3] - cx * tz) / fx
        ty = (P_rect_00[1, 3] - cy * tz) / fy
        data['T_ego_cam0'][0:3, 3] = np.asarray([tx, ty, tz])

        data['T_ego_cam1'] = np.eye(4)
        fx = data['cam1_K'][0, 0]
        fy = data['cam1_K'][1, 1]
        cx = data['cam1_K'][0, 2]
        cy = data['cam1_K'][1, 2]
        tz = P_rect_10[2, 3]
        tx = (P_rect_10[0, 3] - cx * tz) / fx
        ty = (P_rect_10[1, 3] - cy * tz) / fy
        data['T_ego_cam1'][0:3, 3] = np.asarray([tx, ty, tz])

        data['T_ego_cam2'] = np.eye(4)
        fx = data['cam2_K'][0, 0]
        fy = data['cam2_K'][1, 1]
        cx = data['cam2_K'][0, 2]
        cy = data['cam2_K'][1, 2]
        tz = P_rect_20[2, 3]
        tx = (P_rect_20[0, 3] - cx * tz) / fx
        ty = (P_rect_20[1, 3] - cy * tz) / fy
        data['T_ego_cam2'][0:3, 3] = np.asarray([tx, ty, tz])

        data['T_ego_cam3'] = np.eye(4)
        fx = data['cam3_K'][0, 0]
        fy = data['cam3_K'][1, 1]
        cx = data['cam3_K'][0, 2]
        cy = data['cam3_K'][1, 2]
        tz = P_rect_30[2, 3]
        tx = (P_rect_30[0, 3] - cx * tz) / fx
        ty = (P_rect_30[1, 3] - cy * tz) / fy
        data['T_ego_cam3'][0:3, 3] = np.asarray([tx, ty, tz])

        self.calib = data
