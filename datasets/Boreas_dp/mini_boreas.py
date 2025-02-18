import os.path as osp
import os
from pyboreas.data.sequence import Sequence
from pyboreas.data.calib import Calib
from pyboreas.data.sensors import Camera, Lidar
from pyboreas import BoreasDataset
from pyboreas.data.splits import loc_train
from multiprocessing import Pool
import multiprocessing


class Sequence_U(Sequence):
    def __init__(self, boreas_root, seqSpec):
        self.ID = seqSpec[0]
        if len(seqSpec) > 2:
            assert (
                seqSpec[2] > seqSpec[1]
            ), "Sequence timestamps must go forward in time"
            self.start_ts = str(seqSpec[1])
            self.end_ts = str(seqSpec[2])
        else:
            self.start_ts = "0"  # dummy start and end if not specified
            self.end_ts = "9" * 21
        self.seq_root = osp.join(boreas_root, self.ID)
        self.applanix_root = osp.join(self.seq_root, "applanix")
        self.calib_root = osp.join(self.seq_root, "calib")
        self.camera_root = osp.join(self.seq_root, "camera")
        self.lidar_root = osp.join(self.seq_root, "lidar")

        self._check_dataroot_valid()  # Check if folder structure correct

        self.calib = Calib(self.calib_root)
        # Creates list of frame objects for cam, lidar, radar, and inits poses
        self.get_all_frames()

        self._check_download()  # prints warning when sensor data missing

    def get_all_frames(self):
        """Convenience method for retrieving sensor frames of all types"""
        cfile = osp.join(self.applanix_root, "camera_poses.csv")
        lfile = osp.join(self.applanix_root, "lidar_poses.csv")
        self.camera_frames = self._get_frames(cfile, self.camera_root, ".png", Camera)
        self.lidar_frames = self._get_frames(lfile, self.lidar_root, ".bin", Lidar)
    
    def _get_frames(self, posefile, root, ext, SensorType):
        """Initializes sensor frame objects with their ground truth pose information
        Args:
            posefile (str): path to ../sensor_poses.csv
            root (str): path to the root of the sensor folder ../sensor/
            ext (str): file extension specific to this sensor type
            SensorType (cls): sensor class specific to this sensor type
        Returns:
            frames (list): list of sensor frame objects
        """
        frames = []
        if osp.exists(posefile):
            with open(posefile, "r") as f:
                f.readline()  # header
                for line in f:
                    data = line.split(",")
                    ts = data[0]
                    if self.start_ts <= ts and ts <= self.end_ts:
                        frame = SensorType(osp.join(root, ts + ext))
                        frame.init_pose(data)
                        frames.append(frame)
        elif osp.isdir(root):
            framenames = sorted([f for f in os.listdir(root) if f.endswith(ext)])
            for framename in framenames:
                ts = framename.split(",")[0]
                if self.start_ts <= ts and ts <= self.end_ts:
                    frame = SensorType(osp.join(root, framename))
                    frames.append(frame)
        return frames
    
    def _check_download(self):
        """Checks if all sensor data has been downloaded, prints a warning otherwise"""
        if osp.isdir(self.camera_root) and len(os.listdir(self.camera_root)) < len(
            self.camera_frames
        ):
            print(f'camera_frames: {len(self.camera_frames)}')
            print(f'camera_root: {len(os.listdir(self.camera_root))}')
            print("WARNING: camera images are not all downloaded: {}".format(self.ID))
        if osp.isdir(self.lidar_root) and len(os.listdir(self.lidar_root)) < len(
            self.lidar_frames
        ):
            print("WARNING: lidar frames are not all downloaded: {}".format(self.ID))
        gtfile = osp.join(self.applanix_root, "gps_post_process.csv")
        if not osp.exists(gtfile):
            print(
                "WARNING: this may be a test sequence, or the groundtruth is not yet available: {}".format(
                    self.ID
                )
            )

    def print(self):
        print("SEQ: {}".format(self.ID))
        if self.end_ts != "9" * 21:
            print("START: {} END: {}".format(self.start_ts, self.end_ts))
        print("camera frames: {}".format(len(self.camera_frames)))
        print("lidar frames: {}".format(len(self.lidar_frames)))
        print("-------------------------------")
    
    @property
    def lidar(self):
        for lidar_frame in self.lidar_frames:
            yield lidar_frame
    
    @property
    def camera(self):
        for camera_frame in self.camera_frames:
            yield camera_frame

class BoreasDataset_U(BoreasDataset):
    def __init__(
        self, root=None, split=loc_train, verbose=False
    ):
        self.root = root
        self.split = split
        self.camera_frames = []
        self.lidar_frames = []
        self.sequences = []
        self.seqDict = {}  # seq string to index
        self.map = None  # TODO: Load the HD map data

        if split is None:
            split = sorted(
                [
                    [f]
                    for f in os.listdir(root)
                    if f.startswith("boreas-") and f != "boreas-test-gt"
                ]
            )

        # It takes a few seconds to construct each sequence, so we parallelize this
        global _load_seq

        def _load_seq(seqSpec):
            return Sequence_U(root, seqSpec)

        pool = Pool(2)
        self.sequences = list(pool.map(_load_seq, split))
        self.sequences.sort(key=lambda x: x.ID)

        for seq in self.sequences:
            self.camera_frames += seq.camera_frames
            self.lidar_frames += seq.lidar_frames
            self.seqDict[seq.ID] = len(self.seqDict)
            if verbose:
                seq.print()

        if verbose:
            print("total camera frames: {}".format(len(self.camera_frames)))
            print("total lidar frames: {}".format(len(self.lidar_frames)))

    def get_camera(self, idx):
        return self.camera_frames[idx]

    def get_lidar(self, idx):
        return self.lidar_frames[idx]