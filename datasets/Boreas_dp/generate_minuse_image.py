import os
from mini_boreas import BoreasDataset_U
import pickle
import numpy as np
import cv2
import sys
sys.path.append('/home/pengjianyi/code_projects/CoarseFromFine/datasets')
from augmentation import RGB_intrinscs_Transform
sys.path.remove('/home/pengjianyi/code_projects/CoarseFromFine/datasets')


data_path = '/DATA5'
root_name = 'Boreas_minuse'
root_path = os.path.join(data_path, root_name)
target_name = 'Boreas_224x224_image'
target_path = os.path.join(data_path, target_name)
os.makedirs(target_path, exist_ok=True)
dataset = BoreasDataset_U(root_path)
minuse_lidar_path = os.path.join(root_path, 'my_tool', "minuse_lidar_idxs.pickle")
lidar2image_path = os.path.join(root_path, "my_tool", "lidar2image.pickle")
minuse_lidar = pickle.load(open(minuse_lidar_path, 'rb'))
lidar2image_idx = pickle.load(open(lidar2image_path, 'rb'))
image_transform = RGB_intrinscs_Transform(aug_mode=15,
                                          image_size=[224, 224],
                                          crop_location=dict(
                                              x_min=0,
                                              x_max=2448,
                                              y_min=683,
                                              y_max=1366))

# need the new intrinscs
for sequence in dataset.sequences:
    target_sequence_dir = os.path.join(target_path, sequence.ID)
    os.makedirs(target_sequence_dir, exist_ok=True)
    target_sequence_camera_dir = os.path.join(target_sequence_dir, 'camera')
    os.makedirs(target_sequence_camera_dir, exist_ok=True)
    target_sequence_calib_dir = os.path.join(target_sequence_dir, 'calib')
    os.makedirs(target_sequence_calib_dir, exist_ok=True)
    for lidar_id in minuse_lidar[sequence.ID]:
        curr_image_idxs = lidar2image_idx[sequence.ID][str(lidar_id)]
        curr_image_idx = curr_image_idxs[0]
        curr_image_frame = sequence.camera_frames[curr_image_idx]
        curr_P0 = sequence.calib.P0.astype(np.float32)
        curr_P0 = curr_P0[:3, :3]
        curr_image = cv2.imread(curr_image_frame.path)
        curr_rgb_img = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        _, _, image_file_name = (curr_image_frame.path).split('/')[-3:]
        image_path = os.path.join(target_sequence_camera_dir, image_file_name)
        processed_rgb_image, curr_P0_processed = image_transform(curr_rgb_img, curr_P0)
        processed_bgr_image = cv2.cvtColor(processed_rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, processed_bgr_image)
        intrinsics_path = os.path.join(target_sequence_calib_dir, 'intrinsics.npy')
        np.save(intrinsics_path, curr_P0_processed)
        print(f"Saved {image_path} and {intrinsics_path}")
    print(f"Finished {sequence.ID}")
print("Finished all")
