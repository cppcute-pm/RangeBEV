import numpy as np
import os
from my_pykitti_odometry import my_odometry
import cv2
import copy
import sys
sys.path.append('/home/pengjianyi/code_projects/CoarseFromFine/datasets')
from augmentation import RGB_intrinscs_Transform
sys.path.remove('/home/pengjianyi/code_projects/CoarseFromFine/datasets')

# import torch.multiprocessing as mp

all_seq_ID_list = ['00', '01', '02', '03', '04', 
                     '05', '06', '07', '08', '09', 
                     '10', '11', '12', '13', '14',
                     '15', '16', '17', '18', '19',
                     '20', '21']

image_transform = RGB_intrinscs_Transform(aug_mode=15,
                                          image_size=[96, 576],
                                          crop_location=dict(
                                              x_min=0,
                                              x_max=1226,
                                              y_min=165,
                                              y_max=370))

def process_sequence(seq_ID):
    print(f"enter {seq_ID}")
    curr_seq = my_odometry(sequence=seq_ID, 
                           base_path=dataset_path, 
                           pose_path=os.path.join(data_path, "semanticKITTI", raw_name))
    target_seq_path = os.path.join(target_path, seq_ID)
    os.makedirs(target_seq_path, exist_ok=True)
    target_seq_image_path = os.path.join(target_seq_path, "image_2")
    os.makedirs(target_seq_image_path, exist_ok=True)
    target_seq_intrinsics_path = os.path.join(target_seq_path, "image_2_intrinsic")
    os.makedirs(target_seq_intrinsics_path, exist_ok=True)

    curr_calib = curr_seq.calib
    curr_cam2_K = curr_calib['cam2_K']
    img_H = 370
    img_W = 1226

    for idx in range(len(curr_seq.timestamps)):
        filename = str(idx).zfill(6)
        save_image_path = os.path.join(target_seq_image_path, filename + ".png")
        save_intrinsics_path = os.path.join(target_seq_intrinsics_path, filename + ".npy")
        if os.path.exists(save_intrinsics_path):
            print(f"pass the {filename} in {seq_ID}")
            continue
        
        image_path = curr_seq.cam2_files[idx]
        curr_image = cv2.imread(image_path)
        curr_rgb_img = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        processed_rgb_image, curr_cam2_K_processed = image_transform(curr_rgb_img, copy.deepcopy(curr_cam2_K))
        processed_bgr_image = cv2.cvtColor(processed_rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_image_path, processed_bgr_image)
        np.save(save_intrinsics_path, curr_cam2_K_processed)
        print(f"saved {filename} in {seq_ID}")

    


if __name__ == "__main__":
    data_path = "/DATA1/pengjianyi"
    dataset_root = os.path.join(data_path, "KITTI")
    raw_name = "dataset"
    target_name = f"576x96_image"
    dataset_path = os.path.join(dataset_root, raw_name)
    target_path = os.path.join(dataset_root, target_name)
    os.makedirs(target_path, exist_ok=True)

    seq_ID_list_inuse = all_seq_ID_list
    for seq_ID in seq_ID_list_inuse:
        process_sequence(seq_ID)