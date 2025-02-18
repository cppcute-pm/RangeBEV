import os
import pickle
import numpy as np
import cv2
import sys
import torch
sys.path.append('/home/pengjianyi/code_projects/Depth-Anything-V2-main')
from depth_anything_v2.dpt import DepthAnythingV2
sys.path.remove('/home/pengjianyi/code_projects/Depth-Anything-V2-main')


data_root = '/DATA5'
source_dir = 'Boreas_224x224_image'
source_path = os.path.join(data_root, source_dir)
target_dir = 'Boreas_minuse_image_depth_postprocess'
target_path = os.path.join(data_root, target_dir)
os.makedirs(target_path, exist_ok=True)
sequence_list = sorted(os.listdir(source_path))
device = torch.device('cuda:1')
torch.cuda.set_device(device)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'
model = DepthAnythingV2(**model_configs[encoder])
model_ckpt_path = '/home/pengjianyi/.cache/torch/hub/checkpoints/depth_anything_v2_vitl.pth'
model.load_state_dict(torch.load(model_ckpt_path, map_location='cpu'))
model = model.to(device).eval()

for sequence in sequence_list:
    source_sequence_path = os.path.join(source_path, sequence)
    target_sequence_path = os.path.join(target_path, sequence)
    os.makedirs(target_sequence_path, exist_ok=True)
    source_sequence_camera_path = os.path.join(source_sequence_path, 'camera')
    target_sequence_camera_path = os.path.join(target_sequence_path, 'camera')
    os.makedirs(target_sequence_camera_path, exist_ok=True)
    file_name_list = sorted(os.listdir(source_sequence_camera_path))
    for file_name in file_name_list:
        target_file_path = os.path.join(target_sequence_camera_path, file_name)
        if os.path.exists(target_file_path):
           print('skip', target_file_path)
           continue 
        file_path = os.path.join(source_sequence_camera_path, file_name)
        raw_img = cv2.imread(file_path)
        depth = model.infer_image(raw_img)
        print(np.max(depth))
        depth_output = (((np.max(depth) + 3.0) - depth) / (np.max(depth) + 3.0) * 65535).astype(np.uint16)
        cv2.imwrite(target_file_path, depth_output)
        print('save', target_file_path)