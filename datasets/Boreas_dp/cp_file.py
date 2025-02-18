import shutil
import os
from multiprocessing import Pool
import subprocess

# def process_sequence(seq_ID):
#     source_seq_path = os.path.join(source_dataset_root, seq_ID)
#     source_seq_lidar_path = os.path.join(source_seq_path, "lidar")
#     target_seq_path = os.path.join(target_dataset_root, seq_ID)
#     os.makedirs(target_seq_path, exist_ok=True)
#     target_seq_lidar_path = os.path.join(target_seq_path, "lidar")
#     os.makedirs(target_seq_lidar_path, exist_ok=True)
#     lidar_list = sorted(os.listdir(source_seq_lidar_path))
#     for curr_lidar_name in lidar_list:
#         source_curr_lidar = os.path.join(source_seq_lidar_path, curr_lidar_name)
#         target_curr_lidar = os.path.join(target_seq_lidar_path, curr_lidar_name)
#         if os.path.exists(target_curr_lidar):
#             continue
#         print("Copying {} to {}".format(source_curr_lidar, target_curr_lidar))
#         shutil.copyfile(source_curr_lidar, target_curr_lidar)

def process_sequence(seq_ID):
    print("Processing sequence {}".format(seq_ID))
    source_seq_path = os.path.join(source_dataset_root, seq_ID)
    target_seq_path = os.path.join(target_dataset_root, seq_ID)
    os.makedirs(target_seq_path, exist_ok=True)
    cp_cmd = "cp -r {}/* {}/".format(source_seq_path, target_seq_path)
    print(cp_cmd)
    subprocess.run(cp_cmd, shell=True)
    print("Done copying sequence {}".format(seq_ID))

def process_sequence_v2(seq_ID):
    print("Processing sequence {}".format(seq_ID))
    source_seq_path = os.path.join(source_dataset_root, seq_ID)
    target_seq_path = os.path.join(target_dataset_root, seq_ID, seq_ID)
    os.makedirs(target_seq_path, exist_ok=True)
    # cp_cmd = "mv {}/* {}".format(target_seq_path, source_seq_path)
    # print(cp_cmd)
    # subprocess.run(cp_cmd, shell=True)
    del_cmd = "rm -r {}".format(target_seq_path)
    print(del_cmd)
    subprocess.run(del_cmd, shell=True)
    print("Done copying sequence {}".format(seq_ID))
        

if __name__ == "__main__":
    source_dataset_root = "/DATA1/Boreas_minuse"
    target_dataset_root = "/DATA1/Boreas_minuse"
    os.makedirs(target_dataset_root, exist_ok=True)
    sequence_list = sorted(os.listdir(source_dataset_root))
    
    # for seq_list in sequence_list:
    #     process_sequence(seq_list)
    with Pool() as p:
        p.map(process_sequence_v2, sequence_list)