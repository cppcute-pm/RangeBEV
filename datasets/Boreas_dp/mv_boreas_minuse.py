import os
import subprocess
root_dir = '/media/group1/data/pengjianyi/Boreas_minuse'

loc_train = [
    ["boreas-2020-11-26-13-58"],
    ["boreas-2020-12-01-13-26"],
    ["boreas-2020-12-18-13-44"],
    ["boreas-2021-01-15-12-17"],
    ["boreas-2021-01-19-15-08"],
    ["boreas-2021-01-26-11-22"],
    ["boreas-2021-02-02-14-07"],
    ["boreas-2021-03-02-13-38"],
    ["boreas-2021-03-23-12-43"],
    ["boreas-2021-03-30-14-23"],
    ["boreas-2021-04-08-12-44"],
    ["boreas-2021-04-13-14-49"],
    ["boreas-2021-04-15-18-55"],
    ["boreas-2021-04-20-14-11"],
    ["boreas-2021-04-29-15-55"],
    ["boreas-2021-05-06-13-19"],
    ["boreas-2021-05-13-16-11"],
    ["boreas-2021-06-03-16-00"],
    ["boreas-2021-06-17-17-52"],
    ["boreas-2021-08-05-13-34"],
    ["boreas-2021-09-02-11-42"],
    ["boreas-2021-09-07-09-35"],
    ["boreas-2021-10-15-12-35"],
    ["boreas-2021-10-22-11-36"],
    ["boreas-2021-11-02-11-16"],
    ["boreas-2021-11-14-09-47"],
    ["boreas-2021-11-16-14-10"],
    ["boreas-2021-11-23-14-27"],
]

for seq_list in loc_train:
    seq_ID = seq_list[0]
    seq_path = os.path.join(root_dir, seq_ID)
    seq_camera_path = os.path.join(seq_path, 'camera')
    seq_lidar_path = os.path.join(seq_path, 'lidar')
    camera_rm_cmd = f'rm -rf {seq_camera_path}/*'
    lidar_rm_cmd = f'rm -rf {seq_lidar_path}/*'
    subprocess.run(camera_rm_cmd, shell=True)
    subprocess.run(lidar_rm_cmd, shell=True)
    print(f'{seq_ID} cleaned')