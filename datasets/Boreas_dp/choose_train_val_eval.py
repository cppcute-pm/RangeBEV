import os
import tqdm
import pickle
import numpy as np
from PIL import Image
from multiprocessing import Pool
from typing import Dict
from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from mini_boreas import BoreasDataset_U
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


if __name__ == '__main__':

    data_root = '/media/data/pengjianyi/Boreas'
    dataset = BoreasDataset_U(root=data_root, verbose=True)
    minuse_lidar_idxs_path = os.path.join(data_root, 'my_tool', 'minuse_lidar_idxs.pickle')
    minuse_lidar = pickle.load(open(minuse_lidar_idxs_path, 'rb'))
    lidar_pose_x = []
    lidar_pose_y = []
    for sq in dataset.sequences:
        for lidar_id in minuse_lidar[sq.ID]:
            curr_lidar = sq.lidar_frames[lidar_id]
            curr_lidar_pose_x = curr_lidar.pose[0, 3]
            curr_lidar_pose_y = curr_lidar.pose[1, 3]
            lidar_pose_x.append(curr_lidar_pose_x)
            lidar_pose_y.append(curr_lidar_pose_y)
    x1, x2 = [-150.0, -50.0]
    y1, y2 = [0.0, 100.0]
    x3, x4 = [-420.0, -380.0]
    y3, y4 = [950.0, 1100.0]
    x5, x6 = [-1200.0, -1100.0]
    y5, y6 = [950.0, 1050.0]
    x7, x8 = [-950.0, -830.0]
    y7, y8 = [1950.0, 2100.0]
    lidar_pose_x_array = np.array(lidar_pose_x)
    lidar_pose_y_array = np.array(lidar_pose_y)
    lidar_pose_x_flag = np.logical_and(lidar_pose_x_array > x1, lidar_pose_x_array < x2)
    lidar_pose_y_flag = np.logical_and(lidar_pose_y_array > y1, lidar_pose_y_array < y2)
    lidar_pose_flag = np.logical_and(lidar_pose_x_flag, lidar_pose_y_flag)
    total_num = lidar_pose_flag.shape[0]
    chosen_num_1 = np.count_nonzero(lidar_pose_flag)
    lidar_pose_x_flag = np.logical_and(lidar_pose_x_array > x3, lidar_pose_x_array < x4)
    lidar_pose_y_flag = np.logical_and(lidar_pose_y_array > y3, lidar_pose_y_array < y4)
    lidar_pose_flag = np.logical_and(lidar_pose_x_flag, lidar_pose_y_flag)
    chosen_num_2 = np.count_nonzero(lidar_pose_flag)
    lidar_pose_x_flag = np.logical_and(lidar_pose_x_array > x5, lidar_pose_x_array < x6)
    lidar_pose_y_flag = np.logical_and(lidar_pose_y_array > y5, lidar_pose_y_array < y6)
    lidar_pose_flag = np.logical_and(lidar_pose_x_flag, lidar_pose_y_flag)
    chosen_num_3 = np.count_nonzero(lidar_pose_flag)
    lidar_pose_x_flag = np.logical_and(lidar_pose_x_array > x7, lidar_pose_x_array < x8)
    lidar_pose_y_flag = np.logical_and(lidar_pose_y_array > y7, lidar_pose_y_array < y8)
    lidar_pose_flag = np.logical_and(lidar_pose_x_flag, lidar_pose_y_flag)
    chosen_num_4 = np.count_nonzero(lidar_pose_flag)
    print('total_num:', total_num)
    print('chosen_num_1:', chosen_num_1)
    print('chosen_num_2:', chosen_num_2)
    print('chosen_num_3:', chosen_num_3)
    print('chosen_num_4:', chosen_num_4)
    plt.scatter(lidar_pose_x, lidar_pose_y, s=float(1/10000), color='red')
    # 在图中添加竖直线
    rectangle1 = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue')
    plt.gca().add_patch(rectangle1)
    rectangle2 = Rectangle((x3, y3), x4-x3, y4-y3, fill=False, edgecolor='green')
    plt.gca().add_patch(rectangle2)
    rectangle3 = Rectangle((x5, y5), x6-x5, y6-y5, fill=False, edgecolor='red')
    plt.gca().add_patch(rectangle3)
    rectangle4 = Rectangle((x7, y7), x8-x7, y8-y7, fill=False, edgecolor='yellow')
    plt.gca().add_patch(rectangle4)
    # plt.axvline(x=x1, color='blue')
    # plt.axvline(x=x2, color='blue')
    # plt.axvline(x=x3, color='green')
    # plt.axvline(x=x4, color='green')
    # plt.axvline(x=x5, color='red')
    # plt.axvline(x=x6, color='red')
    # plt.axvline(x=x7, color='yellow')
    # plt.axvline(x=x8, color='yellow')

    # # 在图中添加水平线
    # plt.axhline(y=y1, color='blue')
    # plt.axhline(y=y2, color='blue')
    # plt.axhline(y=y3, color='green')
    # plt.axhline(y=y4, color='green')
    # plt.axhline(y=y5, color='red')
    # plt.axhline(y=y6, color='red')
    # plt.axhline(y=y7, color='yellow')
    # plt.axhline(y=y8, color='yellow')
    plt.xlabel('Longitude (x)')
    plt.ylabel('Latitude (y)')
    plt.title('Lidar global pose')
    plt.savefig('/home/test5/code_project/visualization/lidar_pose.png', bbox_inches='tight', pad_inches=0, dpi=200)




