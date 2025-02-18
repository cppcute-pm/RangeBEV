# For each LiDAR scan in the dataset find the corresponding RGB image based on the timestamp
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



# ***********************************

def find_cross_over(arr, low, high, x):
    if arr[high] <= x:  # x is greater than all
        return high

    if arr[low] > x:  # x is smaller than all
        return low

    # Find the middle point
    mid = (low + high) // 2  # low + (high - low)// 2

    # If x is same as middle element, then return mid
    if arr[mid] <= x and arr[mid + 1] > x:
        return mid

    # If x is greater than arr[mid], then either arr[mid + 1] is ceiling of x or ceiling lies in arr[mid+1...high]
    if arr[mid] < x:
        return find_cross_over(arr, mid + 1, high, x)

    return find_cross_over(arr, low, mid - 1, x)


def find_k_closest(arr, x, k):
    # This function returns indexes of k closest elements to x in arr[]
    n = len(arr)
    # Find the crossover point
    l = find_cross_over(arr, 0, n - 1, x)
    r = l + 1  # Right index to search
    ndx_l = []

    # If x is present in arr[], then reduce left index. Assumption: all elements in arr[] are distinct
    if arr[l] == x:
        l -= 1

    # Compare elements on left and right of crossover point to find the k closest elements
    while l >= 0 and r < n and len(ndx_l) < k:
        if x - arr[l] < arr[r] - x:
            ndx_l.append(l)
            l -= 1
        else:
            ndx_l.append(r)
            r += 1

    # If there are no more elements on right side, then print left elements
    while len(ndx_l) < k and l >= 0:
        ndx_l.append(l)
        l -= 1

    # If there are no more elements on left side, then print right elements
    while len(ndx_l) < k and r < n:
        ndx_l.append(r)
        r += 1

    return ndx_l


if __name__ == '__main__':

    # only make the correspndings
    print('enter the main function')
    nn_threshold = 1000  # Nearest neighbour threshold in miliseconds
    k = 50               # Number of nearest neighbour images to find for each LiDAR scan
    data_root = '/media/data/pengjianyi/Boreas'
    dataset = BoreasDataset_U(root=data_root, verbose=True)
    print('finish loading the dataset')
    timestamp2ind = {}
    cam_timestamps = {}
    for curr_sq in dataset.sequences:
        timestamp2ind[curr_sq.ID] = {}
        cam_timestamps[curr_sq.ID] = []
        camera_iter = curr_sq.get_camera_iter()
        for idx, curr_cam in enumerate(camera_iter):
            timestamp2ind[curr_sq.ID][str(curr_cam.timestamp_micro)] = idx
            cam_timestamps[curr_sq.ID].append(curr_cam.timestamp_micro)
    print('finish reverse indexing the timestamp')
    minuse_lidar_idxs_path = os.path.join(data_root, 'my_tool', 'minuse_lidar_idxs.pickle')
    minuse_lidar = pickle.load(open(minuse_lidar_idxs_path, 'rb'))
    print('finish loading the minuse_lidar')
    lidar2image = {}
    for sq in dataset.sequences:
        lidar2image[sq.ID] = {}
        cam_timestamps[sq.ID] = sorted(cam_timestamps[sq.ID])
        curr_cam_timestamps = np.array(cam_timestamps[sq.ID], dtype=np.int64)
        curr_minuse_lidar = minuse_lidar[sq.ID]
        curr_lidar_timestamps = []
        for id in curr_minuse_lidar:
            curr_lidar_timestamp = sq.lidar_frames[id].timestamp_micro
            curr_lidar_timestamps.append(curr_lidar_timestamp)
        curr_lidar_timestamps = np.array(curr_lidar_timestamps, dtype=np.int64)
        indices = np.searchsorted(curr_cam_timestamps, curr_lidar_timestamps)
        for i in range(len(indices)):
            image_timestamps = curr_cam_timestamps[max(0, indices[i] - 2 * k):min(len(curr_cam_timestamps), indices[i] + 2 * k)]
            nn_ndx = find_k_closest(list(image_timestamps), curr_lidar_timestamps[i], k)
            nn_ts = image_timestamps[nn_ndx]
            image_idxs = []
            for ts in nn_ts:
                image_id = timestamp2ind[sq.ID][str(ts)]
                image_idxs.append(image_id)
            lidar2image[sq.ID][str(curr_minuse_lidar[i])] = image_idxs

    lidar2img_pickle = 'lidar2image.pickle'
    filepath = os.path.join(data_root, 'my_tool', lidar2img_pickle)
    with open(filepath, 'wb') as f:
        pickle.dump(lidar2image, f)