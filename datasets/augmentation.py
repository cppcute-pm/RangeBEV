# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import math
from scipy.linalg import expm, norm
import random
import torch

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional
import cv2
from albumentations import RandomResizedCrop
from typing import Any
from copy import deepcopy
from warnings import warn
import copy

def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop

def camera_matrix_scaling(K: np.ndarray, sx: float, sy: float):
    K_scale = np.copy(K)
    K_scale[0, 2] *= sx
    K_scale[0, 0] *= sx
    K_scale[1, 2] *= sy
    K_scale[1, 1] *= sy
    return K_scale

class RandomResizedCrop_intrinscs(RandomResizedCrop):

    def apply(
        self,
        img: np.ndarray,
        crop_height: int = 0,
        crop_width: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        interpolation: int = cv2.INTER_LINEAR,
        **params: Any,
    ) -> list:
        params_used = {}
        params_used['crop_height'] = crop_height
        params_used['crop_width'] = crop_width
        params_used['h_start'] = h_start
        params_used['w_start'] = w_start
        params_used['raw_height'] = img.shape[0]
        params_used['raw_width'] = img.shape[1]
        params_used['new_height'] = self.size[0]
        params_used['new_width'] = self.size[1]
        img_out = super(RandomResizedCrop_intrinscs, self).apply(img, crop_height, crop_width, h_start, w_start, interpolation, **params)
        return img_out, params_used

class Resize_intrinscs(A.Resize):

    def apply(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR, **params: Any) -> list:
        params_used = {}
        params_used['raw_height'] = img.shape[0]
        params_used['raw_width'] = img.shape[1]
        params_used['new_height'] = self.height
        params_used['new_width'] = self.width
        img_out = super(Resize_intrinscs, self).apply(img, interpolation, **params)
        
        return img_out, params_used

class LongestMaxSize_intrinscs(A.LongestMaxSize):

    def apply(self, img: np.ndarray, max_size: int, **params: Any) -> list:
        params_used = {}
        params_used['raw_height'] = img.shape[0]
        params_used['raw_width'] = img.shape[1]
        scale = max_size / float(max(img.shape[:2]))
        new_height, new_width = tuple(round(dim * scale) for dim in img.shape[:2])
        params_used['new_height'] = new_height
        params_used['new_width'] = new_width
        img_out = super(LongestMaxSize_intrinscs, self).apply(img, max_size, **params)
        return img_out, params_used
    
class SmallestMaxSize_intrinscs(A.SmallestMaxSize):
    
    def apply(self, img: np.ndarray, max_size: int, **params: Any) -> list:
        params_used = {}
        params_used['raw_height'] = img.shape[0]
        params_used['raw_width'] = img.shape[1]
        scale = max_size / float(min(img.shape[:2]))
        new_height, new_width = tuple(round(dim * scale) for dim in img.shape[:2])
        params_used['new_height'] = new_height
        params_used['new_width'] = new_width
        img_out = super(SmallestMaxSize_intrinscs, self).apply(img, max_size, **params)
        return img_out, params_used

class Crop_intrinscs(A.Crop):

    def apply(self, img: np.ndarray, **params: Any) -> list:
        params_used = {}
        params_used['raw_height'] = img.shape[0]
        params_used['raw_width'] = img.shape[1]
        params_used['y_min'] = self.y_min
        params_used['x_min'] = self.x_min
        img_out = super(Crop_intrinscs, self).apply(img, **params)
        return img_out, params_used

class PCTransform:
    def __init__(self, aug_mode, num_points=None):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = None
        elif self.aug_mode == 1:
            t = [JitterPoints(sigma=0.001, clip=0.002), 
                 RemoveRandomPoints(r=(0.0, 0.1), return_mask=False),
                 RandomTranslation(max_delta=0.01), 
                 RemoveRandomBlock(p=0.4, return_mask=False),
                 PointShuffle(shuffle_indices=False)]
        elif self.aug_mode == 2:
            t = [JitterPoints(sigma=0.002, clip=0.004), 
                 RemoveRandomPoints(r=(0.0, 0.1), return_mask=False),
                 RandomTranslation(max_delta=0.02), 
                 RemoveRandomBlock(p=0.4, return_mask=False),
                 PointSample(num_points=num_points, sample_range=0.2, replace=True),
                 PointShuffle(shuffle_indices=False)]
        elif self.aug_mode == 3:
            t = [PointShuffle()]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        if t is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class PCTransform_Pose:
    def __init__(self, aug_mode, num_points=None):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = None
        elif self.aug_mode == 1:
            t = [JitterPoints(sigma=0.01, clip=0.02), 
                 RemoveRandomPoints(r=(0.0, 0.3)),
                 RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 RemoveRandomBlock(p=0.3),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 2:
            t = [JitterPoints(sigma=0.01, clip=0.02), 
                 RemoveRandomPoints(r=(0.0, 0.3)),
                 RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 RemoveRandomBlock(p=0.3),
                 PointSample(num_points=num_points, sample_range=150.0, replace=True),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 3:
            t = [PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 4:
            t = [RemoveRandomPoints(r=(0.0, 0.3)),
                 RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 RemoveRandomBlock(p=0.3),
                 PointSample(num_points=num_points, sample_range=150.0, replace=True),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 5:
            t = [RemoveRandomPoints(r=(0.0, 0.3)),
                 RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 RemoveRandomBlock(p=0.3),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 6:
            t = [RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 RemoveRandomBlock(p=0.3),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 7:
            t = [RemoveRandomPoints(r=(0.0, 0.3)),
                RandomTransform(P_tx_amplitude=10.0, 
                                 P_ty_amplitude=10.0, 
                                 P_tz_amplitude=10.0, 
                                 P_Rx_amplitude=0.1 * math.pi, 
                                 P_Ry_amplitude=0.1 * math.pi, 
                                 P_Rz_amplitude=2.0 * math.pi), 
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 8:
            t = [RemoveRandomPoints(r=(0.0, 0.3)),
                 RemoveRandomBlock(p=0.3),
                 PointShuffle(shuffle_indices=True)]
        elif self.aug_mode == 9:
            t = [RemoveRandomPoints(r=(0.0, 0.01)),
                 RemoveRandomBlock(p=0.01),
                 PointShuffle(shuffle_indices=True)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        if t is None:
            self.transform = None
        else:
            self.transform = t

    def __call__(self, e):
        P_random = None
        P_remove_mask = np.zeros(e.shape[0], dtype=np.bool_)
        shuffle_indices = np.arange(e.shape[0], dtype=np.int64)
        if self.transform is not None:
            for curr_transform in self.transform:
                if isinstance(curr_transform, RandomTransform):
                    e, P_random = curr_transform(e)
                elif isinstance(curr_transform, RemoveRandomBlock) or isinstance(curr_transform, RemoveRandomPoints):
                    e, mask = curr_transform(e)
                    P_remove_mask = P_remove_mask | mask
                elif isinstance(curr_transform, PointShuffle):
                    e, shuffle_indices = curr_transform(e)
                    if np.count_nonzero(P_remove_mask) != 0:
                        P_remove_mask = P_remove_mask[shuffle_indices] 
                else:
                    e = curr_transform(e)
        return e, P_random, P_remove_mask, shuffle_indices


class RGB_intrinscs_Transform:
    def __init__(self, aug_mode, image_size, crop_location=None, max_size=None):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 ToTensorV2()]
        elif self.aug_mode == 1:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 2:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Resize_intrinscs(image_size[0], image_size[1]),
                 ToTensorV2()]
        elif self.aug_mode == 3:
            t = [Resize_intrinscs(image_size[0], image_size[1]),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 4:
            t = [ToTensorV2()]
        elif self.aug_mode == 5:
            t = [Resize_intrinscs(image_size[0], image_size[1]),
                 A.Normalize([0.5], [0.5], 1.),
                 ToTensorV2()]
        elif self.aug_mode == 6:
            t = [Resize_intrinscs(image_size[0], image_size[1], cv2.INTER_NEAREST),
                 A.Normalize([0.5], [0.5], 14.),
                 ToTensorV2()]
        elif self.aug_mode == 7:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Resize_intrinscs(image_size[0], image_size[1]),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 8:
            t = [A.GaussNoise(var_limit=(0.0, 0.01)),
                 A.GaussianBlur(blur_limit=(3, 5)),
                 RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 A.Normalize([0.5], [0.5], 1.),
                 ToTensorV2()]
        elif self.aug_mode == 9:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 ToTensorV2()]
        elif self.aug_mode == 10:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 11:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 Resize_intrinscs(image_size[0], image_size[1]),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 12:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 RandomResizedCrop_intrinscs(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 ToTensorV2()]
        elif self.aug_mode == 13:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 Resize_intrinscs(image_size[0], image_size[1]),
                 ToTensorV2()]
        elif self.aug_mode == 14:
            t = [Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 Resize_intrinscs(image_size[0], image_size[1]),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 15:
            t = [Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 Resize_intrinscs(image_size[0], image_size[1]),]
        elif self.aug_mode == 16:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()],
        elif self.aug_mode == 17:
            t = [A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()],
        elif self.aug_mode == 18:
            t = [Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True)]
        elif self.aug_mode == 19:
            t = [Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 LongestMaxSize_intrinscs(max_size),]
        elif self.aug_mode == 20:
            t = [Crop_intrinscs(x_min=crop_location['x_min'], 
                                y_min=crop_location['y_min'], 
                                x_max=crop_location['x_max'], 
                                y_max=crop_location['y_max'], 
                                always_apply=True), 
                 SmallestMaxSize_intrinscs(max_size),]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        if t is None:
            self.transform = None
        else:
            self.transform = t

    def __call__(self, e, intrinscs=None):
        if intrinscs is not None:
            if self.transform is not None:
                for curr_transform in self.transform:
                    if isinstance(curr_transform, RandomResizedCrop_intrinscs): 
                        e, params = curr_transform(image=e)["image"]
                        params['aug_type'] = 'RandomResizedCrop'
                        intrinscs = self.__intrinscs_transform__(intrinscs, params)
                    elif (isinstance(curr_transform, Resize_intrinscs) 
                          or isinstance(curr_transform, LongestMaxSize_intrinscs)
                          or isinstance(curr_transform, SmallestMaxSize_intrinscs)):
                        e, params = curr_transform(image=e)["image"]
                        params['aug_type'] = 'Resize'
                        intrinscs = self.__intrinscs_transform__(intrinscs, params)
                    elif isinstance(curr_transform, Crop_intrinscs):
                        e, params = curr_transform(image=e)["image"]
                        params['aug_type'] = 'Crop'
                        intrinscs = self.__intrinscs_transform__(intrinscs, params)
                    else:
                        e = curr_transform(image=e)["image"]
            return e, intrinscs
        else:
            if self.transform is not None:
                for curr_transform in self.transform:
                    if isinstance(curr_transform, RandomResizedCrop_intrinscs): 
                        e, params = curr_transform(image=e)["image"]
                    elif isinstance(curr_transform, Resize_intrinscs):
                        e, params = curr_transform(image=e)["image"]
                    elif isinstance(curr_transform, LongestMaxSize_intrinscs):
                        e, params = curr_transform(image=e)["image"]
                    elif isinstance(curr_transform, SmallestMaxSize_intrinscs):
                        e, params = curr_transform(image=e)["image"]
                    elif isinstance(curr_transform, Crop_intrinscs):
                        e, params = curr_transform(image=e)["image"]
                    else:
                        e = curr_transform(image=e)["image"]
            return e
        
                
    
    def __intrinscs_transform__(self, intrinscs, params):
        if params['aug_type'] == 'RandomResizedCrop':
            crop_height = params["crop_height"]
            crop_width = params['crop_width']
            h_start = params['h_start']
            w_start = params['w_start']
            raw_height = params['raw_height']
            raw_width = params['raw_width']
            new_height = params['new_height']
            new_width = params['new_width']
            dy = int((raw_height - crop_height + 1) * h_start)
            dx = int((raw_width - crop_width + 1) * w_start)
            intrinscs = camera_matrix_cropping(intrinscs, dx, dy)
            intrinscs = camera_matrix_scaling(intrinscs, new_width * 1.0 / crop_width, new_height * 1.0 / crop_height)
        elif params['aug_type'] == 'Resize':
            raw_height = params['raw_height']
            raw_width = params['raw_width']
            new_height = params['new_height']
            new_width = params['new_width']
            intrinscs = camera_matrix_scaling(intrinscs, new_width * 1.0 / raw_width, new_height * 1.0 / raw_height)
        elif params['aug_type'] == 'Crop':
            raw_height = params['raw_height']
            raw_width = params['raw_width']
            y_min = params['y_min']
            x_min = params['x_min']
            dy = y_min
            dx = x_min
            intrinscs = camera_matrix_cropping(intrinscs, dx, dy)
        return intrinscs


class RGBTransform:
    def __init__(self, aug_mode, image_size):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 ToTensorV2()]
        elif self.aug_mode == 1:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 2:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 A.Resize(image_size[0], image_size[1]),
                 ToTensorV2()]
        elif self.aug_mode == 3:
            t = [A.Resize(image_size[0], image_size[1]),
                 A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 4:
            t = [ToTensorV2()]
        elif self.aug_mode == 5:
            t = [A.Resize(image_size[0], image_size[1]),
                 A.Normalize([0.5], [0.5], 1.),
                 ToTensorV2()]
        elif self.aug_mode == 6:
            t = [A.Resize(image_size[0], image_size[1], cv2.INTER_NEAREST),
                 A.Normalize([0.5], [0.5], 14.),
                 ToTensorV2()]
        elif self.aug_mode == 7:
            t = [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 A.Resize(image_size[0], image_size[1]),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ToTensorV2()]
        elif self.aug_mode == 8:
            t = [A.GaussNoise(var_limit=(0.0, 0.01)),
                 A.GaussianBlur(blur_limit=(3, 5)),
                 A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                 A.Normalize([0.5], [0.5], 1.),
                 ToTensorV2()]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = A.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(image=e)
        return e["image"]



class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class PointSample:
    def __init__(self, 
                 num_points: int,
                 sample_range: Optional[float] = None,
                 replace: bool = False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def __call__(self, points):

        if self.num_points is not None:
            if not self.replace:
                replace = (points.shape[0] < self.num_points)
            point_range = range(len(points))
            if self.sample_range is not None and not replace:
                # Only sampling the near points when len(points) >= num_samples
                dist = np.linalg.norm(points.numpy(), axis=1)
                far_inds = np.where(dist >= self.sample_range)[0]
                near_inds = np.where(dist < self.sample_range)[0]
                # in case there are too many far points
                if len(far_inds) > self.num_points:
                    far_inds = np.random.choice(
                        far_inds, self.num_points, replace=False)
                point_range = near_inds
                self.num_points -= len(far_inds)
            choices = np.random.choice(point_range, self.num_points, replace=replace)
            if self.sample_range is not None and not replace:
                choices = np.concatenate((far_inds, choices))
                # Shuffle points after sampling
                np.random.shuffle(choices)
            return points[choices]
        else:
            return points


class PointShuffle:
    def __init__(self, shuffle_indices=False):
        self.shuffle_indices = shuffle_indices

    def __call__(self, points):
        if not self.shuffle_indices:
            np.random.shuffle(points)
            return points
        else:
            indices = np.arange(points.shape[0], dtype=np.float32)
            indices = np.expand_dims(indices, axis=-1)
            points_indices_mixed = np.concatenate((points, indices), axis=1)
            np.random.shuffle(points_indices_mixed)
            return points_indices_mixed[:, :3], points_indices_mixed[:, 3].astype(np.int64)


class RandomRotation_Rmat:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
            R_mat = np.linalg.inv(R)
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n
            R_mat = np.linalg.inv(R_n) @ np.linalg.inv(R)

        return coords, R_mat

class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords

class RandomTranslation_trans:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32), trans.astype(np.float32)

class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)

class RandomTransform:

    def __init__(self, P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                 P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        self.P_tx_amplitude = P_tx_amplitude
        self.P_ty_amplitude = P_ty_amplitude
        self.P_tz_amplitude = P_tz_amplitude
        self.P_Rx_amplitude = P_Rx_amplitude
        self.P_Ry_amplitude = P_Ry_amplitude
        self.P_Rz_amplitude = P_Rz_amplitude
    
    def __call__(self, pc):

        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
                random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
                random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                    random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                    random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t
        P_random = P_random.astype(np.float32)
        pc = np.concatenate((pc, np.ones((pc.shape[0], 1), dtype=np.float32)), axis=1) # (N, 4)
        pc = np.dot(pc, P_random.T)
        pc = pc[:, 0:3]
        return pc, P_random

class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
            Nx3 array, original batch of point clouds
            Return:
            Nx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = np.random.choice([0, 1], size=sample_shape, p=[1 - self.p, self.p])
        else:
            m = np.ones(sample_shape, dtype=np.int64)

        mask = m == 1
        jitter = self.sigma * np.random.randn(*e[mask].shape)

        if self.clip is not None:
            jitter = np.clip(jitter, a_min=-self.clip, a_max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r, return_mask=True):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)
        self.return_mask = return_mask

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = np.random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        mask_out = np.zeros((n,), dtype=np.bool_)
        mask_out[mask] = True
        e[mask] = np.zeros_like(e[mask], dtype=np.float32)
        if self.return_mask:
            return e, mask_out
        else:
            return e

class RemoveRandomPoints_numreduce:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = np.random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e_out = e[~mask, :]
        return e_out

class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), return_mask=True):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.return_mask = return_mask

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.reshape(-1, 3)
        min_coords = np.min(flattened_coords, axis=0)
        max_coords = np.max(flattened_coords, axis=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = np.zeros_like(coords[mask], np.float32)
            if self.return_mask:
                return coords, mask
            else:
                return coords
        else:
            mask = np.zeros(coords.shape[0], dtype=np.bool_)
            if self.return_mask:
                return coords, mask
            else:
                return coords

class RemoveRandomBlock_numreduce:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.reshape(-1, 3)
        min_coords = np.min(flattened_coords, axis=0)
        max_coords = np.max(flattened_coords, axis=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords_out = coords[~mask, :]
            return coords_out
        else:
            return coords


class TrainSetTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        self.transform = None
        if aug_mode == 0:
            t = None
        elif aug_mode == 1:
            t = [RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(aug_mode))
        if t is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)



#点云随机旋转
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


# 含法向量
def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


#z方向点云随机旋转
def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

#欧拉角随机旋转
def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


# 含法向量
def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


#指定角度旋转点云

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data