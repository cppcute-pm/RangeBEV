import cv2
import sys
OR_sdk_path = '../robotcar-dataset-sdk-master/python'
sys.path.append(OR_sdk_path)
from camera_model import CameraModel
from image import load_image
sys.path.remove(OR_sdk_path)

from PIL import Image
import os
from multiprocessing import Pool

def process_traversal(traversal):
    traversal_path = os.path.join(raw_image_dir, traversal, 'stereo', 'centre')
    if not os.path.exists(os.path.join(out_image_dir, traversal)):
        os.makedirs(os.path.join(out_image_dir, traversal))
    if not os.path.exists(os.path.join(out_image_dir, traversal, 'stereo')):
        os.makedirs(os.path.join(out_image_dir, traversal, 'stereo'))
    if not os.path.exists(os.path.join(out_image_dir, traversal, 'stereo', 'centre')):
        os.makedirs(os.path.join(out_image_dir, traversal, 'stereo', 'centre'))
    images_list = sorted(os.listdir(traversal_path))
    for image_name in images_list:
        image_path = os.path.join(traversal_path, image_name)
        img = load_image(image_path, camera_model)
        img = Image.fromarray(img)
        img.save(os.path.join(out_image_dir, traversal, 'stereo', 'centre', image_name))
        print('save image: ', os.path.join(out_image_dir, traversal, 'stereo', 'centre', image_name))

def main():
    traversal_list = sorted(os.listdir(traversal_list_dir))
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    # done_traversal_list = []
    # for traversal in traversal_list:
    #     if os.path.exists(os.path.join(out_image_dir, traversal)):
    #         print(traversal + ' is already done, skip.')
    #         done_traversal_list.append(traversal)
    # if len(done_traversal_list) >= 1:
    #     for done_traversal in done_traversal_list:
    #         traversal_list.remove(done_traversal)
    with Pool() as p:
        p.map(process_traversal, traversal_list)

if __name__ == '__main__':
    raw_image_dir = '/DATA5/Oxford_Robotcar/rgb'
    out_image_dir = '/DATA5/Oxford_Robotcar/real_rgb'
    traversal_list_dir = '/DATA5/Oxford_Robotcar/rgb'
    camera_model = CameraModel("../robotcar-dataset-sdk-master/models", 'stereo_centre')
    main()