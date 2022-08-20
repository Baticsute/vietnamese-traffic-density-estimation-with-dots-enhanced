import os
import sys
import pathlib

import numpy as np

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import expand_labels

ROOT_PATH = str(pathlib.Path().absolute())
DATA_STORAGE_PATH = '/data_storage/'
STORAGE_PATH = ROOT_PATH + DATA_STORAGE_PATH

DATASET_PATH = '/datasets/final_data'
FINAL_DATASET_PATH = ROOT_PATH + DATASET_PATH

TRAIN_PATH_IMAGES = FINAL_DATASET_PATH + '/train/images/'
TEST_PATH_IMAGES = FINAL_DATASET_PATH + '/test/images/'
VALI_PATH_IMAGES = FINAL_DATASET_PATH + '/validation/images/'

TRAIN_PATH_MASKS = FINAL_DATASET_PATH + '/train/masks/'
TEST_PATH_MASKS = FINAL_DATASET_PATH + '/test/masks/'
VALI_PATH_MASKS = FINAL_DATASET_PATH + '/validation/masks/'

def preprocess_img(img):
    """
    Preprocessing for the image
    z-score normalize
    """
    # return (img - img.mean()) / img.std()
    return img / 255.0

def preprocess_label(mask):
    """
    Predict whole tumor. If you want to predict tumor sections, then
    just comment this out.
    """
    mask[mask > 0] = 1.0

    return mask

def mapping_rescale_dot(mask_scale, mask_original):
    scale_factor_h = mask_scale.shape[0] / mask_original.shape[0]
    scale_factor_w = mask_scale.shape[1] / mask_original.shape[1]
    non_zero_points = np.array(np.nonzero(mask_original))
    non_zero_points[0] = non_zero_points[0] * scale_factor_h
    non_zero_points[1] = non_zero_points[1] * scale_factor_w
    non_zero_points = np.transpose(non_zero_points)
    for point in non_zero_points:
        x = point[0]
        y = point[1]
        mask_scale[x][y] = 1.0

    return mask_scale


def prepare_and_save_data(data_type, image_path, mask_path, dataset_name, img_h=96, img_w=128, img_ch=1):
    sys.stdout.flush()

    file_ids = next(os.walk(image_path))[2]

    storage_path = STORAGE_PATH + dataset_name + '/' + data_type
    if not os.path.exists(STORAGE_PATH + dataset_name):
        os.makedirs(storage_path + '/images')
        os.makedirs(storage_path + '/masks')
        os.makedirs(storage_path + '/original_shapes')
        os.makedirs(storage_path + '/dot_counts')
    else:
        os.makedirs(storage_path + '/images', exist_ok=True)
        os.makedirs(storage_path + '/masks', exist_ok=True)
        os.makedirs(storage_path + '/original_shapes', exist_ok=True)
        os.makedirs(storage_path + '/dot_counts', exist_ok=True)

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc='Image data preparing ..'):
        # Read image files iteratively
        img = imread(image_path + id_)[:, :, :img_ch]
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        img = preprocess_img(img)
        file_save_name = storage_path + '/images/' + os.path.splitext(id_)[0]
        np.save(file_save_name, preprocess_img(img))

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc='Mask data preparing ..'):
        # Read corresponding mask files iteratively
        mask_file_name = os.path.splitext(id_)[0] + '.png'
        mask = np.zeros((img_h, img_w))

        mask_ = imread(mask_path + mask_file_name, as_gray=True)
        file_save_name = storage_path + '/original_shapes/' + os.path.splitext(id_)[0]
        mask_original_shape = np.array([mask_.shape[0], mask_.shape[1]])
        np.save(file_save_name, mask_original_shape)

        file_save_name = storage_path + '/dot_counts/' + os.path.splitext(id_)[0]
        mask_count = np.array([np.count_nonzero(mask_)])
        np.save(file_save_name, mask_count)


        # original size div to scale size
        mask = mapping_rescale_dot(mask, mask_)
        mask = mask.reshape((img_h, img_w, 1))
        file_save_name = storage_path + '/masks/' + os.path.splitext(id_)[0]
        np.save(file_save_name, preprocess_img(mask))