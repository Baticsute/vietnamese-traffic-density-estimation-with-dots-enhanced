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
    return (img - img.mean()) / img.std()

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


def prepare_and_save_data(image_path, mask_path, data_type, dataset_name, img_h=96, img_w=128, img_ch=1):
    sys.stdout.flush()

    file_ids = next(os.walk(image_path))[2]
    # X_train, Y_train or X_test, Y_test etc.
    image_files_type = 'X_' + data_type
    mask_files_type = 'Y_' + data_type

    image_data = np.zeros((len(file_ids), img_h, img_w, img_ch), dtype=np.float32)
    mask_data = np.zeros((len(file_ids), img_h, img_w, 1), dtype=np.float32)
    count_data = np.zeros((len(file_ids), 1), dtype=np.float32)
    mask_sizes = []

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc='Image data preparing ..'):
        # Read image files iteratively
        img = imread(image_path + id_)[:, :, :img_ch]
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        image_data[n] = preprocess_img(img)

    save_path = STORAGE_PATH + dataset_name + '/' + image_files_type
    np.save(save_path, image_data)
    print("{0}.npy has been saved at {1} ".format(image_files_type, STORAGE_PATH + dataset_name))

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc='Mask data preparing ..'):
        # Read corresponding mask files iteratively
        mask_file_name = os.path.splitext(id_)[0] + '.png'
        mask = np.zeros((img_h, img_w))

        mask_ = imread(mask_path + mask_file_name, as_gray=True)
        mask_sizes.append([mask_.shape[0], mask_.shape[1]])
        count_data[n] = np.array([np.count_nonzero(mask_)])

        # original size div to scale size
        mask = mapping_rescale_dot(mask, mask_)

        mask_data[n] = mask.reshape((img_h, img_w, 1))

    save_path = STORAGE_PATH + dataset_name + '/' + mask_files_type
    np.save(save_path, mask_data)
    print("{0}.npy has been saved at {1} ".format(mask_files_type, STORAGE_PATH + dataset_name))
    np.save(save_path + '_size', mask_sizes, allow_pickle=True)
    print("{0}_size.npy has been saved at {1} ".format(mask_files_type, STORAGE_PATH + dataset_name))
    np.save(save_path + '_count', count_data)
    print("{0}_count.npy has been saved at {1} ".format(mask_files_type, STORAGE_PATH + dataset_name))


def load_train_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_train = np.load(file_path + '/' + 'X_train.npy')
    Y_train = np.load(file_path + '/' + 'Y_train.npy')
    Y_train_count = np.load(file_path + '/' + 'Y_train_count.npy')

    if is_dots_expanded:
        Y_train = expand_labels(Y_train, distance=expand_size)

    return {
        'train_data': X_train,
        'train_label_data': Y_train,
        'train_count_label_data': Y_train_count
    }

def load_test_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_test = np.load(file_path + '/' + 'X_test.npy')
    Y_test = np.load(file_path + '/' + 'Y_test.npy')
    Y_test_count = np.load(file_path + '/' + 'Y_test_count.npy')

    if is_dots_expanded:
        Y_test = expand_labels(Y_test, distance=expand_size)

    return {
        'test_data': X_test,
        'test_label_data': Y_test,
        'test_count_label_data': Y_test_count
    }

def load_val_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_val = np.load(file_path + '/' + 'X_val.npy')
    Y_val = np.load(file_path + '/' + 'Y_val.npy')
    Y_val_count = np.load(file_path + '/' + 'Y_val_count.npy')

    if is_dots_expanded:
        Y_val = expand_labels(Y_val, distance=expand_size)

    return {
        'val_data': X_val,
        'val_label_data': Y_val,
        'val_count_label_data': Y_val_count
    }