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


def prepare_image_data(path, save_file_name, dataset_name, img_h=96, img_w=128, img_ch=1):
    sys.stdout.flush()

    file_ids = next(os.walk(path))[2]
    data = np.zeros((len(file_ids), img_h, img_w, img_ch), dtype=np.uint8)

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids)):
        # Read image files iteratively
        img = imread(path + id_)[:, :, :img_ch]
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        data[n] = img

    save_path = STORAGE_PATH + dataset_name + '/' + save_file_name
    np.save(save_path, data)
    print("{0}.npy has been saved at {1} ".format(save_file_name, STORAGE_PATH + dataset_name))


def prepare_mask_data(path, save_file_name, dataset_name, img_h=96, img_w=128):
    sys.stdout.flush()

    file_ids = next(os.walk(path))[2]
    data = np.zeros((len(file_ids), img_h, img_w, 1), dtype=np.bool)
    mask_sizes = []

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids)):
        # Read corresponding mask files iteratively
        mask = np.zeros((img_h, img_w, 1), dtype=np.bool)

        mask_ = imread(path + id_)
        mask_sizes.append([mask.shape[0], mask.shape[1]])
        # Expand individual mask dimensions
        mask_ = np.expand_dims(resize(mask_, (img_h, img_w), mode='constant', preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)
        data[n] = mask

    save_path = STORAGE_PATH + dataset_name + '/' + save_file_name
    np.save(save_path, data)
    print("{0}.npy has been saved at {1} ".format(save_file_name, STORAGE_PATH + dataset_name))

    np.save(save_path + '_size', mask_sizes, allow_pickle=True)
    print("{0}.npy has been saved at {1} ".format(save_file_name + '_size', STORAGE_PATH + dataset_name))

def get_train_data(dataset_name, img_h=96, img_w=128, img_ch=1):
    prepare_image_data(TRAIN_PATH_IMAGES, 'X_train', dataset_name, img_h=img_h, img_w=img_w, img_ch=img_ch)
    prepare_mask_data(TRAIN_PATH_MASKS, 'Y_train', dataset_name, img_h=img_h, img_w=img_w)


def load_train_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_train = np.load(file_path + '/' + 'X_train.npy')
    Y_train = np.load(file_path + '/' + 'Y_train.npy')

    if is_dots_expanded:
        Y_train = expand_labels(Y_train, distance=expand_size)

    return {
        'train_data': X_train,
        'train_label_data': Y_train
    }


def get_test_data(dataset_name, img_h=96, img_w=128, img_ch=1):
    prepare_image_data(TEST_PATH_IMAGES, 'X_test', dataset_name, img_h=img_h, img_w=img_w, img_ch=img_ch)
    prepare_mask_data(TEST_PATH_MASKS, 'Y_test', dataset_name, img_h=img_h, img_w=img_w)


def load_test_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_test = np.load(file_path + '/' + 'X_test.npy')
    Y_test = np.load(file_path + '/' + 'Y_test.npy')

    if is_dots_expanded:
        Y_test = expand_labels(Y_test, distance=expand_size)

    return {
        'test_data': X_test,
        'test_label_data': Y_test
    }


def get_validation_data(dataset_name, img_h=96, img_w=128, img_ch=1):
    prepare_image_data(VALI_PATH_IMAGES, 'X_val', dataset_name, img_h=img_h, img_w=img_w, img_ch=img_ch)
    prepare_mask_data(VALI_PATH_MASKS, 'Y_val', dataset_name, img_h=img_h, img_w=img_w)

def load_val_data(dataset_name, is_dots_expanded=True, expand_size=1):
    file_path = STORAGE_PATH + dataset_name

    X_val = np.load(file_path + '/' + 'X_val.npy')
    Y_val = np.load(file_path + '/' + 'Y_val.npy')

    if is_dots_expanded:
        Y_val = expand_labels(Y_val, distance=expand_size)

    return {
        'val_data': X_val,
        'val_label_data': Y_val
    }