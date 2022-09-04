import os
import sys
import pathlib

import numpy as np

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import gc

import cv2

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
        np.save(file_save_name, img)

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
        np.save(file_save_name, mask)


def prepare_and_save_bulk_data(
        image_path,
        mask_path,
        data_type,
        dataset_name,
        train_val_split_size=0.2,
        img_h=96,
        img_w=128,
        img_ch=1
):
    sys.stdout.flush()

    file_ids = next(os.walk(image_path))[2]
    val_ids = None
    image_val_files_type = None
    mask_val_files_type = None

    if data_type == 'train' and train_val_split_size > 0:
        file_ids, val_ids = train_test_split(file_ids, test_size=train_val_split_size, random_state=2022)
        image_val_files_type = 'X_val'
        mask_val_files_type = 'Y_val'
    # X_train, Y_train or X_test, Y_test etc.
    image_files_type = 'X_' + data_type
    mask_files_type = 'Y_' + data_type

    image_data = np.zeros((len(file_ids), img_h, img_w, img_ch), dtype=np.float32)
    mask_data = np.zeros((len(file_ids), img_h, img_w, 1), dtype=np.float32)
    count_data = np.zeros((len(file_ids), 1), dtype=np.float32)
    mask_sizes = []

    image_val_data = None
    mask_val_data = None
    count_val_data = None
    mask_val_sizes = None
    if val_ids is not None:
        image_val_data = np.zeros((len(val_ids), img_h, img_w, img_ch), dtype=np.float32)
        mask_val_data = np.zeros((len(val_ids), img_h, img_w, 1), dtype=np.float32)
        count_val_data = np.zeros((len(val_ids), 1), dtype=np.float32)
        mask_val_sizes = []

    if not os.path.exists(STORAGE_PATH + dataset_name):
        os.makedirs(STORAGE_PATH + dataset_name)
    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc=data_type + ' image data preparing ..'):
        # Read image files iteratively
        img = imread(image_path + id_)[:, :, :img_ch]
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        image_data[n] = preprocess_img(img)

    save_path = STORAGE_PATH + dataset_name + '/' + image_files_type
    np.save(save_path, image_data)
    del image_data
    gc.collect()
    print("{0}.npy has been saved at {1} ".format(image_files_type, STORAGE_PATH + dataset_name))

    if val_ids is not None:
        for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids), desc='validation image data preparing ..'):
            # Read image files iteratively
            img = imread(image_path + id_)[:, :, :img_ch]
            img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
            image_val_data[n] = preprocess_img(img)

        save_path = STORAGE_PATH + dataset_name + '/' + image_val_files_type
        np.save(save_path, image_val_data)
        del image_val_data
        gc.collect()
        print("{0}.npy has been saved at {1} ".format(image_val_files_type, STORAGE_PATH + dataset_name))

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc=data_type + ' mask data preparing ..'):
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
    del mask_data
    del mask_sizes
    del count_data
    gc.collect()

    if val_ids is not None:
        for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids), desc='validation mask data preparing ..'):
            # Read corresponding mask files iteratively
            mask_file_name = os.path.splitext(id_)[0] + '.png'
            mask = np.zeros((img_h, img_w))

            mask_ = imread(mask_path + mask_file_name, as_gray=True)
            mask_val_sizes.append([mask_.shape[0], mask_.shape[1]])
            count_val_data[n] = np.array([np.count_nonzero(mask_)])

            # original size div to scale size
            mask = mapping_rescale_dot(mask, mask_)

            mask_val_data[n] = mask.reshape((img_h, img_w, 1))

        save_path = STORAGE_PATH + dataset_name + '/' + mask_val_files_type
        np.save(save_path, mask_val_data)
        print("{0}.npy has been saved at {1} ".format(mask_val_files_type, STORAGE_PATH + dataset_name))
        np.save(save_path + '_size', mask_val_sizes, allow_pickle=True)
        print("{0}_size.npy has been saved at {1} ".format(mask_val_files_type, STORAGE_PATH + dataset_name))
        np.save(save_path + '_count', count_val_data)
        print("{0}_count.npy has been saved at {1} ".format(mask_val_files_type, STORAGE_PATH + dataset_name))
        del mask_val_data
        del mask_val_sizes
        del count_val_data
        gc.collect()


def load_train_bulk_data(dataset_name):
    train_data_path = STORAGE_PATH + dataset_name + '/X_train.npy'
    train_mask_path = STORAGE_PATH + dataset_name + '/Y_train.npy'
    val_data_path = STORAGE_PATH + dataset_name + '/X_val.npy'
    val_mask_path = STORAGE_PATH + dataset_name + '/Y_val.npy'

    train_data = np.load(train_data_path)
    train_mask = np.load(train_mask_path)

    val_data = np.load(val_data_path)
    val_mask = np.load(val_mask_path)

    return (train_data, train_mask), (val_data, val_mask)


def load_test_bulk_data(dataset_name):
    test_data_path = STORAGE_PATH + dataset_name + '/X_test.npy'
    test_mask_path = STORAGE_PATH + dataset_name + '/Y_test.npy'

    test_data = np.load(test_data_path)
    test_mask = np.load(test_mask_path)

    return test_data, test_mask
