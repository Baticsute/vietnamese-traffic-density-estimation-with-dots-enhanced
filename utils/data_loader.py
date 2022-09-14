import os
import sys
import pathlib

import numpy as np

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import scipy
from scipy.ndimage.filters import gaussian_filter

import gc

import cv2

ROOT_PATH = str(pathlib.Path().absolute())
DATA_STORAGE_PATH = '/data_storage/'
STORAGE_PATH = ROOT_PATH + DATA_STORAGE_PATH


def gaussian_filter_density(ground_truth_shape, points, k_nearest=4, beta=0.3, leafsize=2048, fixed_sigma=None):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    density = np.zeros(ground_truth_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=k_nearest, workers=8)

    for i, pt in enumerate(points):
        pt2d = np.zeros(ground_truth_shape, dtype=np.float32)
        if int(pt[1]) < ground_truth_shape[0] and int(pt[0]) < ground_truth_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            if gt_count <= k_nearest:
                sigma = (np.average(distances[i][1: gt_count])) * beta
            else:
                sigma = (np.average(distances[i][1: k_nearest + 1])) * beta
        else:
            sigma = gt_count  # case: 1 point
        if fixed_sigma is not None:
            sigma = fixed_sigma
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def preprocess_img(img):
    """
    Preprocessing for the image
    z-score normalize
    """
    img = img / 255.0

    return img


def mapping_rescale_dot(mask_scale, mask_original):
    scale_factor_h = mask_scale.shape[0] / mask_original.shape[0]
    scale_factor_w = mask_scale.shape[1] / mask_original.shape[1]
    non_zero_points = np.array(np.nonzero(mask_original))
    non_zero_points[0] = non_zero_points[0] * scale_factor_h
    non_zero_points[1] = non_zero_points[1] * scale_factor_w
    non_zero_points = np.transpose(non_zero_points)
    for point in non_zero_points:
        x, y = point[0], point[1]
        mask_scale[x][y] = 1.0

    return mask_scale, non_zero_points


def prepare_and_save_data(
        data_type,
        image_path,
        mask_path,
        dataset_name,
        img_h=96,
        img_w=128,
        img_ch=3
):
    sys.stdout.flush()

    file_ids = next(os.walk(image_path))[2]

    storage_path = STORAGE_PATH + dataset_name + '/' + data_type
    if not os.path.exists(STORAGE_PATH + dataset_name):
        os.makedirs(storage_path + '/images')
        os.makedirs(storage_path + '/masks')
        os.makedirs(storage_path + '/original_shapes')
    else:
        os.makedirs(storage_path + '/images', exist_ok=True)
        os.makedirs(storage_path + '/masks', exist_ok=True)
        os.makedirs(storage_path + '/original_shapes', exist_ok=True)

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

        # original size div to scale size
        mask, points = mapping_rescale_dot(mask, mask_)
        density_map = gaussian_filter_density(mask, np.fliplr(points), k_nearest=3, fixed_sigma=None)
        density_map = density_map.reshape((img_h, img_w, 1))
        file_save_name = storage_path + '/masks/' + os.path.splitext(id_)[0]
        np.save(file_save_name, density_map)

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
        file_ids, val_ids = train_test_split(file_ids, test_size=train_val_split_size, random_state=1996)
        image_val_files_type = 'X_val'
        mask_val_files_type = 'Y_val'

    # X_train, Y_train or X_test, Y_test etc.
    image_files_type = 'X_' + data_type
    mask_files_type = 'Y_' + data_type

    image_data = np.zeros((len(file_ids), img_h, img_w, img_ch), dtype=np.float32)
    mask_data = np.zeros((len(file_ids), img_h, img_w, 1), dtype=np.float32)

    image_val_data = None
    mask_val_data = None
    if val_ids is not None:
        image_val_data = np.zeros((len(val_ids), img_h, img_w, img_ch), dtype=np.float32)
        mask_val_data = np.zeros((len(val_ids), img_h, img_w, 1), dtype=np.float32)

    if not os.path.exists(STORAGE_PATH + dataset_name):
        os.makedirs(STORAGE_PATH + dataset_name)
    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc=data_type + ' image data preparing ..'):
        # Read image files iteratively
        img = imread(image_path + id_)[:, :, :img_ch]
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        image_data[n] = preprocess_img(img)

    save_path = STORAGE_PATH + dataset_name + '/' + image_files_type
    np.save(save_path, image_data)
    print("{0}.npy has been saved at {1} ".format(image_files_type, STORAGE_PATH + dataset_name))
    del image_data
    gc.collect()

    if val_ids is not None:
        for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids), desc='validation image data preparing ..'):
            # Read image files iteratively
            img = imread(image_path + id_)[:, :, :img_ch]
            img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
            image_val_data[n] = preprocess_img(img)

        save_path = STORAGE_PATH + dataset_name + '/' + image_val_files_type
        np.save(save_path, image_val_data)
        print("{0}.npy has been saved at {1} ".format(image_val_files_type, STORAGE_PATH + dataset_name))
        del image_val_data
        gc.collect()

    for n, id_ in tqdm(enumerate(file_ids), total=len(file_ids), desc=data_type + ' mask data preparing ..'):
        # Read corresponding mask files iteratively
        mask_file_name = os.path.splitext(id_)[0] + '.png'
        mask = np.zeros((img_h, img_w))

        mask_ = imread(mask_path + mask_file_name, as_gray=True)

        # original size div to scale size
        mask, points = mapping_rescale_dot(mask, mask_)
        density_map = gaussian_filter_density(mask.shape, np.fliplr(points), k_nearest=3, fixed_sigma=None)

        mask_data[n] = np.expand_dims(density_map, axis=-1)

    save_path = STORAGE_PATH + dataset_name + '/' + mask_files_type
    np.save(save_path, mask_data)
    print("{0}.npy has been saved at {1} ".format(mask_files_type, STORAGE_PATH + dataset_name))
    del mask_data
    gc.collect()

    if val_ids is not None:
        for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids), desc='validation mask data preparing ..'):
            # Read corresponding mask files iteratively
            mask_file_name = os.path.splitext(id_)[0] + '.png'
            mask = np.zeros((img_h, img_w))

            mask_ = imread(mask_path + mask_file_name, as_gray=True)

            # original size div to scale size
            mask, points = mapping_rescale_dot(mask, mask_)
            density_map = gaussian_filter_density(mask.shape, np.fliplr(points), k_nearest=3, fixed_sigma=None)

            mask_val_data[n] = np.expand_dims(density_map, axis=-1)

        save_path = STORAGE_PATH + dataset_name + '/' + mask_val_files_type
        np.save(save_path, mask_val_data)
        print("{0}.npy has been saved at {1} ".format(mask_val_files_type, STORAGE_PATH + dataset_name))
        del mask_val_data
        gc.collect()


def load_bulk_data(dataset_name, section='train'):
    data_file_name = 'X_' + section + '.npy'
    mask_file_name = 'Y_' + section + '.npy'

    data_path = STORAGE_PATH + dataset_name + '/' + data_file_name
    mask_path = STORAGE_PATH + dataset_name + '/' + mask_file_name

    data = np.load(data_path)
    mask = np.load(mask_path)

    return data, mask