import os
import sys
import pathlib

import numpy as np

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import gc
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers


ROOT_PATH = str(pathlib.Path().absolute())
DATA_STORAGE_PATH = '/data_storage/'
STORAGE_PATH = ROOT_PATH + DATA_STORAGE_PATH


def gaussian_filter_density(mask_img, k_nearest=4, beta=0.3, leafsize=2048, fixed_sigma=None):
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
    ground_truth_shape = mask_img.shape
    points = np.transpose(np.nonzero(mask_img))
    points = np.fliplr(points)
    density = np.zeros(ground_truth_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    distances = None
    if fixed_sigma is None:
        # build kdtree
        tree = KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=k_nearest, workers=16)

    for i, pt in enumerate(points):
        pt2d = np.zeros(ground_truth_shape, dtype=np.float32)
        if int(pt[1]) < ground_truth_shape[0] and int(pt[0]) < ground_truth_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = mask_img[int(pt[1]), int(pt[0])]
        else:
            continue

        if fixed_sigma is not None:
            sigma = fixed_sigma
        else:
            if gt_count > 1:
                if gt_count <= k_nearest:
                    sigma = (np.average(distances[i][1: gt_count])) * beta
                else:
                    sigma = (np.average(distances[i][1: k_nearest + 1])) * beta
            else:
                sigma = gt_count  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')

    return density

def dot_illusion(ground_truth_shape, points, gap_spaces=(5,5)):
    illusion_map = np.zeros(ground_truth_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return illusion_map

    for i, pt in enumerate(points):
        pt2d = np.zeros(ground_truth_shape, dtype=np.float32)

        center = (pt[0], int(pt[1]))
        gap_left = (int(pt[0]), int(pt[1] - gap_spaces[1]))
        gap_right = (int(pt[0]), int(pt[1] + gap_spaces[1]))
        gap_top = (int(pt[0] - gap_spaces[0]), int(pt[1]))
        gap_bottom = (int(pt[0] + gap_spaces[0]), int(pt[1]))

        if center[0] < ground_truth_shape[0] and center[1] < ground_truth_shape[1]:
            pt2d[center[0], center[1]] += 0.5
        else:
            continue

        if gap_left[0] < ground_truth_shape[0] and gap_left[1] < ground_truth_shape[1]:
             pt2d[gap_left[0], gap_left[1]] += 0.125
        else:
            pt2d[center[0], center[1]] += 0.125

        if gap_right[0] < ground_truth_shape[0] and gap_right[1] < ground_truth_shape[1]:
             pt2d[gap_right[0], gap_right[1]] += 0.125
        else:
            pt2d[center[0], center[1]] += 0.125

        if gap_top[0] < ground_truth_shape[0] and gap_top[1] < ground_truth_shape[1]:
             pt2d[gap_top[0], gap_top[1]] += 0.125
        else:
            pt2d[center[0], center[1]] += 0.125

        if gap_bottom[0] < ground_truth_shape[0] and gap_bottom[1] < ground_truth_shape[1]:
             pt2d[gap_bottom[0], gap_bottom[1]] += 0.125
        else:
            pt2d[center[0], center[1]] += 0.125

        illusion_map = illusion_map + pt2d

    return illusion_map

def mapping_rescale_dot(mask_scale, mask_original):
    scale_factor_h = mask_scale.shape[0] / mask_original.shape[0]
    scale_factor_w = mask_scale.shape[1] / mask_original.shape[1]
    non_zero_points = np.array(np.nonzero(mask_original))
    non_zero_points[0] = non_zero_points[0] * scale_factor_h
    non_zero_points[1] = non_zero_points[1] * scale_factor_w
    non_zero_points = np.transpose(non_zero_points)
    for point in non_zero_points:
        mask_scale[point[0]][point[1]] = 1.0

    return mask_scale, non_zero_points

def generate_density_maps_from_groundtruths(
        dataset_name,
        is_dot_illusion=False,
        gap_spaces=(5,5),
        k_nearest=3, beta=0.3, leafsize=2048,
        fixed_sigma=None
):
    path = ROOT_PATH + '/datasets/' + dataset_name

    sys.stdout.flush()
    train_mask_path_string = path + '/train/masks/'
    test_mask_path_string = path + '/test/masks/'
    validation_mask_path_string = path + '/val/masks/'

    train_dm_path_string = path + '/train/density_maps/'
    test_dm_path_string = path + '/test/density_maps/'
    validation_dm_path_string = path + '/val/density_maps/'

    if not os.path.exists(train_dm_path_string):
        os.makedirs(train_dm_path_string, exist_ok=True)

    if not os.path.exists(validation_dm_path_string):
        os.makedirs(validation_dm_path_string, exist_ok=True)

    if not os.path.exists(test_dm_path_string):
        os.makedirs(test_dm_path_string, exist_ok=True)

    train_mask_file_ids = next(os.walk(train_mask_path_string))[2]
    validation_mask_file_ids = []
    test_mask_file_ids = next(os.walk(test_mask_path_string))[2]
    if os.path.exists(validation_mask_path_string):
        validation_mask_file_ids = next(os.walk(validation_mask_path_string))[2]

    minimum_height = 480
    minimum_width = 640
    for n, id in tqdm(
            enumerate(train_mask_file_ids),
            total=len(train_mask_file_ids),
            desc='Train density maps data preparing ..'
    ):
        file_save_name = os.path.splitext(id)[0]
        mask = imread(train_mask_path_string + id, as_gray=True)
        mask_resized = np.zeros((minimum_height, minimum_width))

        # original size div to scale size
        mask, points = mapping_rescale_dot(mask_resized, mask)
        if is_dot_illusion:
            mask = dot_illusion(ground_truth_shape=mask.shape, points=points, gap_spaces=gap_spaces)
        dm = gaussian_filter_density(
            mask,
            k_nearest=k_nearest,
            beta=beta,
            leafsize=leafsize,
            fixed_sigma=fixed_sigma
        )
        dm = np.expand_dims(dm, axis=-1)

        np.save(train_dm_path_string + file_save_name, dm)

    for n, id in tqdm(
            enumerate(validation_mask_file_ids),
            total=len(validation_mask_file_ids),
            desc='Validation density maps data preparing ..'
    ):
        file_save_name = os.path.splitext(id)[0]
        mask = imread(validation_mask_path_string + id, as_gray=True)
        mask_resized = np.zeros((minimum_height, minimum_width))

        # original size div to scale size
        mask, points = mapping_rescale_dot(mask_resized, mask)
        if is_dot_illusion:
            mask = dot_illusion(ground_truth_shape=mask.shape, points=points, gap_spaces=gap_spaces)
        dm = gaussian_filter_density(
            mask,
            k_nearest=k_nearest,
            beta=beta,
            leafsize=leafsize,
            fixed_sigma=fixed_sigma
        )
        dm = np.expand_dims(dm, axis=-1)

        np.save(validation_dm_path_string + file_save_name, dm)

    for n, id in tqdm(
            enumerate(test_mask_file_ids),
            total=len(test_mask_file_ids),
            desc='Test density maps data preparing ..'
    ):
        file_save_name = os.path.splitext(id)[0]
        mask = imread(test_mask_path_string + id, as_gray=True)
        mask_resized = np.zeros((minimum_height, minimum_width))

        # original size div to scale size
        mask, points = mapping_rescale_dot(mask_resized, mask)
        if is_dot_illusion:
            mask = dot_illusion(ground_truth_shape=mask.shape, points=points, gap_spaces=gap_spaces)
        dm = gaussian_filter_density(
            mask,
            k_nearest=k_nearest,
            beta=beta,
            leafsize=leafsize,
            fixed_sigma=fixed_sigma
        )
        dm = np.expand_dims(dm, axis=-1)

        np.save(test_dm_path_string + file_save_name, dm)


TARGET_TYPE = tf.dtypes.float32


def load_image(path):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, TARGET_TYPE)
    minimum_width = 640
    minimum_height = 480
    if image.shape[0] == minimum_height and image.shape[1] == minimum_width:
        return image
    image = tf.image.resize(
        image,
        size=[minimum_height, minimum_width],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False
    )

    return image

def load_density_map(path):
    dm = np.load(path.numpy().decode())
    dm = tf.convert_to_tensor(dm, dtype=TARGET_TYPE)

    return dm

def set_shapes(img):
    ## https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-573663963
    img.set_shape((480, 640, 1))
    return img

def load_mask(path):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.image.convert_image_dtype(image, TARGET_TYPE)
    minimum_width = 640
    minimum_height = 480
    image = tf.image.resize(
        image,
        size=[minimum_height, minimum_width],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False
    )

    return image

def load_dataset_paths(dataset_name='final_data', validation_split_size=0.2, is_exist_validation=False, density_map_folder_name='density_maps'):
    train_image_paths = []
    validation_image_paths = []
    test_image_paths = []

    train_label_paths = []
    train_dm_paths = []

    validation_label_paths = []
    validation_dm_paths = []

    test_label_paths = []
    test_dm_paths = []

    path = ROOT_PATH + '/datasets/' + dataset_name

    sys.stdout.flush()
    train_image_path_string = path + '/train/images/'
    test_image_path_string = path + '/test/images/'
    validation_image_path_string = None
    is_split_from_train = False
    if is_exist_validation:
        is_split_from_train = False
        validation_image_path_string = path + '/validation/images/'

    train_image_file_ids = next(os.walk(train_image_path_string))[2]
    validation_image_file_ids = []
    if validation_split_size > 0:
        is_split_from_train = True
        train_image_file_ids, validation_image_file_ids = train_test_split(train_image_file_ids,
                                                                           test_size=validation_split_size,
                                                                           random_state=1996)
    if is_exist_validation and validation_image_path_string is not None:
        validation_image_file_ids = next(os.walk(validation_image_path_string))[2]

    test_image_file_ids = next(os.walk(test_image_path_string))[2]

    for id in train_image_file_ids:
        file_path = os.path.join(path + '/train/images/', id)
        label_file_path = os.path.join(path + '/train/masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + '/train/' + density_map_folder_name + '/', os.path.splitext(id)[0] + '.npy')

        train_image_paths.append(str(file_path))
        train_label_paths.append(str(label_file_path))
        train_dm_paths.append(str(dm_file_path))

    for id in validation_image_file_ids:
        sub_path = '/train/'
        if not is_split_from_train:
            sub_path = '/validation/'

        file_path = os.path.join(path + sub_path + 'images/', id)
        label_file_path = os.path.join(path + sub_path + 'masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + sub_path + density_map_folder_name + '/', os.path.splitext(id)[0] + '.npy')

        validation_image_paths.append(str(file_path))
        validation_label_paths.append(str(label_file_path))
        validation_dm_paths.append(str(dm_file_path))

    for id in test_image_file_ids:
        file_path = os.path.join(path + '/test/images/', id)
        label_file_path = os.path.join(path + '/test/masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + '/test/' + density_map_folder_name + '/', os.path.splitext(id)[0] + '.npy')

        test_image_paths.append(str(file_path))
        test_label_paths.append(str(label_file_path))
        test_dm_paths.append(str(dm_file_path))

    dataset_dict = {
        'train': {
            'images': train_image_paths,
            'masks': train_label_paths,
            'density_maps': train_dm_paths
        },
        'validation': {
            'images': validation_image_paths,
            'masks': validation_label_paths,
            'density_maps': validation_dm_paths
        },
        'test': {
            'images': test_image_paths,
            'masks': test_label_paths,
            'density_maps': test_dm_paths
        },
    }

    return dataset_dict


def gen_pre_process_func(downsampling=8, method='nearest', is_multi_outputs=False, is_regression=False):
    batch_add = 1
    @tf.function
    def _pre_process_(img, gth):

        before_resize = tf.reduce_sum(gth)

        if is_regression:
            return img, tf.expand_dims(before_resize, axis=0)

        if downsampling > 1:

            gth_shape = tf.shape(gth)
            resize_gth = tf.image.resize(
                gth,
                (gth_shape[0 + batch_add] // downsampling, gth_shape[1 + batch_add] // downsampling),
                method=method, antialias=False
            )

            after_resize = tf.reduce_sum(resize_gth)

            if after_resize > 0:
                resize_gth = resize_gth * (before_resize / after_resize)
            gth = resize_gth
        if is_multi_outputs:
            return img, {"density_map": gth, "estimation": before_resize}
        else:
            return img, gth

    return _pre_process_


def load_dataset(input_paths=[], output_paths=[], output_type='density_maps',
                 batch_size=1, buffer_size=256,
                 shuffle=False,
                 downsampling_size=8, resize_method=tf.image.ResizeMethod.BILINEAR,
                 is_multi_outputs=False,
                 is_regression=False
                 ):
    data_size = len(input_paths)
    input_data = tf.data.Dataset.from_tensor_slices(input_paths)
    input_data = input_data.map(load_image)

    mask_data = tf.data.Dataset.from_tensor_slices(output_paths)
    if output_type == 'density_maps':
        mask_data = mask_data.map(lambda x: tf.py_function(
                                                            func=load_density_map,
                                                            inp=[x],
                                                            Tout=tf.dtypes.float32
                                            )
        )
        mask_data = mask_data.map(lambda x: set_shapes(x))
    else:
        mask_data = mask_data.map(load_mask)

    dataset = tf.data.Dataset.zip((input_data, mask_data))

    if shuffle:
        dataset = dataset.batch(batch_size).repeat().shuffle(buffer_size)
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        gen_pre_process_func(downsampling=downsampling_size, method=resize_method, is_multi_outputs=is_multi_outputs, is_regression=is_regression)
    )

    return dataset, data_size