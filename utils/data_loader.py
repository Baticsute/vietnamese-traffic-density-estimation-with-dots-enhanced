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
from PIL import Image

import cv2
import tensorflow as tf

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
    distances, locations = tree.query(points, k=k_nearest, workers=16)

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
        image_data[n] = img

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
            image_val_data[n] = img

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

def generate_density_maps_from_groundtruths(dataset_name, k_nearest=3, beta=0.3, leafsize=2048, fixed_sigma=None):
    path = ROOT_PATH + '/datasets/' + dataset_name

    sys.stdout.flush()
    train_mask_path_string = path + '/train/masks/'
    test_mask_path_string = path + '/test/masks/'
    validation_mask_path_string = path + '/validation/masks/'

    train_dm_path_string = path + '/train/density_maps/'
    test_dm_path_string = path + '/test/density_maps/'
    validation_dm_path_string = path + '/validation/density_maps/'

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
        dm, points = mapping_rescale_dot(mask_resized, mask)
        dm = gaussian_filter_density(
            dm.shape, np.fliplr(points),
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
        dm, points = mapping_rescale_dot(mask_resized, mask)
        dm = gaussian_filter_density(
            dm.shape, np.fliplr(points),
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
        dm, points = mapping_rescale_dot(mask_resized, mask)
        dm = gaussian_filter_density(
            dm.shape, np.fliplr(points),
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
    image = tf.image.resize_with_pad(
        image,
        target_height=minimum_height,
        target_width=minimum_width,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False
    )

    return image

def load_density_map(path):
    dm = np.load(path.numpy().decode())
    dm = tf.convert_to_tensor(dm, dtype=TARGET_TYPE)
    dm = tf.image.convert_image_dtype(dm, dtype=TARGET_TYPE)

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
    image = tf.image.resize_with_pad(
        image,
        target_height=minimum_height,
        target_width=minimum_width,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False
    )

    return image

def load_dataset_paths(dataset_name='final_data', validation_split_size=0.2):
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

    train_image_file_ids = next(os.walk(train_image_path_string))[2]
    validation_image_file_ids = []
    if validation_split_size > 0:
        train_image_file_ids, validation_image_file_ids = train_test_split(train_image_file_ids,
                                                                           test_size=validation_split_size,
                                                                           random_state=1996)
    test_image_file_ids = next(os.walk(test_image_path_string))[2]

    for id in train_image_file_ids:
        file_path = os.path.join(path + '/train/images/', id)
        label_file_path = os.path.join(path + '/train/masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + '/train/density_maps/', os.path.splitext(id)[0] + '.npy')

        train_image_paths.append(str(file_path))
        train_label_paths.append(str(label_file_path))
        train_dm_paths.append(str(dm_file_path))

    for id in validation_image_file_ids:
        file_path = os.path.join(path + '/train/images/', id)
        label_file_path = os.path.join(path + '/train/masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + '/train/density_maps/', os.path.splitext(id)[0] + '.npy')

        validation_image_paths.append(str(file_path))
        validation_label_paths.append(str(label_file_path))
        validation_dm_paths.append(str(dm_file_path))

    for id in test_image_file_ids:
        file_path = os.path.join(path + '/test/images/', id)
        label_file_path = os.path.join(path + '/test/masks/', os.path.splitext(id)[0] + '.png')
        dm_file_path = os.path.join(path + '/test/density_maps/', os.path.splitext(id)[0] + '.npy')

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


def gen_pre_process_func(downsampling=8, method='nearest'):
    batch_add = 1
    @tf.function
    def _pre_process_(img, gth):
        img = img / tf.constant(255.0, dtype=tf.float32)

        if downsampling > 1:
            before_resize = tf.reduce_sum(gth)

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

        return img, gth

    return _pre_process_


def load_dataset(input_paths=[], output_paths=[], output_type='density_maps',
                 batch_size=1, buffer_size=256,
                 shuffle=False, repeat=True,
                 downsampling_size=8, resize_method=tf.image.ResizeMethod.BILINEAR,
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
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(
        gen_pre_process_func(downsampling=downsampling_size, method=resize_method)
    )

    return dataset, data_size