from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf

import numpy as np
from utils import feed_data_generator
from sklearn.model_selection import train_test_split
import pathlib
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

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
batch_size = 32
#
# image_files_ids = next(os.walk(TRAIN_PATH_IMAGES))[2]
# training_ids, validation_ids = train_test_split(image_files_ids, test_size=0.2, random_state=2022)

# training_generator = feed_data_generator.DatasetGenerator(
#     image_files_ids=training_ids,
#     dataset_name='final_data_color_1',
#     data_type='train',
#     batch_size=16,
#     img_h=192,
#     img_w=256,
#     n_channels=3
# )
#
# validation_generator = feed_data_generator.DatasetGenerator(
#     image_files_ids=validation_ids,
#     dataset_name='final_data_color_1',
#     data_type='train',
#     batch_size=16,
#     img_h=192,
#     img_w=256,
#     n_channels=3
# )

def gen_pre_process_func(downsampling=8, method='nearest', is_use_imagenet=False):
    batch_axis = 1

    @tf.function
    def _pre_process_(img, gth):
        if is_use_imagenet:
            img = img - tf.constant([0.485, 0.456, 0.406], dtype=tf.float32) / tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        down_ratio = downsampling
        before_resize = tf.reduce_sum(gth)

        gth_shape = tf.shape(gth)
        out_gth = tf.image.resize(gth,
                                  (gth_shape[0 + batch_axis] // down_ratio, gth_shape[1 + batch_axis] // down_ratio),
                                  method=method, antialias=False)
        out_gth = tf.cast(out_gth, dtype=tf.float32)
        after_resize = tf.reduce_sum(out_gth)

        if after_resize > 0:
            out_gth = out_gth * before_resize / after_resize

        return img, out_gth

    return _pre_process_


def load_dataset(
        dataset_name, section='train', batch_size=32, buffer_size=256,
        shuffle=False, downsampling_size=8, resize_method=tf.image.ResizeMethod.BILINEAR,
        is_use_imagenet=False
):
    data, mask = data_loader.load_bulk_data(dataset_name, section)

    input_data = tf.data.Dataset.from_tensor_slices(data)
    mask_data = tf.data.Dataset.from_tensor_slices(mask)

    dataset = tf.data.Dataset.zip((input_data, mask_data))

    if shuffle:
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    if downsampling_size > 1:
        dataset = dataset.map(
            gen_pre_process_func(downsampling=downsampling_size, method=resize_method, is_use_imagenet=is_use_imagenet)
        )

    return dataset


BATCH_SIZE = 32

# train_dataset = load_dataset('unet_mini_data', section='train', batch_size=BATCH_SIZE, shuffle=True,
#                              downsampling_size=1, is_use_imagenet=False)
# validation_dataset = load_dataset('unet_mini_data', section='val', batch_size=BATCH_SIZE,
#                                   downsampling_size=1,
#                                   is_use_imagenet=False)

train_dataset = load_dataset('unet_final_data', section='train', batch_size=BATCH_SIZE, shuffle=True,
                             downsampling_size=2, is_use_imagenet=True)
validation_dataset = load_dataset('unet_final_data', section='val', batch_size=BATCH_SIZE,
                                  downsampling_size=2,
                                  is_use_imagenet=True)
# test_dataset = load_dataset('unet_mini_data', section='test', batch_size=BATCH_SIZE,
#                                   downsampling_size=8,
#                                   is_use_imagenet=True)

# for img, mask in train_dataset:
#     print(tf.reduce_sum(mask))

net = model.get_wnet_model(img_h=192, img_w=256, img_ch=3)

# for img, mask in train_dataset:
#     result = net.predict(img)
#     print(result.shape)

#
model.train_model(
    model=net,
    train_data=train_dataset,
    valid_data=validation_dataset,
    batch_size=BATCH_SIZE,
    n_epochs=500,
    model_checkpoint_filename='model_unet_checkpoint',
    patience=100,
    monitor='val_loss'
)