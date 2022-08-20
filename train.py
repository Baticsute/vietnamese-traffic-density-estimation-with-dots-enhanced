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

image_files_ids = next(os.walk(TRAIN_PATH_IMAGES))[2]
training_ids, validation_ids = train_test_split(image_files_ids, test_size=0.2, random_state=2022)

training_generator = feed_data_generator.DatasetGenerator(
    image_files_ids=training_ids,
    dataset_name='final_data_color_1',
    data_type='train',
    batch_size=16,
    img_h=192,
    img_w=256,
    n_channels=3
)

validation_generator = feed_data_generator.DatasetGenerator(
    image_files_ids=validation_ids,
    dataset_name='final_data_color_1',
    data_type='train',
    batch_size=16,
    img_h=192,
    img_w=256,
    n_channels=3
)

#
unet = model.get_unet_model(img_h=192, img_w=256, img_ch=3)
model.train_generator_model(
    model=unet,
    train_data_generator=training_generator,
    validation_data_generator=validation_generator,
    batch_size=batch_size,
    n_epochs=500,
    model_checkpoint_filename='model_unet_checkpoint',
    patience=100
)