from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
import pathlib

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

BATCH_SIZE = 1
BATCH_SAMPLE_SIZE = 64
DATASET_LOOP = 10

dataset_dict = data_loader.load_dataset_paths(dataset_name='final_data', validation_split_size=0.1)

train_input_data = dataset_dict['train']['images']
train_output_data = dataset_dict['train']['density_maps']

validation_input_data = dataset_dict['validation']['images']
validation_output_data = dataset_dict['validation']['density_maps']

train_dataset, train_size = data_loader.load_dataset(
    input_paths=train_input_data,
    output_paths=train_output_data,
    output_type='density_maps',
    batch_size=BATCH_SIZE,
    shuffle=True,
    downsampling_size=2,
    buffer_size=512
)

validation_dataset, val_size = data_loader.load_dataset(
    input_paths=validation_input_data,
    output_paths=validation_output_data,
    output_type='density_maps',
    batch_size=BATCH_SIZE,
    shuffle=False,
    downsampling_size=2
)

# for image, mask in train_dataset:
#     print(image.shape)
#     print(mask.shape)
#     print(tf.reduce_sum(mask))


net = model.get_wnet_model(img_h=480, img_w=640, img_ch=3)

model.train_model(
    model=net,
    train_data=train_dataset,
    valid_data=validation_dataset,
    steps_per_epoch=int(train_size / BATCH_SAMPLE_SIZE),
    validation_steps=val_size,
    n_epochs=BATCH_SAMPLE_SIZE * DATASET_LOOP,
    model_checkpoint_filename='model_WNet_checkpoint',
    patience=100,
    monitor='val_density_mae'
)