from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
import pathlib

import math

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
BATCH_SAMPLE_SIZE = 128
DATASET_LOOP = 10
DOWN_SAMPLING = 2
is_multi_outputs = False
BUFFER_SIZE = 512

dataset_dict = data_loader.load_dataset_paths(dataset_name='final_data', validation_split_size=0.2, density_map_folder_name='density_maps')

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
    downsampling_size=DOWN_SAMPLING,
    buffer_size=BUFFER_SIZE,
    is_multi_outputs=is_multi_outputs
)

validation_dataset, val_size = data_loader.load_dataset(
    input_paths=validation_input_data,
    output_paths=validation_output_data,
    output_type='density_maps',
    batch_size=BATCH_SIZE,
    shuffle=False,
    downsampling_size=DOWN_SAMPLING,
    is_multi_outputs=is_multi_outputs
)

# net = model.get_csrnet_model(img_h=480, img_w=640, img_ch=3, is_multi_output=is_multi_outputs)
net = model.get_u_asd_net(img_h=480, img_w=640, img_ch=3, is_use_max_unpool=False, BN=False)

# net.summary()

model.train_model(
    model=net,
    train_data=train_dataset,
    valid_data=validation_dataset,
    steps_per_epoch=int(math.ceil((1. * train_size) / BATCH_SAMPLE_SIZE)),
    validation_steps=val_size,
    n_epochs=BATCH_SAMPLE_SIZE * DATASET_LOOP,
    model_checkpoint_filename='model_U-ASD_Net_No_BN_checkpoint',
    patience=100,
    monitor='val_loss'
)