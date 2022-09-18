from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
from utils import feed_data_generator
import pathlib
import os


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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

validation_dataset, val_size = data_loader.load_dataset(
    'final_data', section='val',
    batch_size=1, shuffle=False, downsampling_size=8
)

test_dataset, test_size = data_loader.load_dataset(
    'final_data', section='test',
    batch_size=1, shuffle=False, downsampling_size=8
)

model.evaluate_model('./model_checkpoints/model_CSRnet_model_09_15_2022_060843.h5', validation_dataset)