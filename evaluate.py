import matplotlib.pyplot as plt

from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

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

image_files_ids = next(os.walk(TEST_PATH_IMAGES))[2]

test_generator = feed_data_generator.DatasetGenerator(
    image_files_ids=image_files_ids,
    dataset_name='final_data_color_1',
    data_type='test',
    batch_size=16,
    img_h=192,
    img_w=256,
    n_channels=3
)

model.evaluate_model('./model_checkpoints/model_unet_checkpoint_08_21_2022_041653.h5', test_generator)