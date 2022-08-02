from utils import data_loader
import pathlib


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

data_loader.prepare_and_save_data(TRAIN_PATH_IMAGES, TRAIN_PATH_MASKS, type='train', img_h=96, img_w=128, img_ch=1)
data_loader.prepare_and_save_data(TEST_PATH_IMAGES, TEST_PATH_MASKS, type='train', img_h=96, img_w=128, img_ch=1)
# data_loader.get_test_data('final_data', img_h=96, img_w=128, img_ch=1)