from utils import data_loader
import pathlib

ROOT_PATH = str(pathlib.Path().absolute())

DATASET_PATH = '/datasets/night_traffic'
FINAL_DATASET_PATH = ROOT_PATH + DATASET_PATH

TRAIN_PATH_IMAGES = FINAL_DATASET_PATH + '/train/images/'
TEST_PATH_IMAGES = FINAL_DATASET_PATH + '/test/images/'
VALI_PATH_IMAGES = FINAL_DATASET_PATH + '/validation/images/'

TRAIN_PATH_MASKS = FINAL_DATASET_PATH + '/train/masks/'
TEST_PATH_MASKS = FINAL_DATASET_PATH + '/test/masks/'
VALI_PATH_MASKS = FINAL_DATASET_PATH + '/validation/masks/'

data_loader.generate_density_maps_from_groundtruths('night_traffic', is_dot_illusion=False, gap_spaces=(15,5), fixed_sigma=5)
