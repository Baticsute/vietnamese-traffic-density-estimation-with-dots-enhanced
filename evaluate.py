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



dataset_dict = data_loader.load_dataset_paths(dataset_name='final_data', validation_split_size=0.1)

validation_input_data = dataset_dict['validation']['images']
validation_output_data = dataset_dict['validation']['density_maps']

test_input_data = dataset_dict['test']['images']
test_output_data = dataset_dict['test']['density_maps']

validation_dataset, val_size = data_loader.load_dataset(
    input_paths=validation_input_data,
    output_paths=validation_output_data,
    output_type='density_maps',
    batch_size=1,
    shuffle=False,
    downsampling_size=2
)


test_dataset, test_size = data_loader.load_dataset(
    input_paths=test_input_data,
    output_paths=test_output_data,
    output_type='density_maps',
    batch_size=1,
    shuffle=False,
    downsampling_size=2
)

model.evaluate_model('./model_checkpoints/model_WNet_checkpoint_09_19_2022_071819.h5', validation_dataset)

model.evaluate_model('./model_checkpoints/model_WNet_checkpoint_09_19_2022_071819.h5', test_dataset)