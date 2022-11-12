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



dataset_dict = data_loader.load_dataset_paths(dataset_name='trancos_v3', validation_split_size=0.1)

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
    downsampling_size=8
)


test_dataset, test_size = data_loader.load_dataset(
    input_paths=test_input_data,
    output_paths=test_output_data,
    output_type='density_maps',
    batch_size=1,
    shuffle=False,
    downsampling_size=8
)

model.evaluate_model('./model_checkpoints/model_CSRNet_checkpoint_11_06_2022_180257.h5', validation_dataset)

model.evaluate_model('./model_checkpoints/model_CSRNet_checkpoint_11_06_2022_180257.h5', test_dataset)